# === linear_elastic.py =========================================
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ufl
from dolfin import *


# ---------------------------------------------------------------
@dataclass
class Params:
    """Primitive material constants (+ 2-D mode flag)."""
    E0: float
    nu: float
    rho: float
    stress_c: float
    ci: float
    plane_stress: bool = False          # ← NEW (only meaningful when dim == 2)


# ---------------------------------------------------------------
class LinearElasticBase(ABC):
    """Abstract isotropic small-strain linear-elastic model."""
    _dim: int | None = None

    def __init__(self, params: Params):
        if self.dim is None:
            raise TypeError(
                "LinearElasticBase cannot be instantiated directly")

        # --- store primitives ------------------------------------------------
        self.E0 = params.E0_ice
        self.nu = params.nu_ice
        self.rho = params.rho_i_ice
        self.critical_stress = params.sigma_c_ice
        self.ci = params.ci

        # --- derived constants ----------------------------------------------
        self.critical_energy = 0.5 * self.critical_stress**2 / (2 * self.E0)
        self.mu = self.E0 / (2 * (1 + self.nu))
        self.lmbda = self.E0 * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    # -----------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self._dim

    # --- kinematics / constitutive -----------------------------------------
    def strain(self, u):
        return 0.5 * (grad(u) + grad(u).T)

    def stress(self, u):
        eps = self.strain(u)
        return 2 * self.mu * eps + self.lmbda * tr(eps) * Identity(self.dim)

    def strain_energy(self, u):
        eps = self.strain(u)
        return self.lmbda / 2 * tr(eps) ** 2 + self.mu * tr(eps * eps)

    def split_plus_minus(self, T):
        x_plus = self.applyElementwise(lambda x: 0.5 * (abs(x) + x), T)
        x_minus = self.applyElementwise(lambda x: 0.5 * (abs(x) - x), T)
        return x_plus, x_minus

    @abstractmethod
    def principal_strain(self, u):
        """
        Return the principal strain tensor of the displacement field `u`.
        The result is a 2x2 or 3x3 tensor depending on the dimension.
        """
        pass

    @abstractmethod
    def principal_stress(self, u):
        """
        Return the principal stress tensor of the displacement field `u`.
        The result is a 2x2 or 3x3 tensor depending on the dimension.
        """
        pass

    @abstractmethod
    def crack_driving_force(self, u):
        """
        Return the crack driving force (CDF) for the displacement field `u`.
        The result is a scalar field.
        """
        pass


# ---------------------------------------------------------------
class LinearElastic2D(LinearElasticBase):
    """2-D concrete class with built-in plane-stress / plane-strain switch."""
    _dim = 2

    def __init__(self, params: Params):
        self.plane_stress = bool(params.plane_stress)
        super().__init__(params)

        # replace λ for plane-stress formulation
        if self.plane_stress:
            self.lmbda = self.E0 * self.nu / (1 - self.nu ** 2)

    def eigenstate(self, t):
        eig1 = 0.5 * (tr(t) + safeSqrt(tr(t) * tr(t) - 4 * det(t)))
        eig2 = 0.5 * (tr(t) - safeSqrt(tr(t) * tr(t) - 4 * det(t)))
        return as_tensor([[eig1, 0], [0, eig2]])

    def applyElementwise(self, f, T):
        sh = ufl.shape(T)
        if len(sh) == 0:
            return f(T)
        fT = []
        for i in range(0, sh[0]):
            fT += [self.applyElementwise(f, T[i])]
        return as_tensor(fT)

    def principal_strain(self, u):
        return self.eigenstate(self.strain(u))

    def principal_stress(self, u):
        return self.eigenstate(self.stress(u))

    def crack_driving_force(self, u):
        stress_plus, _ = self.split_plus_minus(self.principal_stress(u))
        cdf_expr = self.ci * (
            (stress_plus[0, 0] / self.critical_stress) ** 2
            + (stress_plus[1, 1] / self.critical_stress) ** 2
            - 1
        )
        cdf_expr = ufl.Max(cdf_expr, 0)
        # Apply the threshold condition
        # cdf_expr = ufl.conditional(ufl.le(cdf_expr, 8), 0, cdf_expr)
        return cdf_expr


class LinearElastic3D(LinearElasticBase):
    """3-D concrete class (plane-stress flag is ignored)."""
    _dim = 3

    def eigenstate(self, A):
        if ufl.shape(A) != (3, 3):
            raise RuntimeError(
                f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
        #
        eps = 1.0e-10
        #
        A = ufl.variable(A)
        q = ufl.tr(A) / 3
        B = A - q * ufl.Identity(3)
        # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
        # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
        j = ufl.tr(B * B) / 2
        b = ufl.tr(B * B * B) / 3  # == I3(B) for trace-free B

        p = 2 / ufl.sqrt(3) * ufl.sqrt(j + eps ** 2)  # eps: MMM
        r = 4 * b / p ** 3
        r = ufl.Max(ufl.Min(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
        phi = ufl.acos(r) / 3
        # sorted eigenvalues: λ0 <= λ1 <= λ2
        λ0 = q + p * ufl.cos(phi + 2 / 3 * ufl.pi)  # low
        λ1 = q + p * ufl.cos(phi + 4 / 3 * ufl.pi)  # middle
        λ2 = q + p * ufl.cos(phi)  # high

        return as_tensor([[λ2, 0, 0], [0, λ1, 0], [0, 0, λ0]])

    # Apply some scalar-to-scalar mapping `f` to each component of `T`:

    def applyElementwise(self, f, T):
        from ufl import shape

        sh = shape(T)
        if len(sh) == 0:
            return f(T)
        fT = []
        for i in range(0, sh[0]):
            fT += [self.applyElementwise(f, T[i, i])]
        return as_tensor([[fT[0], 0, 0], [0, fT[1], 0], [0, 0, fT[2]]])
    
    def principal_strain(self, u):
        return self.eigenstate(self.strain(u))

    def principal_stress(self, u):
        return self.eigenstate(self.stress(u))
    
    def crack_driving_force(self, disp):
        stress_plus, stress_minus =  self.split_plus_minus(self.eigenstate(self.stress(disp)))
        energy_expr = self.ci * (
            (stress_plus[0, 0] / self.critical_stress) ** 2 + (stress_plus[1, 1] / self.critical_stress) ** 2 + (stress_plus[2, 2] / self.critical_stress) ** 2 - 1
        )
        energy_expr = ufl.Max(energy_expr, 0)
        # Apply the threshold condition
        # energy_expr = ufl.conditional(ufl.le(energy_expr, self.energy_thsd), 0, energy_expr)

        return energy_expr  


# ---------------------------------------------------------------
class LinearElastic(LinearElasticBase):
    """
    Factory alias: ``LinearElastic(mesh, params)`` returns a ready
    *LinearElastic2D* or *LinearElastic3D* instance.
    """

    def __new__(cls, mesh: Mesh, params: Params):
        dim = mesh.geometry().dim()

        if dim == 2:
            return LinearElastic2D(params)

        if dim == 3:
            if params.plane_stress:
                raise ValueError("`plane_stress` is only valid for 2-D meshes")
            return LinearElastic3D(params)

        raise ValueError(f"Unsupported mesh dimension {dim}")

    # __init__ never runs because __new__ returns a subclass instance.


# Safe square root function to avoid numerical issues

def safeSqrt(x):
    return sqrt(x + DOLFIN_EPS)
