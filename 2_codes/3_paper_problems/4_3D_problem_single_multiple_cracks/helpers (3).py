from dolfin import *
import ufl as ufl


class MaterialModel:
    def __init__(self, E0=9500e6, nu=0.35, rho=917 * 9.81, stress_c=0.1185e6, ci=1):
        # Initialize material parameters
        self.E0 = E0
        self.nu = nu
        self.rho = rho
        self.critical_stress = stress_c
        self.critical_energy = 0.5 * stress_c**2 / (2 * E0)

        self.mu = self.E0 / (2 * (1 + self.nu))
        self.lmbda = self.E0 * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.ci = ci

    def strain(self, u):
        return 0.5 * (grad(u) + grad(u).T)

    def stress(self, u):
        return 2.0 * self.mu * self.strain(u) + self.lmbda * tr(
            self.strain(u)
        ) * Identity(len(u))

    def get_strain_energy(self, u):
        return self.lmbda / 2 * (tr(self.strain(u)) ** 2) + self.mu * tr(
            self.strain(u) * self.strain(u)
        )

    def get_principal_strain(self, u):
        return get_eigenstate(self.strain(u))

    def get_principal_stress(self, u):
        return get_eigenstate(self.stress(u))

    def get_crack_driving_force(self, u):
        stress_plus, _ = split_plus_minus(self.get_principal_stress(u))
        cdf_expr = self.ci * (
            (stress_plus[0, 0] / self.critical_stress) ** 2 + (stress_plus[1, 1] / self.critical_stress) ** 2+ (stress_plus[2,2] / self.critical_stress) ** 2 - 1)
        cdf_expr = ufl.Max(cdf_expr, 0)
#         cdf_expr = ufl.Max(cdf_expr, 0)
        # Apply the threshold condition
        cdf_expr = ufl.conditional(ufl.le(cdf_expr, 1.5), 0, cdf_expr)

        return cdf_expr


# Helper functions -------------------------------------------------
# def applyElementwise(f, T):
#     sh = ufl.shape(T)
#     if len(sh) == 0:
#         return f(T)
#     fT = []
#     for i in range(0, sh[0]):
#         fT += [applyElementwise(f, T[i])]
#     return as_tensor(fT)

# Apply some scalar-to-scalar mapping `f` to each component of `T`:
def applyElementwise(f, T):
    from ufl import shape

    sh = shape(T)
    if len(sh) == 0:
        return f(T)
    fT = []
    for i in range(0, sh[0]):
        fT += [applyElementwise(f, T[i, i])]
    return as_tensor([[fT[0], 0,0], [0, fT[1],0], [0, 0, fT[2]]])


def split_plus_minus(T):
    x_plus = applyElementwise(lambda x: 0.5 * (abs(x) + x), T)
    x_minus = applyElementwise(lambda x: 0.5 * (abs(x) - x), T)
    return x_plus, x_minus


# def get_eigenstate(t):
#     eig1 = 0.5 * (tr(t) + safeSqrt(tr(t) * tr(t) - 4 * det(t)))
#     eig2 = 0.5 * (tr(t) - safeSqrt(tr(t) * tr(t) - 4 * det(t)))
#     return as_tensor([[eig1, 0], [0, eig2]])

def get_eigenstate(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.
    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    #
    eps = 1.0e-10
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = ufl.tr(A) / 3
    B = A - q * ufl.Identity(3)
    # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
    j = ufl.tr(B * B) / 2  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B * B * B) / 3  # == I3(B) for trace-free B
    # solve: 0 = ω**3 - j * ω - b  by substitution  ω = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4 / 3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2 / ufl.sqrt(3) * ufl.sqrt(j + eps ** 2)  # eps: MMM
    r = 4 * b / p ** 3
    r = ufl.Max(ufl.Min(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
    phi = ufl.acos(r) / 3
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ0 = q + p * ufl.cos(phi + 2 / 3 * ufl.pi)  # low
    λ1 = q + p * ufl.cos(phi + 4 / 3 * ufl.pi)  # middle
    λ2 = q + p * ufl.cos(phi)  # high
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    # E0 = ufl.diff(λ0, A).T
    # E1 = ufl.diff(λ1, A).T
    # E2 = ufl.diff(λ2, A).T
    #
    # return [λ0, λ1, λ2], [E0, E1, E2]
    return as_tensor([[λ2, 0,0], [0, λ1,0], [0, 0, λ0]])



def safePower(x, pw):
    # return pow(x + DOLFIN_EPS, pw)
    return pow(x, pw)


def safeSqrt(x):
    return sqrt(x + DOLFIN_EPS)


def degradation(p):
    return (1 - p) ** 2 + 1e-4
