import ufl
import matplotlib.pyplot as plt
import time
import numpy as np
from dolfin import *
comm = MPI.comm_world  # MPI communicator
from fenics import PETScKrylovSolver
from petsc4py import PETSc

mesh = Mesh()
with XDMFFile(comm, "mesh/mesh.xdmf") as infile:
    infile.read(mesh)

# Get the mesh coordinates
coordinates = mesh.coordinates()

xdmf = XDMFFile(comm, "output/output.xdmf")
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False
xdmf.parameters["flush_output"] = True

energy_thsd = 30 # thres = 5 for 0.5
lx,ly,lz=500.0,750.0,125.0
L, B, H = lx, ly, lz

# ---------------------------------------------------------------------------------
# SIMULATION PARAMETERS        ------------------------------------------------
# ---------------------------------------------------------------------------------
time_elapsed = 0
time_total = 500  # arbitrary
delta_t = 1.0  # time increment
time_counter = 0
output_increment = 5


air, water, ice = 0, 1, 2
water_left, water_right = 3, 4
right = 2

# ---------------------------------------------------------------------------------
# MATERIAL PARAMETERS     ------------------------------------------------
# ---------------------------------------------------------------------------------

rho_ice = 917  # density of ice (kg/m^3)
rho_H2O = 1000  # density of freshwater (kg/m^3)
rho_sea = 1020  # density of seawater (kg/m^3)
grav = 9.81  # gravity acceleration (m/s**2)
E = 9.5e9  # Young's modulus (Pa)
nu = 0.35  # Poisson's ratio
mu = E / 2 / (1 + nu)  # shear modulus
K = E / 3 / (1 - 2 * nu)  # bulk modulus
lmbda = K - 2 / 3 * mu  # Lame's first parameter
KIc = 0.1e6  # critical stress intensity factor (Pa*m^0.5)
lc = 10  # nonlocal length scale (m)
# hw = hw_ratio * H  # water level at terminus (absolute height)
ci = 1
sigmac = 0.1185e6  # critical stress only for stress based method



# ---------------------------------------------------------------------------------
# PARAMETERS FOR PARALLEL COMPUTATIONS   -----------------------------
# ---------------------------------------------------------------------------------

rank = comm.Get_rank()  # number of current process
size = comm.Get_size()  # total number of processes


def mwrite(filename, my_list):
    MPI.barrier(comm)
    if rank == 0:
        with open(filename, "w") as f:
            for item in my_list:
                f.write("%s" % item)


def mprint(*argv):
    if rank == 0:
        out = ""
        for arg in argv:
            out = out + str(arg)
        # this forces program to output when run in parallel
        print(out, flush=True)


# ---------------------------------------------------------------------------------
# SET SOME COMMON FENICS FLAGS     ---------------------------------------
# ---------------------------------------------------------------------------------
set_log_level(50)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


# ---------------------------------------------------------------------------------
# MATERIAL MODEL  --------------------------------------------------------
# ---------------------------------------------------------------------------------
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))


# ---------------------------------------------------------------------------------
# ICE FRACTURE     --------------------------------------------------------
# ---------------------------------------------------------------------------------
def get_crack_tip_coord(dmg, damage_threshold=0.99):
    dmg_dg_1 = project(dmg, D1,solver_type="cg",
            preconditioner_type="hypre_euclid")
    coord_vec = z_co_ord.vector()[dmg_dg_1.vector()[:] >= damage_threshold]
    coord = 115  # ToDo: Change this
    if coord_vec.size > 0:
        coord = coord_vec.min()
    return MPI.min(comm, coord)


def get_density_mf(dmg, Dcr = 0.6):
    density_mf = MeshFunction("size_t", mesh, 2)
    density_mf.set_all(ice)
    density_mf.array()[dmg.vector()[:] > Dcr] = air
    crevasse_depth = surface_crevasse_level(dmg)
    water_zone = Function(D)

    density_mf.array()
    return density_mf


def surface_crevasse_level(dmg):
    # We have to figure out the crevasse_depth from top and the
    # crevasse_surface from bottom. We have the damage function
    x2_dmg_min = get_crack_tip_coord(dmg, damage_threshold=0.6)
    crevasse_depth = H - x2_dmg_min
    # water_surface_from_bottom = 0crevasse_depth * hs_ratio + x2_dmg_min
    return crevasse_depth

# -----------------------------------------------------------------------------
# 3D eigen decomposition
# -----------------------------------------------------------------------------

def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = ufl.tr(A)
    i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
    i3 = ufl.det(A)
    return i1, i2, i3


def invariants_main(A):
    """Main invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    j1 = ufl.tr(A)
    j2 = ufl.tr(A * A)
    j3 = ufl.tr(A * A * A)
    return j1, j2, j3


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


# ---------------------------------------------------------------------------------
# ENERGY CALCULATIONS        ------------------------------------------------
# ---------------------------------------------------------------------------------


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


def safeSqrt(x):
    return sqrt(x + DOLFIN_EPS)


def get_energy(disp, type="stress"):
    if type == "stress":
        stress_plus, stress_minus =  split_plus_minus(get_eigenstate(sigma(unew)))
        energy_expr = ci * (
            (stress_plus[0, 0] / sigmac) ** 2 + (stress_plus[1, 1] / sigmac) ** 2+ (stress_plus[2,2] / sigmac) ** 2 - 1
        )
        energy_expr = ufl.Max(energy_expr, 0)
        # Apply the threshold condition
        energy_expr = ufl.conditional(ufl.le(energy_expr, energy_thsd), 0, energy_expr)

        energy = project(energy_expr, D,solver_type="cg",
            preconditioner_type="hypre_euclid").vector()[:]
        # energy = assemble(energy_expr*TestFunction(D)*dx())
    else:
        eig = principal_tensor(strain(disp))
        eig_1, eig_2 = eig[0, 0], eig[1, 1]

        # Define the energy function
        energy_expr = ufl.conditional(
            ufl.ge(eig_2, 0),
            (0.5 * lmbda * (eig_1 + eig_2) ** 2 + mu * (eig_1**2 + eig_2**2)) / g1c,
            ufl.conditional(
                ufl.And(ufl.ge(eig_1, 0), ufl.le(eig_2, 0)),
                conditional(
                    ufl.gt((1 - nu) * eig_1 + nu * eig_2, 0),
                    (0.5 * lmbda / nu / (1 - nu) * ((1 - nu) * eig_1 + nu * eig_2) ** 2)
                    / g1c,
                    0,
                ),
                0,
            ),
        )

        # Apply the threshold condition
        energy_expr = ufl.conditional(ufl.le(energy_expr, energy_thsd), 0, energy_expr)
        # https://fenicsproject.discourse.group/t/unable-to-use-conditional-module/6502
        energy = assemble(energy_expr * TestFunction(D) * dx())

    return energy


def check_point(unew, pnew, time_current, check_point_every):
   if time_current % check_point_every == 0:
        file_name = "output/check_point/" + str(time_current) + ".h5"
        hdf5 = HDF5File(comm, file_name, "w")
        
        # time_current_fn = Function(FunctionSpace(IntervalMesh(1,0,1),'DG',0))
        # time_current_fn.vector().vec().array[:] = time_current
        
        hdf5.write(unew.leaf_node().function_space().mesh(),"mesh")
        hdf5.write(unew, "displacement")
        hdf5.write(pnew, "damage")
        # hdf5.write(time_current_fn, "time_current_fn")
        hdf5.close()



#  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄
# ▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌      ▐░▌
# ▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌     ▐░▌
# ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌▐░▌    ▐░▌
# ▐░▌ ▐░▐░▌ ▐░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌ ▐░▌   ▐░▌
# ▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░▌  ▐░▌  ▐░▌
# ▐░▌   ▀   ▐░▌▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌   ▐░▌ ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌    ▐░▌▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌ ▄▄▄▄█░█▄▄▄▄ ▐░▌     ▐░▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌      ▐░░▌
#  ▀         ▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀


# ---------------------------------------------------------------------------------
# FUNCTION SPACES  ---------------------------------------------------------
# ---------------------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 1)  # displacement shape function
P = FunctionSpace(mesh, "CG", 1)
D = FunctionSpace(mesh, "DG", 0)  # phase-field shape function
D1 = FunctionSpace(mesh, "DG", 1)
T = TensorFunctionSpace(mesh, "DG", 0)

mprint("-----------------------------------------------------")
mprint("Number of vertices      : {}".format(P.dim()))  # Number of mesh vertices
mprint("Number of cells         : {}".format(D.dim()))  # Number of mesh elements
mprint("Number of DoF (u)       : {}".format(V.dim()))  # DOFs for displacement
mprint("Number of DoF (p)       : {}".format(D.dim()))  # DOFs for damage (DG0, per element)
mprint("Total Number of DoF     : {}".format(V.dim() + D.dim()))  # Total DOFs in the system
mprint("Number of processes     : {}".format(size))  # Number of MPI processes
mprint("Number of DoF/process   : {}".format(int((V.dim() + D.dim()) / size)))  # DOFs per process
mprint("Number of DoF_dis/process   : {}".format(int(V.dim() / size)))
mprint("Number of DoF_damge/process   : {}".format(int(D.dim() / size)))

mprint("-----------------------------------------------------")



z_co_ord = Function(D)
z_co_ord.interpolate(Expression("x[2]", degree=1))

# ---------------------------------------------------------------------------------
# BOUNDARIES AND MEASURES    ------------------------------------------------
# ---------------------------------------------------------------------------------

front = CompiledSubDomain("near(x[0], 0)")
back = CompiledSubDomain("near(x[0], 500.0)")
left = CompiledSubDomain("near(x[1], 0)")
right_csd = CompiledSubDomain("near(x[1], 750.0)")
bottom = CompiledSubDomain("near(x[2], 0)")
top = CompiledSubDomain("near(x[2], 125.0)")

bottom_roller = DirichletBC(V.sub(2), Constant(0), bottom)
left_roller = DirichletBC(V.sub(1), Constant(0), left)
front_roller = DirichletBC(V.sub(0), Constant(0), front)
right_roller = DirichletBC(V.sub(1), Constant(0), right_csd)
back_roller = DirichletBC(V.sub(0), Constant(0), back)

disp_bc = [bottom_roller, left_roller,front_roller,right_roller]



z_co_ord = Function(D1)
z_co_ord.interpolate(Expression("x[2]", degree=1))

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
left.mark(boundaries, 1)
back.mark(boundaries, 2)
# right_csd.mark(boundaries, 2)

bottom.mark(boundaries, 3)
top.mark(boundaries, 4)
ds = Measure("ds", subdomain_data=boundaries)


# ---------------------------------------------------------------------------------
# FUNCTIONS    --------------------------------------------------------------
# ---------------------------------------------------------------------------------

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(P), TestFunction(P)

unew, uold = Function(V, name="displacement"), Function(V)
pnew, pold = Function(P, name="damage"), Function(P)
Hold = Function(D)
energy = Function(D)
cdf = Function(D, name="energy")


phase_a = (lc**2 * inner(grad(p), grad(q)) + (1) * inner(p, q)+ cdf * inner(p, q))*dx
phase_L = inner(cdf, q) * dx 


phase_problem = LinearVariationalProblem(phase_a, phase_L, pnew)
phase_solver = LinearVariationalSolver(phase_problem)



prm_phase = phase_solver.parameters
prm_phase["linear_solver"] = "cg"
prm_phase["preconditioner"] = "hypre_euclid"
# prm_phase["preconditioner"] = "petsc_amg"
# prm_phase["preconditioner"] = "hypre_amg"
# prm_phase["preconditioner"] = "jacobi"

# prm_disp['maximum_iterations'] = 10

# ---------------------------------------------------------------------------------
# START ITERATIONS    -----------------------------------------------------
# ---------------------------------------------------------------------------------
Dcr = 0.01
def psi(p):
    pis = ufl.conditional(ufl.le(p, Dcr), 1, 0)
    return pis

start = time.time()
restart_simulation = False
if restart_simulation:
    hdf5 = HDF5File(comm, "cp.h5", "r")

    hdf5.read(uold, "displacement")
    hdf5.read(pold, "damage")

while time_elapsed <= time_total:


    # while time_counter <= 300:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    crevasse_depth = surface_crevasse_level(pold)
    # dx = Measure("dx", subdomain_data=get_density_mf(pold))

    disp_a = inner(((1 - pold) ** 2 + 1e-4) * sigma(u), epsilon(v)) * dx
    disp_L = psi(pold)*inner(v, Constant((0,0, -rho_ice * grav))) * dx

    unew = Function(V, name="displacement")
    disp_problem = LinearVariationalProblem(disp_a, disp_L, unew, disp_bc)
    disp_solver = LinearVariationalSolver(disp_problem)

    prm_disp = disp_solver.parameters
    prm_disp["linear_solver"] = "cg"
    # prm_disp["linear_solver"] = "bicgstab"
    prm_disp["preconditioner"] = "hypre_euclid"
    # prm_disp["preconditioner"] = "hypre_amg"
    # prm_disp["preconditioner"] = "jacobi"
    # prm_disp["preconditioner"] = "bjacobi"
    # prm_disp['maximum_iterations'] = 10
    # prm_disp['relative_tolerance'] = 1e-3
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    disp_solver.solve()

    # update history variable --------------------------------------------------
    energy.vector().vec().array[:] = get_energy(unew)[:]
    energy.vector().vec().array[:] = np.maximum(energy.vector()[:], cdf.vector()[:])

    cdf.assign(energy)
    # kappa = np.fmax(energy, kappa)
    phase_solver.solve()

    # Clip the damage solution to 0-1 ------------------------------------------
    pnew.vector().vec().array[:] = np.clip(pnew.vector()[:], 0, 1)

    # pnew will be max(pold,pnew)-----------------------------------------------
    # pnew.vector().vec().array[:] = np.maximum(pnew.vector()[:], pold.vector()[:])

    # Update the measurse based on current damage ------------------------------
    pold.assign(pnew)

    # update counter -----------------------------------------------------------

    cracktip_coor = get_crack_tip_coord(pnew)
    depth = 1 - cracktip_coor/H
    if time_counter% 1 == 0:
        # Only write data every 10 iterations
        xdmf.write(unew, time_counter)
        xdmf.write(pnew, time_counter)
        xdmf.write(cdf, time_counter)
        # xdmf.write(unew, time_counter)
        # xdmf.write(pnew, time_counter)
        # xdmf.write(cdf, time_counter)
    mprint(
        "step: {0:3}, Depth: {1:6.4f}, time: {2:6.0f}, energy: {3:6.2f} ".format(
            time_counter,
            depth, 
            time.time() - start,
            get_energy(unew)[:].max(), 
        )
    )

    check_point(unew, pnew, time_current=time_counter, check_point_every=10)

    time_elapsed += delta_t
    time_counter += 1
