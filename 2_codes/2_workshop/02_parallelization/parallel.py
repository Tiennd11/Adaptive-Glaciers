# %%
from dolfin import *
import numpy as np
import ufl
import time
import meshio
import pygmsh
import matplotlib.pyplot as plt
import csv
from ufl import shape
from mpi4py import MPI as pyMPI
set_log_level(0)


def mproject(fun, fun_space):
    return project(fun, fun_space, solver_type="gmres", preconditioner_type="hypre_euclid")


# Set log level
set_log_level(LogLevel.ERROR)
comm = MPI.comm_world  # MPI communicator

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


# %%
l_mul = 4.0 
hw_ratio = 0.5

lc = 2.5 * l_mul    # nonlocal length scale (m)
l = lc
targeted_him = 2.0 * l_mul  
# ---------------------------------------------------------------------------------
# DOMAIN SIZE # PROBLEM PARAMETERS    ---------------------------------------------
# ---------------------------------------------------------------------------------
hs_ratio = 0.0  # water level in crevasse (normalized with crevasse height)
# water level at right terminus (normalized with glacier thickness)

prec_ratio = 0.08  # depth of pre-crack (normalized with glacier thickness)

# Threshold for energy
energy_thsd = 6   # hw = 0.5-thres = 0.5, hw = 0-thres = 10

alpha_2 = 0.01  # threshold for damage

Lx, Ly, Lz = 500, 750, 125
L, B, H = Lx, Ly, Lz
hw = hw_ratio*H


# ---------------------------------------------------------------------------------
# MATERIAL PARAMETERS     ------------------------------------------------
# ---------------------------------------------------------------------------------

E = 9.5e9  # Young's modulus (Pa)
nu = 0.35  # Poisson's ratio
KIc = 0.1e6  # critical stress intensity factor (Pa*m^0.5)
rho_ice = 917  # density of ice (kg/m^3)

# rho_ice = 917  # density of ice (kg/m^3)
rho_H2O = 1000  # density of freshwater (kg/m^3)
rho_sea = 1020  # density of seawater (kg/m^3)
grav = 9.81  # gravity acceleration (m/s**2)

mu = E / 2 / (1 + nu)  # shear modulus
K = E / 3 / (1 - 2 * nu)  # bulk modulus
lmbda = K - 2 / 3 * mu  # Lame's first parameter


# water level at terminus (absolute height)
ci = 1

sigmac = 0.1185e6  # critical stress only for stress based method


# Step 3: Loading the Mesh
mesh = Mesh()
with XDMFFile(comm, "mesh/adaptive/mesh.xdmf") as infile:
    infile.read(mesh)

mprint(mesh.hmin())

model = "stress_cdf"
is_adaptive = True

# Step 4: Preparing output file
xdmf = XDMFFile(comm, "output/adaptive/"+str(hw_ratio)+"/output.xdmf")
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = True
xdmf.parameters["flush_output"] = True
# ---------------------------------------------------------------------------------
# MATERIAL MODEL  --------------------------------------------------------
# ---------------------------------------------------------------------------------
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))

# ---------------------------------------------------------------------------------
# CRACK DEPTH FUNCTION     --------------------------------------------------------
# ---------------------------------------------------------------------------------


# def get_crack_tip_coord(dmg, Dx,  damage_threshold=0.5):
#     y_co_ord = Function(Dx)
#     y_co_ord.interpolate(Expression("x[2]", degree=1))
#     dmg_dg_1 = mproject(dmg, Dx)
#     coord_vec = y_co_ord.vector()[dmg_dg_1.vector()[:] >= damage_threshold]
#     coord = 115  # ToDo: Change this
#     if coord_vec.size > 0:
#         coord = coord_vec.min()
#     return MPI.min(comm, coord)



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
    """Eigenvalues and eigenmprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenmprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenmprojectors of eigenvalues with multiplicity n are returned as 1/n-fold mprojector.
    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(
            f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    eps = 1.0e-10

    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
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


# ---------------------------------------------------------------------------------
# ENERGY CALCULATIONS        ------------------------------------------------
# ---------------------------------------------------------------------------------


# Apply some scalar-to-scalar mapping `f` to each component of `T`:
def applyElementwise(f, T):


    sh = shape(T)
    if len(sh) == 0:
        return f(T)
    fT = []
    for i in range(0, sh[0]):
        fT += [applyElementwise(f, T[i, i])]
    return as_tensor([[fT[0], 0, 0], [0, fT[1], 0], [0, 0, fT[2]]])


def split_plus_minus(T):
    x_plus = applyElementwise(lambda x: 0.5 * (abs(x) + x), T)
    x_minus = applyElementwise(lambda x: 0.5 * (abs(x) - x), T)
    return x_plus, x_minus


def safeSqrt(x):
    return sqrt(x + DOLFIN_EPS)


def get_energy(disp, pnew, D):
    if model == "stress_cdf":
        stress_plus, stress_minus = split_plus_minus(
            get_eigenstate(sigma(disp)))
        energy_expr = ci * (
            (stress_plus[0, 0] / sigmac) ** 2 + (stress_plus[1, 1] /
                                                sigmac) ** 2 + (stress_plus[2, 2] / sigmac) ** 2 - 1
        )
        energy_expr = ufl.Max(energy_expr, 0)
        # Apply the threshold condition
        energy_expr = ufl.conditional(
            ufl.le(energy_expr, energy_thsd), 0, energy_expr)

    return energy_expr


def problem(mesh, ms_iterations=1, u_adaptive=None, p_adaptive=None, cdf_adaptive=None):
    # Step 6: Defining spaces
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    F1 = FunctionSpace(mesh, "CG", 1)

    FDG0 = FunctionSpace(mesh, "DG", 0)
    TDG0 = TensorFunctionSpace(mesh, "DG", 0)

    # Step 7: Define functions
    u = TrialFunction(V1)
    v = TestFunction(V1)
    unew, uold = Function(V1, name="disp"), Function(V1, name="disp")
    stress_xx = Function(F1, name="sigma_xx")
    cdf = Function(FDG0, name="cdf")

    p = TrialFunction(F1)
    q = TestFunction(F1)
    pnew, pold = Function(F1, name="damage"), Function(F1, name="damage")

    # Step 8: Update old data (New to old)

    # if u_adaptive is not None:
    #     uold.assign(mproject(u_adaptive, V1))
        # unew.assign(mproject(u_adaptive, V1))

    if p_adaptive is not None:
        pold.assign(mproject(p_adaptive, F1))
        # pnew.assign(mproject(p_adaptive, F1))

    if cdf_adaptive is not None:
        cdf.assign(mproject(cdf_adaptive, FDG0))

    front = CompiledSubDomain("near(x[1], 0)")
    back = CompiledSubDomain("near(x[1], L)", L=Ly)
    left = CompiledSubDomain("near(x[0], 0)")
    right_csd = CompiledSubDomain("near(x[0], B)", B=Lx)
    bottom = CompiledSubDomain("near(x[2], 0)")
    top = CompiledSubDomain("near(x[2], H)", H=Lz)

    bottom_roller = DirichletBC(V1.sub(2), Constant(0), bottom)
    left_roller = DirichletBC(V1.sub(0), Constant(0), left)
    front_roller = DirichletBC(V1.sub(1), Constant(0), front)
    back_roller = DirichletBC(V1.sub(1), Constant(0), back)
    bc_u = [bottom_roller, left_roller, front_roller, back_roller]

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    right_csd.mark(boundaries, 1)

    # print(boundaries.where_equal(1))

    ds = Measure("ds", subdomain_data=boundaries)

    # Step 10: Define Loads
    t = Expression(("(h - x[2] >= 0 ?-rho_H2O * grav*(h - x[2]) : 0)",
                0, 0), h=hw, rho_H2O=rho_sea, grav=grav, degree=1)
    f = Constant((0, 0, -rho_ice * grav))

    # Step 11: Displacement and Damage Problems
    disp_a = inner(((1 - pold) ** 2 + 1e-4) * sigma(u), epsilon(v)) * dx
    step_fun = conditional(gt(pold, 0.001), 0.0, 1.0)
    disp_L = step_fun * dot(f, v) * dx + dot(t, v) * ds(1)

    phase_a = (l**2 * inner(grad(p), grad(q)) + (1)
            * inner(p, q) + cdf * inner(p, q)) * dx
    phase_L = inner(cdf, q) * dx

    # Step 12: Solve
    disp_problem = LinearVariationalProblem(disp_a, disp_L, unew, bc_u)
    disp_solver = LinearVariationalSolver(disp_problem)

    phase_problem = LinearVariationalProblem(phase_a, phase_L, pnew)
    phase_solver = LinearVariationalSolver(phase_problem)

    prm_disp = disp_solver.parameters
    prm_disp["linear_solver"] = "gmres"
    prm_disp["preconditioner"] = "hypre_euclid"

    prm_phase = phase_solver.parameters
    prm_phase["linear_solver"] = "gmres"
    prm_phase["preconditioner"] = "hypre_euclid"

    # Start iterations
    disp_solver.solve()

    # print(unew.vector()[:].max(), " -- ", unew.vector()[:].min())

    # Update history variable
    cdf.vector()[:] = np.maximum(
        mproject(get_energy(unew, pnew, FDG0), FDG0).vector()[:], cdf.vector()[:])

    # print(cdf.vector()[:].max(), " -- ", cdf.vector()[:].min())

    phase_solver.solve()
    # Clip the damage solution to 0-1
    pnew.vector()[:] = np.clip(pnew.vector()[:], 0, 1)

    tol = 1e-14

    norm_u = assemble(unew**2 * dx)
    norm_p = assemble(pnew**2 * dx)

    err_u = 0#sqrt(assemble((unew - uold)**2 * dx) / norm_u) if norm_u > tol else 0.0
    err_phi = sqrt(assemble((pnew - pold)**2 * dx) /
                norm_p) if norm_p > tol else 0.0

    ms_err = max(err_u, err_phi)

    return unew, pnew, cdf, ms_err


def get_markers(_phi, mesh, target_hmin):
    adaptivity_converged = False

    marker = MeshFunction("bool", mesh, mesh.topology().dim())
    marker.set_all(False)

    DG = FunctionSpace(mesh, "DG", 0)
    phi = (mproject(_phi, DG))
    marker.array()[phi.vector()[:] > alpha_2] = True
    # ---------------------------------------------------------------------------------
    # Scheme S4    -----------------------------------------------------
    # ---------------------------------------------------------------------------------
    cell_dia = Circumradius(mesh)
    dia_vector = mproject(cell_dia, DG).vector()[:]
    marker.array()[dia_vector < target_hmin] = False
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    DG = None
    del DG

    # Check adaptivity convergence across all MPI processes
    local_converged = np.all(np.invert(marker.array()))
    adaptivity_converged = comm.allreduce(local_converged, op=pyMPI.LAND)
    return marker, adaptivity_converged


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
# cpp_code = """
# #include <pybind11/pybind11.h>
# #include <dolfin/adaptivity/adapt.h>
# #include <dolfin/function/Function.h>
# #include <dolfin/mesh/Mesh.h>
# namespace py = pybind11;
# PYBIND11_MODULE(SIGNATURE, m)
# {
# m.def("adapt", [](const dolfin::Function &function,
#         std::shared_ptr<const dolfin::Mesh> adapted_mesh,
#                 bool interpolate){
#             return dolfin::adapt(function, adapted_mesh, interpolate);});
# }
# """
# m = compile_cpp_code(cpp_code)


# def adaptFunction(f, mesh, interp=True):
#     return m.adapt(f, mesh, interp)


# def transfer(_p, mesh):
#     _p = Function(adaptFunction(_p._cpp_object, mesh))
#     return _p

def transfer(from_fun, to_mesh):
    V1 = from_fun.ufl_function_space()
    V2 = FunctionSpace(to_mesh, "DG", 0)
    A = PETScDMCollection.create_transfer_matrix(V1,V2)
    f2 = Function(V2)
    f2.vector()[:] = A*from_fun.vector()
    return f2

# import csv, os
# # Specify the CSV file path
# csv_file_path = "output/adaptive/crevasse_depth_"+str(hw_ratio)+".csv"

# if os.path.exists(csv_file_path):
#     os.remove(csv_file_path)

# file = open(csv_file_path, mode='a', newline='')
# writer = csv.writer(file)
# writer.writerow([
#     f"{'step':>8}",
#     f"{'crack_depth':>12}",
#     f"{'time':>8}",
#     f"{'ndof':>8}",
#     f"{'ms_err':>8}"
# ])
# ---------------------------------------------------------------------------------
D = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
DG = FunctionSpace(mesh, "DG", 0)

pold = Function(D, name="damage")
uold = Function(V, name="displacement")
cdf_old = Function(DG, name="energy")

u_adaptive_n = uold
p_adaptive_n = pold
cdf_adaptive_n = cdf_old
start = time.time()

psi_old = Function(DG, name="energy")
psi_new = Function(DG, name="energy")
sigma_new = Function(DG, name="sigma")

time_elapsed = 0
adaptivity_converged = False

ms_err = 1.0  # Initialize ms_err to enter the while loop
tol = 1e-4  # Tolerance for convergence

while ms_err > tol:

    adaptivity_converged = False
    while not adaptivity_converged:
        u_adaptive_n_1, p_adaptive_n_1, cdf_adaptive_n_1, ms_err = problem(
            mesh, 1, u_adaptive_n, p_adaptive_n, cdf_adaptive_n)
        marker, adaptivity_converged = get_markers(
            p_adaptive_n_1, mesh, target_hmin=targeted_him)
        mesh = refine(mesh, marker)
        p_adaptive_n = transfer(p_adaptive_n, mesh)
        # u_adaptive_n = transfer(u_adaptive_n, mesh)
        cdf_adaptive_n = transfer(cdf_adaptive_n, mesh)
        # print("Adaptivity converged: ", adaptivity_converged)
        # adaptivity_converged = True

    p_adaptive_n.assign(transfer(p_adaptive_n_1, mesh))
    # u_adaptive_n.assign(transfer(u_adaptive_n_1, mesh))
    cdf_adaptive_n.assign(transfer(cdf_adaptive_n_1, mesh))

    # Write the data
    p_adaptive_n.rename("damage", "damage")
    xdmf.write(p_adaptive_n, time_elapsed+1)
    xdmf.write(u_adaptive_n, time_elapsed+1)

    # Increse time_elapsed
    time_elapsed += 1

    # Define new functions to print data
    Dx = FunctionSpace(mesh, "CG", 1)  # CDF

    # Determine crack tip coordinate
    cracktip_coor = 0# get_crack_tip_coord(p_adaptive_n, Dx=Dx)

    # Print
    mprint(
        "step: {0:3}, crack_depth: {1:6.0f}, hmin: {2:5.2f}, ndof: {5:6}, time: {3:6.0f}, ms_err: {4:6.2e}".format(
            time_elapsed,
            H - cracktip_coor,
            MPI.min(comm, mesh.hmin()),
            time.time() - start,
            ms_err,
            Dx.dim()*4
        )
    )

    # writer.writerow([
    #     f"{time_elapsed:>8.2f}",
    #     f"{H - cracktip_coor:>12.1f}",
    #     f"{time.time() - start:>8.1f}",
    #     f"{Dx.dim()*4:>8}",
    #     f"{ms_err:>8.1f}"
    # ])
    # file.flush()


