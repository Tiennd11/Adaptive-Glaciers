"""Run an updated-Lagrangian Stokes flow simulation with:
    - nonlinear viscosity
    - incompressibility
    - poro-mechanics formulation for hydraulic fracture
"""

from __future__ import division
from fenics import *
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


# suppress FEniCS output to terminal
set_log_active(False)


# output directory
output_dir = ""


comm = MPI.COMM_WORLD  # MPI communications
rank = comm.Get_rank()  # number of current process
size = comm.Get_size()  # total number of processes


if rank == 0:
    start = time.clock()
    time_log = open(output_dir + "output/details/time_log.txt", "w")
    time_log.close()
    with open(output_dir
              + "output/details/simulation_log.txt", "w") as sim_log:
        if size == 1:
            sim_log.write("Stokes flow in FEniCS.\nRunning on "
                          "1 processor.\n" + "-"*64 + "\n")
        else:
            sim_log.write("Stokes flow in FEniCS.\nRunning on "
                          "%d processors.\n" % size + "-"*64 + "\n")


def write_hdf5(timestep, mesh, data, time=0):
    """Write output from the simulation in .h5 file format."""
    output_path = output_dir + "output/data/" + str(timestep) + ".h5"
    hdf5 = HDF5File(mpi_comm_world(), output_path, "w")
    hdf5.write(mesh, "mesh")
    m = hdf5.attributes("mesh")
    m["current time"] = float(time)
    m["current step"] = int(timestep)
    for value in sorted(data):
        hdf5.write(data[value], value)
    hdf5.close()


def load_mesh(path):
    """Load a mesh in .h5 file format."""
    mesh = Mesh()
    hdf5 = HDF5File(mpi_comm_world(), path, "r")
    hdf5.read(mesh, "mesh", False)
    hdf5.close()
    return mesh


"""Material parameters."""

rho_ice = 917  # density of ice (kg/m^3)
rho_H2O = 1020  # density of seawater (kg/m^3)
grav = 9.81  # gravity acceleration (m/s**2)
temp = -10 + 273  # temperature (K)
B0 = 2.207e-3  # viscosity coefficient (kPa * yr**(1/3))
B0 *= 1e3  # convert to (Pa * yr**(1/3))
B0 *= (365*24*3600)**(1/3)  # convert to (Pa * second**(1/3))
BT = B0*np.exp(3155/temp - 0.16612/(273.39 - temp)**1.17)
ci = 1

sigmac = 0.1185e6
# sigmac = 0.0274e6
# energy_thsd = 10
# energy_thsd =  27.802

# energy_thsd = 13.541

# energy_thsd = 22.103 # 0H
# energy_thsd = 10.676 # 0.5H
energy_thsd = 13.541 
# energy_thsd = 50
# energy_thsd = 0   # 0.9H


eta1 = 50


"""Damage parameters."""

# alpha = 0.21  # weight of max principal stress  in Hayhurst criterion
# beta = 0.63  # weight of von Mises stress in Hayhurst criterion
# r = 0.43  # damage exponent
# B = 5.232e-7  # damage coefficient
# k1, k2 = -2.63, 7.24  # damage rate dependency parameters
Dcr = 0.6  # critical damage
Dmax = 0.99  # maximum damage
# lc = 0.625  # length scale
lc = 10  # length scale


"""Set simulation time and timestepping options."""

t_total = 200  # total time (hours)
t_elapsed = 0  # current elapsed time (hours)
t_delay_dmg = 0  # delay damage (hours)
max_Delta_t = 0.5  # max time increment (hours)
max_Delta_D = 0.1  # max damage increment
output_increment = 5  # number of steps between output
time_counter = 0  # current time step
delta_t = 1

"""Mesh details."""

L, H = 500, 125  # domain dimensions
hs = 0  # water level in crevasse (normalized with crevasse height)
# hw = 0.5*H  # water level at terminus (absolute height)
hw = 0.0*H  # water level at terminus (absolute height)

# mesh = load_mesh(output_dir + "mesh/hdf5/glacier.h5")
mesh = load_mesh(output_dir + "mesh/hdf5/notch.h5")

nd = 2  # mesh dimensions (2D or 3D)

# # Create a Mesh object
# mesh = Mesh()

# # Initialize the XDMFFile object for reading mesh data
# xdmf_file = XDMFFile("mesh/glacier.xdmf")

# # Read the mesh data into the mesh object
# xdmf_file.read(mesh)

# # Close the XDMFFile object
# xdmf_file.close()


"""Define function spaces."""

S1 = FunctionSpace(mesh, "CG", 1)  # first order scalar space
S2 = FunctionSpace(mesh, "CG", 2)  # second order scalar space
V1 = VectorFunctionSpace(mesh, "CG", 1)  # first order vector space
V2 = VectorFunctionSpace(mesh, "CG", 2)  # second order vector space
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)  # first order scalar element
P2 = VectorElement("CG", mesh.ufl_cell(), 2)  # second order vector element
V = FunctionSpace(mesh, MixedElement([P2, P1]))  # mixed finite element
S = FunctionSpace(mesh, "CG", 1)  # first order scalar space


"""Quadrature elements and function spaces."""

deg_quad = 2
scalar_quad = FiniteElement("Quadrature", cell=mesh.ufl_cell(),
                            degree=deg_quad, quad_scheme="default")
vector_quad = VectorElement("Quadrature", cell=mesh.ufl_cell(),
                            degree=deg_quad, quad_scheme="default")
SQ = FunctionSpace(mesh, scalar_quad)  # quadrature points in scalar space
VQ = FunctionSpace(mesh, vector_quad)  # quadrature points in vector space
form_params = {"quadrature_degree": deg_quad}


"""Coordinates of nodes on initial mesh configuration."""

X1, X2 = S1.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
n_local = len(X1)  # number of coordinates on local process
n_global = S1.dim()  # number of coordinates in global system


"""Coordinates of quadrature points on initial mesh configuration."""

XQ1, XQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T  # coordinates
nQ_local = len(XQ1)  # number of quadrature points on local process
nQ_global = SQ.dim()  # number of quadrature points in global system


class left_edge(SubDomain):
    """Boundary on the left domain edge."""
    def inside(self, x, on_boundary): return near(x[0], 0) and on_boundary


class right_edge(SubDomain):
    """Boundary on the right domain edge."""
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary


class bottom_edge(SubDomain):
    """Boundary on the bottom domain edge."""
    def inside(self, x, on_boundary): return near(x[1], 0) and on_boundary


class top_edge(SubDomain):
    """Boundary on the top domain edge."""
    def inside(self, x, on_boundary): return near(x[1], H) and on_boundary


""" Define boundaries and boundary conditions. """

left = left_edge()
right = right_edge()
bottom = bottom_edge()
top = top_edge()

boundaries = FacetFunction("size_t", mesh, 0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
bottom.mark(boundaries, 3)
top.mark(boundaries, 4)
ds = Measure("ds", subdomain_data=boundaries)

free_slip_left = DirichletBC(V.sub(0).sub(0), Constant(0), left)
free_slip_bottom = DirichletBC(V.sub(0).sub(1), Constant(0), bottom)
BC = [free_slip_left, free_slip_bottom]


"""Define loading functions."""


class hydrostatic(Expression):
    """Hydrostatic pressure class (applied as a Neumann BC)."""
    def __init__(self, h=0, **kwargs):
        self.h = float(h)  # water level

    def eval(self, value, x):
        value[0] = -rho_H2O*grav*max(self.h - x[1], 0)

    def value_shape(self):
        return ()


def bodyforce(dmg, y, h=0):
    """Gravity loading as a body force. Fully failed points have
        no density unless they are filled with water. Then they
        have the same density as water.
    """
    b = Function(VQ)  # body force as vector function
    by = Function(SQ)  # y-component of vector function
    ice_points = dmg.vector()[:] < Dcr  # ice material points
    H2O_points = (dmg.vector()[:] > Dcr)*(y <= h)  # water material points
    by.vector()[ice_points] = -rho_ice*grav
    by.vector()[H2O_points] = -rho_H2O*grav
    assign(b.sub(1), by)
    return b


# def pore_pressure(dmg, y, h):
#     """Hydraulic pressure from poro-mechanics formulation."""
#     pHD = Function(SQ)
#     pHD.vector()[:] = rho_H2O*grav*np.fmax(h - y, 0)*dmg.vector().array()
#     return pHD
# def pore_pressure(dmg, y, h):
#     """Hydraulic pressure from poro-mechanics formulation."""
#     pHD = Function(SQ)
#     pHD.vector()[:] = rho_H2O*grav*np.fmax(h - y, 0)*(1- (1 - dmg.vector().array())**2)
#     return pHD

def pore_pressure(dmg,x, y, h):
    """Hydraulic pressure from poro-mechanics formulation."""
    b_poro = Function(SQ)
    dmg_local = dmg.vector().get_local()
    for i in range(nQ_local):
        if dmg_local[i] >= Dcr:
            b_poro.vector()[i] = rho_H2O*grav*np.fmax(h - y[i], 0)*(1-(1-dmg_local[i])**2)
    return b_poro

"""Define constitutive and kinematic relationships."""


def D(u):
    """Symmetric gradient operator."""
    return sym(nabla_grad(u))


def DII(u):
    """Second strain invariant."""
    return (0.5*(D(u)[0, 0]**2 + D(u)[1, 1]**2) + D(u)[0, 1]**2)


def eta(u, n=3, gam=1e-14):
    """Nonlinear viscosity."""
    return 0.5*BT*(DII(u) + gam)**((1 - n)/2/n)


""" Define damage functions. """


def c_bar(dmg=None):
    """Implicit gradient damage constant term."""
    c = Function(SQ)
    if dmg:
        c.vector()[dmg.vector()[:] < Dcr] = 0.5*(lc**2)
    else:
        c.vector()[:] = 0.5*(lc**2)
    return c


def update_psi(dmg):
    """Function for enforcing incompressibility."""
    psi = Function(SQ)
    psi.vector()[:] = 1
    failed_points = dmg.vector()[:] > Dcr  # failed material points
    psi.vector()[failed_points] = 1e-32
    return psi


def surface_crevasse_level(dmg,hrs,y):
    """dynamically compute water level in the surface crevasse"""
    cs = 0
    if hrs > 0:
        x2_dmg = y[dmg.vector().array() > Dcr]
        if len(x2_dmg) > 0:
            x2_dmg_local_max = min(max(x2_dmg), H)
            x2_dmg_local_min = max(min(x2_dmg), 0)
        else:
            x2_dmg_local_min = H
            x2_dmg_local_max = H-10
        x2_dmg_max = comm.allreduce(x2_dmg_local_max, op=MPI.MAX)
        x2_dmg_min = comm.allreduce(x2_dmg_local_min, op=MPI.MIN)
        cs = float(x2_dmg_max - x2_dmg_min)*hs + x2_dmg_min
    return cs

def np_array(x):
    """converts x in to a vector format computed at the nodes"""
    return x.vector().get_local()


def mac(x):
    """Macaulay's bracket"""
    return (x+abs(x))/2

"""Define damage function."""

dmg = Function(SQ)
#dmg.vector()[(L/2 - 2*lc < XQ1)*(XQ1 < L/2 + 2*lc)*(H*(1-0.08) < XQ2)] = Dmax
# dmg.vector()[(L/2 - 5 < XQ1)*(XQ1 < L/2 + 5)*(H - 10 < XQ2)] = Dmax

deg= Function(SQ) #degradation function
phi = Function(SQ) #history variable functionspace
kappa = np.zeros(nQ_local) #history variable
d = TrialFunction(S)
omega = TestFunction(S)

"""Initial guess for Picard iterations."""

uk = Function(V2)  # velocity
pk = Function(S1)  # pressure
cracktip_coor = H - 10


comm.Barrier()


"""Main time loop."""

while time_counter <1:

    time_counter += 1
    # get current configuration coordinates
    x1, x2 = S1.tabulate_dof_coordinates().reshape((-1, nd)).T
    xQ1, xQ2 = SQ.tabulate_dof_coordinates().reshape((-1, nd)).T

    u, p = TrialFunctions(V)  # trial functions in (V2, S1) space
    v, q = TestFunctions(V)  # test functions in (V2, S1) space

    # hydraulic pressure in surface crevasse
    cs = surface_crevasse_level(dmg=dmg,y=xQ2, hrs=hs)  # height of water column
    pHD = pore_pressure(dmg=dmg,x=xQ1, y=xQ2, h=cs)  # hydraulic pressure
    # pHD = pore_pressure(dmg=dmg, y=xQ2, h=cs)  # hydraulic pressure


    # define loading terms
    b_grav = bodyforce(dmg=dmg, y=xQ2, h=cs)  # gravity for ice and water
    b_hw = hydrostatic(h=hw, degree=1)  # terminus pressure

    # normal function to mesh
    nhat = FacetNormal(mesh)

  # incompressibility terms

    penalty = False

    psi = update_psi(dmg)
     ### update degradation function with damage value from previous step
    deg.vector()[:]= np.fmax((1-np_array(dmg))**2,1e-6)

    # define variational form
    LHS = (inner(D(v), 2*deg*eta(uk)*D(u)) - deg*div(v)*p
           + psi*q*div(u))*dx
    if penalty:
        LHS += 1e12*inner(nabla_div(u), psi*div(v))*dx  # penalty term
    # RHS = (1-dmg)*inner(v, b_grav)*dx  # ice and water gravity
    RHS = inner(v, b_grav)*dx  # ice and water gravity
    if hs > 0:
        RHS += inner(div(v), pHD)*dx  # hydraulic pressure in damage zone
    if hw > 0:
        RHS += inner(v, b_hw*nhat)*ds(2)  # terminus pressure

    """ Picard iterations. """

    eps_local = 1  # local error norm
    eps_global = 1  # global error norm
    tol = 1e-5  # error tolerance
    picard_count = 0  # iteration count
    picard_max = 100  # maximum iterations
    w = Function(V)  # empty function to dump solution



    while (abs(eps_global) > tol) and (picard_count < picard_max):
        
        # phase_problem = LinearVariationalProblem(LHS, RHS, w, BC, form_compiler_parameters=form_params)
        # phase_solver = LinearVariationalSolver(phase_problem)

        # prm_phase = phase_solver.parameters
        # prm_phase["linear_solver"] = "cg"
        # prm_phase["preconditioner"] = "hypre_euclid"

        # solve(LHS == RHS, w, BC,form_compiler_parameters=form_params,
        # solver_parameters={'linear_solver': 'gmres',
        #                  'preconditioner': 'ilu'})

        # phase_solver.solve()
        # solve the variational form
        solve(LHS == RHS, w, BC, form_compiler_parameters=form_params)
        u, p = w.split(deepcopy=True)
        u1, u2 = u.split(deepcopy=True)

        # compute error norms
        u1k, u2k = uk.split(deepcopy=True)
        diff1 = u1.vector().array() - u1k.vector().array()
        diff2 = u2.vector().array() - u2k.vector().array()
        diffp = p.vector().array() - pk.vector().array()
        eps1 = np.linalg.norm(diff1)/np.linalg.norm(u1.vector().array())
        eps2 = np.linalg.norm(diff2)/np.linalg.norm(u2.vector().array())
        epsp = np.linalg.norm(diffp)/np.linalg.norm(p.vector().array())
        # eps1 = np.linalg.norm(diff1)
        # eps2 = np.linalg.norm(diff2)
        # epsp = np.linalg.norm(diffp)

        # update solution for next iteration
        assign(uk, u)
        assign(pk, p)

        comm.Barrier()

        # obtain the max error on the local process
        eps_local = max(eps1, eps2, epsp)
        # obtain the max error on all processes
        eps_global = comm.allreduce(eps_local, op=MPI.MAX)

        # update iteration count
        picard_count += 1
    print(" Steps, eps1, eps2, eps3, epslocal: %d, %9f, %9f, %9f",picard_count, eps1, eps2, epsp, eps_local )

    if rank == 0:
        with open(output_dir + "output/details/simulation_log.txt",
                  "a") as sim_log:
            sim_log.write("\nTime step "
                          "%d: %g hours\n" % (time_counter, t_elapsed))
            if picard_count < picard_max:
                sim_log.write("Convergence after "
                              "%d Picard iterations.\n" % picard_count)
            else:
                sim_log.write("WARNING: no convergence after "
                              "%d Picard iterations!\n" % picard_count)

    """ Generate numpy arrays from output. """

    # build effective deviatoric stress tensor
    tau = 2*eta(u)*D(u)
    t11 = project(tau[0, 0], SQ,
                  form_compiler_parameters=form_params).vector().array()
    t22 = project(tau[1, 1], SQ,
                  form_compiler_parameters=form_params).vector().array()
    t33 = np.zeros(nQ_local)
    t12 = project(tau[0, 1], SQ,
                  form_compiler_parameters=form_params).vector().array()

    dmg0 = dmg.vector().array()  # damage from previous time step
    prs = interpolate(p, SQ).vector().array()  # effective pressure

    t23 = np.zeros(nQ_local)

    t13 = np.zeros(nQ_local)


    # effective Cauchy stress
    s11, s22, s33, s12, s23, s13 = t11 - prs, t22 - prs, t33 - prs, t12, t23, t13

    energy= np.zeros(nQ_local)
        #   # find eigen values of strain tensor
    # for i in range(nQ_local):
    #     # s_i= np.array([[s11[i],s12[i]],[s12[i],s22[i]]])
    #     s_i= np.array([[s11[i],s12[i], s13[i]],[s12[i],s22[i], s23[i]], [s13[i],s23[i], s33[i]]])

    #     eig,vec= np.linalg.eigh(s_i)
    #     # Clayton
    #     # energy[i] = ci*mac((mac(eig[0])**2 + (mac(eig[1]))**2)/(sigmac**2) - 1)
    #     energy[i] = ci*mac((mac(eig[0])**2 + (mac(eig[1]))**2 + mac(eig[2])**2)/(sigmac**2)-1)
    #     if energy[i] <= energy_thsd:
    #         energy[i] = 0



    # # update history variable
    # kappa = np.fmax(energy, kappa)
    # phi.vector()[:] = kappa  # Crack driving force

    # #define variational problem for phase-field
    # Lhs= ((Constant(eta1/delta_t +1/lc)+2*phi)*inner(d,omega)*dx
    # + Constant(lc)*inner(nabla_grad(d),nabla_grad(omega))*dx)
    # Rhs= (inner(omega,2*phi)*dx + Constant(eta1/delta_t)*inner(omega,dmg)*dx)

    # #solve the variational problem
    # w= Function(S)
    # solve(Lhs==Rhs,w, form_compiler_parameters=form_params)

    # #update damage
    # dmgv = np.clip(interpolate(w, SQ).vector().get_local(), 0, 1)
    # dmg.vector()[:] = np.fmax(dmgv, dmg.vector().get_local())


    # #update counter
    # # time_counter+= 1

    # #write Output data to mesh file.
    # if ((time_counter - 1) % output_increment == 0 or time_counter >= 300 ):
    #     # data = {
    #     #     "displacement": disp,
    #     #     "damage": dmg,
    #     #     "crack driving force": Function(SQ),
    #     #     }
    #     # data["crack driving force"].vector()[:] = kappa
    #     # write_hdf5(time_counter, mesh, data, time_elapsed)

    #     dmgc = dmg.vector().get_local()
    #     for i in range(nQ_local):
    #         if dmgc[i] >= 0.99:
    #         # if dmgc[i] >= Dcr:
    #             if XQ2[i] < cracktip_coor:
    #                 cracktip_coor = XQ2[i]
    #     print("Step %d, Crack depth: %g" % (time_counter,1-cracktip_coor/H))

    """ Updated Lagrangian implementation. """

    # split velocity into components in S1 space
    u1, u2 = u.split(deepcopy=True)
    u1 = interpolate(u1, S1).vector().array()
    u2 = interpolate(u2, S1).vector().array()

    # compute the displacement increment vector Delta_u
    Delta_u1 = Function(S1)
    Delta_u2 = Function(S1)
    ind1 = x1 > 0  # indices of coordinates where x1 > 0
    ind2 = x2 > 0  # indices of coordinates where x2 > 0
    Delta_u1.vector()[ind1] = u1[ind1]*delta_t
    Delta_u2.vector()[ind2] = u2[ind2]*delta_t
    Delta_u = Function(V1)
    assign(Delta_u.sub(0), Delta_u1)
    assign(Delta_u.sub(1), Delta_u2)

    # move the mesh, update coordinates
    ALE.move(mesh, Delta_u)


    #update time
    t_elapsed+= delta_t

# # dmg = project(dmg, S1, form_compiler_parameters=form_params)
# # plot_11 = plot(dmg)
# # plt.colorbar(plot_11)
# # plt.savefig('dmg_vis_e.jpg', dpi=300)
# # # u_plot = project(tem, S)
jet = cm.jet
rainbow = cm.rainbow
redblue = cm.coolwarm
viridis = cm.viridis
plasma = cm.plasma
magma = cm.magma

plt.figure(figsize=(20, 5))
# tem = dmg.vector().array()
plot = plt.tricontourf(XQ1, XQ2, s11, 360, cmap=rainbow)
plt.colorbar(plot)
plt.savefig('s11_hw_00.jpg', dpi=300)

# # print(max((s11)))
# plt.figure(figsize=(20, 5))
# tem = dmg.vector().array()
# plot12 = plt.tricontourf(XQ1, XQ2, tem, 360, cmap=rainbow)
# # phi = project(phi, SQ,form_compiler_parameters=form_params)
# # plot12 = plot(phi)
# plt.colorbar(plot12)
# plt.savefig('dmg.jpg', dpi=300)

# s11 = project(s11, SQ, form_compiler_parameters=form_params)
# # # Create a File object to save the PVD file
# file = File("s11.pvd")

# # Save the function to the PVD file
# file << s11
# s1x = project(tau[0, 0] - p, SQ,form_compiler_parameters=form_params)
# plot11 = plot(s1x)
# plt.colorbar(plot11)
# plt.savefig('s1x.jpg', dpi=300)


# # Create a File object to save the PVD file
# file1 = File("phi.pvd")

# # Save the function to the PVD file
# file1 << phi
print(max(np_array(phi)))