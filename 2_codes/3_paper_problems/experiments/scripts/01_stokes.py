import sys
import os

# Add src to the system path to allow imports from the src directory
notebook_dir = os.getcwd()
src_path = os.path.abspath(os.path.join(notebook_dir, '../..', 'src'))
sys.path.insert(0, src_path)

from utils import *
import my_mesh
import params
import physics

set_log_active(False)

# output directory
output_dir = "output"

# Material parameters
m = params.Params()

# load mesh
mprint("Rank: ", rank, " Size: ", size)
msh = my_mesh.MyMesh(m)

# Define output files
xdmf = XDMFFile("Solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["rewrite_function_mesh"] = True
xdmf.parameters["functions_share_mesh"] = True


phys = physics.Physics(msh, m)
phys.Define_BC()

dZmax = 5e-2 #max crack depth increment

t_elapsed = 0  # current elapsed time (s)
time_counter = 0  # current time step
tVec = [0.0]
cVec = [msh.dZ0/msh.H]  # initial crack depth
ctVec = [phys.GetCrackLength()/msh.H] # total combined crack lengths

# Time loop
counter = 0
m.print_params()
# while t_elapsed < m.t_total:
for i in range(1): # for testing purposes
    time_counter += 1
    m.Delta_t = m.Delta_t*1.1

    AcceptableDT = False
    while AcceptableDT==False:
        for i in range(m.nPasses):
            phys.SolveStokes()
            phys.CalculateDrivingEnergy()
            phys.SolveDamage()
        print(phys.u.vector().max(), " ", phys.u.vector().min())
        print(phys.p.vector().max(), " ", phys.p.vector().min())
        print(phys.dmg.vector().max(), " ", phys.dmg.vector().min())
        print("-"*80)
        TCL = phys.GetCrackLength()/msh.H
        mprint(phys.GetCrackLength()," ", msh.H," ", TCL)
        dTCL = TCL-ctVec[-1]
        print("TCL: %g, dTCL: %g, Crack depth dz: %g" % (TCL, dTCL, msh.dZ0/msh.H))

       
        if (dTCL < 2*m.dx/msh.H or m.Delta_t<1.0e-3):
            AcceptableDT = True
        else:
            m.Delta_t = 0.5*m.Delta_t
            mprint("\t Reducing time step to %f, Crack depth dz: %g " % (m.Delta_t, dTCL))
        AcceptableDT = True       
    
    # update elapsed time
    t_elapsed += m.Delta_t
    phys.CommitHistory()
    
    #print info
    z = 1.0-phys.GetCrackTip()/msh.H
    if (z < cVec[-1]):
        z=cVec[-1] #correct for geometric crack
    print(z)
    cVec.append(z)
    ctVec.append(TCL)
    tVec.append(t_elapsed)
    
    mprint("Simulation time %f, Crack depth: %g " % (t_elapsed,cVec[-1]))
    print(phys.GetCrackTip())
    # plotting 
    if (time_counter % 1 == 0):
        phys.Plot(tVec, cVec, ctVec)
        phys.SaveOutputs(t_elapsed)

# #pause to show all plots
# plt.pause(0.001)
# input("Press [enter] to continue.")