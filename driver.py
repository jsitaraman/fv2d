from Mesh import Mesh
from Solver import Solver
import numpy as np
# read the mesh (TODO: read BC's also)
mesh = Mesh("coarse.ugrid")
# create the solver
solver= Solver(mesh)
print("\n-- testing cell wise gradient reconstruction --")
solver.test_gradients()
print("\n-- testing nodal gradients --")
solver.test_nodal_gradients()
# initialize the isentropic vortex data
solver.initData()
print("\n-- check linearization of fluxes --")
solver.checkJac_fd()
print("\n-- check jacobian products ---")
solver.jacobian_check(solver.q)
# show some plots
solver.output()
dt=0.5
nsubit=5
nlinear=10
nsteps=100
time=0
explicit=False
#
if not explicit:
  qn=np.zeros_like(solver.q)
  qnn=qn.copy()
for i in range(nsteps):
   print(f"i={i}")
   time+=dt
   if explicit:
      solver.advance(dt)
   else:
      solver.shiftTime(dt,solver.q,qn,qnn)
      for s in range(nsubit):
         R,dtfac=solver.getUnsteadyResidual(qn,qnn,dt)
         print(f"\tsubiter{s} {np.linalg.norm(R)}")
         D=solver.getDiagJacobian(dt,dtfac)
         dq=solver.linearIterations(D, R, nlinear)
         solver.update(dq)
solver.output()
