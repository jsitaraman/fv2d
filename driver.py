from Mesh import Mesh
from Solver import Solver
# read the mesh (TODO: read BC's also)
mesh = Mesh("finer.ugrid")
# create the solver
solver= Solver(mesh)
print("-- testing cell wise gradient reconstruction --")
solver.test_gradients()
print("-- testing nodal gradients --")
solver.test_nodal_gradients()
# initialize the isentropic vortex data
solver.initData()
# show some plots
solver.output()
dt=0.02
for i in range(50):
   print(f"i={i}")
   solver.advance(dt)
solver.output()
