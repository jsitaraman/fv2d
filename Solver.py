import numpy as np
from IsentropicVortex import IsentropicVortex
from solver_utils import solver_utils
class Solver:
    def __init__(self, mesh, use_cupy=False):
        self.use_cupy=use_cupy
        try:
          self.xp = __import__('cupy' if use_cupy else 'numpy')
        except:
          self.use_cupy=False
          self.xp = np
        self.su=solver_utils(use_cupy=self.use_cupy)
        #self.xp=np
        self.mesh=mesh

        # info of the mesh
        mesh.info()
        self.nelem=mesh.n_owned
        self.nnodes=mesh.coords.shape[0]

        # this ghost map contains both edge and vertex neighbors
        self.ghostMap=mesh.add_halo_cells()
        mesh.remove_duplicate_nodes_tol()

        # get the coordinates and element connectivity
        self.coords = self.xp.asarray(mesh.coords)
        self.elems  = self.xp.asarray(mesh.elems)

        # optionally plot the mesh
        print("\n..after adding ghosts and removing duplicates \n")
        mesh.info()
        #mesh.plot2d()        

        # element 2 element connectivity
        self.elem2elem = mesh.build_elem2elem()
        self.elem2elem = self.xp.asarray(self.elem2elem)

        # centroids
        self.centroids,self.areas = mesh.compute_centroids_and_areas()
        self.centroids = self.xp.asarray(self.centroids)
        self.areas=self.areas.reshape((self.areas.shape[0],1))
        self.areas     = self.xp.asarray(self.areas)

        # number of halo cells and nodes
        self.nhalo=mesh.elems.shape[0]-self.nelem
        self.nnodes_halo = mesh.coords.shape[0]-self.nnodes

        # node2element graph with a nodemask to say which ones exist
        self.node2elem,self.nodemask = mesh.build_node2elem_dense_vec()
        self.node2elem=self.xp.asarray(self.node2elem)
        self.nodemask=self.xp.asarray(self.nodemask)
        
        # these are interior and boundary edges, but no halo edges
        self.edges,self.edge2elem,self.elem2edge = mesh.build_edge_to_elem()
        self.edges=self.xp.asarray(self.edges)
        self.edge2elem=self.xp.asarray(self.edge2elem)

        # boundary edges
        self.boundary_edges = []
        for eidx, (left, right) in enumerate(self.edge2elem):
            if left < self.nelem and right >= self.nelem :   # boundary edge
                self.boundary_edges.append((eidx, left))
                                
    def initData(self):
        #
        self.props={'gamma' : 1.4,
                    'uinf'  : 0.5,
                    'vinf'  : 0.0,
                    'rinf'  : 1.0,
                    'pinf'  : 1.0/1.4}
        self.gamma= self.props['gamma']
        self.uinf = self.props['uinf']
        self.vinf = self.props['vinf']
        self.rinf = self.props['rinf']
        self.pinf = self.props['pinf']
        self.su.setParams(self.props)
        self.time=0
        self.nq=4
        # generate an isentropic vortex
        X=self.centroids[:,:2]
        self.vortex=IsentropicVortex(use_cupy=self.use_cupy)
        self.q=self.vortex.init_Q(X,x0=10.0,y0=5,uinf=0.5,vinf=0)
        # residual storage
        self.R=self.xp.zeros_like(self.q)
        
    def output(self):
        self.su.plot_field_with_cut(self.coords[:,:2],self.elems,self.q, y0=5.0,
                                    time=self.time)
        # tecplot_file='test.dat')


    def test_gradients(self):
        """
        Verification of LSQ gradient reconstruction using a linear field.
        """        
        # -----------------------------
        # 1. Create a linear field
        # -----------------------------
        cx, cy = self.centroids[:, 0], self.centroids[:, 1]
        # Linear field: f(x,y) = ax + by + c
        a, b, c = 2.0, -1.0, 0.5
        q = a * cx + b * cy + c
        q = self.xp.vstack((q,self.xp.vstack((q,q)))).T
        # -----------------------------
        # 2. Gather stencils (elem2elem)
        # -----------------------------
        max_neigh = self.elem2elem.shape[1]
        Ne=self.nelem
        ndim = self.centroids.shape[1]
        # --- neighbor indices, replacing -1 with self indices ---
        idx = self.xp.where(self.elem2elem[:Ne,:] >= 0,
                       self.elem2elem[:Ne,:], self.xp.arange(Ne)[:, None])
        # --- differences in centroids (dx = neighbor - self) ---
        dx = self.centroids[idx] - self.centroids[:Ne, None, :]   # (Ne, max_neigh, ndim)
        # --- mask: 1 if neighbor is real, 0 if padded ---
        mask = self.xp.array((self.elem2elem[:Ne,:] >= 0))
        #
        weights = self.su.lsq_wts_batched(dx, mask)
        # -----------------------------
        # 4. Compute gradients
        # -----------------------------
        dq =  q[idx,...] 
        grads = self.xp.einsum('nij,nik->nkj', weights, dq)
        # -----------------------------
        # 5. Verification
        # -----------------------------
        exact = self.xp.array([a, b])
        error = grads - exact[None, :]
        l2err = self.xp.sqrt(self.xp.mean(self.xp.sum(error**2, axis=1)))
        print(f"Expected gradient = {exact}")
        print(f"Recovered mean gradient = {grads.mean(axis=0)}")
        print(f"L2 error = {l2err:.3e}")

    def test_nodal_gradients(self):
        """
        Verification of LSQ gradient reconstruction using a linear field.
        """        
        # -----------------------------
        # 1. Create a linear field
        # -----------------------------
        cx, cy = self.centroids[:, 0], self.centroids[:, 1]
        # Linear field: f(x,y) = ax + by + c
        a, b, c = 2.0, -1.0, 0.5
        q = a * cx + b * cy + c
        # -----------------------------
        # 2. Gather stencils (elem2elem)
        # -----------------------------
        # Node coordinates
        x_nodes = self.coords[:self.nnodes,:2]  # (nnodes, dim)        

        # Element centroids (assume already computed)
        x_elem = self.centroids  # (Ne, dim)
        
        # For each node, gather the neighboring element centroids
        # Use advanced indexing
        neighbor_idx = self.node2elem[:self.nnodes,:]  # (nnodes, max_degree)
        neighbor_mask = self.nodemask[:self.nnodes,:]    # (nnodes, max_degree)

        # Replace invalid (-1) indices with 0 to allow safe indexing
        safe_idx = neighbor_idx.copy()
        safe_idx[~neighbor_mask] = 0

        # Gather relative positions: dx = x_elem[elem_j] - x_node_i
        dx = x_elem[safe_idx] - x_nodes[:, self.xp.newaxis, :]  # (nnodes, max_degree, dim)

        # Zero-out invalid entries
        dx[~neighbor_mask] = 0.0
        #
        weights = self.su.lsq_wts_batched(dx, neighbor_mask)
        # -----------------------------
        # 4. Compute gradients
        # -----------------------------
        dq =  q[safe_idx]
        grads = self.xp.einsum('nij,ni->nj', weights, dq)

        # -----------------------------
        # 5. Verification
        # -----------------------------
        exact = self.xp.array([a, b])
        error = grads - exact[None, :]
        l2err = self.xp.sqrt(self.xp.mean(self.xp.sum(error**2, axis=1)))

        print(f"Expected gradient = {exact}")
        print(f"Recovered mean gradient = {grads.mean(axis=0)}")
        print(f"L2 error = {l2err:.3e}")

    def computeGradientWeights(self):
        if hasattr(self,'weights'):
            return
        # find gradient weights
        max_neigh = self.elem2elem.shape[1]
        Ne=self.elem2elem.shape[0]
        ndim = self.centroids.shape[1]
        # --- neighbor indices, replacing -1 with self indices ---
        idx = self.xp.where(self.elem2elem[:Ne,:] >= 0,
                       self.elem2elem[:Ne,:], self.xp.arange(Ne)[:, None])
        self.idx=idx
        # --- differences in centroids (dx = neighbor - self) ---
        self.dx = self.centroids[idx] - self.centroids[:Ne, None, :] # (Ne, max_neigh, ndim)
        # --- mask: 1 if neighbor is real, 0 if padded ---
        mask = self.xp.array((self.elem2elem[:Ne,:] >= 0))
        #
        self.weights = self.su.lsq_wts_batched(self.dx, mask)
        # zero out gradients for halo cells for now
        self.weights[self.nelem:,...]=0
        
    def residual(self,q , use_grad=True):
        self.computeGradientWeights()
        dq=q[self.idx]
        # compute least-square gradients by taking the tensor product
        # with LSQ weights
        if use_grad:
            grads = self.xp.einsum('nij,nik->nkj', self.weights, dq)
        else:
            grads = None
        # compute the residual
        R=self.su.residual(q,
                           self.centroids[:,:2],
                           self.coords[:,:2], 
                           self.edges,
                           self.edge2elem, self.nelem, gradQ=grads)
        return -R/self.areas

    def fd_check(self, Qc, dq):
        """Finite-difference check for Jacobian–vector product."""
        R0 = self.residual(Qc, use_grad=False)       # baseline residual
        R1 = self.residual(Qc + dq, use_grad=False)  # perturbed residual
        return (R1 - R0)                             # approximate J*dq

    def checkJac_fd(self):
        xp=self.xp
        eps=1e-8
        dq_test = xp.random.randn(self.q.shape[0],4)*eps
        dq_test[self.nelem:,...]=0
        Jdq_fd = self.fd_check(self.q,dq_test)
        Ff=self.su.fullJacobianProduct(dq_test,self.q,self.centroids[:,:2],self.coords[:,:2],
                                       self.edges,self.edge2elem, self.nelem)
        Ff=-Ff/self.areas
        # print("Jdq_fd=",Jdq_fd)
        # print("Ff=",Ff)
        print("‖FD - Matvec‖ =", xp.linalg.norm(Jdq_fd[:self.nelem,:] - Ff[:self.nelem,:]))
        
    def jacobian_check(self,q):
        xp=self.xp
        eps=1e-8
        dq_test = xp.random.randn(self.q.shape[0],4)*eps
        dq_test[self.nelem:,...]=0

        Fd=self.su.diagProduct(dq_test,q,self.centroids[:,:2],self.coords[:,:2],
                               self.edges,self.edge2elem, self.nelem)
        Fo=self.su.offDiagProduct(dq_test,q,self.centroids[:,:2],self.coords[:,:2],
                                  self.edges,self.edge2elem,self.nelem)
        Ff=self.su.fullJacobianProduct(dq_test,q,self.centroids[:,:2],self.coords[:,:2],
                                       self.edges,self.edge2elem, self.nelem)
        D=self.su.diagJacobians(q,self.centroids[:,:2],self.coords[:,:2],
                                self.edges,self.edge2elem,self.nelem)
        print("‖(Fd+Fo) - Ff‖ =", xp.linalg.norm((Fd + Fo) - Ff))
        print("‖(Fd - Dq‖ =", xp.linalg.norm(Fd - self.xp.einsum('nij,nj->ni',D,dq_test)))

    def getDiagJacobian(self,dt, factor):
        D=self.su.diagJacobians(self.q,self.centroids[:,:2],self.coords[:,:2],
                                self.edges,self.edge2elem,self.nelem)
        D/=self.areas[:,None]
        alpha=factor/dt
        for i in range(self.nq):
            D[:, i, i] += alpha
        return D

    def getUnsteadyResidual(self, qn, qnn, dt):
        R=self.residual(self.q)
        factor=1.5
        if self.time > dt:
            R -= (3*self.q - 4*qn + qnn)/(2*dt)            
        else:
            R -= (self.q - qn)/dt
            factor=1
        return R, factor
        
    def advance(self,dt,scheme="RK3"):
        self.time+=dt
        if scheme=="ForwardEuler":
            R=self.residual(self.q)
            self.q += R*dt
        elif scheme=="RK3":
            """
            Low-storage RK scheme from dgsand.c
            """
            rk = [0.25, 8/15, 5/12, 3/4]
            # Stage 1
            R = self.residual(self.q)
            self.qstar = self.q + rk[1]*dt*R
            self.q     = self.q + rk[0]*dt*R
            # Stage 2
            R = self.residual(self.qstar)
            self.qstar = self.q + rk[2]*dt*R
            # Stage 3
            R = self.residual(self.qstar)
            self.q = self.q + rk[3]*dt*R
        elif scheme=="SSP-RK3":
            qn = self.q.copy()            
            # --- Stage 1
            R = self.residual(self.q)
            q1 = qn + dt * R            
            # --- Stage 2
            self.q = q1
            R = self.residual(self.q)
            q2 = 0.75 * qn + 0.25 * (q1 + dt * R)            
            # --- Stage 3
            self.q = q2
            R = self.residual(self.q)
            self.q = (1.0/3.0) * qn + (2.0/3.0) * (q2 + dt * R)
        self.applyBC()
        print(f"\t Residual {np.linalg.norm(R)}")
            
    def linearIterations(self, D, R, nlinear, dq=None):
        dq=self.xp.zeros_like(R)
        for k in range(nlinear):
            if k > 0:
                Fo=self.su.offDiagProduct(dq,self.q,self.centroids[:,:2],self.coords[:,:2],
                                          self.edges,self.edge2elem,self.nelem)
                B = R-Fo/self.areas
            else:
                B = R
            Ddq = self.xp.einsum('nij,nj->ni',D,dq)
            print(f"\t\t linear{k} {self.xp.linalg.norm(B-Ddq)}")
            dq = self.xp.linalg.solve(D, B[..., None])[..., 0]
        return dq

    def shiftTime(self,dt,q,qn,qnn):
        qnn[:]=qn[:]
        qn[:]=q[:]
        self.time+=dt

    def applyBC(self):
        X=self.centroids[self.nelem:,:2]
        bc_q = self.vortex.init_Q(X,x0=10+self.uinf*self.time,y0=5,uinf=self.uinf,vinf=0)
        self.q[self.nelem:,:] = bc_q
        
    def update(self,dq):
        self.q+=dq
        self.applyBC()
        
