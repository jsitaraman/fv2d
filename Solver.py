import numpy as np
from IsentropicVortex import IsentropicVortex
from solver_utils import solver_utils
class Solver:
    def __init__(self, mesh):
        self.su=solver_utils()
        self.xp=np
        # this is element to element connectivity
        self.mesh=mesh
	# info of the mesh
        mesh.info()
        self.nelem=mesh.n_owned
        self.nnodes=mesh.coords.shape[0]
        # this ghost map contains both edge and vertex neighbors
        self.ghostMap=mesh.add_halo_cells()
        mesh.remove_duplicate_nodes_tol()
        # get the coordinates and element connectivity
        self.coords = mesh.coords
        self.elems  = mesh.elems
        # optionally plot the mesh
        mesh.info()
        #mesh.plot2d()        
        # element 2 element connectivity
        self.elem2elem = mesh.build_elem2elem()
        # centroids
        self.centroids,self.areas = mesh.compute_centroids_and_areas()
        self.areas=self.areas.reshape((self.areas.shape[0],1))
        # number of halo cells and nodes
        self.nhalo=mesh.elems.shape[0]-self.nelem
        self.nnodes_halo = mesh.coords.shape[0]-self.nnodes
        # node2element graph with a nodemask to say which ones exist
        self.node2elem,self.nodemask = mesh.build_node2elem_dense_vec()
        # these are interior and boundary edges, but no halo edges
        self.edges,self.edge2elem,self.elem2edge = mesh.build_edge_to_elem()
        #
        self.boundary_edges = []
        for eidx, (left, right) in enumerate(self.edge2elem):
            if left < self.nelem and right >= self.nelem :   # boundary edge
                self.boundary_edges.append((eidx, left))
        # --- geometry on edges ---
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        x0 = self.coords[i0,:2]                   # (Nedge,2)
        x1 = self.coords[i1,:2]                   # (Nedge,2)
        evec = x1 - x0                # edge vector
        # base (left-handed) unit normal (rotated +90°): n_perp = [ey, -ex] / L
        self.ds = self.xp.stack((evec[:,1], -evec[:,0]), axis=1)
        self.faceVel=np.zeros((self.ds.shape[0],))
        self.xf = 0.5 * (x0 + x1)      # edge midpoint
                        
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
        self.vortex=IsentropicVortex(use_cupy=False)
        self.q=self.vortex.init_Q(X,x0=10.0,y0=5,uinf=0.5,vinf=0)
        # residual storage
        self.R=np.zeros_like(self.q)
        
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
        q = np.vstack((q,np.vstack((q,q)))).T
        # -----------------------------
        # 2. Gather stencils (elem2elem)
        # -----------------------------
        max_neigh = self.elem2elem.shape[1]
        Ne=self.nelem
        ndim = self.centroids.shape[1]
        # --- neighbor indices, replacing -1 with self indices ---
        idx = np.where(self.elem2elem[:Ne,:] >= 0,
                       self.elem2elem[:Ne,:], np.arange(Ne)[:, None])
        # --- differences in centroids (dx = neighbor - self) ---
        dx = self.centroids[idx] - self.centroids[:Ne, None, :]   # (Ne, max_neigh, ndim)
        # --- mask: 1 if neighbor is real, 0 if padded ---
        mask = np.array((self.elem2elem[:Ne,:] >= 0))
        #
        weights = self.su.lsq_wts_batched(dx, mask)
        # -----------------------------
        # 4. Compute gradients
        # -----------------------------
        dq =  q[idx,...] 
        grads = np.einsum('nij,nik->nkj', weights, dq)
        # -----------------------------
        # 5. Verification
        # -----------------------------
        exact = np.array([a, b])
        error = grads - exact[None, :]
        l2err = np.sqrt(np.mean(np.sum(error**2, axis=1)))
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
        grads = np.einsum('nij,ni->nj', weights, dq)

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
        idx = np.where(self.elem2elem[:Ne,:] >= 0,
                       self.elem2elem[:Ne,:], np.arange(Ne)[:, None])
        self.idx=idx
        # --- differences in centroids (dx = neighbor - self) ---
        self.dx = self.centroids[idx] - self.centroids[:Ne, None, :] # (Ne, max_neigh, ndim)
        # --- mask: 1 if neighbor is real, 0 if padded ---
        mask = np.array((self.elem2elem[:Ne,:] >= 0))
        #
        self.weights = self.su.lsq_wts_batched(self.dx, mask)
        # zero out gradients for halo cells for now
        self.weights[self.nelem:,...]=0
        
    def residual(self,q):
        self.computeGradientWeights()
        dq=q[self.idx]
        # compute least-square gradients by taking the tensor product
        # with LSQ weights
        grads = np.einsum('nij,nik->nkj', self.weights, dq)
        # compute the residual
        R=self.su.residual(q,
                           self.centroids[:,:2],
                           self.coords[:,:2], 
                           self.edges,
                           self.edge2elem, self.nelem, gradQ=grads)
        return -R/self.areas
    
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
            
