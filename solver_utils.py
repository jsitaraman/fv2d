from IsentropicVortex import IsentropicVortex
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata
import numpy as np
from flux import roeflux,  flux_jacobians_fd

class solver_utils:
    def __init__(self, use_cupy=False):
        try:
            self.xp = __import__('cupy' if use_cupy else 'numpy')
        except:
            self.xp = __import__('numpy')
        
    def setParams(self, props):
        self.gamma=props['gamma']
        self.uinf=props['uinf']
        self.vinf=props['vinf']
        self.rinf=props['rinf']
        self.pinf=props['pinf']

    def plot_field_with_cut(self, Xv, E, Qc, field="rho", y0=0.0, levels=10, 
                            time=0, show_mesh=True,plot_exact=True, cmap="viridis"):
        """
        Project cell-centered data to nodes, plot contour field and a line cut.

        Args:
            Xv : (Nv,2) vertex coordinates
            E  : (Ne,4) element connectivity (tris padded with -1)
            Qc : (Ne,4) conservative variables at centroids
            field : which scalar to plot ("rho","p","u","v","mach")
            y0 : y-coordinate of horizontal cut line
            levels : number of contour levels
            cmap : colormap
        """
        gamma = self.gamma
        rho = Qc[:, 0]
        u   = Qc[:, 1] / rho
        v   = Qc[:, 2] / rho
        Econs = Qc[:, 3]
        p   = (gamma - 1.0) * (Econs - 0.5 * rho * (u**2 + v**2))
        a   = np.sqrt(gamma * p / rho)
        mach = np.sqrt(u**2 + v**2) / a

        if field == "rho":
            values = rho
        elif field == "p":
            values = p
        elif field == "u":
            values = u
        elif field == "v":
            values = v
        elif field == "mach":
            values = mach
        else:
            raise ValueError(f"Unknown field: {field}")

        Nv = Xv.shape[0]
        qnodal = np.zeros(Nv)
        counts = np.zeros(Nv)

        # --- project to nodes by averaging ---
        for ei, conn in enumerate(E):
            if conn[3] == -1:  # triangle
                nodes = conn[:3]
            else:              # quad
                nodes = conn
            qnodal[nodes] += values[ei]
            counts[nodes] += 1.0
        qnodal /= np.maximum(counts, 1.0)

        # --- triangulation (split quads into 2 tris) ---
        tris = []
        for conn in E:
            if conn[3] == -1:
                tris.append([conn[0],conn[1],conn[2]])
            else:
                tris.append([conn[0], conn[1], conn[2]])
                tris.append([conn[0], conn[2], conn[3]])
        tris = np.array(tris)
        triang = mtri.Triangulation(Xv[:,0], Xv[:,1], tris)
        # --- make plots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

        # contour plot
        tcf = ax1.tricontourf(triang, qnodal, levels=levels, cmap=cmap)
        fig.colorbar(tcf, ax=ax1, label=field, orientation='horizontal',pad=0.1)
        if show_mesh:
            ax1.triplot(triang, color="k", lw=0.3, alpha=0.5)
        ax1.set_aspect("equal")
        ax1.set_title(f"{field} contours with mesh")
        ax1.set_aspect("equal")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.axhline(y0, color="r", linestyle="--", lw=1)

        # --- interpolated line cut at y = y0 ---
        xmin, xmax = Xv[:,0].min(), Xv[:,0].max()
        xline = np.linspace(xmin, xmax, 400)
        yline = np.full_like(xline, y0)
        qline = griddata(Xv, qnodal, (xline, yline), method="linear")

        ax2.plot(xline, qline, "-", lw=1.5)
        ax2.set_title(f"{field} at y={y0}")
        if plot_exact:
            # TODO this is hard coded for vortex and
            # its start location at (10,5) now
            vortex=IsentropicVortex()
            qline_exact=vortex.init_Q(np.vstack((xline,yline)).T,
                                      x0=10.0+self.uinf*time,y0=5,
                                      uinf=self.uinf,vinf=self.vinf)
            ax2.plot(xline, qline_exact[:,0], "r-", lw=1.5)
        ax2.set_xlabel("x")
        ax2.set_ylabel(field)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        
    def plot_field(self,Xv, E, Qc, field="rho", levels=30, cmap="viridis", show_mesh=True, tecplot_file=None):
        """
        Project cell-centered values to nodes, plot filled contours, optionally overlay mesh,
        and optionally write Tecplot ASCII file.

        Args:
            Xv : (Nv,2) vertex coordinates
            E  : (Ne,4) element connectivity (tris padded with -1)
            Qc : (Ne,4) conservative variables at centroids
            field : scalar to plot ("rho","p","u","v","mach")
            levels : number of contour levels
            cmap : colormap
            show_mesh : if True, overlay mesh edges
            tecplot_file : str or None, if provided writes Tecplot ASCII file
        """
        gamma = self.gamma
        rho = Qc[:, 0]
        u   = Qc[:, 1] / rho
        v   = Qc[:, 2] / rho
        Econs = Qc[:, 3]
        p   = (gamma - 1.0) * (Econs - 0.5 * rho * (u**2 + v**2))
        a   = np.sqrt(gamma * p / rho)
        mach = np.sqrt(u**2 + v**2) / a

        # --- select field ---
        if field == "rho":
            values = rho
        elif field == "p":
            values = p
        elif field == "u":
            values = u
        elif field == "v":
            values = v
        elif field == "mach":
            values = mach
        else:
            raise ValueError(f"Unknown field: {field}")

        # --- project to nodes ---
        Nv = Xv.shape[0]
        qnodal = np.zeros(Nv)
        counts = np.zeros(Nv)
        for ei, conn in enumerate(E):
            if conn[3] == -1:
                nodes = conn[:3]
            else:
                nodes = conn
            qnodal[nodes] += values[ei]
            counts[nodes] += 1.0
        qnodal /= np.maximum(counts, 1.0)

        # --- build triangulation (split quads into two tris) ---
        tris = []
        for conn in E:
            if conn[3] == -1:
                tris.append(conn[:3])
            else:
                tris.append([conn[0], conn[1], conn[2]])
                tris.append([conn[0], conn[2], conn[3]])
        tris = np.array(tris)

        triang = mtri.Triangulation(Xv[:,0], Xv[:,1], tris)
        #interp = mtri.LinearTriInterpolator(triang, qnodal)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(6,6))
        tcf = ax.tricontourf(triang, qnodal, levels=levels, cmap=cmap)
        fig.colorbar(tcf, ax=ax, label=field)
        if show_mesh:
            ax.triplot(triang, color="k", lw=0.3, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{field} contours with mesh")
        plt.show()

        # --- write Tecplot ASCII if requested ---
        if tecplot_file is not None:
            with open(tecplot_file, "w") as f:
                f.write(f"TITLE = \"{field} field\"\n")
                f.write(f"VARIABLES = \"X\", \"Y\", \"{field}\"\n")
                f.write(f"ZONE N={Nv}, E={len(tris)}, DATAPACKING=POINT, ZONETYPE=FETRIANGLE\n")
                # Write nodal coordinates and field
                for i in range(Nv):
                    f.write(f"{Xv[i,0]} {Xv[i,1]} {qnodal[i]}\n")
                # Write connectivity (1-based indexing for Tecplot)
                for tri in tris:
                    f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
            print(f"Tecplot ASCII file written to {tecplot_file}")
            
    def lsq_wts_batched(self,dx, mask, r=-1, eps=1e-12):
        """
        Batched LSQ weights for gradient reconstruction in 2D or 3D.

        Parameters
        ----------
        dx : (N, M, dim) array
            Relative coordinates (x_j - x0_i).
        mask : (N, M) bool array
            True where a neighbor is valid, False for padding.
        r : float
            Weight exponent (default 0 → uniform, -2 typical).
        eps : float
            Stabilization for singular matrices.

        Returns
        -------
        wgts : (N, M, dim) array
            LSQ weights for each neighbor and dimension.
        """
        xp = self.xp
        N, M, dim = dx.shape
        assert dim in (2, 3), "Only 2D or 3D supported"

        # --- distances and weights ---
        dist2 = xp.sum(dx**2, axis=2) + eps
        w = xp.where(mask, dist2**(r/2), 0.0)  # (N, M)

        dx0 = dx[:, :, 0]
        dx1 = dx[:, :, 1]
        if dim == 3:
            dx2 = dx[:, :, 2]

        # --- build moment matrix entries ---
        a00 = xp.sum(w, axis=1)
        a01 = xp.sum(w * dx0, axis=1)
        a02 = xp.sum(w * dx1, axis=1)
        if dim == 3:
            a03 = xp.sum(w * dx2, axis=1)

        a11 = xp.sum(w * dx0 * dx0, axis=1)
        a12 = xp.sum(w * dx0 * dx1, axis=1)
        if dim == 3:
            a13 = xp.sum(w * dx0 * dx2, axis=1)
        a22 = xp.sum(w * dx1 * dx1, axis=1)
        if dim == 3:
            a23 = xp.sum(w * dx1 * dx2, axis=1)
            a33 = xp.sum(w * dx2 * dx2, axis=1)

        # --- determinant and inverse ---
        if dim == 2:
            # symmetric 3x3
            det = (
                a00 * (a11 * a22 - a12 * a12)
              - a01 * (a01 * a22 - a12 * a02)
              + a02 * (a01 * a12 - a11 * a02)
            )
            det = xp.where(xp.abs(det) < eps, eps * (2*(det >= 0)-1), det)
            ia00 =  (a11 * a22 - a12 * a12) / det
            ia01 = -(a01 * a22 - a12 * a02) / det
            ia02 =  (a01 * a12 - a11 * a02) / det
            ia11 =  (a00 * a22 - a02 * a02) / det
            ia12 = -(a00 * a12 - a01 * a02) / det
            ia22 =  (a00 * a11 - a01 * a01) / det

            wx0 = w * dx0
            wx1 = w * dx1

            wgtx = wx0 * ia11[:, None] + wx1 * ia12[:, None] + w * ia01[:, None]
            wgty = wx0 * ia12[:, None] + wx1 * ia22[:, None] + w * ia02[:, None]

            wgts = xp.stack([wgtx, wgty], axis=-1)

        else:  # dim == 3
            # safer path: build full 4x4 matrices and invert batched
            mats = xp.zeros((N, 4, 4), dtype=dx.dtype)
            mats[:, 0, 0] = a00
            mats[:, 0, 1] = mats[:, 1, 0] = a01
            mats[:, 0, 2] = mats[:, 2, 0] = a02
            mats[:, 0, 3] = mats[:, 3, 0] = a03
            mats[:, 1, 1] = a11
            mats[:, 1, 2] = mats[:, 2, 1] = a12
            mats[:, 1, 3] = mats[:, 3, 1] = a13
            mats[:, 2, 2] = a22
            mats[:, 2, 3] = mats[:, 3, 2] = a23
            mats[:, 3, 3] = a33

            invmats = xp.linalg.inv(mats + eps * xp.eye(4)[None])

            wx0 = w * dx0
            wx1 = w * dx1
            wx2 = w * dx2

            rhs = xp.stack([w, wx0, wx1, wx2], axis=-1)  # (N, M, 4)
            wgts = rhs @ invmats[:, 1:, :].swapaxes(1, 2)  # (N, M, 3)

        # Zero padded neighbors
        wgts = xp.where(mask[..., None], wgts, 0.0)
        return wgts

    # --- helper: build freestream conservative state ---
    def _freestream_U(self):
        xp = self.xp
        gamma = self.gamma
        # Freestream primitives
        rho = self.rinf
        u   = getattr(self, "uinf", 0.0)
        v   = getattr(self, "vinf", 0.0)
        p   = self.pinf
        E   = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v)
        return xp.array([rho, rho*u, rho*v, E])

    # --- helper: conservative -> primitives ---
    def _cons_to_prims(self, U):
        xp = self.xp
        gamma = self.gamma
        rho = U[..., 0]
        u   = U[..., 1] / rho
        v   = U[..., 2] / rho
        E   = U[..., 3]
        p   = (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v))
        return rho, u, v, p

    # --- helper: F(U) · n (projected flux for 2D Euler) ---
    def _flux_dot_n(self, U, nx, ny):
        xp = self.xp
        gamma = self.gamma
        rho, u, v, p = self._cons_to_prims(U)
        vn = u*nx + v*ny
        F0 = rho * vn
        F1 = U[...,1]*vn + nx * p
        F2 = U[...,2]*vn + ny * p
        F3 = (U[...,3] + p) * vn
        return xp.stack((F0, F1, F2, F3), axis=-1)

    def safe_add_at(self, arr, indices, values):
        """
        Safe in-place accumulation:
        - NumPy backend: uses np.add.at (deterministic)
        - CuPy backend: uses cupyx.scatter_add (atomic)
        
        Args:
        xp      : numpy or cupy module
        arr     : target array
        indices : integer indices (broadcastable to values)
        values  : array of values to add
        """
        if self.xp.__name__ == "numpy":
            self.xp.add.at(arr, indices, values)
        elif self.xp.__name__ == "cupy":
            import cupyx
            cupyx.scatter_add(arr, indices, values)
        else:
            raise TypeError(f"Unsupported xp module: {xp}")

    def getLeftRightStates(self, Qc, Xc, Xv, edges, edge2elem, ncell, gradQ, bc_type):
        
        xp = self.xp
        Ne = Qc.shape[0]
        Nedge = edges.shape[0]
        gamma = self.gamma

        # --- geometry on edges ---
        i0 = edges[:, 0]
        i1 = edges[:, 1]
        x0 = Xv[i0]                   # (Nedge,2)
        x1 = Xv[i1]                   # (Nedge,2)
        evec = x1 - x0                # edge vector
        L = xp.sqrt(xp.sum(evec**2, axis=1))  # edge length (Nedge,)
        # base (left-handed) unit normal (rotated +90°): n_perp = [ey, -ex] / L
        n_perp = xp.stack((evec[:,1], -evec[:,0]), axis=1)
        n_perp = n_perp / L[:, None]
        xf = 0.5 * (x0 + x1)          # edge midpoint

        # --- left/right elements ---
        Lidx = edge2elem[:, 0]        # left element
        Ridx = edge2elem[:, 1]        # right element (-1 if boundary)

        internal_mask = Ridx >= 0
        bnd_mask = ~internal_mask

        # --- orient normals from left -> right (or outward for boundary) ---
        nx = xp.empty(Nedge)
        ny = xp.empty(Nedge)        

        # internal edges: orient with vector from L centroid to R centroid
        if xp.any(internal_mask):
            dc = Xc[Ridx[internal_mask]] - Xc[Lidx[internal_mask]]
            sgn = xp.sign(xp.sum(n_perp[internal_mask] * dc, axis=1))
            sgn = xp.where(sgn == 0, 1.0, sgn)
            n_int = n_perp[internal_mask] * sgn[:, None]
            nx[internal_mask] = n_int[:, 0]
            ny[internal_mask] = n_int[:, 1]

        # boundary edges: orient outward relative to left centroid
        if xp.any(bnd_mask):
            dc = xf[bnd_mask] - Xc[Lidx[bnd_mask]]
            sgn = xp.sign(xp.sum(n_perp[bnd_mask] * dc, axis=1))
            sgn = xp.where(sgn == 0, 1.0, sgn)
            n_bnd = n_perp[bnd_mask] * sgn[:, None]
            nx[bnd_mask] = n_bnd[:, 0]
            ny[bnd_mask] = n_bnd[:, 1]

        # --- reconstruction at edge midpoint ---
        # First-order (piecewise constant)
        UL = Qc[Lidx]                 # (Nedge,4)
        if gradQ is not None:
            # linear MUSCL-style reconstruction: Q_L(xf) = Qc_L + gradQ_L · (xf - xc_L)
            dL = xf - Xc[Lidx]        # (Nedge,2)
            UL = UL + xp.einsum('eak,ek->ea', gradQ[Lidx], dL)

        # Right state
        UR = xp.empty_like(UL)
        # internal edges: from neighbor
        UR[internal_mask] = Qc[Ridx[internal_mask]]
        if gradQ is not None and xp.any(internal_mask):
            dR = xf[internal_mask] - Xc[Ridx[internal_mask]]
            UR[internal_mask] = UR[internal_mask] + xp.einsum('eak,ek->ea',
                                                              gradQ[Ridx[internal_mask]], dR)

        # boundary edges: build by BC
        if bc_type is None:
            # default: far-field on boundary
            bc_type = xp.array(["farfield" if b else "internal" for b in bnd_mask.tolist()])
        else:
            # convert Python list/np array of strings to xp array of dtype object if needed
            bc_type = xp.array(bc_type)

        if xp.any(bnd_mask):
            # slice arrays for boundary set
            UL_b = UL[bnd_mask]
            nx_b = nx[bnd_mask]; ny_b = ny[bnd_mask]

            # primitives on left
            rhoL, uL, vL, pL = self._cons_to_prims(UL_b)

            # allocate UR on boundary subset
            UR_b = xp.empty_like(UL_b)

            # masks per BC
            is_wall = xp.array([bt == "wall" for bt in bc_type[bnd_mask].tolist()])
            is_far  = xp.array([bt == "farfield" for bt in bc_type[bnd_mask].tolist()])

            # --- WALL (slip): reflect normal velocity ---
            if xp.any(is_wall):
                vnL = uL[is_wall]*nx_b[is_wall] + vL[is_wall]*ny_b[is_wall]
                uR = uL[is_wall] - 2.0 * vnL * nx_b[is_wall]
                vR = vL[is_wall] - 2.0 * vnL * ny_b[is_wall]
                rhoR = rhoL[is_wall]
                pR   = pL[is_wall]
                ER   = pR/(gamma - 1.0) + 0.5 * rhoR * (uR*uR + vR*vR)
                UR_b[is_wall, 0] = rhoR
                UR_b[is_wall, 1] = rhoR * uR
                UR_b[is_wall, 2] = rhoR * vR
                UR_b[is_wall, 3] = ER

            # --- FARFIELD: impose freestream as exterior state ---
            if xp.any(is_far):
                Uinf = self._freestream_U()  # (4,)
                UR_b[is_far] = Uinf

            # copy back
            UR[bnd_mask] = UR_b
        return UL,UR,nx,ny, Lidx, Ridx, L
        
    # --- main: residual ---
    def residual(self,
                 Qc,          # (Ne,4) cell-centered conservative variables
                 Xc,          # (Ne,2) element centroids
                 Xv,          # (Nv,2) node coordinates
                 edges,       # (Nedge,2) node index pairs
                 edge2elem,   # (Nedge,2) left/right element ids; -1 on boundary
                 ncells,
                 gradQ=None,  # optional (Ne,4,2) gradients for linear recon; None => 1st order
                 bc_type=None, # optional (Nedge,) strings: "internal"|"farfield"|"wall"
                 fluxType="Roe"
                 ):
        """
        Vectorized finite-volume residual for 2D Euler using Rusanov flux.

        Returns:
            R : (Ne,4) residual (sum of fluxes over edges, outward positive)
        """
        xp=self.xp
        UL,UR,nx,ny, Lidx, Ridx, L = self.getLeftRightStates( Qc, Xc, Xv, edges,
                                                           edge2elem, ncells,
                                                           gradQ, bc_type)
        if fluxType=="Lax" or fluxType=="Rusanov":
            # --- Rusanov flux: F* = 0.5[(F_L + F_R)·n - smax (UR - UL)] ---
            FLn = self._flux_dot_n(UL, nx, ny)
            FRn = self._flux_dot_n(UR, nx, ny)
            
            # wavespeed
            rhoL, uL, vL, pL = self._cons_to_prims(UL)
            aL = xp.sqrt(gamma * pL / rhoL)
            rhoR, uR, vR, pR = self._cons_to_prims(UR)

            aR = xp.sqrt(gamma * pR / rhoR)
            vnL = uL*nx + vL*ny
            vnR = uR*nx + vR*ny
            smax = xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR)

            Fe = 0.5 * (FLn + FRn - smax[:, None] * (UR - UL))  # (Nedge,4)
        elif fluxType == "Roe":
            ds=xp.vstack((nx,ny)).T
            faceVel=xp.zeros((UL.shape[0],))
            Fe,_ = roeflux(UL,UR,ds,faceVel)
            
        # integrate over edge length (flux per edge)
        Fe *= L[:, None]

        # --- scatter to residuals (left +, right -) ---
        R = xp.zeros_like(Qc)
        mask = Lidx < ncells
        self.safe_add_at(R, Lidx[mask],  Fe[mask])
        # only subtract for valid right neighbors
        mask = (Ridx >= 0 ) & (Ridx < ncells)
        if xp.any(mask):
            self.safe_add_at(R, Ridx[mask], -Fe[mask])
        return R

    # --- main: residual ---
    def diagJacobians(self,
                      Qc,          # (Ne,4) cell-centered conservative variables
                      Xc,          # (Ne,2) element centroids
                      Xv,          # (Nv,2) node coordinates
                      edges,       # (Nedge,2) node index pairs
                      edge2elem,   # (Nedge,2) left/right element ids; -1 on boundary
                      ncells,
                      gradQ=None,  # optional (Ne,4,2) gradients for linear recon; None => 1st order
                      bc_type=None, # optional (Nedge,) strings: "internal"|"farfield"|"wall"
                      fluxType="Roe"
                      ):        
        """
        Vectorized finite-volume Jacobians 

        Returns:
            D : (Ne, 4, 4)
            O : (Ne, max_neigh, 4, 4)
        """
        xp=self.xp
        UL,UR,nx,ny, Lidx, Ridx, L = self.getLeftRightStates( Qc, Xc, Xv, edges,
                                                           edge2elem, ncells,
                                                           gradQ, bc_type)
        ds=xp.vstack((nx,ny)).T
        faceVel=xp.zeros((UL.shape[0],))
        jacL,jacR = flux_jacobians_fd(UL,UR,ds,faceVel, xp=self.xp)
            
        # integrate over edge length (flux per edge)
        jacL *= L[:,None,None]
        jacR *= L[:,None,None]

        # --- scatter to residuals (left +, right -) ---
        D = xp.zeros((Qc.shape[0],4,4),'d')
        mask = Lidx < ncells
        self.safe_add_at(D, Lidx[mask],  jacL[mask])
        # only subtract for valid right neighbors
        mask = (Ridx >= 0 ) & (Ridx < ncells)
        if xp.any(mask):
            self.safe_add_at(D, Ridx[mask], -jacR[mask])
        return D

    # ---------------------------------------------------------
    # Full Jacobian–vector product (diag + offdiag together)
    # ---------------------------------------------------------
    def fullJacobianProduct(self, dq, Qc, Xc, Xv, edges, edge2elem, ncells,
                            gradQ=None, bc_type=None, fluxType="Roe"):
        """
        Full Jacobian–vector product (diag + offdiag together).
        """
        xp=self.xp
        UL, UR, nx, ny, Lidx, Ridx, L = self.getLeftRightStates(
            Qc, Xc, Xv, edges, edge2elem, ncells, gradQ, bc_type )
        ds = xp.vstack((nx, ny)).T

        faceVel = xp.zeros((UL.shape[0],))
        jacL, jacR = flux_jacobians_fd(UL, UR, ds, faceVel, xp=self.xp)

        jacL *= L[:,None,None]
        jacR *= L[:,None,None]

        Odq = xp.zeros_like(Qc)

        # Fe0,_ = roeflux(UL,UR,ds,faceVel)
        # Fe0 *= L[:,None]
        # UL, UR, nx, ny, Lidx, Ridx, L = self.getLeftRightStates(
        #     Qc+dq, Xc, Xv, edges, edge2elem, ncells, gradQ, bc_type )
        # Fe1,_ = roeflux(UL,UR,ds,faceVel)
        # Fe1 *= L[:,None]
        # print("Fe1-Fe0=",Fe1-Fe0)

        # dF=Fe1-Fe0
        # mask = Lidx < ncells
        # dJF = xp.zeros_like(UL)
        # dJF[mask]=np.einsum('nij,nj->ni',jacL[mask],dq[Lidx[mask]])
        # mask = (Ridx >=0) & (Ridx < ncells)
        # dJF[mask]+=np.einsum('nij,nj->ni',jacR[mask],dq[Ridx[mask]])
        # print("norm(dF-dJF)=",xp.linalg.norm(dF-dJF))
        
        # Left residual: + (J_L dq_L + J_R dq_R)
        maskL = Lidx < ncells

        dFl = xp.einsum('nij,nj->ni',jacL,dq[Lidx])
        dFr = xp.einsum('nij,nj->ni',jacR,dq[Ridx])
        
        if xp.any(maskL):
            self.safe_add_at(Odq, Lidx[maskL], (dFl+dFr)[maskL])
            #self.safe_add_at(Odq, Lidx[maskL],
            #                 xp.einsum('nij,nj->ni', jacL[maskL], dq[Lidx[maskL]])
            #               + xp.einsum('nij,nj->ni', jacR[maskR], dq[Ridx[maskR]]))

        # Right residual: - (J_L dq_L + J_R dq_R)
        maskR = (Ridx >= 0) & (Ridx < ncells)

        if xp.any(maskR):
            self.safe_add_at(Odq, Ridx[maskR], -(dFl+dFr)[maskR])
            #self.safe_add_at(Odq, Ridx[maskR],
            #                -(xp.einsum('nij,nj->ni', jacL[maskL], dq[Lidx[maskL]])
            #                + xp.einsum('nij,nj->ni', jacR[maskR], dq[Ridx[maskR]])))
        # print("Odq=",Odq)
        # print("sum(Odq)=",np.sum(Odq))
        return Odq

    def diagProduct(self, dq, Qc, Xc, Xv, edges, edge2elem, ncells,
                    gradQ=None, bc_type=None, fluxType="Roe"):
        xp=self.xp
        UL, UR, nx, ny, Lidx, Ridx, L = self.getLeftRightStates(
            Qc, Xc, Xv, edges, edge2elem, ncells, gradQ, bc_type)
        ds = xp.vstack((nx, ny)).T

        faceVel = xp.zeros((UL.shape[0],))
        jacL, jacR = flux_jacobians_fd(UL, UR, ds, faceVel, xp=self.xp)

        jacL *= L[:,None,None]
        jacR *= L[:,None,None]

        Odq = xp.zeros_like(Qc)

        dFl = xp.einsum('nij,nj->ni',jacL,dq[Lidx])
        dFr = xp.einsum('nij,nj->ni',jacR,dq[Ridx])
        
        # Left diagonal: + J_L dq_L
        maskL = Lidx < ncells
        self.safe_add_at(Odq, Lidx[maskL], dFl[maskL])

        # Right diagonal: - J_R dq_R
        maskR = (Ridx >= 0) & (Ridx < ncells)
        if xp.any(maskR):
            self.safe_add_at(Odq, Ridx[maskR], -dFr[maskR])

        return Odq


    def offDiagProduct(self, dq, Qc, Xc, Xv, edges, edge2elem, ncells,
                       gradQ=None, bc_type=None, fluxType="Roe"):
        xp=self.xp
        UL, UR, nx, ny, Lidx, Ridx, L = self.getLeftRightStates(
            Qc, Xc, Xv, edges, edge2elem, ncells, gradQ, bc_type )
        ds = xp.vstack((nx, ny)).T
        
        faceVel = xp.zeros((UL.shape[0],))
        jacL, jacR = flux_jacobians_fd(UL, UR, ds, faceVel, xp=self.xp)
        
        jacL *= L[:,None,None]
        jacR *= L[:,None,None]
        
        Odq = xp.zeros_like(Qc)

        dFl = xp.einsum('nij,nj->ni',jacL,dq[Lidx])
        dFr = xp.einsum('nij,nj->ni',jacR,dq[Ridx])
        
        # Left off-diagonal: + J_R dq_R
        maskL = Lidx < ncells
        self.safe_add_at(Odq, Lidx[maskL], dFr[maskL])
        # Right off-diagonal: - J_L dq_L
        maskR = (Ridx >= 0) & (Ridx < ncells)
        if xp.any(maskR):
            self.safe_add_at(Odq, Ridx[maskR],-dFl[maskR])
        return Odq    
    
