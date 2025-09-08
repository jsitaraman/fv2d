import numpy as np

class IsentropicVortex:
    def __init__(self, use_cupy=False):
        self.xp = __import__('cupy' if use_cupy else 'numpy')

        # --- Hardcoded parameters (from your C++ snippet) ---
        self.gamma = 1.4
        self.pinf  = 1.0
        self.rinf  = 1.0

        pi    = np.pi
        self.cinf  = self.xp.sqrt(self.gamma * self.pinf / self.rinf)
        self.sinf  = self.pinf / (self.rinf**self.gamma)
        self.tinf  = self.pinf / self.rinf
        self.strnth = 1.0
        self.sigma  = 1.0
        self.scale  = 1.0
        self.gm1    = self.gamma - 1.0
        self.afac   = self.strnth / (2.0 * pi)
        self.bfac   = -0.5 * self.gm1 / self.gamma * self.afac * self.afac

    def init_Q(self, X, x0, y0, uinf, vinf):
        """
        Initialize conservative variables Q for 2D isentropic vortex.

        Args:
            X : array of shape (N, 2) with coordinates
            x0, y0 : vortex center
            uinf, vinf : free-stream velocities

        Returns:
            Q : array of shape (N, 4) with conservative variables
                Q[:,0] = rho
                Q[:,1] = rho * u
                Q[:,2] = rho * v
                Q[:,3] = total energy
        """
        xp = self.xp

        # --- Coordinates ---
        xx = X[:, 0] - x0
        yy = X[:, 1] - y0
        rsq = xx**2 + yy**2

        # --- Exponential envelope ---
        ee = xp.exp(0.5 * (1 - self.sigma * rsq)) * self.scale

        # --- Velocity field ---
        u = uinf - (self.afac * ee) * yy
        v = vinf + (self.afac * ee) * xx

        # --- Temperature and pressure ---
        t = self.tinf + self.bfac * ee * ee
        p = (t**self.gamma / self.sinf) ** (1.0 / self.gm1)

        # --- Density (EOS with R=1) ---
        rho = p / t

        # --- Conservative variables (N,4) ---
        N = X.shape[0]
        Q = xp.zeros((N, 4))
        Q[:, 0] = rho
        Q[:, 1] = rho * u
        Q[:, 2] = rho * v
        Q[:, 3] = p / self.gm1 + 0.5 * rho * (u**2 + v**2)

        return Q


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
    
    # --- main: residual ---
    def residual(self,
                 Qc,          # (Ne,4) cell-centered conservative variables
                 Xc,          # (Ne,2) element centroids
                 Xv,          # (Nv,2) node coordinates
                 edges,       # (Nedge,2) node index pairs
                 edge2elem,   # (Nedge,2) left/right element ids; -1 on boundary
                 ncells,
                 gradQ=None,  # optional (Ne,4,2) gradients for linear recon; None => 1st order
                 bc_type=None # optional (Nedge,) strings: "internal"|"farfield"|"wall"
                 ):
        """
        Vectorized finite-volume residual for 2D Euler using Rusanov flux.

        Returns:
            R : (Ne,4) residual (sum of fluxes over edges, outward positive)
        """
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
        # integrate over edge length (flux per edge)
        Fe *= L[:, None]

        # --- scatter to residuals (left +, right -) ---
        R = xp.zeros_like(Qc)
        mask = Lidx < ncells
        xp.add.at(R, Lidx[mask],  Fe[mask])
        # only subtract for valid right neighbors
        mask = (Ridx >= 0 ) & (Ridx < ncells)
        if xp.any(mask):
            xp.add.at(R, Ridx[mask], -Fe[mask])
        return R
