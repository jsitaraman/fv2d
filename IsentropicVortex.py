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

