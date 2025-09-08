import numpy as np
# import jax
# import jax.numpy as jnp

# def roeflux_jax(left, right, ds, faceVel, gamma=1.4, entropy_fix=True):
#     # identical to vectorized routine above, but using jnp not np/cp
#     # return flux (N,4)
#     flux, _ = roeflux(left, right, ds, faceVel, gamma, entropy_fix, xp=jnp)
#     return flux

# def flux_jacobians(left, right, ds, faceVel, gamma=1.4):
#     """
#     Compute Jacobians dF/dUL and dF/dUR.
#     Returns (N,4,4), (N,4,4)
#     """
#     # single-sample flux function for jacrev
#     def flux_LR(UL, UR, ds_i, v_i):
#         return roeflux_jax(UL[None,:], UR[None,:], ds_i[None,:], v_i[None])[0]

#     # vectorize across N faces
#     jacL = jax.vmap(jax.jacrev(flux_LR, argnums=0))(left, right, ds, faceVel)
#     jacR = jax.vmap(jax.jacrev(flux_LR, argnums=1))(left, right, ds, faceVel)
#     return jacL, jacR

def flux_jacobians_fd(left, right, ds, faceVel, gamma=1.4, eps=1e-8, xp=None):
    """
    Finite-difference Jacobians of Roe flux.
    Returns (N,4,4), (N,4,4).
    """
    flux0, _ = roeflux(left, right, ds, faceVel, gamma, xp=xp)
    N = left.shape[0]
    jacL = xp.zeros((N,4,4))
    jacR = xp.zeros((N,4,4))

    # perturb UL
    for j in range(4):
        dU = xp.zeros_like(left)
        dU[:,j] = eps
        f_plus, _ = roeflux(left+dU, right, ds, faceVel, gamma, xp=xp)
        f_minus, _ = roeflux(left-dU, right, ds, faceVel, gamma, xp=xp)
        jacL[:,:,j] = (f_plus - f_minus)/(2*eps)

    # perturb UR
    for j in range(4):
        dU = xp.zeros_like(right)
        dU[:,j] = eps
        f_plus, _ = roeflux(left, right+dU, ds, faceVel, gamma, xp=xp)
        f_minus, _ = roeflux(left, right-dU, ds, faceVel, gamma, xp=xp)
        jacR[:,:,j] = (f_plus - f_minus)/(2*eps)

    return jacL, jacR


def roeflux(left, right, ds, faceVel, gamma=1.4, entropy_fix=True, xp=None):
    """
    Vectorized Roe flux for 2D Euler with moving grids.

    Parameters
    ----------
    left : (N,4) xp.ndarray
        Left conservative states [rho, rho*u, rho*v, rho*E].
    right : (N,4) xp.ndarray
        Right conservative states.
    ds : (N,2) xp.ndarray
        Face area vectors (nx*|S|, ny*|S|).
    faceVel : (N,) xp.ndarray
        Normal grid velocities.
    gamma : float
        Ratio of specific heats.
    entropy_fix : bool
        If True, apply Harten–Hyman entropy fix.
    xp : module
        Array module, e.g. numpy or cupy. If None, inferred from `left`.

    Returns
    -------
    flux : (N,4) xp.ndarray
        Roe flux across each face.
    spec_radius : (N,) xp.ndarray
        Spectral radius (for CFL calculation).
    """
    if xp is None:
        # autodetect numpy vs cupy from input
        import numpy as np
        xp = np if type(left).__module__.startswith("numpy") else __import__("cupy")

    gm1 = gamma - 1.0
    eps = 1e-14
    
    # --- Left state ---
    rl = left[:, 0]
    ul = left[:, 1] / rl
    vl = left[:, 2] / rl
    uv_l = 0.5*(ul**2 + vl**2)
    pl = gm1*(left[:, 3] - rl*uv_l)
    pl = xp.maximum(pl, 1e-16)
    el = pl/gm1 + rl*uv_l
    hl = (el + pl)/rl
    cl2 = gm1*(hl - uv_l)
    cl2 = xp.maximum(cl2, 0.0)
    cl = xp.sqrt(cl2)

    # --- Right state ---
    rr = right[:, 0]
    ur = right[:, 1] / rr
    vr = right[:, 2] / rr
    uv_r = 0.5*(ur**2 + vr**2)
    pr = gm1*(right[:, 3] - rr*uv_r)
    pr = xp.maximum(pr, 1e-16)
    er = pr/gm1 + rr*uv_r
    hr = (er + pr)/rr
    cr2 = gm1*(hr - uv_r)
    cr2 = xp.maximum(cr2, 0.0)
    cr = xp.sqrt(cr2)

    # --- Roe averages ---
    rat = xp.sqrt(rr/rl)
    rati = 1.0/(rat+1.0)
    rav = rat*rl
    uav = (rat*ur + ul)*rati
    vav = (rat*vr + vl)*rati
    hav = (rat*hr + hl)*rati
    uv = 0.5*(uav**2 + vav**2)
    cav2 = gm1*(hav - uv)
    cav2 = xp.maximum(cav2, 0.0)
    cav = xp.sqrt(cav2)

    # --- Deltas ---
    dq1 = rr - rl
    dq2 = ur - ul
    dq3 = vr - vl
    dq4 = pr - pl

    # --- Geometry ---
    ri1, ri2 = ds[:, 0], ds[:, 1]
    rr2 = ri1**2 + ri2**2
    rrn = xp.sqrt(rr2)
    rrn = xp.maximum(rrn, 1e-300)
    r0 = 1.0/rrn
    r1 = ri1*r0
    r2 = ri2*r0
    r3 = faceVel*r0

    # --- Normal velocities ---
    uul = r1*ul + r2*vl + r3
    uur = r1*ur + r2*vr + r3
    uu = r1*uav + r2*vav + r3

    # --- Eigenvalues ---
    auu = xp.abs(uu)
    aupc = xp.abs(uu+cav)
    aumc = xp.abs(uu-cav)

    if entropy_fix:
        upcl = uul + cl
        upcr = uur + cr
        umcl = uul - cl
        umcr = uur - cr

        dauu = xp.maximum(4*(uur-uul)+eps, 0.0)
        auu = xp.where(auu <= 0.5*dauu,
                       auu*auu/(dauu+eps)+0.25*dauu, auu)

        daupc = xp.maximum(4*(upcr-upcl)+eps, 0.0)
        aupc = xp.where(aupc <= 0.5*daupc,
                        aupc*aupc/(daupc+eps)+0.25*daupc, aupc)

        daumc = xp.maximum(4*(umcr-umcl)+eps, 0.0)
        aumc = xp.where(aumc <= 0.5*daumc,
                        aumc*aumc/(daumc+eps)+0.25*daumc, aumc)

    # --- Spectral radius ---
    spec_radius = (xp.abs(uu)+cav)*rrn

    # --- |A|*dq ---
    rcav = rav*cav
    aquu = uur - uul
    c2i = xp.where(cav2 > 0.0, 1.0/cav2, 0.0)
    c2ih = 0.5*c2i
    ruuav = auu*rav

    b1 = auu*(dq1 - c2i*dq4)
    b2 = c2ih*aupc*(dq4 + rcav*aquu)
    b3 = c2ih*aumc*(dq4 - rcav*aquu)
    b4 = b1 + b2 + b3
    b5 = cav*(b2 - b3)
    b6 = ruuav*(dq2 - r1*aquu)
    b7 = ruuav*(dq3 - r2*aquu)

    dq1 = b4
    dq2 = uav*b4 + r1*b5 + b6
    dq3 = vav*b4 + r2*b5 + b7
    dq4 = hav*b4 + (uu-r3)*b5 + uav*b6 + vav*b7 - cav2*b1/gm1

    # --- Flux assembly ---
    aj = 0.5*rrn
    plar = pl + pr
    epl = el + pl
    epr = er + pr

    flux = xp.empty_like(left)
    flux[:, 0] = aj*(rl*uul + rr*uur - dq1)
    flux[:, 1] = aj*(rl*ul*uul + rr*ur*uur + r1*plar - dq2)
    flux[:, 2] = aj*(rl*vl*uul + rr*vr*uur + r2*plar - dq3)
    flux[:, 3] = aj*(epl*uul + epr*uur - r3*plar - dq4)

    return flux, spec_radius

if __name__ == "__main__":
    tol = 1e-6

    # Define two test left/right states (N=2 interfaces, 2D Euler, 4 conservative vars)
    left  = np.array([[1,   0.0, 0.0, 1.8 ],
                      [1,   0.1, 0.0, 1.8 ]])
    right = np.array([[1,   0.1, 0.1, 1.9 ],
                      [1,   0.1, 0.1, 1.91]])

    # Face velocity (grid speed) = 0 for both faces
    faceVel = np.array([0.0, 0.0], dtype='d')

    # Face normals (two different directions)
    ds = np.array([[1, 1],
                   [2, 2]])

    # --- Step 1: Evaluate baseline flux ---
    # roeflux returns (flux, specRadius), so take [0] = flux
    flux0 = roeflux(left, right, ds, faceVel)[0]

    # --- Step 2: Add small random perturbations to left/right states ---
    # Random perturbations ~ tol = 1e-6
    eps1 = np.random.rand(2,4) * tol   # perturbations for left states
    eps2 = np.random.rand(2,4) * tol   # perturbations for right states

    # Perturbed flux
    flux1 = roeflux(left + eps1, right + eps2, ds, faceVel)[0]

    # Difference in fluxes due to perturbations
    deltaflux = flux1 - flux0

    # --- Step 3: Compute Jacobians via finite differences ---
    jacL, jacR = flux_jacobians_fd(left, right, ds, faceVel, xp=np)

    # --- Step 4: Linearized flux difference using Jacobians ---
    # deltaF ≈ J_L * eps1 + J_R * eps2
    # einsum('nij,nj->ni', jac, eps) does batch-matrix-vector multiplication
    deltafluxjac = (np.einsum('nij,nj->ni', jacL, eps1) +
                    np.einsum('nij,nj->ni', jacR, eps2))

    # --- Step 5: Compare the true flux difference vs. Jacobian-predicted one ---
    # This should be small (~O(tol^2)) if Jacobians are consistent
    error_norm = np.linalg.norm(deltafluxjac - deltaflux)
    print(f"Norm of difference between linearized and actual flux change: {error_norm:.3e}")
#tol=1e-6
#left=np.array([[1,0,0,1.8],[1,0.1,0,1.8]])
#right=np.array([[1,0.1,0.1,1.9],[1,0.1,0.1,1.91]])
#faceVel=np.array([0,0],'d')
#ds=np.array([[1,1],[2,2]])
#flux0=roeflux(left,right,ds,faceVel)[0]
#eps1=np.random.rand(2,4)*tol
#eps2=np.random.rand(2,4)*tol
#flux1=roeflux(left+eps1,right+eps2,ds,faceVel)[0]
#deltaflux = flux1-flux0
#jacL,jacR = flux_jacobians_fd(left,right,ds,faceVel,xp=np)
#deltafluxjac = (np.einsum('nij,nj->ni',jacL,eps1)+np.einsum('nij,nj->ni',jacR,eps2))
#print(np.linalg.norm(deltafluxjac-deltaflux))
#
