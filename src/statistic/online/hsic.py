import jax
import jax.numpy as jnp
from utils.type import KernelFn
__all__ = [# online hsic
           "loo_online_HSIC",
           "loo_online_HSIC_from_matrices",
           # martingale hsic
           "mHSIC",
           "mHSIC_from_matrices",
           "loo_mHSIC",
           "loo_mHSIC_from_matrices",]


# ===============================
#          Online MMD
# ===============================

def loo_online_HSIC(X: jax.Array,
                    Y: jax.Array,
                    kernel_x: KernelFn,
                    kernel_y: KernelFn,) -> jax.Array:
    assert X.shape == Y.shape and X.ndim == 2
    X0, X1 = jnp.split(X, 2)
    Y0, Y1 = jnp.split(Y, 2)
    K00 = kernel_x(X0, X0)
    K01 = kernel_x(X0, X1)
    K11 = kernel_x(X1, X1)
    L00 = kernel_y(Y0, Y0)
    L01 = kernel_y(Y0, Y1)
    L11 = kernel_y(Y1, Y1)
    return loo_online_HSIC_from_matrices(K00, K01, K11, L00, L01, L11)

def loo_online_HSIC_from_matrices(K00: jax.Array,
                                  K01: jax.Array,
                                  K11: jax.Array,
                                  L00: jax.Array,
                                  L01: jax.Array,
                                  L11: jax.Array,) -> jax.Array:
    r"""Computes the HSIC (v-statistic) sequentially from time t=1 to t=n, while also excluding a leave-out sample l.
    Args:
        (K00, K01, K11): (n,n) kernel matrices for the X0, X1 samples
        (L00, L01, L11): (n,n) kernel matrices for the Y0, Y1 samples
    Returns:
        res: (n,n) matrix of HSIC values; the rows index time from t=1 to t=n, and columns index the leave-out sample l.
    """
    assert K00.shape == K01.shape == K11.shape and K00.ndim == 2
    assert L00.shape == L01.shape == L11.shape and L00.ndim == 2
    K10 = K01.mT
    L10 = L01.mT
    n = K00.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)

    # compute |μ_yx|^2 term
    A = K00*L00 + K01*L01 + K10*L10 + K11*L11   # indices: i x j
    sA_tj = jnp.cumsum(A, axis=0)
    sA_t = jnp.sum(uptot * sA_tj, axis=1)
    sA_atleast1l_tl = 2*sA_tj - jnp.diagonal(A)
    S1 = sA_t[:,jnp.newaxis] - uptot * sA_atleast1l_tl

    # compute <μ_yx, μ_y⊗μ_x> terms
    # part 1
    Bx = K00 + K10  # indices: i x k
    By = L00 + L10  # indices: j x k
    Bxii = jnp.diagonal(Bx)
    Byjj = jnp.diagonal(By)
    sBx_tk = jnp.cumsum(Bx, axis=0)
    sBy_tk = jnp.cumsum(By, axis=0)
    sBxBy_t = jnp.sum(uptot * sBx_tk * sBy_tk, axis=1)
    sBxBy_ieql_tl = (uptot * sBy_tk) @ Bx.mT    # i=l
    sBxBy_jeql_tl = (uptot * sBx_tk) @ By.mT    # j=l
    sBxBy_keql_tl = sBx_tk * sBy_tk             # k=l
    sBxBy_ijeql_tl = jnp.cumsum(Bx * By, axis=1).mT  # i=j=l
    sBxBy_ikeql_tl = Bxii * sBy_tk  # i=k=l
    sBxBy_jkeql_tl = sBx_tk * Byjj  # j=k=l
    sBxBy_ijkeql = Bxii * Byjj      # i=j=k=l
    sBxBy_atleast1l_tl = (
        (sBxBy_ieql_tl + sBxBy_jeql_tl + sBxBy_keql_tl)
        - (sBxBy_ijeql_tl + sBxBy_ikeql_tl + sBxBy_jkeql_tl)
        + sBxBy_ijkeql
    )
    p1 = sBxBy_t[:,jnp.newaxis] - uptot * sBxBy_atleast1l_tl
    # part 2
    Cx = K01 + K11  # indices: i x k
    Cy = L01 + L11  # indices: j x k
    Cxii = jnp.diagonal(Cx)
    Cyjj = jnp.diagonal(Cy)
    sCx_tk = jnp.cumsum(Cx, axis=0)
    sCy_tk = jnp.cumsum(Cy, axis=0)
    sCxCy_t = jnp.sum(uptot * sCx_tk * sCy_tk, axis=1)
    sCxCy_ieql_tl = (uptot * sCy_tk) @ Cx.mT    # i=l
    sCxCy_jeql_tl = (uptot * sCx_tk) @ Cy.mT    # i=l
    sCxCy_keql_tl = sCx_tk * sCy_tk             # k=l
    sCxCy_ijeql_tl = jnp.cumsum(Cx * Cy, axis=1).mT  # i=j=l
    sCxCy_ikeql_tl = Cxii * sCy_tk  # i=k=l
    sCxCy_jkeql_tl = sCx_tk * Cyjj  # j=k=l
    sCxCy_ijkeql = Cxii * Cyjj      # i=j=k=l
    sCxCy_atleast1l_tl = (
        (sCxCy_ieql_tl + sCxCy_jeql_tl + sCxCy_keql_tl)
        - (sCxCy_ijeql_tl + sCxCy_ikeql_tl + sCxCy_jkeql_tl)
        + sCxCy_ijkeql
    )
    p2 = sCxCy_t[:,jnp.newaxis] - uptot * sCxCy_atleast1l_tl
    S2 = p1 + p2

    # compute |μ_y⊗μ_x|^2 term
    D = K00 + K01 + K10 + K11   # indices: i x j
    E = L00 + L01 + L10 + L11   # indices: u x v
    sD_tj = jnp.cumsum(D, axis=0)
    sE_tv = jnp.cumsum(E, axis=0)
    sD_t = jnp.sum(uptot * sD_tj, axis=1)
    sE_t = jnp.sum(uptot * sE_tv, axis=1)
    sD_atleast1l_tl = 2*sD_tj - jnp.diagonal(D)
    sE_atleast1l_tl = 2*sE_tv - jnp.diagonal(E)
    p1 = sD_t[:,jnp.newaxis] - uptot * sD_atleast1l_tl
    p2 = sE_t[:,jnp.newaxis] - uptot * sE_atleast1l_tl
    S3 = p1 * p2

    # normalize sums
    t = jnp.arange(n)[:,jnp.newaxis] + 1
    Nt = uptot*(2*jnp.clip(t-1, min=1)) + ~uptot*(2*t)
    M1 = jnp.sign(S1) * jnp.exp(jnp.log(jnp.abs(S1)) - 2*jnp.log(Nt))
    M2 = jnp.sign(S2) * jnp.exp(jnp.log(jnp.abs(S2)) - 3*jnp.log(Nt))
    M3 = jnp.sign(S3) * jnp.exp(jnp.log(jnp.abs(S3)) - 4*jnp.log(Nt))
    return jnp.clip(M1 - 2*M2 + M3, min=0)


# ===============================
#        Martingale HSIC
# ===============================

def loo_mHSIC(X: jax.Array,
              Y: jax.Array,
              kernel_x: KernelFn,
              kernel_y: KernelFn,
              radius: float = 0.5,
              eps: float = 1e-8,) -> jax.Array:
    """Computes the HSIC by learning the witness Hilbert-Schmidt critic on up-to-current samples (<=t), excluding the sample (=l) used for evaluation.
    Args:
        X: (n,d) samples
        Y: (n,d) samples
        kernel_x: Callable function on X
        kernel_y: Callable function on X
        radius: Norm of HS critic (default=0.5)
        eps: constant used for numerical stabilization (default=1e-8). 
    Returns:
        (n,n) array of leave-one-out mHSIC values; the rows correspond to time t and the columns correspond t o the leave-out sample l.
        The diagonal is exactly mHSIC.
    """
    assert X.shape == Y.shape and X.ndim == 2
    X0, X1 = jnp.split(X, 2)
    Y0, Y1 = jnp.split(Y, 2)
    K00 = kernel_x(X0, X0)
    K01 = kernel_x(X0, X1)
    K11 = kernel_x(X1, X1)
    L00 = kernel_y(Y0, Y0)
    L01 = kernel_y(Y0, Y1)
    L11 = kernel_y(Y1, Y1)
    return loo_mHSIC_from_matrices(K00, K01, K11, L00, L01, L11, radius, eps)

def loo_mHSIC_from_matrices(K00: jax.Array,
                            K01: jax.Array,
                            K11: jax.Array,
                            L00: jax.Array,
                            L01: jax.Array,
                            L11: jax.Array,
                            radius: float = 0.5,
                            eps: float = 1e-8,) -> jax.Array:
    """Computes the HSIC by learning the witness Hilbert-Schmidt critic on up-to-current samples (<=t), excluding the sample (=l) used for evaluation.
    Args:
        (K00, K01, K11): (n,n) kernel matrices for the split X = [X0, X1] samples
        (L00, L01, L11): (n,n) kernel matrices for the split Y = [Y0, Y1] samples
        radius: norm of HS critic (default=0.5)
        eps: constant used for numerical stabilization (default=1e-8). 
    Returns:
        (n,n) array of leave-one-out mHSIC values; the rows correspond to time t and the columns correspond t o the leave-out sample l.
        The diagonal is exactly mHSIC.
    """
    assert K00.shape == K01.shape == K11.shape and K00.ndim == 2
    assert L00.shape == L01.shape == L11.shape and L00.ndim == 2
    assert K00.shape == L00.shape
    K10 = K01.mT
    L10 = L01.mT
    n = K00.shape[0]
    uptot = jnp.tri(n, dtype=jnp.bool)

    A = (K00 - K01) * (L00 - L01) + (K10 - K11) * (L10 - L11)
    Bx = K00 - K01 + K10 - K11
    By = L00 - L01 + L10 - L11
    S1 = jnp.cumsum(A, axis=0) - uptot * jnp.diagonal(A)
    S2 = jnp.cumsum(Bx, axis=0) - uptot * jnp.diagonal(Bx)
    S3 = jnp.cumsum(By, axis=0) - uptot * jnp.diagonal(By)

    norm2 = loo_online_HSIC_from_matrices(K00, K01, K11, L00, L01, L11)
    norm = jnp.sqrt(jnp.clip(norm2, min=eps))
    t = jnp.arange(n)[:,jnp.newaxis] + 1
    Nt = uptot*(2*jnp.clip(t-1, min=1)) + ~uptot*(2*t)
    M1 = jnp.sign(S1) * jnp.exp(jnp.log(jnp.abs(S1)) - jnp.log(Nt))
    M2 = jnp.sign(S2*S3) * jnp.exp(jnp.log(jnp.abs(S2*S3)) - 2*jnp.log(Nt))
    return radius * (M1 - M2) / norm

