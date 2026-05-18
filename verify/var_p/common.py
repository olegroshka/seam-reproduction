"""Shared utilities for VAR(p) sigma^2_C exploration.

Provides:
- build_VAR_p_DGP : random stable VAR(p) generator with LOC/REM block structure
- companion_matrix, companion_Sigma_F : companion-form embedding helpers
- yule_walker_R : autocovariances R(0), R(1), ..., R(p) from (A_h, Sigma_eps)
- block_toeplitz_C_Gamma : C and Gamma block-Toeplitz construction at REM
- closed_form_sigma2_C : the lifted Theorem 1^(p) formula
- companion_form_sigma2_C : paper's Theorem 1 applied at lifted REM_comp partition
- companion_cluster_decomposition : Theorem 2^(p) per-LOC diagonal + cross
- simulate_VAR_p : simulate a stationary trajectory
- ols_var_p : OLS-VAR(p) estimator
"""
from __future__ import annotations

import numpy as np


def build_companion_matrix(A_list, K, p):
    """Companion-form transition matrix Atilde of size (Kp x Kp).

    Atilde acts on F_t = (f_t, f_{t-1}, ..., f_{t-p+1})^T as F_t = Atilde F_{t-1} + B eps_t,
    where B = (I_K, 0, ..., 0)^T selects the innovation into the first block.
    """
    Kp = K * p
    Atilde = np.zeros((Kp, Kp))
    for h in range(p):
        Atilde[0:K, h*K:(h+1)*K] = A_list[h]
    for h in range(1, p):
        Atilde[h*K:(h+1)*K, (h-1)*K:h*K] = np.eye(K)
    return Atilde


def rescale_to_target_rho(A_list, K, p, target_rho=0.85, max_iter=200, tol=1e-10):
    """Iteratively scale all A_h by a common factor so the companion-form
    spectral radius equals target_rho. Returns scaled A_list and final Atilde.
    """
    s = 1.0
    last_test = None
    last_At = None
    for _ in range(max_iter):
        test = [s*A for A in A_list]
        At = build_companion_matrix(test, K, p)
        rho = float(np.max(np.abs(np.linalg.eigvals(At))))
        last_test, last_At = test, At
        if abs(rho - target_rho) < tol:
            break
        s = s * target_rho / max(rho, 1e-12)
    return last_test, last_At


def companion_Sigma_F(Atilde, Sigma_eps, K, p):
    """Solve companion-form Lyapunov: Sigma_F = Atilde Sigma_F Atilde^T + B Sigma_eps B^T.

    Returns Sigma_F (Kp x Kp), symmetrized for numerical hygiene.
    """
    Kp = K * p
    Sigma_eps_comp = np.zeros((Kp, Kp))
    Sigma_eps_comp[0:K, 0:K] = Sigma_eps
    AkronA = np.kron(Atilde, Atilde)
    vec_S_eps_comp = Sigma_eps_comp.flatten('F')
    vec_Sigma_F = np.linalg.solve(np.eye(Kp**2) - AkronA, vec_S_eps_comp)
    Sigma_F = vec_Sigma_F.reshape((Kp, Kp), order='F')
    return (Sigma_F + Sigma_F.T) / 2


def extract_R_list(Sigma_F, A_list, K, p):
    """Extract autocovariances R(0), R(1), ..., R(p) from companion Sigma_F.

    R(0) = Sigma_f = top-left K x K of Sigma_F.
    R(h) for h = 1..p-1: top-right K x K of Sigma_F at column-offset h*K.
    R(p) via Yule-Walker recursion (one step beyond the companion extraction).
    """
    R_list = [Sigma_F[0:K, 0:K]]
    for h in range(1, p):
        R_list.append(Sigma_F[0:K, h*K:(h+1)*K])
    R_p = sum(A_list[q-1] @ R_list[p-q] for q in range(1, p+1))
    R_list.append(R_p)
    return R_list


def build_C_Gamma_REM(R_list, K_LOC, K_REM, p):
    """Build C (K_REM x K_REM*p) and Gamma (K_REM*p x K_REM*p) at the REM block.

    C[:, (h-1)*K_REM:h*K_REM] = (R(h))_REM for h = 1..p.
    Gamma is block-Toeplitz with blocks (R(|i-j|))_REM if j >= i else (R(i-j))_REM^T.
    """
    rem = slice(K_LOC, K_LOC + K_REM)

    C_blocks = [R_list[h][rem, rem] for h in range(1, p+1)]
    C = np.hstack(C_blocks)

    Gamma_blocks = []
    for i in range(1, p+1):
        row_blocks = []
        for j in range(1, p+1):
            if j >= i:
                blk = R_list[j-i][rem, rem]
            else:
                blk = R_list[i-j][rem, rem].T
            row_blocks.append(blk)
        Gamma_blocks.append(np.hstack(row_blocks))
    Gamma = np.vstack(Gamma_blocks)
    return C, Gamma


def closed_form_sigma2_C(A_list, Sigma_eps, R_list, K_LOC, K_REM, N_total):
    """sigma^2_C^(p) via the block-Toeplitz formula.

    Returns (sigma^2_C, dict_of_intermediates).
    """
    K = K_LOC + K_REM
    p = len(A_list)
    rem = slice(K_LOC, K)

    joint_term_full = sum(A_list[h-1] @ R_list[h].T for h in range(1, p+1))
    joint_term_REM = joint_term_full[rem, rem]

    C, Gamma = build_C_Gamma_REM(R_list, K_LOC, K_REM, p)
    C_Ginv_CT = C @ np.linalg.solve(Gamma, C.T)

    sigma2_C = float(np.trace(joint_term_REM - C_Ginv_CT)) / N_total
    Gamma_cond = float(np.linalg.cond(Gamma))
    Gamma_lambda_min = float(np.linalg.eigvalsh(Gamma).min())

    return sigma2_C, {
        'joint_term_REM_trace': float(np.trace(joint_term_REM)),
        'C_Ginv_CT_trace': float(np.trace(C_Ginv_CT)),
        'Gamma_condition_number': Gamma_cond,
        'Gamma_lambda_min': Gamma_lambda_min,
        'C_shape': C.shape,
        'Gamma_shape': Gamma.shape,
    }


def BLP_direct_sigma2_C(Sigma_f, Sigma_eps, R_list, K_LOC, K_REM, N_total):
    """Same number via (Sigma_f)_REM - C Gamma^{-1} C^T - (Sigma_eps)_REM.

    Algebraically identical to closed_form by Yule-Walker; cross-check path
    that does not invoke YW substitution.
    """
    K = K_LOC + K_REM
    p = len(R_list) - 1
    rem = slice(K_LOC, K)

    C, Gamma = build_C_Gamma_REM(R_list, K_LOC, K_REM, p)
    rest_var = Sigma_f[rem, rem] - C @ np.linalg.solve(Gamma, C.T)
    joint_var = Sigma_eps[rem, rem]
    return float(np.trace(rest_var - joint_var)) / N_total


def companion_form_sigma2_C(Atilde, Sigma_F, K, K_LOC, K_REM, p, N_total):
    """Apply paper's Theorem 1 at the companion VAR(1) with lifted REM_comp partition.

    REM_comp = REM components at all p lags (in F_t = (f_t, ..., f_{t-p+1})^T):
        idx = U_{h=0..p-1} [h*K + K_LOC, h*K + K_LOC + K_REM)
    """
    rem_comp_idx = np.concatenate([
        np.arange(h*K + K_LOC, h*K + K_LOC + K_REM) for h in range(p)
    ])

    SF_REM_comp = Sigma_F[np.ix_(rem_comp_idx, rem_comp_idx)]
    AtildeSF = Atilde @ Sigma_F
    AtildeSF_AtildeT = AtildeSF @ Atilde.T

    block_11 = AtildeSF_AtildeT[np.ix_(rem_comp_idx, rem_comp_idx)]
    block_21 = AtildeSF[np.ix_(rem_comp_idx, rem_comp_idx)]
    block_12 = block_21.T

    Schur = block_11 - block_21 @ np.linalg.solve(SF_REM_comp, block_12)
    sigma2_C = float(np.trace(Schur)) / N_total
    return sigma2_C, Schur, rem_comp_idx


def companion_cluster_decomposition(Atilde, Sigma_F, K, K_LOC_list, K_REM, p, N_total):
    """Theorem 2^(p) cluster decomposition at companion-state level.

    K_LOC_list: list of LOC block sizes [K_LOC_1, K_LOC_2, ..., K_LOC_J].

    Returns dict with per-LOC diagonals, cross-term, and M^comp(j, k) Frobenius norms.
    """
    J = len(K_LOC_list)
    K_LOC_total = sum(K_LOC_list)
    assert K_LOC_total + K_REM == K

    # Build companion-state index arrays per LOC block and for REM_comp
    loc_indices = []
    cum = 0
    for j, K_LOC_j in enumerate(K_LOC_list):
        idx_j = np.concatenate([np.arange(h*K + cum, h*K + cum + K_LOC_j) for h in range(p)])
        loc_indices.append(idx_j)
        cum += K_LOC_j
    rem_comp_idx = np.concatenate([
        np.arange(h*K + K_LOC_total, h*K + K_LOC_total + K_REM) for h in range(p)
    ])

    SF_REM = Sigma_F[np.ix_(rem_comp_idx, rem_comp_idx)]
    SF_REM_inv = np.linalg.inv(SF_REM)

    def M_comp(a_idx, b_idx):
        SF_ab = Sigma_F[np.ix_(a_idx, b_idx)]
        SF_a_REM = Sigma_F[np.ix_(a_idx, rem_comp_idx)]
        SF_REM_b = Sigma_F[np.ix_(rem_comp_idx, b_idx)]
        return SF_ab - SF_a_REM @ SF_REM_inv @ SF_REM_b

    diag_contribs = []
    cross_contribs_pairs = {}
    M_frobenius = {}
    for j in range(J):
        a_idx = loc_indices[j]
        A_REM_j = Atilde[np.ix_(rem_comp_idx, a_idx)]
        M_jj = M_comp(a_idx, a_idx)
        diag_j = float(np.trace(A_REM_j @ M_jj @ A_REM_j.T)) / N_total
        diag_contribs.append(diag_j)

    sigma2_cross = 0.0
    for j in range(J):
        for k in range(J):
            if j == k:
                continue
            a_idx, b_idx = loc_indices[j], loc_indices[k]
            A_REM_j = Atilde[np.ix_(rem_comp_idx, a_idx)]
            A_REM_k = Atilde[np.ix_(rem_comp_idx, b_idx)]
            M_jk = M_comp(a_idx, b_idx)
            cross_jk = float(np.trace(A_REM_j @ M_jk @ A_REM_k.T)) / N_total
            cross_contribs_pairs[f'{j}_{k}'] = cross_jk
            sigma2_cross += cross_jk
            M_frobenius[f'M_{j}_{k}_fro'] = float(np.linalg.norm(M_jk, 'fro'))

    return {
        'diag_per_LOC': diag_contribs,
        'sigma2_diag_total': sum(diag_contribs),
        'sigma2_cross': sigma2_cross,
        'sigma2_total': sum(diag_contribs) + sigma2_cross,
        'cross_contribs_pairs': cross_contribs_pairs,
        'M_frobenius': M_frobenius,
    }


def build_VAR_p_DGP(K, K_LOC_list, K_REM, sigma_cross, seed, p,
                     target_rho=0.85, sigma_eps_block_diag=False,
                     A_LOC_LOC_zero=False, A_h_REM_REM_zero_at_higher_lags=False,
                     A_h_REM_LOC_only_at_lag1=False,
                     A_h_LOC_REM_only_at_lag1=False):
    """Build a stable VAR(p) DGP with controlled block structure.

    Options:
    - sigma_eps_block_diag: Sigma_eps block-diagonal between each LOC_j and REM
    - A_LOC_LOC_zero: A_h has zero entries between LOC_j and LOC_k for j != k, at all h
    - A_h_REM_REM_zero_at_higher_lags: (A_h)_REM,REM = 0 for h >= 2 (REM dynamics are AR(1)-within)
    - A_h_REM_LOC_only_at_lag1: (A_h)_REM,LOC = 0 for h >= 2 (LOC affects REM only at lag 1)
    - A_h_LOC_REM_only_at_lag1: (A_h)_LOC,REM = 0 for h >= 2 (REM affects LOC only at lag 1)

    Returns (A_list, Sigma_eps, Sigma_F, R_list, Atilde, info_dict).
    """
    rng = np.random.default_rng(seed)
    K_LOC_total = sum(K_LOC_list)
    assert K_LOC_total + K_REM == K
    J = len(K_LOC_list)

    # LOC block index ranges
    loc_starts = np.cumsum([0] + list(K_LOC_list[:-1]))
    loc_ends = np.cumsum(K_LOC_list)

    A_list = []
    for h in range(p):
        A_h = np.zeros((K, K))
        decay = 0.6 ** h

        # Within-LOC_j self dynamics
        for j in range(J):
            s, e = loc_starts[j], loc_ends[j]
            A_h[s:e, s:e] = rng.standard_normal((K_LOC_list[j], K_LOC_list[j])) * 0.3 * decay

        # Within-REM self dynamics
        if not (A_h_REM_REM_zero_at_higher_lags and h >= 1):
            A_h[K_LOC_total:, K_LOC_total:] = rng.standard_normal((K_REM, K_REM)) * 0.3 * decay

        # REM <- LOC (cross-coupling)
        if not (A_h_REM_LOC_only_at_lag1 and h >= 1):
            for j in range(J):
                s, e = loc_starts[j], loc_ends[j]
                A_h[K_LOC_total:, s:e] = rng.standard_normal((K_REM, K_LOC_list[j])) * sigma_cross * decay

        # LOC <- REM
        if not (A_h_LOC_REM_only_at_lag1 and h >= 1):
            for j in range(J):
                s, e = loc_starts[j], loc_ends[j]
                A_h[s:e, K_LOC_total:] = rng.standard_normal((K_LOC_list[j], K_REM)) * sigma_cross * 0.5 * decay

        # LOC_j <-> LOC_k coupling (off-diagonal LOC blocks)
        if not A_LOC_LOC_zero:
            for j in range(J):
                for k in range(J):
                    if j != k:
                        s_j, e_j = loc_starts[j], loc_ends[j]
                        s_k, e_k = loc_starts[k], loc_ends[k]
                        A_h[s_j:e_j, s_k:e_k] = rng.standard_normal((K_LOC_list[j], K_LOC_list[k])) * sigma_cross * 0.3 * decay

        A_list.append(A_h)

    A_list, Atilde = rescale_to_target_rho(A_list, K, p, target_rho=target_rho)

    # Sigma_eps
    if sigma_eps_block_diag:
        Sigma_eps = np.zeros((K, K))
        for j in range(J):
            s, e = loc_starts[j], loc_ends[j]
            W_j = rng.standard_normal((K_LOC_list[j], K_LOC_list[j]))
            Sigma_eps[s:e, s:e] = W_j @ W_j.T + 0.5 * np.eye(K_LOC_list[j])
        WR = rng.standard_normal((K_REM, K_REM))
        Sigma_eps[K_LOC_total:, K_LOC_total:] = WR @ WR.T + 0.5 * np.eye(K_REM)
    else:
        W = rng.standard_normal((K, K))
        Sigma_eps = W @ W.T + 0.5 * np.eye(K)

    Sigma_F = companion_Sigma_F(Atilde, Sigma_eps, K, p)
    R_list = extract_R_list(Sigma_F, A_list, K, p)

    info = {
        'rho_Atilde': float(np.max(np.abs(np.linalg.eigvals(Atilde)))),
        'sigma_eps_block_diag': sigma_eps_block_diag,
        'A_LOC_LOC_zero': A_LOC_LOC_zero,
        'A_h_REM_REM_zero_at_higher_lags': A_h_REM_REM_zero_at_higher_lags,
        'A_h_REM_LOC_only_at_lag1': A_h_REM_LOC_only_at_lag1,
        'A_h_LOC_REM_only_at_lag1': A_h_LOC_REM_only_at_lag1,
        'sigma_cross': sigma_cross,
    }
    return A_list, Sigma_eps, Sigma_F, R_list, Atilde, info


def simulate_VAR_p(A_list, Sigma_eps, T, n_burn=500, seed=0):
    """Simulate a single stationary trajectory of VAR(p) of length T (after burn-in).

    Returns f of shape (T, K).
    """
    p = len(A_list)
    K = A_list[0].shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma_eps)
    eps = rng.standard_normal((T + n_burn + p, K)) @ L.T
    f = np.zeros((T + n_burn + p, K))
    for t in range(p):
        f[t] = rng.standard_normal(K) * 0.1
    for t in range(p, T + n_burn + p):
        f[t] = eps[t]
        for h in range(1, p+1):
            f[t] += A_list[h-1] @ f[t-h]
    return f[n_burn + p:]


def ols_var_p(f, p):
    """OLS estimate of VAR(p) coefficients and Sigma_eps from trajectory f of shape (T, K).

    Returns (A_hat_list, Sigma_eps_hat).
    A_hat_list: list of p (K x K) matrices.
    """
    T, K = f.shape
    # regression: f_t on (f_{t-1}, ..., f_{t-p}) for t = p..T-1
    n_eff = T - p
    Y = f[p:]                                          # (n_eff, K)
    X = np.zeros((n_eff, K * p))
    for h in range(1, p+1):
        X[:, (h-1)*K:h*K] = f[p-h:p-h+n_eff]
    # solve X B = Y in least squares: B = (X^T X)^{-1} X^T Y, B is (Kp, K)
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y                              # (Kp, K)
    # split B into A_hat_1, ..., A_hat_p: each (K, K)
    # B row block [h*K:(h+1)*K] corresponds to A_h^T, so A_h = B[h*K:(h+1)*K].T
    A_hat_list = [B[h*K:(h+1)*K].T for h in range(p)]
    resid = Y - X @ B
    Sigma_eps_hat = resid.T @ resid / n_eff
    return A_hat_list, Sigma_eps_hat


def plug_in_sigma2_C(A_hat_list, Sigma_eps_hat, K_LOC, K_REM, K, p, N_total):
    """Plug-in sigma^2_C^(p) via OLS-estimated (A_hat, Sigma_eps_hat) + derived Sigma_F."""
    Atilde_hat = build_companion_matrix(A_hat_list, K, p)
    rho = float(np.max(np.abs(np.linalg.eigvals(Atilde_hat))))
    if rho >= 1.0:
        return np.nan, {'rho_Atilde_hat': rho, 'unstable': True}
    Sigma_F_hat = companion_Sigma_F(Atilde_hat, Sigma_eps_hat, K, p)
    R_list_hat = extract_R_list(Sigma_F_hat, A_hat_list, K, p)
    sigma2_C, info = closed_form_sigma2_C(A_hat_list, Sigma_eps_hat, R_list_hat,
                                           K_LOC, K_REM, N_total)
    info['rho_Atilde_hat'] = rho
    info['unstable'] = False
    return sigma2_C, info
