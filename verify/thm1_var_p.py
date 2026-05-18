"""E1 — verify the closed-form sigma^2_C^(p) extension for VAR(p), p=2.

Closed-form derivation (orchestrator scratch, post-v5):
  Joint predictor:    yhat^joint = sum_h (A_h f_{t-h})_REM,  variance = (Sigma_eps)_REM
  Restricted predictor: BLP of f_REM,t given Z = (f_REM,{t-1}, ..., f_REM,{t-p})
                      variance = (Sigma_f)_REM - C Gamma^{-1} C^T
                      where C    = [(R(1))_REM, ..., (R(p))_REM]    (K_REM x K_REM*p)
                            Gamma = block-Toeplitz of (R(|i-j|))_REM (K_REM*p x K_REM*p)
  Difference + Yule-Walker REM-block substitution:
      sigma^2_C^(p) = (1/N_total) tr[ sum_h (A_h R(h)^T)_REM - C Gamma^{-1} C^T ]
  For p=1: C = (R(1))_REM = (A Sigma_f)_REM, Gamma = (Sigma_f)_REM, recovers Theorem 1.

Verifications (this script):
  V1. Direct closed form (above) == BLP-direct form (sigma_f)_REM - C Gamma^{-1} C^T - (sigma_eps)_REM.
  V2. Direct closed form == MC empirical at T large.
  V3. At p=1 (set A_2 = 0), recovers the paper's Theorem 1 numerical value.
"""
from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def build_VAR_p(K, K_LOC, K_REM, sigma_cross, seed, p, target_rho=0.85):
    """Build a stable VAR(p) DGP. Returns (A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F).

    A_list: [A_1, ..., A_p], each K x K.
    R_list: [R(0), R(1), ..., R(p)], each K x K, R(0) = Sigma_f.
    Atilde: K*p x K*p companion matrix.
    Sigma_F: K*p x K*p companion-form stationary covariance.
    """
    rng = np.random.default_rng(seed)

    # initial A_h with decreasing magnitude across lags
    A_list = []
    for h in range(p):
        A_h = rng.standard_normal((K, K)) * 0.4 * (0.6 ** h)
        # cross-block coupling at lag 1 only (will be relaxed in E3)
        if h == 0:
            A_h[K_LOC:, 0:K_LOC] = rng.standard_normal((K_REM, K_LOC)) * sigma_cross
            A_h[0:K_LOC, K_LOC:] = rng.standard_normal((K_LOC, K_REM)) * sigma_cross * 0.5
        A_list.append(A_h)

    def build_Atilde(A_list_):
        At = np.zeros((K*p, K*p))
        for h in range(p):
            At[0:K, h*K:(h+1)*K] = A_list_[h]
        for h in range(1, p):
            At[h*K:(h+1)*K, (h-1)*K:h*K] = np.eye(K)
        return At

    # iteratively scale all A_h by a common factor to hit target_rho
    s = 1.0
    for _ in range(80):
        A_list_test = [s * A for A in A_list]
        At = build_Atilde(A_list_test)
        rho = float(np.max(np.abs(np.linalg.eigvals(At))))
        if abs(rho - target_rho) < 1e-10:
            break
        s = s * target_rho / max(rho, 1e-12)
    A_list = A_list_test
    Atilde = At

    W = rng.standard_normal((K, K))
    Sigma_eps = W @ W.T + 0.5 * np.eye(K)

    # Companion-form Sigma_F: vec(Sigma_F) = (I - Atilde kron Atilde)^{-1} vec(B Sigma_eps B^T)
    Kp = K * p
    Sigma_eps_comp = np.zeros((Kp, Kp))
    Sigma_eps_comp[0:K, 0:K] = Sigma_eps
    AkronA = np.kron(Atilde, Atilde)
    vec_S_eps_comp = Sigma_eps_comp.flatten('F')
    vec_Sigma_F = np.linalg.solve(np.eye(Kp**2) - AkronA, vec_S_eps_comp)
    Sigma_F = vec_Sigma_F.reshape((Kp, Kp), order='F')
    Sigma_F = (Sigma_F + Sigma_F.T) / 2  # symmetrize for numerical hygiene

    # Sigma_f = R(0)
    Sigma_f = Sigma_F[0:K, 0:K]

    # R(h) for h = 1..p via direct extraction from Sigma_F + Yule-Walker for h = p
    R_list = [Sigma_f]
    for h in range(1, p):
        # E[f_t f_{t-h}^T] = Sigma_F block at (rows 0:K, cols h*K:(h+1)*K)
        R_h = Sigma_F[0:K, h*K:(h+1)*K]
        R_list.append(R_h)
    # R(p) via Yule-Walker: R(p) = sum_{q=1}^p A_q R(p-q)
    # uses R(p-q) for q=1..p, i.e., R(p-1), R(p-2), ..., R(0).
    R_p = sum(A_list[q-1] @ R_list[p-q] for q in range(1, p+1))
    R_list.append(R_p)

    return A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F


def closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total):
    """sigma^2_C^(p) = (1/N) tr[ sum_h (A_h R(h)^T)_REM - C Gamma^{-1} C^T ]."""
    K = K_LOC + K_REM
    p = len(A_list)
    rem = slice(K_LOC, K)

    # sum_h (A_h R(h)^T)_REM
    joint_term_full = sum(A_list[h-1] @ R_list[h].T for h in range(1, p+1))
    joint_term_REM = joint_term_full[rem, rem]

    # C: K_REM x K_REM*p
    C_blocks = [R_list[h][rem, rem] for h in range(1, p+1)]
    C = np.hstack(C_blocks)

    # Gamma: K_REM*p x K_REM*p block-Toeplitz of (R(|i-j|))_REM
    # rows indexed by lag i in 1..p, cols by lag j in 1..p
    # block (i, j) = E[f_REM,{t-i} f_REM,{t-j}^T] = R(j-i)_REM if j >= i, else R(i-j)^T_REM
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

    # tr[C Gamma^{-1} C^T]
    C_Ginv_CT = C @ np.linalg.solve(Gamma, C.T)

    sigma2_C = float(np.trace(joint_term_REM - C_Ginv_CT)) / N_total
    return sigma2_C, joint_term_REM, C, Gamma


def BLP_direct_sigma2_C(Sigma_f, Sigma_eps, R_list, K_LOC, K_REM, N_total):
    """Same number computed via (Sigma_f)_REM - C Gamma^{-1} C^T - (Sigma_eps)_REM.

    Algebraically identical to closed_form_sigma2_C by Yule-Walker; this is the
    'sanity' path that does not invoke the YW substitution.
    """
    K = K_LOC + K_REM
    p = len(R_list) - 1
    rem = slice(K_LOC, K)

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

    rest_var = Sigma_f[rem, rem] - C @ np.linalg.solve(Gamma, C.T)
    joint_var = Sigma_eps[rem, rem]
    return float(np.trace(rest_var - joint_var)) / N_total


def MC_empirical_sigma2_C(A_list, Sigma_eps, K_LOC, K_REM, N_total,
                          T=20000, n_burn=500, n_rep=50, seed=10001):
    """MC: simulate VAR(p), estimate Var(rest err - joint err) at REM_t empirically."""
    K = K_LOC + K_REM
    p = len(A_list)
    rng = np.random.default_rng(seed)

    L = np.linalg.cholesky(Sigma_eps)
    rem = slice(K_LOC, K)

    diffs = []
    for r in range(n_rep):
        # simulate trajectory of length T + n_burn + p
        eps = rng.standard_normal((T + n_burn + p, K)) @ L.T  # K x K Cholesky
        f = np.zeros((T + n_burn + p, K))
        # init: random small
        for t in range(p):
            f[t] = rng.standard_normal(K) * 0.1
        for t in range(p, T + n_burn + p):
            f[t] = eps[t]
            for h in range(1, p+1):
                f[t] += A_list[h-1] @ f[t-h]

        f = f[n_burn:]  # discard burn-in

        # joint err at REM: f_REM,t - sum_h (A_h f_{t-h})_REM = eps_REM,t
        # restricted err at REM: f_REM,t - BLP(f_REM,t | f_REM,{t-1..t-p})
        # We compute restricted err empirically via OLS of f_REM,t on (f_REM,{t-1..t-p})
        # (the BLP coefficients are the OLS coefficients in expectation, for stationary
        # Gaussian; numerical computation on this trajectory recovers them)

        f_REM = f[:, K_LOC:]  # (T+p) x K_REM
        # build regressors: X_t = (f_REM,{t-1}, ..., f_REM,{t-p}) for t = p..T+p-1
        T_use = f_REM.shape[0] - p
        X = np.zeros((T_use, K_REM * p))
        for h in range(1, p+1):
            X[:, (h-1)*K_REM:h*K_REM] = f_REM[p-h:p-h+T_use]
        Y = f_REM[p:p+T_use]

        # OLS: coef = (X^T X)^{-1} X^T Y, but we want population BLP, which we recover
        # by using true (R(h))_REM and Gamma instead. For empirical comparison, fit OLS:
        coef = np.linalg.solve(X.T @ X, X.T @ Y)
        Yhat_rest = X @ coef
        rest_err = Y - Yhat_rest
        rest_var_emp = rest_err.T @ rest_err / T_use

        # joint err: eps_REM,t at times t = p..T+p-1
        joint_err = np.zeros((T_use, K_REM))
        for t in range(p, p + T_use):
            joint_err[t - p] = f[t, K_LOC:] - sum(A_list[h-1] @ f[t-h] for h in range(1, p+1))[K_LOC:]
        joint_var_emp = joint_err.T @ joint_err / T_use

        diff = float(np.trace(rest_var_emp - joint_var_emp)) / N_total
        diffs.append(diff)

    return float(np.mean(diffs)), float(np.std(diffs) / np.sqrt(n_rep))


def main():
    print("=" * 88)
    print(" E1 — sigma^2_C^(p) closed-form verification at p=2")
    print("=" * 88)

    results = {}

    # ---------- Instance 1: VAR(2), K=4, blocks (2 LOC, 2 REM), moderate cross-coupling
    print("\n[Instance 1] VAR(2) K=4 K_LOC=2 K_REM=2 sigma_cross=0.3 seed=1001")
    K, K_LOC, K_REM = 4, 2, 2
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.3, seed=1001, p=2)
    print(f"  rho(Atilde) = {float(np.max(np.abs(np.linalg.eigvals(Atilde)))):.6f}")
    print(f"  (Sigma_eps)_REM eigenvalues: {np.linalg.eigvalsh(Sigma_eps[K_LOC:, K_LOC:])}")
    print(f"  (Sigma_f)_REM eigenvalues: {np.linalg.eigvalsh(Sigma_f[K_LOC:, K_LOC:])}")

    sigma2_closed, _, _, _ = closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total)
    sigma2_BLP = BLP_direct_sigma2_C(Sigma_f, Sigma_eps, R_list, K_LOC, K_REM, N_total)
    print(f"  V1.a closed form sigma^2_C^(2)  = {sigma2_closed:.10f}")
    print(f"  V1.b BLP-direct sigma^2_C^(2)   = {sigma2_BLP:.10f}")
    diff_V1 = abs(sigma2_closed - sigma2_BLP)
    print(f"  V1 residual (closed vs BLP)     = {diff_V1:.3e}  ({'PASS' if diff_V1 < 1e-10 else 'FAIL'} at < 1e-10)")

    # MC
    print(f"  V2 MC empirical (T=20000, n_rep=50)...")
    t0 = time.time()
    mc_mean, mc_se = MC_empirical_sigma2_C(A_list, Sigma_eps, K_LOC, K_REM, N_total,
                                            T=20000, n_burn=500, n_rep=50, seed=10001)
    mc_elapsed = time.time() - t0
    print(f"  V2 MC mean ± 1 SE                = {mc_mean:.6f} ± {mc_se:.6f}  ({mc_elapsed:.1f}s)")
    mc_rel_err = abs(mc_mean - sigma2_closed) / max(abs(sigma2_closed), 1e-10) * 100
    n_se_away = abs(mc_mean - sigma2_closed) / max(mc_se, 1e-10)
    print(f"  V2 rel err vs closed             = {mc_rel_err:.3f}%  ({n_se_away:.2f} MC SE away)")
    v2_pass = n_se_away < 3.0  # within 3 MC SE
    print(f"  V2 {'PASS' if v2_pass else 'FAIL'} at < 3 MC SE")

    results['instance_1'] = {
        'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': 2, 'sigma_cross': 0.3, 'seed': 1001,
        'sigma2_C_closed_form': sigma2_closed,
        'sigma2_C_BLP_direct': sigma2_BLP,
        'V1_residual': float(diff_V1),
        'V1_pass': bool(diff_V1 < 1e-10),
        'sigma2_C_MC_mean': mc_mean,
        'sigma2_C_MC_SE': mc_se,
        'V2_rel_err_pct': float(mc_rel_err),
        'V2_n_SE_away': float(n_se_away),
        'V2_pass': bool(v2_pass),
    }

    # ---------- Instance 2: VAR(2), K=6, blocks (LOC1=2, LOC2=2, REM=2)
    print("\n[Instance 2] VAR(2) K=6 K_LOC=4 K_REM=2 sigma_cross=0.3 seed=2002")
    K, K_LOC, K_REM = 6, 4, 2
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.3, seed=2002, p=2)
    print(f"  rho(Atilde) = {float(np.max(np.abs(np.linalg.eigvals(Atilde)))):.6f}")

    sigma2_closed, _, _, _ = closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total)
    sigma2_BLP = BLP_direct_sigma2_C(Sigma_f, Sigma_eps, R_list, K_LOC, K_REM, N_total)
    print(f"  V1.a closed form sigma^2_C^(2)  = {sigma2_closed:.10f}")
    print(f"  V1.b BLP-direct sigma^2_C^(2)   = {sigma2_BLP:.10f}")
    diff_V1 = abs(sigma2_closed - sigma2_BLP)
    print(f"  V1 residual                     = {diff_V1:.3e}  ({'PASS' if diff_V1 < 1e-10 else 'FAIL'} at < 1e-10)")

    print(f"  V2 MC empirical (T=20000, n_rep=50)...")
    t0 = time.time()
    mc_mean, mc_se = MC_empirical_sigma2_C(A_list, Sigma_eps, K_LOC, K_REM, N_total,
                                            T=20000, n_burn=500, n_rep=50, seed=20002)
    mc_elapsed = time.time() - t0
    print(f"  V2 MC mean ± 1 SE                = {mc_mean:.6f} ± {mc_se:.6f}  ({mc_elapsed:.1f}s)")
    mc_rel_err = abs(mc_mean - sigma2_closed) / max(abs(sigma2_closed), 1e-10) * 100
    n_se_away = abs(mc_mean - sigma2_closed) / max(mc_se, 1e-10)
    print(f"  V2 rel err                       = {mc_rel_err:.3f}%  ({n_se_away:.2f} MC SE away)")
    v2_pass = n_se_away < 3.0
    print(f"  V2 {'PASS' if v2_pass else 'FAIL'} at < 3 MC SE")

    results['instance_2'] = {
        'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': 2, 'sigma_cross': 0.3, 'seed': 2002,
        'sigma2_C_closed_form': sigma2_closed,
        'sigma2_C_BLP_direct': sigma2_BLP,
        'V1_residual': float(diff_V1),
        'V1_pass': bool(diff_V1 < 1e-10),
        'sigma2_C_MC_mean': mc_mean,
        'sigma2_C_MC_SE': mc_se,
        'V2_rel_err_pct': float(mc_rel_err),
        'V2_n_SE_away': float(n_se_away),
        'V2_pass': bool(v2_pass),
    }

    # ---------- V3: p=1 reduces to paper's Theorem 1
    print("\n[Instance 3] V3 — set A_2 = 0, verify VAR(2) closed form reduces to Theorem 1 (p=1)")
    K, K_LOC, K_REM = 4, 2, 2
    N_total = K_REM * 2
    A_list_2, Sigma_eps, Sigma_f, R_list_2, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.3, seed=3003, p=2)
    A_list_2[1] = np.zeros_like(A_list_2[1])  # zero out A_2

    # rebuild Sigma_F, Sigma_f, R(0..2) with A_2 = 0
    Kp = K * 2
    Atilde = np.zeros((Kp, Kp))
    Atilde[0:K, 0:K] = A_list_2[0]
    Atilde[0:K, K:2*K] = A_list_2[1]
    Atilde[K:2*K, 0:K] = np.eye(K)
    Sigma_eps_comp = np.zeros((Kp, Kp))
    Sigma_eps_comp[0:K, 0:K] = Sigma_eps
    AkronA = np.kron(Atilde, Atilde)
    vec_S_eps_comp = Sigma_eps_comp.flatten('F')
    vec_Sigma_F = np.linalg.solve(np.eye(Kp**2) - AkronA, vec_S_eps_comp)
    Sigma_F_2 = vec_Sigma_F.reshape((Kp, Kp), order='F')
    Sigma_F_2 = (Sigma_F_2 + Sigma_F_2.T) / 2
    Sigma_f_2 = Sigma_F_2[0:K, 0:K]
    R_list_2 = [Sigma_f_2, Sigma_F_2[0:K, K:2*K]]
    R_2 = A_list_2[0] @ R_list_2[1] + A_list_2[1] @ R_list_2[0]
    R_list_2.append(R_2)

    sigma2_p2 = closed_form_sigma2_C(A_list_2, R_list_2, Sigma_eps, K_LOC, K_REM, N_total)[0]

    # paper's Theorem 1 (p=1) for the same A_1, Sigma_f (note: Sigma_f for VAR(1) is solved
    # from discrete Lyapunov of A_1, not from the VAR(2) companion; should equal Sigma_f_2)
    # Actually Sigma_f_2 with A_2 = 0 IS the VAR(1) stationary covariance because the
    # companion-form Lyapunov collapses.
    A = A_list_2[0]
    rem = slice(K_LOC, K)
    sigma2_p1 = float(np.trace(
        (A @ Sigma_f_2 @ A.T)[rem, rem]
        - (A @ Sigma_f_2)[rem, rem] @ np.linalg.solve(Sigma_f_2[rem, rem], (Sigma_f_2 @ A.T)[rem, rem])
    )) / N_total

    print(f"  VAR(2) closed-form with A_2=0:  {sigma2_p2:.10f}")
    print(f"  paper's Theorem 1 (p=1):         {sigma2_p1:.10f}")
    diff_V3 = abs(sigma2_p2 - sigma2_p1)
    print(f"  V3 residual                      = {diff_V3:.3e}  ({'PASS' if diff_V3 < 1e-10 else 'FAIL'} at < 1e-10)")

    results['instance_3'] = {
        'description': 'A_2 = 0 reduces VAR(2) to VAR(1); closed form must match Theorem 1',
        'sigma2_C_VAR2_A2_zero': sigma2_p2,
        'sigma2_C_VAR1_Theorem1': sigma2_p1,
        'V3_residual': float(diff_V3),
        'V3_pass': bool(diff_V3 < 1e-10),
    }

    # ---------- SUMMARY
    print("\n" + "=" * 88)
    print(" E1 SUMMARY")
    print("=" * 88)
    n_pass = sum(r.get('V1_pass', False) for r in results.values()) \
           + sum(r.get('V2_pass', False) for r in results.values()) \
           + sum(r.get('V3_pass', False) for r in results.values())
    print(f"  Instance 1: V1 closed=BLP {'PASS' if results['instance_1']['V1_pass'] else 'FAIL'}, "
          f"V2 MC {'PASS' if results['instance_1']['V2_pass'] else 'FAIL'} "
          f"({results['instance_1']['V2_rel_err_pct']:.2f}% rel err)")
    print(f"  Instance 2: V1 closed=BLP {'PASS' if results['instance_2']['V1_pass'] else 'FAIL'}, "
          f"V2 MC {'PASS' if results['instance_2']['V2_pass'] else 'FAIL'} "
          f"({results['instance_2']['V2_rel_err_pct']:.2f}% rel err)")
    print(f"  Instance 3: V3 (p=1 reduction) {'PASS' if results['instance_3']['V3_pass'] else 'FAIL'} "
          f"(residual {results['instance_3']['V3_residual']:.3e})")
    overall_pass = (results['instance_1']['V1_pass'] and results['instance_1']['V2_pass']
                    and results['instance_2']['V1_pass'] and results['instance_2']['V2_pass']
                    and results['instance_3']['V3_pass'])
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  Closed-form sigma^2_C^(p) verified for VAR(2) at K in {{4, 6}}: " +
          ("YES" if overall_pass else "PARTIAL"))

    out_path = HERE / 'E1_thm1_var_p_results.json'
    with open(out_path, 'w') as f:
        json.dump({'overall_pass': bool(overall_pass), 'instances': results}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
