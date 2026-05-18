"""E9 — Wald test for cross-block coupling under VAR(p): H_0: A_{h, REM, LOC} = 0 for all h.

Test statistic:
   W_T = T * vec(A_hat_{REM, LOC, all-h})^T Omega^{-1} vec(A_hat_{REM, LOC, all-h})
where Omega is the asymptotic covariance of vec(A_hat_{REM, LOC}) across all p lags,
estimated under the joint OLS-VAR(p) framework.

Under H_0, W_T -> chi^2(p * K_REM * K_LOC) asymptotically (since p * K_REM * K_LOC zero
constraints are tested simultaneously).

Verifications:
   V1. Empirical size at T in {200, 500, 1000, 2000}, n_rep=2000.
   V2. Empirical power at fixed T=1000 across sigma_cross sweep.
   V3. mean(W_T) at H_0 should approach E[chi^2(df)] = df as T grows.
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
import scipy.stats as stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common


def compute_wald_stat(A_hat_list, Sigma_eps_hat, K_LOC, K_REM, K, p, T):
    """Wald statistic for H_0: A_{h, REM, LOC} = 0 for all h.

    The asymptotic covariance of vec(A_hat) under OLS-VAR(p) is
        (Z^T Z / T)^{-1} otimes Sigma_eps
    where Z is the (T-p) x (K*p) regressor matrix of stacked lagged f's.
    Equivalently, the *population* asymptotic covariance is
        Gamma_p^{-1} otimes Sigma_eps
    where Gamma_p is the (K*p x K*p) population covariance of (f_{t-1}, ..., f_{t-p})^T,
    which is the block-Toeplitz of (R(|i-j|))_full.

    Wald statistic uses Gamma_p_hat^{-1} (LOC sub-block) otimes Sigma_eps_REM^{-1}_hat.

    We compute it sub-block-wise: extract A_hat_{h, REM, LOC} for h=1..p,
    stack into a (K_REM, K_LOC * p) matrix, vectorize, and use Kronecker structure.
    """
    # Stack A_hat_{h, REM, LOC} across h
    A_stack = np.zeros((K_REM, K_LOC * p))
    for h in range(p):
        A_stack[:, h*K_LOC:(h+1)*K_LOC] = A_hat_list[h][K_LOC:, :K_LOC]

    # vec(A_stack)
    vec_A = A_stack.flatten('F')  # length K_REM * K_LOC * p

    # Build full Gamma_p = E[Z_t Z_t^T] (K*p x K*p) where Z_t = (f_{t-1}, ..., f_{t-p})^T,
    # then take the Schur complement at the LOC sub-block:
    #   M_LOC^(p) = ((Gamma_p^{-1})_LOC,LOC)^{-1}
    # By Lutkepohl Prop 3.1, Cov(vec(B_hat)) ~ (1/T) * Gamma_p^{-1} kron Sigma_eps. The asymptotic
    # inverse covariance of vec(B_hat[LOC_at_all_lags, REM]) is M_LOC^(p) kron Sigma_eps_REM^{-1}.
    # This is the natural generalization of paper's Theorem 5 to VAR(p): partitioned-inverse
    # identity (Gamma_p^{-1})_LOC,LOC = M_LOC^(p),-1.
    Atilde_hat = common.build_companion_matrix(A_hat_list, K, p)
    if float(np.max(np.abs(np.linalg.eigvals(Atilde_hat)))) >= 1.0:
        return np.nan
    Sigma_F_hat = common.companion_Sigma_F(Atilde_hat, Sigma_eps_hat, K, p)
    R_hat_list = common.extract_R_list(Sigma_F_hat, A_hat_list, K, p)

    # Build full Gamma_p (K*p x K*p) — block-Toeplitz of R(|i-j|) of the FULL state
    Gamma_p_blocks = []
    for i in range(1, p+1):
        row_blocks = []
        for j in range(1, p+1):
            if j >= i:
                blk = R_hat_list[j-i]
            else:
                blk = R_hat_list[i-j].T
            row_blocks.append(blk)
        Gamma_p_blocks.append(np.hstack(row_blocks))
    Gamma_p = np.vstack(Gamma_p_blocks)  # (K*p, K*p)

    # Index arrays: LOC at all lags, REM at all lags within Gamma_p
    loc_idx_all_lags = np.concatenate([np.arange(h*K, h*K + K_LOC) for h in range(p)])
    rem_idx_all_lags = np.concatenate([np.arange(h*K + K_LOC, h*K + K) for h in range(p)])

    # M_LOC^(p) = Schur complement of (Gamma_p)_REM,REM at LOC sub-block
    G_LL = Gamma_p[np.ix_(loc_idx_all_lags, loc_idx_all_lags)]
    G_LR = Gamma_p[np.ix_(loc_idx_all_lags, rem_idx_all_lags)]
    G_RL = Gamma_p[np.ix_(rem_idx_all_lags, loc_idx_all_lags)]
    G_RR = Gamma_p[np.ix_(rem_idx_all_lags, rem_idx_all_lags)]
    M_LOC_p = G_LL - G_LR @ np.linalg.solve(G_RR, G_RL)

    Sigma_eps_REM = Sigma_eps_hat[K_LOC:, K_LOC:]
    Sigma_eps_REM_inv = np.linalg.inv(Sigma_eps_REM)

    # Wald: W_T = T * vec_A^T (M_LOC_p kron Sigma_eps_REM^{-1}) vec_A
    # = T * trace(Sigma_eps_REM^{-1} A_stack M_LOC_p A_stack^T)
    Wald = T * float(np.trace(Sigma_eps_REM_inv @ A_stack @ M_LOC_p @ A_stack.T))
    return Wald


def main():
    print("=" * 100)
    print(" E9 — Wald test for cross-block coupling, VAR(p), p=2")
    print("=" * 100)

    K_LOC, K_REM, p = 4, 2, 2
    K = K_LOC + K_REM
    df = p * K_REM * K_LOC
    print(f"  df = p * K_REM * K_LOC = {p} * {K_REM} * {K_LOC} = {df}")
    print(f"  chi^2_{df}: mean={df}, var={2*df}")
    print(f"  critical value chi^2_{df} 95%: {stats.chi2.ppf(0.95, df):.4f}")

    # ---------- Under H_0: build DGP with A_{h, REM, LOC} = 0 for all h ----------
    print(f"\n[H_0 size test]")

    def build_H0_DGP(seed):
        A_list, Sigma_eps, Sigma_F, R_list, Atilde, info = common.build_VAR_p_DGP(
            K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.0,  # implies REM<-LOC blocks zero (initially)
            seed=seed, p=p)
        # Explicitly enforce A_{h, REM, LOC} = 0 for all h (in case there's any residual from rescaling)
        for h in range(p):
            A_list[h][K_LOC:, :K_LOC] = 0.0
        # rescale
        A_list, Atilde = common.rescale_to_target_rho(A_list, K, p, target_rho=0.85)
        Sigma_F = common.companion_Sigma_F(Atilde, Sigma_eps, K, p)
        R_list = common.extract_R_list(Sigma_F, A_list, K, p)
        return A_list, Sigma_eps, Sigma_F, R_list, Atilde, info

    A_H0_list, Sigma_eps_H0, _, _, _, _ = build_H0_DGP(seed=90001)

    print(f"  {'T':>8} {'emp size (alpha=0.05)':>22} {'mean W_T':>12} {'expected':>10}")
    size_results = []
    for T in [200, 500, 1000, 2000]:
        t0 = time.time()
        n_rep = 2000
        W_stats = []
        n_unstable = 0
        for r in range(n_rep):
            f = common.simulate_VAR_p(A_H0_list, Sigma_eps_H0, T, n_burn=500, seed=90000 + r)
            A_hat, S_eps_hat = common.ols_var_p(f, p)
            W = compute_wald_stat(A_hat, S_eps_hat, K_LOC, K_REM, K, p, T)
            if np.isnan(W):
                n_unstable += 1
                continue
            W_stats.append(W)
        W_stats = np.array(W_stats)
        emp_size = float(np.mean(W_stats > stats.chi2.ppf(0.95, df)))
        mean_W = float(W_stats.mean())
        elapsed = time.time() - t0
        print(f"  {T:>8} {emp_size:>22.4f} {mean_W:>12.2f} {df:>10}  ({n_unstable} unstable, {elapsed:.1f}s)")
        size_results.append({
            'T': T, 'n_rep': n_rep, 'n_unstable': n_unstable,
            'emp_size_alpha05': emp_size, 'mean_W': mean_W, 'df_expected': df,
        })

    # ---------- Power at T=1000 across sigma_cross sweep ----------
    print(f"\n[Power at T=1000]")
    print(f"  {'sigma_cross':>12} {'sigma^2_C':>14} {'emp power (alpha=0.05)':>22}")
    power_results = []
    T_power = 1000
    for sigma_cross in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
        A_list_alt, Sigma_eps_alt, Sigma_F_alt, R_list_alt, Atilde_alt, info_alt = \
            common.build_VAR_p_DGP(K=K, K_LOC_list=[K_LOC], K_REM=K_REM,
                                    sigma_cross=sigma_cross, seed=91000 + int(sigma_cross*100),
                                    p=p)
        sigma2_C_alt, _ = common.closed_form_sigma2_C(
            A_list_alt, Sigma_eps_alt, R_list_alt, K_LOC, K_REM, K_REM * 2)
        n_rep = 1000
        n_reject = 0
        n_unstable = 0
        for r in range(n_rep):
            f = common.simulate_VAR_p(A_list_alt, Sigma_eps_alt, T_power, n_burn=500, seed=92000 + int(sigma_cross*100) + r)
            A_hat, S_eps_hat = common.ols_var_p(f, p)
            W = compute_wald_stat(A_hat, S_eps_hat, K_LOC, K_REM, K, p, T_power)
            if np.isnan(W):
                n_unstable += 1
                continue
            if W > stats.chi2.ppf(0.95, df):
                n_reject += 1
        emp_power = n_reject / max(n_rep - n_unstable, 1)
        print(f"  {sigma_cross:>12.2f} {sigma2_C_alt:>14.6f} {emp_power:>22.4f}")
        power_results.append({
            'sigma_cross': sigma_cross, 'sigma2_C_true': sigma2_C_alt,
            'emp_power_alpha05': emp_power, 'n_unstable': n_unstable,
        })

    # SUMMARY
    print("\n" + "=" * 100)
    print(" E9 SUMMARY")
    print("=" * 100)
    print(f"  H_0 size at T=1000: {[r['emp_size_alpha05'] for r in size_results if r['T']==1000][0]:.4f} "
          f"(nominal 0.05; PASS if in [0.025, 0.075])")
    print(f"  Mean W_T at T=2000: {[r['mean_W'] for r in size_results if r['T']==2000][0]:.2f} "
          f"(expected df = {df})")
    print(f"  Power at sigma_cross=0.4: {[r['emp_power_alpha05'] for r in power_results if r['sigma_cross']==0.4][0]:.4f}")

    out_path = HERE / 'E9_wald_test_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'df': df,
            'size_results': size_results,
            'power_results': power_results,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
