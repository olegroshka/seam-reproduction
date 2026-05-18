"""E7 — sampling theory for sigma^2_C^(p): consistency, asymptotic V, CI coverage at p=2.

Plug-in estimator sigma_hat^2_C^(p) via OLS-VAR(p) on simulated trajectories.

V1. Consistency: sigma_hat^2_C^(p) -> sigma^2_C^(p) as T grows. Report rel err at
    T in {500, 1000, 2000, 5000, 10000, 20000} with n_rep replications each.
V2. sqrt(T)-rate: T * Var(sigma_hat^2_C^(p)) approximately constant across T.
V3. Asymptotic V via delta method:
       V = (grad phi^(p))^T Omega^(p) (grad phi^(p))
    grad phi^(p) computed via finite-diff w.r.t. (A_1, ..., A_p, Sigma_f) parameters.
    Omega^(p) estimated via MC on a long auxiliary panel.
V4. CI coverage at T=2000: nominal 95% from N(sigma_hat^2_C, V/T) should achieve ~95%.
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
sys.path.insert(0, str(HERE))
import common


def plug_in_one_rep(A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total, T, seed):
    f = common.simulate_VAR_p(A_list, Sigma_eps, T, n_burn=500, seed=seed)
    A_hat_list, Sigma_eps_hat = common.ols_var_p(f, p)
    sigma2_hat, info = common.plug_in_sigma2_C(A_hat_list, Sigma_eps_hat, K_LOC, K_REM, K, p, N_total)
    return sigma2_hat, info


def consistency_sweep(A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total, sigma2_true,
                      T_values, n_rep, base_seed=20000):
    rows = []
    for T in T_values:
        t0 = time.time()
        ests = []
        n_unstable = 0
        for r in range(n_rep):
            s, info = plug_in_one_rep(A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total,
                                       T=T, seed=base_seed + r)
            if info.get('unstable', False):
                n_unstable += 1
                continue
            ests.append(s)
        ests = np.array(ests)
        mean = float(ests.mean()) if len(ests) > 0 else float('nan')
        sd = float(ests.std(ddof=1)) if len(ests) > 1 else float('nan')
        rel_err = (mean - sigma2_true) / max(abs(sigma2_true), 1e-12) * 100
        sqrtT_sd = np.sqrt(T) * sd
        T_var = T * float(ests.var(ddof=1)) if len(ests) > 1 else float('nan')
        elapsed = time.time() - t0
        rows.append({
            'T': T, 'n_rep_used': int(len(ests)), 'n_unstable_dropped': int(n_unstable),
            'MC_mean': mean, 'MC_SD': sd,
            'rel_err_pct': float(rel_err),
            'sqrt_T_times_SD': float(sqrtT_sd),
            'T_times_Var': float(T_var),
            'elapsed_s': elapsed,
        })
        print(f"    T={T:>6}: MC mean = {mean:.6f}  rel err = {rel_err:+.3f}%  "
              f"sqrt(T)*SD = {sqrtT_sd:.4f}  T*Var = {T_var:.4f}  "
              f"(unstable dropped: {n_unstable})")
    return rows


def finite_diff_grad_phi_p(A_list, Sigma_eps_for_Sigma_f, K_LOC, K_REM, K, p, N_total, eps=1e-6):
    """Finite-difference gradient of phi^(p) w.r.t. (vec A_1, ..., vec A_p, vech Sigma_f).

    Note: parameter is (A_h)_{h=1..p} and Sigma_f directly (NOT Sigma_eps); for VAR(p)
    the natural sampling parameters are coefficients + Sigma_f (or equivalently Sigma_eps,
    related via Lyapunov). We parameterize by (A_h)_{h=1..p}, Sigma_f and compute the
    gradient.

    Returns gradient of length p*K^2 + K*(K+1)/2, in order
    [vec(A_1), vec(A_2), ..., vec(A_p), vech(Sigma_f)].
    """
    # Build reference Sigma_F to get R(h) list at the perturbed parameter values.
    Atilde = common.build_companion_matrix(A_list, K, p)
    Sigma_F = common.companion_Sigma_F(Atilde, Sigma_eps_for_Sigma_f, K, p)
    Sigma_f = Sigma_F[0:K, 0:K]

    def phi_at(A_list_p, Sigma_f_p):
        # Given (A_list_p, Sigma_f_p), construct R(h) consistent with these.
        # Approach: solve Yule-Walker as a linear system in R(0)..R(p-1).
        # Or simpler: since we have Sigma_f directly, derive R(h) for h=1..p via
        # Yule-Walker recursion using A_h:
        #   R(h) = sum_q A_q R(h-q) for h >= 1, with R(0) = Sigma_f, R(-h) = R(h)^T.
        # For h=1: R(1) = A_1 R(0) + A_2 R(-1) + ... + A_p R(-(p-1))
        #               = A_1 Sigma_f + sum_{q=2..p} A_q R(q-1)^T
        # This is a linear system in (R(1), ..., R(p-1)).
        # Easier to just simulate it implicitly via companion-form Sigma_F if we ensure
        # Sigma_f matches. To ensure consistency between (A, Sigma_f), use Sigma_eps =
        # Sigma_f - sum_q A_q (R(q))^T (the Lyapunov inverse). But this is circular.
        #
        # Cleaner: parameterize via (A_list_p, Sigma_eps_p) and recover Sigma_f from
        # companion-form Lyapunov. This is the natural parameterization anyway.
        # For our purposes here, we use (A_list, Sigma_eps) as the underlying parameters.
        return None  # not used in this implementation; use param-via-Sigma_eps instead

    # Use (A_list, Sigma_eps) parameterization
    sigma2_base, _ = compute_sigma2_C_from_params(A_list, Sigma_eps_for_Sigma_f, K, K_LOC, K_REM, p, N_total)

    # Grad w.r.t. each A_h entry
    grad_A_list = []
    for h_idx in range(p):
        grad_A_h = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                A_p = [A.copy() for A in A_list]
                A_p[h_idx][i, j] += eps
                A_m = [A.copy() for A in A_list]
                A_m[h_idx][i, j] -= eps
                sp, _ = compute_sigma2_C_from_params(A_p, Sigma_eps_for_Sigma_f, K, K_LOC, K_REM, p, N_total)
                sm, _ = compute_sigma2_C_from_params(A_m, Sigma_eps_for_Sigma_f, K, K_LOC, K_REM, p, N_total)
                grad_A_h[i, j] = (sp - sm) / (2 * eps)
        grad_A_list.append(grad_A_h)

    # Grad w.r.t. each unique Sigma_eps entry (symmetric)
    grad_Sigma_eps = np.zeros((K, K))
    for i in range(K):
        Sp = Sigma_eps_for_Sigma_f.copy(); Sp[i, i] += eps
        Sm = Sigma_eps_for_Sigma_f.copy(); Sm[i, i] -= eps
        sp, _ = compute_sigma2_C_from_params(A_list, Sp, K, K_LOC, K_REM, p, N_total)
        sm, _ = compute_sigma2_C_from_params(A_list, Sm, K, K_LOC, K_REM, p, N_total)
        grad_Sigma_eps[i, i] = (sp - sm) / (2 * eps)
    for i in range(K):
        for j in range(i+1, K):
            Sp = Sigma_eps_for_Sigma_f.copy(); Sp[i, j] += eps; Sp[j, i] += eps
            Sm = Sigma_eps_for_Sigma_f.copy(); Sm[i, j] -= eps; Sm[j, i] -= eps
            sp, _ = compute_sigma2_C_from_params(A_list, Sp, K, K_LOC, K_REM, p, N_total)
            sm, _ = compute_sigma2_C_from_params(A_list, Sm, K, K_LOC, K_REM, p, N_total)
            dphi = (sp - sm) / (2 * eps)
            grad_Sigma_eps[i, j] = dphi / 2
            grad_Sigma_eps[j, i] = dphi / 2

    return grad_A_list, grad_Sigma_eps, sigma2_base


def compute_sigma2_C_from_params(A_list, Sigma_eps, K, K_LOC, K_REM, p, N_total):
    """Compute sigma^2_C^(p) from (A_list, Sigma_eps) — handles consistency via companion Lyapunov."""
    Atilde = common.build_companion_matrix(A_list, K, p)
    rho = float(np.max(np.abs(np.linalg.eigvals(Atilde))))
    if rho >= 1.0:
        return float('nan'), {'unstable': True}
    Sigma_F = common.companion_Sigma_F(Atilde, Sigma_eps, K, p)
    R_list = common.extract_R_list(Sigma_F, A_list, K, p)
    s, info = common.closed_form_sigma2_C(A_list, Sigma_eps, R_list, K_LOC, K_REM, N_total)
    info['unstable'] = False
    return s, info


def estimate_Omega(A_list, Sigma_eps, K, p, T_aux, n_rep_omega, base_seed=33000):
    """MC-estimate Omega = joint asymptotic covariance of (vec A_hat, vech Sigma_eps_hat)
    by repeated OLS-VAR(p) at T=T_aux.

    Returns Omega as a flat covariance matrix of dimension p*K^2 + K(K+1)/2,
    in the same ordering as the gradient: [vec(A_1), ..., vec(A_p), vech(Sigma_eps)].
    """
    # Compute reference (A_list, Sigma_eps) parameter vector
    def pack(A_l, S_eps):
        vec_A = np.concatenate([A.flatten('F') for A in A_l])
        vech_S = []
        for i in range(K):
            for j in range(i, K):
                vech_S.append(S_eps[i, j])
        return np.concatenate([vec_A, np.array(vech_S)])

    estimates = []
    for r in range(n_rep_omega):
        f = common.simulate_VAR_p(A_list, Sigma_eps, T_aux, n_burn=500, seed=base_seed + r)
        A_hat, S_eps_hat = common.ols_var_p(f, p)
        estimates.append(pack(A_hat, S_eps_hat))

    estimates = np.array(estimates)  # (n_rep_omega, dim_param)
    # Centered, scaled by T_aux to get T_aux * Cov(theta_hat - theta)
    centered = estimates - estimates.mean(axis=0, keepdims=True)
    Cov_emp = centered.T @ centered / (n_rep_omega - 1)
    Omega_hat = T_aux * Cov_emp
    return Omega_hat


def main():
    print("=" * 100)
    print(" E7 — sampling theory for sigma^2_C^(p=2): consistency, V, coverage")
    print("=" * 100)

    K_LOC, K_REM, p = 4, 2, 2
    K = K_LOC + K_REM
    N_total = K_REM * 2

    A_list, Sigma_eps, Sigma_F, R_list, Atilde, info = common.build_VAR_p_DGP(
        K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.3, seed=70001, p=p)
    sigma2_true, _ = common.closed_form_sigma2_C(A_list, Sigma_eps, R_list, K_LOC, K_REM, N_total)
    print(f"  DGP: K={K}, K_LOC={K_LOC}, K_REM={K_REM}, p={p}, sigma_cross=0.3, rho={info['rho_Atilde']:.4f}")
    print(f"  sigma^2_C^(p=2) true value: {sigma2_true:.10f}")

    # ---------- V1 + V2: consistency sweep ----------
    print(f"\n[V1+V2] consistency sweep across T (n_rep=100 per T)")
    T_values = [500, 1000, 2000, 5000, 10000, 20000]
    rows = consistency_sweep(A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total, sigma2_true,
                              T_values, n_rep=100, base_seed=20000)

    # ---------- V3: asymptotic V via delta method ----------
    print(f"\n[V3] asymptotic V via delta method")
    print(f"  Computing finite-diff gradient of phi^(p) ...")
    t0 = time.time()
    grad_A_list, grad_Sigma_eps, sigma2_check = finite_diff_grad_phi_p(
        A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total, eps=1e-6)
    fd_elapsed = time.time() - t0
    print(f"  finite-diff completed in {fd_elapsed:.1f}s; sigma^2_C check = {sigma2_check:.10f}")
    # Pack gradient: [vec A_1, vec A_2, ..., vec A_p, vech Sigma_eps]
    grad_vec_A = np.concatenate([g.flatten('F') for g in grad_A_list])
    grad_vech_S = []
    for i in range(K):
        for j in range(i, K):
            if i == j:
                grad_vech_S.append(grad_Sigma_eps[i, i])
            else:
                # off-diagonal in vech: the directional-derivative coefficient is 2*grad
                grad_vech_S.append(2 * grad_Sigma_eps[i, j])
    grad_vech_S = np.array(grad_vech_S)
    grad_phi = np.concatenate([grad_vec_A, grad_vech_S])
    grad_norm = float(np.linalg.norm(grad_phi))
    print(f"  ||grad phi^(p)||_2 = {grad_norm:.4f}  (dim = {len(grad_phi)})")

    # Estimate Omega via MC
    T_aux, n_rep_omega = 10000, 600
    print(f"  Estimating Omega via MC at T={T_aux}, n_rep={n_rep_omega} ...")
    t0 = time.time()
    Omega_hat = estimate_Omega(A_list, Sigma_eps, K, p, T_aux, n_rep_omega, base_seed=33000)
    Omega_elapsed = time.time() - t0
    print(f"  Omega MC completed in {Omega_elapsed:.1f}s; ||Omega||_F = {np.linalg.norm(Omega_hat, 'fro'):.4f}")

    V_analytic = float(grad_phi @ Omega_hat @ grad_phi)
    print(f"\n  V_analytic = (grad phi)^T Omega (grad phi) = {V_analytic:.6f}")

    # Compare V_analytic to MC empirical T*Var
    print(f"\n  Comparison to consistency-sweep T*Var(sigma_hat):")
    print(f"  {'T':>8} {'MC mean':>14} {'T*MC Var':>14} {'V_analytic':>14} {'rel err':>10}")
    for row in rows:
        rel_err_V = abs(row['T_times_Var'] - V_analytic) / V_analytic * 100
        print(f"  {row['T']:>8} {row['MC_mean']:>14.6f} {row['T_times_Var']:>14.4f} "
              f"{V_analytic:>14.4f} {rel_err_V:>9.2f}%")

    # ---------- V4: CI coverage at T=2000 ----------
    print(f"\n[V4] CI coverage at T=2000, n_rep=500 (nominal 95%)")
    rows_covg = []
    n_in_CI_95 = 0
    n_total = 500
    t0 = time.time()
    for r in range(n_total):
        f = common.simulate_VAR_p(A_list, Sigma_eps, T=2000, n_burn=500, seed=80000 + r)
        A_hat, S_eps_hat = common.ols_var_p(f, p)
        s, inf = common.plug_in_sigma2_C(A_hat, S_eps_hat, K_LOC, K_REM, K, p, N_total)
        if inf.get('unstable', False):
            continue
        se = np.sqrt(V_analytic / 2000)
        lo, hi = s - 1.96 * se, s + 1.96 * se
        if lo <= sigma2_true <= hi:
            n_in_CI_95 += 1
        rows_covg.append((s, lo, hi))
    covg_pct = n_in_CI_95 / len(rows_covg) * 100
    print(f"  CI95 coverage: {n_in_CI_95}/{len(rows_covg)} = {covg_pct:.1f}% "
          f"({'PASS' if 92 <= covg_pct <= 98 else 'CHECK'} at [92, 98])")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ---------- SUMMARY ----------
    print("\n" + "=" * 100)
    print(" E7 SUMMARY")
    print("=" * 100)
    print(f"  Consistency: rel err at T=20000 = {rows[-1]['rel_err_pct']:+.3f}%  "
          f"({'PASS' if abs(rows[-1]['rel_err_pct']) < 2 else 'CHECK'} at < 2%)")
    sqrtT_SDs = [r['sqrt_T_times_SD'] for r in rows[2:]]  # T >= 2000
    ratio = max(sqrtT_SDs) / min(sqrtT_SDs)
    print(f"  sqrt(T)-rate: sqrt(T)*SD max/min at T>=2000 = {ratio:.3f}  "
          f"({'PASS' if ratio < 1.2 else 'CHECK'} at < 1.2)")
    rel_err_V_T20000 = abs(rows[-1]['T_times_Var'] - V_analytic) / V_analytic * 100
    print(f"  V_analytic vs T*MC Var at T=20000: {rel_err_V_T20000:.2f}% "
          f"({'PASS' if rel_err_V_T20000 < 15 else 'CHECK'} at < 15%)")
    print(f"  CI95 coverage at T=2000: {covg_pct:.1f}% "
          f"({'PASS' if 92 <= covg_pct <= 98 else 'CHECK'} at [92, 98])")

    out_path = HERE / 'E7_sampling_theory_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'sigma2_C_true': sigma2_true,
            'V_analytic': V_analytic,
            'grad_norm': grad_norm,
            'consistency_rows': rows,
            'CI95_coverage_at_T2000_pct': covg_pct,
            'CI95_n_in_CI': n_in_CI_95,
            'CI95_n_total': len(rows_covg),
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
