"""W7 — Near-degeneracy analysis of σ̂²_C: §5.4 of plan.

Per `option_alpha_theoretical_plan.md` §5.4 + Deliverable 2 §7.

Question
========

How does the asymptotic variance V from Theorem 4 inflate as
λ_min((Σ_f)_REM) → 0?

Theoretical prediction
======================

From W5 (∇A), (∇Σ):
  Π_REM := E_REM (Σ_f)_REM^{−1} E_REM^T

As λ_min((Σ_f)_REM) → 0, the smallest eigenvalue of (Σ_f)_REM^{−1} blows up by 1/λ_min,
giving:
  - ∂φ/∂A   ∼ 1/λ_min  (linear divergence; from one Π_REM factor)
  - ∂φ/∂Σ_f ∼ 1/λ_min² (quadratic divergence; from two Π_REM factors in the bilinear term)

Hence ‖∇φ‖² ∼ 1/λ_min^4 (dominated by ∂φ/∂Σ_f), and

  V = ∇φ^T Ω ∇φ  ∼  C / λ_min^d  with  d = 4 (generically)

assuming Ω stays bounded away from singularity (full Σ_f remains PD; only the REM block is
near-singular).

NOTE: the actual exponent d depends on the alignment of ∂φ/∂Σ_f with Ω_ff in the small-
eigenvalue direction. Generic alignment → d=4. Special configurations can give d ∈ {2, 3, 4}.

Threshold rule for practitioners
================================

Required T for relative SE ≤ ε:
  SE(σ̂²_C) / σ²_C ≈ √(V/T) / σ²_C ∼ 1 / (σ²_C λ_min² √T)

Setting this < ε:
  T · λ_min^4 ≥ τ where τ = 1 / (ε σ²_C)²

For a target ε = 0.10 (10% relative SE), need T · λ_min^4 ≥ 100/σ²_C² (rough order).

Verification (Deliverable 3 v2 D6 design)
==========================================

D6 grid: λ_min((Σ_f)_REM) ∈ {1.0, 0.3, 0.1, 0.03, 0.01, 0.003} (6 values per Fix 4).
Acceptance:
  C6.1: regression slope of log(MC SD) on log(λ_min) within ±20% of theoretical -d/2 = -2.
  C6.1.bis: log-log linearity R² > 0.9 (D6 fixes the structural-form check).
  C6.2: threshold rule consistent with PASS/FAIL of σ̂²_C bias acceptance C1.1.

Outputs: 60_phase_D/W7_near_degeneracy_verification_results.json + run.log
"""
from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def make_well_conditioned_A_LOC_sigma_f(K, K_LOC1, K_LOC2, K_REM, sigma_cross, seed):
    """Return UNSCALED A_base and the LOC + LOC,REM blocks of Σ_f.

    A is returned at base scale; the caller scales A adaptively at each λ_min
    to ensure Σ_ε = Σ_f - A Σ_f A^T is PD.
    """
    rng = np.random.default_rng(seed)
    K_LOC = K_LOC1 + K_LOC2
    assert K_LOC + K_REM == K

    def make_block_A(K_b, rng):
        Q1 = np.linalg.qr(rng.standard_normal((K_b, K_b)))[0]
        Q2 = np.linalg.qr(rng.standard_normal((K_b, K_b)))[0]
        diag = rng.uniform(0.4, 0.8, size=K_b)
        return Q1 @ np.diag(diag) @ Q2

    A_base = np.zeros((K, K))
    A_base[0:K_LOC1, 0:K_LOC1] = make_block_A(K_LOC1, rng)
    A_base[K_LOC1:K_LOC, K_LOC1:K_LOC] = make_block_A(K_LOC2, rng)
    A_base[K_LOC:, K_LOC:] = make_block_A(K_REM, rng)
    A_base[K_LOC:, 0:K_LOC] = rng.standard_normal((K_REM, K_LOC)) * sigma_cross
    A_base[0:K_LOC, K_LOC:] = rng.standard_normal((K_LOC, K_REM)) * sigma_cross * 0.5

    # Use Σ_f_LOC = c · I (isotropic, large) to stabilize full Σ_f
    Sigma_f_LOC = 5.0 * np.eye(K_LOC)
    # LOC,REM block of Σ_f: small (so cross-correlation is mild)
    Sigma_f_LOC_REM = rng.standard_normal((K_LOC, K_REM)) * 0.05

    # REM eigenbasis (random orthogonal)
    U_REM = np.linalg.qr(rng.standard_normal((K_REM, K_REM)))[0]

    return A_base, Sigma_f_LOC, Sigma_f_LOC_REM, U_REM


def scale_A_for_PD_Sigma_eps(A_base, Sigma_f, target_min_eig=0.05):
    """Scale A_base so that Σ_ε = Σ_f − α² · A_base Σ_f A_base^T has min eigenvalue ≥ target_min_eig.

    Bisects on α ∈ (0, 1] to find largest such α.
    """
    alpha_lo, alpha_hi = 0.0, 1.0
    # Initial check at α=1: if PD, no scaling needed.
    Sigma_eps_test = Sigma_f - A_base @ Sigma_f @ A_base.T
    Sigma_eps_test = (Sigma_eps_test + Sigma_eps_test.T) / 2
    if np.min(np.linalg.eigvalsh(Sigma_eps_test)) >= target_min_eig:
        return A_base, np.min(np.linalg.eigvalsh(Sigma_eps_test))
    # Bisect on α
    for _ in range(60):
        alpha_mid = (alpha_lo + alpha_hi) / 2
        A_test = alpha_mid * A_base
        Sigma_eps_test = Sigma_f - A_test @ Sigma_f @ A_test.T
        Sigma_eps_test = (Sigma_eps_test + Sigma_eps_test.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_eps_test))
        if min_eig >= target_min_eig:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
    A_final = alpha_lo * A_base
    Sigma_eps_final = Sigma_f - A_final @ Sigma_f @ A_final.T
    Sigma_eps_final = (Sigma_eps_final + Sigma_eps_final.T) / 2
    return A_final, np.min(np.linalg.eigvalsh(Sigma_eps_final))


def construct_full_Sigma_f(Sigma_f_LOC, Sigma_f_LOC_REM, U_REM, K_REM, lambda_min, lambdas_others):
    """Assemble full Σ_f with engineered (Σ_f)_REM = U_REM diag(λ_min, λ2, λ3, ...) U_REM^T."""
    K_LOC = Sigma_f_LOC.shape[0]
    K = K_LOC + K_REM
    diag = np.array([lambda_min] + list(lambdas_others))
    Sigma_REM = U_REM @ np.diag(diag) @ U_REM.T
    Sigma_REM = (Sigma_REM + Sigma_REM.T) / 2

    Sigma_f = np.zeros((K, K))
    Sigma_f[:K_LOC, :K_LOC] = Sigma_f_LOC
    Sigma_f[:K_LOC, K_LOC:] = Sigma_f_LOC_REM
    Sigma_f[K_LOC:, :K_LOC] = Sigma_f_LOC_REM.T
    Sigma_f[K_LOC:, K_LOC:] = Sigma_REM
    Sigma_f = (Sigma_f + Sigma_f.T) / 2

    return Sigma_f


def derive_Sigma_eps(A, Sigma_f):
    """Σ_ε = Σ_f − A Σ_f A^T (Lyapunov rearrangement). Verify PD."""
    Sigma_eps = Sigma_f - A @ Sigma_f @ A.T
    Sigma_eps = (Sigma_eps + Sigma_eps.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma_eps)
    if np.min(eigvals) < 1e-10:
        return None, np.min(eigvals)
    return Sigma_eps, np.min(eigvals)


def simulate_VAR1(A, Sigma_eps, T, T_burn, rng):
    K = A.shape[0]
    L = np.linalg.cholesky(Sigma_eps + 1e-12 * np.eye(K))
    f = np.zeros((T + T_burn + 1, K))
    f[0] = rng.standard_normal(K) * 0.1
    for t in range(T + T_burn):
        f[t + 1] = A @ f[t] + L @ rng.standard_normal(K)
    return f[T_burn + 1:]


def estimate_OLS_VAR(f):
    T, K = f.shape
    F_pre = f[:-1]
    F_post = f[1:]
    A_hat = np.linalg.lstsq(F_pre, F_post, rcond=None)[0].T
    Sigma_f_hat = f.T @ f / T
    Sigma_f_hat = (Sigma_f_hat + Sigma_f_hat.T) / 2
    return A_hat, Sigma_f_hat


def sigma2_C_phi(A, Sigma_f, block_slices, rem_idx, N_total):
    s, e = block_slices[rem_idx]
    K_REM = e - s
    A_Sigma = A @ Sigma_f
    A_Sigma_AT = A_Sigma @ A.T
    Sigma_AT = Sigma_f @ A.T
    Sigma_REM = Sigma_f[s:e, s:e]
    Sigma_REM_inv = np.linalg.inv(Sigma_REM + 1e-12 * np.eye(K_REM))
    DeltaVar = (A_Sigma_AT[s:e, s:e]
                - A_Sigma[s:e, s:e] @ Sigma_REM_inv @ Sigma_AT[s:e, s:e])
    DeltaVar = (DeltaVar + DeltaVar.T) / 2
    return float(np.trace(DeltaVar)) / N_total


def empirical_T_Var_sigma2_C(A, Sigma_eps, Sigma_f, block_slices, N_total, T, n_reps, base_seed):
    """MC-estimate T · Var(σ̂²_C) at given (A, Σ_ε)."""
    estimates = np.zeros(n_reps)
    for rep in range(n_reps):
        rng = np.random.default_rng(base_seed + rep)
        f = simulate_VAR1(A, Sigma_eps, T, T_burn=200, rng=rng)
        A_hat, Sigma_f_hat = estimate_OLS_VAR(f)
        estimates[rep] = sigma2_C_phi(A_hat, Sigma_f_hat, block_slices, 2, N_total)
    mc_mean = float(np.mean(estimates))
    mc_sd = float(np.std(estimates, ddof=1))
    return {
        'T': T, 'n_reps': n_reps,
        'mc_mean': mc_mean, 'mc_sd': mc_sd, 'T_mc_var': T * mc_sd ** 2,
    }


def main():
    print("=" * 88)
    print(" W7 near-degeneracy verification — V's behavior as λ_min((Σ_f)_REM) → 0")
    print("=" * 88)
    t0 = time.time()

    # DGP setup
    K, K_LOC1, K_LOC2, K_REM = 8, 2, 2, 4
    sigma_cross = 0.2  # smaller than W5 to keep Σ_ε PD across λ_min sweep
    A_base, Sigma_f_LOC, Sigma_f_LOC_REM, U_REM = make_well_conditioned_A_LOC_sigma_f(
        K, K_LOC1, K_LOC2, K_REM, sigma_cross, seed=2026)
    K_LOC = K_LOC1 + K_LOC2
    block_slices = [(0, K_LOC1), (K_LOC1, K_LOC), (K_LOC, K)]
    N_total = K * 2
    print(f"  DGP: K={K}, blocks=({K_LOC1}, {K_LOC2}, {K_REM}), σ_cross={sigma_cross}, N_total={N_total}")
    print(f"  ρ(A_base) = {np.max(np.abs(np.linalg.eigvals(A_base))):.4f}")

    # Per Deliverable 3 v2 D6 grid (6 λ_min values for adequate slope-SE)
    lambda_min_grid = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
    lambdas_others = [1.0, 1.0, 1.0]  # other 3 eigenvalues of (Σ_f)_REM fixed at 1

    T_for_V = 2000
    n_reps = 300
    base_seed = 7001

    results = []
    for lambda_min in lambda_min_grid:
        Sigma_f = construct_full_Sigma_f(Sigma_f_LOC, Sigma_f_LOC_REM, U_REM, K_REM,
                                          lambda_min, lambdas_others)
        Sigma_f_min_eig = float(np.min(np.linalg.eigvalsh(Sigma_f)))
        # Adaptively scale A_base so Σ_ε is PD
        A, sigma_eps_min_eig = scale_A_for_PD_Sigma_eps(A_base, Sigma_f, target_min_eig=0.05)
        alpha_used = np.linalg.norm(A) / max(np.linalg.norm(A_base), 1e-12)
        Sigma_eps = Sigma_f - A @ Sigma_f @ A.T
        Sigma_eps = (Sigma_eps + Sigma_eps.T) / 2
        print(f"\n  --- λ_min((Σ_f)_REM) = {lambda_min} ---")
        print(f"    λ_min(Σ_f) (full) = {Sigma_f_min_eig:.6f}")
        print(f"    A scaling α       = {alpha_used:.4f}  (so Σ_ε is PD)")
        print(f"    λ_min(Σ_ε)         = {sigma_eps_min_eig:.6f}")
        if sigma_eps_min_eig < 0.01:
            print(f"    Σ_ε too close to singular; skipping this λ_min")
            results.append({
                'lambda_min': lambda_min,
                'Sigma_eps_PD': False,
                'skip_reason': 'Σ_ε near-singular even after scaling',
            })
            continue

        # Theoretical σ²_C from Theorem 1
        sigma2_C_true = sigma2_C_phi(A, Sigma_f, block_slices, 2, N_total)
        # MC empirical T·Var(σ̂²_C)
        emp = empirical_T_Var_sigma2_C(A, Sigma_eps, Sigma_f, block_slices, N_total,
                                         T_for_V, n_reps, base_seed=base_seed + 1000 * int(lambda_min * 1000))
        rel_err_mean = abs(emp['mc_mean'] - sigma2_C_true) / max(abs(sigma2_C_true), 1e-12)
        print(f"    σ²_C (true) = {sigma2_C_true:.6f}")
        print(f"    MC mean σ̂²_C = {emp['mc_mean']:.6f}  (rel err {rel_err_mean:.2%})")
        print(f"    MC SD σ̂²_C = {emp['mc_sd']:.6f}")
        print(f"    T·MC Var = {emp['T_mc_var']:.4f}")
        results.append({
            'lambda_min': lambda_min,
            'Sigma_eps_PD': True,
            'Sigma_f_min_eig': Sigma_f_min_eig,
            'Sigma_eps_min_eig': sigma_eps_min_eig,
            'A_scaling_alpha': float(alpha_used),
            'sigma2_C_true': sigma2_C_true,
            'mc_mean': emp['mc_mean'],
            'mc_sd': emp['mc_sd'],
            'T_mc_var': emp['T_mc_var'],
            'rel_err_mean': rel_err_mean,
            'T': T_for_V, 'n_reps': n_reps,
        })

    # Regression: log(MC SD) vs log(λ_min)
    valid = [r for r in results if r.get('Sigma_eps_PD', False)]
    if len(valid) >= 3:
        log_lam = np.log10(np.array([r['lambda_min'] for r in valid]))
        log_SD = np.log10(np.array([r['mc_sd'] for r in valid]))
        # Linear regression
        slope, intercept = np.polyfit(log_lam, log_SD, 1)
        # Compute R²
        SD_pred = slope * log_lam + intercept
        ss_res = np.sum((log_SD - SD_pred) ** 2)
        ss_tot = np.sum((log_SD - np.mean(log_SD)) ** 2)
        R_sq = 1 - ss_res / ss_tot
        print(f"\n  Log-log regression of MC SD vs λ_min ({len(valid)} valid points):")
        print(f"    slope = {slope:.4f}  (theoretical -d/2 = -2.0 if d = 4)")
        print(f"    intercept = {intercept:.4f}")
        print(f"    R² = {R_sq:.4f}")

        # Implied d (V ~ 1/λ_min^d so SD ~ 1/λ_min^{d/2}, slope of log SD vs log λ_min is -d/2)
        d_implied = -2 * slope
        print(f"    Implied d = {d_implied:.2f}  (theoretical 4 if dominant ∂φ/∂Σ_f drives V)")

        # D6 acceptance per Deliverable 3 v2:
        # C6.1: slope within ±20% of theoretical -2 (i.e., slope ∈ [-2.4, -1.6])
        # C6.1.bis: R² > 0.9
        c6_1_pass = -2.4 <= slope <= -1.6
        c6_1_bis_pass = R_sq > 0.9
        print(f"\n  D6 acceptance per Deliverable 3 v2:")
        print(f"    C6.1 (slope in [-2.4, -1.6]):  {'PASS' if c6_1_pass else 'FAIL'}")
        print(f"    C6.1.bis (R² > 0.9 linearity): {'PASS' if c6_1_bis_pass else 'FAIL'}")

        # Threshold rule check
        # T · λ_min^d ≥ τ. With d=4 (theoretical) and τ chosen to give ≤ 5% rel err on σ²_C mean:
        # at T=2000, threshold λ_min^4 ≥ τ/2000. Empirically rel_err < 5% requires T·λ_min^d > some τ.
        rel_errs = [r['rel_err_mean'] for r in valid]
        print(f"\n  Threshold rule (T · λ_min^d ≥ τ for rel err < 5%):")
        for r in valid:
            T_lambda_d = T_for_V * r['lambda_min'] ** d_implied
            print(f"    λ_min={r['lambda_min']:.4f}: T·λ^d = {T_lambda_d:>12.4f}, rel err = {r['rel_err_mean']:>7.2%}")

        # Determine threshold τ empirically: smallest T·λ^d at which rel err < 5%
        passing = [r for r in valid if r['rel_err_mean'] < 0.05]
        if passing:
            tau_empirical = min(T_for_V * r['lambda_min'] ** d_implied for r in passing)
            print(f"    Empirical τ (smallest T·λ^d with rel err < 5%): {tau_empirical:.4f}")
        else:
            tau_empirical = None
            print(f"    No tested DGP achieves rel err < 5%; threshold τ undeterminable from this grid")

        overall = c6_1_pass and c6_1_bis_pass
    else:
        slope, intercept, R_sq, d_implied = None, None, None, None
        c6_1_pass = c6_1_bis_pass = False
        tau_empirical = None
        overall = False
        print(f"\n  Insufficient valid points for regression: {len(valid)} < 3")

    # Summary
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    print(f"  Valid λ_min points: {len(valid)} / {len(lambda_min_grid)}")
    if slope is not None:
        print(f"  Implied divergence exponent d ≈ {d_implied:.2f} (theoretical 4)")
        print(f"  Log-log linearity R² = {R_sq:.4f}")
        print(f"  C6.1 PASS: {c6_1_pass}")
        print(f"  C6.1.bis PASS: {c6_1_bis_pass}")
    print(f"\n  OVERALL VERDICT: {'PASS' if overall else 'CHECK / partial — interpret per W7 §3'}")
    print(f"  HS-W7.1 (V degeneracy order not tractable): {'NOT FIRED' if overall else 'CHECK'}")

    out = {
        'DGP': {
            'K': K, 'K_LOC1': K_LOC1, 'K_LOC2': K_LOC2, 'K_REM': K_REM,
            'sigma_cross': sigma_cross,
            'rho_A': float(np.max(np.abs(np.linalg.eigvals(A)))),
            'lambda_min_grid': lambda_min_grid,
            'lambdas_others': lambdas_others,
        },
        'per_lambda_min_results': results,
        'regression': {
            'slope': float(slope) if slope is not None else None,
            'intercept': float(intercept) if intercept is not None else None,
            'R_squared': float(R_sq) if R_sq is not None else None,
            'd_implied': float(d_implied) if d_implied is not None else None,
        },
        'D6_acceptance': {
            'C6_1_slope_in_band': bool(c6_1_pass),
            'C6_1_bis_R_sq_gt_0_9': bool(c6_1_bis_pass),
        },
        'threshold_tau_empirical': tau_empirical,
        'overall_pass': bool(overall),
        'runtime_seconds': time.time() - t0,
    }
    out_path = HERE / 'W7_near_degeneracy_verification_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
