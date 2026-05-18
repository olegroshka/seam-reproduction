"""E10 — near-degeneracy V ~ 1/lambda_min(Gamma)^4 conjecture at VAR(p=2).

The paper's §5.4 / App F derives V ~ 1/lambda_min((Sigma_f)_REM)^4 for VAR(1) via
the two-factor (Sigma_f)_REM^{-1} dependence in (∇Σ_f φ). For VAR(p), the natural
analog is V ~ 1/lambda_min(Gamma)^4 where Gamma is the block-Toeplitz of REM
autocovariances.

Verification path (parametric, like paper's V3.a): construct Sigma_F at the companion
level such that the eigenvalues of (Sigma_F)_REM_comp are controlled, evaluate V via
the delta-method-style formula, and observe scaling.

Two paths:
  Path A: scale lambda_min((Sigma_f)_REM) directly via constructed Sigma_f (not
          Lyapunov-consistent — same approach as paper V3.a).
  Path B: study the natural extension of V scaling at the block-Toeplitz Gamma level.

We use Path A for simplicity: build a VAR(2) and parametrically scale lambda_min
of (Sigma_f)_REM by perturbing the REM diagonal of Sigma_eps.
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


def main():
    print("=" * 100)
    print(" E10 — near-degeneracy V scaling at VAR(p=2), parametric path")
    print("=" * 100)

    K_LOC, K_REM, p = 4, 2, 2
    K = K_LOC + K_REM
    N_total = K_REM * 2

    A_list, Sigma_eps_base, Sigma_F_base, R_list_base, Atilde, info = common.build_VAR_p_DGP(
        K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.3, seed=100001, p=p)
    print(f"  Base DGP: K={K}, K_LOC={K_LOC}, K_REM={K_REM}, p={p}, rho={info['rho_Atilde']:.4f}")

    # Compute V at base DGP via finite-diff grad + MC Omega
    # (For E10 we just compute V from analytical Omega blocks — use placeholder Omega = I for scaling test)
    # Actually we'll use the closed-form sigma^2_C / lambda_min relation directly to verify scaling.
    # V depends on (grad phi)^T Omega (grad phi); we'll observe the magnitude of (grad phi)
    # as lambda_min varies, since Omega is bounded as lambda_min -> 0 (Omega depends on Sigma_eps
    # and the DGP, not on (Sigma_f)_REM directly).
    # The argument: ||grad phi||_F ~ 1/lambda_min^2 (from two-factor inverse dependence), so
    # V ~ ||grad phi||^2 * ||Omega|| ~ 1/lambda_min^4.

    # ---------- Sweep: scale Sigma_eps_REM to control lambda_min((Sigma_f)_REM) ----------
    # Parametric: replace Sigma_eps with Sigma_eps' = Sigma_eps - mu * P_REM where P_REM
    # is the projection onto REM (subtract a small constant from REM diagonal entries).
    # We control mu to drive lambda_min((Sigma_f)_REM) -> 0.
    # Note this is NOT Lyapunov-consistent in general; for scaling test we don't require it.

    # Simpler: parametrically scale (Sigma_f)_REM directly via a perturbation of Sigma_f
    # (treat as independent parameter — paper V3.a approach).

    def compute_grad_norm_at_perturbed_Sigma_f(Sigma_f_perturbed):
        """Compute ||grad_Sigma_eps phi^(p)||_F at perturbed Sigma_f using closed-form formulas."""
        # Reconstruct R_list from perturbed Sigma_f
        # For VAR(p), R(h) = A_1 R(h-1) + ... + A_p R(h-p) with R(0) = Sigma_f
        # This is a nontrivial reconstruction without re-solving Lyapunov.
        # Simpler: use finite-diff w.r.t. Sigma_eps with the perturbed setup.

        # Since the closed form for sigma^2_C^(p) uses Sigma_f and the A_h list, we can perturb
        # Sigma_f directly (treating it as independent) and recompute R(h) for h>=1 via YW recursion.
        R_pert = [Sigma_f_perturbed]
        for h in range(1, p+1):
            R_h = sum(A_list[q-1] @ R_pert[h-q] for q in range(1, min(h, p)+1) if h-q >= 0)
            R_pert.append(R_h)
        # implied Sigma_eps_pert from Yule-Walker R(0) = sum A_q R(q)^T + Sigma_eps
        Sigma_eps_pert = Sigma_f_perturbed - sum(A_list[h-1] @ R_pert[h].T for h in range(1, p+1))
        sigma2_C_pert, info_pert = common.closed_form_sigma2_C(
            A_list, Sigma_eps_pert, R_pert, K_LOC, K_REM, N_total)
        Gamma_lambda_min = info_pert['Gamma_lambda_min']
        return sigma2_C_pert, Gamma_lambda_min, R_pert, Sigma_eps_pert

    # Eigendecompose base (Sigma_f)_REM
    Sigma_f_base = Sigma_F_base[0:K, 0:K]
    Sigma_f_REM_base = Sigma_f_base[K_LOC:, K_LOC:]
    eigvals, eigvecs = np.linalg.eigh(Sigma_f_REM_base)
    print(f"  Base (Sigma_f)_REM eigenvalues: {eigvals}")

    # Target lambda_min values
    target_lams = [10**(-k) for k in [1, 2, 3, 4]]
    print(f"\n{'target_lam_min':>16} {'gamma_lam_min':>16} {'sigma^2_C':>14} {'V_approx (||grad||^2)':>24}")
    results = []
    for tgt in target_lams:
        # Construct perturbed (Sigma_f)_REM with smallest eigenvalue = tgt
        eigvals_pert = eigvals.copy()
        eigvals_pert[0] = tgt  # smallest eigenvalue set to tgt
        Sigma_f_REM_pert = eigvecs @ np.diag(eigvals_pert) @ eigvecs.T
        Sigma_f_REM_pert = (Sigma_f_REM_pert + Sigma_f_REM_pert.T) / 2

        Sigma_f_pert = Sigma_f_base.copy()
        Sigma_f_pert[K_LOC:, K_LOC:] = Sigma_f_REM_pert

        sigma2_C_pert, gam_lam, R_pert, Sigma_eps_pert = compute_grad_norm_at_perturbed_Sigma_f(Sigma_f_pert)

        # Approximate V ~ ||grad_Sigma_f phi||_F^2: use finite-diff
        eps_fd = max(1e-6, tgt * 0.01)
        gradnorm_sq = 0.0
        for i in range(K_LOC, K):
            for j in range(K_LOC, K):
                if i > j:
                    continue
                Sp = Sigma_f_pert.copy()
                Sp[i, j] += eps_fd
                Sp[j, i] += eps_fd if i != j else 0.0
                # adjust to make symmetric for off-diag
                Sm = Sigma_f_pert.copy()
                Sm[i, j] -= eps_fd
                Sm[j, i] -= eps_fd if i != j else 0.0
                sp, _, _, _ = compute_grad_norm_at_perturbed_Sigma_f(Sp)
                sm, _, _, _ = compute_grad_norm_at_perturbed_Sigma_f(Sm)
                d = (sp - sm) / (2 * eps_fd)
                if i == j:
                    gradnorm_sq += d**2
                else:
                    gradnorm_sq += 2 * d**2  # off-diag counted twice in the symmetric form
        results.append({
            'target_lam_min': float(tgt),
            'gamma_lambda_min': float(gam_lam),
            'sigma2_C': float(sigma2_C_pert),
            'grad_phi_norm_sq': float(gradnorm_sq),
        })
        print(f"  {tgt:>16.6e} {gam_lam:>16.6e} {sigma2_C_pert:>14.6f} {gradnorm_sq:>24.6e}")

    # Fit log-log regression: log(grad_norm_sq) vs log(target_lam_min)
    # Expected slope: -4 (since V ~ ||grad||^2 ~ 1/lam^4)
    logs_lam = np.log10([r['target_lam_min'] for r in results])
    logs_gnsq = np.log10([r['grad_phi_norm_sq'] for r in results])
    slope, intercept = np.polyfit(logs_lam, logs_gnsq, 1)
    print(f"\n  Log-log regression of ||grad||^2 vs lambda_min:")
    print(f"    slope = {slope:.4f}  (theoretical -4 if V ~ 1/lam^4)")
    print(f"    intercept = {intercept:.4f}")

    out_path = HERE / 'E10_near_degeneracy_results.json'
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'slope': float(slope), 'intercept': float(intercept),
                   'theoretical_slope': -4.0}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"\n  E10 OVERALL: empirical slope {slope:.3f} vs theoretical -4 "
          f"({'PASS' if abs(slope - (-4)) < 0.5 else 'CHECK'} at |error| < 0.5)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
