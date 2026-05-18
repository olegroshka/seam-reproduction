"""E10 v2 — near-degeneracy V scaling at VAR(p=2), via direct Sigma_F parametric path.

The paper's §5.4 / App F derives V ~ 1/lambda_min((Sigma_f)_REM)^4 for VAR(1) from the
two-factor (Sigma_f)_REM^{-1} dependence in (∇Σ_f φ). For VAR(p), the natural analog
is V ~ 1/lambda_min(Gamma)^4 where Gamma is the block-Toeplitz of REM autocovariances —
equivalently, Gamma = (Sigma_F)_REM_comp at the companion-state level.

V3.a path (parametric, non-Lyapunov-consistent): treat (Atilde, Sigma_F) as separate
parameters and parametrically scale the smallest eigenvalue of (Sigma_F)_REM_comp.
Compute V proxy = ||grad_Sigma_F^(p) phi^(p)||_F^2. Expected scaling: ||grad||^2 ~ 1/lam^4.
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


def companion_form_sigma2_C_from_params(Atilde, Sigma_F, K, K_LOC, K_REM, p, N_total):
    """Pure-functional path: given (Atilde, Sigma_F) as independent inputs, compute
    sigma^2_C^(p) via the companion-form Theorem 1 application. No Lyapunov consistency
    check. This is the natural extension of the paper's V3.a parametric path.
    """
    s, _, _ = common.companion_form_sigma2_C(Atilde, Sigma_F, K, K_LOC, K_REM, p, N_total)
    return s


def main():
    print("=" * 100)
    print(" E10 v2 — near-degeneracy V scaling at VAR(p=2), parametric Sigma_F path")
    print("=" * 100)

    K_LOC, K_REM, p = 4, 2, 2
    K = K_LOC + K_REM
    Kp = K * p
    N_total = K_REM * 2

    A_list, Sigma_eps, Sigma_F_base, R_list, Atilde, info = common.build_VAR_p_DGP(
        K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.3, seed=100002, p=p)

    print(f"  Base DGP: K={K}, K_LOC={K_LOC}, K_REM={K_REM}, p={p}")
    print(f"  rho(Atilde) = {info['rho_Atilde']:.4f}")

    # Index REM_comp at companion level
    rem_comp_idx = np.concatenate([np.arange(h*K + K_LOC, h*K + K) for h in range(p)])
    # Base (Sigma_F)_REM_comp
    SF_REM_base = Sigma_F_base[np.ix_(rem_comp_idx, rem_comp_idx)]
    eigvals, eigvecs = np.linalg.eigh(SF_REM_base)
    print(f"  Base (Sigma_F)_REM_comp eigenvalues: {eigvals}")

    # Sweep over target smallest eigenvalue
    target_lams = [10**(-k) for k in [0, 1, 2, 3, 4]]

    print(f"\n{'target_lam_min':>16} {'sigma^2_C':>14} {'||grad||^2':>16} {'cond(SF_REM)':>14}")
    print("-" * 70)
    results = []
    for tgt in target_lams:
        # Construct perturbed (Sigma_F)_REM_comp with smallest eigenvalue = tgt
        eigvals_pert = eigvals.copy()
        eigvals_pert[0] = tgt
        SF_REM_pert = eigvecs @ np.diag(eigvals_pert) @ eigvecs.T
        SF_REM_pert = (SF_REM_pert + SF_REM_pert.T) / 2

        # Embed back into a perturbed Sigma_F (replace REM_comp block, keep rest)
        Sigma_F_pert = Sigma_F_base.copy()
        for i, idx_i in enumerate(rem_comp_idx):
            for j, idx_j in enumerate(rem_comp_idx):
                Sigma_F_pert[idx_i, idx_j] = SF_REM_pert[i, j]

        sigma2_pert = companion_form_sigma2_C_from_params(Atilde, Sigma_F_pert, K, K_LOC, K_REM, p, N_total)

        # Compute ||grad_Sigma_F^(p) phi^(p)||_F^2 via finite-diff
        # Only perturb entries of (Sigma_F)_REM_comp (parametric path)
        eps_fd = max(1e-6, tgt * 0.001)
        grad_REM = np.zeros((K_REM * p, K_REM * p))
        for i in range(K_REM * p):
            for j in range(i, K_REM * p):
                Sp_full = Sigma_F_pert.copy()
                Sm_full = Sigma_F_pert.copy()
                ii = rem_comp_idx[i]
                jj = rem_comp_idx[j]
                Sp_full[ii, jj] += eps_fd
                Sp_full[jj, ii] += eps_fd if i != j else 0.0
                Sm_full[ii, jj] -= eps_fd
                Sm_full[jj, ii] -= eps_fd if i != j else 0.0
                sp = companion_form_sigma2_C_from_params(Atilde, Sp_full, K, K_LOC, K_REM, p, N_total)
                sm = companion_form_sigma2_C_from_params(Atilde, Sm_full, K, K_LOC, K_REM, p, N_total)
                d = (sp - sm) / (2 * eps_fd)
                if i == j:
                    grad_REM[i, j] = d
                else:
                    # the directional derivative w.r.t. symmetric perturbation is 2 * grad_ij
                    grad_REM[i, j] = d / 2
                    grad_REM[j, i] = d / 2
        grad_norm_sq = float(np.sum(grad_REM**2) + np.sum(np.triu(grad_REM, k=1)**2))
        # (diagonal entries weighted 1, off-diagonal weighted 2 in the symmetric Frobenius inner product)

        cond_SF_REM = float(np.linalg.cond(SF_REM_pert))
        results.append({
            'target_lam_min': float(tgt),
            'sigma2_C': float(sigma2_pert),
            'grad_phi_norm_sq': float(grad_norm_sq),
            'SF_REM_condition': cond_SF_REM,
        })
        print(f"  {tgt:>16.6e} {sigma2_pert:>14.6f} {grad_norm_sq:>16.4e} {cond_SF_REM:>14.3e}")

    # Log-log regression
    logs_lam = np.log10([r['target_lam_min'] for r in results])
    logs_gnsq = np.log10([r['grad_phi_norm_sq'] for r in results])
    slope, intercept = np.polyfit(logs_lam, logs_gnsq, 1)
    print(f"\n  Log-log regression of ||grad||^2 vs lambda_min:")
    print(f"    slope = {slope:.4f}  (theoretical -4 if V ~ 1/lam^4)")
    print(f"    intercept = {intercept:.4f}")
    # Use only small-lam_min points for asymptotic slope
    logs_lam_small = logs_lam[2:]  # lam <= 1e-2
    logs_gnsq_small = logs_gnsq[2:]
    if len(logs_lam_small) >= 2:
        slope_small, _ = np.polyfit(logs_lam_small, logs_gnsq_small, 1)
        print(f"    slope (lam <= 1e-2 only) = {slope_small:.4f}  (theoretical -4)")
    else:
        slope_small = float('nan')

    overall_pass = abs(slope_small - (-4)) < 0.5
    print(f"\n  E10 v2 OVERALL: empirical slope (small lam) {slope_small:.3f} vs theoretical -4 "
          f"({'PASS' if overall_pass else 'CHECK'} at |error| < 0.5)")

    out_path = HERE / 'E10_near_degeneracy_v2_results.json'
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'slope_all': float(slope),
                   'slope_small_lam': float(slope_small),
                   'theoretical_slope': -4.0, 'overall_pass': bool(overall_pass)}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
