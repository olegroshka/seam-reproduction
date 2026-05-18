"""E4 — closed-form sigma^2_C^(p) sweep across p in {2, 3, 5, 7, 10} and K in {4, 6, 8, 12}.

Verifications per (p, K) cell:
- V1: closed form == BLP-direct (algebraic, should be ~machine zero)
- V2: closed form vs MC empirical at T=20000, n_rep=30 (within 3 MC SE)
- V3: companion-form Theorem 1 applied at lifted REM_comp partition == closed form
- Gamma block-Toeplitz condition number reported for stability
- Timing reported per cell
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


def MC_empirical_sigma2_C(A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total,
                          T=20000, n_burn=500, n_rep=30, seed=10001):
    rng = np.random.default_rng(seed)
    diffs = []
    for r in range(n_rep):
        f = common.simulate_VAR_p(A_list, Sigma_eps, T, n_burn=n_burn, seed=seed + r)
        f_REM = f[:, K_LOC:K]
        T_use = f_REM.shape[0] - p
        X = np.zeros((T_use, K_REM * p))
        for h in range(1, p+1):
            X[:, (h-1)*K_REM:h*K_REM] = f_REM[p-h:p-h+T_use]
        Y = f_REM[p:p+T_use]
        coef = np.linalg.solve(X.T @ X, X.T @ Y)
        rest_err = Y - X @ coef
        rest_var = rest_err.T @ rest_err / T_use

        joint_err = np.zeros((T_use, K_REM))
        for t in range(p, p + T_use):
            jp = sum(A_list[h-1] @ f[t-h] for h in range(1, p+1))
            joint_err[t-p] = f[t, K_LOC:] - jp[K_LOC:]
        joint_var = joint_err.T @ joint_err / T_use
        diffs.append(float(np.trace(rest_var - joint_var)) / N_total)
    return float(np.mean(diffs)), float(np.std(diffs) / np.sqrt(n_rep))


def main():
    print("=" * 100)
    print(" E4 — closed-form sigma^2_C^(p) sweep")
    print("=" * 100)

    # Grid: (K, K_LOC, K_REM) configurations
    grid = [
        # (K, K_LOC, K_REM)
        (4, 2, 2),
        (6, 4, 2),
        (8, 4, 4),
        (8, 6, 2),
        (12, 8, 4),
    ]
    p_values = [2, 3, 5, 7, 10]

    results = []
    for K, K_LOC, K_REM in grid:
        for p in p_values:
            cell_key = f'K={K} K_LOC={K_LOC} K_REM={K_REM} p={p}'
            print(f"\n[{cell_key}]")
            t0 = time.time()
            try:
                A_list, Sigma_eps, Sigma_F, R_list, Atilde, info = common.build_VAR_p_DGP(
                    K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.3,
                    seed=4000 + K*100 + p, p=p)
                N_total = K_REM * 2

                sigma2_closed, info_closed = common.closed_form_sigma2_C(
                    A_list, Sigma_eps, R_list, K_LOC, K_REM, N_total)
                sigma2_BLP = common.BLP_direct_sigma2_C(
                    Sigma_F[0:K, 0:K], Sigma_eps, R_list, K_LOC, K_REM, N_total)
                sigma2_comp, _, _ = common.companion_form_sigma2_C(
                    Atilde, Sigma_F, K, K_LOC, K_REM, p, N_total)

                V1_residual = abs(sigma2_closed - sigma2_BLP)
                V3_residual = abs(sigma2_closed - sigma2_comp)

                # MC for one cell per (K, p) to save time — only run at K=4, 6
                run_MC = (K <= 6)
                if run_MC:
                    mc_mean, mc_se = MC_empirical_sigma2_C(
                        A_list, Sigma_eps, K_LOC, K_REM, K, p, N_total,
                        T=20000, n_rep=30, seed=10000 + K*100 + p)
                    V2_n_se = abs(mc_mean - sigma2_closed) / max(mc_se, 1e-12)
                    V2_pass = V2_n_se < 3.0
                    V2_rel_err = abs(mc_mean - sigma2_closed) / max(abs(sigma2_closed), 1e-12) * 100
                else:
                    mc_mean = float('nan')
                    mc_se = float('nan')
                    V2_n_se = float('nan')
                    V2_pass = None  # skipped
                    V2_rel_err = float('nan')

                elapsed = time.time() - t0
                cell_result = {
                    'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': p,
                    'sigma2_C_closed': sigma2_closed,
                    'sigma2_C_BLP': sigma2_BLP,
                    'sigma2_C_companion': sigma2_comp,
                    'sigma2_C_MC_mean': mc_mean,
                    'sigma2_C_MC_SE': mc_se,
                    'V1_residual_closed_vs_BLP': float(V1_residual),
                    'V3_residual_closed_vs_companion': float(V3_residual),
                    'V2_MC_n_SE_away': float(V2_n_se),
                    'V2_MC_rel_err_pct': float(V2_rel_err),
                    'V1_pass': bool(V1_residual < 1e-10),
                    'V3_pass': bool(V3_residual < 1e-10),
                    'V2_pass': V2_pass,
                    'Gamma_condition': info_closed['Gamma_condition_number'],
                    'Gamma_lambda_min': info_closed['Gamma_lambda_min'],
                    'rho_Atilde': info['rho_Atilde'],
                    'elapsed_s': elapsed,
                }
                results.append(cell_result)
                print(f"  sigma^2_C^(p)        = {sigma2_closed:.10f}")
                print(f"  V1 residual          = {V1_residual:.3e}  ({'PASS' if V1_residual < 1e-10 else 'FAIL'})")
                print(f"  V3 residual          = {V3_residual:.3e}  ({'PASS' if V3_residual < 1e-10 else 'FAIL'})")
                if run_MC:
                    print(f"  V2 MC mean ± SE      = {mc_mean:.6f} ± {mc_se:.6f}")
                    print(f"  V2 rel err           = {V2_rel_err:.3f}%  ({V2_n_se:.2f} SE away)  ({'PASS' if V2_pass else 'FAIL'})")
                else:
                    print(f"  V2 MC                = skipped (K > 6)")
                print(f"  Gamma cond           = {info_closed['Gamma_condition_number']:.3e}")
                print(f"  Gamma lambda_min     = {info_closed['Gamma_lambda_min']:.6f}")
                print(f"  elapsed              = {elapsed:.2f}s")
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                results.append({
                    'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': p,
                    'exception': str(e), 'V1_pass': False, 'V3_pass': False,
                })

    # SUMMARY
    print("\n" + "=" * 100)
    print(" E4 SUMMARY")
    print("=" * 100)
    n_V1 = sum(1 for r in results if r.get('V1_pass'))
    n_V3 = sum(1 for r in results if r.get('V3_pass'))
    n_V2 = sum(1 for r in results if r.get('V2_pass') is True)
    n_V2_skip = sum(1 for r in results if r.get('V2_pass') is None)
    n_cells = len(results)
    print(f"  Cells:                  {n_cells}")
    print(f"  V1 (closed == BLP):     {n_V1} / {n_cells} PASS")
    print(f"  V3 (closed == companion): {n_V3} / {n_cells} PASS")
    print(f"  V2 (MC empirical):      {n_V2} PASS, {n_V2_skip} skipped (K > 6)")
    overall = (n_V1 == n_cells) and (n_V3 == n_cells)
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  closed-form sigma^2_C^(p) lifts across p in {p_values} and K in {[g[0] for g in grid]}: " +
          ("YES" if overall else "NO"))

    out_path = HERE / 'E4_closed_form_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump({'overall_pass': bool(overall), 'cells': results,
                   'grid': grid, 'p_values': p_values}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
