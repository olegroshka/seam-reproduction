"""E5 — companion-form vs direct equivalence sweep at larger p, K.

Extends E2 to a grid sweep. The companion-form Theorem 1 applied at the lifted
REM_comp partition should equal the direct VAR(p) closed form regardless of:
- singularity of Sigma_eps^companion (rank K, dim K*p — always singular for p>=2)
- block-Toeplitz Gamma conditioning at larger p

Reports: V1 residual (closed vs companion), V2 (companion top-block trace == full trace),
Schur off-block magnitude (should be machine zero), timing, Gamma condition.
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
    print(" E5 — companion-form equivalence sweep at larger p, K")
    print("=" * 100)

    grid = [
        # (K, K_LOC, K_REM)
        (4, 2, 2),
        (6, 4, 2),
        (8, 4, 4),
        (10, 6, 4),
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
                    K=K, K_LOC_list=[K_LOC], K_REM=K_REM, sigma_cross=0.25,
                    seed=5000 + K*100 + p, p=p)
                N_total = K_REM * 2

                sigma2_direct, _ = common.closed_form_sigma2_C(
                    A_list, Sigma_eps, R_list, K_LOC, K_REM, N_total)
                sigma2_comp, Schur, _ = common.companion_form_sigma2_C(
                    Atilde, Sigma_F, K, K_LOC, K_REM, p, N_total)

                V1_residual = abs(sigma2_direct - sigma2_comp)
                # V2: Schur should have only the top-left K_REM x K_REM block nonzero
                top_block = Schur[:K_REM, :K_REM]
                top_block_trace_normalized = float(np.trace(top_block)) / N_total
                off_block_max = max(
                    np.max(np.abs(Schur[K_REM:, K_REM:])),
                    np.max(np.abs(Schur[:K_REM, K_REM:])),
                    np.max(np.abs(Schur[K_REM:, :K_REM])),
                )
                V2_residual = abs(sigma2_comp - top_block_trace_normalized)

                elapsed = time.time() - t0
                cell_result = {
                    'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': p,
                    'sigma2_C_direct': sigma2_direct,
                    'sigma2_C_companion': sigma2_comp,
                    'V1_residual_direct_vs_companion': float(V1_residual),
                    'Schur_top_block_trace_normalized': float(top_block_trace_normalized),
                    'Schur_off_block_max_abs': float(off_block_max),
                    'V2_residual_full_vs_top': float(V2_residual),
                    'rho_Atilde': info['rho_Atilde'],
                    'elapsed_s': elapsed,
                }
                results.append(cell_result)
                print(f"  sigma^2_C^(p) direct      = {sigma2_direct:.10f}")
                print(f"  sigma^2_C^(p) companion   = {sigma2_comp:.10f}")
                print(f"  V1 residual               = {V1_residual:.3e}")
                print(f"  Schur off-block max abs   = {off_block_max:.3e}")
                print(f"  elapsed                   = {elapsed:.2f}s")
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                results.append({
                    'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': p,
                    'exception': str(e),
                })

    print("\n" + "=" * 100)
    print(" E5 SUMMARY")
    print("=" * 100)
    n_pass = sum(1 for r in results if r.get('V1_residual_direct_vs_companion', 1e10) < 1e-9)
    n_cells = len(results)
    print(f"  {n_pass} / {n_cells} cells with V1 residual < 1e-9")
    max_resid = max((r.get('V1_residual_direct_vs_companion', 0) for r in results), default=0)
    print(f"  max V1 residual across all cells: {max_resid:.3e}")
    max_offblock = max((r.get('Schur_off_block_max_abs', 0) for r in results), default=0)
    print(f"  max Schur off-block magnitude:    {max_offblock:.3e}")
    overall = n_pass == n_cells
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")

    out_path = HERE / 'E5_companion_equiv_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump({'overall_pass': bool(overall), 'cells': results}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
