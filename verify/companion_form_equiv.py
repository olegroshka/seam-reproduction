"""E2 — companion-form vs direct equivalence for sigma^2_C^(p), p=2.

Theoretical claim: sigma^2_C^(p) (direct VAR(p) closed form derived in E1) equals
the trace of the Schur-complement Theorem-1 form applied at the companion VAR(1)
level with the LIFTED partition REM_comp = {REM at lags t, t-1, ..., t-p+1}, i.e.

  sigma^2_C^(p) = (1/N_total) tr[
        (Atilde Sigma_F Atilde^T)_REM_comp
      - (Atilde Sigma_F)_REM_comp (Sigma_F)_REM_comp^{-1} (Sigma_F Atilde^T)_REM_comp
    ]

The companion-form innovation covariance B Sigma_eps B^T is singular (rank K in dim
K*p), so (A2) Sigma_eps ≻ 0 fails at the lifted system; the Schur-complement form
still holds because the algebra only requires (Sigma_F)_REM_comp ≻ 0 (which is the
block-Toeplitz Gamma).

We also confirm: the rest-minus-joint difference at the companion level is block-
diagonal with the only nonzero block at the (current-time REM, current-time REM)
position; tracing over REM_comp equals tracing over just the current-time REM block.

Verifications:
  V1. companion-form Theorem-1 form == direct sigma^2_C^(p) from E1.
  V2. (rest-joint) difference at REM_comp is block-diagonal at current-time-REM.
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

# reuse the DGP builder from E1
sys.path.insert(0, str(HERE))
from thm1_var_p import build_VAR_p, closed_form_sigma2_C


def companion_form_sigma2_C(A_list, Sigma_eps, Sigma_F, Atilde, K, K_LOC, K_REM, N_total):
    """Apply Theorem 1 at companion-form VAR(1) with lifted REM_comp partition.

    REM_comp indices: REM at lag 0, REM at lag 1, ..., REM at lag p-1
    In F_t = (f_t, f_{t-1}, ..., f_{t-p+1})^T ordering, REM_comp_idx within F is
    [K_LOC..K) ∪ [K + K_LOC..2K) ∪ ... ∪ [(p-1)K + K_LOC..pK)

    Returns: sigma^2_C^companion (full trace), and Schur-complement matrix block.
    """
    p = len(A_list)
    Kp = K * p
    rem_comp_idx = np.concatenate([
        np.arange(h*K + K_LOC, h*K + K) for h in range(p)
    ])

    Sigma_F_REM_comp = Sigma_F[np.ix_(rem_comp_idx, rem_comp_idx)]

    # (Atilde Sigma_F Atilde^T)_REM_comp
    AtildeSF = Atilde @ Sigma_F
    AtildeSFAtildeT = AtildeSF @ Atilde.T
    block_11 = AtildeSFAtildeT[np.ix_(rem_comp_idx, rem_comp_idx)]
    block_21 = AtildeSF[np.ix_(rem_comp_idx, rem_comp_idx)]      # (Atilde Sigma_F)_REM_comp
    block_12 = block_21.T                                         # (Sigma_F Atilde^T)_REM_comp

    Schur = block_11 - block_21 @ np.linalg.solve(Sigma_F_REM_comp, block_12)
    sigma2_C_companion_full = float(np.trace(Schur)) / N_total
    return sigma2_C_companion_full, Schur


def companion_form_top_block_only(Schur, K_REM):
    """Trace of just the top-left K_REM x K_REM block of the companion Schur complement.

    Should equal companion_form_sigma2_C's full trace (since other diagonal blocks
    of the Schur complement should be zero — restricted err minus joint err is zero
    at lagged-REM positions which are observed).
    """
    top_block = Schur[:K_REM, :K_REM]
    return float(np.trace(top_block))


def main():
    print("=" * 88)
    print(" E2 — companion-form vs direct sigma^2_C^(p) equivalence, p=2")
    print("=" * 88)

    results = {}

    # ---- Instance 1: VAR(2) K=4 blocks (2, 2)
    print("\n[Instance 1] VAR(2) K=4 K_LOC=2 K_REM=2 sigma_cross=0.3 seed=1001")
    K, K_LOC, K_REM = 4, 2, 2
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.3, seed=1001, p=2)

    sigma2_direct, _, _, _ = closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total)
    sigma2_comp, Schur = companion_form_sigma2_C(A_list, Sigma_eps, Sigma_F, Atilde,
                                                  K, K_LOC, K_REM, N_total)
    top_only = companion_form_top_block_only(Schur, K_REM) / N_total

    print(f"  direct VAR(p) closed form         = {sigma2_direct:.10f}")
    print(f"  companion-form Theorem-1 (full)   = {sigma2_comp:.10f}")
    print(f"  companion-form top-block only     = {top_only:.10f}")
    diff_V1 = abs(sigma2_direct - sigma2_comp)
    diff_V2 = abs(sigma2_comp - top_only)
    print(f"  V1 |direct - companion-full|      = {diff_V1:.3e}  ({'PASS' if diff_V1 < 1e-10 else 'FAIL'} at < 1e-10)")
    print(f"  V2 |companion-full - top-only|    = {diff_V2:.3e}  ({'PASS' if diff_V2 < 1e-10 else 'FAIL'} at < 1e-10)")

    # Inspect Schur complement off-block elements for V2 evidence
    off_block_max = max(
        np.max(np.abs(Schur[K_REM:, K_REM:])),
        np.max(np.abs(Schur[:K_REM, K_REM:])),
        np.max(np.abs(Schur[K_REM:, :K_REM])),
    )
    print(f"  Schur off-block max abs           = {off_block_max:.3e}")
    print(f"  Schur top-left K_REM x K_REM block:")
    print(f"  {Schur[:K_REM, :K_REM]}")

    results['instance_1'] = {
        'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': 2, 'seed': 1001,
        'sigma2_C_direct': sigma2_direct,
        'sigma2_C_companion_full': sigma2_comp,
        'sigma2_C_companion_top_only': top_only,
        'V1_residual_direct_vs_companion': float(diff_V1),
        'V2_residual_companion_full_vs_top': float(diff_V2),
        'Schur_off_block_max_abs': float(off_block_max),
        'V1_pass': bool(diff_V1 < 1e-10),
        'V2_pass': bool(diff_V2 < 1e-10),
    }

    # ---- Instance 2: VAR(2) K=6 blocks (4, 2)
    print("\n[Instance 2] VAR(2) K=6 K_LOC=4 K_REM=2 sigma_cross=0.3 seed=2002")
    K, K_LOC, K_REM = 6, 4, 2
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.3, seed=2002, p=2)

    sigma2_direct, _, _, _ = closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total)
    sigma2_comp, Schur = companion_form_sigma2_C(A_list, Sigma_eps, Sigma_F, Atilde,
                                                  K, K_LOC, K_REM, N_total)
    top_only = companion_form_top_block_only(Schur, K_REM) / N_total

    print(f"  direct VAR(p) closed form         = {sigma2_direct:.10f}")
    print(f"  companion-form Theorem-1 (full)   = {sigma2_comp:.10f}")
    print(f"  companion-form top-block only     = {top_only:.10f}")
    diff_V1 = abs(sigma2_direct - sigma2_comp)
    diff_V2 = abs(sigma2_comp - top_only)
    print(f"  V1 |direct - companion-full|      = {diff_V1:.3e}  ({'PASS' if diff_V1 < 1e-10 else 'FAIL'} at < 1e-10)")
    print(f"  V2 |companion-full - top-only|    = {diff_V2:.3e}  ({'PASS' if diff_V2 < 1e-10 else 'FAIL'} at < 1e-10)")

    off_block_max = max(
        np.max(np.abs(Schur[K_REM:, K_REM:])),
        np.max(np.abs(Schur[:K_REM, K_REM:])),
        np.max(np.abs(Schur[K_REM:, :K_REM])),
    )
    print(f"  Schur off-block max abs           = {off_block_max:.3e}")

    results['instance_2'] = {
        'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': 2, 'seed': 2002,
        'sigma2_C_direct': sigma2_direct,
        'sigma2_C_companion_full': sigma2_comp,
        'sigma2_C_companion_top_only': top_only,
        'V1_residual_direct_vs_companion': float(diff_V1),
        'V2_residual_companion_full_vs_top': float(diff_V2),
        'Schur_off_block_max_abs': float(off_block_max),
        'V1_pass': bool(diff_V1 < 1e-10),
        'V2_pass': bool(diff_V2 < 1e-10),
    }

    # ---- Instance 3: p=3 (push further)
    print("\n[Instance 3] VAR(3) K=4 K_LOC=2 K_REM=2 sigma_cross=0.2 seed=3003")
    K, K_LOC, K_REM = 4, 2, 2
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_f, R_list, Atilde, Sigma_F = \
        build_VAR_p(K, K_LOC, K_REM, sigma_cross=0.2, seed=3003, p=3)

    sigma2_direct, _, _, _ = closed_form_sigma2_C(A_list, R_list, Sigma_eps, K_LOC, K_REM, N_total)
    sigma2_comp, Schur = companion_form_sigma2_C(A_list, Sigma_eps, Sigma_F, Atilde,
                                                  K, K_LOC, K_REM, N_total)
    top_only = companion_form_top_block_only(Schur, K_REM) / N_total

    print(f"  direct VAR(p=3) closed form       = {sigma2_direct:.10f}")
    print(f"  companion-form Theorem-1 (full)   = {sigma2_comp:.10f}")
    print(f"  companion-form top-block only     = {top_only:.10f}")
    diff_V1 = abs(sigma2_direct - sigma2_comp)
    diff_V2 = abs(sigma2_comp - top_only)
    print(f"  V1 residual                       = {diff_V1:.3e}  ({'PASS' if diff_V1 < 1e-10 else 'FAIL'} at < 1e-10)")
    print(f"  V2 residual                       = {diff_V2:.3e}  ({'PASS' if diff_V2 < 1e-10 else 'FAIL'} at < 1e-10)")

    results['instance_3'] = {
        'K': K, 'K_LOC': K_LOC, 'K_REM': K_REM, 'p': 3, 'seed': 3003,
        'sigma2_C_direct': sigma2_direct,
        'sigma2_C_companion_full': sigma2_comp,
        'sigma2_C_companion_top_only': top_only,
        'V1_residual_direct_vs_companion': float(diff_V1),
        'V2_residual_companion_full_vs_top': float(diff_V2),
        'V1_pass': bool(diff_V1 < 1e-10),
        'V2_pass': bool(diff_V2 < 1e-10),
    }

    # ---- SUMMARY
    print("\n" + "=" * 88)
    print(" E2 SUMMARY")
    print("=" * 88)
    all_pass = all(r['V1_pass'] and r['V2_pass'] for r in results.values())
    for k, r in results.items():
        print(f"  {k} (p={r['p']}): V1 {'PASS' if r['V1_pass'] else 'FAIL'}, V2 {'PASS' if r['V2_pass'] else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  companion-form Theorem 1 evaluates to the direct VAR(p) closed form: " +
          ("YES (singular Sigma_eps^comp not an obstacle)" if all_pass else "NO"))

    out_path = HERE / 'E2_companion_form_equiv_results.json'
    with open(out_path, 'w') as f:
        json.dump({'overall_pass': bool(all_pass), 'instances': results}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
