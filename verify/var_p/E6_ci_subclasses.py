"""E6 — CI condition battery: map subclasses where contemporaneous CI suffices vs not.

For VAR(p), the cluster decomposition cross-term sigma^2_{C, cross_comp} vanishes
(structurally over Atilde entries) iff M^comp(LOC_j_comp, LOC_k_comp) = 0 ∀ j ≠ k,
i.e., REM_comp separates LOC_j_comp and LOC_k_comp in the (K*p)-dim companion-state
conditional-independence graph.

E3 showed that engineering "no LOC_j <-> LOC_k coupling at any lag" + block-diagonal
Sigma_eps is NOT sufficient — the cross-term remains ~3% of total because finite p-lag
REM conditioning leaks dependence through unobserved past REM.

E6 tests structural subclasses to find what additional restrictions on the DGP make the
contemporaneous CI condition sufficient:

  S1 - Baseline: random DGP with cross-coupling everywhere
       Expected: large cross-term
  S2 - No LOC_j <-> LOC_k at any lag + block-diagonal Sigma_eps (= E3 DGP A)
       Expected: small but NONZERO cross-term (the finite-p leakage)
  S3 - Add: REM dynamics restricted to AR(1)-within-REM (no (A_h)_REM,REM for h >= 2)
       Conjecture: still nonzero (LOC->REM and REM->LOC propagation across lags still leaks)
  S4 - Further: REM-LOC coupling only at lag 1 (no (A_h)_REM,LOC for h >= 2)
       Conjecture: still nonzero
  S5 - Further: LOC-REM coupling only at lag 1 (no (A_h)_LOC,REM for h >= 2)
       Conjecture: nonzero, perhaps smaller
  S6 - All of S3+S4+S5: REM appears only at lag 1 in any block
       Conjecture: zero or near-zero (the system effectively reduces to VAR(1) for
       LOC <-> REM communication; contemporaneous CI may now suffice)
  S7 - True VAR(1): set p=2 but compute against ground-truth A_2 = 0 baseline (already in E1)
       Expected: zero exactly (the VAR(1) limit)

Each subclass tested at K=6 blocks (2, 2, 2), p=2, with engineered structure.
"""
from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common


def run_subclass(name, build_kwargs, K_LOC_list=[2, 2], K_REM=2, p=2, sigma_cross=0.3, seed=6000):
    K = sum(K_LOC_list) + K_REM
    N_total = K_REM * 2
    A_list, Sigma_eps, Sigma_F, R_list, Atilde, info = common.build_VAR_p_DGP(
        K=K, K_LOC_list=K_LOC_list, K_REM=K_REM, sigma_cross=sigma_cross,
        seed=seed, p=p, **build_kwargs)
    sigma2_total, _ = common.closed_form_sigma2_C(
        A_list, Sigma_eps, R_list, sum(K_LOC_list), K_REM, N_total)
    decomp = common.companion_cluster_decomposition(
        Atilde, Sigma_F, K, K_LOC_list, K_REM, p, N_total)
    rel_cross = abs(decomp['sigma2_cross']) / max(abs(sigma2_total), 1e-20) * 100
    return {
        'name': name,
        'build_kwargs': build_kwargs,
        'sigma2_C_total': sigma2_total,
        'sigma2_C_cross_comp': decomp['sigma2_cross'],
        'sigma2_C_diag_per_LOC': decomp['diag_per_LOC'],
        'rel_cross_pct': float(rel_cross),
        'M_frobenius': decomp['M_frobenius'],
        'rho_Atilde': info['rho_Atilde'],
    }


def main():
    print("=" * 100)
    print(" E6 — CI condition battery: subclass mapping (K=6, blocks (2,2,2), p=2)")
    print("=" * 100)

    K_LOC_list = [2, 2]
    K_REM = 2

    subclasses = [
        ('S1_baseline',
         dict(),  # no restrictions
         'Random DGP, all cross-couplings present'),

        ('S2_no_LOC_LOC_plus_blk_diag_Sigma_eps',
         dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True),
         'No LOC_j<->LOC_k at any lag + Sigma_eps block-diag'),

        ('S3_S2_plus_REM_REM_AR1',
         dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True,
              A_h_REM_REM_zero_at_higher_lags=True),
         'S2 + (A_h)_REM,REM = 0 for h>=2 (REM AR(1)-within)'),

        ('S4_S3_plus_REM_LOC_only_lag1',
         dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True,
              A_h_REM_REM_zero_at_higher_lags=True,
              A_h_REM_LOC_only_at_lag1=True),
         'S3 + (A_h)_REM,LOC = 0 for h>=2 (LOC->REM only at lag 1)'),

        ('S5_S3_plus_LOC_REM_only_lag1',
         dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True,
              A_h_REM_REM_zero_at_higher_lags=True,
              A_h_LOC_REM_only_at_lag1=True),
         'S3 + (A_h)_LOC,REM = 0 for h>=2 (REM->LOC only at lag 1)'),

        ('S6_S3_S4_S5_REM_only_lag1_everywhere',
         dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True,
              A_h_REM_REM_zero_at_higher_lags=True,
              A_h_REM_LOC_only_at_lag1=True,
              A_h_LOC_REM_only_at_lag1=True),
         'S3 + S4 + S5: REM appears only at lag 1 in all blocks'),
    ]

    results = []
    print(f"\n{'name':<55}{'rho':>7}{'sigma^2_C':>14}{'cross':>15}{'|cross|/total':>16}")
    print("-" * 110)
    for name, kwargs, description in subclasses:
        r = run_subclass(name, kwargs, K_LOC_list, K_REM, p=2, sigma_cross=0.3, seed=6000)
        r['description'] = description
        results.append(r)
        print(f"  {name:<53} {r['rho_Atilde']:>6.3f} "
              f"{r['sigma2_C_total']:>14.6f} {r['sigma2_C_cross_comp']:>+14.6e} "
              f"{r['rel_cross_pct']:>14.3f}%")

    # Repeat at p=3 to see if pattern persists
    print(f"\n  Now repeating at p=3 (push further):")
    print(f"\n{'name':<55}{'rho':>7}{'sigma^2_C':>14}{'cross':>15}{'|cross|/total':>16}")
    print("-" * 110)
    p3_results = []
    for name, kwargs, description in subclasses:
        r = run_subclass(name + '_p3', kwargs, K_LOC_list, K_REM, p=3, sigma_cross=0.3, seed=7000)
        r['description'] = description
        p3_results.append(r)
        print(f"  {name + '_p3':<53} {r['rho_Atilde']:>6.3f} "
              f"{r['sigma2_C_total']:>14.6f} {r['sigma2_C_cross_comp']:>+14.6e} "
              f"{r['rel_cross_pct']:>14.3f}%")

    # Multi-seed sanity for S6 at p=2 (the most-restricted subclass)
    print("\n  S6 robustness across 10 seeds (p=2):")
    s6_cross_values = []
    for seed in range(6000, 6010):
        r = run_subclass('S6_seed{}'.format(seed),
                          dict(A_LOC_LOC_zero=True, sigma_eps_block_diag=True,
                               A_h_REM_REM_zero_at_higher_lags=True,
                               A_h_REM_LOC_only_at_lag1=True,
                               A_h_LOC_REM_only_at_lag1=True),
                          K_LOC_list, K_REM, p=2, sigma_cross=0.3, seed=seed)
        s6_cross_values.append(r['rel_cross_pct'])
        print(f"    seed={seed}: rel cross = {r['rel_cross_pct']:.4f}%")
    print(f"    S6 mean rel cross across 10 seeds: {np.mean(s6_cross_values):.4f}%")
    print(f"    S6 max  rel cross across 10 seeds: {np.max(s6_cross_values):.4f}%")

    # SUMMARY
    print("\n" + "=" * 100)
    print(" E6 SUMMARY")
    print("=" * 100)
    print(f"  Subclasses tested: {len(subclasses)} structural patterns at p=2 and p=3")
    print(f"")
    print(f"  Key findings (p=2):")
    for r in results:
        verdict = "ZERO" if r['rel_cross_pct'] < 0.01 else ("SMALL" if r['rel_cross_pct'] < 1.0 else "LARGE")
        print(f"    {r['name']:<55}: rel cross = {r['rel_cross_pct']:.4f}% [{verdict}]")
    print(f"")
    print(f"  S6 mean rel cross (10 seeds, p=2): {np.mean(s6_cross_values):.4f}%")
    print(f"  S6 max  rel cross (10 seeds, p=2): {np.max(s6_cross_values):.4f}%")
    print(f"")
    s6_threshold = 0.01  # arbitrary threshold for "effectively zero"
    s6_zero = max(s6_cross_values) < s6_threshold
    print(f"  Does S6 give effectively zero cross-term? {s6_zero}")
    print(f"  (Threshold: rel cross < {s6_threshold}% = effectively zero)")

    out_path = HERE / 'E6_ci_subclasses_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'subclasses_p2': results,
            'subclasses_p3': p3_results,
            'S6_robustness_seeds': s6_cross_values,
            'S6_mean': float(np.mean(s6_cross_values)),
            'S6_max': float(np.max(s6_cross_values)),
            'S6_zero_at_threshold_01pct': bool(s6_zero),
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
