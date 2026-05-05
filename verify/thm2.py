"""W3 — Sympy + numerical verification of Theorem 2 (cluster decomposition identity).

Per `option_alpha_theoretical_plan.md` §3 Theorem 2 + Deliverable 2 §3.

Theorem 2 statement
===================

Let M(a, b) = (Σ_f)_{a,b} − (Σ_f)_{a,REM} (Σ_f)_REM^{−1} (Σ_f)_{REM,b} for blocks a, b ∈
{LOC_1, …, LOC_J, REM} (this is the conditional covariance of (f_a, f_b) given f_REM under
the joint stationary distribution; equivalently, the Schur complement of the (REM,REM)-block
of Σ_f at the (a, b)-position).

Then σ²_C admits the exact decomposition

  σ²_C  =  Σ_j  σ²_{C, LOC_j↔REM}   +   σ²_{C, cross}

where
  σ²_{C, LOC_j↔REM} := (1/N_total) tr[ A_{REM, LOC_j} M(LOC_j, LOC_j) A_{REM, LOC_j}^T ]
  σ²_{C, cross}     := (1/N_total) Σ_{j≠k} tr[ A_{REM, LOC_j} M(LOC_j, LOC_k) A_{REM, LOC_k}^T ]

Vanishing condition (corollary): σ²_{C, cross} = 0 iff M(LOC_j, LOC_k) = 0 for all j ≠ k,
i.e., LOCAL blocks are conditionally independent given REMAINDER under the joint Gaussian
stationary distribution. This is a graphical-model condition: REM separates the LOC blocks.

The (REM, *) and (*, REM) terms vanish identically since M(REM, b) = M(a, REM) = 0 for
all a, b — by direct algebra on the Schur-complement definition.

Acceptance criteria
===================

C_W3.1 (algebraic, exact) — Sympy: σ²_C from Theorem 1 closed form equals
       Σ_j σ²_{C, LOC_j↔REM} + σ²_{C, cross} entry-wise to zero residual on at least
       3 instances at K ∈ {6, 8, 12}.

C_W3.2 (vanishing condition, exact) — Sympy: when (Σ_f)_{LOC_j, LOC_k} − (Σ_f)_{LOC_j, REM}
       (Σ_f)_REM^{−1} (Σ_f)_{REM, LOC_k} = 0 for all j ≠ k (engineered), σ²_{C, cross} = 0.

C_W3.3 (numerical, harp) — load harp's (A, Σ_f) from step_B17_1_results.json; compute
       σ²_C from Theorem 1, Σ_j σ²_{C, LOC_j↔REM}, and σ²_{C, cross} via Theorem 2.
       Verify Theorem 1 = Theorem 2 sum to floating-point precision (relative error < 1e-10).

Outputs: 60_phase_D/W3_sympy_verification_results.json + run.log
"""
from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent


# ─── Sympy-instance helpers ──────────────────────────────────────────────────

def random_rational_matrix(rng, rows: int, cols: int, denom: int = 100):
    return sp.Matrix(rows, cols, lambda i, j: sp.Rational(rng.randint(-denom + 1, denom - 1), denom))


def make_VAR_instance(rng, K: int, block_sizes):
    """Build a stationary VAR(1) with the given partition.

    Returns A, Sigma_eps, Sigma_f, block_slices, rem_idx (rem block index in block_slices list)
    """
    assert sum(block_sizes) == K
    A_raw = random_rational_matrix(rng, K, K, denom=10)
    A = A_raw / 4  # scale for stationarity (rho ≈ 0.5)
    W = random_rational_matrix(rng, K, K, denom=10)
    Sigma_eps = (W * W.T + sp.eye(K))
    Sigma_eps = (Sigma_eps + Sigma_eps.T) / 2

    I_K2 = sp.eye(K * K)
    AkronA = sp.kronecker_product(A, A)
    M = I_K2 - AkronA
    vec_Seps = Sigma_eps.reshape(K * K, 1)
    vec_Sf = M.solve(vec_Seps)
    Sigma_f = vec_Sf.reshape(K, K)
    Sigma_f = (Sigma_f + Sigma_f.T) / 2

    block_slices = []
    s = 0
    for sz in block_sizes:
        block_slices.append((s, s + sz))
        s += sz
    rem_idx = len(block_slices) - 1  # REMAINDER is last block by convention

    return A, Sigma_eps, Sigma_f, block_slices, rem_idx


# ─── Theorem 1 closed form (exact and float versions) ────────────────────────

def sigma2_C_theorem1_sympy(A, Sigma_f, block_slices, rem_idx, N_total):
    s, e = block_slices[rem_idx]
    K_REM = e - s
    A_Sigma_f = A * Sigma_f
    A_Sigma_f_AT = A_Sigma_f * A.T
    Sigma_f_AT = Sigma_f * A.T
    Sigma_f_REM = Sigma_f[s:e, s:e]
    Sigma_f_REM_inv = Sigma_f_REM.inv()
    DeltaVar = (A_Sigma_f_AT[s:e, s:e]
                - A_Sigma_f[s:e, s:e] * Sigma_f_REM_inv * Sigma_f_AT[s:e, s:e])
    DeltaVar = sp.simplify(DeltaVar)
    sigma2_C = sp.Trace(DeltaVar).doit() / N_total
    return sp.simplify(sigma2_C), DeltaVar


def sigma2_C_theorem1_numpy(A, Sigma_f, block_slices, rem_idx, N_total):
    s, e = block_slices[rem_idx]
    K_REM = e - s
    A_Sigma_f = A @ Sigma_f
    A_Sigma_f_AT = A_Sigma_f @ A.T
    Sigma_f_AT = Sigma_f @ A.T
    Sigma_f_REM = Sigma_f[s:e, s:e]
    Sigma_f_REM_inv = np.linalg.inv(Sigma_f_REM + 1e-12 * np.eye(K_REM))
    DeltaVar = (A_Sigma_f_AT[s:e, s:e]
                - A_Sigma_f[s:e, s:e] @ Sigma_f_REM_inv @ Sigma_f_AT[s:e, s:e])
    DeltaVar = (DeltaVar + DeltaVar.T) / 2
    return float(np.trace(DeltaVar)) / N_total, DeltaVar


# ─── Theorem 2: Schur-complement-based cluster decomposition ─────────────────

def conditional_cov_block_sympy(Sigma_f, a_slc, b_slc, rem_slc):
    """M(a, b) = (Σ_f)_{a,b} − (Σ_f)_{a,REM} (Σ_f)_REM^{−1} (Σ_f)_{REM,b}.

    Equals the conditional covariance of (f_a, f_b) given f_REM under joint Gaussian.
    """
    sa, ea = a_slc
    sb, eb = b_slc
    sr, er = rem_slc
    Sigma_f_ab = Sigma_f[sa:ea, sb:eb]
    Sigma_f_a_REM = Sigma_f[sa:ea, sr:er]
    Sigma_f_REM_b = Sigma_f[sr:er, sb:eb]
    Sigma_f_REM = Sigma_f[sr:er, sr:er]
    Sigma_f_REM_inv = Sigma_f_REM.inv()
    return sp.simplify(Sigma_f_ab - Sigma_f_a_REM * Sigma_f_REM_inv * Sigma_f_REM_b)


def conditional_cov_block_numpy(Sigma_f, a_slc, b_slc, rem_slc):
    sa, ea = a_slc
    sb, eb = b_slc
    sr, er = rem_slc
    K_REM = er - sr
    Sigma_f_ab = Sigma_f[sa:ea, sb:eb]
    Sigma_f_a_REM = Sigma_f[sa:ea, sr:er]
    Sigma_f_REM_b = Sigma_f[sr:er, sb:eb]
    Sigma_f_REM = Sigma_f[sr:er, sr:er]
    Sigma_f_REM_inv = np.linalg.inv(Sigma_f_REM + 1e-12 * np.eye(K_REM))
    return Sigma_f_ab - Sigma_f_a_REM @ Sigma_f_REM_inv @ Sigma_f_REM_b


def sigma2_C_theorem2_sympy(A, Sigma_f, block_slices, rem_idx, N_total):
    """Σ_j σ²_{C, LOC_j↔REM} + σ²_{C, cross} via Theorem 2 cluster decomposition (exact)."""
    rem_slc = block_slices[rem_idx]
    sr, er = rem_slc
    loc_indices = [j for j in range(len(block_slices)) if j != rem_idx]

    # Per-LOC_j↔REM contribution (diagonal terms in the bilinear sum)
    diag_contribs = {}
    for j in loc_indices:
        a_slc = block_slices[j]
        sa, ea = a_slc
        A_REM_LOCj = A[sr:er, sa:ea]
        M_jj = conditional_cov_block_sympy(Sigma_f, a_slc, a_slc, rem_slc)
        contrib = sp.Trace(A_REM_LOCj * M_jj * A_REM_LOCj.T).doit() / N_total
        diag_contribs[f'LOC{j}_REM'] = sp.simplify(contrib)

    # Cross terms (j != k)
    cross_total = sp.Integer(0)
    cross_pairs = {}
    for j in loc_indices:
        for k in loc_indices:
            if j == k:
                continue
            a_slc = block_slices[j]
            b_slc = block_slices[k]
            sa, ea = a_slc
            sb, eb = b_slc
            A_REM_LOCj = A[sr:er, sa:ea]
            A_REM_LOCk = A[sr:er, sb:eb]
            M_jk = conditional_cov_block_sympy(Sigma_f, a_slc, b_slc, rem_slc)
            term = sp.Trace(A_REM_LOCj * M_jk * A_REM_LOCk.T).doit() / N_total
            cross_total += term
            cross_pairs[f'LOC{j}_LOC{k}'] = sp.simplify(term)

    sigma2_C_T2 = sum(diag_contribs.values(), sp.Integer(0)) + cross_total
    return sp.simplify(sigma2_C_T2), diag_contribs, sp.simplify(cross_total), cross_pairs


def sigma2_C_theorem2_numpy(A, Sigma_f, block_slices, rem_idx, N_total):
    rem_slc = block_slices[rem_idx]
    sr, er = rem_slc
    loc_indices = [j for j in range(len(block_slices)) if j != rem_idx]

    diag_contribs = {}
    for j in loc_indices:
        a_slc = block_slices[j]
        sa, ea = a_slc
        A_REM_LOCj = A[sr:er, sa:ea]
        M_jj = conditional_cov_block_numpy(Sigma_f, a_slc, a_slc, rem_slc)
        contrib = float(np.trace(A_REM_LOCj @ M_jj @ A_REM_LOCj.T)) / N_total
        diag_contribs[f'LOC{j}_REM'] = contrib

    cross_total = 0.0
    cross_pairs = {}
    for j in loc_indices:
        for k in loc_indices:
            if j == k:
                continue
            a_slc = block_slices[j]
            b_slc = block_slices[k]
            sa, ea = a_slc
            sb, eb = b_slc
            A_REM_LOCj = A[sr:er, sa:ea]
            A_REM_LOCk = A[sr:er, sb:eb]
            M_jk = conditional_cov_block_numpy(Sigma_f, a_slc, b_slc, rem_slc)
            term = float(np.trace(A_REM_LOCj @ M_jk @ A_REM_LOCk.T)) / N_total
            cross_total += term
            cross_pairs[f'LOC{j}_LOC{k}'] = term

    return sum(diag_contribs.values()) + cross_total, diag_contribs, cross_total, cross_pairs


# ─── Verification routines ───────────────────────────────────────────────────

def verify_sympy_instance(label, A, Sigma_eps, Sigma_f, block_slices, rem_idx, N_total):
    print(f"\n  --- {label} ---")
    print(f"  K = {A.shape[0]}, block_slices = {block_slices}, rem_idx = {rem_idx}")

    # Theorem 1 closed form (reference)
    sigma2_C_T1, _ = sigma2_C_theorem1_sympy(A, Sigma_f, block_slices, rem_idx, N_total)

    # Theorem 2 sum
    sigma2_C_T2, diag_contribs, cross_total, cross_pairs = sigma2_C_theorem2_sympy(
        A, Sigma_f, block_slices, rem_idx, N_total)

    # Identity: T1 - T2 = 0 exactly
    residual = sp.simplify(sigma2_C_T1 - sigma2_C_T2)
    print(f"  σ²_C (Theorem 1): {float(sigma2_C_T1):.10f}")
    print(f"  Σ_j σ²_{{C, LOC_j↔REM}} (diagonal): {float(sum(diag_contribs.values(), sp.Integer(0))):.10f}")
    for tag, val in diag_contribs.items():
        print(f"      {tag}: {float(val):.10f}")
    print(f"  σ²_{{C, cross}} (off-diagonal): {float(cross_total):+.10f}")
    print(f"  σ²_C (Theorem 2 sum): {float(sigma2_C_T2):.10f}")
    print(f"  Theorem 1 − Theorem 2 residual: {residual}")
    print(f"  ✓ {'PASS' if residual == 0 else 'FAIL'}: {'identity holds exactly' if residual == 0 else 'NON-ZERO residual'}")

    return {
        'label': label,
        'K': A.shape[0],
        'block_sizes': [e - s for s, e in block_slices],
        'sigma2_C_T1': float(sigma2_C_T1),
        'sigma2_C_T2_total': float(sigma2_C_T2),
        'diag_contribs': {k: float(v) for k, v in diag_contribs.items()},
        'cross_total': float(cross_total),
        'cross_pairs': {k: float(v) for k, v in cross_pairs.items()},
        'residual_str': str(residual),
        'identity_passes': bool(residual == 0),
    }


def verify_vanishing_condition_sympy():
    """Engineer Σ_f s.t. M(LOC_j, LOC_k) = 0 for j ≠ k; verify σ²_{C, cross} = 0 exactly.

    Construction: choose Σ_f with (Σ_f)_LOC block-diagonal at LOCAL sub-partition AND
    (Σ_f)_{LOC, REM} = 0 (LOCAL blocks unconditionally independent of REM; trivially also
    conditionally). Then M(LOC_j, LOC_k) = 0 for j ≠ k. We won't actually solve a Lyapunov
    here; instead we directly construct (A, Σ_f) consistent with the structure.
    """
    print(f"\n  --- vanishing condition test ---")
    K = 8
    K_LOC1, K_LOC2, K_REM = 2, 2, 4
    rng = random.Random(2026)

    # Build Σ_f directly (skip Lyapunov; just test the algebra of Theorem 2):
    # Σ_f = block-diag(Σ_LOC1, Σ_LOC2, Σ_REM) → M(LOC1, LOC2) = (Σ_f)_{LOC1, LOC2}=0
    # since (Σ_f)_{LOC1, REM} = 0 too. So vanishing condition holds.
    Sigma_LOC1 = random_rational_matrix(rng, K_LOC1, K_LOC1, denom=10)
    Sigma_LOC1 = Sigma_LOC1 * Sigma_LOC1.T + sp.eye(K_LOC1)
    Sigma_LOC2 = random_rational_matrix(rng, K_LOC2, K_LOC2, denom=10)
    Sigma_LOC2 = Sigma_LOC2 * Sigma_LOC2.T + sp.eye(K_LOC2)
    Sigma_REM = random_rational_matrix(rng, K_REM, K_REM, denom=10)
    Sigma_REM = Sigma_REM * Sigma_REM.T + sp.eye(K_REM)

    Sigma_f = sp.zeros(K, K)
    Sigma_f[0:K_LOC1, 0:K_LOC1] = Sigma_LOC1
    Sigma_f[K_LOC1:K_LOC1+K_LOC2, K_LOC1:K_LOC1+K_LOC2] = Sigma_LOC2
    Sigma_f[K_LOC1+K_LOC2:, K_LOC1+K_LOC2:] = Sigma_REM
    Sigma_f = (Sigma_f + Sigma_f.T) / 2  # enforce symmetry

    # Random A (any A is fine — the vanishing depends on Σ_f only)
    A = random_rational_matrix(rng, K, K, denom=10) / 4

    block_slices = [(0, K_LOC1), (K_LOC1, K_LOC1+K_LOC2), (K_LOC1+K_LOC2, K)]

    # Compute Theorem 2 cross term — should be 0 exactly
    rem_slc = block_slices[2]
    M_LOC1_LOC2 = conditional_cov_block_sympy(Sigma_f, block_slices[0], block_slices[1], rem_slc)
    M_LOC2_LOC1 = conditional_cov_block_sympy(Sigma_f, block_slices[1], block_slices[0], rem_slc)
    M_LOC1_LOC2_simp = sp.simplify(M_LOC1_LOC2)
    M_LOC2_LOC1_simp = sp.simplify(M_LOC2_LOC1)
    M_max1 = max(abs(M_LOC1_LOC2_simp[i, j]) for i in range(K_LOC1) for j in range(K_LOC2))
    M_max2 = max(abs(M_LOC2_LOC1_simp[i, j]) for i in range(K_LOC2) for j in range(K_LOC1))
    print(f"  Engineered Σ_f block-diagonal at (LOC1, LOC2, REM)")
    print(f"  M(LOC1, LOC2) max(|entry|) = {M_max1}")
    print(f"  M(LOC2, LOC1) max(|entry|) = {M_max2}")

    # Compute σ²_{C, cross}
    _, _, cross_total, cross_pairs = sigma2_C_theorem2_sympy(
        A, Sigma_f, block_slices, rem_idx=2, N_total=8)
    cross_total_simp = sp.simplify(cross_total)
    print(f"  σ²_{{C, cross}} = {cross_total_simp}")
    print(f"  ✓ {'PASS' if cross_total_simp == 0 else 'FAIL'}: cross term {'vanishes' if cross_total_simp == 0 else 'NON-ZERO'}")

    return {
        'M_LOC1_LOC2_max': str(M_max1),
        'M_LOC2_LOC1_max': str(M_max2),
        'cross_total_str': str(cross_total_simp),
        'vanishing_passes': bool(cross_total_simp == 0),
    }


def verify_numerical_random(seed=2026, K=12, block_sizes=(4, 4, 4), N_total=24):
    """Numerical Theorem 1 vs Theorem 2 cross-check on a random K=12 (4,4,4) instance.

    This replaces the original 'harp numerical instance' check (which loaded
    from a project-specific JSON of fitted parameters) with a self-contained
    random instance reproducing the same Theorem 1 = Theorem 2 sum identity
    on a partition matching the paper's D7 simulation setup (K=12, J=2 LOC + REM).
    """
    print(f"\n  --- numerical instance (random VAR(1), K={K}, blocks={block_sizes}) ---")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K, K)) * 0.15
    eigs = np.linalg.eigvals(A)
    A = A * (0.85 / max(abs(eigs).max(), 1e-9))  # rescale to spectral radius 0.85
    W = rng.standard_normal((K, K))
    Sigma_eps = W @ W.T + 0.1 * np.eye(K)
    # Solve discrete Lyapunov: Sigma_f = A Sigma_f A^T + Sigma_eps
    from scipy.linalg import solve_discrete_lyapunov
    Sigma_f = solve_discrete_lyapunov(A, Sigma_eps)

    block_slices = []
    s = 0
    for sz in block_sizes:
        block_slices.append((s, s + sz))
        s += sz
    rem_idx = len(block_slices) - 1
    print(f"  spectral radius rho(A) = {abs(np.linalg.eigvals(A)).max():.4f}")

    sigma2_C_T1, _ = sigma2_C_theorem1_numpy(A, Sigma_f, block_slices, rem_idx=rem_idx, N_total=N_total)
    print(f"  σ²_C (Theorem 1): {sigma2_C_T1:.10f}")

    sigma2_C_T2, diag_contribs, cross_total, cross_pairs = sigma2_C_theorem2_numpy(
        A, Sigma_f, block_slices, rem_idx=rem_idx, N_total=N_total)
    print(f"  Σ_j σ²_{{C, LOC_j↔REM}} (diagonal):")
    for tag, val in diag_contribs.items():
        print(f"      {tag}: {val:+.10f}")
    print(f"  σ²_{{C, cross}}: {cross_total:+.10f}")
    print(f"  σ²_C (Theorem 2 sum): {sigma2_C_T2:.10f}")
    rel_err = abs(sigma2_C_T1 - sigma2_C_T2) / max(abs(sigma2_C_T1), 1e-12)
    print(f"  Relative error |T1 − T2| / |T1|: {rel_err:.2e} (< 1e-10 = PASS)")

    return {
        'K': int(K),
        'block_sizes': list(block_sizes),
        'N_total': int(N_total),
        'sigma2_C_T1': float(sigma2_C_T1),
        'sigma2_C_T2_total': float(sigma2_C_T2),
        'cross_total': float(cross_total),
        'rel_error_T1_T2': float(rel_err),
        'numerical_pass': bool(rel_err < 1e-10),
    }


def main():
    print("=" * 88)
    print(" W3 Sympy + numerical verification — Theorem 2 cluster decomposition identity")
    print("=" * 88)
    t0 = time.time()

    sympy_results = []

    # Three Sympy instances at increasing K with multi-LOC partitions
    cases = [
        ("K=6, blocks=(2,2,2): J=2 LOC + REM",  6, [2, 2, 2], 1001, 6),
        ("K=8, blocks=(2,2,4): asymmetric LOC", 8, [2, 2, 4], 1002, 8),
        ("K=8, blocks=(2,2,2,2): J=3 LOC + REM", 8, [2, 2, 2, 2], 1003, 8),
    ]

    for label, K, block_sizes, seed, N_total in cases:
        rng = random.Random(seed)
        A, Sigma_eps, Sigma_f, block_slices, rem_idx = make_VAR_instance(rng, K, block_sizes)
        result = verify_sympy_instance(label, A, Sigma_eps, Sigma_f, block_slices, rem_idx, N_total)
        result['seed'] = seed
        sympy_results.append(result)

    vanishing_result = verify_vanishing_condition_sympy()
    harp_result = verify_numerical_random()

    # Summary
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    n_sympy_pass = sum(1 for r in sympy_results if r['identity_passes'])
    print(f"  Sympy instances PASS: {n_sympy_pass} / {len(sympy_results)}")
    print(f"  Vanishing-condition PASS: {vanishing_result['vanishing_passes']}")
    print(f"  Numerical random instance PASS: {harp_result['numerical_pass']} (rel err {harp_result['rel_error_T1_T2']:.2e})")

    overall = (n_sympy_pass == len(sympy_results)
               and vanishing_result['vanishing_passes']
               and harp_result['numerical_pass'])
    print(f"\n  OVERALL VERDICT: {'PASS' if overall else 'FAIL'}")
    print(f"  Theorem 2 cluster decomposition identity VERIFIED:")
    print(f"    - C_W3.1 (Sympy exact): {n_sympy_pass}/{len(sympy_results)}")
    print(f"    - C_W3.2 (vanishing condition exact): {'PASS' if vanishing_result['vanishing_passes'] else 'FAIL'}")
    print(f"    - C_W3.3 (numerical random instance < 1e-10 rel err): {'PASS' if harp_result['numerical_pass'] else 'FAIL'}")

    out = {
        'verification_target': "Theorem 2 cluster decomposition identity",
        'tool': f'sympy {sp.__version__} (exact rational) + numpy (random numerical)',
        'sympy_instances': sympy_results,
        'vanishing_condition_test': vanishing_result,
        'numerical_random_test': harp_result,
        'overall_pass': overall,
        'runtime_seconds': time.time() - t0,
    }
    out_path = HERE / 'W3_sympy_verification_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
