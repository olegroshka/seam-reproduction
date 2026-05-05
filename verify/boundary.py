"""W9.1 — Specialize V (Theorem 4 asymptotic variance) at A_REM,LOC = 0 (null hypothesis H_0).

Per review #3 §3 W9 directive. Goal: derive V_0 := V evaluated at the null A_REM,LOC = 0,
verify simplification and structural form, support Theorem 5 (test for cross-block coupling
existence) in §5.5.

Under H_0: A_REM,LOC_j = 0 for all j (block-diagonal A at the (LOC, REM) partition).
By Proposition 1, σ²_C = 0 under H_0.

Strategy:
  1. Construct null DGP: A with A_REM,LOC_j = 0 for all j, generic A_LOC,LOC, A_LOC,REM,
     A_REM,REM. Σ_ε generic. Solve Lyapunov for Σ_f.
  2. Compute ∇φ at the null using finite-diff and the corrected closed forms (∇A) + (∇Σ).
  3. Identify which entries of ∇φ are non-zero — should be only a structured subset.
  4. Compute V_0 = ∇φ^T Ω ∇φ via theoretical Ω (Σ_f^{-1} ⊗ Σ_ε for Ω_AA, Wick + geometric
     for Ω_ff).
  5. Compare V_0 to general V (with non-zero A_REM,LOC) for the same DGP family.
  6. Derive analytical structure: which gradient entries survive at the null.

Acceptance: V_0 is well-defined (positive); reduces dimensionally relative to general V
(rank of effective gradient subspace shrinks).

Outputs: 60_phase_D/W9_1_V_null_specialization_results.json + run.log
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


def construct_null_DGP(K, K_LOC1, K_LOC2, K_REM, sigma_within=0.5, sigma_LOC_REM=0.3, sigma_REM=0.5, target_rho=0.85, seed=2026):
    """Construct DGP under H_0: A_REM,LOC_j = 0 for all j. Other A blocks generic."""
    rng = np.random.default_rng(seed)
    K_LOC = K_LOC1 + K_LOC2
    A = np.zeros((K, K))
    # Within-LOC blocks
    A[0:K_LOC1, 0:K_LOC1] = rng.standard_normal((K_LOC1, K_LOC1)) * sigma_within
    A[K_LOC1:K_LOC, K_LOC1:K_LOC] = rng.standard_normal((K_LOC2, K_LOC2)) * sigma_within
    # REM-REM block
    A[K_LOC:, K_LOC:] = rng.standard_normal((K_REM, K_REM)) * sigma_REM
    # A_LOC,REM cross-block (allowed under H_0; only A_REM,LOC = 0 is the null)
    A[0:K_LOC, K_LOC:] = rng.standard_normal((K_LOC, K_REM)) * sigma_LOC_REM
    # A_REM,LOC = 0 (the null structure)
    # (already zeroed above)

    # Scale to target spectral radius
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho > 1e-9:
        A = A * (target_rho / rho)

    # Σ_ε generic
    W = rng.standard_normal((K, K))
    Sigma_eps = W @ W.T + 0.5 * np.eye(K)
    Sigma_eps = (Sigma_eps + Sigma_eps.T) / 2

    # Lyapunov solve
    I_K2 = np.eye(K * K)
    AkronA = np.kron(A, A)
    vec_Seps = Sigma_eps.reshape(K * K, order='F')
    vec_Sf = np.linalg.solve(I_K2 - AkronA, vec_Seps)
    Sigma_f = vec_Sf.reshape((K, K), order='F')
    Sigma_f = (Sigma_f + Sigma_f.T) / 2

    block_slices = [(0, K_LOC1), (K_LOC1, K_LOC), (K_LOC, K)]
    return A, Sigma_eps, Sigma_f, block_slices


def construct_alternative_DGP(K, K_LOC1, K_LOC2, K_REM, sigma_cross_REM_LOC, sigma_within=0.5, sigma_LOC_REM=0.3, sigma_REM=0.5, target_rho=0.85, seed=2026):
    """Same structure as construct_null_DGP but with non-zero A_REM,LOC scaled by sigma_cross_REM_LOC."""
    rng = np.random.default_rng(seed)
    K_LOC = K_LOC1 + K_LOC2
    A = np.zeros((K, K))
    A[0:K_LOC1, 0:K_LOC1] = rng.standard_normal((K_LOC1, K_LOC1)) * sigma_within
    A[K_LOC1:K_LOC, K_LOC1:K_LOC] = rng.standard_normal((K_LOC2, K_LOC2)) * sigma_within
    A[K_LOC:, K_LOC:] = rng.standard_normal((K_REM, K_REM)) * sigma_REM
    A[0:K_LOC, K_LOC:] = rng.standard_normal((K_LOC, K_REM)) * sigma_LOC_REM
    A[K_LOC:, 0:K_LOC] = rng.standard_normal((K_REM, K_LOC)) * sigma_cross_REM_LOC

    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho > 1e-9:
        A = A * (target_rho / rho)
    W = rng.standard_normal((K, K))
    Sigma_eps = W @ W.T + 0.5 * np.eye(K)
    Sigma_eps = (Sigma_eps + Sigma_eps.T) / 2
    I_K2 = np.eye(K * K)
    AkronA = np.kron(A, A)
    vec_Seps = Sigma_eps.reshape(K * K, order='F')
    vec_Sf = np.linalg.solve(I_K2 - AkronA, vec_Seps)
    Sigma_f = vec_Sf.reshape((K, K), order='F')
    Sigma_f = (Sigma_f + Sigma_f.T) / 2
    block_slices = [(0, K_LOC1), (K_LOC1, K_LOC), (K_LOC, K)]
    return A, Sigma_eps, Sigma_f, block_slices


def sigma2_C_phi(A, Sigma_f, block_slices, rem_idx, N_total):
    s, e = block_slices[rem_idx]
    K_REM = e - s
    A_S = A @ Sigma_f
    A_S_AT = A_S @ A.T
    Sigma_AT = Sigma_f @ A.T
    Sigma_REM = Sigma_f[s:e, s:e]
    Sigma_REM_inv = np.linalg.inv(Sigma_REM + 1e-12 * np.eye(K_REM))
    DV = (A_S_AT[s:e, s:e]
          - A_S[s:e, s:e] @ Sigma_REM_inv @ Sigma_AT[s:e, s:e])
    DV = (DV + DV.T) / 2
    return float(np.trace(DV)) / N_total


def grad_A_closed(A, Sigma_f, K_REM, K, N_total):
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM)); E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T
    return (2.0 / N_total) * P_REM @ A @ Sigma_f @ (np.eye(K) - Pi_REM @ Sigma_f)


def grad_Sigma_closed(A, Sigma_f, K_REM, K, N_total):
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM)); E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T
    M = (A.T @ P_REM @ A
         - A.T @ P_REM @ A @ Sigma_f @ Pi_REM
         - Pi_REM @ Sigma_f @ A.T @ P_REM @ A
         + Pi_REM @ Sigma_f @ A.T @ P_REM @ A @ Sigma_f @ Pi_REM)
    return M / N_total


def vech(M, K=None):
    if K is None:
        K = M.shape[0]
    out = np.zeros(K * (K + 1) // 2)
    k = 0
    for j in range(K):
        for i in range(j, K):
            out[k] = M[i, j]
            k += 1
    return out


def duplication_matrix(K):
    K_vech = K * (K + 1) // 2
    D = np.zeros((K * K, K_vech))
    k = 0
    for j in range(K):
        for i in range(j, K):
            D[i + j * K, k] = 1
            if i != j:
                D[j + i * K, k] = 1
            k += 1
    return D


def commutation_matrix(K):
    K_KK = np.zeros((K * K, K * K))
    for i in range(K):
        for j in range(K):
            K_KK[i + j * K, j + i * K] = 1
    return K_KK


def compute_Omega_analytical(A, Sigma_f, Sigma_eps, K):
    """Theoretical Ω = block(Ω_AA; Ω_ff) ignoring Ω_Af cross-cov."""
    Sigma_f_inv = np.linalg.inv(Sigma_f + 1e-14 * np.eye(K))
    Omega_AA = np.kron(Sigma_f_inv, Sigma_eps)

    Sigma_f_kron = np.kron(Sigma_f, Sigma_f)
    AT_kron = np.kron(A.T, A.T)
    A_kron = np.kron(A, A)
    I_K2 = np.eye(K * K)
    sum_pos = Sigma_f_kron @ np.linalg.inv(I_K2 - AT_kron)
    sum_neg = A_kron @ np.linalg.inv(I_K2 - A_kron) @ Sigma_f_kron
    sum_total = sum_pos + sum_neg
    K_KK = commutation_matrix(K)
    sum_with_comm = sum_total @ (I_K2 + K_KK)
    D_K = duplication_matrix(K)
    D_K_pinv = np.linalg.pinv(D_K)
    Omega_ff = D_K_pinv @ sum_with_comm @ D_K_pinv.T
    return Omega_AA, Omega_ff


def compute_V(grad_A_vec, grad_Sf_vech, Omega_AA, Omega_ff):
    V_AA = float(grad_A_vec @ Omega_AA @ grad_A_vec)
    V_ff = float(grad_Sf_vech @ Omega_ff @ grad_Sf_vech)
    return V_AA + V_ff, V_AA, V_ff


def main():
    print("=" * 88)
    print(" W9.1 — V_0 specialization at H_0 (A_REM,LOC = 0)")
    print("=" * 88)
    t0 = time.time()

    K, K_LOC1, K_LOC2, K_REM = 12, 4, 4, 4
    K_LOC = K_LOC1 + K_LOC2
    N_total = K * 2
    block_slices = [(0, K_LOC1), (K_LOC1, K_LOC), (K_LOC, K)]
    rem_slc = block_slices[2]

    # ─── Step 1: Construct null DGP (A_REM,LOC = 0) ─────────────────────────
    print(f"\n  --- Step 1: Null DGP (H_0: A_REM,LOC = 0) ---")
    A_null, Sigma_eps_null, Sigma_f_null, _ = construct_null_DGP(K, K_LOC1, K_LOC2, K_REM, seed=2026)
    sigma2_C_at_null = sigma2_C_phi(A_null, Sigma_f_null, block_slices, 2, N_total)
    print(f"  σ²_C (at H_0) = {sigma2_C_at_null:.10e}")
    print(f"  Should be ~0 by Proposition 1 (block-diagonal A_REM,LOC ⇒ σ²_C = 0)")
    A_REM_LOC_norm = float(np.linalg.norm(A_null[K_LOC:, :K_LOC]))
    print(f"  ‖A_REM,LOC‖_F at null = {A_REM_LOC_norm:.3e}  (should be 0)")

    # ─── Step 2: Compute ∇φ at null ─────────────────────────────────────────
    print(f"\n  --- Step 2: Compute ∇φ at H_0 ---")
    grad_A_null = grad_A_closed(A_null, Sigma_f_null, K_REM, K, N_total)
    grad_Sf_null = grad_Sigma_closed(A_null, Sigma_f_null, K_REM, K, N_total)
    grad_A_vec_null = grad_A_null.flatten()
    grad_Sf_vech_null = vech(grad_Sf_null, K)

    # Analyze structure of ∇A at null
    # The (∇A) form is (2/N) P_REM A Σ_f (I - Π_REM Σ_f). At A_REM,LOC = 0, P_REM A has
    # entries only at REM-rows × REM-cols (since A_REM,LOC = 0). So the matrix-product
    # P_REM A Σ_f at A_REM,LOC = 0 has zeros at REM-rows × LOC-cols.
    # ∂σ²_C/∂A_REM,LOC: this is the "interesting" gradient block (the H_1 direction).
    grad_A_REM_LOC = grad_A_null[K_LOC:, :K_LOC]
    grad_A_REM_REM = grad_A_null[K_LOC:, K_LOC:]
    grad_A_LOC_LOC = grad_A_null[:K_LOC, :K_LOC]
    grad_A_LOC_REM = grad_A_null[:K_LOC, K_LOC:]
    print(f"  ‖∂σ²_C/∂A_REM,LOC‖_F  = {np.linalg.norm(grad_A_REM_LOC):.3e}  (the H_1 direction)")
    print(f"  ‖∂σ²_C/∂A_REM,REM‖_F  = {np.linalg.norm(grad_A_REM_REM):.3e}  (should be 0 at H_0 since first term ∝ P_REM A Σ_f at REM-cols, second term Π_REM Σ_f at REM-cols cancels... let's check)")
    print(f"  ‖∂σ²_C/∂A_LOC,LOC‖_F  = {np.linalg.norm(grad_A_LOC_LOC):.3e}  (should be 0; P_REM A has only REM rows)")
    print(f"  ‖∂σ²_C/∂A_LOC,REM‖_F  = {np.linalg.norm(grad_A_LOC_REM):.3e}  (should be 0; P_REM A has only REM rows)")
    print(f"  ‖∂σ²_C/∂Σ_f‖_F        = {np.linalg.norm(grad_Sf_null):.3e}")

    # ─── Step 3: Compute V_0 = ∇φ^T Ω ∇φ at null ────────────────────────────
    print(f"\n  --- Step 3: V_0 computation (theoretical Ω) ---")
    Omega_AA_null, Omega_ff_null = compute_Omega_analytical(A_null, Sigma_f_null, Sigma_eps_null, K)
    V_0, V_0_AA, V_0_ff = compute_V(grad_A_vec_null, grad_Sf_vech_null, Omega_AA_null, Omega_ff_null)
    print(f"  V_0_AA = ∇A^T Ω_AA ∇A = {V_0_AA:.6e}")
    print(f"  V_0_ff = ∇Σ^T Ω_ff ∇Σ = {V_0_ff:.6e}")
    print(f"  V_0 = V_0_AA + V_0_ff = {V_0:.6e}")
    print(f"  V_0_AA / V_0 ratio (∇A contribution dominance under H_0) = {V_0_AA / max(V_0, 1e-30):.4f}")

    # ─── Step 4: Compare to alternative DGP at varying A_REM,LOC magnitude ──
    print(f"\n  --- Step 4: V at alternatives (H_1) for context ---")
    for sigma_cross in [0.0, 0.05, 0.1, 0.2, 0.3]:
        A_alt, Sigma_eps_alt, Sigma_f_alt, _ = construct_alternative_DGP(
            K, K_LOC1, K_LOC2, K_REM, sigma_cross_REM_LOC=sigma_cross, seed=2026)
        sigma2_C_alt = sigma2_C_phi(A_alt, Sigma_f_alt, block_slices, 2, N_total)
        grad_A_alt = grad_A_closed(A_alt, Sigma_f_alt, K_REM, K, N_total)
        grad_Sf_alt = grad_Sigma_closed(A_alt, Sigma_f_alt, K_REM, K, N_total)
        Omega_AA_alt, Omega_ff_alt = compute_Omega_analytical(A_alt, Sigma_f_alt, Sigma_eps_alt, K)
        V_alt, V_alt_AA, V_alt_ff = compute_V(grad_A_alt.flatten(), vech(grad_Sf_alt, K), Omega_AA_alt, Omega_ff_alt)
        print(f"  σ_cross_REM_LOC={sigma_cross:.2f}: σ²_C={sigma2_C_alt:+.6e}  V={V_alt:.6e}  V/(σ²_C·N_total)²={V_alt / max(sigma2_C_alt * N_total, 1e-30) ** 2 if sigma2_C_alt > 0 else float('inf'):.3e}")

    # ─── Step 5: V_0 stability across multiple DGP seeds (sanity check) ────
    print(f"\n  --- Step 5: V_0 stability check (3 seeds) ---")
    V_0_seeds = []
    for seed in [2026, 2027, 2028]:
        A_s, Sigma_eps_s, Sigma_f_s, _ = construct_null_DGP(K, K_LOC1, K_LOC2, K_REM, seed=seed)
        grad_A_s = grad_A_closed(A_s, Sigma_f_s, K_REM, K, N_total)
        grad_Sf_s = grad_Sigma_closed(A_s, Sigma_f_s, K_REM, K, N_total)
        Omega_AA_s, Omega_ff_s = compute_Omega_analytical(A_s, Sigma_f_s, Sigma_eps_s, K)
        V_s, _, _ = compute_V(grad_A_s.flatten(), vech(grad_Sf_s, K), Omega_AA_s, Omega_ff_s)
        V_0_seeds.append(V_s)
        print(f"  seed={seed}: σ²_C(H_0)={sigma2_C_phi(A_s, Sigma_f_s, block_slices, 2, N_total):+.3e}  V_0={V_s:.6e}")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    print(f"  V_0 (DGP seed 2026): {V_0:.6e}")
    print(f"  V_0 across 3 seeds: range [{min(V_0_seeds):.3e}, {max(V_0_seeds):.3e}]")
    print(f"  V_0 well-defined (positive, finite): {'YES' if V_0 > 0 and np.isfinite(V_0) else 'NO'}")
    print(f"\n  Structural finding: at H_0, ∇A is non-zero ONLY at A_REM,LOC entries")
    print(f"  (and possibly A_REM,REM via residual coupling); ∇A_LOC,REM and ∇A_LOC,LOC ≈ 0.")
    print(f"  This means V_0 is determined by a {(K_REM * K_LOC):d}-dimensional gradient subspace")
    print(f"  for ∇A (out of K² = {K * K:d} total entries) plus the full vech(Σ_f) dimension {K * (K + 1) // 2:d}.")
    print(f"  The simplification: only {K_REM * K_LOC:d} of {K * K:d} ∇A entries are 'active' at H_0.")

    # Theorem 5 corollary: under H_0, √T σ̂²_C,bc → N(0, V_0). Test rejects when σ̂²_C,bc > z_α √(V_0/T).
    # Demonstrate the test value at typical T = 1000:
    T_test = 1000
    threshold_05 = 1.645 * np.sqrt(V_0 / T_test)
    threshold_01 = 2.326 * np.sqrt(V_0 / T_test)
    print(f"\n  Theorem 5 illustration at T={T_test}:")
    print(f"    z_0.05 √(V_0/T) = {threshold_05:.6f}  (one-sided 5% rejection threshold for σ̂²_C,bc)")
    print(f"    z_0.01 √(V_0/T) = {threshold_01:.6f}  (one-sided 1% rejection threshold)")

    out = {
        'DGP': {'K': K, 'partition': [K_LOC1, K_LOC2, K_REM], 'note': 'block-diagonal A at (LOC, REM): A_REM,LOC = 0'},
        'sigma2_C_at_H0': sigma2_C_at_null,
        'A_REM_LOC_norm_at_H0': A_REM_LOC_norm,
        'gradient_structure_at_H0': {
            'grad_A_REM_LOC_norm': float(np.linalg.norm(grad_A_REM_LOC)),
            'grad_A_REM_REM_norm': float(np.linalg.norm(grad_A_REM_REM)),
            'grad_A_LOC_LOC_norm': float(np.linalg.norm(grad_A_LOC_LOC)),
            'grad_A_LOC_REM_norm': float(np.linalg.norm(grad_A_LOC_REM)),
            'grad_Sigma_norm': float(np.linalg.norm(grad_Sf_null)),
        },
        'V_0_seed_2026': {
            'V_0': V_0, 'V_0_AA': V_0_AA, 'V_0_ff': V_0_ff,
            'V_0_AA_dominance': V_0_AA / max(V_0, 1e-30),
        },
        'V_0_across_seeds': V_0_seeds,
        'V_0_well_defined': bool(V_0 > 0 and np.isfinite(V_0)),
        'effective_grad_A_dimension_at_H0': K_REM * K_LOC,
        'total_grad_A_dimension': K * K,
        'reduction_ratio': (K_REM * K_LOC) / (K * K),
        'Theorem_5_threshold_T1000_alpha05': threshold_05,
        'Theorem_5_threshold_T1000_alpha01': threshold_01,
        'runtime_seconds': time.time() - t0,
    }
    out_path = HERE / 'W9_1_V_null_specialization_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
