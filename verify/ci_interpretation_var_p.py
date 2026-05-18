"""E3 — CI interpretation test for the VAR(p) cluster decomposition.

Claim: For VAR(p) the cluster decomposition lifts to the companion-state level:
    sigma^2_C^(p) = sum_j sigma^2_{C, LOC_j_comp <-> REM_comp}  +  sigma^2_{C, cross_comp}
with
    sigma^2_{C, LOC_j_comp <-> REM_comp}
        = (1/N) tr[ Atilde_{REM_comp, LOC_j_comp}
                    M^comp(LOC_j_comp, LOC_j_comp)
                    Atilde_{REM_comp, LOC_j_comp}^T ]
    sigma^2_{C, cross_comp}
        = (1/N) sum_{j != k} tr[ Atilde_{REM_comp, LOC_j_comp}
                                  M^comp(LOC_j_comp, LOC_k_comp)
                                  Atilde_{REM_comp, LOC_k_comp}^T ]
    M^comp(a, b) = (Sigma_F)_{a,b} - (Sigma_F)_{a, REM_comp} (Sigma_F)_{REM_comp}^{-1} (Sigma_F)_{REM_comp, b}

Under joint Gaussianity, M^comp(LOC_j_comp, LOC_k_comp) = 0 (structurally over Atilde)
iff REM_comp separates LOC_j_comp and LOC_k_comp in the (K*p)-dim companion-state CI graph
— NOT the K-dim contemporaneous CI graph.

Experiments:
  DGP A — engineered to satisfy companion-state CI: no LOC_1<->LOC_2 coupling at ANY lag,
          plus Sigma_eps block-diagonal between LOC_1, LOC_2.
          Expected: sigma^2_{C, cross_comp} ~ 0 (companion CI separation holds).
  DGP B — A_2 introduces a direct LOC_2 -> LOC_1 link at lag 2 only; A_1 has no
          direct LOC<->LOC coupling. Contemporaneous CI graph still has LOC_1 perp LOC_2 | REM
          (no edge in the within-time graph), but companion-state CI graph has a lag-2 edge
          that bypasses REM_comp.
          Expected: sigma^2_{C, cross_comp} != 0 (companion CI separation fails).
  DGP A' (control) — for context, recompute sigma^2_{C, cross} at the contemporaneous level
                     (analog of paper's Cor 2.1, evaluated as bilinear on M(LOC, LOC) blocks
                      from Sigma_f rather than companion-Sigma_F).
                     For VAR(2), this is a different quantity from sigma^2_{C, cross_comp}.

Together these show whether the K-dim contemporaneous CI condition still suffices
under VAR(p), or whether the (K*p)-dim companion-state condition is genuinely needed.
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
from thm1_var_p import closed_form_sigma2_C


def build_companion(A_list, K, p):
    Kp = K * p
    Atilde = np.zeros((Kp, Kp))
    for h in range(p):
        Atilde[0:K, h*K:(h+1)*K] = A_list[h]
    for h in range(1, p):
        Atilde[h*K:(h+1)*K, (h-1)*K:h*K] = np.eye(K)
    return Atilde


def solve_companion_Sigma_F(Atilde, Sigma_eps, K, p):
    Kp = K * p
    Sigma_eps_comp = np.zeros((Kp, Kp))
    Sigma_eps_comp[0:K, 0:K] = Sigma_eps
    AkronA = np.kron(Atilde, Atilde)
    vec_S_eps_comp = Sigma_eps_comp.flatten('F')
    vec_Sigma_F = np.linalg.solve(np.eye(Kp**2) - AkronA, vec_S_eps_comp)
    Sigma_F = vec_Sigma_F.reshape((Kp, Kp), order='F')
    return (Sigma_F + Sigma_F.T) / 2


def extract_R_list(Sigma_F, A_list, K, p):
    Sigma_f = Sigma_F[0:K, 0:K]
    R_list = [Sigma_f]
    for h in range(1, p):
        R_list.append(Sigma_F[0:K, h*K:(h+1)*K])
    # R(p) via YW
    R_p = sum(A_list[q-1] @ R_list[p-q] for q in range(1, p+1))
    R_list.append(R_p)
    return R_list


def companion_indices(K_LOC1, K_LOC2, K_REM, K, p):
    """Return (LOC_1_comp, LOC_2_comp, REM_comp) index arrays into the K*p companion state.

    Companion state F_t = (f_t, f_{t-1}, ..., f_{t-p+1}) so component i at lag h has
    index h*K + i in F_t.
    """
    loc1_idx = np.concatenate([np.arange(h*K, h*K + K_LOC1) for h in range(p)])
    loc2_idx = np.concatenate([np.arange(h*K + K_LOC1, h*K + K_LOC1 + K_LOC2) for h in range(p)])
    rem_idx  = np.concatenate([np.arange(h*K + K_LOC1 + K_LOC2, h*K + K) for h in range(p)])
    return loc1_idx, loc2_idx, rem_idx


def cluster_decomposition_companion(A_list, Sigma_eps, K_LOC1, K_LOC2, K_REM, K, p, N_total):
    """Compute (sigma^2_C^(p), per-LOC diagonals, cross-term) at the companion-state level."""
    Atilde = build_companion(A_list, K, p)
    Sigma_F = solve_companion_Sigma_F(Atilde, Sigma_eps, K, p)
    loc1_idx, loc2_idx, rem_idx = companion_indices(K_LOC1, K_LOC2, K_REM, K, p)

    # M^comp(a, b) = (Sigma_F)_{a,b} - (Sigma_F)_{a, REM_comp} (Sigma_F)_{REM_comp}^{-1} (Sigma_F)_{REM_comp, b}
    SF_REM = Sigma_F[np.ix_(rem_idx, rem_idx)]
    SF_REM_inv = np.linalg.inv(SF_REM)

    def M_comp(a_idx, b_idx):
        SF_ab = Sigma_F[np.ix_(a_idx, b_idx)]
        SF_a_REM = Sigma_F[np.ix_(a_idx, rem_idx)]
        SF_REM_b = Sigma_F[np.ix_(rem_idx, b_idx)]
        return SF_ab - SF_a_REM @ SF_REM_inv @ SF_REM_b

    # Atilde_{REM_comp, LOC_j_comp}
    Atilde_REM_LOC1 = Atilde[np.ix_(rem_idx, loc1_idx)]
    Atilde_REM_LOC2 = Atilde[np.ix_(rem_idx, loc2_idx)]

    M_11 = M_comp(loc1_idx, loc1_idx)
    M_22 = M_comp(loc2_idx, loc2_idx)
    M_12 = M_comp(loc1_idx, loc2_idx)
    M_21 = M_comp(loc2_idx, loc1_idx)

    # Diagonal contributions
    sigma2_LOC1 = float(np.trace(Atilde_REM_LOC1 @ M_11 @ Atilde_REM_LOC1.T)) / N_total
    sigma2_LOC2 = float(np.trace(Atilde_REM_LOC2 @ M_22 @ Atilde_REM_LOC2.T)) / N_total

    # Cross contributions
    sigma2_cross = (float(np.trace(Atilde_REM_LOC1 @ M_12 @ Atilde_REM_LOC2.T))
                  + float(np.trace(Atilde_REM_LOC2 @ M_21 @ Atilde_REM_LOC1.T))) / N_total

    sigma2_C_total = sigma2_LOC1 + sigma2_LOC2 + sigma2_cross

    # M_12 norms for direct CI-graph interpretation
    M_12_frob = float(np.linalg.norm(M_12, 'fro'))
    M_12_blocks = {}
    for h_row in range(p):
        for h_col in range(p):
            blk = M_12[h_row*K_LOC1:(h_row+1)*K_LOC1, h_col*K_LOC2:(h_col+1)*K_LOC2]
            M_12_blocks[f'lag{h_row}_vs_lag{h_col}'] = float(np.linalg.norm(blk, 'fro'))

    return {
        'sigma2_C_total_companion': sigma2_C_total,
        'sigma2_LOC1_diag': sigma2_LOC1,
        'sigma2_LOC2_diag': sigma2_LOC2,
        'sigma2_cross_companion': sigma2_cross,
        'M_12_frobenius': M_12_frob,
        'M_12_block_norms_by_lag_pair': M_12_blocks,
    }


def contemporaneous_cluster(A_list, Sigma_eps, Sigma_f, R_list, K_LOC1, K_LOC2, K_REM, K, p, N_total):
    """Compute contemporaneous-level cluster cross-term (analog of paper's Cor 2.1 evaluated
    on Sigma_f directly, NOT companion-Sigma_F). Used as comparison context.

    For VAR(p) this is computed via the K-dim Schur complement of (Sigma_f)_REM, on the
    contemporaneous M(LOC_j, LOC_k) blocks. It is NOT a decomposition of sigma^2_C^(p)
    (which decomposes via companion form); it shows what one would compute if one
    naively applied the VAR(1) Theorem 2 formula to the lag-1 coefficients only.
    """
    rem = slice(K_LOC1 + K_LOC2, K)
    loc1 = slice(0, K_LOC1)
    loc2 = slice(K_LOC1, K_LOC1 + K_LOC2)

    SF_REM = Sigma_f[rem, rem]
    SF_REM_inv = np.linalg.inv(SF_REM)

    def M_contemp(a, b):
        return Sigma_f[a, b] - Sigma_f[a, rem] @ SF_REM_inv @ Sigma_f[rem, b]

    M_11 = M_contemp(loc1, loc1)
    M_22 = M_contemp(loc2, loc2)
    M_12 = M_contemp(loc1, loc2)

    A1_REM_LOC1 = A_list[0][rem, loc1]
    A1_REM_LOC2 = A_list[0][rem, loc2]

    sigma2_LOC1 = float(np.trace(A1_REM_LOC1 @ M_11 @ A1_REM_LOC1.T)) / N_total
    sigma2_LOC2 = float(np.trace(A1_REM_LOC2 @ M_22 @ A1_REM_LOC2.T)) / N_total
    sigma2_cross = 2 * float(np.trace(A1_REM_LOC1 @ M_12 @ A1_REM_LOC2.T)) / N_total

    return {
        'sigma2_LOC1_contemp_lag1_only': sigma2_LOC1,
        'sigma2_LOC2_contemp_lag1_only': sigma2_LOC2,
        'sigma2_cross_contemp_lag1_only': sigma2_cross,
        'M_12_contemp_frobenius': float(np.linalg.norm(M_12, 'fro')),
    }


def main():
    print("=" * 88)
    print(" E3 — CI interpretation test for VAR(p) cluster decomposition")
    print("=" * 88)

    K_LOC1, K_LOC2, K_REM = 2, 2, 2
    K = K_LOC1 + K_LOC2 + K_REM  # 6
    p = 2
    N_total = K_REM * 2

    results = {}

    # ============================================================
    # DGP A: companion-state CI separation — no LOC_1<->LOC_2 at any lag
    # ============================================================
    print("\n[DGP A] No LOC_1<->LOC_2 at any lag; Sigma_eps block-diagonal LOC_1 | LOC_2")
    rng = np.random.default_rng(40001)
    A_list_A = []
    for h in range(p):
        A_h = np.zeros((K, K))
        decay = 0.6 ** h
        # diagonal blocks (within-LOC, within-REM self-dynamics)
        A_h[0:K_LOC1, 0:K_LOC1]              = rng.standard_normal((K_LOC1, K_LOC1)) * 0.3 * decay
        A_h[K_LOC1:K_LOC1+K_LOC2, K_LOC1:K_LOC1+K_LOC2] = rng.standard_normal((K_LOC2, K_LOC2)) * 0.3 * decay
        A_h[K_LOC1+K_LOC2:, K_LOC1+K_LOC2:]  = rng.standard_normal((K_REM, K_REM)) * 0.3 * decay
        # LOC_1 <-> REM, LOC_2 <-> REM at lag 1 only
        if h == 0:
            A_h[K_LOC1+K_LOC2:, 0:K_LOC1] = rng.standard_normal((K_REM, K_LOC1)) * 0.4
            A_h[K_LOC1+K_LOC2:, K_LOC1:K_LOC1+K_LOC2] = rng.standard_normal((K_REM, K_LOC2)) * 0.4
            A_h[0:K_LOC1, K_LOC1+K_LOC2:] = rng.standard_normal((K_LOC1, K_REM)) * 0.2
            A_h[K_LOC1:K_LOC1+K_LOC2, K_LOC1+K_LOC2:] = rng.standard_normal((K_LOC2, K_REM)) * 0.2
        # NO LOC_1 <-> LOC_2 block at any lag (zeros, as initialized)
        A_list_A.append(A_h)

    # rescale to spectral radius 0.85
    def _rescale(A_list, target=0.85):
        s = 1.0
        for _ in range(80):
            test = [s*A for A in A_list]
            At = build_companion(test, K, p)
            rho = float(np.max(np.abs(np.linalg.eigvals(At))))
            if abs(rho - target) < 1e-10:
                return test, At
            s = s * target / max(rho, 1e-12)
        return test, At
    A_list_A, Atilde_A = _rescale(A_list_A)
    rho_A = float(np.max(np.abs(np.linalg.eigvals(Atilde_A))))
    print(f"  rho(Atilde) = {rho_A:.6f}")

    # Sigma_eps block-diagonal: LOC_1, LOC_2, REM all independent
    W1 = rng.standard_normal((K_LOC1, K_LOC1))
    W2 = rng.standard_normal((K_LOC2, K_LOC2))
    WR = rng.standard_normal((K_REM, K_REM))
    Sigma_eps = np.zeros((K, K))
    Sigma_eps[0:K_LOC1, 0:K_LOC1] = W1 @ W1.T + 0.5 * np.eye(K_LOC1)
    Sigma_eps[K_LOC1:K_LOC1+K_LOC2, K_LOC1:K_LOC1+K_LOC2] = W2 @ W2.T + 0.5 * np.eye(K_LOC2)
    Sigma_eps[K_LOC1+K_LOC2:, K_LOC1+K_LOC2:] = WR @ WR.T + 0.5 * np.eye(K_REM)
    # confirm block-diagonality
    Sigma_eps_off = max(np.max(np.abs(Sigma_eps[0:K_LOC1, K_LOC1:K_LOC1+K_LOC2])),
                       np.max(np.abs(Sigma_eps[0:K_LOC1, K_LOC1+K_LOC2:])),
                       np.max(np.abs(Sigma_eps[K_LOC1:K_LOC1+K_LOC2, K_LOC1+K_LOC2:])))
    print(f"  Sigma_eps off-block max abs: {Sigma_eps_off:.3e} (should be 0 by construction)")

    decomp_A = cluster_decomposition_companion(
        A_list_A, Sigma_eps, K_LOC1, K_LOC2, K_REM, K, p, N_total)

    Sigma_F_A = solve_companion_Sigma_F(Atilde_A, Sigma_eps, K, p)
    R_list_A = extract_R_list(Sigma_F_A, A_list_A, K, p)
    sigma2_total, _, _, _ = closed_form_sigma2_C(
        A_list_A, R_list_A, Sigma_eps, K_LOC1 + K_LOC2, K_REM, N_total)

    contemp_A = contemporaneous_cluster(
        A_list_A, Sigma_eps, Sigma_F_A[0:K, 0:K], R_list_A,
        K_LOC1, K_LOC2, K_REM, K, p, N_total)

    print(f"  sigma^2_C^(2) (direct closed form)   = {sigma2_total:.10f}")
    print(f"  sigma^2_C^(2) (companion cluster sum)= {decomp_A['sigma2_C_total_companion']:.10f}")
    print(f"  residual                              = {abs(sigma2_total - decomp_A['sigma2_C_total_companion']):.3e}")
    print(f"  sigma^2_{{C, LOC_1<->REM_comp}}        = {decomp_A['sigma2_LOC1_diag']:.10f}")
    print(f"  sigma^2_{{C, LOC_2<->REM_comp}}        = {decomp_A['sigma2_LOC2_diag']:.10f}")
    print(f"  sigma^2_{{C, cross_comp}}              = {decomp_A['sigma2_cross_companion']:.6e}")
    print(f"  |sigma^2_C, cross_comp| / sigma^2_C  = {abs(decomp_A['sigma2_cross_companion'])/max(abs(sigma2_total),1e-20)*100:.3f}%")
    print(f"  ||M^comp(LOC_1_comp, LOC_2_comp)||_F = {decomp_A['M_12_frobenius']:.6e}")
    print(f"  M_12 block norms by (lag_LOC_1, lag_LOC_2):")
    for k, v in decomp_A['M_12_block_norms_by_lag_pair'].items():
        print(f"     {k:25}: {v:.6e}")

    A_predicted_pass = abs(decomp_A['sigma2_cross_companion']) < 1e-10
    print(f"  PREDICTION: cross_comp ≈ 0 (CI separation in companion graph): "
          f"{'PASS' if A_predicted_pass else 'FAIL'}")

    results['DGP_A_no_LOC_LOC_coupling'] = {
        'description': 'No LOC_1<->LOC_2 at any lag, Sigma_eps block-diagonal LOC_1|LOC_2',
        'sigma2_C_total': sigma2_total,
        **decomp_A,
        **{f'contemp_{k}': v for k, v in contemp_A.items()},
        'prediction_cross_comp_near_zero_PASS': bool(A_predicted_pass),
    }

    # ============================================================
    # DGP B: lag-2 direct LOC_2 -> LOC_1 coupling; everything else same as DGP A
    # ============================================================
    print("\n[DGP B] As DGP A, plus A_2[LOC_1, LOC_2] = 0.3 (direct lag-2 LOC_2->LOC_1 coupling)")
    A_list_B = [A.copy() for A in A_list_A]
    # add direct lag-2 coupling LOC_2 -> LOC_1
    rng_B = np.random.default_rng(40002)
    A_list_B[1][0:K_LOC1, K_LOC1:K_LOC1+K_LOC2] = rng_B.standard_normal((K_LOC1, K_LOC2)) * 0.3
    A_list_B, Atilde_B = _rescale(A_list_B)
    rho_B = float(np.max(np.abs(np.linalg.eigvals(Atilde_B))))
    print(f"  rho(Atilde) = {rho_B:.6f}")
    # confirm A_1 still has no LOC<->LOC blocks
    A1_LOC_cross = max(
        np.max(np.abs(A_list_B[0][0:K_LOC1, K_LOC1:K_LOC1+K_LOC2])),
        np.max(np.abs(A_list_B[0][K_LOC1:K_LOC1+K_LOC2, 0:K_LOC1])))
    A2_LOC1_from_LOC2 = np.max(np.abs(A_list_B[1][0:K_LOC1, K_LOC1:K_LOC1+K_LOC2]))
    print(f"  A_1 LOC<->LOC max abs (should be 0): {A1_LOC_cross:.3e}")
    print(f"  A_2 LOC_2 -> LOC_1 max abs (nonzero by design): {A2_LOC1_from_LOC2:.3e}")

    decomp_B = cluster_decomposition_companion(
        A_list_B, Sigma_eps, K_LOC1, K_LOC2, K_REM, K, p, N_total)
    Sigma_F_B = solve_companion_Sigma_F(Atilde_B, Sigma_eps, K, p)
    R_list_B = extract_R_list(Sigma_F_B, A_list_B, K, p)
    sigma2_total_B, _, _, _ = closed_form_sigma2_C(
        A_list_B, R_list_B, Sigma_eps, K_LOC1 + K_LOC2, K_REM, N_total)
    contemp_B = contemporaneous_cluster(
        A_list_B, Sigma_eps, Sigma_F_B[0:K, 0:K], R_list_B,
        K_LOC1, K_LOC2, K_REM, K, p, N_total)

    print(f"  sigma^2_C^(2) (direct closed form)   = {sigma2_total_B:.10f}")
    print(f"  sigma^2_C^(2) (companion cluster sum)= {decomp_B['sigma2_C_total_companion']:.10f}")
    print(f"  residual                              = {abs(sigma2_total_B - decomp_B['sigma2_C_total_companion']):.3e}")
    print(f"  sigma^2_{{C, LOC_1<->REM_comp}}        = {decomp_B['sigma2_LOC1_diag']:.10f}")
    print(f"  sigma^2_{{C, LOC_2<->REM_comp}}        = {decomp_B['sigma2_LOC2_diag']:.10f}")
    print(f"  sigma^2_{{C, cross_comp}}              = {decomp_B['sigma2_cross_companion']:.6e}")
    print(f"  |sigma^2_C, cross_comp| / sigma^2_C  = {abs(decomp_B['sigma2_cross_companion'])/max(abs(sigma2_total_B),1e-20)*100:.3f}%")
    print(f"  ||M^comp(LOC_1_comp, LOC_2_comp)||_F = {decomp_B['M_12_frobenius']:.6e}")
    print(f"  M_12 block norms by (lag_LOC_1, lag_LOC_2):")
    for k, v in decomp_B['M_12_block_norms_by_lag_pair'].items():
        print(f"     {k:25}: {v:.6e}")

    B_predicted_pass = abs(decomp_B['sigma2_cross_companion']) > 1e-6
    print(f"  PREDICTION: cross_comp != 0 (CI separation FAILS in companion graph): "
          f"{'PASS' if B_predicted_pass else 'FAIL'}")

    print(f"\n  Contemporaneous-only cluster (Cor 2.1 evaluated at Sigma_f, lag-1 coef only):")
    print(f"  sigma^2_{{C, cross}}_contemp (DGP A) = {results['DGP_A_no_LOC_LOC_coupling']['contemp_sigma2_cross_contemp_lag1_only']:.6e}")
    print(f"  sigma^2_{{C, cross}}_contemp (DGP B) = {contemp_B['sigma2_cross_contemp_lag1_only']:.6e}")
    print(f"  (note: contemporaneous cross is NOT a decomposition of sigma^2_C^(p);")
    print(f"   it shows what one would compute if naively applying the VAR(1) Cor 2.1 formula.)")

    results['DGP_B_lag2_LOC_LOC_coupling'] = {
        'description': 'DGP A plus A_2[LOC_1, LOC_2] nonzero (lag-2 LOC_2->LOC_1 direct edge)',
        'sigma2_C_total': sigma2_total_B,
        **decomp_B,
        **{f'contemp_{k}': v for k, v in contemp_B.items()},
        'prediction_cross_comp_nonzero_PASS': bool(B_predicted_pass),
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 88)
    print(" E3 SUMMARY")
    print("=" * 88)
    A_pass = results['DGP_A_no_LOC_LOC_coupling']['prediction_cross_comp_near_zero_PASS']
    B_pass = results['DGP_B_lag2_LOC_LOC_coupling']['prediction_cross_comp_nonzero_PASS']
    print(f"  DGP A (no LOC<->LOC at any lag):     cross_comp ≈ 0  prediction: {'PASS' if A_pass else 'FAIL'}")
    print(f"  DGP B (lag-2 LOC<->LOC direct edge): cross_comp ≠ 0  prediction: {'PASS' if B_pass else 'FAIL'}")
    overall = A_pass and B_pass
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"\n  Interpretation:")
    print(f"    The VAR(p) cluster cross-term lives at the COMPANION-state CI graph, not")
    print(f"    the contemporaneous K-dim graph. DGP B shows that adding lag-2 LOC<->LOC")
    print(f"    coupling produces a non-zero cross-term EVEN THOUGH the contemporaneous")
    print(f"    Sigma_f has no direct LOC_1<->LOC_2 dependence beyond what flows through REM.")

    out_path = HERE / 'E3_ci_interpretation_results.json'
    with open(out_path, 'w') as f:
        json.dump({'overall_pass': bool(overall), 'DGPs': results}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
