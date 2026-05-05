"""VR1 — verify CORRECT ∂φ/∂Σ_f closed form against finite-diff.

Catches and fixes the algebraic error in submission §5.2 / W5 proof markdown:

WRONG (submission): -A^T Π_REM Σ_f A^T P_REM - P_REM A Σ_f Π_REM A
CORRECT (re-derived): -A^T P_REM A Σ_f Π_REM - Π_REM Σ_f A^T P_REM A

Both have first term A^T P_REM A and last term Π_REM Σ_f A^T P_REM A Σ_f Π_REM.
Only the middle two terms differ.

Verification strategy:
  1. Build VAR(1) DGP at K=8.
  2. Compute φ at the DGP (Theorem 1).
  3. Compute (∇A) and (∇Σ) via finite-diff.
  4. Compute proposed (∇A) closed form — should match finite-diff.
  5. Compute WRONG (∇Σ) closed form — should NOT match (validates the error claim).
  6. Compute CORRECT (∇Σ) closed form — should match finite-diff.

Outputs: 60_phase_D/VR1_grad_sigma_correction_results.json + run.log
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


def make_VAR1_DGP(K, K_LOC1, K_LOC2, K_REM, sigma_cross, seed):
    rng = np.random.default_rng(seed)
    K_LOC = K_LOC1 + K_LOC2

    def make_block_A(K_b, rng):
        Q1 = np.linalg.qr(rng.standard_normal((K_b, K_b)))[0]
        Q2 = np.linalg.qr(rng.standard_normal((K_b, K_b)))[0]
        diag = rng.uniform(0.4, 0.85, size=K_b)
        return Q1 @ np.diag(diag) @ Q2

    A = np.zeros((K, K))
    A[0:K_LOC1, 0:K_LOC1] = make_block_A(K_LOC1, rng)
    A[K_LOC1:K_LOC, K_LOC1:K_LOC] = make_block_A(K_LOC2, rng)
    A[K_LOC:, K_LOC:] = make_block_A(K_REM, rng)
    A[K_LOC:, 0:K_LOC] = rng.standard_normal((K_REM, K_LOC)) * sigma_cross
    A[0:K_LOC, K_LOC:] = rng.standard_normal((K_LOC, K_REM)) * sigma_cross * 0.5

    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    A = A * (0.85 / rho)
    W = rng.standard_normal((K, K))
    Sigma_eps = W @ W.T + 0.5 * np.eye(K)

    I_K2 = np.eye(K * K)
    AkronA = np.kron(A, A)
    vec_Seps = Sigma_eps.reshape(K * K, order='F')
    vec_Sf = np.linalg.solve(I_K2 - AkronA, vec_Seps)
    Sigma_f = vec_Sf.reshape((K, K), order='F')
    Sigma_f = (Sigma_f + Sigma_f.T) / 2

    return A, Sigma_eps, Sigma_f


def phi_unconstrained(A, Sigma, K_REM, K, N_total):
    """Theorem 1 closed form, treating Σ as unconstrained K×K (not necessarily symmetric)."""
    s, e = K - K_REM, K
    A_S = A @ Sigma
    A_S_AT = A_S @ A.T
    Sigma_AT = Sigma @ A.T
    Sigma_REM = Sigma[s:e, s:e]
    Sigma_REM_inv = np.linalg.inv(Sigma_REM + 1e-14 * np.eye(K_REM))
    DV = A_S_AT[s:e, s:e] - A_S[s:e, s:e] @ Sigma_REM_inv @ Sigma_AT[s:e, s:e]
    return float(np.trace(DV)) / N_total


def grad_A_FD(A, Sigma, K_REM, K, N_total, eps=1e-7):
    """Finite-diff ∂φ/∂A as K×K matrix."""
    G = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Ap = A.copy(); Ap[i, j] += eps
            Am = A.copy(); Am[i, j] -= eps
            G[i, j] = (phi_unconstrained(Ap, Sigma, K_REM, K, N_total)
                       - phi_unconstrained(Am, Sigma, K_REM, K, N_total)) / (2 * eps)
    return G


def grad_Sigma_FD_unsym(A, Sigma, K_REM, K, N_total, eps=1e-7):
    """Finite-diff ∂φ/∂Σ — Σ treated as UNCONSTRAINED K×K (no symmetry).

    Result is the unsymmetrized matrix gradient.
    """
    G = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Sp = Sigma.copy(); Sp[i, j] += eps
            Sm = Sigma.copy(); Sm[i, j] -= eps
            G[i, j] = (phi_unconstrained(A, Sp, K_REM, K, N_total)
                       - phi_unconstrained(A, Sm, K_REM, K, N_total)) / (2 * eps)
    return G


def grad_Sigma_FD_sym(A, Sigma, K_REM, K, N_total, eps=1e-7):
    """Finite-diff ∂φ/∂Σ — Σ constrained symmetric.

    For diagonal entries: perturb just Σ_{ii} by ε.
    For off-diagonal (i < j): perturb both Σ_{ij} and Σ_{ji} by ε simultaneously.

    Returns the K×K symmetric gradient matrix (vech form rolled out to full matrix).
    """
    G = np.zeros((K, K))
    for i in range(K):
        # Diagonal
        Sp = Sigma.copy(); Sp[i, i] += eps
        Sm = Sigma.copy(); Sm[i, i] -= eps
        G[i, i] = (phi_unconstrained(A, Sp, K_REM, K, N_total)
                   - phi_unconstrained(A, Sm, K_REM, K, N_total)) / (2 * eps)
    for i in range(K):
        for j in range(i + 1, K):
            Sp = Sigma.copy(); Sp[i, j] += eps; Sp[j, i] += eps
            Sm = Sigma.copy(); Sm[i, j] -= eps; Sm[j, i] -= eps
            # The directional derivative is M_unsym[i,j] + M_unsym[j,i] = 2 M_sym[i,j]
            dphi = (phi_unconstrained(A, Sp, K_REM, K, N_total)
                    - phi_unconstrained(A, Sm, K_REM, K, N_total)) / (2 * eps)
            # Convert to M_sym by halving (since M_sym = (M_unsym + M_unsym^T) / 2 has off-diag = M_unsym off-diag)
            G[i, j] = dphi / 2
            G[j, i] = dphi / 2
    return G


def grad_A_closed_form(A, Sigma_f, K_REM, K, N_total):
    """(∇A) closed form: (2/N) P_REM A Σ_f (I − Π_REM Σ_f)."""
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM))
    E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T
    return (2.0 / N_total) * P_REM @ A @ Sigma_f @ (np.eye(K) - Pi_REM @ Sigma_f)


def grad_Sigma_closed_form_WRONG(A, Sigma_f, K_REM, K, N_total):
    """WRONG version from submission §5.2:

    (1/N)[A^T P_REM A − A^T Π_REM Σ A^T P_REM − P_REM A Σ Π_REM A + Π_REM Σ A^T P_REM A Σ Π_REM]^(symm)
    """
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM))
    E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T

    M = (A.T @ P_REM @ A
         - A.T @ Pi_REM @ Sigma_f @ A.T @ P_REM
         - P_REM @ A @ Sigma_f @ Pi_REM @ A
         + Pi_REM @ Sigma_f @ A.T @ P_REM @ A @ Sigma_f @ Pi_REM)
    M_sym = (M + M.T) / 2
    return M_sym / N_total


def grad_Sigma_closed_form_CORRECT(A, Sigma_f, K_REM, K, N_total):
    """CORRECT re-derivation:

    (1/N)[A^T P_REM A − A^T P_REM A Σ_f Π_REM − Π_REM Σ_f A^T P_REM A + Π_REM Σ_f A^T P_REM A Σ_f Π_REM]

    (Already symmetric — middle terms are transposes; first and last symmetric.)
    """
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM))
    E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T

    M = (A.T @ P_REM @ A
         - A.T @ P_REM @ A @ Sigma_f @ Pi_REM
         - Pi_REM @ Sigma_f @ A.T @ P_REM @ A
         + Pi_REM @ Sigma_f @ A.T @ P_REM @ A @ Sigma_f @ Pi_REM)
    return M / N_total


def grad_Sigma_closed_form_CORRECT_unsym(A, Sigma_f, K_REM, K, N_total):
    """CORRECT unsymmetrized form for comparison with finite-diff unsymmetric:

    (1/N)[A^T P_REM A − 2 A^T P_REM A Σ_f Π_REM + Π_REM Σ_f A^T P_REM A Σ_f Π_REM]

    This is df₁ - df₂ where df₁ = A^T P_REM A and df₂_unsym = 2 A^T P_REM A Σ_f Π_REM
    - Π_REM Σ_f A^T P_REM A Σ_f Π_REM. (Term 1 of df₂ is A^T P_REM A Σ_f Π_REM with multiplicity
    2 from 2tr(H^-1 G^T dG); Term 2 of df₂ is -Π_REM Σ_f A^T P_REM A Σ_f Π_REM.)
    """
    K_LOC = K - K_REM
    E = np.zeros((K, K_REM))
    E[K_LOC:, :] = np.eye(K_REM)
    P_REM = E @ E.T
    Sigma_REM_inv = np.linalg.inv(Sigma_f[K_LOC:, K_LOC:] + 1e-14 * np.eye(K_REM))
    Pi_REM = E @ Sigma_REM_inv @ E.T

    M = (A.T @ P_REM @ A
         - 2.0 * A.T @ P_REM @ A @ Sigma_f @ Pi_REM
         + Pi_REM @ Sigma_f @ A.T @ P_REM @ A @ Sigma_f @ Pi_REM)
    return M / N_total


def main():
    print("=" * 88)
    print(" VR1 — verify ∂φ/∂Σ_f closed form against finite-diff")
    print("=" * 88)
    t0 = time.time()

    # DGP
    K, K_LOC1, K_LOC2, K_REM = 8, 2, 2, 4
    sigma_cross = 0.3
    A, Sigma_eps, Sigma_f = make_VAR1_DGP(K, K_LOC1, K_LOC2, K_REM, sigma_cross, seed=2026)
    N_total = 16

    print(f"  DGP: K={K}, partition (2, 2, 4), σ_cross={sigma_cross}, N_total={N_total}")
    print(f"  ρ(A) = {np.max(np.abs(np.linalg.eigvals(A))):.4f}")
    sigma2_C_true = phi_unconstrained(A, Sigma_f, K_REM, K, N_total)
    print(f"  σ²_C (Theorem 1): {sigma2_C_true:.10f}")

    # ─── (∇A) verification ───────────────────────────────────────────────────
    print(f"\n  --- (∇A) verification ---")
    grad_A_fd = grad_A_FD(A, Sigma_f, K_REM, K, N_total)
    grad_A_cf = grad_A_closed_form(A, Sigma_f, K_REM, K, N_total)
    diff_A = float(np.max(np.abs(grad_A_fd - grad_A_cf)))
    rel_diff_A = diff_A / float(np.max(np.abs(grad_A_fd)))
    print(f"  max |finite-diff − closed-form| = {diff_A:.3e}")
    print(f"  rel diff vs ‖finite-diff‖_∞ = {rel_diff_A:.3e}")
    pass_A = rel_diff_A < 1e-5
    print(f"  ✓ {'PASS' if pass_A else 'FAIL'}: (∇A) closed form {'matches' if pass_A else 'DIVERGES from'} finite-diff")

    # ─── (∇Σ) symmetric verification — WRONG vs CORRECT ──────────────────────
    print(f"\n  --- (∇Σ) symmetric verification ---")
    grad_Sf_fd_sym = grad_Sigma_FD_sym(A, Sigma_f, K_REM, K, N_total)
    grad_Sf_wrong = grad_Sigma_closed_form_WRONG(A, Sigma_f, K_REM, K, N_total)
    grad_Sf_correct = grad_Sigma_closed_form_CORRECT(A, Sigma_f, K_REM, K, N_total)

    fd_norm = float(np.max(np.abs(grad_Sf_fd_sym)))
    diff_wrong = float(np.max(np.abs(grad_Sf_fd_sym - grad_Sf_wrong)))
    diff_correct = float(np.max(np.abs(grad_Sf_fd_sym - grad_Sf_correct)))
    rel_wrong = diff_wrong / max(fd_norm, 1e-12)
    rel_correct = diff_correct / max(fd_norm, 1e-12)

    print(f"  ‖finite-diff (sym)‖_∞ = {fd_norm:.6f}")
    print(f"  max |finite-diff − WRONG closed-form|   = {diff_wrong:.3e}  (rel = {rel_wrong:.3e})")
    print(f"  max |finite-diff − CORRECT closed-form| = {diff_correct:.3e}  (rel = {rel_correct:.3e})")

    pass_correct = rel_correct < 1e-5
    fail_wrong = rel_wrong > 1e-3   # wrong should be OBVIOUSLY wrong
    print(f"  ✓ {'PASS' if pass_correct else 'FAIL'}: CORRECT (∇Σ) {'matches' if pass_correct else 'DIVERGES from'} finite-diff")
    print(f"  ✓ {'CONFIRMED ERROR' if fail_wrong else 'NO ERROR'}: WRONG (∇Σ) form from submission DIVERGES (as expected)")

    # ─── (∇Σ) unsymmetric verification ──────────────────────────────────────
    print(f"\n  --- (∇Σ) unsymmetric verification (sanity check on derivation) ---")
    grad_Sf_fd_unsym = grad_Sigma_FD_unsym(A, Sigma_f, K_REM, K, N_total)
    grad_Sf_correct_unsym = grad_Sigma_closed_form_CORRECT_unsym(A, Sigma_f, K_REM, K, N_total)
    diff_unsym = float(np.max(np.abs(grad_Sf_fd_unsym - grad_Sf_correct_unsym)))
    fd_unsym_norm = float(np.max(np.abs(grad_Sf_fd_unsym)))
    rel_unsym = diff_unsym / max(fd_unsym_norm, 1e-12)
    print(f"  ‖finite-diff (unsym)‖_∞ = {fd_unsym_norm:.6f}")
    print(f"  max |unsym FD − unsym CORRECT closed-form| = {diff_unsym:.3e}  (rel = {rel_unsym:.3e})")
    pass_unsym = rel_unsym < 1e-5
    print(f"  ✓ {'PASS' if pass_unsym else 'FAIL'}: unsymmetrized form match")

    # ─── Sample value table for inspection ──────────────────────────────────
    print(f"\n  Sample (∇Σ_f) entries comparison [diagonal + selected off-diag]:")
    print(f"    {'(i,j)':<10s}  {'finite-diff':>14s}  {'CORRECT':>14s}  {'WRONG':>14s}")
    for i, j in [(0, 0), (0, 4), (0, 5), (4, 4), (4, 5), (5, 5), (5, 7), (7, 7)]:
        print(f"    ({i},{j})       {grad_Sf_fd_sym[i, j]:>14.8f}  "
              f"{grad_Sf_correct[i, j]:>14.8f}  {grad_Sf_wrong[i, j]:>14.8f}")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    overall = pass_A and pass_correct and fail_wrong and pass_unsym
    print(f"  (∇A) closed form matches finite-diff: {'PASS' if pass_A else 'FAIL'}")
    print(f"  (∇Σ) CORRECT closed form matches finite-diff: {'PASS' if pass_correct else 'FAIL'}")
    print(f"  (∇Σ) WRONG submission form DIVERGES from finite-diff: {'CONFIRMED' if fail_wrong else 'UNCONFIRMED'}")
    print(f"  (∇Σ) unsymmetric closed form matches finite-diff: {'PASS' if pass_unsym else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  → submission §5.2 (∇Σ) form needs correction; CORRECT form replaces it.")

    out = {
        'DGP': {'K': K, 'partition': [K_LOC1, K_LOC2, K_REM], 'rho_A': float(np.max(np.abs(np.linalg.eigvals(A))))},
        'sigma2_C_true': sigma2_C_true,
        'grad_A_check': {
            'max_abs_diff_FD_vs_closed': diff_A, 'rel_diff': rel_diff_A, 'pass': pass_A,
        },
        'grad_Sigma_sym_check': {
            'fd_norm': fd_norm,
            'wrong_max_abs_diff': diff_wrong, 'wrong_rel_diff': rel_wrong, 'wrong_diverges_as_expected': fail_wrong,
            'correct_max_abs_diff': diff_correct, 'correct_rel_diff': rel_correct, 'correct_pass': pass_correct,
        },
        'grad_Sigma_unsym_check': {
            'unsym_max_abs_diff': diff_unsym, 'unsym_rel_diff': rel_unsym, 'pass': pass_unsym,
        },
        'overall_pass': bool(overall),
        'runtime_seconds': time.time() - t0,
    }
    out_path = HERE / 'VR1_grad_sigma_correction_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
