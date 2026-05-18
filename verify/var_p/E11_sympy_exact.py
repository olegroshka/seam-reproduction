"""E11 — Sympy exact verification of sigma^2_C^(p) closed form at small p, K.

Build a VAR(2) at K=3 with rational entries; solve companion-form Lyapunov for
exact rational Sigma_F; extract R(0), R(1), R(2) exactly; verify
   sigma^2_C^(p=2) [closed form] == sigma^2_C^(p=2) [BLP-direct]
to exact rational arithmetic.

Then verify the companion-form Theorem 1 application gives the same exact rational.
"""
from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import json
import time
from pathlib import Path

import sympy as sp
import numpy as np


def main():
    print("=" * 100)
    print(" E11 — Sympy exact verification of sigma^2_C^(p=2) closed form")
    print("=" * 100)

    # Test instance: K=3, K_LOC=2, K_REM=1, p=2 with small rational A_1, A_2, Sigma_eps
    K = 3
    K_LOC = 2
    K_REM = 1
    p = 2
    N_total = K_REM * 2

    A_1 = sp.Matrix([
        [sp.Rational(1, 4), sp.Rational(-1, 10), sp.Rational(1, 5)],
        [sp.Rational(1, 8), sp.Rational(1, 5), sp.Rational(-1, 10)],
        [sp.Rational(1, 10), sp.Rational(1, 8), sp.Rational(1, 4)],
    ])
    A_2 = sp.Matrix([
        [sp.Rational(1, 10), sp.Rational(0), sp.Rational(-1, 20)],
        [sp.Rational(1, 20), sp.Rational(1, 10), sp.Rational(0)],
        [sp.Rational(0), sp.Rational(1, 20), sp.Rational(1, 10)],
    ])
    Sigma_eps = sp.Matrix([
        [sp.Rational(1, 1), sp.Rational(1, 4), sp.Rational(1, 8)],
        [sp.Rational(1, 4), sp.Rational(1, 1), sp.Rational(1, 6)],
        [sp.Rational(1, 8), sp.Rational(1, 6), sp.Rational(1, 1)],
    ])

    # check stationarity (use numerical eigvals)
    A_1_np = np.array(A_1.tolist(), dtype=float)
    A_2_np = np.array(A_2.tolist(), dtype=float)
    Atilde_np = np.block([[A_1_np, A_2_np], [np.eye(K), np.zeros((K, K))]])
    rho = float(np.max(np.abs(np.linalg.eigvals(Atilde_np))))
    print(f"  Companion-form rho (numerical check): {rho:.6f}")
    if rho >= 1.0:
        print(f"  EXCEPTION: not stationary, rho >= 1")
        return 1

    print(f"\n  Building Sympy exact companion-form Sigma_F (6 x 6) ...")
    Kp = K * p
    Atilde = sp.zeros(Kp, Kp)
    Atilde[0:K, 0:K] = A_1
    Atilde[0:K, K:2*K] = A_2
    Atilde[K:2*K, 0:K] = sp.eye(K)

    Sigma_eps_comp = sp.zeros(Kp, Kp)
    Sigma_eps_comp[0:K, 0:K] = Sigma_eps

    # Solve vec(Sigma_F) = (I - A kron A)^{-1} vec(B Sigma_eps B^T)
    # In Sympy: build the linear system explicitly.
    # Use sympy.solve_linear_system or matrix algebra.
    # Approach: vec(Sigma_F) is Kp^2-dim. With Kp=6, system is 36-dim.
    t0 = time.time()
    I_K2p2 = sp.eye(Kp**2)
    AkronA = sp.zeros(Kp**2, Kp**2)
    # Use Kronecker product manually
    for i in range(Kp):
        for j in range(Kp):
            for k in range(Kp):
                for l in range(Kp):
                    AkronA[i*Kp+k, j*Kp+l] = Atilde[i, j] * Atilde[k, l]
    # vec(B Sigma_eps B^T) where B = (I_K, 0)
    BSeBt = Sigma_eps_comp
    vec_BSeBt = sp.Matrix([BSeBt[i % Kp, i // Kp] for i in range(Kp**2)])

    M = I_K2p2 - AkronA
    print(f"    inverting {Kp**2} x {Kp**2} system (slow in Sympy)...")
    vec_Sigma_F = M.solve(vec_BSeBt)
    print(f"    solved in {time.time()-t0:.1f}s")

    # Reshape to Kp x Kp
    Sigma_F = sp.zeros(Kp, Kp)
    for i in range(Kp):
        for j in range(Kp):
            Sigma_F[i, j] = vec_Sigma_F[j*Kp + i]
    # symmetrize
    Sigma_F = (Sigma_F + Sigma_F.T) / 2

    # Extract R(0), R(1)
    Sigma_f = Sigma_F[0:K, 0:K]
    R_1 = Sigma_F[0:K, K:2*K]
    # R(2) via Yule-Walker
    R_2 = A_1 * R_1 + A_2 * Sigma_f

    # Build C, Gamma
    rem = slice(K_LOC, K)
    C_REM = sp.Matrix.hstack(R_1[rem, rem], R_2[rem, rem])
    # Gamma 2x2 block-Toeplitz
    Gamma_REM = sp.Matrix.vstack(
        sp.Matrix.hstack(Sigma_f[rem, rem], R_1[rem, rem].T),
        sp.Matrix.hstack(R_1[rem, rem], Sigma_f[rem, rem]),
    )

    # closed form: tr[sum_h (A_h R(h)^T)_REM - C Gamma^{-1} C^T] / N_total
    joint_part = (A_1 * R_1.T + A_2 * R_2.T)[rem, rem]
    Schur_term = C_REM * Gamma_REM.inv() * C_REM.T
    sigma2_closed = (joint_part - Schur_term).trace() / N_total
    sigma2_closed_simp = sp.simplify(sigma2_closed)

    # BLP-direct: (Sigma_f)_REM - C Gamma^{-1} C^T - (Sigma_eps)_REM
    Sigma_eps_REM = Sigma_eps[rem, rem]
    sigma2_BLP = (Sigma_f[rem, rem] - Schur_term - Sigma_eps_REM).trace() / N_total
    sigma2_BLP_simp = sp.simplify(sigma2_BLP)

    residual = sp.simplify(sigma2_closed - sigma2_BLP)
    print(f"\n  sigma^2_C^(p=2) closed form (exact rational):")
    print(f"    {sigma2_closed_simp}")
    print(f"    numerical: {float(sigma2_closed_simp):.10f}")
    print(f"  sigma^2_C^(p=2) BLP-direct (exact rational):")
    print(f"    {sigma2_BLP_simp}")
    print(f"    numerical: {float(sigma2_BLP_simp):.10f}")
    print(f"  residual: {residual}")
    print(f"  V1 (closed == BLP exactly): {'PASS' if residual == 0 else 'FAIL'}")

    # Now companion-form Theorem 1: extract REM_comp indices
    rem_comp_idx = [K_LOC, K + K_LOC]  # K_REM=1, p=2, so REM at lag 0 = idx 2, lag 1 = idx 5
    Sigma_F_REM_comp = Sigma_F.extract(rem_comp_idx, rem_comp_idx)
    AtildeSigmaF = Atilde * Sigma_F
    block_11 = (AtildeSigmaF * Atilde.T).extract(rem_comp_idx, rem_comp_idx)
    block_21 = AtildeSigmaF.extract(rem_comp_idx, rem_comp_idx)
    Schur_comp = block_11 - block_21 * Sigma_F_REM_comp.inv() * block_21.T
    sigma2_companion = Schur_comp.trace() / N_total
    sigma2_companion_simp = sp.simplify(sigma2_companion)

    residual_V3 = sp.simplify(sigma2_closed - sigma2_companion)
    print(f"\n  sigma^2_C^(p=2) companion-form (exact rational):")
    print(f"    {sigma2_companion_simp}")
    print(f"    numerical: {float(sigma2_companion_simp):.10f}")
    print(f"  residual (closed - companion): {residual_V3}")
    print(f"  V3 (closed == companion exactly): {'PASS' if residual_V3 == 0 else 'FAIL'}")

    print("\n" + "=" * 100)
    print(" E11 SUMMARY")
    print("=" * 100)
    V1_pass = (residual == 0)
    V3_pass = (residual_V3 == 0)
    overall = V1_pass and V3_pass
    print(f"  V1 closed == BLP-direct (exact): {'PASS' if V1_pass else 'FAIL'}")
    print(f"  V3 closed == companion-form (exact): {'PASS' if V3_pass else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  Closed-form sigma^2_C^(p) certified exact at K=3, p=2 via Sympy rational arithmetic: " +
          ("YES" if overall else "NO"))

    out_path = Path(__file__).parent / 'E11_sympy_exact_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'overall_pass': bool(overall),
            'V1_closed_equals_BLP_exact': bool(V1_pass),
            'V3_closed_equals_companion_exact': bool(V3_pass),
            'sigma2_C_closed_form_rational': str(sigma2_closed_simp),
            'sigma2_C_companion_form_rational': str(sigma2_companion_simp),
            'sigma2_C_BLP_direct_rational': str(sigma2_BLP_simp),
            'sigma2_C_numerical': float(sigma2_closed_simp),
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
