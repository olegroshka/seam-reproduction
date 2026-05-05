# seam-reproduction

Reproduction code for **"Asymptotic variance reduction at remainder blocks under joint VAR(1) over per-block AR(1)"** (Roshka, 2026).

The paper derives a closed-form expression $\sigma^2_C$ for the asymptotic forecast-error variance reduction at REMAINDER blocks of a partitioned VAR(1) under joint vs block-diagonal-restricted modeling, with a cluster decomposition along the LOCAL partition (Theorem 2), full sampling theory (Theorems 3–4, Proposition 6), and a Wald test for cross-block coupling (Theorem 5). Five named limiting regimes admit closed-form simplifications or bounds (Propositions 1–5).

This repo reproduces every numerical and symbolic verification cited in the paper.

## Layout

```
seam-reproduction/
├── paper/                  — paper PDF + LaTeX source
│   ├── seam_paper.pdf
│   └── seam_paper.tex
├── verify/                 — symbolic + numerical verifications
│   ├── thm1.py             — Theorem 1 (closed-form σ²_C), Sympy exact
│   ├── thm2.py             — Theorem 2 + Cor 2.1 + Lemma 1, Sympy + numpy
│   ├── props.py            — Propositions 1–5 (limiting cases), Sympy + numpy
│   ├── sampling.py         — Theorems 3 + 4 (consistency, asymptotic V), MC
│   ├── bias.py             — Proposition 6 (finite-sample bias + bc estimator)
│   ├── near_deg.py         — §5.4 / Appendix F (V ~ 1/λ_min⁴ scaling)
│   ├── grad_sigma.py       — VR1 (∇Σ) closed-form vs finite-diff cross-check
│   └── boundary.py         — W9.1 (V_0 = 0 at H_0; Theorem 5 boundary)
├── sim/                    — simulation suite (Appendix H, D1–D7)
│   ├── harness.py          — unified D1–D7 harness
│   └── results/            — cached JSON results from a clean run
├── requirements.txt
└── README.md
```

## Reproducing the paper's claims

Python 3.10+ recommended. Install dependencies:

```
pip install -r requirements.txt
```

### Verifications (`verify/`)

Each script is self-contained and prints PASS/FAIL plus a JSON results file next to it. Total runtime ~3–5 minutes on a laptop.

```
python verify/thm1.py        # Theorem 1 — Sympy exact at K ∈ {4, 6, 8}
python verify/thm2.py        # Theorem 2 — 3 Sympy instances + vanishing condition + numerical
python verify/props.py       # Propositions 1–5 — Sympy P1-P4 + numerical bound P5
python verify/sampling.py    # Theorems 3 + 4 — MC consistency rate + V coverage
python verify/bias.py        # Proposition 6 — finite-sample bias decomposition + bc check
python verify/near_deg.py    # V ~ 1/λ_min⁴ — analytical scaling + parametric V3.a
python verify/grad_sigma.py  # (∇Σ) closed form — VR1 finite-diff cross-check
python verify/boundary.py    # V_0 at H_0 — W9.1 boundary criticality (V_0 ≈ 1.5e-29)
```

### Simulation suite (`sim/`)

Reproduces Appendix H tables D1–D7. Outputs JSON per design under `sim/results/`. Total runtime ~37s on a laptop.

```
python sim/harness.py
```

Cached results from a clean run are already in `sim/results/`. To match the paper's reported numbers exactly, run with the same Python/numpy/scipy versions; minor MC noise is expected otherwise.

## What this repo does NOT contain

- The companion harp paper (forecasting-side empirical work) and its data — that lives in a separate repo at journal-submission time.
- Project-history files (drafts, referee correspondence, intermediate phase records). Only the artifacts cited by the published paper are here.

## License

MIT (see `LICENSE`).

## Citation

```
@article{Roshka2026SigmaC,
  author  = {Oleg Roshka},
  title   = {Asymptotic variance reduction at remainder blocks under joint VAR(1) over per-block AR(1)},
  year    = {2026},
  note    = {Working paper}
}
```
