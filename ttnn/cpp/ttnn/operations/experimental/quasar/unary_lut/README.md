<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# `ttnn.experimental.quasar.unary_lut` — DFB generic-LUT activation op

`unary_lut` is the **Metal-2.0 / Dataflow-Buffer (DFB)** path for the Quasar
generic-LUT eltwise activation flow. It applies an embedded piecewise-LUT
activation (per-segment **polynomial** or **rational** `P(x)/Q(x)`, with optional
range reduction and asymptotic factoring) to a height/block-sharded bf16 L1 input,
entirely through the DFB framework (`QuasarDataMovementKernel` + `ProgramSpec` +
`dataflow_buffer_spec`). It is the path the Quasar team will use; the DFB add op
that unblocked it landed via PR #47739.

The LUT (boundaries + per-segment coefficients + range-reduction / asymptotic
metadata) is baked into the compute kernel at JIT time from a
`ttnn.experimental.quasar.LutConfig`. There is **zero per-activation hardcoding** —
every distinct activation is just a different `LutConfig` built from a
tt-polynomial-fitter coefficient CSV.

## Relationship to the tt-llk flow

This is **one of two feature-equivalent flows** that validate the SAME deployed
fitter picks (from `best.csv`):

1. **DFB flow (this op)** — Metal-2.0, `QuasarDataMovementKernel` + DFB, runs the
   real TTNN op on craq-sim Quasar. The path that ships.
2. **tt-llk flow** — standalone SFPI-compiled TRISC test kernels, slow-dispatch,
   in `/localdev/nkapre/tt-metal/tt_metal/tt-llk/tests/quasar_eltwise/`.

The two kernels are **separate source files** (no shared eval header). The feature
set was ported (duplicated) into tt-llk "for now" — tt-llk is being phased out in
favor of DFB, and a shared-eval-header refactor is the eventual clean architecture
but deliberately deferred. The canonical, cross-flow reference is
`tt_metal/tt-llk/tests/quasar_eltwise/QUASAR_ELTWISE.md`.

## Files

| File | Role |
|------|------|
| `unary_lut.{hpp,cpp}` | host front-end (`unary_lut(input, …, lut_config)`) |
| `unary_lut_nanobind.{hpp,cpp}` | Python bindings (`ttnn.experimental.quasar.unary_lut`, `LutConfig`) |
| `device/kernels_dfb/compute/unary_lut_dfb.cpp` | compute kernel entry (DFB tile loop) |
| `device/kernels_dfb/compute/unary_lut_sfpu.h` | the SFPU LUT evaluator (poly / rational / RR / asymptotic) |
| `device/kernels_dfb/dataflow/reader_sharded_dfb.cpp` | DFB sharded reader |
| `device/kernels_dfb/dataflow/writer_sharded_dfb.cpp` | DFB sharded writer |

Driver + sweep (drive any fitter CSV through the op):

| File | Role |
|------|------|
| `tests/ttnn/unit_tests/operations/experimental/quasar/dfb_lut_driver.py` | parses any fitter CSV → `LutConfig`, runs the DFB op, compares vs fitter `ground_truth` |
| `tests/ttnn/unit_tests/operations/experimental/quasar/test_dfb_sweep_60.py` | the 60-activation EXHAUSTIVE bf16 deployment sweep |
| `tests/ttnn/unit_tests/operations/experimental/quasar/DFB_SWEEP_60.md` | latest sweep results table |

## Eval-method coverage (all data-driven from the CSV)

The DFB SFPU evaluator (`unary_lut_sfpu.h`) implements the full feature set — there
is no per-activation branch; the selected behavior is entirely a function of the
`LutConfig` / CSV:

- **POLY_CASCADE** — piecewise polynomial, single- or multi-segment (boundary
  cascade, per-segment degree).
- **RATIONAL** — per-segment `P(x)/Q(x)` (iterative reciprocal, deferred reciprocal).
- **Range reduction** (all methods):
  - **Cody-Waite REDUCED_POLY** — `exp` / trig (reduce to a small domain, then a
    single reduced-domain polynomial, then reconstruct).
  - **EXPONENT_ALU** — `exp2` / `log2` / `log` / `log10` / `sigmoid` via direct
    exponent-field manipulation (`exman` / `exexp` / `setexp`).
  - **NEWTON_ROOT** — `sqrt` / `rsqrt` magic-seed + Newton iteration. Parameters
    are read from the CSV metadata (`newton_root_reciprocal` / `_n` / `_magic`);
    these distinguish `sqrt` from `rsqrt`. A consumer that lacks them marks the
    activation out-of-scope rather than hardcode.
- **Asymptotic factoring** — `f(x) = dominant(x) * correction_poly(x)` for
  `is_asymptotic` tail segments. Four dominant classes (`EXP_QUADRATIC`,
  `EXP_LINEAR`, `X_EXP_LINEAR`, `X`) reproducing `eval.py`'s `DOMINANT_FACTORS`
  (e.g. `gelu`'s left tail). Per-segment `LUT_ASYM_MASK` + `LUT_DOMINANT_CLASS`;
  asymptotic tails are evaluated properly and never dropped.

## Validation regime (EXHAUSTIVE bf16)

Inputs are **exhaustive**, not linspace/random: every distinct representable bf16
value in the activation's **full fit domain** `[lo, hi]` (≈33k values for `[-10,10]`
— bf16 is 16-bit, so this is cheap). Linspace under-reports near-zero ULP, so it is
not used.

Metrics, all measured on craq-sim (never assumed):

- **bf16 sign-magnitude bit-distance (Goldberg) ULP** — `max` / `mean` / `p99`.
  Near a root / zero-crossing the bf16 ordinals straddle 0, so a 1-bf16-ULP value
  error reads as a large bit-distance — a harmless artifact, so `ULP_max` (and,
  through it, `ULP_p99`) is **not** the headline gate.
- **`ml_pass`** — fraction of exhaustive bf16 inputs within a `1e-3` rel+abs Torch
  tolerance band (`atol = rtol = 1e-3`). This is the headline faithfulness gate
  (pure ULP explodes near output zeros).
- **PCC** vs the fitter `ground_truth`.

CLEAN gate: `PCC ≥ 0.99` **and** (`ml_pass ≥ 0.95` **or** bf16 `ULP_p99 ≤ 1.0`).
`ULP_mean` / `ULP_max` never gate.

## Current state — 58/60 CLEAN exhaustively

The latest sweep (`DFB_SWEEP_60.md`) is **58/60 CLEAN, 0 out-of-scope**. The 2 fails
are **fitter-side fundamentals, not kernel bugs**:

- **`hardshrink`** — a step discontinuity, unfittable by any smooth approximation.
- **`polygamma`** — a hard function; the best available fit is ~16 ULP.

The underlying craq-sim Quasar enablers that made the DFB path + `exponent_alu`
work: the vectored-`mtvec` fix (craq-sim `61695922`) and native `SFPEXMAN` /
`SFPDIVP2` (`b4358134`).

## How to run the exhaustive sweep

```bash
cd /localdev/nkapre/tt-metal-dfbport
export TT_METAL_HOME=$PWD ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
       TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
       PYTHONPATH=$PWD:$PWD/ttnn
/localdev/nkapre/tt-metal/python_env/bin/python -m pytest \
    tests/ttnn/unit_tests/operations/experimental/quasar/test_dfb_sweep_60.py -q
```

Each activation JIT-recompiles the kernel with its own coefficients — no full
rebuild. Sim runs are sequential (one sim at a time). For a single CSV, use
`dfb_lut_driver.py` directly. Branch: `dfb-lut-port` @ `c9aae9b8dcd`.

## DFB-vs-tt-llk exhaustive comparison table (TBD)

_Side-by-side per-activation bf16-ULP / `ml_pass` for the DFB flow and the tt-llk
flow — to be pasted here once the tt-llk exhaustive sweep finishes._
