<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# `ttnn.experimental.quasar.unary_lut` — DFB generic-LUT activation op

The Metal-2.0 / Dataflow-Buffer (DFB) path for the Quasar generic-LUT eltwise
activation flow: a per-segment polynomial / rational LUT (with range reduction +
asymptotic factoring) baked into the compute kernel at JIT time from a
`LutConfig` (built from a tt-polynomial-fitter coefficient CSV — **zero
per-activation hardcoding**).

## Files (this op)

| File | Role |
|------|------|
| `unary_lut.{hpp,cpp}`, `unary_lut_nanobind.{hpp,cpp}` | host front-end + Python bindings (`ttnn.experimental.quasar.unary_lut`, `LutConfig`) |
| `device/unary_lut_metal_v2_factory.cpp` | DFB program factory + `make_lut_defines` (bakes the LUT into kernel `-D` defines) |
| `device/kernels_dfb/compute/unary_lut_sfpu.h` | the SFPU LUT evaluator (poly / rational / RR / asymptotic) |
| `device/kernels_dfb/compute/unary_lut_dfb.cpp`, `dataflow/{reader,writer}_sharded_dfb.cpp` | DFB compute + sharded reader/writer |

Sweep/driver: `tests/ttnn/unit_tests/operations/experimental/quasar/{dfb_lut_driver.py, test_dfb_sweep_60.py}` → `DFB_SWEEP_60.md`.

## Full documentation

This is **one of two feature-equivalent flows** (DFB + tt-llk). The canonical
reference — both flows, the shared data-driven feature set, the exhaustive bf16
validation regime, how each sweep runs, the current state, and the DFB-vs-tt-llk
comparison table — is:

**`tt_metal/tt-llk/tests/quasar_eltwise/QUASAR_ELTWISE.md`**
