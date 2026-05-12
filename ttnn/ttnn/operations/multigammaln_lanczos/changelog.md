# Changelog: multigammaln_lanczos

## Phase 0 — Core Implementation

- **Date**: 2026-05-12
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). The op fuses `torch.special.multigammaln(x, p=4)` into a single TTNN `generic_op` dispatch: a reader streams fp32 input tiles into `cb_input_tiles`, a single compute kernel runs 4 Lanczos-lgamma evaluations per tile (accumulating through `cb_accumulator`) plus the `3·log(π)` finalisation, and a writer drains the result back to DRAM. Hard-coded HiFi4 + fp32_dest_acc with per-CB `UnpackToDestFp32` for the two fp32 reload edges.
- **Verifier fix (one-line precision lever)**: Added `unpack_to_dest_mode[cb_input_tiles] = unpack_to_dest_mode[cb_accumulator] = UnpackToDestMode::UnpackToDestFp32` to the compute config in the program descriptor. Without this, the unpacker's SrcA/SrcB path truncates Float32 L1 reloads to TF32 (~10-bit mantissa) even when `fp32_dest_acc_en=True`. The fix is correct (no kernel changes needed; both reload sites already operate on Float32 CBs) and the numerical impact is large.
- **Accuracy achieved** (after the verifier fix, 4 shapes):

  | Shape | PCC | Max abs err | Mean abs err | Rel RMS err |
  |-------|-----|-------------|--------------|-------------|
  | `(1,1,32,32)` | 0.999999996 | 0.0048 | 0.00079 | 5.2e-5 |
  | `(1,1,64,64)` | 0.999999996 | 0.0051 | 0.00081 | 5.3e-5 |
  | `(1,1,256,256)` | 0.999999996 | 0.0052 | 0.00083 | 5.4e-5 |
  | `(2,4,64,128)` | 0.999999996 | 0.0052 | 0.00083 | 5.4e-5 |

  Pre-fix baseline (for reference): max abs ≈ 0.12–0.14, mean abs ≈ 0.031, rel RMS ≈ 1.7e-3 across the same shapes — i.e. the UnpackToDestFp32 fix delivered ~24× max-abs / ~39× mean-abs / ~32× RMS improvement.

- **Issues encountered**:
  - Numerical stability subagent flagged that the fp32 L1 round-trips (4 per output element through `cb_accumulator`, 4 input re-reads through `cb_input_tiles`) were silently truncating to TF32 because `unpack_to_dest_mode` was not configured. Fixed in the program descriptor (one-line addition, no kernel changes).
  - No correctness defects in the kernel itself — the 5-push/5-pop `cb_accumulator` cycle balances correctly, pole-zeroing operates on `a` rather than the result (avoiding the `NaN × 0` failure mode), and the unpacker/packer reconfigs cover every CB switch.

- **Tests added**:
  - `test_multigammaln_lanczos.py` — Phase 0 acceptance test (immutable, planner-supplied). 14 cases. All pass.
  - `test_multigammaln_lanczos_precision_baseline.py` — PCC ≥ 0.999 plus tight `max_abs < 0.05` and `rel_rms < 5e-4` regression gates over 4 shapes (`(1,1,32,32)`, `(1,1,64,64)`, `(1,1,256,256)`, `(2,4,64,128)`). 4 cases, all pass.
  - `test_multigammaln_lanczos_extended.py` — focused extra coverage: L1-interleaved input, constant inputs at the safe-domain pole (`a = 2.0`), near-pole (`a = 2.5`), upper edge (`a = 10.0`), linspace-over-safe-domain, and a `(1,1,32,2080)` shape that forces both `core_group_1` and `core_group_2` non-empty in `split_work_to_cores`. 6 cases, all pass.

  Total: 24 tests, 24 passing.

- **Companion analysis (already in this directory)**:
  - `numerical_stability.md` — error sources, accumulation strategy, fidelity sensitivity, pole-mask analysis. Identifies the UnpackToDestFp32 finding that the verifier acted on.
  - `data_transfer.md` — DRAM bandwidth, NoC channel balance, per-core L1 footprint, intra-core L1 traffic for the accumulator round-trip.
  - `capabilities.md` — what the op currently accepts (data formats, layouts, memory configs, etc.).
