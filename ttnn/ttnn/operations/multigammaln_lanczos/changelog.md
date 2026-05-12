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

## Refinement 1 — Reuse DST across lgamma iterations

- **Date**: 2026-05-12
- **What was done**: Eliminated the `cb_accumulator` L1 round-trip between the 4 Lanczos lgamma sub-evaluations. The global accumulator now lives in DST (D0) across the entire per-tile block, wrapped in a single `tile_regs_acquire`/`release` instead of Phase-0's 6 acquire/commit/release cycles per tile (1 init-zero + 4 lgamma updates + 1 finalize). `cb_accumulator` (intermediate CB at index 24) was removed from the program descriptor entirely; the kernel now uses only `cb_input_tiles` and `cb_output_tiles`. Per-core L1 footprint dropped from 24 KB → 16 KB.

  Slot-budget tactic for the 4-DST-slot fp32_dest_acc half-sync mode: with D0 (global accum), D1 (a), D2 (local lgamma sum), D3 (scratch) all in use, the `(a − 0.5) · log(a + 4.5)` multiply needs a 5th live value. We resolve this by corrupting D1 to `a − 0.5` for the one multiply, then reloading D1 from `cb_input_tiles` (held resident across all 4 iterations; popped only at end of tile) before pole zeroing. One extra `copy_tile` per iteration — negligible alongside the ~150 SFPU ops per tile.

- **Accuracy achieved** (4 shapes, same domain `[2.0, 10.0]` as Phase 0):

  | Shape | PCC | Max abs err | Mean abs err | Rel RMS err |
  |-------|-----|-------------|--------------|-------------|
  | `(1,1,32,32)` | 0.999999996 | 0.0048 | 0.00079 | 5.2e-5 |
  | `(1,1,64,64)` | 0.999999996 | 0.0051 | 0.00081 | 5.3e-5 |
  | `(1,1,256,256)` | 0.999999996 | 0.0052 | 0.00083 | 5.4e-5 |
  | `(2,4,64,128)` | 0.999999996 | 0.0052 | 0.00083 | 5.4e-5 |

  Numerically identical to the Phase-0 measured ceiling — no precision regression, as required by the refinement acceptance.

- **Golden test progress**: N/A — no golden suite exists for `multigammaln_lanczos` (not yet wired into `eval/golden_tests/`).

- **Issues encountered**: None. The kernel rewrite compiled and passed all 24 existing tests on the first attempt; precision baseline numbers were bit-identical to Phase 0 within the measured PCC.

- **Tests added**:
  - `test_multigammaln_lanczos_dst_reuse.py` — 8 cases pinning Refinement 1:
    - **Structural** (would catch a re-introduced `cb_accumulator` immediately):
      - `test_program_descriptor_has_no_cb_accumulator` — asserts `len(pd.cbs) == 2` and that CB index 24 is unused.
      - `test_compute_kernel_has_two_cb_compile_time_args` — asserts the compute kernel CT args are exactly `[0, 16]`.
      - `test_compute_config_unpack_to_dest_fp32_on_input_only` — asserts `cb_input_tiles` keeps `UnpackToDestFp32` and the legacy slot 24 is `Default`.
    - **Behavioural** (exercises the new code path):
      - `test_dst_resident_accumulator_matches_reference` (3 shapes) — same precision floor as Phase 0 baseline.
      - `test_d1_corrupt_and_reload_correctness` — stresses the D1 corrupt/reload contract at pole-hitting (`x ∈ {2.0, 2.5}`) and high-magnitude (`x = 10.0`) values; verifies pole zeroing fires on the correct reloaded `a`, not the corrupted `a − 0.5`.
      - `test_single_acquire_block_stress` — 65-tile multi-core shape with one long `tile_regs_acquire` per tile, verifies math/pack synchronisation at scale.

  Total: 32 tests (24 pre-existing + 8 new), all passing in both `--dev` and production modes.
