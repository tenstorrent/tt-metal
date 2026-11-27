# Changelog: g6_interleaved_to_sharded

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Owner-fetch on the shard grid:
  - each shard owner reads its tile pages from DRAM straight into its L1 shard;
  - one NoC request per tile, a 544 B valid-row span on partial tile rows;
  - one read barrier per issuer;
  - no CB, no compute kernel, no writer kernel;
  - one `generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], rank=[2], alignment=[h_non_aligned], out_scheme=[width_sharded]
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, rms_err=0, max ULP=0, got/true ratio exactly 1 (measured on 4 shapes via test_g6_interleaved_to_sharded_precision_baseline.py).
- **Golden suite at Phase 0**: 4 supported_pass, 2 xfail_expected (rank 4 / height → Refinement 1), 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verify_results/verifier_report.json`). 9/13 tests pass overall. `test_refuse_uncovered_shard_spec` fails only in the eval L1-profiling harness (`peak_total=0` on a refusal test), not in the op.
- **Perf**: FLUX #58 p1 at 1101 ns profiled device kernel duration (golden harness: 1106 ns; stock ttnn 1.43 µs). Per core: ~233 ns issuing 5 reads and ~326 ns DRAM round trip. On top of that, ~450 ns of kernel-start skew across the 32 cores, which is outside the op.
- **Issues encountered** (verifier fixes):
  - `validate()` now refuses a width shard taller / height shard wider than the tensor with `ValueError`. TensorSpec cannot allocate such a shard; before the fix the call failed with a TT_FATAL after caching a program.
  - The cache key uses `device.id()` instead of the Python wrapper's `id()`.
  - The shard-multiple check derives the tile shape from the tensor.
  - `describe_plan()` returns a deep copy.
  - Argument loading moved inside the `I2S-FETCH` zone.
  - `KNOB_VALUES["issuers_per_core"]` is derived from `ISSUERS`.
- **Tests added**: test_g6_interleaved_to_sharded.py (acceptance), test_g6_interleaved_to_sharded_knobs.py, test_g6_interleaved_to_sharded_perf.py, test_g6_interleaved_to_sharded_precision_baseline.py, test_g6_interleaved_to_sharded_validation.py

## Refinement 1 — Height-sharded placement and rank-4 inputs
- **Date**: 2026-11-27
- **What was done**: Knob-turn, as planned. `SUPPORTED` now holds `rank: [2, 4]` and `out_scheme: ["width_sharded", "height_sharded"]`. No kernel or descriptor change was needed:
  - `owner_block()` already gives height shard `s` tile rows `[s*sh, min((s+1)*sh, rows))` with all tile columns;
  - `compute_geometry()` already folds leading dims into tile rows at `ceil(H/32)` rows per image;
  - the reader picks the partial-row span by `r mod image_tile_rows`.

  The height path holds the Phase 0 perf bar: one block per issuer covering the whole shard, every read in flight, one barrier.
- **Accuracy achieved**: bit-exact (PCC=1.0, relative RMS=0, NaN padding poisoned in the knob cases) on:
  - height: `(1,1,1,128)` p2 on 1 core; `(2,1,33,128)` on 4 cores; `(2,3,33,128)` on a 4x2 COL_MAJOR grid (6 owners, 2 idle); `(3,1,1,256)` whose last shard is partial;
  - rank-4 width: `(1,1,1,5120)` on 8x4; `(2,1,33,256)` on 4 cores.
- **Golden test progress**: the two former xfail_expected cells pass (`test_op[1x1x1x128-height_sharded…]`, loose `flux57_p2`), as does regression `test_height_multi_shard`. Golden: 6/6 cells supported_pass. I ran only a `-k` slice: the 2 height cells plus `flux58_p1` as the width representative.
- **Issues encountered**: None.
- **Tests added**:
  - `test_g6_interleaved_to_sharded.py`: 4 `test_placement` cases (multi-image COL_MAJOR height, partial last height shard, 2 rank-4 width) and `test_plan_report_height_map` (shard `s` on `G[s]`, idle cores past the last shard, short last block);
  - `test_g6_interleaved_to_sharded_knobs.py`: the 4-core `(2,1,33,128)` height scenario, bit-exact on all 6 `issuers_per_core x partial_tile_read` values.
