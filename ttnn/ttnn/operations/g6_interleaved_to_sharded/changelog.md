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
