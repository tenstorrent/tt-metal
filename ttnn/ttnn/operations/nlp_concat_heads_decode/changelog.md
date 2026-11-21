# Changelog: nlp_concat_heads_decode

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `pull_rows`: one Metal 2.0 `ProgramSpec`, one DM gather kernel per output core (one head per core on
  `num_cores_to_corerangeset(num_heads, grid, row_wise=True)`), tagged NoC reads of row `head_idx` of each user's
  input tiles straight into the local output shard, padding rows zeroed by a core-local iDMA. One
  `ttnn.generic_op` dispatch. Thread plan (R, C, W) = (1, 0, 0). Target quasar on ttsim.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[L1], sharding=[height], users=[1],
  alignment=[tile_aligned]
- **Accuracy achieved**: PCC=1.000, max_abs_err=0.0, rms_err=0.0 (relative), ULP p99=0, measured on 4 shapes
  ((1,1,32,32) h1, (1,1,32,128) h2/h32, (1,1,32,256) h32) via test_nlp_concat_heads_decode_precision_baseline.py
  (bit-exact, as R1 requires)
- **Golden suite at Phase 0**: 12 / 12 supported cells passing, 18 xfail_expected, 7 / 7 rejections, 0 loud
  categories (per `verifier_report.json`)
- **Issues encountered**: none failing. Verifier hardening: the kernel now `static_assert`s the host's
  thread-split invariants from the previously unused `reader_threads` CT arg; added a host-side R5 test (the exact
  `Target number of cores ...` message on a 2-core grid).
- **Tests added**: test_nlp_concat_heads_decode.py (generated acceptance), test_nlp_concat_heads_decode_plan.py
  (implementer: all PHASE0 INPUTS, every-reader-thread-ran, padding knob, session order, rejections; verifier: R5),
  test_nlp_concat_heads_decode_precision_baseline.py
