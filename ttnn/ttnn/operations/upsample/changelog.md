# Changelog: upsample

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `stick_stream`: one `ttnn.generic_op` (Metal 2.0 `ProgramSpec`, Quasar) with a reader and a writer and no compute kernel; input sticks split over the full grid, each read once into a STRIDED DM→DM ring and written `sh·sw` times. Thread plan (R, C, W) = (2, 0, 4).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR], memory_config=[DRAM, L1], sharding=[interleaved], mode=[nearest], channel_alignment=[tile_aligned], scale=[uniform] (= the specification's PHASE0)
- **Accuracy achieved**: bit-exact — PCC=1.0, max_abs_err=0.0, rms_err=0.0 (measured on 4 shapes via test_upsample_precision_baseline.py, incl. the YOLOv8 calls `[1,20,20,512]` and `[1,40,40,512]` by (2,2))
- **Golden suite at Phase 0**: 26 / 26 supported cells passing; 142 xfail_expected, 84 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: no drift. Verifier added the missing R6 test (sharded input refused by name) and the DRAM half of R5. Verifier probe (`probes/probe_002.py`) showed float32, non-aligned C and per-axis factors already bit-exact on the RM path with the gate bypassed — SUPPORTED left at PHASE0, queued as Refinement 2.
- **Tests added**: test_upsample.py (harness acceptance), test_upsample_threads.py (thread plan, session order, regime predicate, R2, R5-L1), test_upsample_placement.py (R5-DRAM, R6), test_upsample_precision_baseline.py
- **Structural perf baseline** (for Refinement 3, `[1,40,40,512]` (2,2), no clock on ttsim): 50 sticks/cluster, `block_sticks` = 7, 8 blocks (last 1); max output pages per writer thread per cluster = 56 (writer threads 14/14/14/8 sticks); sticks read before a cluster's first write = 7; 1 input read per stick, 1 write per output page.
