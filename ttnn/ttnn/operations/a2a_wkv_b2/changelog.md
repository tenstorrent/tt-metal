# Changelog: a2a_wkv_b2

## Phase 0 — Core Implementation
- **Date**: 2026-11-26
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` over a `MeshProgramDescriptor`.
  - Matmul-first: 96 heads over 96 worker cores per chip, with an in-kernel tilize of the ROW_MAJOR attn and `matmul_block` with W resident.
  - 4 fabric_tx streams per chip (neighbours plus an antipode half each way) scatter-write bf8_b results straight into each destination's L1-interleaved output.
  - Cross-call ready/done handshakes on cached `GlobalSemaphore`s.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR_LAYOUT], weight_dtype=[bfloat8_b], fp32_dest_acc_en=[False]. No shape-derived axes. This equals TARGET.
- **Accuracy achieved**: PCC=0.99991, max_abs_err=0.149–0.160 (N(0,1) inputs), rms_err (relative)=0.0177, measured on 5 input distributions via `test_a2a_wkv_b2_precision_baseline.py`.
  - The op has one legal shape, so the baseline varies seeds and magnitude ×0.01 / ×8.
  - Median got/true ratio is 1.013: Blackhole 16-bit-DEST rounding bias, not a scale bug. Stock `ttnn.matmul` shows 1.05.
- **Performance**: 131.6 µs per call (trace replay) against a SOL of 43.8 µs. The critical path is the fabric_tx sender path (about 1 µs per 4.3 KB packet). Workers finish at about 65 µs.
- **Golden suite at Phase 0**: 7 / 7 passing, with `EVAL_CAPTURE_L1=0`.
  - Per `verifier_report.json`: supported_pass 2, no_axes_found 5 (all passed), 0 in every loud category.
- **Issues encountered**:
  - Verifier DRY fixes:
    - the tx writer's `dst_addrs[4]` is now sized by `tiles_per_packet`, with a `static_assert` against `NOC_SCATTER_WRITE_MAX_CHUNKS`;
    - the worker reader's hard-coded `tile_h = 32` now comes from the host `TILE` CT arg.
  - Harness: the eval runner's default L1 capture (`EVAL_CAPTURE_L1=1`) disables the program cache for each test. That makes `test_trace_replay_matches_eager` fail at capture and hangs the device for the tests after it. The op passes the test outside that mode. Reported in `verification_report.md`.
- **Tests added**:
  - `test_a2a_wkv_b2_precision_baseline.py`;
  - `test_a2a_wkv_b2_trace_timing.py` (perf baseline);
  - probes `probes/test_probe_scale.py` (scale-bias triage) and `probes/test_probe_profile.py` (device-profiler run).
  - `test_a2a_wkv_b2.py` (acceptance) came from the planner.
