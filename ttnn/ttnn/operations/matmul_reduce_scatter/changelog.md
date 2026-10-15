# Changelog: matmul_reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-10-15
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `generic_op` per call: a per-chip program with a 2D dual-multicast `matmul_block` grid that computes the G output blocks in transport order. Each finished block is handed to port cores that forward it over Fabric into the L1 arrival zones of the matching matmul core on the next chip, which adds its own partial in fp32 DEST. Ring (balanced / unbalanced) and line schedules, 1-2 links per hop, resident-xblock and stream-xblock regimes, L1 landing (zero partial bytes in DRAM).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat16, bfloat8_b], layout=[TILE], alignment=[tile_aligned], cluster_axis=[0, 1], scatter_dim=[-1, -2], topology=[Linear, Ring], num_links=[1, 2]
- **Accuracy achieved**: PCC ≥ 0.99988, max_abs_err ≤ 0.085, rel_rms_err = 0.0047, median 1 bf16 ULP (HiFi2; measured on 4 shapes on the 1x4 TORUS_XY ring via test_matmul_reduce_scatter_precision_baseline.py). HiFi4 gives rel-RMS 0.0024.
- **Golden suite at Phase 0**: 411 supported_pass, 624 xfail_expected, 6 regression tests passing; 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode (`verifier_report.json`, run with `EVAL_CAPTURE_L1=0`, see below).
- **Perf at Phase 0** (untraced, max over 4 chips, QB2 1x4 ring, 2 links): Kimi o_proj focus 142-144 µs (target 124.8), GLM o_proj 206-207 µs (target 207.3), MiMo o_proj 255-261 µs (target 220.6).
- **Issues encountered**:
  - Verifier DRY fix: the port reader restated the host's mailbox stride (16) as a literal, and sized its handoff table by a magic 256. Both are now CT args from the host (`MAILBOX_STRIDE`, `ceil(n_cores / L)`).
  - Verifier ledger fix: `l1_ledger.md` worked example updated to the implemented plan (row strip, P=14, kb=16, 80 cores, 893 KiB), plus the MiMo stream-regime example.
  - Tried `UnpackToDestFp32` on the fp32 hold CB: no measurable precision gain (the error is HiFi2 + bf16 output), so it was reverted.
  - Golden harness: with the default L1 graph capture, every executed cell fails with `peak_total=0`. The op's heterogeneous MeshWorkload allocates CBs on thread-pool workers, and graph processors are thread-local. This is a framework/harness limitation, escalated in `verification_report.md`; graded with `EVAL_CAPTURE_L1=0`.
  - After a `--dev` fabric run, the next mesh open can fail on an Ethernet heartbeat; `touch /tmp/tt-device.dirty` makes the wrapper reset first.
- **Tests added**: test_matmul_reduce_scatter.py (planner), test_matmul_reduce_scatter_precision_baseline.py, test_matmul_reduce_scatter_perf.py (LOOSE_CASES perf harness for `--profile`)
