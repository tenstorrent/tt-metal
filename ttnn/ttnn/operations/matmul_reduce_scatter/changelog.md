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

## Refinement 1 — Numerical formats: bfloat8_b activations + bfloat4_b weights
- **Date**: 2026-10-16
- **What was done**: Added `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `ttnn.bfloat4_b` to `SUPPORTED["weight_dtype"]`. This was a knob-turn with no kernel or descriptor change. The descriptor already sets `cb_act_k_window` / `cb_weight_k_window` format and page size from the tensor (`a.dtype`, `buffer_aligned_page_size()`), and `footprint()` / `choose_k_block` already take `tA` / `tW` from those page sizes. Checks the verifier asked for:
  - The bf4b page (576 B = 18·32) and the bf8b page (1088 B = 34·32) are 32 B multiples, so every window slot / part offset (`a_win_bytes`, `a_part_off`, `w_part_off`) stays NoC / L1 aligned.
  - `matmul_block` runs with its default `DataFormatReconfig::INPUT_AND_OUTPUT`, so `InitMode::Short` re-issues the unpack reconfig to the A / W formats after the fp32 / bf16 finish chains.
  - `choose_k_block` sees the smaller tiles. With a bf4b W, MiMo (2048x2048x4096, scatter -2) now fits resident-xblock (kb=16), which overlaps Refinement 3. The focus shape stays resident-A, kb=16, grid 10x8.
  - `compute_kernel_config` behaviour is unchanged.
- **Accuracy achieved** (1x4 TORUS_XY ring, HiFi2, fp32 reference over device-quantized inputs): focus 640x2048x7168 with A bf8b × W bf4b: PCC 1.000000, rel-RMS 0.00236, max_abs 0.051. MiMo 2048x2048x4096 with W bf4b: PCC 0.999885, rel-RMS 0.00468, max_abs 0.087. Prior cases unchanged (rel-RMS 0.0047).
- **Golden test progress**: all 624 former xfail cells (`dtype=bfloat8_b` ∪ `weight_dtype=bfloat4_b`, every shape, Linear and Ring, 1-2 links) pass, plus the LOOSE perf cases: 659 passed / 0 failed in the targeted slices (EVAL_CAPTURE_L1=0). EXCLUSIONS stays empty.
- **Issues encountered**: None.
- **Tests added**: `test_matmul_reduce_scatter_precision_baseline.py` has a new activation-dtype column and two cases (`focus_a_bf8b_w_bf4b`, `mimo_w_bf4b`), and it prints `_LAST_PLAN` (xblock / kb / grid) per case.
