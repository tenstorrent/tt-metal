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

## Refinement 2 — Speed up the PERF FOCUS profile (Kimi K2.7 o_proj)
- **Date**: 2026-10-16
- **What was done**: Measured first. Zones and ablations broke the focus case down as fill + 4 × matmul entry + finish adds + transport tail; the fixes below target those terms. Numbers are traced (`MRS_PERF_TRACE=1`), max over the 4 chips, QB2 1x4 TORUS_XY ring, 2 links:

  | Case | Before (µs) | After (µs) |
  |---|---|---|
  | Focus | 136.5-141 | 124.2-125.8 (target 124.8) |
  | GLM | 206-207 | 198.5 |
  | MiMo | 245-276 | 244-262 (no regression; noisy) |

  Untraced, chip 0 (MMIO) runs 8-25 µs longer than the others. Under trace it matches the other chips, so that skew was host launch skew, not kernel time.

  Levers, in order of gain:
  1. **Mirrored half orders** (`MIRROR_HALF_ORDERS`, host-only schedule change: `build_schedule` / `plan_transport` give `entries[h]` per half). The shared order sent a whole block one way in each middle entry, so the bwd ports finished at ~131 µs and the fwd ports at ~102 µs. Now the two halves walk the equal-distance blocks in opposite orders, so every entry sends half a block each way. The halves are split along the streamed operand's multicast lines (`half_split` n / m), so no mcast line spans two blocks. The design's deadlock argument is re-proved per half in `op_design.md` and asserted in `plan_transport`. It applies to ring_balanced with a resident regime only. The transport tail went from 131 to 115 µs, which is now under the last matmul.
  2. **Finish adds on the FPU** (`FIN_ADD_FPU`): `add<hold, arrival>` and, for add-both, `BinaryFpu` + `DestReuseBinary`. The fp32 hold reaches DEST through srcA exactly as it did through CopyTile in the SFPU chain, so precision is unchanged (focus rel-RMS 0.00454). A relay / fin add went from 4.5 to 1.2 µs per entry, and that time is serial on the TRISCs between matmul entries.
  3. **Streamed fin write** (writer): each tile row of the fin block is written as soon as the finish packs it. Before, the writer waited for all P tiles, and every core's DRAM write landed after the last add. About −2 µs.
  4. **Fill levers** (`READ_NOCS`, `SENDERS_PER_LINE`, `LOOKAHEAD_AFTER_BARRIER`, `MCAST_SENDER_PLACEMENT`, each now a per-regime dict). Entry 0 of the resident regime is DRAM-bound: 39 µs against 24.7 µs per later entry, and the `MRS_ABLATE=dram` stub brings it to 28 µs. The resident regime now reads W on NoC1 (the reader owns every read on both NoCs of its core) with 2 senders per line, issues look-ahead reads after the current step's barrier, and uses diagonal_disjoint senders. Together these are worth ~−3.5 µs. On the stream regime (MiMo) each of these measured neutral-to-worse, so it keeps the pre-refinement values.

  Measured and not kept as defaults (still live knobs):
  - kb 8 / 32 (131 / 134-139 µs traced; kb 16 stays).
  - `PREFETCH_STEPS` 2 (+4 µs).
  - Both operands on NoC1 (+8 µs).
  - Column strip (P = 12 on 100 cores cuts each matmul entry 24.7 → 21.3 µs, but one port lands far from its Ethernet core, so transport ends at ~150 µs; 150-160 µs in total, with "eth" or "spread" port placement).
- **Accuracy achieved**: focus PCC 0.999991, rel-RMS 0.00454 (gate 0.035), max_abs 0.066. MiMo PCC 0.999888, rel-RMS 0.00455. A bf8b × W bf4b rel-RMS 0.00238. Unchanged from Refinement 1, from `test_matmul_reduce_scatter_precision_baseline.py`.
- **Golden test progress**: targeted slices (EVAL_CAPTURE_L1=0):
  - `test_golden.py` loose cases + the 640x2048x7168 / 640x1536x32 / 2048x4096x4096 Linear cells: 123/123 pass.
  - `test_regression.py` 6/6, `test_fabric_configs.py` 24/24, `test_ring_mock.py` 72/72.
  - Acceptance file: 38 line + 24 ring pass. Precision baseline: 6/6.
- **Issues encountered**:
  - The traced harness needed a trace region on the mesh open; it is opt-in (`MRS_PERF_TRACE=1`) so the default run is unchanged.
  - The perf-only env switches grew by `MRS_KB_MAX`, `MRS_MIRROR`, `MRS_FIN_ADD_FPU`, `MRS_LOOKAHEAD_AFTER_BARRIER`, `MRS_A_READ_NOC`, `MRS_W_READ_NOC` and `MRS_ABLATE=mcast`. They follow the existing `MRS_*` convention; the defaults are the module constants.
- **Remaining headroom (finding)**: the focus case is now compute-bound (4 × 24.7 µs matmul entries, of which ~13% is fp32 full-sync pack time), plus an ~11 µs DRAM fill in entry 0 (6.5 MB at ~175-200 GB/s aggregate), ~3 µs of finishes and a ~3 µs fin-write tail.
- **Tests added**: `test_matmul_reduce_scatter_perf.py` gained a traced mode (`MRS_PERF_TRACE=1`). Reader ablation `MRS_ABLATE=mcast` (perf scaffolding).

## Refinement 3 — Speed up the rows-scattered MiMo-V2 profile (W residency L1 fit)
- **Date**: 2026-10-16
- **What was done**: Brought MiMo (2048x2048x4096, scatter -2, bf8b W) into the resident-xblock regime. These are knob-turns on the existing block surface, with no kernel change.
  - **Hold depth from the schedule.** `hold_depth()` derives the fp32 hold depth from the schedule's lag decisions: `DEPTH_HOLD` (2) if any entry of the walk lags, else 1. The lag rule moved into `entry_finish_plan()`, one source shared by the descriptor's runtime args and the footprint. The mirrored ring_balanced walks of the resident regime never lag, so they hold one slot.
  - **Fallback depth for the streamed operand.** `RESIDENT_STREAM_DEPTHS = (3, 2)`: `choose_k_block` tries every resident kb at depth 3, then at depth 2, before falling back to stream. It now returns `(resident, kb, depth)`; `window_depths()` is the single source for footprint and CB sizes.
  - **Plan order.** Both regimes' schedules (and hold depths) are planned before the regime choice.
  - **`_LAST_PLAN` additions**: `depths` and `hold_depth`.
  - Result: MiMo resident kb 8 / A depth 3 / hold 1 = 1296 KiB (budget 1363 KiB; before: 1488 KiB, which forced stream). W crosses DRAM once (48.2 → 21.5 MB).
- **Perf** (traced, max over 4 chips, QB2 1x4 TORUS_XY, 2 links):

  | Case | Before (µs) | After (µs) |
  |---|---|---|
  | MiMo | 243-256 | 218.4-218.9 (target 220.6) |
  | Focus | 124.2-125.8 | 124.1-125.8 (unchanged; still resident A, kb 16, depth 3, now hold 1) |
  | GLM | 198.5 | 197.9-199.2 |

  Also measured on MiMo:
  - kb 8 / depth 2 / hold 2 and kb 16 / depth 2 / hold 1: same within noise.
  - Senders 1, diagonal placement, look-ahead before the barrier, W on NoC0, prefetch 0 / 2: neutral or worse.
  - Unmirrored halves: +12 µs.
- **Bottleneck now (finding)**: compute-bound. The zones show:
  - entry 0 at 57.5 µs (W fill; its first K-block alone ~11 µs);
  - 3 × 45.9 µs matmul entries on 88 of 99 rectangle cores;
  - ~10 µs of serial finishes;
  - a ~7-13 µs fin-write tail.

  Ablations: compute alone 206 µs; everything stubbed (transport only) 113 µs.

  Next levers: a smaller first K-block, Refinement 4's sub-block sends, a ragged-m grid for the 11 idle cores.
- **Accuracy achieved**:
  - MiMo bf8b W: PCC 0.999888, rel-RMS 0.00455 (precision baseline) / 0.00487 (perf harness).
  - MiMo bf4b W: rel-RMS 0.00456.
  - Focus: rel-RMS 0.00486 (perf harness); A bf8b × W bf4b: 0.00238.
- **Golden test progress**: targeted slices (EVAL_CAPTURE_L1=0):
  - `test_golden.py -k "2048x2048_2048x4096 or test_op_loose"`: 99/99.
  - `test_regression.py` 6/6, `test_fabric_configs.py` 24/24, `test_ring_mock.py` 72/72.
  - Acceptance file: 38 line + 24 ring. Precision baseline: 6/6.
- **Issues encountered**: None.
- **Tests added**: the perf harness prints `_LAST_PLAN` and rel-RMS per case. New perf-only env knob: `MRS_RESIDENT_DEPTHS`.
