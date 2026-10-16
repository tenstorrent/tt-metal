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

## Refinement 4 — Shrink the fill term: first send before a whole block is finished (overlap)
- **Date**: 2026-10-16
- **What was done**: Built the design's overlap lamp as a live schedule knob, `ENTRIES_PER_BLOCK = W` (`MRS_WAVES`). It is a schedule change. Each per-core block is cut into W waves along the streamed operand's axis: n for resident A / scatter -1, with a ragged last wave allowed; m for resident W / scatter -2, only when W divides core_m. Each wave is one entry: a full-K matmul of the wave, then its finish and its send.
  - **Host.** `expand_waves` turns block b into virtual blocks `b·W + w`, which carry b's roles and sit consecutively in b's place of each half's order. `plan_transport` / `entry_finish_plan` then re-derive the following, unchanged, from the expanded schedule: entries, zone slots, arrival thresholds, port units, lag decisions and hold depth.
  - **`wave_plan()`** is the single source of the per-wave descriptors `(row_off, col_off, rows, cols)`, the streamed window extents and the slot size P (the largest wave). It feeds `footprint`, `choose_k_block` (resident kb cap = `KB_MAX·W`, so a wave's window keeps the unsplit bytes and multicast rounds per block) and every runtime arg. The stream regime keeps W = 1.
  - **Kernels**, all five, as small deltas on the existing path:
    - Reader: wave offsets / valid extents in the DRAM reads.
    - Compute: per-wave `MatmulBlockShape` + subblock, `in1_per_core_w` = nominal window width, `out_row_width` = the wave's columns (dense wave tiles at the front of the slot).
    - Writer: per-wave output pages.
    - Port reader / sender: per-wave tile counts.
  - **`ARRIVAL_INC_PER_WAVE`** (default 1 = per wave; 0 = a block's last wave only).
  - **`LAG_POLICY = "fin"`** (lag every fin wave but the last; env only) and **`PORT_GROUP`** (env override) are perf knobs.
  - **Deadlock-freedom argument** for the wave-split order is written in `op_design.md` next to the schedule tables.
- **Perf** (traced, max over 4 chips, QB2 1x4 TORUS_XY, 2 links; default = W 1):

  | Case | W 1 (default) | Best W 2 variant |
  |---|---|---|
  | Focus | 124.0-125.1 µs | 123.4-123.8 µs (kb 32, fin lag, 6 samples), −0.5 µs |
  | GLM | 197.8-198.8 µs | 197.8-202 µs (kb 32); 198.0-198.9 µs with fin lag: neutral |
  | MiMo | 218.2-219.3 µs | 222 µs (kb 16), 230 µs (kb 8 / 32): regression |

  Because the gain does not hold across the profile (focus −0.5, GLM ±0, MiMo +3-11), the knob stays at its byte-identical default of 1. The default path re-measured unchanged: focus 124.1-124.9, GLM 197.9-198.7, MiMo 218.2-219.3 µs.

  Why (zones, chip 2):
  - **The fill is the resident operand's DRAM load, and wave 0 needs all of it.** Focus entry 0 takes 37 µs at W 1 vs 27.4 + 10.2 µs at W 2. GLM: 60 vs 41.3 + 20.5. MiMo: wave 0 waits for the whole 8.9 MB resident W, 48 µs against 23 µs of wave compute, so the fill grows by 14 µs. The first send does move earlier (focus 37 → 28 µs, GLM 60 → 42 µs), but the transport was not on the critical path at the start.
  - **The matmul itself gets slightly faster** with waves and kb 32: focus 23.7 vs 24.7 µs per block (subblocks 2x4 / 2x3 instead of 1x7). With kb 16 the doubled multicast rounds cost +4 µs (focus) and +15 µs (GLM).
  - **On focus the transport becomes the binding stage.** core_n 7 → 4 + 3 sends 8 + 6 tiles per core-block, which is 3 packets (7+1, 6) instead of 2. The ports are per-packet bound, ~0.17 µs per packet plus ~36 GB/s, so a block's send grows from ~21 to ~24 µs, which is above the 23.7 µs block compute. Fin wave 0 then waits up to 14 µs for the last-gathered arrivals, and one late core holds its whole W multicast line.
  - **Increments are not the bound.** One increment per block instead of per wave left the send units unchanged (10.5 / 13.7 µs per wave) and made wave-0 relays and fins wait for the second wave (focus 131 µs). Lagging the fin waves hides most of the straggler wait (focus 123.6), which is the −0.5 above.
  - Port batch 8: −0.8 µs at W 2, neutral at W 1.
- **Accuracy achieved**: unchanged at the default. Focus rel-RMS 0.00486, GLM 0.00487, MiMo 0.00487 (perf harness, max over devices' last check). The same values with W = 2 / 3 on all three cases.
- **Golden test progress** (targeted slices, EVAL_CAPTURE_L1=0):
  - `test_regression.py` 6/6.
  - `test_golden.py -k "loose or 640x2048_2048x7168"` 99/99.
  - `test_fabric_configs.py` 24/24, `test_ring_mock.py` 72/72.
  - Acceptance file: 38 line + 24 ring at the default. With `MRS_WAVES=2`: 38 + 24. With `MRS_WAVES=3` (ring): 24. With `MRS_WAVES=2 MRS_INC_PER_WAVE=0` (line): 38.
- **Issues encountered**:
  - The first W 2 build missed one `core_m → win_m` in the multicast sender plan (a static_assert caught it at compile time).
  - The positional deadlock assertion holds only for mirrored orders. Under the shared descending-key order (line), an arrival can be produced at the same entry index and is lagged, so the assertion stays mirror-only (as at W 1).
- **Tests added**: `test_every_partial_exactly_once_waves` (acceptance file): the exact-integer partial check with the wave split engaged (scatter -1 with 2 and 3 waves, ragged; scatter -2 with 2 waves), asserting `_LAST_PLAN["waves"] > 1`. `_LAST_PLAN` gained `waves`. Perf-only env knobs: `MRS_WAVES`, `MRS_INC_PER_WAVE`, `MRS_PORT_GROUP`, `MRS_LAG=fin`.

## Perf 1 — perf tournament round 1 (focus: Kimi K2.7 o_proj, 640x2048x7168, scatter -1)
- **Date**: 2026-10-16
- **Config**: the flagged LOOSE_CASES[0] exactly: A bf16 (1,1,640,2048), W bf8b (2048,7168), scatter -1, Ring, cluster_axis 1, 2 links, HiFi2, user `fp32_dest_acc_en=False`. The op's own fp32 full-sync DEST stays as is; no precision knob was touched.
- **Setup**: QB2 1x4 TORUS_XY, traced (`MRS_PERF_TRACE=1`). Op latency = max over the 4 chips of DEVICE KERNEL DURATION. The new permanent stage zones are on in every number below unless noted.

### Instrumentation (permanent)
- Restored `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`. `MaybeDeviceZoneScope` is `DeviceZoneScopedN`, so it is free when the profiler is off.
- The opt-in `MRS_ZONES` / `MRS_ZONE` macros in all 5 kernels became permanent `MaybeDeviceZoneScope` zones, and the `MRS_ZONES` define plumbing was removed from the descriptor.
- New zones: `RD_BARRIER` (the reader's read barrier, split out of `RD_RESV`), `WR_BARRIER` (the fin write barrier) and `PORT_GATHER_UNIT` (the port reader, per unit).
- Two per-core / per-chunk zones were removed because they could silently exhaust the 250-marker budget: `MBOX_WAIT` and `CHUNK_WAIT`. Max markers per RISC is now 140.
- Tools:
  - `tests/.../matmul_reduce_scatter/perf_zone_report.py`: per-zone and per-occurrence timeline, plus a marker-budget check.
  - `perf_summary.py`: per-case traced replay maxima.
  - `test_matmul_reduce_scatter_perf.py` gained `MRS_PERF_GUARD=1`, which adds 6 guard cells.

### Measured breakdown (focus, before)
- **Whole op**: 125.9-126.8 µs with zones on; 124.3 µs median with zones off.
- **Cumulative peel**:

  | Variant | µs |
  |---|---|
  | full | 126.8 |
  | no DRAM | 118.3 |
  | + no mcast (the compute floor) | 112.6 |
  | + no compute (transport only) | 59.9 |
  | compute only stubbed | 100.8 |

  The reader DM path alone takes ~100 µs, so compute and DM are close to balanced.
- **Matmul zone (MM)** per entry: 36.6 / 24.4 / 27.5 / 24.4 µs.
  - The pure compute is 24.4 µs per entry: 896 tile-mults per core at 36.8 cycles each, against a HiFi2 peak of ~32, so ~87% FPU utilization.
  - The gap is consistent with full-sync fp32 packing, since math stalls for each of the 56 per-window L1-acc packs.
- **Fill**: the step-0 reads landed by 6.5 µs, but the step-0 multicast ended at 12.7 µs.
- **Entry 2**: +3 µs, from a reader blocked in `land_arrivals_block` on a late arrival (ARR_WAIT max 63.5 vs median 57.6), which stalled its multicast lines.
- **Ranked bottleneck**:
  1. MM payload: 97.6 µs, ~13% headroom to FPU peak.
  2. Input-DM exposure: ~14 µs on top of the compute floor, mostly the fill.
  3. FIN adds plus write tail: ~10 µs.
  4. Transport: 60 µs alone, hidden.
- **Roofline gate**: the matmul has ~13% left before FPU peak. The DM fill has headroom: the first window (1.6 MB) needs about 4 µs at 400 GB/s, but takes 12.7 µs.

### Portfolio (hard cap: 2 experiments)
1. **`entry0_fill`** (reader): cut the entry-0 fill and the reader's per-step cost, including non-blocking arrival landing.
2. **`dest_resident_k`** (compute): amortize the full-sync fp32 pack by keeping each output subblock in DEST across K-windows.

Considered but not tested (cap):
- More matmul cores (the column strip). It measured 21.3 µs per entry in Refinement 2, but transport became binding.
- FIN / tail work.

### Verdicts
- **`entry0_fill` — WIN.** Focus 126.1-126.6 → 123.1-124.0 µs.
  - Cause: `ReceiverPipe::receive()` fuses the consumer-ready ack with the data wait. The walk is a → a2 → b → b2, so an A sender waited for W-reading receivers' DRAM barriers, and W waited for the slowest A column.
  - Winner: A receivers ack right after reserving their slot, and W keeps the fused order. Arrivals are landed by polling in every loader wait (`try_land`), so the loader never blocks on the fabric.
  - Nulls and regressions from the menu:

    | Variant | Focus µs |
    |---|---|
    | full 4-family split | 142-159 (concurrent A-column / W-row multicasts stalled in the NoC) |
    | per-operand split | 134.3 |
    | look-ahead after own send | 126.8-127.0 |
    | deferred step-0 W look-ahead | 124.8 (net loss) |
    | fast DRAM address table | null |
    | early ack for all families | 125.7 (noisy) |

  - Domain, standalone: GLM −5.0, MiMo −2.3, Linear −7.7..−11, 1 link −1.8 µs; stream regime flat.
- **`dest_resident_k` — WIN.** Subblock-outer / K-inner raw-LLK matmul over super-blocks of S windows.

  | Option | Compute floor µs | Focus whole op µs |
  |---|---|---|
  | helper (baseline) | 112.8 | 126.4 |
  | raw S=1 | 111.6 | 124.7 |
  | **raw S=2, cap 4** | 109.1 | 121.9-123.8 |
  | raw S=2, cap 3 | — | 133.3 (regression: no look-ahead room) |
  | raw S=4 (full K) | 106.4 | 124.4-125.9 |

  - S=4 also regressed Linear (+3..+6.6 µs) and bf16 W (+10.5 µs). The bursts consumed windows faster than the ~4.5 µs-per-window W stream could supply them, so S=2 was chosen.
  - The "late credit" reader variant was null.
  - Precision was identical in every option (rel-RMS 0.00486).

### What graduated (both; they touch disjoint stages)
- **One unqualified path for both changes. No predicate and no dual path. The replaced code was deleted.**
- **Reader** (`kernels/matmul_reduce_scatter_reader.cpp`):
  - split `rx_ack` / `rx_wait` with A acked early;
  - polled `try_land`, plus a final `ARR_WAIT` drain;
  - `land_arrivals_block` and the reader's lag handling were deleted. Compute still uses the lag bit.
- **Compute** (`kernels/matmul_reduce_scatter_compute.cpp`): the `matmul_block` helper call was replaced by the raw `matmul_entry`.
  - It packs once per super-block of `super_block` K-windows (CT arg 13).
  - Entry 0 always uses 1, because its windows arrive at DRAM pace.
  - It honors `SKIP_COMPUTE`, so `MRS_ABLATE=compute` still works.
- **Descriptor**: `super_block_plan()` sets `SUPER_BLOCK = 2` when `Kt/kb` is even and the streamed windows deepened to 4 still fit L1. Otherwise it uses S = 1 at the old depths. The S=1 fallback is the "cannot" case: an odd K-block count, or no L1 room.
  - `footprint()` takes a `depths` override.
  - `_LAST_PLAN` gains `super_block`.
- **No carve-outs.**
- **Docs**: `l1_ledger.md` (focus 837 → 956 KiB, MiMo 1328 KiB) and the `op_design.md` buffer-depth table were updated.

### Whole op, before → after (traced, median of 3 replays, max over chips, zones on in both)
| Cell (guard set: one per kernel path x placement) | before | after | Δ |
|---|---|---|---|
| **Focus** 640x2048x7168, ring, 2 links (resident A) | 126.0 | **119.3** | −6.7 (−5.3%) |
| GLM 640x4096x6144 (resident A) | 200.1 | 186.0 | −14.1 |
| MiMo 2048x2048x4096 sd -2, fp32acc (resident W) | 219.9 | 205.8 | −14.1 |
| Focus, Linear | 168.3 | 152.9 | −15.4 |
| Focus, 1 link | 155.7 | 151.3 | −4.4 |
| K3 FFN 640x8448x7168 (stream regime, kb 12) | 539.1 | 540.8 | +1.7 (+0.3%, noise) |
| 640x1536x2048 (one 2x2 subblock, S=1) | 55.8 | 54.5 | −1.3 |
| 640x1536x128 (grid 10x1, S=1) | 30.3-30.4 | 31.0-31.3 | +0.8 (+2.6%) |
| A bf8b x W bf16 640x2048x6144 | 138.6 | 137.6 | −1.0 |

- **Zones off** (shipping default): focus 124.3 → ≤119.3 µs, against a target of 124.8. The "after" column was measured with zones on, so this is conservative.
- **The narrow-N cell's +0.8 µs** reproduced in 3 sessions. It comes from the reader change: the new reader with the old compute gave 31.4 µs.
  - At 2.6% it is inside the ±3% noise band, so it is not a measured material regression and does not earn a carve-out. A carve-out would mean a second reader path.
  - It is recorded here for round 2.
- **Accuracy**: unchanged.
  - Precision baseline 6/6: focus rel-RMS 0.00454, MiMo 0.00455, A bf8b × W bf4b 0.00238.
  - Perf harness: rel-RMS 0.00486 / 0.00487 / 0.00487.
- **Golden** (EVAL_CAPTURE_L1=0):
  - `test_golden.py` 939/939 (run in shape slices), `test_regression.py` 6/6, `test_fabric_configs.py` 24/24, `test_ring_mock.py` 72/72.
  - Acceptance file: 41 + 12 + 12, run per placement group. That covers line, ring and test_ring linear, including `test_every_partial_exactly_once_waves` at S=2.
  - When the acceptance file runs as one session, 12 fixture errors appear (`SetFabricConfig ... while devices are still open`). The baseline shows the same errors, so they come from the harness, not the op.
- **SUPPORTED unchanged.**

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` | capability | The loop order is fixed K-block-outer / subblock-inner. It acquires and packs DEST per (K-block, subblock), so there is no mode that keeps an output subblock in DEST across several K-blocks and packs it once. It also cannot address a fronted multi-window region that the CB ring wrapped: the caller has to rewrite the UNPACK `fifo_rd_ptr` per window and pop the windows on the last subblock. | 112800 (focus compute floor); 126400 whole op | 109100 floor (S=2); 123500 whole op | `matmul_reduce_scatter_compute.cpp:99` (`matmul_entry`, raw `ckernel::matmul_block` at :150) |
| `dataflow_kernel_lib` mcast `ReceiverPipe::receive()` | capability | `receive()` fuses the consumer-ready ack (semaphore up to the sender) with the blocking data-valid wait. There is no split ack / wait API, so a receiver cannot ack as soon as its slot is reserved and wait later, and the wait cannot run a caller poll hook (here, landing fabric arrivals). | 126100-126600 (focus whole op) | 123100-124000 | `matmul_reduce_scatter_reader.cpp:44` (`rx_ack`), `:51` (`rx_wait`) |
