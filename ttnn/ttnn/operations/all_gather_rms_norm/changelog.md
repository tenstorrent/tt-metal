# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-07
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One
  fused `ttnn.generic_op` (one MeshProgramDescriptor) per call: ports (one per direction × link) relay raw input
  tiles over Fabric 2D straight into the L1 landing slots of the member cores (balanced ring / line, ready fence,
  one fused write + increment per item). Members (round-robin row groups of k slice-nested cores) compute Σx²,
  combine partials in-chip (symmetric k ≤ 8 / owner k > 8, ported from the base op `rms_norm`) and normalize as rows
  land. Regime N composes dense output tiles for W_local % 32 ≠ 0.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False], layout=[TILE],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma],
  gamma_dtype=[float32, bfloat16, none], gamma_layout=[TILE, ROW_MAJOR, none],
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2]. EXCLUSIONS: {float32, fp32_dest_acc_en=False}, {bfloat8_b, w_non_aligned}.
- **Accuracy achieved** (4 shapes × 4 precision corners, `test_all_gather_rms_norm_precision_baseline.py`):
  bf16 + fp32 DEST PCC ≥ 0.999997, max_abs ≤ 0.086, rel_rms ≤ 0.0038; bf16 + fp16 DEST PCC ≥ 0.999992,
  rel_rms ≤ 0.0062; fp32 PCC = 1.000000, max_abs ≤ 0.025, rel_rms ≤ 0.0012; bf8b PCC ≥ 0.99995, rel_rms ≤ 0.0099.
- **Golden suite at Phase 0**: 10,503 / 10,503 supported cells passing; 3,000 xfail_expected (the two EXCLUSIONS),
  180 infeasible_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`, run with
  `EVAL_CAPTURE_L1=0`, see verification_report.md).
- **Perf at Phase 0** (eval harness, fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 31.36 µs (target 10.1);
  640×512 41.51 µs (target 13.5); 640×1792 99.89 µs (target 47.3). Focus breakdown: TRANSPORT-only 26.22,
  LOCAL-only 14.73, empty 0.75 µs.
- **Issues encountered** (verifier fixes):
  - Correctness: race in the in-chip partial combine. A single cumulative partials semaphore let a fast peer's
    block-(b+1) increment complete a slow peer's block-b count, so a stale partial produced a wrong per-row
    inv_rms (intermittent `inf` / precision misses on the 2048-row golden cells after a program switch). Fixed with
    one semaphore per block parity (symmetric and owner combine). Reproducer: 3/20 bad → 0/40.
  - Per-stage device zones gated behind `AGRMS_PROFILE_ZONES=1`: they overflowed the perf harness's profiler buffers
    and inflated the headline number.
  - `GAMMA_READ_ALIGN` single-sourced (writer CT arg). `dst_full_sync_en` added to the program hash key. Ledger and
    design docs updated.
- **Tests added**: `test_all_gather_rms_norm_precision_baseline.py` (acceptance `test_all_gather_rms_norm.py` and
  `test_all_gather_rms_norm_debug.py` came from the planner / implementer).

## Refinement 1 — Native sharded placement: consume the input shard in place, pack into the output shard
- Date: 2026-11-07
- What was done: native shard-resident path for HEIGHT_SHARDED / WIDTH_SHARDED (scheme change: the shard grid fixes
  the split), ported from the base op's Refinements 2 / 4 and adapted to the fused landing.
  - HEIGHT (input and output share the row partition): one group per shard, k = 1, contiguous rows
    (`rows_per_group` in the port walk). `cb_x_in` (new CB 11) is backed on the input shard. Compute reads the
    own-slice columns from it for Σx² and the normalize segments, the regime-N compose reads them from it, and the
    ports send the own slice straight from it. No own-slice load and no NoC read of the core's own shard. Regime N
    zeroes the own tail tile's W padding in place: that is tile padding, not logical data.
  - WIDTH (full-height output shards, regime A): one group, k = output shard columns, member m = output shard core m.
    Own tiles are gathered from the input shards (cross-core reads: the input partition is not the output's). A
    ragged last member pushes pad pages so each row keeps the shard pitch.
  - Both: `cb_out_tiles` is backed on the output shard. Compute packs every row at its final place; the writer only
    waits / pops. Ports are placed off the member cores (`plan(avoid=...)`): the shard grid starts at core (0,0),
    where Phase 0 placed ports.
  - Remote slices still land in `cb_x_landing`, not in the output shard. The ports relay from the landing, so
    normalizing it in place would forward normalized data (the verifier note's "keep cb_x_landing" branch, with no copy
    at all: compute packs the landing straight into the shard).
  - Predicate (`native_shard_split` + `native_keeps_up`): TILE L1 shards, ROW_MAJOR orientation, the partitions above,
    the CBs fit L1, >= 2L free cores off the shard grid, and the shard-fixed split keeps up with the transport
    (per-core work ≤ `NATIVE_KEEP_UP_FRACTION` × T_xfer, the split's own cost model). Otherwise the sharded
    `TensorAccessor` fallback (Phase 0 path) runs. `AGRMS_NATIVE_SHARD=0` forces the fallback for A/B.
  - Reused: the landing / port item walk / combine / gamma paths, unchanged except a row-placement hook and an
    own-slice source switch. Added: `native_shard_split`, `native_keeps_up` / `transfer_time` / `combine_cost`
    (the cost model factored out of `select_split`), `cb_x_in`, the shard-backed `cb_out_tiles`, compute's
    `normalize_segment` / `pad_out_row`, and the writer's zero-copy store.
- Perf (fastest chip, QB2 1×4 ring L2, fallback → native): HEIGHT 640×384 41.41 → 34.93 µs; HEIGHT 2048×384
  93.41 → 68.21 µs. Kept on the fallback by the gate (natively slower): HEIGHT 128×512 (20.25 vs 27.60), WIDTH 64×512
  (14.28 vs 32.63; TRANSPORT-only 9.45 vs 21.15, LOCAL-only 6.66 vs 14.56: 64 one-tile members), WIDTH 32×4096 (24.92
  vs 32.71, k = 103). Perf cases: 31.46 / 41.70 / 100.02 µs (Phase 0 31.36 / 41.51 / 99.89, within noise).
- Accuracy achieved: unchanged numerics (same compute chain). Acceptance PCC gates (bf16 0.995, fp32 0.999, bf8b 0.99)
  met on every case.
- Golden test progress: targeted slices of sharded cells: 76/76 (one per shape × layout, final code) + 114/114 on a
  second random sample, plus 6 infeasible-L1 skips, all in Phase 0's infeasible HEIGHT shapes. The full suite is the
  harness's run.
- Issues encountered: native WIDTH loses on every golden geometry wider than a few tiles. Its shard grid gives
  one-tile members, which fragments the fabric packets and widens the combine, so the keep-up gate holds it to small
  shapes. Few-shard HEIGHT loses for the same reason (whole gathered rows on 1–4 cores).
- Tests added: `test_all_gather_rms_norm_native_shard.py` (6 cases: HEIGHT regime A / N, WIDTH symmetric / owner /
  ragged, each asserting the native placement was taken, plus a WIDTH case asserting the gate's fallback).
  `perf_ablation_all_gather_rms_norm.py` gains `AGRMS_PERF_SHAPE` / `AGRMS_PERF_MEMORY_LAYOUT` overrides for sharded A/B.

## Refinement 2 — bfloat8_b with W_local non-tile-aligned (regime N compose on a bf16 staging row)
- Date: 2026-11-07
- What was done: regime N on bfloat8_b. The transport still moves whole bfp8 tiles unchanged.
  - Reader: the slice-tail zero-fill is bfp8-aware (it skips the 64 B shared-exponent header and zeroes mantissa bytes,
    which gives exact zeros). The compose reads 2-byte datums from the bf16 staging row (all slices, own included) and
    writes bf16 `cb_x_aligned` tiles, stepping by the bf16 page.
  - Compute: `stage_row` copies (eltwise_chain `CopyTile` → `PackTile`, helper path) the row's landing tiles into
    `cb_x_stage` before normalizing that row. On native HEIGHT the own slice comes from `cb_x_in`. The normalize then
    reads bf16 `cb_x_aligned` and packs bfp8 into `cb_out_tiles` (the chain reconfigures the formats).
  - Host: `CB_X_STAGE = 12` (`STAGE_ROWS · land_w_tiles` bf16 pages, `STAGE_ROWS = 1`), `Sizes.T_stage`, and
    `cb_x_aligned` is Float16_b when staged. The new CB is in `cb_plan`, so the split's L1 fit check counts it. There is
    one new CT flag each for the reader and compute.
  - The `{bfloat8_b, w_non_aligned}` EXCLUSION is removed.
  - Reused: the regime N compose (`compose_tile`, now parameterized on datum size and page), the tail zero-fill, the
    landing / normalize chain. Added: `cb_x_stage`, compute's `stage_segment` / `stage_row`.
- Accuracy achieved: PCC ≥ 0.99 against the device-quantized input on (1,1,17,50), (1,1,64,1000), (4,128,47) and on every
  golden cell run. The column-exact debug test (monotonic per-device input) shows no bad columns.
- Golden test progress: bfp8 × w_non_aligned slice 252/252 supported cells pass (cluster_axis 1 / 2 links: 180; cluster_axis 0 /
  1 link, no-gamma + bf16 RM gamma: 72), across all six shapes and the INTERLEAVED / HEIGHT / WIDTH layouts. The rest
  of the slice is INVALID gamma skips.
- Issues encountered: the first run stepped `cb_x_aligned` by the bfp8 page (1088 B) instead of the bf16 page. The
  column debug test exposed it (every column past the first tile was shifted).
- Tests added: 3 bfp8 regime N cases in `test_all_gather_rms_norm.py`, and `test_bf8b_regime_n_columns` (2 shapes) in
  `test_all_gather_rms_norm_debug.py`.

## Refinement 3 — TRANSPORT: fill the packets (perf, focus case)
- Date: 2026-11-07
- What was done (TRANSPORT; three measured wins on the focus case, fastest chip, `eval.ccl_perf compare`):
  - Far slice of the balanced ring split between the two directions by **tile-row parity** instead of by columns.
    Each link carries its whole column run, so packets are 6 tiles instead of 3, and each port sends 30 packets
    instead of 40. Entries gained a row filter `rf` (packed in bits 31:30). Items with no row of their parity send
    nothing and carry no arrival increment (`Geom::items_before`). Landing notifications became a per-stream touch
    flag, and the host coverage check runs per row parity. Knob `FAR_ROW_SPLIT` (env `AGRMS_FAR_ROW_SPLIT=0` for A/B).
    31.47 → 29.38 µs.
  - Per-packet device zones showed the port sender was bound by its own RISC work, not the link: the send call is
    0.10 µs, the slot wait 0.03 µs, and the walk's address math ~0.25 µs per packet plus ~0.5 µs per item. The port
    walk now goes members outer and rows inner, with incremental (group, local-row) placement and one chunk split per
    member per item. 29.38 → 26.98 µs.
  - Host-precomputed member segments per send entry (`_segment_args`: member, landing offset, run length) replace
    the `ws[0..k]` boundary list in the port args. No per-item scan of k members. 26.98 → **25.08 µs**.
  - Re-swept with defaults kept: `relay_lag` 1/2/3 → 29.29 / 25.08 / 25.82; `item_min_chunks` 2/4/8 → 30.99 / 25.08 /
    26.37; `port_cb_bytes` 56/112/224 KiB (now env `AGRMS_PORT_CB_KIB`) → 31.29 (earlier tree) / 25.08 / 26.28.
  - Scatter packets not built: after the row split a packet is one member's 6-tile run, two runs exceed the 7-tile
    payload, and the item's fused increment ends a packet anyway.
  - Reused: the port reader / sender / walk, the schedule, the arrival counters. Added: the entry row filter,
    `items_before`, `Geom::advance`, `_segment_args`. No CB change, no SUPPORTED change.
- Perf (fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 31.47 → 25.08 µs (slowest 32.37 → 26.50); 640×512
  41.5 → 35.54 µs; 640×1792 99.9 → 85.96 µs. Focus TRANSPORT-only 26.37 → 19.81 µs, i.e. 1.47 MB received per chip
  at 74 GB/s. The `fabric_all_gather` example moves 256 tiles/chip on 2 links in ~18.3 µs, so the gap to it went from
  ~8 to ~1.5 µs. LOCAL-only 14.72 µs (unchanged).
- Still binding (finding, not a follow-up): the sender's per-packet RISC work (~0.25 µs above the send), the relay
  tail (the last relay item waits ~2 µs on its upstream), and the own-slice load before the first packet (~1.6 µs,
  Refinement 4's lamp).
- Accuracy achieved: unchanged numerics (transport moves the same bytes; compute untouched). Acceptance PCC gates
  (bf16 0.995, fp32 0.999, bf8b 0.99) met on every unit case; precision baseline test green.
- Golden test progress: guard slice of 144 cells (one per shape × topology × links × cluster_axis × memory_layout
  over 12 shapes): 38 passed, 5 xfailed (EXCLUSIONS), 101 skipped (INVALID / not buildable on this mesh), 0 failed.
  The three loose perf cases pass their golden check in `perf_measure.py`. Unit dir: 54 passed + native shard 6/6.
- Issues encountered: the full three-case perf run was killed once by host memory pressure (exit 137, other
  processes held 227 of 249 GB). The 640×1792 case was re-run alone.
- Tests added: 3 cases in `test_all_gather_rms_norm.py` for the row-split edges: `160x384-ring-L2-rowsplit_tail` and
  `96x384-ring-L1-rowsplit_tail` (a 1-row last item, one direction's far half empty), and `96x1792-ring-L2-rowsplit_xr1`
  (1-row items, every other far item empty).

## Refinement 4 — OVERLAP: shorten the exposed tail and co-tune the split around the ports (perf, flagged cases)
- Date: 2026-11-07
- What was done (OVERLAP; measured with `perf_measure.py`, fastest chip, `eval.ccl_perf compare`):
  - Breakdown first. Gated member zones (`AG_C_SUMSQ` / `AG_C_COMBINE` / `AG_C_NORM`, `AG_W_XCHG` / `AG_W_STORE`)
    show the exposed time is the chain after the **last landing**, not a transport slowed by the local work (the last
    landing happens at the same time as in TRANSPORT-only). Focus ~5 µs: ~1.5 exchange (the row's latest member gates
    it) + ~0.5 combine + ~1.3 normalize + ~1 store. 640×1792 ~11 µs: normalize of 28-tile members ~5.7 µs.
  - **x·γ pre-scale** in the combine wait (design lamp). Compute CT `prescale` (regime A, k > 1, gamma, round-robin
    split; `prescale_on`): `BinaryFpu` Mul with gamma row-broadcast from its compact pages → `cb_x_gamma` (CB 13,
    T_acc, one block), between the partial collapse and the combine. The normalize after the combine is one
    bcast-col multiply. Knob `AGRMS_PRESCALE=0` for A/B. Focus 25.09 → 24.57 µs, 640×1792 (k = 8) 81.47 → 78.41 µs.
  - **Packet-aware split objective** (`select_split`): the estimated end of the call, T_xfer(k) + tail(k)·310 ns.
    T_xfer(k) scales the byte term by `1 + 0.55·(packets(k)/packets_ref − 1)` (`packets_per_row`: link parts cut at
    member boundaries into run_pages chunks; `PACKET_TIME_SHARE` calibrated on 640×1792: TRANSPORT-only 76.2 µs at
    k = 12 vs 70.0 at k = 8). keep-up uses the same T_xfer(k). With equal packet counts the selection is unchanged.
    640×1792 L2 k = 12 owner → 8 symmetric (86.06 → 81.47 µs before the pre-scale). Off-focus checks of changed
    selections: 640×1792 L1 k = 28 → 16 (145.0 → 108.0 µs), 640×512 L1 k = 8 → 12 (47.30 → 47.02 µs).
  - Measured, defaults kept: `block_row_tiles` 2 (focus 28.84, 640×1792 89.57 µs; env `AGRMS_BLOCK_ROW_TILES`),
    `out_unit` cap 1 / 2 (25.00 / 25.20 vs 25.09; env `AGRMS_OUT_UNIT_CAP`), focus k = 4 / 12 (26.01 / 34.37),
    640×1792 k = 4 / 12-sym / 16-sym (89.2 / 87.2 / 85.2 vs 81.5 at k = 8). Per-row-group readiness: measured null
    (the ports' own-slice wait ends 4.2–5.1 µs, before their ready fence at 4.6–10.3 µs, on every chip), not built.
  - Not built: member-direct landing notifications (an atomic on every packet vs rule 4; only a port-notify hop on
    the tail), the compute lookahead (no effect on the last block), and the Linear line-end notify loop (not on the
    flagged cases, untouched).
  - Reused: the compute pipeline (sum_squares / collapse / combine helpers), the eltwise_chain BinaryFpu path, the
    cb_plan L1 fit check, the split candidate set. Added: `cb_x_gamma`, `prescale_gamma_block` /
    `normalize_prescaled`, `prescale_on`, `packets_per_row`, `PACKET_TIME_SHARE`.
- Perf (fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 25.09 → **24.57 µs** (slowest 26.53 → 25.88);
  640×512 35.55 → **34.20 µs** (37.45 → 36.45); 640×1792 86.06 → **78.38 µs** (89.63 → 81.46). Focus TRANSPORT-only
  19.81 µs (same build), so the exposed time went 5.27 → 4.76 µs.
- Still binding (finding): the last landed row's chain. Focus: ~1.5 µs land → exchange complete, then combine,
  normalize, store. 640×1792: Σx² + normalize of 28-tile members.
- Accuracy achieved: acceptance PCC gates (bf16 0.995, fp32 0.999, bf8b 0.99) met on every unit case. The precision
  baseline test is green: x·γ at T_acc keeps the fused chain's rounding count, also with 16-bit DEST
  (`test_all_gather_rms_norm_fp16_dest` passes).
- Golden test progress: slice of 176 cells (bf16, gamma bf16 RM, INTERLEAVED, both fp32_dest_acc_en, both cluster
  axes, L = 1 / 2, Linear (the ring cells are pruned at collection here), on the shapes whose split can change:
  640×512 / 640×896 / 640×1536 / 640×1792 / 1×640×1792, 2048×384 / 2048×1280, 128×4096, 4×1×512×512, 32×8192): 88 passed,
  88 skipped, 0 failed. The three loose perf cases pass their golden check in `perf_measure.py`. Unit dir:
  acceptance + precision 47 passed / 1 skipped, native shard + debug 13/13.
- Issues encountered: `test_split_pins` pinned the old rule's 640×1792 choice (k = 12 owner). It now pins k = 8 on
  2 links and k = 16 owner on 1 link. The owner combine stays numerically covered by the 32×1792 fp32 L1 and
  96×1792 L2 acceptance cases.
- Tests added: `test_split_pins` updated (4 pins + the prescale gate). `perf_ablation_all_gather_rms_norm.py` gains
  `AGRMS_PERF_NUM_LINKS`.

## Refinement 5 — Local work: cut the members' compute + combine time (perf, focus case)
- Date: 2026-11-07
- What was done (local work; measured with `perf_measure.py`, fastest chip, `eval.ccl_perf compare`):
  - Breakdown first (gated zones, the writer's exchange split into partial wait / write issue / flush / increments /
    peer wait). The LOCAL-only ablation and the full op's tail were both dominated by the symmetric exchange's SEND
    round: 1.2–2.9 µs per block, NoC-bound, because all 96 member cores ship k = 8 column-0 partials (faces 0/2,
    2 KiB at fp32) at the same moment — 16× the 128 B of row sums actually needed.
  - **Row-form symmetric partials** (knob `ROW_PARTIALS`, env `AGRMS_ROW_PARTIALS=0` for A/B; compute CT 12, writer
    CT 22): compute collapses with the owner path's `collapse_rows_to_row0` (ones-row × sumsq^T, row 0 = the 32 row
    sums), the writer sends two 64 B face rows into row `member_idx` of the row's gather tile on every peer, and every
    member runs `combine_rows_block<cb_inv_rms>` (REDUCE_COL + transpose_dest + finalize). The unused gather rows
    k..31 are zeroed once (`zero_unused_gather_rows`, now also for the row form). `cb_gathered_sumsq` shrinks k×.
    Focus 24.57 → 23.81 µs, LOCAL-only 13.16 → 11.56 µs.
  - Stateful one-packet writes (`noc_async_write_one_packet_set_state` per peer) for the 2·rows face-row writes:
    measured null (23.81 → 23.81; the slowest peer, not the issue rate, gates the wait), kept (correct, harmless).
  - Re-checked k with the cheaper exchange: k = 4 24.76, k = 12 symmetric 33.59 µs; k = 8 stays.
  - Not built: group-rectangle multicast (harvested columns differ per chip, one logical core set serves every chip);
    writer order "send partial(b+1) before store(b)" (helps LOCAL-only only; needs cb_out to hold a block).
  - Reused: `collapse_rows_to_row0`, `combine_rows_block` (templated on its output CB), `gather_row_offset`,
    `zero_unused_gather_rows`, the parity semaphores and lockstep slots. Added: `ROW_PARTIALS` / `row_partials_on`,
    one CT flag per kernel, the row-form send branch.
- Perf (fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 24.57 → **23.81 µs** (slowest 25.86 → 24.76);
  640×512 34.20 → **33.84 µs**; 640×1792 78.38 → **78.02 µs**. TRANSPORT-only 19.82 µs (unchanged), so the exposed
  time went 4.76 → 3.99 µs.
- Still binding (finding): the last landed row's chain (~4 µs): landing → partial ~0.85 µs of CB hand-offs, exchange
  ~1.1 µs (slowest member), combine ~0.65, normalize ~0.5, store ~1.3 µs (DRAM write latency at the tail).
- Accuracy achieved: acceptance PCC gates (bf16 0.995, fp32 0.999, bf8b 0.99) met on every unit case; precision
  baseline test green (15 passed, 1 skipped) — the row form uses the same FPU collapse / combine as the owner path.
- Golden test progress: slice of 60 cells (640×384, 1×1×32×2048, 1×1×64×1000, 2×4×128×512, 4×1×512×512; every dtype ×
  fp32_dest_acc_en; INTERLEAVED + WIDTH_SHARDED; cluster_axis 1, 2 links, gamma bf16 RM): 50 passed, 10 xfailed
  (the permanent fp32 / fp32_dest_acc_en=False EXCLUSION), 0 failed. The three loose perf cases pass their golden check
  in `perf_measure.py`. Unit dir: acceptance 32/32, precision 15 + 1 skipped, native shard 6/6, debug 7/7.
- Issues encountered: none.
- Tests added: none (the symmetric path is covered by the existing acceptance cases, now on the row form).

## Perf 1 — pre-send critical path (perf tournament, round 1)
- Date: 2026-11-13. Focus case: LOOSE_CASES[0], per-device (1,1,640,384) bf16 TILE INTERLEAVED, fp32_dest_acc_en=True,
  gamma bf16 ROW_MAJOR, Ring, 2 links, QB2 1×4 ring under FABRIC_2D_TORUS_XY. All numbers are from the eval CCL
  harness (`perf_measure.py` / the ablation probe, 5 experiments = 2500 calls, fastest chip, `eval.ccl_perf compare`).

### Measured breakdown (before)
- Whole op 23.81 µs (slowest 24.74). Ablations (`AGRMS_ABLATE`): TRANSPORT-only 19.81, LOCAL-only 11.56, EMPTY 0.78,
  NOREAD (port reader skips its NoC reads) 22.31. Exposed OVERLAP = 23.81 − 19.81 = 4.0 µs.
- Per-chip zone timeline (µs from kernel start, the same on all 4 chips within ~0.3 µs):
  - port sender setup: **AG_SETUP_HDRS 1.46 µs** (routing 4 packet headers, ~0.36 µs each), so the fence started at
    1.64 and ended at 2.8–3.2; the own-slice load ended at ~2.0 (member launch skew within a chip is up to 1.1 µs);
    first packet at ~2.9–3.5;
  - TRANSPORT streaming to the last member landing at ~20.1: 30 packets per port (20 own + 10 relayed), ~0.33–0.48
    µs per 12 KiB packet (~37 GB/s against 48.5 for the link); relay arrival waits of 0.9–1.5 µs from step 4 on; port
    sender done at 16.5–17.4;
  - OVERLAP tail after the last landing (4.25 µs): partial ready +0.83, exchange done +0.88, combine +0.6, normalize
    +0.5, store +1.35.
- Ranked: (1) TRANSPORT streaming (17 µs against a 7.6 µs link roofline), (2) the OVERLAP tail (4.25 µs), (3) the
  pre-send path (~2.9 µs, of which 1.46 µs was header routing in front of the fence).
- Instrumentation: the existing `AGRMS_ZONE` probes (compiled in only with `AGRMS_PROFILE_ZONES=1`, so the macro costs
  nothing when off) are extended with `AG_SETUP_HDR0`, `AG_OPEN`, `AG_READY_OUT`, `AG_SETUP_REST`, `AG_READY_WAIT`,
  `AG_SEND`, `AG_CLOSE` (port sender), `AG_INPUT_ITEM`, `AG_FINAL_ARRIVAL`, `AG_FINAL_NOTIFY` (port reader).
  `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree, so the op's
  own gated macro is used. With every zone on, the 500-call harness overflows the port cores' profiler buffers
  ("programs per chip 570, expected 600"). For a timeline, run 1 experiment and read the first, untraced call.

### Portfolio (hard cap: 2 experiments)
1. **TRANSPORT: far slice by fabric multicast** (2D line mcast, write-and-forward to p±1 and p±2) instead of the
   store-and-forward relay: 30 → 20 packets per port, no relay reads or relay waits.
2. **TRANSPORT handshake + OVERLAP data availability: shorten the pre-send critical path**: (a) route one header and
   copy it, (b) send the first own items without waiting for the members' own-slice load, (c) send the ready before
   the rest of the setup.

### Verdicts
| idea | verdict | focus fastest / slowest (µs) | domain |
|---|---|---|---|
| far-slice fabric mcast | **BLOCKED** | — (hangs) | the QB2 1×4 ring mock is a snake over the 2×2 torus. The mesh positions 0..3 are fabric nodes D0, D1, D3, D2, and the ring turns a corner at every hop (E, S, W, N). A 2-hop line mcast in one direction never reaches p+2. A 2D mcast (N/S spine, E/W branch) covers only half the chips per direction. A 2-hop unicast injected on the 1-hop link is misrouted. Three runs hung in the fence or in the slot wait (`perf_experiments/far_slice_fabric_mcast/evidence/`). Expressible only on a straight ring axis (Galaxy 8×4 TP axis), which is untested here. A QB2-specific variant (whole slice as a spine+branch mcast, unicast to the E/W neighbour) is designed but not built: it raises the E/W link load from 1.5 to 2 slices (link floor 7.6 → 10.1 µs). |
| pre-send critical path | **WIN** | 23.80 / 24.76 → **22.83 / 24.03** | applies everywhere, no exceptions found |

Pre-send menu (focus, fastest / slowest µs; same precision everywhere):
| variant | fastest | slowest |
|---|---|---|
| baseline | 23.80 | 24.76 |
| A: route one header, copy the rest | 23.74 | 24.70 |
| C: ready out before the count / copies | 23.70 | 24.75 |
| D2: first 2 own items read from the input tensor | 23.64 | 24.56 |
| A+D2 / C+D2 | 23.26 / 22.97 | 24.18 / 24.02 |
| **A+C+D2 (graduated)** | **22.83** | **24.03** |
| A+C+D1 / D3 / all items from the input | 23.00 / 22.92 / 23.18 | 23.99 / 24.02 / 24.16 |
| + host chunk count / early `open_start` / header copies as local NoC writes | 22.83 / 22.82 / 22.82 | null |
| + copy each header at its first use | 23.10 | regression |
| + input items pushed per chunk (D1 / D2 / D3) | 23.68 / 25.05 / 26.32 | regression (the DRAM reads serialize) |
| (b) as per-item member readiness (early semaphores, 1 / 2 / 3 items) | 23.36 / 23.31 / 23.33 | ~0.5 µs worse than D2: all ports and members read DRAM at once, and D2 makes the port independent of the members' launch skew |

The levers only pay off together: each alone is worth 0.06–0.17 µs, because removing one wait exposes the next.
The first packet moved from 3.3–3.6 to 2.3–2.6 µs on every chip. The ready goes out at ~1.05 µs instead of ~2.3. What
binds now is the peer's ready arriving at ~2.0 µs.

### What graduated (one unqualified path; the code it replaced is deleted)
- `all_gather_rms_norm_port_sender.cpp`: the sender routes header 0 only, then opens the connection and sends the
  ready. In the shadow of the peer's ready it counts the chunks and fills headers 1.. as word copies of header 0. The
  per-header routing loop is deleted.
- `all_gather_rms_norm_port_reader.cpp` + `all_gather_rms_norm_port_walk.hpp` (`for_each_chunk_of_item_src`): the port
  reader reads own items `xb < PORT_INPUT_ITEMS` (= 2) straight from the input tensor through a `TensorAccessor`. The
  chunks, order and batches are the same as the sender's walk. The reader waits on `sem_local_loaded` only before the
  first own item it takes from the members' L1.
- Host: `PORT_INPUT_ITEMS = 2` (env `AGRMS_PORT_INPUT_ITEMS` for A/B), port reader CT arg 14 plus the input's
  `TensorAccessorArgs` (CT 15+), and the program hash key.
- Carve-out (it has no meaning there, not a perf guard): native HEIGHT_SHARDED already reads the own slice in place from
  its resident input shard (`own_from_input`), so the input-direct items are 0 there and only A and C act.
- Whole op (harness, 5 experiments): focus **23.81 → 22.83 µs** (slowest 24.74 → 24.03; `compare`: win, noise 0.05);
  640×512 33.84 → **32.92** (slowest 34.97); 640×1792 78.02 → **75.36** (slowest 78.48). The "before" numbers are from
  Refinement 5's changelog; the subagent's same-session baselines were 34.04 and 78.29. Other cells, from the subagent's
  copy: focus Linear 36.30 → 35.29, focus 1 link 35.82 → 34.79, focus HEIGHT_SHARDED native 33.82 → 33.58 µs.

### No-regression / golden
- Unit dir on the graduated op: acceptance 32/32, native shard 6/6, debug 7/7, precision baseline 15 passed + 1 skipped.
- Golden guard slice (one cell per dtype × memory_layout × topology × num_links × alignment × cluster_axis ×
  fp32_dest_acc_en, shapes rotated over 34 shapes, plus the 3 loose cases): 183 passed, 36 xfailed (EXCLUSIONS), 0
  failed. Random 1,500-cell sample of `test_op` (seed 1): 375 passed, 1060 skipped (INVALID / not buildable on this
  mesh), 65 xfailed, 0 failed. The three perf cases pass their golden check in `perf_measure.py`.
- Precision: unchanged. The same bytes move (raw input tiles), and the compute is untouched.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| ccl::FabricStream (kernel_lib/ccl) | capability | it arms one fixed payload size and routes its header(s) before it can send anything. There is no way to send a fence ready on the open connection after routing one header and build the rest of the header ring (copies of a routed header) in the shadow of the peer's ready, and `signal()` closes the connection. This extends the port sender's existing documented raw-fabric bypass. | not measured (cannot express the pre-send order; the old raw path with every header routed up front was 23808) | 22830 | all_gather_rms_norm_port_sender.cpp:105-155 |

No raw LLK in the compute kernels changed.

## Perf 2 — landing-notification fan-out (perf tournament, round 2)
- Date: 2026-11-13. Focus case: LOOSE_CASES[0], per-device (1,1,640,384) bf16 TILE INTERLEAVED, fp32_dest_acc_en=True,
  gamma bf16 ROW_MAJOR, Ring, 2 links, cluster_axis 1, QB2 1×4 ring under FABRIC_2D_TORUS_XY (the exact flagged config,
  all of it in SUPPORTED). Headline numbers come from the eval CCL harness (5 experiments = 2500 calls, fastest chip,
  `eval.ccl_perf compare`).

### Measured breakdown (before, on the Perf 1 tree)
- Whole op (harness): focus 22.81 / 24.03 µs (fastest / slowest chip); 640×512 32.91 / 34.99; 640×1792 75.39 / 78.39.
  Ablations (focus, 3 experiments): TRANSPORT-only 18.88, LOCAL-only 11.56, EMPTY 0.78. Exposed OVERLAP 3.9 µs.
- Instrumentation: the port reader's `AG_ARRIVAL_WAIT` zone enclosed both the arrival wait and the member landing
  notifications (`notify_until`), so it read as waiting. It is split: `AG_ARRIVAL_WAIT` is now the wait only and the
  new `AG_STEP_NOTIFY` is the fan-out (same gated `AGRMS_ZONE` macro; `MaybeDeviceZoneScope` still does not exist in
  this tree).
- Per-chip zone timeline (1 untraced call, the same shape on all 4 chips, µs from the chip's first marker):
  - TRANSPORT stream: `AG_SEND` 5.0 → 20.4 per port (30 packets, ~0.48 µs per 12 KiB packet vs 0.25 µs at the link
    rate). On the port reader: arrival waits 0.45–1.7 µs per port, **landing-notify fan-out 2.51 µs per port**
    (up to 0.85 µs per step), in front of the relay reads the sender waits on.
  - Tail: last relayed item lands at ~21.9–22.4, **`AG_FINAL_NOTIFY` 1.58 µs on every port of every chip** (16
    serial notifications), last member `AG_LAND_WAIT` ends ~24.1, then exchange / combine / normalize / store to
    28.2.
- Ranked: (1) TRANSPORT stream (15 µs vs the 7.6 µs link roofline; notify fan-out and sender per-packet cost inside
  it); (2) the OVERLAP tail after the last landing (~6 µs, of which 1.58 µs is serial notification issue); (3) the
  pre-send path (~2 µs, Perf 1's subject).

### Portfolio (hard cap: 2 experiments)
1. **OVERLAP (TRANSPORT side effect): landing-notification fan-out** (`perf_experiments/landing_notify_fanout/`).
   Cut the port reader's per-(row, member) semaphore-increment fan-out, on the tail and in the stream.
2. **TRANSPORT: port stream rate** (`perf_experiments/port_stream_rate/`). Raise the per-port packet rate: the
   sender's per-packet issue cost and the reader → sender CB handoff.

### Verdicts
| idea | verdict | focus fastest / slowest (µs), same-session base → candidate | domain |
|---|---|---|---|
| landing-notify fan-out (SF+H+P) | **WIN** | 23.04 / 24.24 → **20.85 / 22.19** | applies everywhere, no exceptions found |
| port stream rate (RELAY_FIRST) | WIN alone, **superseded** | 22.84 / 24.07 → 22.14 / 23.14 alone; on top of SF+H+P 20.86 → **21.75** (regression) | conflicts with idea 1 |

Landing-notify menu (focus; µs fastest / slowest):
| option | result |
|---|---|
| SF: split the final notify per row-block (notify block n_xb−2 while the last block is still in flight) | 21.76 / 22.84 |
| H: hoist the stream's notified-member list (no per-row scan of k flags through `get_arg_val`) | 22.10 / 23.80 |
| SF+H | 21.05 / 22.29 |
| **SF+H+P: placement computed per row (`g.place`) instead of `g.advance` (graduated)** | **20.85 / 22.19** |
| SF+H + reversed row order | 20.91 / 22.16 (ordering is null; P's gain is the code shape) |
| SF+H + raw stateful NoC atomic / absolute 4-byte write | 20.98 / 22.27; 21.08 / 22.33 (null: one issue costs ~24 ns, the cost was loop overhead) |
| SF+H + stateful + cached coordinate table | 21.30 / 22.42 (worse) |
| SF+H + notify after the step's relays ("LATE") | 21.74 / 23.13 (worse) |
| SF+H + early polling between items | 22.18 / 23.36 (worse) |
| SF + NoC multicast increment per member rectangle | 34.95 / 50.74 (correct, but the multicast stalls behind member traffic) |
| one summed increment per member | not built: no group repeats inside one notification burst on any tested cell |

Port-stream-rate menu (focus; full op fastest / slowest µs, TRANSPORT-only µs):
| option | full op | TRANSPORT-only |
|---|---|---|
| baseline | 22.84 / 24.07 | 18.90 (78.0 GB/s received per chip) |
| stateful fabric send + cached free-slot count | 22.78 / 23.97 | 18.83 |
| kernels at O3 (descriptor kernels already build at O2; op_design's "-Os" note was wrong) | 22.82 / 24.01 | — |
| **RELAY_FIRST: the step's relay reads before the notify fan-out** | **22.14 / 23.14** | **18.04 (81.7 GB/s)** |
| reader passes destinations to the sender (no sender walk) | 22.83 / 23.85 | 18.31 (sender 0.22 µs per packet; it then waits 6–8 µs on the reader) |
| per-chunk push with transaction ids / port CB 168 or 224 KiB / own read before notify | 22.15–22.26 | — |

Per-packet finding: inside a batch the sender issues a packet every ~0.32 µs (zones on; ~0.03 CB wait, 0.11 send incl.
slot wait, ~0.12 header + walk). Each item boundary adds ~0.3 µs. The real losses were sender CB stalls of 3–4 µs per
port behind the reader. After RELAY_FIRST the port reader, not the sender, sets the stream rate.

Aggregation: both winners move the reader's notify relative to its relay reads, so they are not independent.
RELAY_FIRST won only because the fan-out was expensive. Once SF+H+P made it cheap, notifying first is right again
(the notify optimizer's "LATE" option says the same thing). Measured in the real op on top of SF+H+P, focus 20.86 →
21.75, 640×512 30.64 → 31.20, 640×1792 74.04 → 74.39. RELAY_FIRST is not graduated.

### What graduated (one unqualified path; the code it replaced is deleted)
- `all_gather_rms_norm_port_reader.cpp`:
  - `notify_rows` walks a `notified[]` list of the stream's members, built once, and calls `g.place(r)` per row.
    The per-row scan of all k flags and the incremental `g.advance` are deleted.
  - The final notify is a per-row-block loop: wait for `upstream_through(block's last step)`, then notify it.
    The single "wait for every item, then notify everything" is deleted; the total-items wait stays as a no-op guard.
- Domain: every cell (Ring / Linear, 1 / 2 links, both cluster axes, interleaved and native sharded). No carve-out:
  every measured cell won and nothing is incorrect or inexpressible.
- Whole op (real op, harness, 5 experiments, `compare` win on each):

| cell | before (fastest / slowest µs) | after | Δ fastest |
|---|---|---|---|
| focus 640×384 Ring L2 | 22.81 / 24.03 | **20.86 / 22.19** | −1.95 (noise 0.05) |
| 640×512 Ring L2 | 32.91 / 34.99 | **30.64 / 32.61** | −2.27 |
| 640×1792 Ring L2 | 75.39 / 78.39 | **74.04 / 77.10** | −1.35 |
| focus, num_links 1 | 35.01 / 35.90 (subagent's base) | 32.70 / 33.76 | −2.31 |
| focus, HEIGHT_SHARDED native | 33.39 / 35.05 (subagent's base) | 33.11 / 34.80 | −0.28 |
| focus, Linear (subagent's copy) | 35.30 / 36.07 | 28.17 / 28.82 | −7.13 |

- Zone evidence (focus, chip 0): `AG_FINAL_NOTIFY` 1.58 µs after the last arrival → one 0.42 µs block (the other block
  runs inside the wait); `AG_STEP_NOTIFY` 2.48 → 1.36 µs per port; last member's `AG_LAND_WAIT` end after the last
  arrival ~1.7 → ~0.4 µs.
- Still binding (finding): the transport stream itself (~15 µs vs 7.6 µs link roofline; the port reader now sets the
  rate), then the tail after the last landing (exchange, combine, normalize, store ≈ 4 µs).

### No-regression / golden
- Unit dir on the graduated op: 60 passed, 1 skipped (acceptance, native shard, debug, precision baseline).
- Golden guard slice: 2 cells per (dtype × memory_layout × topology × num_links × alignment × cluster_axis ×
  fp32_dest_acc_en × layout × gamma_mode) group, a random 600 cells of `test_op` (seed 2), and the 3 loose cases. That
  is 1,456 cells: 311 passed, 1068 skipped (INVALID / not buildable on this mesh), 77 xfailed (EXCLUSIONS), 0 failed.
  The three perf cases pass their golden check in `perf_measure.py`.
- Precision: unchanged. The same bytes move; only when and how the landing semaphores are bumped changes. A block is
  still notified only after its last item has landed.
- Artifacts: the raw profiler CSV dumps of `port_stream_rate` (391 MB) are not committed. `zone_run.sh` regenerates
  them. The harness JSONs are committed.

### Helper bypasses — none
The graduated path uses `noc_semaphore_inc` and no raw LLK or raw NoC. The rejected stateful-atomic options bypassed
`noc_semaphore_inc` (it has no set-state / with-state form for atomics) and measured null, so they were not graduated.
No compute kernel changed.
