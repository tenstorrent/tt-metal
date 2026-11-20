# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-19
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). R1
  mirror_resident_gather end to end: fabric plan (probe placement, balanced ring / line schedules, ports_union
  excluded from the compute grid of every chip), port reader / sender (ready fence, relay, one fused arrival increment
  per (wave, shard) entry, landing notifications), member reader / compute / writer (base-op local norm with symmetric
  partial exchange), LN pivot + Chan combine, dense assembly for `w_non_aligned`. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False], layout=[TILE],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode both, gamma_dtype=[fp32, bf16,
  none], gamma_layout=[TILE, ROW_MAJOR, none], norm_type=[rms_norm, layer_norm], beta_mode both,
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2] (== TARGET). EXCLUSIONS: {float32, fp32_dest_acc_en=False}, {bfloat8_b, w_non_aligned}.
- **Accuracy achieved** (fp32 DEST, 1×4 ring, worst device, 4 shapes via
  `test_all_gather_rms_norm_precision_baseline.py`): bf16 rms PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019;
  bf16 layer_norm PCC ≥ 0.999998, max_abs ≤ 0.045, rel_rms ≤ 0.0019; fp32 layer_norm PCC 1.000000, max_abs ≤ 0.011,
  rel_rms ≤ 0.0006; bf8b rms PCC ≥ 0.99995, rel_rms ≤ 0.0097. Known gap: bf16 layer_norm with fp32_dest_acc_en=False
  rel_rms 0.005–0.014 here, 0.015–0.025 on half of the golden family (Refinement 2).
- **Golden suite at Phase 0** (hash-sampled matrix + all ring-mock / regression / loose tests, `verifier_report.json`):
  2357 supported_pass, 174 supported_fail (143 numerical-precision, 31 OOM — both queued), 644 xfail_expected,
  0 xpass_drift, 0 xfail_wrong_mode.
- **Perf** (eval ccl_perf protocol, 5 experiments, fastest chip): focus 640×384 rms 21.29 µs (roofline 7.6); indexer
  640×32 layer_norm 16.33 µs (roofline 0.6). Implementer's rounds: focus 22.23 → 21.25 µs (direct gamma reads after the
  exchange, out unit 4); indexer 22.68 → 16.16 µs (LN FPU fast path, k = 1 combine); the verifier's precision fix
  below then cost +0.05 / +0.17 µs.
- **Issues encountered / verifier fixes**:
  - FPU normalize chain truncated its fp32 intermediate to the input's 16-bit format at every dest-reuse step
    (DEST → srcA in cb_x's format): −0.6 % per-row scale bias on LN bf16, −0.3 % on rms bf16, and one borderline
    golden failure (bf16 LN 1×384 rel_rms 0.0100). Fixed in `DestReuseBcast` (srcA switched to the DEST width once per
    chain, restored once). LN bf16 rel_rms 0.0072 → 0.0019.
  - `{bfloat8_b, h_non_aligned}` exclusion dropped (measured unnecessary: 241/241 sampled cells pass);
    `{bfloat8_b, w_non_aligned}` kept (measured necessary: 238/238 inf).
  - Split-fit refusal reworded to carry `OOM` (classified as OOM, not `other`).
  - L1: `cb_moment_a` no longer allocated on the LN FPU path (dead), `cb_moment_b` = rows there, `cb_out` capped at
    the member block; the writer's cb_out ring size reads `cb_plan()` (was a restated `OUT_DEPTH·unit`).
  - DRY: one `MulScalar` reduce post-op (was two identical structs).
  - Environment: with `--dev` (watcher), re-opening a fabric mesh after any fabric traffic times out in the Ethernet
    teardown — reproduced with the `fabric_all_gather` example, so not this op. `perf_measure.py` fails with
    `ENAMETOOLONG` on its result file name; `perf_agrms.py` runs the same protocol.
  - Deferred: owner combine for k > 8 (Refinement 1), compact gamma pages (Refinement 1), zero-copy own shard,
    credited landing.
- **Tests added**: `test_all_gather_rms_norm.py` (acceptance, 19), `test_all_gather_rms_norm_precision_baseline.py`
  (20), `perf_agrms.py` (perf harness wrapper).

## Refinement 1 — L1 fit for wide rows and L1-sharded I/O: port the base op's owner combine (k > 8) and compact gamma pages
- Date: 2026-11-19
- What was done:
  - **Owner combine for k > 8** (base op R3 / Perf 2 row partial, unicast return): `_k_candidates` now runs up to
    `K_CAP_MAX = 128` (writer RT-arg bound) with the slice-aligned rule, and `select_split` prices k > 8 with the base
    op's `OWNER_MEMBER_COST · ceil(rows/k) · k`. Row r is owned by member r % k. Partials travel in ROW form (two 16-datum
    face rows into row m % 32 of gather tile m / 32 of the owner: `ceil(k/32)` gather pages per owned row instead of k);
    the owner zero-fills the never-written rows, combines, and returns column-form statistics (faces 0/2) into slot r of
    every member's `cb_rowstat` (+ `cb_shift`, `cb_kg`), one `SEM_INV` increment per row. rms: member collapse = ones-row
    × sumsqᵀ matmul, owner = `reduce_tile` COL + `transpose_dest` + the column-0 finalize (base op code). layer_norm (not
    in the base op): member finalize transposes (mean_m, M2_m) in DEST; the owner's Chan combine is centered on a pivot
    `K = bf16(mean of member 0)` (SFPU, fp32), weights n_m from a column-0 constant (`cb_wcol`), sums over members by a
    ones-row matmul of centered terms only, then `transpose_dest` → (inv_std, shift, K_g). Every perf-flagged shape
    keeps its regime (focus k = 4, indexer k = 1).
  - **Compact gamma / beta pages** (base op Perf 2): `cb_gamma` / `cb_beta` page stride = `gamma_page_layout()` (192 B
    for bf16 / fp32, + tail pages for the last page's unpacker read) instead of a whole gamma tile; loaders unchanged
    except the stride.
  - **Port stage batch fitted** to L1 (`port_batch_candidates`: 4 → 2 → 1 packets per batch); the cap stays the default.
  - **LN L1**: live-disjoint scratch CBs share one region (one CBDescriptor, several buffer indices: `cb_alias_groups`):
    moments scratch (`cb_moment_a`/`cb_moment_b` or `cb_cent`) with the owner's `cb_dev`, and `cb_mstat` with `cb_mu`.
    With fp32 DEST the LN FPU path falls back to the SFPU moments path (no block-sized `cb_cent`) when no split fits.
  - **Global semaphores no longer pin L1**: the cached (arrival, ready) pairs were created under the first call's L1
    shards and, once those were freed, set the CB budget of every later call (measured: budgets 625 KB → 101 KB → 68 KB
    as configs accumulated; the isolated cells all fit). `_relocate_pinning_sems` re-creates a pair once free L1 has
    opened above it (allocator is top-down) and keeps the higher one; a no-op in steady state.
  - Program hash now covers the fitted knobs (out unit, port batch, LN path). `FORCE_K` / `LAST_INFO` test hooks.
- Accuracy achieved (1×4 ring, owner path k = 24 / 16, worst device and slice): bf16 rms (1,1,32,2048) PCC 0.999998,
  rel RMS 0.0018; bf16 LN PCC 0.999998, rel RMS 0.0021; bf16 LN fp32_dest_acc_en=False PCC 0.99997, rel RMS 0.0095;
  fp32 LN offset 50 (1,1,32,1024) PCC 1.000000, rel RMS 0.0015.
- Golden test progress: the 31 Phase-0 OOM cells 31/31; family slices `1x1x32x2048` 840 pass / 0 fail;
  `1x1x32x1280, 1x1x32x1792, 1x1x8x1792, 1x1x640x384, 1x1x64x1000, 1x1x32x50, 4x128x47, 32x17` 5009 pass / 256 fail (all
  bf16 LN acc16 precision, Refinement 2's family; none new); ring-mock + regression 160/160.
- Perf (eval protocol, 5 experiments, fastest chip): case0 21.29 µs, case1 16.32 µs — unchanged (baseline 21.29 / 16.33).
- Issues encountered: the "OOM" budgets in the batch were mostly L1 fragmentation by the op's own persistent global
  semaphores, not the cells' footprint (each remaining cell passed alone); fixed at the source. Not done (not needed by
  any named cell): zero-copy own shard / output shard (item 3), owner-spread / root multicast return (base op perf).
- Tests added: `test_all_gather_rms_norm_owner.py` (13: forced k = 12 / 16 / 20 / 40 incl. zero-filled gather rows, two
  gather tiles, multi-row owners, rms / LN FPU / LN SFPU, 16-bit DEST, non-aligned dense assembly);
  `test_all_gather_rms_norm_debug_l1.py` (debug, kept: L1 block view across calls, owner metrics).

## Refinement 1b — L1 fit for wide rows and L1-sharded I/O (debug: fix gate violations)
- Date: 2026-11-20
- What was done: the gate's regression (92 prior-passing cells OOM, 3 `L1ProfilingError`) was order dependent: each
  cell passed alone. Root cause, measured with the harness plugins on the real 700-cell order: L1 tensors of FAILED
  cells stay alive in reference cycles (pytest's exception traceback → the test frames → input / output shards) until a
  full (gen-2) GC, which a 200k-item pytest process runs rarely. Dead shards accumulated to ~1.3 MB per bank and
  fragmented L1, so `cb_l1_budget` (free L1 below the lowest allocated buffer) dropped to 133–363 KB. Refinement 1 made
  more sharded cells run to the precision check (leaking input + output shards) instead of raising OOM early, which is
  why Phase 0 did not hit it. When such a GC finally fires inside a later DRAM cell's graph capture, the profiler's
  `size_t` L1 base wraps (`peak_total = 2^64 − 61184`) — the 3 profiler errors.
  Fix (host side, op file): after each dispatch (this call's CBs already placed) `_collect_dead_l1` runs one
  `gc.collect()` when L1 holds blocks no call argument / cached semaphore owns AND that were not already seen to survive
  a collection (live foreign tensors therefore do not re-trigger it); on the OOM path it collects with `force` and
  retries the split once. The OOM message now lists the allocated L1 blocks.
- Accuracy achieved: unchanged (no kernel change).
- Golden test progress (FULL suite, eval_test_runner, same order/plugins as the gate): 29118 passed / 1284 failed /
  0 hangs / 0 OOM / 0 `L1ProfilingError` (previous attempt 27115 / 1867 incl. 799 OOM; Phase 0 23269 / 5289).
  Regressions vs Phase 0: 0. All 1284 failures are `numerical-precision`, bf16 layer_norm with fp32_dest_acc_en=False
  (Refinement 2's family). 40 of them (1x1x17x1280 / 1x1x640x1792 HEIGHT_SHARDED, failed in Phase 0) had passed in the
  previous attempt only because leaked L1 forced a wider member split (k = 16 instead of 8 → narrower members →
  fewer tiles accumulated in 16-bit DEST): evidence for Refinement 2 that narrower members lower the 16-bit error.
- Perf (eval protocol, 5 experiments, fastest chip): case0 21.31 µs, case1 16.32 µs (unchanged).
- Issues encountered: none beyond the above; unit dir 58/58.
- Tests added: none (repro = the golden order with `EVAL_CAPTURE_L1=1 -p eval.metrics_plugin`).

## Refinement 2 — layer_norm precision with 16-bit DEST (fp32_dest_acc_en=False)
- Date: 2026-11-20
- What was done: measured the error split per row (new debug test, no gamma / beta: fit out = a·ref + b): the dominant
  term was a per-row SCALE bias +0.3…1.3 % growing with member width (σ² under-estimated), plus an offset of rms
  0.002–0.006 with row offsets. Cause: with 16-bit DEST every intermediate is bf16 and an SFPU store truncates toward
  zero (LLK: "SFPSTORE → bf16 dst defaults to TRUNCATION"), so tile-by-tile DEST accumulation and the per-member SFPU op
  chains of the finalize / Chan combine all bias downward. 16-bit-DEST LN now (fp32 DEST untouched):
  (1) `moments16_block`: x tiles transposed into DEST, loaded with the Welford LLK's SFPTRANSP pattern (one LREG lane
  per row), shifted sums S1, S2 (shift = the row's first sample) accumulated in fp32 LREGs across the member's tiles,
  slice-last tiles limited to their valid columns (the segs RT flag now carries v); mean_m / M2_m formed in fp32 and
  stored once with RNE; (2) `combine16_block` (symmetric, k = 1) and `owner16_ln` (k > 8): one-pass pivoted Chan in
  fp32 LREGs, fp32 rsqrt (with `rsqrt_tile_init()` — the first cut left its programmable constants to the previous
  kernel: inv_std ≈ 1e−22, zero outputs), K_g = RNE bf16(mu), shift = (K_g − mu)·inv_std carrying the sub-bf16 mean;
  row → column form through `cb_mu` + unpack transpose; (3) FPU normalize for every bf16 / bf8b 16-bit-DEST cell (the
  SFPU chain truncates: −0.6 % scale on w_non_aligned); (4) moment scratch CBs (`cb_moment_a/b`, `cb_mstat`, `cb_cent`,
  `cb_km`, `cb_dev`) not allocated on this path (`Sizes.m16`). Reused: cb_partial / cb_gathered / owner exchange payloads,
  writer, reader, normalize chain; Added: raw-TTI LREG accumulators (documented bypass), `cb_mu` 3-page round trip.
- Accuracy achieved (1x4 ring, bf16, HiFi4): precision baseline `bf16_ln_acc16` rel RMS 0.0041–0.0044, PCC 0.99999 on
  [1x1x32x384, 1x1x640x32, 1x1x640x384, 1x1x128x1792] (was 0.0094–0.0142); debug decomposition rel RMS 0.0029–0.0039
  incl. owner (32x1280, 32x4096), dense w_non_aligned (64x1000), multi-row (2048x384); residual ≈ output rounding.
- Golden test progress: one-per-shape slice of the family (39 cells, Linear, 1 link, gamma+beta RM) 33 → 39/39; two
  random samples over all axes of the family 51/51 + 126/126 executed (rest INVALID-skipped / expected xfail);
  test_regression 32/32; unit dir 70/70. Perf (fp32 DEST, path unchanged): case0 21.30 µs, case1 16.32 µs.
- Issues encountered: stale SFPU programmable constants (rsqrt init missing) produced inf / zero outputs only on some
  runs — found with DEVICE_PRINT of the row-form statistics.
- Tests added: `test_all_gather_rms_norm_debug_ln16.py` (per-row scale / offset decomposition, 6 shapes × 2 offset
  modes, gated on the (bf16, False) tolerance); precision baseline drops the `bf16_ln_acc16` known-gap carve-out.

## Refinement 3 — TRANSPORT: raise the focus case's link bandwidth
- Date: 2026-11-20
- What was done: measured the per-port send with device zones + port ablations (new probe `zones_agrms.py`,
  probe-only `AGRMS_ABL` bits): 11.8 µs real, 11.1 without the port's L1 reads, 10.9 without relay waits, 10.4 with
  both stubbed (= the link floor at the current 30 packets / 368 KB per port). Then:
  (1) port reader **stream mode** (`PORT_READ_STREAM = 1`, default): each stage slot's L1 read carries its own NoC
  transaction id, the whole ring is in flight, each slot is pushed the moment its own read has landed (no batch
  barrier) — 21.29 → 20.98 µs; (2) **`port_batch_chunks` 4 → 2** (the sender's flush / pop granularity) — 20.98 →
  20.75 µs, case1 16.24 → 15.68 µs; (3) **scatter packing** knob (`PORT_PACK`, fabric scatter write of up to 4 run
  pieces per packet, fused scatter + inc / separate flushed inc at entry end, shared packer header
  `kernels/all_gather_rms_norm_port_pack.hpp`) built and parked at 0: +0.4 µs here, link floor unchanged (byte-bound,
  not packet-rate-bound). Knobs single-sourced in the descriptor (`PORT_READ_STREAM`, `PORT_PACK`, `STAGE_BATCHES`,
  `PORT_PUSH_PACKETS`, env-overridable for probes); the sender no longer takes a host packet count (flushes its tail
  batch after the loop). Reused: the port kernels, entry lists, stage CB; Added: the packer header, stream read loop.
- Measured (eval `ccl_perf` via `perf_agrms.py`, 5 experiments, fastest / slowest chip): case0 21.29 / 22.12 →
  **20.75 / 21.53 µs**; case1 16.24 / 17.05 (stream, batch 4) → **15.68 / 16.44 µs** (verifier 16.33). Nulls / losses:
  per-packet read barrier 24.94, INC_EVERY 4 → 22.75, 16 → 23.69, batch 1 → 22.62, 6 → 21.69, 3 → 20.77, stage depth
  3 / 4 batches → 20.72 / 20.73, packing → 21.39. Not dispatch-bound.
- Accuracy achieved: unchanged (transport only; bit-identical data movement) — unit dir 70/70 pass.
- Golden test progress: stratified slice (one cell per memory_layout × norm_type × topology × num_links × alignment,
  random dtype / gamma / beta, both perf loose cases) + `test_ring_mock`: 137 passed, 29 skipped, 0 failed.
- Issues encountered: the eval CSV of zones is written only when the mesh closes (the probe parses it afterwards).
  Remaining transport gap ≈ 1 µs per port vs the link floor (relay dependency / launch skew) + the ready fence —
  Refinement 4 territory.
- Tests added: `tests/ttnn/unit_tests/operations/all_gather_rms_norm/zones_agrms.py` (zone breakdown probe, not
  collected: no `test_` file prefix).

## Refinement 4 — OVERLAP: shorten the focus case's exposed tail and start
- Date: 2026-11-20
- What was done: re-measured the focus case's breakdown after Refinement 3 (zones with 4 traced calls so the
  profiler's marker buffers do not overflow: launch → first packet ≈ 2 µs, link ≈ 14 µs, last landing → end ≈ 5.3 µs).
  Then the OVERLAP levers, each a single-source descriptor knob (env-overridable for probes):
  (1) **per-entry own-loaded masks** (`OWN_LOADED_MASK = 1`, kept): member i of a port's own list adds `1 << i` to the
  port's `sem_own_loaded` (≤ 31 members, else the old wait-all), and the port reader's own entry waits only for its
  own members' bits (wait word = flag bit 31 | mask), so the first packets are staged before the fence ends:
  20.74 → 20.64 µs; (2) `RELAY_LAG_WAVES` env knob, 0 / 2 measured worse, 1 kept; (3) **tail split** (`TAIL_K_STEPS`,
  parked at 0): the last wave's row-groups at the next wider member count (Split.tail_k / classes(), one
  reader / writer / compute kernel set per class, merged CB plan, per-group core offsets in build_entries) — correct,
  +1.2 µs; (4) `TAIL_RELAY_LAG` (send-slot reorder of the last wave, parked at RELAY_LAG_WAVES); (5) `XG_FIRST`
  (x·γ into `cb_xg` before the combine, one multiply by inv_rms after it; gated on the chosen split still fitting L1;
  parked at 0, +0.26 µs); (6) `AFFINE_FIRST = 1` (writer finishes gamma / beta before the partial exchange; null, kept).
  Probe-only: zones_agrms.py takes AGRMS_ZONE_ITERS / AGRMS_ZONE_PATTERNS and prints the group → core map.
  Reused: the port reader's wait loop, the reader's signal loop, the split / entry builder, the chain helpers.
  Added: the member-class split, the mask encoding, the cb_xg pass (parked).
- Measured (eval `ccl_perf` via `perf_agrms.py`, 5 experiments, fastest / slowest chip): case0 20.74 / 21.51 →
  **20.64 / 21.35 µs** (compare: win); case1 15.68 / 16.46 → 15.64 / 16.39 (null). Losses: relay lag 0 25.72, 2 21.93,
  INC_EVERY 12 22.58, tail relay lag 0 22.41, tail split 21.94 / 23.24, both 23.11, x·γ first 21.00 / 22.07.
  Ablations: normalize unchanged with output stores stubbed (2.52 vs 2.51 µs) or with the per-tile srcA format switches
  removed (2.57 vs 2.62) → compute-bound at HiFi4; two identical kernel sets cost nothing (20.744).
- Accuracy achieved: unchanged at the defaults (no compute change; own-loaded masks only reorder port reads) — unit dir
  70/70; the tail split and x·γ variants also passed the golden check of the perf case.
- Golden test progress: random 500-cell sample 77 passed / 28 xfailed / 0 failed (395 topology-skipped on this mesh);
  stratified 110-cell slice incl. both loose cases 13 passed / 5 xfailed / 0 failed.
- Issues encountered: device zones on the late compute cores mis-pair across calls with 20 traced calls (silent marker
  budget); with 4 calls the data is clean. The tail split's last 8-member group finished its partial exchange ~4 µs after
  its last moment — not explained; recorded as the next lever.
- Tests added: none (probe changes only: `zones_agrms.py`).

## Refinement 5 — Indexer layer_norm case: local LN tail (column-0 SFPU scope) + own-slice moments before the landing
- Date: 2026-11-20
- What was done: (perf, `LOOSE_CASES[1]` 640×32 bf16 LN, k = 1) (1) k = 1 fp32-DEST member finalize + Chan combine
  collapsed into one column-0-scoped SFPU window (`ln_finalize_k1_block`, the base op's `finalize_col0_body` pattern);
  with the block pivot K_g = K_m is packed straight to `cb_kg`. (2) OVERLAP: `AGRMS_OWN_PIVOT` — the member pivot is the
  bf16 mean of the own slice, reduced from a reader copy (`cb_own`) before the landing; k = 1 stages K_m in fp32
  `cb_mstat` and forms K_g = bf16_rne(mu) in the finalize. (3) Blocked LN FPU normalize: `eltwise_chain` block mode
  (each step inits once per DEST block; `DestReuseBcast(blk, Wt)` switches srcA per block), block = gcd(writer unit,
  cb_out ring pages, member widths) so a block never straddles the cb_out ring end (compute CT [6], `AGRMS_NORM_BLOCK`).
  (4) `AGRMS_OUT_UNIT` env knob (default 4). Reused: the pivot reduce / centering / D1-D2 reduces, the FPU normalize
  chain and `DestReuseBcast`, the reader's own-read loop. Added: one SFPU body, `stage_pivot_block`, `own_pivot_block`,
  `cb_own`.
- Measured (eval `ccl_perf`, 5 experiments, fastest / slowest chip): case1 15.64 / 16.39 → 12.59 (fused finalize) →
  12.42 (own pivot + staged K) → **12.07 / 12.82 µs** (blocked normalize); own pivot vs block pivot 12.42 vs 12.53 (noise
  band, kept as a knob); writer unit 2 / 1: 12.11 / 12.32 (kept 4). case0 guard 20.65 µs (unchanged).
- Accuracy achieved: precision baseline unchanged — bf16_ln 640×32 PCC 0.999998, rel RMS 0.00185 (was 0.00187), row scale
  0.99973; bf16_ln 640×384 / 128×1792 / 32×384 rel RMS 0.0019 (identical); fp32_ln 0.00053–0.00070; bf16_ln_acc16
  0.0041–0.0044 (identical).
- Golden test progress: stratified LN slice (every dtype × fp32_dest × alignment × memory_layout, 3 shapes each, + 12 rms,
  + both perf cases) 125/125 executed pass (3 skips: sharded inputs too large to allocate, as before; 48 expected xfail),
  second random LN sample 218/218 executed pass (4 such skips, 38 xfail); test_regression + test_ring_mock 160/160; unit
  directory 70/70.
- Issues encountered: the first blocked normalize used block = writer unit; on 2×4×128×512 (pitch 6, members 6 / 4,
  rows 2) the 4 + 2 blocks per row wrapped an 8-page `cb_out` and the 4-tile pack ran past the ring end (PCC 0.85, both
  DEST widths) — fixed with the gcd block. A bf16 pivot copy inside the fused finalize cost +0.27 µs (format switch
  among UnpackToDestFp32 copies) — staged into fp32 `cb_mstat` before the landing instead. Own-slice centered moments
  before the landing not built (1 of 4 tiles, non-contiguous remote columns, per-phase fixed cost dominates).
- Tests added: none (the precision baseline, unit and golden suites cover the touched paths; perf via perf_agrms.py /
  zones_agrms.py).

## Refinement 6 — TRANSPORT: indexer layer_norm case packetization and fixed exchange cost
- Date: 2026-11-20
- What was done: (perf, `LOOSE_CASES[1]` 640×32 bf16 LN, k = 1, 20 row-groups × 1 row, 1-tile runs) a host-gated
  **short-run transport regime** (`short_run_transport()`: on when scatter packing at least halves the busiest port's
  packets; every other shape keeps the original path byte for byte except a constant entry-end increment header):
  (1) scatter packing (`PORT_PACK`, built in Refinement 3, now auto), (2) sender flush + stage-slot release per packet
  (`SHORT_RUN_BATCH = 1`), (3) a packet-header ring on the port cores (`cb_port_hdr`, 16 × 128 B): every header is
  built ahead of its send while the sender waits for the ready fence, a stage slot or a router credit; slots take the
  routed template by a RISC copy at their first build (`HDR_COPY`; DMA measured −0.2 worse), (4) stateful router
  sends with cached credits (`PORT_STATEFUL`). The sender now walks all entries with one `PacketWalk`. Every knob is
  an env-overridable live tunable (-1 = auto). Reused: the port packer, the reader's stream mode, the fence and the
  entry schedule. Added: `cb_port_hdr`, `PacketWalk`, `entry_packets()` / `short_run_transport()` (host mirror of the
  packer). Probe-only: `AGRMS_FORCE_K`, `AGRMS_GROUP_ROWS`, `AGRMS_ABL` bit 4 (one contiguous write per packet),
  `dprint_agrms.py` (DEVICE_PRINT launcher; profiler + DEVICE_PRINT segfault on the mesh).
- Measured (eval `ccl_perf`, 5 experiments, fastest / slowest chip): case1 12.07 / 12.82 → **11.42 / 12.23 µs**
  (compare: win); case0 guard 20.58 / 21.27 (unchanged). Breakdown and every variant: op_design.md "Refinement 6
  measurements" (pack alone 12.09, ring without pack +1.1, pack + ring + stateful 11.77, + batch 1 11.42; no header
  builds in the fence +0.64; 1 link +3.2; waves / relay lag / stage depth flat or worse; k = 4 or taller row-groups
  +8 to +37 µs).
- Accuracy achieved: unchanged (pure data movement: the same tiles land at the same addresses) — unit directory 70/70
  incl. the precision baseline (bf16_ln 640×32 rel RMS 0.00185).
- Golden test progress: every cluster_axis = 1 cell of 1×1×640×32 (the only golden shape in the new regime): 420 passed
  / 84 xfailed / 0 failed (2088 topology / INVALID skips); one LN cell of each of the 39 shapes: 39/39 passed (plan
  print confirmed only 640×32 enters the regime); random 700-cell sample of narrow / decode / focus shapes: 110 passed /
  34 xfailed / 0 failed; test_regression + test_ring_mock 160/160; both loose cases pass their golden check.
- Issues encountered: the incoming entries are counted ≈ 2.5 µs after the upstream issued them regardless of packet
  count or landing pattern (per-hop fabric latency, measured with wall-clock DEVICE_PRINT on the port pairs) — the
  relay pays it twice and it is now the transport's floor here; the next lever is a fabric line multicast instead of
  the worker relay (scheme change). Computing the 2D route per ring slot cost +3.1 µs (fixed by copying one routed
  template).
- Tests added: none (the 640×32 unit and golden cells cover the new regime; perf / debug launchers only).
