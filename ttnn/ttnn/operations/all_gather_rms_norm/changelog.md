# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-23
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). SF-GN scheme (stats-first, gather-normalized), one `ttnn.generic_op` / `MeshProgramDescriptor` per call; details below ("Phase 0 (ttnn-implementer)").
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"], gamma_layout=[TILE, ROW_MAJOR, "none"], norm_type=[rms_norm, layer_norm], beta_mode=[beta, no_beta], memory_layout=[INTERLEAVED], cluster_axis=[0, 1], topology=[Linear, Ring], num_links=[1, 2]; EXCLUSIONS={float32 + fp32_dest_acc_en=False; bfloat8_b + w_non_aligned}
- **Accuracy achieved** (verifier, `test_all_gather_rms_norm_precision_baseline.py`, 10 cases / 7 shapes, 1×4 ring, 2 links): bf16 RMS PCC ≥ 0.999997, max_abs ≤ 0.088, rel_rms ≤ 0.0040; fp32 RMS PCC 1.000000, rel_rms 0.0017; bfloat8_b RMS PCC 0.999953, rel_rms 0.0097; bf16 LN PCC ≥ 0.999998, rel_rms ≤ 0.0019; fp32 LN rel_rms 0.0014.
- **Golden suite at Phase 0**: gate tier (1720 cells): 1080 supported_pass, 480 xfail_expected, 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode after the verifier fixes (before: 404 supported_fail, 3 xfail_wrong_mode). `agent_logs/verifier_report_summary.json`.
- **Perf at Phase 0** (eval harness, 5 experiments, fastest / slowest chip): focus 640×384 RMS 35.56 / 35.94 µs (roofline 7.6, target 10.1); LN 640×32 + γ + β 29.60 / 29.88 µs (roofline 0.6, target 0.8). Not dispatch-bound.
- **Issues encountered**: see "Verifier (Phase 0 hardening)" below — persistent L1 stats-table leak (404 golden failures), the stats-table cap ignored by the split, non-fp32 RMS partial sums (prompt MUST), the base op's bf8b × h_non_aligned exclusion lifted.
- **Tests added**: test_all_gather_rms_norm.py (planner), test_all_gather_rms_norm_debug.py (implementer), test_all_gather_rms_norm_precision_baseline.py (verifier)

## Verifier (Phase 0 hardening)
- **Persistent L1 stats table leaked, then collided with the CB region** (fixed). `stats_table()` cached a mesh L1 tensor *per table size* and never freed it. L1 allocates top-down, so every new shape pushed a new table lower (probe: 1570624 → 1529664 → …) until programs failed with "Statically allocated circular buffers ... clash with L1 buffers" (49 cells). The golden conftest deallocates the device tensors a failed call's frames hold, which freed the cached table under the cache and failed every later call with "Tensor is not allocated" (321 cells). The 27 remaining cells are the next item. **Fix:** the table is now `cb_stats_table` (CB 25), a program-scoped CB on the full core range. Identical CB lists give it the same L1 address on every core of every chip, and peers write into it only after the ready fence, i.e. while the receiving program is live. No persistent L1 is left besides the global semaphores. The cached DRAM staging buffer is re-created if it is no longer allocated. Gate tier: 404 supported_fail → 0.
- **Stats-table cap not honoured by `select_split`** (fixed): LN at R = 64 picked k = 3, a 288 KiB table over `STATS_TABLE_CAP` (27 cells: 2048×1280, 2048×384, 4×1×512×512, LN). `stats_k_cap(R, G, nstats)` now clamps k; the RuntimeError remains only if k = 1 does not fit.
- **RMS partial sums were truncated to tf32 when summed** (fixed; prompt rule "per-row partial sums MUST stay fp32 on the wire and when summed"): the local collapse (`cb_sumsq`) and the cross-chip combine (`cb_gathered`) went through the FPU / AccumulateViaAdd path, which reads Float32 CBs through srcA as tf32. Under fp32 DEST both CBs are now UnpackToDestFp32-tagged and reduced with `ReduceFp32Mode::Accurate` (SFPU), the same pattern LN's S1 uses. Perf null within noise (35.542 → 35.557 µs, noise 0.050).
- **EXCLUSIONS re-measured**: `{bfloat8_b, h_non_aligned}` lifted (19/19 gate cells pass: H padding is never element-masked); `{bfloat8_b, w_non_aligned}` kept (19/19 severity=bug with it lifted).
- Docs: stale reader header comment (word-copy mean broadcast / column scatter, both gone) rewritten; `l1_ledger.md` and `op_design.md` updated for `cb_stats_table`.

## Phase 0 (ttnn-implementer)

**Built:** the SF-GN scheme of `op_design.md` end to end, one `ttnn.generic_op` (one `MeshProgramDescriptor`) per
call.

- Workers = grid − 2L port cores, row-groups × k W members (`select_split`, fused cost model), **common runtime args
  only** (identity from the logical position + the port list). Local stats → 128 B fp32 records into both link-0 hub
  tables → stats round over the fabric (relay, one fused increment per slot) → hub multicast "stats ready" →
  per-worker combine of the P = G·k partitions → normalize into the resident `cb_out_block` → own write.
- Ports pull the own normalized slice from worker L1 and send it into the peers' **final output pages** (bank-
  contiguous runs, up to 4 runs per scatter packet, relay with arrival counters, balanced ring far slot), ready fence
  per call, counters re-armed every call (trace-safe, determinism test passes).
- R_U (W_local % 32 ≠ 0): slot-contiguous DRAM staging + element-shifted assembly by the workers.
- LayerNorm: one stats round with shifted sums (m, T = Σ(x−K)², K = tf32(m)); exact combine in fp32 SFPU;
  centering as (x − K) − (mean − K).
- Gamma / beta: TILE `(1,1,1,W)` or ROW_MAJOR sticks `(1,1,ceil(W/32),32)`, this chip's global columns.

**Acceptance:** `tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm.py` 15/15 (production
mode); debug matrix `test_all_gather_rms_norm_debug.py` 8/8. `--dev` (watcher): the cases that got a device passed;
on this box the device bring-up under `--dev` repeatedly stalls on "waiting for active ethernet core" (fixture
errors, environment, not the op).

**SUPPORTED (Phase 0):** dtype {fp32, bf16, bfp8_b}; fp32_dest_acc_en {True}; TILE; alignment all three; rank 2–4;
gamma / beta all formats; norm_type both; memory_layout {INTERLEAVED} (DRAM input, DRAM-interleaved output);
cluster_axis {0, 1}; topology {Linear, Ring}; num_links {1, 2}.

**Not landed from the from-base list (one generality refinement, base op mechanisms):**
- `memory_layout` HEIGHT_SHARDED / WIDTH_SHARDED: zero-copy `cb_x` on the shard (base op
  `cb_descriptor_from_sharded_tensor`, its `RMS_SHARDED` reader path), port cores chosen off the shard grid, k = shard
  count for WIDTH, W-tail masked (not zeroed) in the caller's shard; sharded output landing walks shard cores
  instead of DRAM banks (the walk's `nb` / page→core mapping).
- `fp32_dest_acc_en=False` (bf16 / bfp8): the LN SFPU paths use UnpackToDestFp32 (needs fp32 DEST); the design's
  16-bit-DEST variant (FPU centering, row 12' of Block Operation Realization) is not built. RMS would only need the
  tag list emptied (the gather transpose would then truncate the partial sums to tf32 — base op Refinement 1
  tolerances).

**Perf (eval `perf_measure.py`, 5 experiments, fastest chip, QB2 1×4 ring, 2 links):**

| Step | focus 640×384 RMS | LN 640×32 + γ + β |
|------|-------------------|-------------------|
| first correct version | 52.63 µs | 37.86 µs |
| scatter packets (4 runs) + x before gamma + gather zeroing off the critical path | 38.41 | 33.19 |
| row-form gather + exact DEST transpose + stats-ready multicast | 35.54 | 32.22 |
| LN centering by the local shift K (no fp32 mean broadcast) | 35.54 | 29.58 |
| roofline / target | 7.6 / 10.1 | 0.6 / 0.8 |

Not dispatch-bound (max gap 0.6 µs). Measured nulls / rejects (knobs kept at the design values): scatter runs 1 / 2
(49.2 / 44.0 µs), batch 2 / 6 / 8 (36.0 / 37.0 / 39.0), inc_every 4 / 16 (38.7 / 38.4, pre-gather-change base
38.4); own entry from DRAM instead of the L1 pull 34.85 (−0.7 µs, the pull is not the bottleneck); aligned slice
forced through the staging regime 40.3 µs (final-address landing stays).

**Where the time goes now (focus, chip without launch-skew wait, single profiled call):** local stats done ~2.3 µs;
stats round to ~6 µs; gather + combine + normalize + own write to ~11 µs; **bulk own entry ~15 µs** (240 KB per
port at ~16 GB/s: interleaved final pages give only 1–2-page bank runs per row, so a 4-run scatter packet is ~6
pages / 4 DRAM writes), relay ~4 µs, landing tail ~3 µs. LN 640×32: local stats ~3 µs (x → partial), records
1.4 µs, combine + normalize ~7 µs of compute for one tile-row, bulk ~11 µs for a 20-tile slot.

**Refinement candidates (ranked by the breakdown):**
1. TRANSPORT — bulk packet efficiency on interleaved final pages (the dominant term): a destination-run layout with
   longer runs, or fewer / larger DRAM writes per packet; measure against `fabric_all_gather`'s long-run rate.
2. OVERLAP — bulk start granularity (lamp): the own entry waits for all 80 workers' own-ready; per-row-group
   readiness would start the first packets earlier.
3. local (LN) — the LN combine / normalize chain (~7 µs per tile-row on the critical path: many small chains and
   reduces with per-call init / reconfig).

## Refinement 1 — Port the base op's remaining input surface: HEIGHT_SHARDED + WIDTH_SHARDED input, and fp32_dest_acc_en=False
- Date: 2026-11-23
- What was done:
  - **fp32_dest_acc_en=False** (base op Refinement 1 mechanism): intermediate CB formats follow the DEST width (`T_acc`
    for `cb_sumsq`, `cb_inv`, `cb_rstd_b`, `cb_ln_sq`, `cb_mask`); the UnpackToDestFp32 tag list is empty under 16-bit
    DEST; LN reads `cb_x` directly (no Float32 copy) and every LN reduce runs Fast. Records stay Float32 pages (wire
    contract); the combine runs in the caller's 16-bit DEST (the hub-side fp32 combine stays `deferred`: not needed).
    The design's FPU-centering row 12' was not needed: the existing SFPU chains run in 16-bit DEST within tolerance.
  - **HEIGHT_SHARDED** (base op Refinement 4): k = 1, one row-group per shard (ragged last → `last_crt`), zero-copy
    `cb_x` (`cb_descriptor_from_sharded_tensor`, empty x accessor args), worker identity from a packed shard map in the
    common args (still no per-core RT args), ports placed off the input shard grid. HEIGHT output with one tile-row
    per shard on the same cores: `cb_out_block` is backed on the own slice of the output shard (no block, no own write).
  - **WIDTH_SHARDED** (base op Refinement 2): one group, k = #shards; the members meet in the stats table; LN combine
    generalized to several gather tiles (P > 32); ragged last shard at R = 1.
  - **Sharded output**: port walk mode `shard_w` (lanes = tile-rows, runs = a slot row inside one shard core, split at
    shard columns); host `Walk` replica in step; R_U keeps the DRAM staging and assembles into the sharded output.
  - **W tail on a sharded input**: masked in compute, never zeroed in the caller's shard — a mask block (row-0 masks,
    `prepare_reduce_mask`) applied with the gamma row-broadcast DestReuse multiply before plain reduces.
  - Reused: SF-GN schedule, stats round, relay / arrival counters, gather, compute chains, DestReuseRowBcast.
    Added: shard split + map, walk mode, mask block, multi-gather-tile LN combine, in-place output CB.
- Accuracy achieved: all gate cells within `helpers.TOLERANCES` (16-bit DEST bf16 (0.999, 0.015), bf8b (0.995, 0.05));
  new unit file 13/13 on shapes [640×384 HEIGHT/WIDTH RMS+LN, 640×1536 fp32 HEIGHT, 32×50 HEIGHT LN, 32×384 WIDTH LN
  (P = 48), 2×1×128×100 WIDTH (tail, crt = 8), 32×8192 WIDTH (ragged, P = 344), 16-bit DEST interleaved + sharded].
- Golden test progress (gate tier, targeted slices): fp32_dest=False interleaved 188 passed / 98 xfailed (EXCLUSIONS);
  sharded 128 passed / 30 failed / 11 skipped / 6 xfailed of 175 (HEIGHT 71/72 runnable; WIDTH ≈ 57/86); interleaved
  fp32-DEST regression sample 91/91 + test_regression 32/32 + loose perf cases 2/2; unit dir green.
- Perf (eval harness, 5 experiments, fastest / slowest chip): focus 35.56 / 35.95 µs, LN 29.56 / 29.83 µs — unchanged.
- Issues encountered: (1) an UnpackToDestFp32-tagged CB cannot be read through srcA — hit twice (masked S1 on
  `cb_x32`, AccumulateViaAdd over several LN gather tiles); (2) AccumulateViaAdd's `only_partial` fold of a lone tail
  tile with several tile-rows accumulates onto an unseeded DEST (rel RMS 0.14), replaced by the mask-block multiply;
  (3) L1: HEIGHT output shards of the gathered shape are G× the input shard — fixed by the in-place output CB and by
  dropping the LN Float32 x copy on sharded input. Remaining (Refinement 1b): WIDTH at large R × k (stats table cap /
  per-row CBs, ≈ 29 cells, RuntimeError); 128×8192 bf8b HEIGHT LN (output shard alone exceeds L1).
- Tests added: `tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_sharded_numeric.py`

## Refinement 1b — WIDTH_SHARDED at large R × k: on-chip pre-combine of the k shard members + row-chunked stats CBs
- Date: 2026-11-23
- What was done:
  - **Row-owner pre-combine** (CT path `pre_combine`, WIDTH_SHARDED input only): member m owns tile-rows
    `[m·rpo, …)`. Every member writes its per-row records into the owners' `cb_pre_table` (+ `sem_pre`); the owner
    gathers the k records of its rows as gather-tile rows (the hub gather's placement), transposes and combines them
    into ONE record per (row, stat) — RMS Σ; LN (mean, M2, K = mean) by the existing chains — and publishes it to the
    hubs (stats table G·R·nstats, kt = 1, whatever k is). After the fabric round the owner runs the existing
    cross-chip combine (P = G) and writes the final records (inv | mean, rstd) into every member's `cb_final_table`
    (+ `sem_final`); members normalize with them. Hubs wait for `n_owners` record writers instead of `n_active`.
  - **Row-chunked per-row CBs**: `row_chunk` (largest divisor of R within `ROW_CHUNK_L1_BUDGET` = 320 KiB, knob) sizes
    every per-row Float32 CB; the local-stats pass pops `cb_x_stats` (second zero-copy view of the shard) per chunk
    while `cb_x` stays whole for the normalize. One block size per CB ring (owner rows a multiple of the chunk, LN final
    set padded to nstats): a block-indexed helper cannot read a block that wraps the ring (first cut: PCC 0 / inf).
  - **LN**: centering shift K = tf32(mean) of the row itself in both combines and the normalize; the combine collapse
    is the exact ReduceTile path for any gather-tile count (AccumulateViaAdd over tagged `cb_ln_tmp` with ≥ 2-row
    chunks gave PCC 0.82).
  - **Ragged WIDTH shards at R > 1** (128×8192: 256 tiles / 3): x addressed with the shard pitch (chain `Strided`,
    reduce `with_row_stride`, raw x² loop pitch); the RuntimeError is gone.
  - Reused: stats round + port kernels, hub gather placement (factored `place_records` / `gather_partials`),
    transpose, RMS / LN combine and normalize chains (factored into templates on gather tiles / partition-width CB),
    writer record extraction. Added: pre / final tables, 2 semaphores, 6 CBs, the PC block functions.
- Accuracy achieved: every WIDTH gate cell within `helpers.TOLERANCES` (fp32 DEST bf16 (0.999, 0.01), 16-bit DEST);
  unit pins on [2048×384 RMS/LN, 2048×384 RMS 16-bit DEST, 640×1792 RMS (k = 56), 640×32 LN (k = 1, R = 20),
  2×1×128×128 LN, 2×1×128×100 LN (tail + R_U), 128×8192 RMS/LN (ragged)].
- Golden test progress (gate tier, targeted slices): WIDTH_SHARDED 85 passed / 2 xfailed of 87 (was ≈ 57/86 + 29
  RuntimeError / L1); HEIGHT + interleaved sample 454 passed / 88 xfailed / 11 skipped / 1 failed (the known
  128×8192 bf8b HEIGHT LN output-shard L1 overflow from Refinement 1); test_regression 32/32; unit dir 55/55.
- Perf (eval harness, 5 experiments, fastest / slowest chip): focus 35.52 / 35.89 µs, LN 29.59 / 29.84 µs — unchanged
  (the interleaved path is the same code, refactored into templates).
- Issues encountered: CB ring wrap of mixed block sizes (fixed by uniform chunks); multi-row AccumulateViaAdd LN
  combine on tagged input (switched to ReduceTile); a name clash of the chunk variable with the port loop's `rc`.
- Tests added: `tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_width_precombine.py`

## Refinement 2 — TRANSPORT: bulk packet efficiency on interleaved final pages (focus case)
- Date: 2026-11-23
- What was done:
  - **Measured the premise first; it was wrong.** Ablations on the focus case (zone profile of a steady call): bulk
    packets landed as one contiguous write in the peer port's L1 (no scatter, no DRAM) → own entry unchanged (~15 µs);
    `fabric_all_gather` moves this slice (640×384, 1×4 ring, 2 links) in 22.2 µs with 1–2-page runs (dim 2) and
    22.6 µs with 7-page runs (dim 0). The own entry was instead bound by per-packet work on both port RISCs:
    ~1200 BRISC ticks per ~10 KiB packet (router slot wait ~40, flush ~15; the rest walk, header-builder address
    re-encoding per chunk, accessor lookups, waiting for a reader doing one NoC read per page and one barrier per
    4-packet batch). A minimal-CPU own entry of the same 24 packets at full size ran 2.5× faster.
  - **Sender**: router-side bank addresses computed once (`bank_base` / `bank_pitch`); non-increment packet headers
    written directly (scatter command fields as ten 32-bit stores, layout pinned by static_asserts); no packet walk
    (runs from the reader).
  - **Reader**: pipelined reads (one NoC transaction id per packet, `READ_DEPTH` = 4 ≤ `BATCH_CHUNKS`, each packet
    pushed as its own reads land; every read packet is handed over before blocking on a relay arrival — without that
    the ring deadlocks); own entry read back from the own output pages one read per bank run (`OWN_SRC_DST` = 1,
    0 = the old L1 pull); bank-table read addresses; writes each slot's runs to `cb_port_desc` (CB 32, 2·batch × 64 B).
  - Knobs: `READ_DEPTH`, `OWN_SRC_DST`, `SPLIT_RUNS` (0: measured +1.2 µs at 1 — fewer, fuller packets with more
    scatter chunks are slower here; host walk replica and device walk both implement it).
  - Measured and dropped: non-blocking header-only increments + a larger header ring (+0.4 µs); an own-entry run table
    walked during the local work with pre-built routed headers (2D route setter ~0.5 µs per header; loading
    precomputed runs is slower than re-walking: +3 µs); the reader writing the command fields into per-slot routed
    headers so the sender only sends (30.75 vs 30.52 µs — the work is already split about evenly over the two RISCs).
  - Reused: the walk, CB layout, arrival-counter protocol, header builders (increment packets); Added: `cb_port_desc`,
    the bank tables, the pipelined reader loop.
- Perf (eval harness, 5 experiments, `eval.ccl_perf compare` vs the pre-refinement build measured in this session):
  focus 640×384 RMS **35.514 → 30.452 µs** fastest (−5.06), slowest 35.889 → 31.354; LN 640×32 + γ + β
  **29.592 → 27.726 µs** fastest (−1.87), slowest 29.841 → 27.935; both "win" (noise 0.05). Not dispatch-bound.
- Accuracy achieved: unchanged (transport only moves bytes; same pages, same values) — unit dir 56/56 within
  `helpers.TOLERANCES` on all its shapes (focus 640×384 RMS/LN, 32×1792, fp32 2×1×128×256, R_U 17×50 / 32×50, sharded
  HEIGHT/WIDTH numerics, WIDTH pre-combine).
- Golden test progress: targeted slice of 254 cells (every 6th bf16 / fp32-DEST / RM-gamma cell over all shapes,
  layouts, topologies, links, cluster axes and alignments, plus bf8b, fp32, fp32_dest=False samples and both loose perf
  cases): 228 passed, 25 xfailed (EXCLUSIONS), 1 failed — 128×8192 bf8b HEIGHT_SHARDED LN, the known L1 overflow
  from Refinement 1 (CB region ends at 1.39 MB; the same failure on the pre-refinement build).
- Issues encountered: a ring deadlock with the pipelined reader (fixed: retire in-flight packets before an arrival
  wait); read depth above the batch size deadlocks (the sender pops whole groups: clamped and static_asserted).
- Remaining headroom (finding, not a queue item): own entry ≈ 11 µs for 24 packets per port (~450 ns per packet vs
  ~345 ns link floor for a full packet), bound about evenly by both port RISCs; the bigger share of the call is now the
  serial chain before the bulk starts (~11 µs: stats → normalize → own-ready) — Refinement 3.
- Tests added: `tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_perf_probe.py`
  (no-check profiling helper for zone timelines).

## Refinement 3 — OVERLAP: start the bulk before the slowest worker finishes, and shorten the stats → normalize → own-entry chain
- Date: 2026-11-23
- What was done:
  - **Measured the chain first** (zones, focus case): stats ready → gather 0.55–2.4 µs (80 workers × G reads hitting
    the 2 hub cores: the hot spot set the slowest worker) → combine ≈ 0.9 → normalize ≈ 1.0 → own DRAM write
    1.1–2.3 → own-ready → first packet +1.6 µs (the port reader pushed only after `READ_DEPTH` packets were issued).
  - **Stats-table multicast** (kept): the hub readers multicast each completed slot into the same `cb_stats_table`
    address on the whole program grid (the fwd hub the own slot after `records_local`, each hub its upstream slots as
    they land). The last slot is linked to the stats-ready flag (kernel_lib mcast pipe Flag protocol), and the reader
    re-arms `stats_arrived` after the sender's `done`. Workers gather from local L1: a flat 0.25 µs, and
    `cb_record_landing` is deleted. The first cut (unlinked + barrier before the flag) won only 0.28 µs on focus and
    regressed LN.
  - **Eager push** (kept, `EAGER_PUSH = 1`): the port reader pushes every in-flight packet whose reads already
    landed after each issue (non-blocking trid check). Own-ready → first packet dropped from 1.6 to 0.8 µs, and every
    relay entry starts sooner.
  - **Own-ready bands** (`OWN_BANDS`, parked at 1): a persistent global-semaphore counter per band of row-groups on
    every port core, re-armed every call; the own entry waits per packet only for its runs' bands. At 16 bands it
    measured +1.4 / +2.4 µs: the own entry's DRAM read-backs started during the workers' 480 KiB own-write burst, and
    the entry ran ~1.5 µs longer.
  - **x·γ in the stats shadow** (`XG_PRE`, parked at 0): x·γ → `cb_xg` during the stats round, then one multiply
    after it. Normalize 1.03 → 0.65 µs in the zone profile, call time unchanged (29.21 vs 29.18 µs). The precision
    baseline passes with it on.
  - Measured and dropped: the L1 own pull (`OWN_SRC_DST = 0`, so own-ready could precede the DRAM write) at +4.5 µs.
  - Reused: hub reader / sender stats protocol, the gather placement (`place_records`), the reader pipeline, the
    RMS chains. Added: slot multicast + runtime args (stats address, slot bytes, own / received slots), band counters
    + args, `EAGER_PUSH` / `OWN_BANDS` / `XG_PRE` / `XG_L1_BUDGET` knobs, `cb_xg`, `ps_own_pkt0` profiling marker.
- Perf (eval harness, 5 experiments, `eval.ccl_perf compare` vs the post-R2 build measured in this session):
  focus 640×384 RMS **30.448 → 29.205 µs** fastest (−1.24), slowest 31.377 → 30.001; LN 640×32 + γ + β
  **27.727 → 26.322 µs** fastest (−1.41), slowest 27.935 → 26.711; both "win" (noise 0.05). Not dispatch-bound.
- Accuracy achieved: unchanged numerics on the default path (the multicast moves the same records); unit dir green
  within `helpers.TOLERANCES` (focus 640×384 RMS/LN, R_U, fp32, sharded HEIGHT/WIDTH, WIDTH pre-combine, precision
  baseline incl. `XG_PRE = 1` and `OWN_BANDS = 16` runs of the main file).
- Golden test progress: targeted slice of 63 cells (perf shapes, interleaved RMS/LN aligned, w/h non-aligned, Linear,
  1 link, cluster_axis 0, HEIGHT/WIDTH sharded, fp32, bf8b, fp32_dest=False, no gamma): 57 passed, 2 skipped,
  4 xfailed (EXCLUSIONS); test_regression 32/32.
- Issues encountered: the per-slot unlinked multicast + write barrier before the flag cost ~1.7 µs between the last
  arrival and stats-ready (LN regressed) — fixed by the linked last payload; band gating loses while the own entry
  reads its source from DRAM during the own-write burst.
- Remaining headroom (finding): the bulk (own entry + relays ≈ 14 µs) is still port-CPU bound (Refinement 2), and the
  chain before it is ≈ 10 µs, of which the own DRAM write burst + read-back is the largest piece left. A cheaper L1
  source for the own entry would make both bands and x·γ pay.
- Tests added: none (existing unit files cover the paths; `ps_own_pkt0` marker for zone timelines).

## Refinement 4 — LN critical-path compute chain (indexer-k case)
- Date: 2026-11-23
- What was done:
  - **Measured first** (zones, LN 640×32 + γ + β, one tile per row, k = 1, 20 workers): after the gather, transpose
    0.8 µs, mean chain + reduce 1.2, d chain + broadcast 0.85, M2 chain (17 elements) 2.4 + reduce / finalize 0.7,
    rstd broadcast 0.2, normalize 1.2. Local stats 2.8 µs before the records. Each chain element costs ~0.1–0.15 µs
    on one tile: the cost is element count.
  - **One-pass fused combine** (`ln_combine_fused`, `gather_tiles == 1`, fp32 DEST): shift c = K_own,
    A = n(m − c), B = T − n(m − K)² + n(m − c)², d = ΣA/W, M2 = ΣB − W d², rstd in a column-0 finalize. One chain:
    the row-form gather tiles are copied from `cb_gather_rows` and transposed in DEST (no `transpose_gathered` pass),
    two fused SFPU passes (faces 0/2 only when P ≤ 16), two SFPU in-tile row sums, two packs (d → `cb_ln_m`,
    rstd → `cb_inv`).
  - **Normalize**: d / rstd broadcast inside the chain (`UnaryBcast`, same FPU column broadcast as the old
    `unary_bcast`), and ((x − K) − d)·rstd as one SFPU pass (`LnCenterScale`, every LN path).
  - **Local stats, cw == 1** (fp32 DEST): S1 is one exact row sum packed into both `cb_partial` and `cb_ln_m`, which
    drops the duplicate reduce without an srcA copy. T is (x − K)² + row sum in the same chain when the member has
    no W tail.
  - Gates: 16-bit DEST, `gather_tiles > 1`, pre_combine, `cw > 1` and W-tail members keep the previous chains.
  - Reused: the record format, reader placement, `cb_ln_m` / `cb_inv` / `cb_ln_shift`, the γ / β dest-reuse elements,
    the FinalizeInvRms SFPU style, the transpose_dest mechanism. Added: chain elements `LnCombCenter`, `LnCombTerms`,
    `LnCombFinalize`, `LnCenterScale`, `SfpuRowSum`, `TransposeDestElem`, `MulScalarElem`; no CB, kernel or host
    change.
- Perf (eval harness, 5 experiments, compared step by step with `eval.ccl_perf compare`): LN **26.324 → 23.269**
  (fused combine) **→ 22.090** (local stats + in-chain broadcast + fused normalize) **→ 21.417 µs** (in-DEST
  transpose) fastest; slowest 26.714 → 21.820. Focus 29.217 → 29.207 µs (null). Not dispatch-bound.
- Accuracy achieved: LN 640×32 bf16 PCC 0.999998 / rel RMS 0.0020; LN fp32 offset 50 (128×384 Linear) PCC 1.000000 /
  0.0015; LN 640×32 fp32 offset 50 PCC 1.000000 / 0.0015 (limits: bf16 0.999 / 0.01, fp32 0.9999 / 0.005).
- Golden test progress: targeted LN slices — 1x1x640x32, 32x17, 17x64, 1x1x32x50, 1x1x17x1280: 179 passed, 45
  xfailed (EXCLUSIONS); 128x8192, 1x640x1792, 640x384: 185 passed, 2 skipped, 27 xfailed; test_regression 32/32
  (`test_layer_norm_distribution[big_offset]` included); unit dir 57/57.
- Issues encountered: none blocking. The DEST limit (4 fp32 slots, half sync) splits the combine into two SFPU passes.
- Remaining headroom (finding): after the gather, combine 2.2 µs → normalize 1.1 → own write 1.1, then the bulk
  (~7 µs, port-CPU bound). The stats round (~7 µs records → stats-ready) is now the largest serial term in the LN call.
- Tests added: `test_perf_probe_ln` in `test_all_gather_rms_norm_perf_probe.py` (no-check LN profiling helper for zone
  timelines).

## Refinement 4b — LN critical-path compute chain (indexer-k case) (debug: fix gate violations)
- Date: 2026-11-23
- What was done: ran the full golden suite to find the gate's regressions. One R4 regression:
  `1x1x64x1000` w_non_aligned LN WIDTH_SHARDED (fp32 DEST) hit `Program size (71536) too large for kernel config
  buffer (70656)`: R4's one-tile-column local-stats fast path (S1 once, T in-chain) was a runtime `cw == 1` branch, so
  every fp32-DEST LN build compiled both it and the general reduce paths. Member cw = min(cw_nom, Wlt − m·cw_nom), so
  "every member has cw == 1" ⇔ `cw_nom == 1`: `ln_fused_local` is now `DST_ACCUM_MODE && cw_nom == 1` and the S1
  branch is `if constexpr`. The LN perf case (cw_nom == 1) compiles the same fast path as before.
- Perf (eval harness, 5 experiments): LN 640×32 + γ + β 21.273 µs fastest / 21.674 slowest (R4: 21.42 / 21.82);
  focus 640×384 RMS 29.196 / 29.997 µs. Not dispatch-bound.
- Accuracy achieved: unchanged numerics (same chains; only the build gate moved).
- Golden test progress: FULL suite: 1457 passed, 139 skipped, 123 xfailed, 1 failed, 0 hangs. Unit dir 57/57.
- Issues encountered: `128x8192` bf8b HEIGHT_SHARDED LN (cluster_axis 0, γ + β) fails with a static-CB / L1-buffer
  clash. This predates R4: it fails identically with the R3 compute kernel, and the host CB plan has not changed since
  R3. The caller's input (272 KB) and output (544 KB) shards leave about 540 KB for CBs. The op needs about 1.2 MB:
  `cb_ln_sq` 512 KB (bf16, cw = 256), `cb_gamma` / `cb_beta` 256 KB each. Finding: wide-row sharded LN needs a
  column-chunked T (chain + accumulating reduce per column window) and a γ / β-streaming column-chunked normalize to
  fit. That is a scheme change outside R4's scope.
- Tests added: none.

## Perf 1
- Date: 2026-11-23
- **Focus cases** (LOOSE_CASES, both perf-flagged, both measured; QB2 1×4 ring, FABRIC_2D_TORUS_XY, 2 links, bf16,
  fp32 DEST, γ ROW_MAJOR sticks, interleaved DRAM): RMS 640×384 (960 output tiles) and LN 640×32 + γ + β (80 tiles).
  The config was optimized exactly as declared (every knob in SUPPORTED); no proxy.
- **Instrumentation**: the op's permanent `AGRN_ZONE` scopes (agrn_common.hpp; free unless `AGRN_PROFILE_ZONES=1`;
  `perf_instrumentation.hpp` / `MaybeDeviceZoneScope` does not exist in this tree, AGRN_ZONE is its equivalent) cover
  every stage; added `wr_stage`, `wr_stage_bar`, `wr_stage_inc`, `pr_stage_wait`, `pr_push0` for the new path.
  Ablation switches (`AGRN_ABLATE=NO_BULK,NO_OWN_WAIT` → kernel defines, output garbage) + harness copy
  `tests/.../all_gather_rms_norm/perf_ablate/perf_ablate.py` (eval ccl_perf protocol, golden check skipped when ablated).

### Measured breakdown (eval harness, traced, fastest chip; zone timelines from the perf probe)
| case | full | chain alone (`NO_BULK`) | bulk unblocked (`NO_OWN_WAIT`) | overlap loss (full − slowest part) |
|---|---|---|---|---|
| focus 640×384 RMS | 29.18 µs (slowest 29.99) | 10.39 | 25.58 | 3.6 |
| LN 640×32 | 21.27 µs (slowest 21.68) | 12.58 | 17.05 | 4.2 |

| part (focus) | measured | part SOL | gap |
|---|---|---|---|
| bulk (own entry 10 µs / 24 pkts per port at ~420 ns, relay ~5 µs / 12 pkts, landing tail ~3 µs) | ~18.5 | 7.6 (links, roofline) | ~11 |
| chain: x read 1.7 → records 3.2 → stats round → 7.0 → combine + normalize → 8.3 → own DRAM write → 10.4 → first packet 11.2 | 10.4 | ~4.5 (x 0.49 MB + 1 relay hop + compute + own write 0.49 MB) | ~6 |
| LN bulk (2-hop store-and-forward of 40 KB slots) | ~8.6–10 | 0.6 link + ~1 hop latency | ~8 |
| LN chain (local stats 2.6, stats round ~3.5, combine 2.0, normalize 1.0, own write 1.1, hand-off 1.0) | 12.6 | ~6 | ~6.5 |

Ranked: (1) bulk relay + tail (TRANSPORT, both cases); (2) the own write → own_ready → DRAM read-back hand-off into
the own entry (OVERLAP, both cases, ~3 µs on focus); (3) stats-round relay; (4) LN compute chain.
The chain and the bulk are strictly serial (the bulk waits for own_ready): the whole overlap loss is that hand-off.

### Portfolio (cap: 2 experiments)
1. **fabric_mcast_relay** (TRANSPORT): replace the store-and-forward relay of the far slot (and the stats round) by
   2D chip-level multicast.
2. **own_entry_port_l1_staging** (OVERLAP): workers write their normalized pages straight into L1 staging in the
   port's packet order, so the own entry needs no DRAM read-back after the own write.
Not tested (cap): interleaving relay packets with the own entry, bidirectional stats round, LN combine chain.

### Verdicts
| idea | kind | verdict | before → after (fastest / slowest, µs) | domain |
|---|---|---|---|---|
| fabric_mcast_relay | TRANSPORT | WIN in isolation on LN-size slots only; **not graduated** | isolated bulk bench: 640×32 10.68 → 9.15 (−14.8 %); 640×384 19.72 → 19.57 (flat); 2 KB stats-size slot 5.97 → 6.51 (+9 %, the 2-hop ready in the fence) | QB2's 1×4 ring is a 2×2 square: every hop turns, and the 2D router only turns N/S → E/W, so straight 2-chip ranges are **inexpressible** here; each chip has one direction with an expressible 2-chip path, so the busiest link carries 2 lanes instead of 1.5 (why the focus is flat). Galaxy torus rings untested. Not graduated this round: the integration is a scheme change of the transport (per-chip route direction from `get_eth_forwarding_direction`, 2-hop ready fence, arrival thresholds per segment, new sender) for a ~1.5 µs LN-only gain; the subagent's sketch is in `perf_experiments/fabric_mcast_relay/` — a round-2 candidate. |
| own_entry_port_l1_staging, staging on the 4 port cores | OVERLAP | REGRESSION | focus 29.21 → 36.61 | 80 workers into 4 cores: ~30 GB/s ingress per core, staging 6.4 µs median vs 1–2 µs DRAM write |
| own_entry_port_l1_staging, **distributed staging** (chunks of run_pages on the active workers, one readiness counter per port) | OVERLAP | **WIN — graduated** | see below | everywhere (feasibility carve-out only: L > 4 or interleaved nb > 16, the CT tables) |
Other measured options of the staging bench (focus): per-chunk counters 29.56–29.60 (atomics ~0.7 µs); chunks on the
first 36 workers 30.45; chunk 4 pages 28.87 (flat vs 7); first 14 pages on the ports 30.03; first packet handed over
early 29.56; no atomic barrier before the DRAM own write: worse on Linear (43.56).

### Graduated: own-entry staging on the active workers (one path, no dual path)
- Each port's own walk is cut into `run_pages`-page chunks; chunk g = lane·nch + c lives in `cb_stage` (CB 34) of
  active worker g·n_active/(L·nch) (round-robin with slots once chunks outnumber workers). Each worker computes every
  page's walk position from compile-time args (per-bank lane offsets, no per-core runtime args), writes it to its
  chunk home (NoC1), barriers, increments one readiness counter on every port (a new global semaphore), atomic
  barrier, then the unchanged DRAM own write + own_ready. The port reader waits the counter once and reads each own
  packet with ≤ 2 L1 reads instead of ≤ 4 DRAM bank runs; it re-arms the counter every call (trace-safe). Past
  `STAGE_L1_BUDGET` (128 KiB of `cb_stage` per core) a lane's walk tail keeps the DRAM read-back.
- Replaced code: the per-packet DRAM read-back is gone from every staged packet; it survives only for an unstaged
  walk tail and the CT-table carve-out. No knob, no predicate on shape / topology / placement.
- Carve-outs: none earned by measurement (the subagent's Linear 640×384 +3.9 % did not reproduce on the graduated
  code: +0.8 % fastest, −2.7 % slowest).
- Precision: transport only (same pages, same values) — no numeric change.

| cell (bf16, RMS, γ RM, cluster_axis 1, 2 links) | before (fastest / slowest) | after | Δ fastest |
|---|---|---|---|
| **focus 640×384 Ring, interleaved** | 29.182 / 29.993 | **28.365 / 28.777** | −0.82 (−2.8 %), slowest −1.22; `ccl_perf compare`: win |
| **LN 640×32 + γ + β Ring** | 21.268 / 21.675 | **20.784 / 21.154** | −0.48 (−2.3 %), slowest −0.52; `ccl_perf compare`: win |
| 640×384 Linear | 40.972 / 44.241 | 41.295 / 43.046 | +0.8 % (noise), slowest −2.7 % |
| 640×384 HEIGHT_SHARDED Ring | 31.618 / 32.338 | 31.852 / 32.546 | +0.7 % (noise) |
| 640×1792 Ring | 105.192 / 107.982 | 79.198 / 81.397 | **−24.7 %** |
| 640×1792 Linear | 127.812 / 148.281 | 108.856 / 114.216 | **−14.8 %** |
| 640×1792 HEIGHT_SHARDED Ring | 127.431 / 130.908 | 127.445 / 129.483 | flat |
| 2048×1280 Ring | 255.224 / 259.626 | 181.869 / 184.154 | **−28.7 %** |
| 2048×1280 Linear | 342.733 / 391.715 | 314.538 / 344.234 | **−8.2 %** |
| 2048×1280 HEIGHT_SHARDED Ring | 197.518 / 201.849 | 195.423 / 198.350 | −1.1 % |
(5 experiments each, same session; before = commit b018d64bbba. Subagent sweep on its copy, also flat: cluster_axis 0,
1 link, R_U 64×1000.) Not dispatch-bound.

- Golden: gate tier 1425 passed, 123 xfailed, 11 skipped, 1 failed — the pre-existing `128x8192` bf8b HEIGHT_SHARDED
  LN L1 clash (CB region still ends at 1,389,248 B, identical to R4b: staging adds nothing there);
  test_regression + test_ring_mock 32 passed / 128 skipped.
- Summary: all ideas measured; 1 graduated, 1 isolated win deferred (mcast, LN-only, scheme change), 1 variant
  regression; focus −0.82 µs (−2.8 %), LN −0.48 µs (−2.3 %), wide slices −8…−29 %; no regression on the guard set.
- Remaining headroom for round 2: bulk relay + tail (~8 µs focus; LN relay turnaround — the mcast sketch), stats round
  (~3.5 µs), LN compute chain (~5.6 µs on one tile per row).
- Artifacts: `perf_experiments/{fabric_mcast_relay, own_entry_port_l1_staging, coordinator_round1}/`,
  `tests/.../all_gather_rms_norm/{perf_ablate, perf_experiments}/`.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (dataflow "scatter block pages to per-chunk L1 homes in a consumer's walk order" — would belong with `kernel_lib/ccl` / `local_copy_helpers_dataflow`) | capability | no helper maps a block page to its position in another core's packet walk or scatters pages to per-chunk L1 homes with a single completion counter; the path is raw `noc_async_write` + `noc_semaphore_inc` | — (no helper path) | focus 28365 (vs 29182 DRAM read-back) | `kernels/agrn_worker_args.hpp` `stage_dist_block` |
| no helper covers this (port-side chunked L1 gather into the fabric packet CB) | capability | the port reader's chunk reads (≤ 2 L1 pieces per packet, chunk-home arithmetic) have no kernel_lib counterpart; raw `noc_async_read` under the existing transaction-id pipeline | — | as above | `kernels/all_gather_rms_norm_port_reader.cpp` staged branch of the own entry |
No raw LLK (compute) was added this round.
