# Changelog: layer_norm_adaln

## Phase 0 — Core Implementation
- **Date**: 2026-11-16
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Pivot-shifted two-pass LayerNorm with a rows × W-slices split over the full grid, rotating-mcast all-gather of per-slice stats and Chan's parallel combine, optional per-call adaLN `·w + b`. One `ttnn.generic_op` dispatch.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], alignment=[tile_aligned], rank=[4], affine=[no_affine, dynamic_wb], affine_dtype=[bfloat16, none], affine_layout=[TILE, none], memory_layout=[INTERLEAVED]. `TARGET − SUPPORTED = ∅`.
- **Accuracy achieved** (fp32 DEST, HiFi4; `test_layer_norm_adaln_precision_baseline.py`, 4 shapes × offset {0, 50} × affine): PCC ≥ 0.99999, max_abs_err ≤ 0.21, rel-RMS ≤ 0.011 (no affine ≤ 0.0061), ULP p99 ≤ 2.5 without affine / ≤ 17 with. 16-bit DEST: PCC ≥ 0.99997, rel-RMS ≤ 0.019.
- **Golden suite at Phase 0**: 16 / 16 supported cells passing, 48 INVALID skipped, 37 regression cells passing; supported_fail = xpass_drift = xfail_wrong_mode = 0 (`verifier_report.json`).
- **Perf at Phase 0** (device kernel duration, µs, no-affine / affine): 256×3072 37.4 / 39.1; 2048×3072 81.4 / 108.3; 2304×3072 88.2 / 110.8.
- **Issues encountered (verifier fixes)**:
  - `fp32_dest_acc_en=False` violated the prompt's large-offset MUST rule (rel-RMS up to 0.998 at mean 50): the bf16-accumulated first-pass mean is too coarse to carry. Fixed with a 16-bit-DEST-only `Σ(x−p)` pass and a pivot-relative `[p, d, M2]` payload; fp32 DEST path unchanged.
  - Affine row-0 reads used one DRAM barrier per tile → chunked (`AFFINE_STREAM_CHUNK`): 2304×3072 affine 122.9 → 110.8 µs.
  - Writer waited for write completion per tile row → `noc_async_writes_flushed` per row, one barrier at the end.
  - `GATHER_TILE_CAP` check now uses the mode-dependent `stats_per_row`.
  - `l1_ledger.md` brought current; `op_design.md` gained a "Verifier amendments" section.
- **Tests added**: `test_layer_norm_adaln_precision_baseline.py` (32 cells, gates at `default_tolerance`), `test_layer_norm_adaln_perf.py` (device-ns measurement, never asserts). Acceptance test `test_layer_norm_adaln.py` was the implementer's.

## Perf 1
- **Date**: 2026-11-16. Perf tournament round 1 (coordinator + 8 `perf-part-optimizer` ideas). Device: Blackhole, 12×10 grid, 1350 MHz.
  `SUPPORTED` unchanged. Precision contract unchanged (bf16, HiFi4, `fp32_dest_acc_en` as given, approx off) in every measurement.
- **Focus shape**: `LOOSE_CASES` is empty (no perf-flagged case), so I picked the largest FLUX shape, (1,1,2304,3072) dynamic_wb fp32 DEST, as the focus. Secondary shape: (1,1,256,3072) dynamic_wb. Only cells > 32 tiles were measured; noise is ±2–3% within one card, ±3–5% across cards.

### Measured breakdown (before; fresh build, one profiled run + cumulative ablation)
- **Instrumentation (permanent).** I added `kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`). It compiles in only with the profiler and `LN_ADALN_PERF_ZONES=1`. Zones `r_*`, `c_*` and `w_*` sit at every stage boundary, with wait, issue and barrier split out. Harness: `tests/.../test_layer_norm_adaln_breakdown.py`; parser: `perf_experiments/breakdown/zones.py`.
- **2304×3072 wb fp32 = 107.8 µs** (S=3, bh=2, bw=32, 120 cores). Zone p50 per core, start→end:
  | stage | p50 time | detail |
  |---|---|---|
  | x load | 0→28.3 µs | slowest core 54 µs; `r_load_issue` 6.7 µs, `r_load_barrier` 10.8 µs |
  | `r_affine` | 19.7 µs, 28.7→52.7 | starts after the x load |
  | `c_mean` | 7.4 µs | |
  | `c_dev_sq` | 7.5 µs | |
  | `c_publish` | 1.6 µs | |
  | `c_prep_affine` | 10.8 µs | starved: waits on `r_affine` |
  | `c_combine` | 7.2 µs (math thread 9.0) | |
  | `c_bcast_stats` | 2.1 µs | |
  | `c_normalize` | 29.2 µs | overlaps the writes |
  | store end | 97.5 µs | |
- **Cumulative ablation (2304 wb fp32).**
  | variant | µs |
  |---|---|
  | full | 107.8 |
  | no-read | 81.9 |
  | no-read + no-write (compute + sync) | 58.4 |
  | no-compute (DM + sync) | 92.1 |
  | DM read-only | 59.2 |
  | DM write-only | 52.0 |
  | sync-only | 11.6 |
  | no-peer | 111.4 |

  So read (~48 µs, ~300 GB/s) and write (~40 µs) run serialized. Removing compute saves only 15.7 µs. The op is DM-dominated, with the stats chain and the w/b fetch serialized between the read and write phases.
- **DM ceiling (`/perf-ceiling-dm` Step 4b; the `noc_estimate` binary is not built on this branch).** DRAM floor ≈ 28.3 MB / ~470 GB/s ≈ 60 µs. Measured DM-only above sync was 80.5 µs, about 0.7 of the floor. The read is DRAM-bound at about 300 GB/s regardless of NoC or issue depth (confirmed by I5).
- **256×3072 wb fp32 = 40.0 µs** (S=12, bh=1, bw=8). Ablation: sync-only 9.2 µs, DM-only 15.7, compute-only 29.8, no-peer 32.9. This cell is compute-bound: `c_combine` on the math thread is 20.8 µs (S=12 SFPU combine), and the exchange rounds take 7.8 µs.
- **Ranked bottlenecks:**
  1. On large shapes: DM-bound read and write, plus critical-path serialization. The w/b fetch sits after the x load and starves `prep_affine`; the stats chain sits between load and exchange.
  2. On small-H shapes: the S-peer combine and the choice of S.

### Portfolio and verdicts
Same-card means median of 3 alternating runs in one process; other numbers are single runs from the subagent's sweep.

| # | idea | verdict | before → after (µs) | domain |
|---|---|---|---|---|
| I1 | affine_prefetch: fetch w/b before or with x; prep before mean | **REGRESSION** as an idea (+4–15%: w/b reads compete with the DRAM-bound x load). **Side win `base_one`**: one 576 B row-0 read per w/b tile instead of two 64 B reads | base_one: 2304 wb fp32 112.0 → 105.3; 2048 wb fp32 110.1 → 97.5; no_affine unchanged | everywhere, bit-identical |
| I2 | stats_row_interleave: per-row mean → pivot → dev_sq as rows land | **WIN** | same-card on HEAD: 2048 noa fp32 75.3 → 70.8, wb fp32 94.3 → 89.8; 2304 noa 16-bit 89.1 → 84.6; bh=1 cells flat | everywhere, bit-identical |
| I3 | combine_fast: fused SFPI fold on faces 0/2, packed payload | **WIN** | 256×3072 wb fp32 39.7 → 28.7; 32×3072 wb 16-bit 41.0 → 22.8; 2304 wb fp32 110.9 → 107.5 | everywhere; fp32 metrics identical, 16-bit equal or better |
| I4 | bcast_reuse_normalize | **WIN** (helper `.block_size(DEST_AUTO_LIMIT)`): 1.00–1.09× on all 16 cells. Raw `reuse_blk` (srcB-broadcast dest-reuse, deletes bcast/prep stages): 1.07× on fp32 wb but a **measured regression** of 0.968× at 2048 wb 16-bit, so not graduated | normalize 29.2 → 22.0 µs; 2304 wb fp32 111.8 → 106.0 | blk: everywhere |
| I5 | dm_dual_noc: split reads and writes across both RISCs/NoCs, stagger, outstanding caps | **NULL / REGRESSION**. Reads are DRAM-bound at ~300 GB/s; splits were +7–16% on several cells. Write-only DM improved 6% but is hidden in the full op | — | not graduated |
| I6 | row_pipeline: 1-row blocks, 2-block stats lookahead, ping-pong input/pivot CBs | **WIN in isolation**: noaff −9–13% on multi-row cells; wb focus flat (−0.5%) | 2304 noa fp32 88.4 → 80.5 | **not graduated**: it restructures the same overlap as I2 (subsumes it) and conflicts with I3; I2 was chosen (smaller, bit-identical, and wins on wb where I6 was flat). The remaining noaff gain needs a re-port onto HEAD (round 2) |
| I7 | slice_count_costmodel: S choice that accounts for exchange and combine cost | **WIN**. I re-fitted it after I3 on a forced-S re-sweep of current kernels | 32×3072 noa fp32 18.4 → 15.7; 256×3072 wb fp32 27.8 → 26.0, noa 26.0 → 23.4 | everywhere (S=12 → 6 on the W=3072 small-H shapes; big shapes keep S=3, confirmed by same-card A/B) |
| I8 | affine_mcast (reuse-shared broadcast, added after the first aggregation): one injector per grid column fetches w/b and multicasts down the column | **WIN** | same-card: 2304 wb fp32 96.6 → 84.2 (−12.8%), wb 16-bit 99.2 → 93.4; 2048 wb fp32 89.6 → 78.0 (−13.0%); every other cell flat | everywhere: full and partial grids, S==1, both DEST modes; bit-exact |

### What graduated
I1 `base_one`, I4 `blk`, I3, I7, I2 and I8, all as the op's single unqualified path. Each replaced path was deleted:
- the two-read w/b fetch;
- the per-tile chain inits;
- the library-op combine/publish and the unpacked payload (`stats_per_row` removed);
- the all-rows stats order;
- the reader's affine path, the `AFFINE_STREAM_*` streams and their CBs (now aliased onto the resident CBs).

**Carve-outs: none.**

**Regression found and fixed during graduation.** The large multi-block guard shapes caught L1 overflows. The op's L1 budget (`get_max_worker_l1_unreserved_size()`) over-counts the CB region by the kernel-config ring (~70 KB on BH). This bug was already there: the baseline OOMs on 23040×2464 wb fp32. The Perf 1 footprint changes moved the failure onto 23040×2464 noaff fp32 and 23040×3072 wb 16-bit, which the baseline ran. The budget is now the device's `cb_limit`. Selections for INPUTS shapes are unchanged, and all 5 large probes run with PCC ≥ 0.99986 (`perf_experiments/coordinator/large_fit.jsonl`).

### Whole op, before → after
Same card (`--device 0`), one run each, pre-tournament op files vs HEAD. Guard set = every INPUTS shape > 32 tiles plus an S==1 rows_only representative (8192×256) × affine × DEST mode. Data: `perf_experiments/breakdown/final_same_card_{baseline,head}.jsonl`.

| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304×3072 (focus)** | **110.9 → 84.3 (−24.0%)** | 112.9 → 93.0 (−17.6%) | 87.1 → 79.6 (−8.5%) | 99.3 → 85.2 (−14.2%) |
| 2048×3072 | 111.5 → 78.2 (−29.8%) | 111.0 → 86.9 (−21.7%) | 81.4 → 70.4 (−13.4%) | 94.6 → 78.7 (−16.7%) |
| 256×3072 | 39.7 → 26.0 (−34.6%) | 48.6 → 28.7 (−40.8%) | 37.7 → 23.0 (−38.9%) | 44.6 → 26.0 (−41.7%) |
| 32×3072 | 31.4 → 18.4 (−41.3%) | 40.9 → 21.3 (−48.0%) | 29.8 → 15.6 (−47.5%) | 37.0 → 18.3 (−50.5%) |
| 256×1024 | 19.5 → 15.8 (−19.0%) | 23.0 → 18.1 (−21.4%) | 18.1 → 12.9 (−28.8%) | 21.5 → 15.1 (−29.6%) |
| 128×512 | 15.7 → 11.7 (−25.8%) | 17.4 → 12.8 (−26.4%) | 13.6 → 10.2 (−25.0%) | 15.1 → 12.2 (−19.7%) |
| 8192×256 (S=1) | 44.7 → 33.4 (−25.4%) | 48.4 → 36.7 (−24.2%) | 35.1 → 28.5 (−18.7%) | 39.4 → 31.7 (−19.5%) |

**Guard-set no-regression:** all 28 cells are faster. Golden: 52 passed, 49 skipped. Acceptance and precision baseline: 73 passed. Acceptance under `--dev` (watcher, NoC sanitizer): 41 passed.

**After-breakdown** (2304 wb fp32, `perf_experiments/breakdown/zones_final_*.txt`):
| stage | p50 timing |
|---|---|
| `w_affine` | done at 14.4 µs, off the critical path |
| x load | ends 23.6 µs; slowest core 40.6 µs |
| stats | done at 31.1 µs |
| `prep_affine` | 3.3 µs |
| combine | 2.2 µs |
| normalize | 39.7→61.7 µs (22 µs) |
| store end | 71.6 µs |
| max core span | 85 µs |

### Leads for round 2 (from this round's data)
- normalize at 22 µs with affine;
- the slowest core's x load (NCRISC max/p50 = 1.72; DRAM serves grid rows roughly in order, so work redistribution is the lever, not NoC issue);
- `dev_sq` at 7.6 µs;
- re-port I6 onto HEAD (noaff overlap);
- re-test I4's raw `reuse_blk` now that I8 removed the w/b DRAM burst it collided with.

### Pre-existing issues found (outside the golden suite; upstream, not perf)
- **16-bit DEST at offset 1000:** the output is already `inf` before this round (I3).
- **16-bit DEST, no_affine, input randn·0.5 + 100 at S=3:** rel-RMS is 0.1355 against a 0.13 limit, identical before and after (I6).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::reduce` | capability | `WaitUpfrontNoPop` / `NoWaitNoPop` always index from the CB front; there is no tile-base offset. So it cannot reduce tile row r of a block that is only partly landed and not popped, which a per-row stats schedule overlapping the load needs. | 94300 (all-rows schedule; 2048×3072 wb fp32, same card) | 89800 (per-row schedule) | `kernels/layer_norm_adaln_compute.cpp:214` |
| `eltwise_chain` (SFPU binary/unary elements) | capability | (a) No element folds over a compile-time-variable count S of CB tiles with accumulators live in DEST across peer copies. (b) No face-offset operand or result, so faces 1/3 can't carry a second value per row. (c) Binary SFPU `*_tile` APIs only run all four faces (`VectorMode::RC`); no faces-0/2 mode. (d) The chain rejects two `PackTile`s into one CB at out-of-order indices. The pre-tournament code was already a raw library-op sequence, for the same reasons (a) and (d). | 39667 (library-op SFPU sequence; 256×3072 wb fp32) | 28727 (fused SFPI routines, packed payload) | `kernels/layer_norm_adaln_compute.cpp:335` (publish), `:396` (combine), `kernels/layer_norm_adaln_stats_sfpu.hpp` |
| `ttnn.Mcast` / `McastArgs` (dataflow mcast helper) | capability | Receivers must split into EQUAL consecutive groups in one core order (RowMajor). Per-grid-column channels on a partially filled grid have unequal column heights, so they can't be expressed. There is also no column-major / strided group order that would map "every core of grid column x" to one group. | n/a, not expressible. Pre-change per-core DRAM fetch: 96600 (2304×3072 wb fp32, same card) | 84200 | `kernels/layer_norm_adaln_writer.cpp:49` (`affine_column_mcast`) |

The kernel-head justification comments are at each site: the compute kernel header, `layer_norm_adaln_stats_sfpu.hpp`, the `c_mean` block comment, and the `affine_column_mcast` comment.

## Perf 2
- **Date**: 2026-11-16. Perf tournament round 2 (coordinator + 4 `perf-part-optimizer` ideas). Device: Blackhole, 12×10 grid, 1350 MHz.
  `SUPPORTED` unchanged. Precision contract unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus shape**: `LOOSE_CASES` is still empty, so the focus stays (1,1,2304,3072) dynamic_wb fp32 DEST. Only cells > 32 tiles were measured. ~2–3% is noise.
- **Tooling note**: the precompile plugin needs `ttnn.graph.up_front_*`, which this build lacks, so every run used `--no-precompile`. Parallel subagents also need their own `TT_METAL_PROFILER_DIR`: the shared zone-source log grows until 16-bit zone hashes collide.

### Measured breakdown (before; HEAD after Perf 1)
- **Zones, 2304 wb fp32 = 85–86 µs** (`perf_experiments/round2_breakdown/zones_head_*.txt`). Times are p50 per core.
  | stage | p50 timing |
  |---|---|
  | x load | 0→24.4 µs (slowest core 40.8, NCRISC max/p50 1.67) |
  | stats chain | done 30.4 µs |
  | publish | 31.7 µs |
  | `prep_affine` | 3.3 µs |
  | exchange | ends 35.9 µs |
  | combine | 2.2 µs |
  | `bcast_stats` | 1.2 µs |
  | normalize | 39.9→61.8 µs (22 µs for 64 tiles) |
  | writer | `w_store_wait` 22.9 µs; first write issued at ~51 µs, after a full 32-tile row was normalized |
  | store end | 72.4 µs p50, BRISC max 85.3 |
- **Cumulative ablation, µs** (`round2_breakdown/ablations.jsonl`, gates in `ablation_gates.patch`).
  | variant | 2304 wb fp32 | 2304 noaff fp32 | 2048 noaff fp32 | 256x3072 wb fp32 |
  |---|---|---|---|---|
  | full | 85.0 | 79.6 | 72.3 | 26.0 |
  | no-write | 74.0 | 63.4 | 60.1 | 22.0 |
  | no-read | 77.9 | 66.5 | 58.1 | 25.6 |
  | compute + sync (no DM) | 45.7 | 34.2 | 33.3 | 20.1 |
  | DM + sync (no compute) | 81.2 | 76.6 | 68.3 | 17.4 |
  | read-only | 40.0 | 38.6 | 35.5 | 12.2 |
  | write-only | 64.1 | 50.9 | 46.2 | 17.3 |
  | sync-only | 19.8 | 10.5 | 10.1 | 12.0 |
- **Ranked bottleneck.** The op is DM-bound on the big shapes: removing all compute saves only 3.8 µs of 85.
  1. DRAM writes: ~40 µs at ~350 GB/s, serialized after the read and starting only after a full normalized row.
  2. DRAM reads: ~28 µs at ~500 GB/s, with a slowest-core tail.
  3. Compute: mostly hidden on the focus. 256x3072 is compute-bound (compute+sync 20.1 vs DM+sync 17.4).
  The DRAM floor is ~60 µs for 28.3 MB (round-1 estimate; the `noc_estimate` binary is still not built), so DM has headroom.

### Portfolio and verdicts
| # | idea | verdict | before → after (µs) | domain |
|---|---|---|---|---|
| J1 | `out_stream`: push `cb_output_tiles` per DEST window; writer drains in chunks so writes start when normalize starts | **WIN** | focus 85.0 → 81.3 (−4.4%, blk_c16). 2048 wb 16-bit −4.3 to −6%. Small shapes flat. Chunk 4/8 flat on focus: early writes compete with the slowest core's reads | everywhere, bit-exact |
| J2 | `dram_bw`: bank-staggered tile visit order (reader + writer). Also tried: posted writes, one-packet writes, lazy pop, writer on NoC0, rebalancing | **WIN** (stagger only) | focus 85.1 → 81.1 (−4.7%); 8192x256 −10%. Posted writes noise; one-packet flat; lazy pop worse; writer on NoC0 **hangs** (dedicated-NoC mode: the reader owns NoC0); moving the 1-row groups to the top rows +1% | everywhere, bit-exact. Slow cores are positional: read tail on the top grid rows (farthest NoC0 hops), write tail on the far-right columns (NoC1). The aggregate DRAM wall is ~370 GB/s read / ~300 GB/s write |
| J3 | `row_pipeline_v2`: 1-row blocks, input CB depth 2 (re-port of round-1 I6 onto HEAD) | **NULL** | focus 84.6 → 83.8 (−0.9%). Regressions: 8192x256 +4.8 to +10.7%, 2048 noaff fp32 +4.4%, 23040x3072 wb fp32 +10.5% | not graduated. Each core's post-read chain is serial compute, and extra blocks add rendezvous and init fixed cost. A lookahead adds the same cost without shortening the chain |
| J4 | `normalize_math`: srcB hardware broadcast in the dest-reuse ops (round-1 `reuse_blk` re-tested), plus per-window push and per-tile writer wait | **WIN** | normalize per-tile cost unchanged (~480 math cycles/tile wb fp32). Gain comes from deleting `bcast_stats` (critical path) and from writer overlap. 256x3072 wb fp32 26.5 → 22.0 (1.20×); helper-only per-window push 22.7. Focus 84.4 → 83.3 (noise). Round 1's 2048 wb 16-bit regression is gone (1.012×) | everywhere, bit-exact (40 correctness cells incl. offset-50) |

Rejected after costing (not built): SFPU affine epilogue (≥ the FPU passes, and it halves the fp32 DEST window); merging −p and −m (srcB is tf32, so large-offset fails); lowering fidelity for w·b (precision contract).

### Aggregation and what graduated
J1, J2 and J4 all touch the output push and the store loop, so I merged them by hand. J4's raw normalize loop subsumes J1's chain push policy. The one open choice was the writer drain. I measured it head-to-head on the merged code (`round2_breakdown/ab_screen.jsonl`, `ab_sweep.jsonl`; same process, 3 alternating reps):
- **Chunk 16, flush per chunk (J1):** focus −6.8%; 256x3072 wb fp32 only −5.3%; 256x1024 noaff +4.5% / +5.8%.
- **Per-tile cumulative wait, flush and pop per row (J4):** focus −5.0%; 256x3072 −17.8%; never slower than HEAD beyond noise.

The two differ by ≤ 3% on the big shapes, which is noise. I graduated the per-tile wait as the one policy. Graduated as the op's single unqualified path:
- **Normalize (J4):** one raw loop per DEST window. FPU `sub_tiles_bcast<COL>`, then dest-reuse binaries with srcB COL broadcast (m and rstd straight from `cb_row_stats`) and ROW broadcast (w and b straight from the row-0 tiles). Each window is pushed as soon as it is packed.
  **Deleted:** `prep_affine_block`, `bcast_stats_block`, `cb_shift_full`, `cb_rstd_full`, `cb_weight_full`, `cb_bias_full`, and the `unary_bcast` include.
- **Writer (J1/J4):** waits per tile and writes each tile as it lands. **Deleted:** the per-row wait.
- **Bank stagger (J2):** core L starts each row at column `(L % num_dram_banks) % bw` and wraps. The reader, normalize and writer all use that order, so the writer still streams in compute's order. **Deleted:** the unrotated loops.
- **Host:** `footprint_bytes` now drops the two deleted per-row acc CBs. Blocking for every INPUTS shape is unchanged; only L1-limited 23040-row shapes pick a taller block.

**Carve-outs: none.** No cell is incorrect, inexpressible or measurably slower.
- 256x1024 noaff 16-bit read +5.3% in one 5-rep A/B. A 14-rep re-check gave −1.2%: the cell's spread is 14.1–16.2 µs (`ab_256x1024.jsonl`).
- 23040-row guard shapes: 12 cells all within ±2.4% (noise), PCC ≥ 0.99987 vs torch (`ab_large.jsonl`). The one cell whose blocking changed (23040x2048 wb 16-bit) is correct but not bit-identical.

**Instrumentation.**
- Kept: `c_normalize`.
- Removed with their stages: `c_prep_affine`, `c_bcast_stats`.
- Writer: `w_store_wait` now measures the starvation before a row's first window lands; `w_store_issue` includes the later per-tile waits (a per-tile zone would blow the marker budget). Max markers per (core, RISC) = 30.

### Whole op, before → after
Same card, same process, HEAD copy (`round2_breakdown/head_pkg`) vs final code, alternating, median of 3 (`ab_sweep.jsonl`). Every cell is bit-identical to HEAD.

| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304x3072 (focus)** | **85.0 → 80.7 (−5.0%)** | 91.9 → 88.0 (−4.3%) | 80.4 → 76.8 (−4.5%) | 85.1 → 78.2 (−8.1%) |
| 2048x3072 | 78.2 → 74.5 (−4.7%) | 87.4 → 80.5 (−7.8%) | 70.7 → 69.0 (−2.4%) | 78.7 → 73.5 (−6.5%) |
| 256x3072 | 26.1 → 21.5 (−17.8%) | 29.0 → 25.1 (−13.2%) | 23.2 → 20.0 (−14.1%) | 25.9 → 23.2 (−10.6%) |
| 256x1024 | 14.6 → 14.5 (−0.5%) | 18.1 → 16.9 (−6.5%) | 12.8 → 12.7 (−1.3%) | 14.4 → 14.7 (+2.2%; 14-rep re-check −1.2%) |
| 128x512 | 11.7 → 11.2 (−3.9%) | 13.0 → 13.0 (−0.2%) | 10.6 → 9.9 (−6.8%) | 11.2 → 11.2 (−0.4%) |
| 32x3072 | 18.4 → 17.3 (−5.8%) | 21.2 → 20.4 (−3.8%) | 15.5 → 14.6 (−5.7%) | 18.4 → 17.7 (−3.6%) |
| 8192x256 (S=1) | 33.4 → 28.1 (−15.9%) | 36.9 → 31.7 (−14.0%) | 28.4 → 24.7 (−12.8%) | 32.3 → 27.8 (−13.7%) |

Final 5-rep confirmation (`ab_final.jsonl`):
| cell | before → after (µs) |
|---|---|
| **focus** | **84.5 → 79.9 (−5.5%)** |
| 2304 noaff fp32 | 80.1 → 76.7 (−4.2%) |
| 2048 wb fp32 | 78.2 → 75.8 (−3.0%) |
| 256x3072 wb fp32 | 26.1 → 21.2 (−18.6%) |
| 8192x256 wb fp32 | 32.9 → 28.3 (−14.1%) |

**Guard-set no-regression:** 0 of 28 INPUTS cells and 0 of 12 large cells are slower beyond noise.
**Golden:** 52 passed, 49 skipped. **Acceptance + precision baseline:** 73 passed. **Acceptance under `--dev`:** 41 passed.

**After-breakdown** (2304 wb fp32, `round2_breakdown/zones_after_2304x3072-wb-fp32.txt`):
| stage | p50 timing |
|---|---|
| normalize | 35.4→57.3 µs (starts 4.5 µs earlier: no `bcast_stats`) |
| first output write | ~38 µs (was ~51) |
| store end | 64.0 µs (was 72.4) |
| BRISC max | 81.2 µs |
| NCRISC max/p50 | 2.03 (load p50 22.8, slowest core 46 µs) |

### Leads for round 3
- **Positional read tail:** the top grid rows' x loads, farthest from DRAM on NoC0. It now sets the makespan through the group exchange.
- **Positional write tail:** the far-right columns on NoC1.
- **Normalize:** ~480 math cycles/tile with affine; only matters once DM shrinks.
- **The 72-rows-over-40-groups imbalance** (2-row vs 1-row groups).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::DestReuseBinary` (in `eltwise_chain`) | capability | Has no broadcast field: it always runs `{add,sub,mul}_reuse_dest_*` with `BroadcastType::NONE`. A per-row (Col) or per-column (Row) operand therefore has to be materialized as full tiles by a separate `unary_bcast` pass and CB first, and that pass sat on the critical path between combine and normalize. The LLK supports it directly (`llk_unpack_A<COL\|ROW, true, DEST_TO_SRCA>` + `llk_math_eltwise_binary<op, COL\|ROW, …, DEST_TO_SRCA>`). Needed: a `BroadcastDim` on `DestReuseBinary` | 22700 (helper chain with per-window push, 256x3072 wb fp32) | 22000 (−3%, at the noise edge: the bypass buys deleting `bcast_stats` + 2 full-tile CBs, not a faster per-tile op) | `kernels/layer_norm_adaln_compute.cpp:167` (`bcast_reuse_init` / `bcast_reuse_tile`), used in `normalize_block` at `:468` |

The kernel-head justification comment is at `kernels/layer_norm_adaln_compute.cpp:38`. The writer change and the bank stagger are plain dataflow, with no helper bypassed. Round 1's bypasses (per-row mean, SFPI combine, affine column mcast) are unchanged.

## Perf 1
- **Date**: 2026-11-16. This is the third tournament overall: it follows "Perf 1" and "Perf 2" above, but the pipeline numbers it "Perf 1" (round 1 of 3 of a new series). Coordinator plus 4 `perf-part-optimizer` ideas. Device: Blackhole, 12×10 grid, 1350 MHz.
  `SUPPORTED` unchanged. The precision contract was unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus shape**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md`:
  - primary: (1,1,2304,3072) dynamic_wb fp32 DEST;
  - then 2048×3072 and 256×3072 (both wb fp32).
  Only cells > 32 tiles were measured. ~2–3% is noise.
- **Tooling**: same as Perf 2. `--no-precompile`, a private `TT_METAL_PROFILER_DIR` per agent, and the `noc_estimate` binary is still not built, so the DRAM floor is the Step-4b estimate.

### Measured breakdown (before; HEAD c3487996f36)
- **Headline sweep** (`perf_experiments/r3_breakdown/baseline_sweep.jsonl`, one run each, µs):
  | shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
  |---|---|---|---|---|
  | 2304×3072 (focus) | 83.1 | 85.7 | 76.9 | 78.8 |
  | 2048×3072 | 76.4 | 81.8 | 67.5 | 72.7 |
  | 256×3072 | 21.3 | 25.2 | 19.7 | 23.5 |
- **Zones, focus 2304 wb fp32** (`r3_breakdown/zones_head_2304x3072-wb-fp32.txt`, p50 per core):
  | stage | p50 |
  |---|---|
  | x load | 0→22.5 µs; slowest core 47.1 (NCRISC max/p50 2.09) |
  | stats chain | done ~29 µs |
  | `w_affine` | 0→14.6 µs on every core |
  | `w_exchange_rounds` | 2.4 µs (max 6.4) |
  | `c_combine` | 2.2 µs |
  | `c_normalize` | 35.2→57.1 µs (22 µs for 64 tiles) |
  | store end | 64.5 µs p50; BRISC max 82.2 |
- **Zones, 256×3072 wb fp32**:
  | stage | p50 |
  |---|---|
  | read | ends 4.5 µs |
  | stats | ready 8.1 µs |
  | `w_affine` | ends 7.5 µs |
  | `w_exchange_rounds` | 3.45 µs (S=6) |
  | normalize | 6.35 µs |
  | span | 20.9 µs |
- **Cumulative ablation, µs** (`r3_breakdown/ablations.jsonl`; gates in `r3_breakdown/ablation_gates.patch`, reverted after the run):
  | variant | 2304 wb fp32 | 2304 noaff fp32 | 2048 wb fp32 | 256×3072 wb fp32 |
  |---|---|---|---|---|
  | full | 81.2 | 76.3 | 74.0 | 21.2 |
  | no-write | 72.5 | 63.7 | 68.1 | 20.9 |
  | no-read | 67.5 | 59.9 | 61.5 | 20.7 |
  | compute + sync | 46.1 | 35.6 | 45.0 | 19.7 |
  | DM + sync (no compute) | 78.3 | 75.3 | 70.4 | 16.8 |
  | read-only | 37.2 | 36.8 | 33.6 | 12.3 |
  | write-only | 63.3 | 49.7 | 57.3 | 16.8 |
  | sync-only | 21.1 | 8.8 | 21.2 | 11.9 |

  The wb sync-only figure (21.1 vs 8.8 µs noaff) exposed the w/b fetch-and-multicast stage at ~12 µs in isolation.
- **DM ceiling.** DRAM floor ≈ 28.3 MB / ~460 GB/s ≈ 61 µs. DM + sync is 78.3 µs: reads run at ~385 GB/s and writes at ~300–355 GB/s. So there is DM headroom, while compute is mostly hidden (removing it saves 2.9 µs).
- **Ranked bottlenecks:**
  1. DRAM writes (removing them saves 8.7 µs).
  2. The positional x-load read tail (removing reads saves 13.7 µs).
  3. The S serialized exchange rounds, which sit on the critical path of the sync-bound small-H shapes.
  4. The w/b fetch (14.6 µs, hidden on big shapes, near-critical on 256×3072).

### Portfolio and verdicts
| # | idea | verdict | before → after (µs) | domain |
|---|---|---|---|---|
| K1 | `affine_fetch`: diagnose and cut the 14.6 µs w/b column fetch+mcast | **NULL** (whole op) | Stage cut 14.6 → 6.6 µs (focus) and 7.8 → 4.1 µs (256×3072) by `linked` multicasts. Whole op flat on all 12 cells (focus 81.4 → 79.9, one run; 2048 wb +2.3%; all noise). A `bulk` 2-multicast variant was also tried: 8.2 µs stage, flat | Everywhere, bit-exact. Not graduated: the stage is off the critical path everywhere; the writer only needs w/b by the exchange. Diagnosis: 8.8 of the 14.6 µs is 64 unlinked multicasts that each re-reserve the column path (~137 ns each); 3.7 µs is the DRAM read queued behind the x flood |
| K2 | `dual_noc_writes`: the reader stores a share of the output on NoC0 after its loads (second output CB) | **NULL / REGRESSION** | Uniform share: +1.7% (12.5%) to +10.8% (25%) on focus. NoC0 writes delay the top grid row's x loads (47 → 62 µs), and those loads set the makespan. Top-row-only share: focus −0.8% (flat); 2304 noaff −2.9 to −3.9% | Not graduated: only the noise-band top-row variant avoids regressing, and it is a placement rule tied to where BH DRAM sits relative to the grid |
| K3 | `dual_noc_reads`: the writer (NoC1, idle until stats) loads a share of x | **REGRESSION** | `uni:0.5` +13% focus. It removes the read tail (NCRISC max 48 → 33.5 µs), but then every core normalizes and writes at once into the DRAM write wall (store end +6 µs). HEAD's slow-read gradient acts as a free write stagger. `uni:0.125`: −2.0% focus (noise), −6.2% 2048 wb fp32, **+4.1% 2304 noaff 16-bit** (measured regression) | Not graduated. Lead: a read split only pays together with something that absorbs the synchronized write burst |
| K4 | `exchange_oneshot`: one-rendezvous all-gather (every member unicasts its payload to the S−1 peers at once, then sets per-sender flags) instead of S handshaked multicast rounds | **WIN** (hybrid) | `w_exchange_rounds`: focus 2.6 → 1.0 µs; 256×3072 3.45 → 2.5 µs. Small-H S>1 cells −3% to −18%; focus flat. Pure unicast regresses 3–16% when a core's outgoing copies are ≥ 56 KiB (4096×4096 −5%, 3072×6144 −5.5%, forced-S=6 2304 −11 to −16%). Concurrent-multicast and token-multicast variants were slower than the helper (256×3072: 28.6 / 24.3 vs 21.3) | Everywhere it is correct (S = 1..12, multi-block, ragged, both DEST modes, `--dev` clean), with one measured carve-out (below). Cost model unchanged: a forced-S re-sweep shows `C_EXCHANGE=2.0` still picks the fastest S |

### What graduated
**K4 hybrid exchange.** It is the op's exchange path for every S > 1 group whose outgoing data, `(S−1) · spr · bh · acc_tile_bytes`, is ≤ 48 KiB. That covers every INPUTS shape.
- **Writer:** new `exchange_oneshot`.
  - Copies its own slot with a local NoC read.
  - Sends S−1 unicast payload writes.
  - Flushes, then sets its per-sender flag (`block_idx + 1`, monotone, never reset) on every peer.
  - Waits until every peer's flag ≥ `block_idx + 1`.
- **Multi-block cores:** `cb_gathered_stats` is double-buffered (`gather_depth`), because a peer can be at most one block ahead.
- **Host:**
  - `exchange_oneshot()`, `gather_depth()` and `ONESHOT_MAX_BYTES` in `layer_norm_adaln.py`.
  - `footprint_bytes` counts the depth.
  - The descriptor adds S flag semaphores, writer named CT args `xchg_oneshot` / `xchg_flag_sem_base`, and the peer NoC coordinates as RT args.
  - The `ttnn.Mcast` channel is attached only on the rotating path.
- **Instrumentation:** the `w_exchange*` zones cover both schedules.
- **L1 ledger:** `l1_ledger.md` was updated for the gather depth.

**Carve-out (measured, earned):** `if (outgoing > 48 KiB) { rotating ttnn.Mcast } else { one-shot }`. Above the threshold, S−1 unicast copies saturate the NoC row and lose to one multicast by 3–16% (`perf_experiments/exchange_oneshot/ab_bigpayload.jsonl`, `ab_forceS.jsonl`). The rotating code is therefore kept, not deleted.

**Not graduated:** K1, K2, K3 (see above).

### Whole op, before → after
Same process, git-HEAD copy vs final code, alternating, median of 3 (`r3_breakdown/ab_guard.jsonl`). Every cell is bit-identical to HEAD.

| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304×3072 (focus)** | **79.9 → 80.2 (+0.4%, noise)** | 85.4 → 84.2 (−1.4%) | 76.8 → 77.6 (+1.1%) | 78.6 → 80.2 (+2.0%) |
| 2048×3072 | 75.8 → 74.2 (−2.2%) | 81.1 → 79.0 (−2.6%) | 67.5 → 68.6 (+1.5%) | 73.1 → 72.7 (−0.5%) |
| 256×3072 | 21.2 → 20.5 (−3.3%) | 25.0 → 23.9 (−4.5%) | 19.5 → 18.7 (−3.7%) | 23.0 → 22.3 (−3.0%) |
| 32×3072 | 17.2 → 15.8 (−8.3%) | 20.2 → 18.8 (−7.2%) | 14.5 → 13.0 (−10.5%) | 17.6 → 16.2 (−8.1%) |
| 256×1024 | 14.7 → 12.1 (−17.6%) | 16.3 → 14.3 (−12.0%) | 12.8 → 10.5 (−18.2%) | 15.7 → 12.9 (−18.2%) |
| 128×512 | 11.1 → 9.6 (−13.3%) | 12.2 → 11.2 (−8.4%) | 9.4 → 8.1 (−14.3%) | 11.4 → 9.8 (−13.5%) |
| 8192×256 (S=1, unchanged path) | 28.2 → 28.2 (−0.2%) | 31.5 → 31.7 (+0.5%) | 25.0 → 25.4 (+1.5%) | 27.9 → 27.6 (−1.2%) |

- **6-rep recheck of the 2304 cells** (`r3_breakdown/ab_recheck.jsonl`): noaff 16-bit +1.2%, noaff fp32 +1.4%, wb fp32 −0.8%. All noise.
- **Large multi-block cells** (`ab_large.jsonl`; 23040×3072 wb fp32 / wb 16-bit / noaff fp32, 23008×3072 wb fp32): within ±0.7%, bit-exact.
- **Guard-set no-regression:** 0 of 28 INPUTS cells and 0 of 4 large cells are slower beyond noise.
- **Golden:** 52 passed, 49 skipped. **Acceptance + precision baseline:** 73 passed. **Acceptance under `--dev`** (watcher, NoC sanitizer): 41 passed.
- **The focus shape is unchanged**: the exchange there was 2.4 µs of a makespan that is set by DRAM traffic. On 256×3072 wb fp32, one of the three roofline shapes, it is −3.3%.

**After-breakdown** (`r3_breakdown/zones_after_*.txt`):
- Focus: `w_exchange_rounds` 2.4 → 1.0 µs p50, `c_wait_gathered` 3.0 → 1.7 µs, BRISC max 82.2 → 80.0 µs.
- 256×3072 wb fp32: span 20.9 → 20.6 µs; `w_exchange_rounds` 3.45 → 2.5 µs (the remainder is skew from waiting on the slowest peer's stats).

### Leads for the next round
- **The focus is DRAM-bound**: about 78 µs of DM + sync against a ~61 µs floor. K2 and K3 show the two DM walls are coupled.
  - Balancing reads (K3) removes the read tail but synchronizes the write burst.
  - Moving writes to NoC0 (K2) slows the top row's reads, which set the makespan.
  - A combined design (K3's read split plus a write schedule that absorbs or staggers the burst, e.g. the reader's idle NoC0 after its now-earlier load end) is untested.
- **K1 `linked`** (1 argument, bit-exact) frees 8 µs of hidden writer time. Revisit it if the 256×3072 stats chain drops below ~7.8 µs.
- **256×3072** is now sync- and compute-bound: compute + sync 19.7 µs; normalize 6.35 µs for 16 tiles is ~395 ns/tile.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ttnn.Mcast` / `McastArgs` (`McastRotatingSenderConfig`, handshake) | capability | No all-to-all schedule. It allows only one sender per round, with a per-round receiver-ready handshake, so an S-member all-gather pays S serialized rendezvous latencies (~575 ns each). It cannot express "every member sends in the same round, with per-sender monotone completion flags and no per-round ready step". Issuing its multicasts concurrently (one channel per sender) was slower than S−1 unicasts (256×3072 wb fp32 28.6 µs). Needed: an all-gather mode with per-sender flags, for payloads small enough to unicast | 21270 (256×3072 wb fp32); 17350 (32×3072 wb fp32); 11020 (128×512 wb fp32) | 20640; 15820; 9730 | `kernels/layer_norm_adaln_writer.cpp:112` (`exchange_oneshot`) |

The kernel-head justification comment is at the `exchange_oneshot` definition (`kernels/layer_norm_adaln_writer.cpp:112`). The large-payload path still uses the helper. Earlier rounds' bypasses are unchanged.

## Perf 2
- **Date**: 2026-11-16. This is the fourth tournament overall and the second round (of 3) of the current series. The pipeline numbers it "Perf 2"; the earlier "Perf 2" heading above belongs to the first series. Coordinator plus 5 `perf-part-optimizer` ideas. Device: Blackhole, 12×10 grid, 1350 MHz.
  `SUPPORTED` unchanged. The precision contract was unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus shape**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md`: (1,1,2304,3072) dynamic_wb fp32 DEST, then 2048×3072 and 256×3072 wb fp32. Only cells > 32 tiles were measured. ~2–3% is noise.
- **Tooling**: `--no-precompile`, and a private `TT_METAL_PROFILER_DIR` per subagent. Coordinator data is in `perf_experiments/r4_breakdown/`: `test_ab.py` (HEAD copy `head_pkg/` vs working tree), `ab_summary.py`, `ablation_gates.patch` (adds `ABL_NO_NORM`), and the attribution tests.

### Measured breakdown (before; HEAD 0f8c5d7ac42)
- **Headline** (`r4_breakdown/baseline_sweep.jsonl`, µs):
  | shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
  |---|---|---|---|---|
  | 2304×3072 (focus) | 80.7 | 84.4 | 77.7 | 79.5 |
  | 2048×3072 | 74.4 | 79.0 | 68.0 | 71.7 |
  | 256×3072 | 20.7 | 23.9 | 19.1 | 22.1 |
- **Zones, focus** (`zones_head_2304x3072-wb-fp32.txt`, p50 per core):
  | stage | p50 |
  |---|---|
  | x load | ends 23.6 µs; slowest core 46.1 (NCRISC max/p50 1.95) |
  | `w_affine` | 14.4 µs |
  | `c_dev_sq` | 7.5 µs |
  | publish | ends 30.1 µs |
  | `c_combine` | 2.2 µs |
  | `c_normalize` | 22.0 µs for 64 tiles; uniform on all 120 cores, so compute-bound |
  | store end | 65.3 µs; BRISC max 78.7 |
- **Zones, 256×3072 wb fp32**:
  | stage | value |
  |---|---|
  | load | ends 4.6 µs |
  | stats | published 8.35 µs |
  | `w_affine` | 7.4 µs |
  | exchange | 2.5 µs |
  | combine | 1.4 µs |
  | normalize | 6.35 µs (16 tiles) |
  | span | 20.5 µs |
- **Cumulative ablation, µs** (`ablations.jsonl`). `ABL_NO_NORM` stubs only normalize.
  | variant | 2304 wb fp32 | 2304 noaff fp32 | 2048 wb fp32 | 256×3072 wb fp32 |
  |---|---|---|---|---|
  | full | 80.2 | 77.4 | 74.4 | 20.6 |
  | no-normalize | 78.7 | 77.5 | 71.4 | 18.1 |
  | no-write | 70.6 | 58.8 | 65.2 | 20.3 |
  | no-normalize + no-write | 50.1 | 48.2 | 44.7 | 14.7 |
  | no-read | 65.3 | 58.9 | 60.6 | 20.2 |
  | compute + sync | 41.4 | 29.8 | 41.4 | 19.4 |
  | DM + sync | 77.8 | 75.8 | 69.3 | 16.2 |
  | read-only | 37.0 | 36.3 | 33.6 | 11.8 |
  | write-only | 62.4 | 50.0 | 56.7 | 16.5 |
  | sync-only | 17.6 | 5.6 | 17.7 | 11.7 |
- **Ranked bottlenecks:**
  1. On the focus, the DRAM write wall (~300 GB/s) and normalize compute are **balanced**. Removing either alone moves the focus 1.5–9.6 µs; removing both moves it 30 µs.
  2. The positional x-load read tail: 46 µs on the slowest core vs 23.6 p50.
  3. On 256×3072 (compute + sync bound): normalize 6.35 µs, the stats chain ~3.7 µs, and the exchange skew, all on the critical path.
- **DM ceiling**, measured this round by the `dram_ceiling` bench (same traffic, pure DM):
  | traffic | ceiling |
  |---|---|
  | read | 399 GB/s |
  | write | 308 GB/s |
  | reads and writes concurrent | ~390 GB/s combined |
  | **COPY floor at the focus** | **72.1 µs** |

  HEAD was at 1.12× this floor. The `roofline_targets.md` 460 GB/s figure (61.4 µs) is ~15% below anything this card sustained.

### Portfolio and verdicts
| # | idea | verdict | before → after (µs) | domain |
|---|---|---|---|---|
| L1 | `bank_coalesce`: bank-major L1 slots, one multi-tile NoC read/write per (bank, row); reader + writer twin | **REGRESSION** | focus `rw` 80.3 → 85.0 (+6.0%), `w` +3.9%, `r` and `r2` flat; 2304 wb 16-bit `rw` +6.5% | not graduated. Issue time fell (6.2 → 0.5 µs), but fewer, larger packets (one bank, one route each) spread worse over banks and links than per-tile round-robin |
| L2 | `dram_ceiling`: real BH DRAM ceilings, then untried levers (alternate DRAM endpoints, bank-phase scheduling, write trids, bank-affine slices) | **NULL** | ceilings as above; every lever flat. Alternate endpoints are **inexpressible** (DRISC firmware keeps only the preferred endpoint per NoC in NOC2AXI mode) | nothing to graduate; established the 72.1 µs floor |
| L3 | `noc_swap`: the 2 easternmost grid columns run reader on NoC1 and writer on NoC0 | **WIN** | focus −1.6 to −4.2% (two cards); 2304 noaff fp32 −4.1 to −7.1% | everywhere except the rotating-mcast path (inexpressible: one NoC per `ttnn.Mcast` channel) and a measured-regression carve-out (below). A third swapped column regressed |
| L4 | `normalize_tile_cost`: diagnosis (math-bound; ~30 ns/init, dest-reuse ~30 cycles/tile/op), then PM2 = D ← c′ = −(p+m)·rstd from SFPU LREGs, D += x·rstd (one FPU MAC), then ·w + b | **WIN (fp32 DEST)** | normalize math 5683 → 4791 ns at 256×3072 (−16%); whole op 2048 wb fp32 −5.8%, 8192×256 −2.8%, focus flat alone (write wall). PM1 (c′ via unpack-to-dest `copy_tile`) was slower: 6390 ns | fp32 DEST everywhere; **16-bit DEST incorrect** (c′ ≈ −mean·rstd does not survive bf16 DEST), so the old path is kept there. Precision **better** than HEAD: no tf32 truncation of the running value |
| L5 | `stats_chain`: chunked reader push (one read trid per 4-tile chunk), p = bf16 mean of the first chunk, one merged Σ(x−p) / Σ(x−p)² pass (SFPU square to a second DEST slot, two L1-acc packs), linked affine multicast | **WIN** | 256×3072 wb fp32 20.6 → 19.0 (−7.7%), wb 16-bit −17.7%; 2304 wb 16-bit −5.5%; focus flat alone. The x0 pivot variant (p = x[:,0]) **failed precision** on outlier first columns and was not graduated | everywhere; both DEST modes |

### Aggregation and what graduated
L3, L4 and L5 touch disjoint stages (host NoC assignment / normalize / stats + reader), so all three graduated together as the op's single path. I merged them by hand:
- L5's experiment knobs were hardcoded: `STATS_CHUNK = 4`, mean pivot, linked multicast.
- The x0 pivot mode was deleted.
- L4's normalize then applied cleanly on L5's `cb_pivot`.
- L3's host patch was rebased onto L5's reader CT args: `noc_swap/graduate_noc_swap_rebased_on_stats_chain.patch`.

**Deleted:**
- `cb_local_mean`, `cb_dev_sum_acc`, `cb_dev_sum`, and the separate `c_mean` / `c_pivot`-copy / `c_dev_sq` / `c_dev_sum` chains;
- the unlinked multicast;
- the single reader/writer kernel descriptor;
- in fp32 DEST, the sub_bcast + dest-reuse sub + mul (16-bit DEST still uses that sequence, as recorded below).

`footprint_bytes` is now exact (see `l1_ledger.md`). Every INPUTS shape's blocking is unchanged; some L1-limited 23040-row shapes pick a taller block.

**Carve-outs:**
1. **PM2 is fp32-DEST only (incorrect elsewhere).** `if constexpr (DST_ACCUM_MODE)`: c′ loses the offset in a 16-bit DEST, so 16-bit keeps the sub/sub/mul sequence.
2. **noc_swap is off on the rotating `ttnn.Mcast` path (inexpressible).** That path covers outgoing stats > 48 KiB, which no INPUTS shape has.
3. **noc_swap is off for affine + 16-bit DEST + `core_h_tiles ≥ 4` (measured regression).** Swap on vs off, same combined build, 3 reps (`r4_breakdown/attr_nocswap.jsonl`):
   | cell | rows/core | swap on vs off |
   |---|---|---|
   | 15360×2048 wb 16-bit | 4 | +9.0% |
   | 23040×1024 wb 16-bit | 6 | +8.3% |
   | 23040×1536 wb 16-bit | 6 | +8.9% |
   | 23040×2048 wb 16-bit | 6 | +8.0% |
   | 23040×2464 wb 16-bit | 4 | +4.8% |
   | 11520×2048 wb 16-bit | 6 | +3.8% |

   The boundary was measured on both sides:
   - wb 16-bit with ≤ 3 rows/core wins: 2304×3072 −5.7%, 4096×2048 −4.5%.
   - fp32 DEST and no-affine on the same tall shapes win: −2.4 to −5.7%.
   - 23040×2464 wb fp32 (+2.2%, +2.5%) is in the noise band, so it keeps the swap.

   I found the regression on the large guard set (+3.8% vs HEAD at 23040×2048 wb 16-bit) and attributed it per package (`attr_pkgs.jsonl`): noc_swap alone +6.7%, stats_chain alone −3.4%, normalize alone flat. The mechanism is not root-caused.

**Attribution of noc_swap inside the combined build** (swap on vs off, 5 reps): focus −2.4%, 2304 noaff fp32 −4.3%, 2304 wb 16-bit −5.7%, 2048 wb fp32 −4.0%, 256×3072 wb fp32 −4.5%, 8192×256 flat, 256×1024 noaff 16-bit −2.7%.

**Instrumentation:**
- Removed with their stages: `c_mean` and `c_dev_sum`.
- `c_pivot` now holds the first-chunk wait, split out as the nested `c_wait_input` zone. `c_dev_sq` contains the per-chunk waits; a zone per chunk would exceed the marker budget.
- `c_normalize` covers both DEST paths. Both swapped-NoC kernel groups get the same zones.
- Max markers per (core, RISC) = 30, the same as HEAD. Zones cover the full span on every RISC.

### Whole op, before → after
Same process, HEAD copy vs final code, alternating, median of 3 (`r4_breakdown/ab_guard.jsonl`). The carve-out cannot touch any of these cells (all have ≤ 3 rows/core).

| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304×3072 (focus)** | **80.3 → 76.1 (−5.3%)** | 83.7 → 76.0 (−9.1%) | 77.9 → 72.3 (−7.1%) | 79.0 → 72.4 (−8.4%) |
| 2048×3072 | 73.4 → 67.5 (−7.9%) | 78.7 → 68.8 (−12.6%) | 68.4 → 64.3 (−6.0%) | 72.1 → 63.5 (−11.9%) |
| 256×3072 | 20.5 → 17.3 (−15.7%) | 24.2 → 18.6 (−22.9%) | 18.8 → 15.5 (−17.6%) | 22.1 → 17.0 (−23.2%) |
| 32×3072 | 15.7 → 14.4 (−8.4%) | 18.8 → 15.6 (−17.0%) | 13.0 → 11.7 (−9.6%) | 16.2 → 13.1 (−19.1%) |
| 256×1024 | 12.0 → 11.0 (−8.8%) | 14.2 → 11.8 (−17.5%) | 10.5 → 9.7 (−7.1%) | 12.9 → 10.5 (−19.0%) |
| 128×512 | 9.6 → 9.3 (−3.7%) | 11.2 → 9.5 (−15.3%) | 8.1 → 7.7 (−4.9%) | 9.8 → 8.2 (−16.4%) |
| 8192×256 (S=1) | 28.1 → 26.9 (−4.4%) | 31.7 → 27.5 (−13.3%) | 25.4 → 24.5 (−3.5%) | 27.6 → 24.4 (−11.5%) |

**Final code, 5 reps** (`ab_final.jsonl`):
| cell | before → after (µs) |
|---|---|
| **focus** | **81.3 → 76.4 (−6.0%)** |
| 2304 wb 16-bit | −10.4% |
| 2304 noaff fp32 | −7.3% |
| 2304 noaff 16-bit | −9.9% |
| 2048 wb fp32 | −10.1% |
| 256×3072 wb fp32 | 20.6 → 17.3 (−16.1%) |
| 8192×256 wb 16-bit | −13.3% |

**Against the 72.1 µs COPY floor**, the focus is now at 1.06×.

**Large guard cells, final code, 3 reps** (`ab_large.jsonl`, tag `large_final`):
- 23040×3072: wb fp32 −4.0%, wb 16-bit −6.5%, noaff 16-bit −7.8%.
- 23008×3072 noaff 16-bit −7.5%.
- 23040×2048 wb 16-bit −3.5% (was +3.8% before the carve-out), wb fp32 −2.3%.
- 23040×2464: wb 16-bit −4.2%, wb fp32 −0.2%.
- 15360×2048 wb 16-bit −2.6%.
- 23040×1024 wb 16-bit −3.9%.
- 11520×2048 wb 16-bit −2.9%.
- 4096×2048 wb 16-bit −7.6%.
- 4096×4096 (rotating) −0.1%.
- 3072×6144 wb fp32 −4.1%.

**Guard-set no-regression:** 0 of 28 INPUTS cells and 0 of 15 large cells are slower beyond noise.

**Precision.** No cell is bit-exact with HEAD.
- **fp32 DEST improves everywhere** (PM2 removes the tf32 truncations): rel-RMS wb 0.0058–0.0077 → 0.0042–0.0044, noaff → 0.0017.
- **16-bit DEST at offset 0 is slightly worse:** rel-RMS +0.0005–0.0012 (e.g. 2304 noaff 0.0081 → 0.0090), with PCC moving by at most ~2e-5. This comes from L5's 4-tile pivot. It stays far inside the 0.13 limit. It is the cost of the fix below, and I'm recording it as such.
- **16-bit DEST at large offsets: an upstream bug in HEAD is fixed** (`r4_breakdown/precision_offsets.jsonl`; it was listed under round 1's pre-existing issues):
  | cell | HEAD PCC | HEAD rel-RMS | now PCC | now rel-RMS |
  |---|---|---|---|---|
  | 256×3072 noaff, offset 1000 | 0.61 | 520 | 0.99998 | 0.0071 |
  | 2304×3072 wb, offset 1000 | 0.70 | 609 | 1.0 | 0.0067 |
  | 32×3072, offset 1000 | 0.53 | — | 0.999995 | — |
  | 8192×256 wb, offset 1000 | 0.46 | — | 0.99999 | — |
  | 128×3040 noaff, offset 50 | 0.53 | — | 0.99998 | — |

  The cause was the bf16-accumulated full-row pivot.

**Test suites:**
- Golden: 52 passed, 49 skipped.
- Acceptance + precision baseline: 73 passed.
- Acceptance under `--dev` (watcher, NoC sanitizer): 41 passed.
- `--dev` on the rotating path, partial grid, the carve-out regime, focus and S=1: 8 passed (`dev_correctness.jsonl`).

**After-breakdown, focus** (`zones_after_2304x3072-wb-fp32.txt`):
| stage | before | after |
|---|---|---|
| normalize | 22.0 µs | 17.6 µs |
| `w_affine` | 14.4 µs | 6.9 µs |
| publish ends | 30.1 µs | 27.9 µs |
| store end (p50) | 65.3 µs | 62.6 µs |
| BRISC max | 78.7 µs | 76.5 µs |
| NCRISC max | 46.1 µs | 43.1 µs |

**After-breakdown, 256×3072 wb fp32:**
| stage | before | after |
|---|---|---|
| span | 20.5 µs | 17.3 µs |
| publish ends | 8.35 µs | 6.5 µs |
| normalize | 6.35 µs | 5.45 µs |

### Leads for round 3
- **The focus is at 1.06× the measured 72.1 µs COPY floor.** What's left is read/write overlap, not bandwidth (L2).
- **The noc_swap regression mechanism** (affine + 16-bit DEST, tall per-core blocks) is not understood. Root-causing it could shrink carve-out 3.
- **256×3072** is now ~17 µs against an ~8.9 µs DM floor. Remaining time: exchange skew (2 µs), normalize (5.45 µs), the 16-bit normalize (which still uses the dest-reuse path), and `w_store_wait`.
- **The PM2 affine tail** (mul w 95 ns + add b 38 ns per tile) is now the largest part of normalize.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `eltwise_chain` (`BinaryFpu` / `DestReuseBinary`) | capability | (a) No element preloads DEST from SFPU registers (LREG → SFPSTORE into each window's slots). (b) `BinaryFpu` has no accumulate-into-DEST flag together with a broadcast operand, so `D += x · bcast(rstd)` cannot be one chain element. The library way to preload DEST (unpack-to-dest `copy_tile` of a c′ tile) costs +96 ns/tile, more than the two ops it removes. Needed: a DEST-preload-from-LREG element and an accumulate flag on broadcast `BinaryFpu` | 6390 (normalize math, PM1 = library preload; 256×3072 wb fp32, 16 tiles) | 4791 (PM2) | `kernels/layer_norm_adaln_compute.cpp:449` (fp32 normalize); `layer_norm_adaln_stats_sfpu.hpp` (`nz_cprime`, `nz_load_cprime`, `nz_store_cprime_tile`) |
| `eltwise_chain` (`BinaryFpu` Sub → `Square` → L1-acc `PackTile`) | capability | (a) No SFPU element writes its result to a second DEST slot while keeping its input; `Square` is in place, so keeping x − p for Σ(x−p) needs a second FPU sub + unpack per tile. (b) The chain rejects two `PackTile`s into one CB (acc[0], acc[1]). (c) Waits are per call, and per-chunk calls (needed to stream behind the chunked reader) would re-seed the L1 accumulator. Needed: a distinct-output-slot SFPU element, multi-target `PackTile` into one CB, and an "accumulate only" L1-acc seed policy across calls | 20630 (whole op, HEAD helper chains, 256×3072 wb fp32); 20560 (4-tile pivot without streaming) | 19049 (merged raw pass + streaming + linked multicast; the multicast alone is −1.0%) | `kernels/layer_norm_adaln_compute.cpp:284` (`c_dev_sq`), `cf_sq_to` in `layer_norm_adaln_stats_sfpu.hpp` |
| `compute_kernel_lib::reduce` | capability | Same gap as round 1's row, now hit by the first-chunk pivot. `WaitUpfrontNoPop` / `NoWaitNoPop` index from the CB front with no tile-base offset or rotated column order, so the helper cannot reduce the landed first chunk of row r of an un-popped block | n/a (same instruction sequence as the helper's FPU REDUCE_ROW path; the bypass is for addressing, not speed) | n/a | `kernels/layer_norm_adaln_compute.cpp:252` (`c_pivot`) |

The kernel-head justification comments are in the compute kernel header (stats chain at the top, normalize fp32 at `:59`). The writer's linked flag extends round 1's existing `affine_column_mcast` bypass (`ttnn.Mcast` cannot express per-column channels on a partial grid). The reader's read-trid chunking and the NoC swap are plain dataflow / host code, so no helper is bypassed. Earlier rounds' bypasses are unchanged.

## Perf 3
- **Date**: 2026-11-16. Fifth tournament overall; round 3 of 3 of the current series (the pipeline numbers it "Perf 3"). Coordinator + 5 portfolio `perf-part-optimizer`s, 1 fusion agent, 1 follow-up idea. Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. Precision contract unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md` (2304×3072, 2048×3072, 256×3072, all dynamic_wb fp32). Only cells > 32 tiles were measured; ~2–3% is noise.
- **Tooling** (`perf_experiments/r5_breakdown/`): `test_run.py` (working tree, `RUN_CELLS`, optional zone CSV capture), `test_ab.py` + `head_pkg/` (HEAD 1414e28e955 copy vs working tree, alternating), `test_forceS_ab.py` (forced slice counts A/B in one process), `ablation_gates.patch` (`LN_ADALN_ABLATE=NO_READ,NO_WRITE,NO_COMPUTE,NO_NORM`; applied for the ablation runs, then reverted), `zones.py`. The precompile pytest plugin fails in this environment (`ttnn.graph.up_front_clear` missing), so every run used `--no-precompile`.

### Measured breakdown (before; HEAD 1414e28e955)
- **Headline** (`baseline_sweep.jsonl`, median of 3, µs):
  | shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
  |---|---|---|---|---|
  | 2304×3072 | 76.3 | 75.4 | 72.9 | 72.1 |
  | 2048×3072 | 68.1 | 69.2 | 64.7 | 63.5 |
  | 256×3072 | 17.4 | 18.6 | 15.7 | 17.0 |
  | 8192×256 | 27.0 | 27.5 | — | — |
- **Cumulative ablation** (`ablations.jsonl`, median of 2, µs):
  | variant | 2304 wb fp32 | 2304 noaff fp32 | 2048 wb fp32 | 256×3072 wb fp32 | 256×3072 wb 16-bit |
  |---|---|---|---|---|---|
  | full | 76.6 | 73.1 | 67.7 | 17.4 | 18.7 |
  | no-normalize | 75.2 | 73.2 | 67.6 | 14.9 | 15.2 |
  | no-write | 62.5 | 51.1 | 58.5 | 17.0 | 17.9 |
  | no-normalize + no-write | 45.7 | 45.0 | 42.1 | 12.3 | 12.7 |
  | no-read | 53.7 | 53.3 | 47.9 | 15.1 | 17.1 |
  | compute + sync (no read, no write) | 34.1 | 24.2 | 34.1 | 14.6 | 15.6 |
  | DM + sync (no compute) | 74.1 | 73.4 | 66.2 | 11.1 | 11.2 |
  | read-only | 36.9 | 35.8 | 33.6 | 7.4 | 7.2 |
  | write-only | 48.6 | 43.3 | 43.6 | 10.9 | 10.8 |
  | sync-only | 9.1 | 3.9 | 9.1 | 6.9 | 6.8 |

  8192×256: full 26.9 / 27.5 (wb fp32 / wb 16-bit), no-normalize 23.2 / 22.9, DM + sync 22.0 / 22.3, compute + sync 19.4 / 19.6.
- **Verdict on the focus: DRAM-bound, no ideas spent on it.** DM + sync is 74.1 µs against full 76.6, and the measured COPY floor (Perf 2, `dram_ceiling`) is 72.1 µs; removing normalize saves 1.4 µs. Every DM lever (posted writes, trids, bank phase, coalescing, dual-NoC reads/writes, endpoints) was measured null or regressing in earlier rounds.
- **Zones, 256×3072 wb fp32** (`zones_head_256x3072_1_1.txt`; S=6, 1 row × 16 tiles per core, 48 cores; math TRISC unless noted): load ends 4.5 µs; stats/dev pass ends 5.9; publish 6.5; the writer's all-gather rendezvous runs 6.65→8.87 (2.1 µs, skew to the slowest peer); combine 8.97→11.23 (**2.26 µs for one tile row**); normalize 11.26→15.97 (**4.7 µs / 16 tiles**; the first output window lands 1.7 µs in); write tail to 16.6. 16-bit DEST: combine 2.8 µs, normalize 5.5 µs.
- **Ranked bottlenecks** (small-H S>1 cells, including roofline target 256×3072; plus normalize on the S=1 cells):
  1. normalize (whole-op ablation −2.5 to −4.6 µs on 256×3072 / 8192×256);
  2. combine, 2.26 µs per row at S=6;
  3. exchange rendezvous + peer skew, 2.1 µs;
  4. stats tail + publish, ~1.9 µs after the last chunk lands.
- **Slice-count model check** (`forceS.jsonl`): the model's S is the measured best on every small-H cell except 256×3072 noaff fp32 (S=6 15.8 vs S=4 14.9). Re-checked after graduation (below).

### Portfolio and verdicts
| # | idea | verdict | before → after (whole op, µs) | domain |
|---|---|---|---|---|
| P1 | `sfpu_normalize`: normalize on the SFPU in fp32 registers. Option `full`: fp32 DEST keeps the c′ preload + FPU MAC and moves the affine tail (·w + b) to the SFPU, with w/b fed from per-call row-replicated "P tiles" built while compute waits on the all-gather; 16-bit DEST: FPU x − p, then SFPU (D·rstd + c″)·w + b | **WIN** | 256×3072 wb fp32 17.30 → 16.47 (−4.8%), wb 16-bit −7.5%, 8192×256 wb −3.9% / −9.1%, 32×3072 wb 16-bit −15.6%; normalize math 4710 → 4105 ns (fp32), 5536 → 3267 ns (16-bit). Precision improves (fp32 rel-RMS 0.0044 → 0.0017). Rejected options: `fullc` (c′ through srcB) is **incorrect** at offset 1000 (PCC 0.93); reader-built P tiles +74% | everywhere |
| P2 | `normalize_fixed_cost`: keep the FPU normalize, cut fixed costs (c′ built in combine and carried in LREGs, math-only re-inits) | **NULL** | 256×3072 wb fp32 −2.1% (noise band), every other cell flat; 2304 wb fp32 +1.3% in 8 of 8 pairs (noise) | superseded by P1 (same stage) |
| P3 | `combine_lean`: column-0-only SFPU fold/finalize, rsqrt inside finalize, hoisted pre-gather inits, one-tile payload in both DEST modes (fp32 tf32-exact hi/lo split copied through srcA, so no `UnpackToDestFp32`; 16-bit `[p, M2 | d]`) | **WIN** | 256×3072 wb fp32 17.4 → 16.2 (−7.3%), wb 16-bit 18.8 → 15.9 (−15.3%), 32×6144 wb 16-bit −23.7%; math combine after the gather 2196 → 764 ns (fp32) | everywhere |
| P4 | `streamed_fold`: the writer pushes each all-gather slot as its peer's flag lands (relative slot order, per-peer write → flush → flag); compute folds per slot; 16-bit compensated one-pass fold | **WIN** | 256×3072 wb fp32 17.2 → 16.0 (−7.1%), 16-bit −7 to −11%, 32×3072 −8.6% | everywhere (`--dev` on 15 cells incl. multi-block and the rotating path) |
| P5 | `stats_publish_fuse`: one acquire per tile row; the dev pass accumulates Σ(x−p), Σ(x−p)² in DEST (SFPU), then SFPU row reduce + pivot copy + publish in the same acquire; `cb_dev_sq_acc` / `cb_dev_sq` deleted | **WIN** | 256×3072 wb fp32 17.33 → 16.75 (−3.3%), 8192×256 wb fp32 −4.4%, 256×1024 −6.2%; 16-bit −0.5 to −3% | everywhere |
| F | `r5_merge`: fusion of P3 + P4 + P5 + P1 (16-bit uses P4's one-pass fold on P3's payload, since streaming needs the fold before m is known; P5's publish emits P3's payload; three added 16-bit round-to-nearest fixes) | **WIN, graduated** | see below; step attribution in `perf_experiments/r5_merge/` (every step monotone within noise) | everywhere |
| P6 | `slice_count_refit` (follow-up, after the post-graduation sweep found forced S=4 beating the model's S=3 on 2304 wb by 4–5%) | **NULL** (model) | **Mechanism**: on the two NoC-swapped east columns the writer (NoC0) issues its 2·bw affine reads behind the chip-wide NoC0 x-read storm, so they finish at ~42 µs at S=3 (35.7 µs at S=4) instead of 6–8 µs, and the exchange waits on them. A smaller bw shortens that stall; no model term is warranted. **Kernel option `affine_poll`** (finish the affine multicast opportunistically, never blocking the exchange): 2304 wb −3.7 / −4.1%, 2048 wb −3.5 / −3.9%, 1024×3072 wb −5 to −6%, 4096×2048 wb 16-bit −3.9%, **but 3072×6144 wb 16-bit +13.5 to +14.5% and 512×3072 wb fp32 +6% (measured regressions)** | not graduated: no physical rule separates the two regressions from the wins (512×3072 has identical blocking in fp32, +6%, and 16-bit, −6.9%), so a guard would be a shape allow-list |

### What graduated (commit 9494156e3de)
`r5_merge` is now the op's single path in both DEST modes, affine and no-affine, one-shot and rotating exchange, S = 1..12, single- and multi-block. **There are no carve-outs.** Code replaced and **deleted**:
- the two-pass 16-bit fold (`cf_fold1_b16`, `cf_fold2_b16`, the old `cf_finalize_b16`);
- `cf_sq_to` and the old publish routines;
- the separate `rsqrt_tile` pass;
- the FPU dest-reuse affine tail and the 16-bit sub / sub / mul / mul / add normalize (`bcast_reuse_*`);
- `cb_dev_sq_acc` and `cb_dev_sq`, plus the reduce-helper pass over them;
- the whole-block push of `cb_gathered_stats` on the one-shot path;
- `UnpackToDestFp32` on every CB.

New: `cb_wbp` (P tiles) and `cb_nstats` (16-bit per-row stats tile), plus `wbp_tiles()`. `footprint_bytes` exactly equals the descriptor's CB total on every `--dev` cell. `l1_ledger.md` was already stale before this round (it still lists `cb_local_mean`) and is not updated. The `footprint_bytes` comments are the current record.

**Instrumentation:**
- New zone `c_wbp_prep`.
- The publish now runs inside the row's acquire, under `c_publish`.
- `c_combine` includes the per-slot waits.
- Markers per (core, RISC): max 22 on 256×3072 (same as HEAD) and 30 on the focus. Zones cover each RISC's full span.

**Cost model.** `C_EXCHANGE` / `C_COMBINE` and the model's payload input are unchanged. Post-graduation forced-S sweep (`forceS_after.jsonl`): the model's S is within noise of the best on every small-H cell, including the old 256×3072 noaff fp32 miss (S6 13.8 vs S4 13.7). Switching the payload input to the real 1 tile flips only 3072×6144 wb 16-bit to S=6, which measures +4.6% slower, so it was rejected (P6).

### Whole op, before → after
HEAD copy (`r5_breakdown/head_pkg`) vs graduated tree, same process, alternating, median of 3 (`r5_breakdown/ab_guard.jsonl`, µs):
| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304×3072 (focus)** | **76.3 → 75.7 (−0.8%, flat)** | 75.7 → 75.9 (+0.3%) | 73.4 → 72.3 (−1.5%) | 72.4 → 72.0 (−0.5%) |
| 2048×3072 | 67.6 → 67.7 (+0.1%) | 68.9 → 66.9 (−2.9%) | 64.1 → 64.6 (+0.8%) | 64.0 → 64.1 (+0.2%) |
| **256×3072 (roofline target)** | **17.3 → 14.2 (−18.0%)** | 18.6 → 14.3 (−23.5%) | 15.6 → 13.5 (−13.3%) | 16.7 → 13.4 (−19.7%) |
| 32×3072 | 14.5 → 11.2 (−22.3%) | 15.6 → 10.6 (−32.0%) | 11.9 → 9.7 (−18.0%) | 13.1 → 9.1 (−30.9%) |
| 256×1024 | 11.0 → 8.5 (−22.8%) | 11.8 → 8.8 (−25.9%) | 9.8 → 8.1 (−17.0%) | 10.5 → 8.1 (−22.9%) |
| 128×512 | 9.2 → 7.4 (−19.0%) | 9.5 → 7.2 (−24.3%) | 7.8 → 6.6 (−15.9%) | 8.3 → 6.8 (−18.4%) |
| 8192×256 (S=1) | 26.7 → 23.7 (−11.2%) | 27.4 → 23.6 (−13.7%) | 24.4 → 23.3 (−4.6%) | 24.0 → 23.6 (−1.8%) |

**Large guard cells** (`ab_large.jsonl`, median of 3):
- 23040×3072: wb 16-bit −0.7%, noaff 16-bit −2.1%. wb fp32 read +3.0% at median of 3; a 6-rep recheck gave −1.2% (samples swing 763–811 µs), so it is noise.
- 23008×3072 noaff 16-bit −2.8%.
- 23040×2048: wb 16-bit −2.4%, wb fp32 −1.7%.
- 23040×2464: wb 16-bit −2.5%, wb fp32 −2.1%.
- 15360×2048 wb 16-bit −2.2%.
- 23040×1024 wb 16-bit −1.1%.
- 11520×2048 wb 16-bit −4.6%.
- 4096×2048 wb 16-bit −9.7%.
- 4096×4096 wb fp32 (rotating) −1.3%.
- 3072×6144 wb fp32 −2.5%.

**Guard-set no-regression: 0 of 28 INPUTS cells and 0 of 14 large cells are slower beyond noise.**

**Precision** is better than HEAD on every cell; PCC ≥ 0.99985 everywhere:
- fp32 DEST rel-RMS: wb 0.0044 → 0.0017; noaff 0.0017, bit-identical to HEAD.
- 16-bit DEST rel-RMS: 0.0082–0.0096 → 0.0031–0.0035.
- Offset 1000: wb fp32 0.0017, wb 16-bit 0.0039, noaff 16-bit 0.0047.

**Test suites:**
- Golden: 52 passed, 49 skipped.
- Acceptance + precision baseline: 73 passed.
- Acceptance under `--dev` (watcher, NoC sanitizer): 41 passed.
- The fusion agent's `--dev` set: 22 cells, including S>1 multi-block (8192×8192 S=4 9 blocks, 4096×8192, 8192×3072) and the rotating path.

**After-breakdown, 256×3072 wb fp32** (`r5_merge/zones_z_s4_norm_256x3072_1_1.txt`, math TRISC):
| stage | before | after |
|---|---|---|
| dev + publish | ends 6.5 µs | ends 6.1 µs |
| `c_wbp_prep` | — | 0.93 µs (overlaps peers landing) |
| combine | 2.26 µs after the gather | ends 8.9 µs, 0.55 µs after the exchange ends (8.3 µs) |
| normalize | 4.7 µs | 4.0 µs |
| BRISC max | 17.3 µs | 14.4 µs |

The all-gather (`w_exchange_rounds` ~2.1 µs) is now the stage between stats and normalize.

### Leads for the next round
- **The focus stays DRAM-bound** at 75.7 µs against the 72.1 µs measured COPY floor (1.05×).
- **`affine_poll`** (P6, `perf_experiments/slice_count_refit/graduate.patch`) is a 3.5–6.9% lever on large affine cells, blocked by two unexplained regressions. Root-causing why releasing the stalled east-column group early slows the west columns' reads on 3072×6144 wb 16-bit could earn a physical carve-out. A related option: un-swap the NoC on the writer only for the affine fetch.
- **256×3072 is now ~14.2 µs against a ~8.9 µs DM floor.** Remaining: the all-gather (~2.1 µs, mostly NoC volume; P4 measured another ~0.5 µs from sending only column-0 bytes), normalize 4.0 µs, and the stats pass ~3 µs.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `rsqrt_tile_init` + `rsqrt_tile` (and the `VectorMode` SFPU dispatch) | capability | `VectorMode` offers only R / C / RC (whole faces); the stats live in column 0, which is 8 of the 32 face-0/2 SFPU vectors, so the helper runs 2× the work and needs its own init pass between the finalize and the pack. Needed: a vector-subset (column-0 / lane-mask) mode and an rsqrt callable inside a caller's SFPU routine. Same `_calculate_sqrt_body_<approx=false>` body, so column 0 is bit-identical | 17300 (whole op, 256×3072 wb fp32, HEAD combine) | 16600 (lean column-0 combine incl. in-finalize rsqrt, option A) | `kernels/layer_norm_adaln_stats_sfpu.hpp:164` (`cf_rsqrt_slot1`), called from `kernels/layer_norm_adaln_compute.cpp:393` (`c_combine`) |
| `ttnn.Mcast` / `McastArgs` (one-shot all-gather) | capability | Extends round 1's row. Beyond the missing all-to-all schedule with per-sender flags, it has no per-slot consumer push: the landing CB can only be released as a whole after the last peer, so the consumer cannot fold early arrivals. Needed: an all-gather mode that pushes each slot as its flag lands | 21270 (helper rotating schedule, round 1, 256×3072 wb fp32) | 17200 → 16000 (HEAD raw one-shot → streamed, same cell) | `kernels/layer_norm_adaln_writer.cpp:129` (`exchange_streamed`) |
| `eltwise_chain` + `compute_kernel_lib::reduce` (stats dev + publish) | capability | (a) No chain element accumulates an SFPU result into a separate DEST slot across windows of one acquire, so Σ(x−p), Σ(x−p)² had to round-trip through L1-acc packs. (b) `reduce()` takes its input only from a CB, which re-introduces the pack → unpack this removes. (c) Each helper owns its own acquire, so reduce + pivot copy + publish cannot share one. Needed: a DEST-resident accumulate element and a `reduce()` variant whose input is already in DEST | 17327 (whole op, 256×3072 wb fp32, L1-acc + reduce helper) | 16748 | `kernels/layer_norm_adaln_compute.cpp:254` (`c_dev_sq`), `:295` (`sfpu_reduce` + publish); `kernels/layer_norm_adaln_stats_sfpu.hpp:52` (`cf_acc_sq`) |
| `eltwise_chain` / `DestReuseBinary` (normalize affine tail, 16-bit normalize) | capability | No SFPU chain element takes a row- or column-broadcast operand from another DEST slot, there is no SFPU multiply-add element with both extra operands from DEST, and nothing keeps values in SFPU registers across chain calls / DEST windows (rstd, c″, c′ per 4-row strip). Needed: an SFPU MAD element with DEST-strip broadcast operands that persist across windows | 4710 / 5536 (normalize math, FPU dest-reuse path, 256×3072 wb fp32 / 16-bit) | 4105 / 3267 | `kernels/layer_norm_adaln_compute.cpp:427` (`c_normalize`), `:344` (`c_wbp_prep`); `kernels/layer_norm_adaln_stats_sfpu.hpp:334-509` (`nz_wbp_prep`, `nz_affine_window`, `nz_full16_window`) |

The kernel-head justification comments for all four are at the top of `kernels/layer_norm_adaln_compute.cpp`, at `exchange_streamed` in the writer, and in the SFPU header. The 16-bit compensated fold (`cf_fold_b16`) extends round 1's existing custom-SFPI combine bypass: a runtime-count fold over CB tiles is still not a chain element. Earlier rounds' bypasses that still apply are unchanged. The `bcast_reuse_*` FPU dest-reuse bypass is gone together with its path.

## Perf 1 (6th tournament)
- **Date**: 2026-11-16. Sixth tournament overall; the pipeline numbers it "Perf 1" (round 1 of 5 of a new series). Coordinator + 4 `perf-part-optimizer`s, then one follow-on option measured by the coordinator. Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. Precision contract unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md` (2304×3072, 2048×3072, 256×3072, all dynamic_wb fp32). Only cells > 32 tiles were measured; ~2–3% is noise.
- **Tooling** (`perf_experiments/r6_breakdown/`): `test_run.py` (single op, optional zone CSV), `test_ab.py` / `test_ab2.py` + `head_pkg/` (HEAD 78f3167cc31) / `stack_pkg/` (alternating same-process A/B), `ablation_gates.patch` (`LN_ADALN_ABLATE=NO_READ,NO_WRITE,NO_COMPUTE,NO_NORM`, rewritten for the Perf 3 kernels; applied for the ablation runs, then reverted), `zones.py`, `ab_summary.py`. Precompile still fails here (`ttnn.graph.up_front_clear` missing): every run used `--no-precompile`. `--dev` runs need the profiler off (`--dev` + profiler overflows BRISC firmware; not op-related).

### Measured breakdown (before; HEAD 78f3167cc31)
- **Headline** (`baseline.jsonl`, one fresh run, µs): 2304×3072 wb fp32 77.3, wb 16-bit 75.4, noaff fp32 73.7; 2048×3072 wb fp32 68.2; 256×3072 wb fp32 14.86, wb 16-bit 14.87, noaff fp32 13.63.
- **Cumulative ablation** (`ablations.jsonl`, µs):
  | variant | 2304 wb fp32 | 2304 noaff fp32 | 2048 wb fp32 | 256×3072 wb fp32 | 256×3072 wb 16-bit |
  |---|---|---|---|---|---|
  | full | 77.3 | 73.7 | 68.2 | 14.86 | 14.87 |
  | no-normalize | 76.0 | 71.3 | 67.7 | 13.13 | 12.40 |
  | no-write | 58.4 | 50.0 | 54.5 | 13.68 | 12.92 |
  | no-normalize + no-write | 45.5 | 43.7 | 40.9 | 10.12 | 9.75 |
  | no-read | 52.9 | 50.9 | 46.8 | 12.17 | 12.12 |
  | compute + sync | 30.2 | 23.1 | 30.2 | 11.22 | 10.12 |
  | DM + sync | 74.7 | 74.0 | 65.9 | 11.16 | 10.31 |
  | read-only | 37.1 | 36.2 | 33.8 | 7.37 | 7.19 |
  | write-only | 49.2 | 42.2 | 44.0 | 10.77 | 9.40 |
  | sync-only | 8.87 | 3.43 | 8.85 | 6.57 | 5.51 |
- **Zones, focus 2304 wb fp32** (`zones_2304x3072_wb32_full.txt`): `w_affine` p50 6.6 µs but **max 41.9 µs** on the two NoC-swapped east columns (their NoC0 writer's w/b reads queue behind the chip-wide NoC0 x-read storm); `w_exchange_rounds` max 34.8 µs waits on it, so those groups normalize/write last (store end max 76.6 vs p50 61.5). Even with no x traffic (sync-only) `w_affine` takes 5.7 µs — the affine accounts for the 8.9 vs 3.4 µs sync-only gap between wb and noaff.
- **Zones, 256×3072 wb fp32** (`zones_256x3072_wb32_full.txt`): reads end 4.4 µs → publish ends 6.2 (1.8 µs stats tail after the last chunk) → `w_exchange_rounds` 6.3–8.3 (2.0 µs, NoC-volume: S−1 = 5 full fp32 tiles per member) → combine to 8.9 → `c_normalize` 8.9–12.9 (4.0 µs / 16 tiles) → store end 13.5.
- **Ranked** (gated by the measured DM ceiling, Perf 2 `dram_ceiling`: COPY floor 72.1 µs at the focus):
  1. focus: DRAM-bound (DM + sync 74.7 vs 72.1 floor) apart from the east-column affine stall on the exchange critical path;
  2. 256×3072: normalize 4.0 µs (ablation −1.7 fp32 / −2.5 µs 16-bit);
  3. 256×3072: all-gather 2.0 µs;
  4. 256×3072: stats tail 1.8 µs.

### Portfolio and verdicts
| # | idea | verdict | before → after (µs, whole op unless noted) | domain |
|---|---|---|---|---|
| Q1 | `affine_fetch_fix`: the w/b fetch + column multicast runs on the core's NoC1 RISC — the writer on default columns, the **reader** on the NoC-swapped east columns, which issues the w/b reads before its first x row and finishes (barrier, mcast, flag) right after it lands. Other options: reader fetches and blocks first (V2: 256×3072 +6.6%, 8192×256 +5.6%); reader reads + writer multicasts on NoC0 (8192×256 +9.2%); cheaper fetch (no headroom: normal columns' affine ends long before compute needs it) | **WIN** | focus −1.2 to −2.7% (noise band); 2048 wb fp32 −2.8%, wb 16-bit −3.2%; 1024×3072 wb −5.8%; 512×3072 wb 16-bit −8.1%. Bit-exact | everywhere S>1 with affine; **carve-out S == 1 (measured: 8192×256 wb fp32 +8.3%, n=3)**: releasing the east columns early lets their NoC0 output writes overlap the west columns' NoC0 read tail, and with S = 1 no exchange waits on the affine, so there is nothing to gain. Same mechanism likely explains Perf 3's `affine_poll` regressions |
| Q2 | `normalize_sfpu_macro`: every normalize SFPU MAD issued as SFPLOADMACRO (load, MAD from a template with operands in fixed LREGs, store on later cycles): fp32 affine `D*w+b`; 16-bit `(D*rstd+c″)` macro + one explicit MAD for `*w+b`; 16-bit noaff macro only | **WIN** | stage (math `c_normalize`, 256×3072, 16 tiles): wb fp32 4000 → 3290 ns, wb 16-bit 3271 → 2948, noaff 16-bit 2481 → 1542; 2304 (64 tiles) 14019 → 11125. Whole op: 256 noaff 16-bit −6.4%, 32×3072 wb fp32 −6.5%, 8192×256 wb fp32 −2.5%; 256 wb fp32 flat alone (the store tail was next). **Bit-identical** | everywhere |
| Q3 | `compact_stats_payload`: after the publish an SFPU pass (Horner insert with `SFPSHFT2 SUBVEC_SHFLROR1` + lane masks) packs each stat's 8 column-0 vectors into one vector, so the payload is the first 512 B (fp32) / 256 B (bf16) of the tile; the one-shot exchange sends only that prefix; fold / finalize run on compact vectors (1 per stat instead of 8) and expand m / rstd back to column 0. Options: compact compute + full-tile send (13.98), one write per peer for all rows (same) | **WIN** | 256×3072 wb fp32 14.82 → 13.41 (−9.5%), wb 16-bit −8.0%, noaff −5.8 to −8.4%, 32×3072 −6 to −9.7%, 256×1024 −7.5%; `w_exchange_rounds` p50 1916 → 1058 ns, combine math 1209 → 351 ns. Focus flat. **Bit-identical** | everywhere. The agent saw 3072×6144 wb 16-bit +10% (median of 3); in the merged stack it measured +2.0% (n=3) and −0.6% (n=5 recheck), and −0.5% in the final A/B: noise on a ±7% cell, **no carve-out** |
| Q3b | `oneshot_model` (follow-on of Q3, measured by the coordinator, `ab_oneshot.jsonl`, stack vs stack + option): size the one-shot byte check by the compact prefix. Every feasible blocking (S·bh ≤ `GATHER_TILE_CAP`) is then one-shot | **WIN** | 8192×3072 wb fp32 274.3 → 256.8 (−6.4%), 3072×6144 wb fp32 −4.1%, 4096×4096 wb fp32 −1.8%; 8192×3072 wb 16-bit / noaff, 3072×6144 wb 16-bit, 8192×8192, 4096×8192 flat. 8192×3072 wb fp32 output differs from the stack by fold order only (rel-RMS 0.0017 both) | everywhere — the rotating path became unreachable and was **deleted** |
| Q4 | `dev_pass_macro`: the 4 SFPU vectors of a tile that cover the same 4 rows fold into one strip accumulator (8 + 8 per row in one DEST tile, D0) → one more work slot per window (fp32 3, 16-bit 7) and 4× less accumulator traffic; the pivot tile is copied into D0 at acquire start and stashed in its parity-1 vectors; after the last window one SFPU pass (`ds_lane_publish`) lane-reduces each row in fp32 LREGs (SFPSHFT2 rotate-and-add) and writes HEAD's payload. Rejected options: HEAD `sfpu_reduce` after strips (2596 ns bench), compensated 16-bit accumulators (no gain), compact-loop code (~100 ns slower tail), RNE of 16-bit M2/d (rel-RMS worse) | **WIN** | bench (1 row × 16 tiles): fp32 3283 → 2434 ns, 16-bit 2760 → 2361; payload ready after last chunk 1790 → 1399 ns. Whole op 256×3072 wb fp32 14.51 → 13.96 (−3.8%), wb 16-bit −2.4%, 256×1024 −3.8%, 8192×256 −2.1%. fp32 bit-identical (multi-block cells: summation order only). **16-bit**: statistics more accurate (M2 max err vs float64 1.5e-2 → 8.3e-3; offset 1000: 0.217 → 0.0075); output rel-RMS +0.3 to +2.8% at offset 0 (e.g. 0.00327 → 0.00329; worst 128×512 noaff 0.00334 → 0.0036), better at offset 1000 (0.00392 → 0.00347 wb, 0.00471 → 0.00386 noaff). Golden green; accepted under "statistics at least as accurate" | everywhere |

Breadcrumbs for every idea and verdict: `agent_logs/perf-coordinator_breadcrumbs.jsonl`. Artifact dirs: `perf_experiments/{affine_fetch_fix,normalize_sfpu_macro,compact_stats_payload,dev_pass_macro}/` (each with `graduate.patch`, `head_pkg/`, candidate packages, A/B logs, zones).

### Aggregation and what graduated
All five are compatible: Q1 is dataflow + host NoC roles, Q2 normalize, Q4 dev pass + publish, Q3 publish tail + exchange + combine, Q3b host model. Applied in that order; the only conflict was the publish: it is now `ds_lane_publish` (Q4, HEAD payload layout) followed by `cf_compact_payload` (Q3). Every win is the op's **single path**:
- **Deleted**: `cf_acc_sq`, the `sfpu_reduce<ROW>` publish, `cf_publish_fp32` / `cf_publish_b16` (Q4); `nz_affine_window`, `nz_full16_window`, `nz_norm16_window` (Q2); `cf_fold_fp32`, `cf_fold_b16`, `cf_finalize_fp32`, `cf_finalize_b16`, `cf_rsqrt_slot1` (Q3); the rotating `ttnn.Mcast` all-gather — the writer's `StatsExchange` / `McastArgs`, the host `ttnn.Mcast` / `attach_absent_mcast` and `STATS_MCAST_PREFIX` (Q3b). The host now asserts `S == 1 or oneshot`. As a consequence the NoC swap no longer has a rotating-path exception (only the measured 16-bit affine ≥ 4-rows carve-out remains), and Q1 applies to the formerly rotating cells too.
- **The one carve-out**: Q1's S == 1 keeps the writer fetch (`affine_in_reader()` in the descriptor; measured +8.3% at 8192×256 wb fp32, mechanism above). It is written as the exception (`S > 1` required for the reader fetch), not an allow-list.
- New kernel header `kernels/layer_norm_adaln_affine.hpp` (issue / finish / receive split of the column multicast, shared by reader and writer).
- `op_design.md` still describes the rotating exchange and the original stats chain; as in earlier rounds, the kernel head comments and this changelog are the current record. `l1_ledger.md` remains stale (the `footprint_bytes` comments are current; Q3b raises `gather_depth` to 2 on formerly rotating multi-block cells).

**Instrumentation:** new zones `aff_issue`, `aff_barrier`, `aff_mcast`, `aff_wait` (both RISCs) and `r_affine` (reader, east columns). Markers per (core, RISC): max 36 on the focus, 28 on 256×3072; every RISC's zones still reach its kernel end (`w_final_barrier`, `c_normalize`).

### Whole op, before → after
HEAD copy vs graduated tree, same process, alternating, median of 3 (`r6_breakdown/ab_final.jsonl`, µs):
| shape | wb fp32 | wb 16-bit | noaff fp32 | noaff 16-bit |
|---|---|---|---|---|
| **2304×3072 (focus)** | **76.1 → 74.6 (−2.0%)** | 75.8 → 75.5 (−0.4%) | 72.3 → 71.1 (−1.6%) | 72.4 → 71.0 (−1.9%) |
| 2048×3072 | 67.7 → 65.0 (−3.9%) | 66.4 → 65.0 (−2.2%) | 64.6 → 64.5 (−0.2%) | 64.4 → 64.7 (+0.5%) |
| **256×3072 (roofline target)** | **14.5 → 12.5 (−13.9%)** | 14.3 → 12.9 (−10.0%) | 13.7 → 11.7 (−14.2%) | 13.6 → 11.8 (−13.0%) |
| 32×3072 | 11.2 → 9.1 (−19.0%) | 10.6 → 8.9 (−16.1%) | 9.6 → 7.9 (−18.6%) | 9.2 → 7.2 (−21.2%) |
| 256×1024 | 8.6 → 7.5 (−11.9%) | 8.8 → 7.7 (−12.2%) | 7.9 → 6.9 (−12.6%) | 8.2 → 6.8 (−16.7%) |
| 128×512 | 7.4 → 6.5 (−12.9%) | 7.2 → 6.5 (−9.0%) | 6.6 → 5.8 (−12.0%) | 6.7 → 5.7 (−15.2%) |

Guard cells: 8192×256 wb fp32 −3.8%, wb 16-bit −4.5%; 512×3072 wb fp32 −23.6%, wb 16-bit −13.5%; 1024×3072 wb fp32 −4.8%; 3072×6144 wb fp32 −9.7%, wb 16-bit −0.5%; 4096×4096 wb fp32 −5.4%; 8192×8192 wb fp32 −6.4%; 8192×3072 wb fp32 −8.8%, wb 16-bit −1.0%; 4096×8192 wb fp32 −6.7%; 23040×3072 wb fp32 −5.0%; 23040×2048 wb 16-bit −2.2%. Borderline cells of the pre-Q3b stack were rechecked at n = 5 (`ab_recheck.jsonl`): 2048 noaff fp32 +1.2%, noaff 16-bit +0.9%, 3072×6144 wb 16-bit −0.6%.

**Guard-set no-regression: 0 of 38 cells slower beyond noise** (largest: 2048×3072 noaff 16-bit +0.5%).

**Precision**: fp32 DEST bit-identical to HEAD on every single-block cell (rel-RMS 0.0017); multi-block / formerly rotating cells differ by summation / fold order only (rel-RMS unchanged). 16-bit DEST: see Q4 (rel-RMS 0.0032–0.0036, PCC ≥ 0.99984).

**Tests**: golden 52 passed, 49 skipped (on the 4-way merge and again on the final tree); `--dev` (watcher, NoC sanitizer, LLK asserts, profiler off) 12 cells on the final tree incl. formerly rotating 8192×3072 / 3072×6144 / 4096×4096 / 4096×8192, 9-block 8192×8192, S = 1 8192×256, partial-grid 256×1024 / 32×3072; plus each agent's own `--dev` sets (10–23 cells).

**After-breakdown** (`zones_*_wb32_after.txt`):
- focus: east-column affine multicast ends 11.6 µs (was ~42); exchange max no longer tracks it. What remains is the NCRISC read tail (max 57.5 µs on the slowest core) — DRAM, 74.6 µs vs the 72.1 µs COPY floor (1.035×).
- 256×3072 wb fp32: publish ends 5.7 (dev pass to 5.3), exchange 5.9–6.9 (p50 0.9 µs), combine 7.0–7.3, normalize 7.4–10.6 (3.2 µs math), store end 11.5; BRISC max 13.0 µs (was 14.8).

### Leads for the next round
- 256×3072 (12.5 µs vs ~8.9 µs DM floor): normalize (3.2 µs; non-SFPU work ~150 ns/tile: FPU MAC, P-tile copy, inits, pack sync) and the writer's store tail are now the critical path; `c_wbp_prep` (0.9 µs math) sits between publish and the exchange end.
- dev pass: the first non-first window runs ~280 ns/tile vs ~100 ns/tile warm (not root-caused); an SFPLOADMACRO accumulate was estimated at ~100 ns off the tail.
- The focus is DRAM-bound (1.035× the measured COPY floor). `roofline_targets.md`'s 460.8 GB/s target remains ~15% above anything this card sustained in the Perf 2 `dram_ceiling` bench.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `eltwise_chain` SFPU MAD / any compute helper (normalize MADs) | capability | No helper or chain element issues a user MAD through SFPLOADMACRO (load / MAD / store overlapped from one instruction per vector); the LLK macro users (typecast / where / mul_int32 / reduce) are fixed functions. Needed: a pipelined (load-macro) MAD element with DEST-strip operands. No helper path exists to measure; the pair is the previous raw sfpi path vs the macro path | 4000 (prior sfpi normalize stage, 256×3072 wb fp32) | 3290 | `kernels/layer_norm_adaln_stats_sfpu.hpp:476-626` (`nm_*`), called from `kernels/layer_norm_adaln_compute.cpp:464,513` |
| pack helpers / `eltwise_chain` (stats payload compaction) | capability | No chain element moves values across SFPU lane-columns (`SFPSHFT2` rotate), and no pack helper emits a partial-tile payload, so a payload with 1/8 meaningful lanes travels as a whole tile. Needed: a lane-gather element and a prefix / partial-face pack | 14820 (whole op, 256×3072 wb fp32, full-tile payload) | 13410 | `kernels/layer_norm_adaln_stats_sfpu.hpp:183-222` (`cf_gather`, `cf_compact_payload`), `kernels/layer_norm_adaln_compute.cpp:329`; prefix send `kernels/layer_norm_adaln_writer.cpp:179` |
| `sfpu_reduce<SUM, REDUCE_ROW>` (+ `eltwise_chain`, `compute_kernel_lib::reduce`; stats dev pass + publish) | capability | Extends Perf 3's row. `sfpu_reduce<ROW>` reduces whole DEST tiles and stores the sums back to DEST (in 16-bit DEST that rounds them to bf16); it cannot reduce strip accumulators held in registers, and no element folds same-row vectors from different faces / tiles into one DEST-resident accumulator across the windows of one acquire. Needed: a DEST-resident strip-accumulate element and a row-reduce over register inputs | 14510 (whole op, 256×3072 wb fp32, HEAD `sfpu_reduce` publish) | 13960 | `kernels/layer_norm_adaln_stats_sfpu.hpp:59` (`ds_acc_strip`), `:131` (`ds_lane_publish`); `kernels/layer_norm_adaln_compute.cpp:310,328` |
| `ttnn.Mcast` / `McastArgs` (affine column multicast) | capability | Extends the existing raw column-multicast bypass (unequal column heights). The helper has no split of a multicast into issue / finish steps that let the producer overlap its own DRAM reads with other work, and no way to put the sender on a different RISC (here the reader, on the NoC-swapped columns) than the CB consumer handshake expects. Needed: an issue/finish split and a sender-RISC choice per core | 67600 (whole op, 2048×3072 wb fp32, writer fetch) | 65700 | `kernels/layer_norm_adaln_affine.hpp:39-108`; `kernels/layer_norm_adaln_reader.cpp:67,116,126` |

The kernel-head justification comments are in the compute kernel header (`normalize_sfpu_macro`, `dev_pass_macro` paragraphs), the SFPU header (`compact_stats_payload` section) and `layer_norm_adaln_affine.hpp`. The writer's one-shot streamed exchange (Perf 3 row, `ttnn.Mcast` capability) is now the op's only all-gather: the rotating `ttnn.Mcast` helper use was deleted, so the op no longer uses that helper for the exchange at all.

## Perf 1 (7th tournament)
- **Date**: 2026-11-17. Seventh tournament overall; the pipeline numbers it "Perf 1" (round 1 of 4 of a new series). Coordinator + 4 `perf-part-optimizer`s. Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. Precision contract unchanged in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Context**: an earlier "Perf 2" of the 6th series was interrupted after its breakdown and while its subagents were still running; nothing of it was committed. Its breakdown (`perf_experiments/t6r2_breakdown/`) ran on this same HEAD (b5a019becf1, kernels unchanged), so this round reuses its cumulative ablation and confirms it with one fresh headline + zone run (`perf_experiments/r7_breakdown/`). Three of its ideas were resumed in their existing dirs. The fourth (`affine_prep_hoist`) had already finished its measurements on this HEAD and is recorded as a null below.
- **Focus**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md`: 2304×3072, 2048×3072 and 256×3072, all dynamic_wb fp32. Only cells > 32 tiles were measured, and a difference of ~2–3% is noise. Precompile still fails here (`ttnn.graph.up_front_clear`), so every run used `--no-precompile`.

### Measured breakdown (before; HEAD b5a019becf1)
- **Headline** (`r7_breakdown/baseline.jsonl`, one fresh run, µs): 2304×3072 wb fp32 74.2, wb 16-bit 73.4, noaff fp32 70.9; 2048×3072 wb fp32 67.7; 256×3072 wb fp32 12.91, wb 16-bit 12.84, noaff fp32 11.66; 1024×3072 wb fp32 33.5; 512×3072 wb fp32 17.8; 8192×256 wb fp32 22.4.
- **Cumulative ablation** (`t6r2_breakdown/ablations.jsonl`, same HEAD, µs):
  - **256×3072 wb fp32**:

    | variant | µs |
    |---|---|
    | full | 12.66 |
    | no-normalize | 11.06 |
    | no-write | 11.70 |
    | no-normalize + no-write | 8.57 |
    | compute + sync | 9.53 |
    | DM + sync | 9.75 |
    | read + sync (no compute, no write) | 5.96 |
    | sync only, wb | 5.21 |
    | sync only, noaff | 2.05 |

  - **Focus (2304×3072) wb fp32**:

    | variant | µs |
    |---|---|
    | full | 73.7 |
    | DM + sync | 74.2 |
    | compute + sync | 25.0 |
    | read-only | 37.1 |
    | write-only | 46.9 |
    | sync only | 8.0 |
    | noaff full | 71.0 |
- **Zones** (`r7_breakdown/zones_*_1_1.txt`):
  - **Focus**: NCRISC span max 56.5 µs vs p50 25.0, because the top grid row's reads are slow. Those slowest cores then carry ~17 µs of serial work after their reads (dev, publish, wbp_prep, exchange, combine, normalize ~11 µs, store).
  - **256×3072 wb fp32** (S = 6, 48 cores, p50 times from kernel start): reads end at 4.5 µs and the publish at 5.9. `c_wbp_prep` runs 5.9–6.8 and is hidden by the exchange, which ends at 6.9. Combine runs to 7.3 and normalize 7.3–10.6 (3.2 µs for 16 tiles); the last store ends at 11.9 and the BRISC max is 13.1.
- **Ranked**:
  1. Focus: DRAM-bound. DM + sync (74.2) ≈ full, at 1.03× the measured COPY floor (72.1). The only lever left is the work split: forced S = 6 measured −3.3% in one run.
  2. 256×3072: normalize. Removing it saves 1.6 µs, or 3.1 µs together with the write.
  3. 256×3072: the store drain after normalize (1.3 µs). Removing the write saves 1.0 µs.
  4. 256×3072: the stats tail (last chunk → publish, 1.4 µs) plus the exchange (0.9 µs).
  5. Affine fetch / prep: hidden. Removing the fetch entirely bounds the gain at −3.2% on 256×3072 and −0.9% on the focus.

### Portfolio and verdicts
| # | idea | verdict | before → after (µs, whole op, same-process alternating, median of 3) | domain |
|---|---|---|---|---|
| R1 | `worksplit_refit`: re-fit `select_blocking` to the current kernels as one model, `cost = core_h·bw + bw·[affine] + C_PEER·S·core_h` with a single `C_PEER = 0.7`. This replaces `C_EXCHANGE = 2.0` + `C_COMBINE = 0.75` (per-peer cost 2.75 fp32 / 3.5 16-bit). The forced-S data bracket `C_PEER` between 0.53 and 0.89. It also deletes the 16-bit-affine NoC-swap carve-out (`core_h ≥ 4` kept the default NoC pair), which now regresses: with the swap on, 23040×3072 wb 16-bit is −6.3% and 8192×8192 −5.2%. The positional row-to-group placement option was inconsistent and was dropped | **WIN** | focus 75.04 → 70.82 (−5.6%; S 3 → 6). 256×3072 wb fp32 −3.7%, wb 16-bit −5.5% (S 6 → 12). 4096×2048 wb 16-bit −6.2%, 8192×8192 wb 16-bit −5.9%, 3072×6144 noaff −4.8%. 0/22 cells slower beyond noise. Summed regret over 30 forced-S cells 0.80 → 0.12; worst cell 13% → 1% | everywhere (host only) |
| R2 | `normalize_cost`: per-tile normalize cost at fixed precision. W1: first DEST windows of 1 and 2 tiles, then 3, so the first output tile reaches the writer earlier. sx2: `D = −m`, then `D += x − p` as a 1-phase accumulating subtract, then `D *= rstd` and the affine on the SFPU. The combine-into-normalize fusion was not built: W1 already showed that an earlier first tile does not move the op | **NULL** | W1: 256 wb fp32 12.85 → 12.47 (−2.9%, borderline); focus +0.5%, 512 +1.2%. Bit-exact. sx2: math `c_normalize` 3206 → 2813 ns (−12%), whole op −3.6% (borderline), **rel-RMS +0.3–0.7%**, which fails the no-precision-loss rule. Mechanism: at 256×3072 the store is DRAM-write-paced (~190 ns/tile, ~520 GB/s), so normalize throughput is not what limits the op | n/a. Neither graduated (W1 flat; sx2 precision) |
| R3 | `stats_tail`. **bflush**, the chosen option: (a) `ds_lane_publish` + `cf_compact_payload` fused into one SFPU pass (`ds_publish_compact`; the payload is built in LREGs and never round-trips through DEST); (b) the writer issues all S−1 payloads, then **one** `noc_async_writes_flushed`, then all S−1 flags. HEAD flushed and flagged once per peer; the own-slot read barrier now waits until after the issue. Option `cand_pkg` (a plain 4 B flag write per peer, no flush) is **BLOCKED**: it relies on undocumented NIU in-order injection per command buffer. A linked unicast is inexpressible (the dataflow API exposes `linked` only for mcast, and the watcher flags it). A payload sentinel would need intra-packet ordering or a NaN that real inputs could produce | **WIN (small)** | 256×3072 wb fp32 12.74 → 12.26 (−3.8%), 512×3072 −3.1%, 64×3072 noaff −4.3%, others flat. Bit-exact. In the stack with R1 (S = 12, 11 peers) it adds −3.1% at 256×3072 wb fp32, −5.3% at 256 noaff fp32 and −5% at 64×3072 (below) | everywhere. Same ordering argument as HEAD (payload flushed before its flag; same static VC + route), applied once per block instead of once per peer |
| R4 | `store_drain`: the DM pair at the op's tail. Recommended option: the reader, idle after its own reads, stores the whole last block on logical columns grid_x−4..−3. Also measured: interleaved shares (+11% to +37%), last-window-only (+7% to +23%), shortest-x-hop NoC per bank (+12%), rotated start column (+5%) | **REGRESSION** | 256 wb fp32 −2.9%, 512 −5.8%, 8192×256 −6.3%, but **focus +3.9%** and **1024×3072 +2.7% to +13%**. The helper's NoC0 writes share the wrap links with the top grid row's read responses (the same coupling as the 4th tournament's K2 `dual_noc_writes`). It also needs a board-specific physical-x table | not graduated. Lead: gate the helper writes on a chip-wide "all reads done" semaphore count rather than on the core's own reads |
| R5 | `affine_prep_hoist` (from the interrupted round, same HEAD): hoist the w/b fetch → P-tile build off the post-read path. cand3 = preemptible poll-hoist | **NULL** | cand3 vs HEAD, median of 3: 256 wb fp32 12.49 → 12.49, focus 74.81 → 74.77, 2048 wb fp32 −1.7%, 256 wb 16-bit +1.5%. Upper bound (no fetch at all): 256 wb fp32 −3.2%, focus −0.9% | n/a |

Breadcrumbs for every idea and verdict are in `agent_logs/perf-coordinator_breadcrumbs.jsonl`. Artifact dirs: `perf_experiments/{worksplit_refit,normalize_cost,stats_tail,store_drain,affine_prep_hoist,t6r2_breakdown,r7_breakdown}/`, each with its `graduate.patch` where one exists, plus head/candidate packages and A/B logs.

### Aggregation and what graduated
R1 (host) and R3 (kernels) are independent and both are applied. Each is the op's **single path** with no guard:
- **Deleted**:
  - `C_EXCHANGE` / `C_COMBINE` and the payload-tile shim in `select_blocking`, replaced by `C_PEER`.
  - The 16-bit-affine NoC-swap carve-out (`swap_regresses`). The NoC swap now applies to every blocking.
  - `ds_lane_publish` and `cf_compact_payload`, replaced by the fused `ds_publish_compact`.
  - The writer's per-peer flush + flag loop.
- **Carve-outs**: none.
- **Instrumentation**: new writer zones `x_issue`, `x_flags` and `x_wait` inside `w_exchange_rounds`. Markers per (core, RISC) reach a max of 58 on the focus (now S = 6, 4 rows per core) and 34 on 256×3072. Every RISC's zones still reach its kernel end (`w_final_barrier` p50 end 55.19 vs BRISC span p50 55.35 µs; `c_normalize` ends at the TRISC span ends).
- `l1_ledger.md` / `op_design.md` remain stale, as in earlier rounds: the kernel head comments and this changelog are the current record. The earlier changelog lines that describe `C_EXCHANGE` / `C_COMBINE` are history.

### Whole op, before → after
HEAD copy (`r7_breakdown/head_pkg`) vs R1-only (`ws_pkg`) vs the graduated tree (`op`), same process, alternating, median of 3. Results are in `r7_breakdown/ab_stack.jsonl`, `ab_recheck.jsonl` (n = 5) and `ab_guard.jsonl`. All values in µs.

**Main cells** (HEAD → R1 only → R1 + R3):

| cell | HEAD | R1 only | R1 + R3 (graduated) |
|---|---|---|---|
| **2304×3072 wb fp32 (focus)** | 74.05 | 71.23 | **71.54 (−3.4%; recheck n=5: 73.94 → 71.43, −3.4%)** |
| 2304×3072 wb 16-bit | 72.46 | 71.06 | 70.74 (−2.4%) |
| 2304×3072 noaff fp32 | 71.50 | 71.12 | 71.21 (−0.4%) |
| 2048×3072 wb fp32 | 64.51 | 65.78 | 65.96 (+2.3%; **recheck n=5: 65.13 → 65.39, +0.4%**) |
| 2048×3072 wb 16-bit | 65.01 | 64.75 | 64.32 (−1.1%) |
| **256×3072 wb fp32 (roofline)** | **12.68** | 12.23 | **11.85 (−6.6%)** |
| 256×3072 wb 16-bit | 13.17 | 12.52 | 12.22 (−7.2%) |
| 256×3072 noaff fp32 | 11.92 | 11.92 | 11.28 (−5.3%) |
| 256×3072 noaff 16-bit | 11.67 | 11.47 | 11.47 (−1.6%) |
| 512×3072 wb fp32 | 17.74 | 17.59 | 17.70 (−0.2%) |
| 1024×3072 wb fp32 | 33.41 | 33.73 | 33.04 (−1.1%; recheck −0.9%) |
| 8192×256 wb fp32 (S=1) | 22.56 | 22.71 | 22.75 (+0.8%) |
| 64×3072 wb fp32 | 11.32 | 8.50 | 8.08 (−28.6%) |
| 64×3072 noaff fp32 | 8.26 | 7.44 | 7.03 (−15.0%) |
| 4096×2048 wb 16-bit | 92.29 | 87.06 | 86.04 (−6.8%) |

**Guard cells** (HEAD → graduated, µs):

| cell | HEAD | graduated | change |
|---|---|---|---|
| 32×3072 wb fp32 | 9.26 | 7.73 | −16.6% |
| 32×3072 noaff 16-bit | 7.24 | 6.73 | −7.0% |
| 128×512 wb fp32 | 6.64 | 6.55 | −1.4% |
| 256×1024 wb fp32 | 7.68 | 7.63 | −0.6% |
| 3072×6144 wb fp32 | 191.2 | 190.7 | −0.2% |
| 3072×6144 wb 16-bit | 199.9 | 192.1 | −3.9% |
| 3072×6144 noaff fp32 | 197.2 | 186.8 | −5.2% |
| 4096×4096 wb fp32 | 179.1 | 177.7 | −0.8% |
| 4096×4096 noaff fp32 | 178.2 | 168.8 | −5.2% |
| 8192×3072 wb fp32 | 263.9 | 256.5 | −2.8% |
| 8192×3072 wb 16-bit | 264.1 | 252.6 | −4.3% |
| 8192×3072 noaff fp32 | 255.2 | 246.6 | −3.4% |
| 23040×3072 wb 16-bit | 769.5 | 744.6 | −3.2% |
| 23040×3072 wb fp32 | 752.5 | 751.1 | −0.2% |
| 8192×8192 wb 16-bit | 733.2 | 696.3 | −5.0% |
| 23040×1024 wb 16-bit | 252.8 | 244.8 | −3.2% |
| 15360×2464 wb 16-bit | 402.7 | 402.7 | 0.0% |

**Guard-set no-regression: 0 of 32 cells slower beyond noise.** The largest increase is 8192×256 wb fp32 at +0.8%. 2048×3072 wb fp32 measured +2.3% at n = 3 and +0.4% on the n = 5 recheck.

**Precision**:
- Every cell whose S is unchanged is bit-identical to HEAD, including those where only the NoC swap changed.
- Cells with a new S differ only in fold / summation order; rel-RMS is unchanged within ±0.0001. The largest shift is 23040×1024 wb 16-bit, 0.00332 → 0.00342 (S 1 → 2), inside tolerance and golden green.
- The fused publish is bit-identical to the two-pass version.

**Tests**:
- Golden: 52 passed, 49 skipped.
- `--dev` (watcher, NoC sanitizer, LLK asserts, profiler off): 12 cells pass. They cover the focus wb fp32 / 16-bit, 256×3072 wb fp32 / noaff 16-bit, S = 12 32×3072 and 64×3072, S = 1 8192×256, the 4096×2048 wb 16-bit swap-now-on cell, 3072×6144 noaff, 9-block 8192×8192 wb 16-bit, multi-block 23040×3072 wb 16-bit, and ragged 96×1056.
- Each agent's own `--dev` sets (R1: 10 cells, R3: 18 cells).

**After-breakdown** (`r7_breakdown/zones_*_after.txt`):
- **Focus** (S = 6): the NCRISC read tail max is 52.7 µs (was 56.5), and the BRISC span max is 71.9 µs (was 73.6). The op is still DRAM-bound at ~0.99× the 72.1 µs COPY floor measured in Perf 2 (`dram_ceiling`), so the work split has taken it to that ceiling.
- **256×3072 wb fp32** (S = 12, 96 cores): reads end at 3.9 µs, publish at 5.2, exchange at 6.8 (`x_issue` 0.33, `x_flags` 0.33, `x_wait` p50 0.63 / max 2.5). Normalize runs to 9.0 (1.9 µs math for 8 tiles), the final barrier is at 10.5 p50, and the BRISC max is 12.1 µs.

### Leads for the next round
- The focus is at the DRAM ceiling measured on this card. `roofline_targets.md`'s 460.8 GB/s target remains above anything measured here.
- 256×3072: the store is DRAM-write-paced. On top of that come peer skew (`x_wait` max 2.5 µs at S = 12) and the normalize-to-first-tile latency. The R4 lead (helper writes gated on a chip-wide "all reads done" count) is the one unbuilt route to small-H tail gains without a shape guard.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `sfpu_reduce<SUM, REDUCE_ROW>` + pack helpers / `eltwise_chain` (stats publish + payload compaction) | capability | Extends the 6th tournament's rows for `ds_lane_publish` and `cf_compact_payload`, which are now fused into one pass. No chain element reduces register-resident strip accumulators and inserts the results into lane-compacted vectors in the same pass. Two helper calls would also need the DEST round trip that this fusion removes. Needed: a row-reduce over SFPU-register inputs whose output is a lane-gather element (`SFPSHFT2` rotate) and a prefix pack | 12740 (whole op, 256×3072 wb fp32, two-pass publish + per-peer flush; the pass alone is TRISC1 `c_publish` 446 ns) | 12260 (with the batched flush; pass alone 340 ns) | `kernels/layer_norm_adaln_stats_sfpu.hpp:184` (`ds_publish_compact`), called from `kernels/layer_norm_adaln_compute.cpp:330` |
| `ttnn.Mcast` / `McastArgs` (stats all-gather; raw `exchange_streamed`) | capability | Extends the existing raw exchange row; the bypass is not new, only its flush / flag discipline changed. The helper has no "issue N unicast payloads, one flush, N flags" batch primitive with per-sender flags that the receiver consumes in streaming order. Needed: a batched unicast all-gather with per-sender semaphores and a receiver-side "push as each lands" callback | 12740 (256×3072 wb fp32, per-peer flush) | 12260 | `kernels/layer_norm_adaln_writer.cpp:59-124` (`exchange_streamed`) |

The kernel-head justification comments are in the writer head comment (exchange paragraph, updated for bflush), the `ds_publish_compact` comment in the SFPU header, and the compute kernel's dev + publish paragraph.

## Perf 2 (8th tournament)
- **Date**: 2026-11-17. Round 2 of 4 of the series that started with "Perf 1 (7th tournament)". Coordinator + 4 `perf-part-optimizer`s (3 in parallel, plus 1 follow-up). Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. The precision contract was the same in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus**: `LOOSE_CASES` is empty, so the focus follows `roofline_targets.md` (2304, 2048 and 256 ×3072, all dynamic_wb fp32). Only cells > 32 tiles were measured. A difference of ~2–3% is noise. Precompile still fails here, so every run used `--no-precompile`.

### Measured breakdown (before; HEAD 91c45e6582f; `perf_experiments/r8_breakdown/`)
- **Headline** (`baseline.jsonl`, one fresh run, µs): 2304×3072 wb fp32 71.3, wb 16-bit 71.7, noaff fp32 71.2; 2048×3072 wb fp32 66.4; 256×3072 wb fp32 11.67, wb 16-bit 12.24, noaff fp32 11.78; 1024×3072 33.8; 512×3072 18.0; 8192×256 21.95.
- **Cumulative ablation** (`ablations.jsonl`, `ablation_gates.patch`, applied temporarily and then reverted), µs:

  | variant | 2304 wb fp32 | 2048 wb fp32 | 256 wb fp32 | 256 noaff fp32 |
  |---|---|---|---|---|
  | full | 71.7 | 65.5 | 12.18 | 11.29 |
  | no normalize | 70.6 | 63.8 | 10.94 | 11.00 |
  | no write | 54.5 | 49.9 | 9.56 | 9.10 |
  | no normalize, no write | 43.1 | 39.7 | 8.06 | 7.96 |
  | DM + sync (no compute) | 75.0 | 66.9 | 8.84 | 8.74 |
  | read + sync | 36.8 | 32.3 | 5.35 | 5.36 |
  | compute + sync (no read, no write) | 27.1 | 27.1 | 6.83 | 5.85 |
  | compute + write (no read) | 53.7 | 47.7 | 9.66 | 9.89 |
  | sync only | 6.5 | 6.5 | 4.12 | 2.17 |
- **Zones** (`zones_*_1_1.txt`): 256×3072 wb fp32 (S = 12, 96 cores, one tile row × 8 tiles per core), p50 from kernel start, µs:

  | stage | window | duration |
  |---|---|---|
  | all reads (chip-wide, DRAM-saturated) | 0.4–3.8 | |
  | dev pass after last chunk lands | 3.3–4.8 | |
  | publish | → 5.2 | |
  | exchange | 5.3–6.8 | `x_wait` max 2.5 |
  | combine | → 7.1 | |
  | normalize | 7.1–9.0 | |
  | store | 6.8–10.0 | final barrier 10.3; BRISC max 12.2 |
- **Ranked**:
  1. **Focus 2304×3072: DRAM-bound.** DM + sync (75.0) ≥ full (71.7), at ~0.99× the 72.1 µs COPY floor. There is no stage lever.
  2. **256×3072: a fully serial chip-wide chain.** Read wall (~3.4 µs DRAM), then the stats chain (dev tail 1.0, publish 0.35, exchange 1.5, combine 0.3; ~3.1 µs), then the write wall (~3.3 µs DRAM). Reads and writes never overlap anywhere on the chip.
  3. **2048×3072 runs ~4% behind byte-proportional.** The core_h imbalance is 3.2 rows, and NCRISC max/p50 is 2.3.

### Portfolio and verdicts
| # | idea | verdict | before → after (whole op, same-process alternating, median of 3) | domain |
|---|---|---|---|---|
| S1 | `read_stagger`: a cross-core read gate. Row groups run in 3 phases, top grid rows first. A phase-p reader waits on a semaphore that its phase p−1 same-slice partner bumps (NoC atomic) once its row has landed, so the early phases' chain and writes overlap the late phases' reads | **WIN in isolation, not graduatable as is** | Agent: 256 wb fp32 11853 → 10603 (−10.5%); 2-phase was flat, bottom-first +3.5%, 4-phase −7%. The coordinator's whole-op sweep (`ab_grad*.jsonl`, `ab_domain.jsonl`; a null A/B of two HEAD copies is within ±1%, `ab_null.jsonl`) was **non-monotone**: 64 / 96 / 128 / 224 / 320 ×3072 +9 / +21 / +10 / +9 / +6%, 256×1024 +11%, 256×2048 +8.5%, but 256 / 384 ×3072 −10 / −5%, 768×1024 −8% | one-row blockings, with no mechanism-based carve-out. Reverted, then re-opened as S1b |
| S1b | `read_stagger_v2`: the follow-up that found why S1 loses and derived its rule from that mechanism (below) | **WIN, graduated** | see "Whole op" below | every one-row blocking (core_h_tiles = 1) where the gated phase is DRAM-bandwidth bound. Elsewhere it compiles out to HEAD's path |
| S2 | `dev_tail`: cut "last chunk lands → publish". `bps` = butterfly publish (bit-exact), one-tile pivot pushes, and a software-pipelined dev loop | **NULL** | `bps` 11670 → 11510 (−1.4%), bit-exact; every guard cell flat. The dev pass is SFPU-pipe-bound at ~112 ns/tile (~96 instr/tile at ~1.6 cycles each); a load-macro accumulate looks blocked because the template writes to the loaded register. `pc1` (fp32 pivot = mean of the first landed tile) is −5.7% but **fails precision**: on elements that differ from HEAD, HEAD is closer to the float64 reference in 57–94% of cases | n/a |
| S3 | `exchange_mcast`: the row-group all-gather as multicasts (absolute slots, a non-loopback 1×S rectangle, a linked payload + flag mcast) instead of S−1 + S−1 unicasts | **REGRESSION** | v2 (linked payload + flag mcast) +16 to +21% at S = 12, +72% at S = 4: issue time drops 660 → 110 ns, but 24 concurrent row multicasts serialize at ~125 ns each (`x_wait` 0.53 → 3.0 µs). v1 +75%, v4 +17.6%, v3 (downstream linked mcast + unicasts) flat but +6.1% on 256 wb 16-bit. All variants bit-exact and `--dev` clean at S = 12/6/4/3/2/1 | none |

Breadcrumbs: `agent_logs/perf-coordinator_breadcrumbs.jsonl`. Artifact dirs: `perf_experiments/{r8_breakdown,read_stagger,read_stagger_v2,dev_tail,exchange_mcast}/`.

**S1b mechanism** (`read_stagger_v2/`, per-core zone timelines). S1 lost for two separate reasons:
1. **Gated groups on DRAM-endpoint rows stall in their exchange.** A gated group whose grid row hosts a **NoC1 DRAM write endpoint** (`get_optimal_dram_bank_to_logical_worker_assignment(device, NOC_1)`, logical rows {0, 2, 5, 8} here) runs its latency-critical exchange writes while the earlier phases' store burst converges on that row. Its `x_issue` then takes 2–4 µs instead of ~0.3 µs. This is what happened at 224×3072 on grid row 5 and 320×3072 on grid row 8. In HEAD every exchange finishes before any store starts, so it never happens there.
2. **Gating a latency-bound phase makes the read wall longer.** A gated row lands ~0.94 µs after release for 16 KB, ~1.3 µs for 32 KB. When the gated phase's bytes at the chip read rate take less time than that, the stagger adds a whole row-read latency (64–160×3072, 256×1024, 256×2048).

The rule S1b uses:
- **Phases**: two. Phase 0 is every endpoint-row group, then the top non-endpoint groups up to half of G. Phase 1 is the remaining (bottom) non-endpoint groups.
- **Bandwidth carve-out**: S1b stays off unless `phase1_bytes ≥ 400 GB/s × (580 ns + row_bytes / 45 GB/s)`. The constants are this board's measured read rate and single-row latency. The endpoint rows are queried from the device.

### What graduated
- **S1b** is the op's one path on one-row blockings: new `_stagger_phases` / `_dram_write_endpoint_rows` and `STAGGER_*` in the program descriptor, and the `stagger` CT arg plus a 4-word RT block in the reader.
- **Carve-outs**, each earned as stated:
  - *No meaning*: G = 1 (no other phase), or the gated phase is empty.
  - *Measured regression*: multi-row blockings, core_h_tiles > 1. Staggering them measured 2304×3072 +2.3 to +2.9% and 8192×256 +3.3%, since they already overlap later rows' reads with earlier rows' writes.
  - *Measured regression, mechanism-derived predicate*: a latency-bound gated phase (the bandwidth inequality above). The predicate is a function of (G, S, row bytes), not a list of shapes.
- **Nothing deleted**: S1b adds a phase and replaces no code. Where it is off, the reader is HEAD's code with 4 zero RT args.
- **Instrumentation**:
  - New zone `r_stagger_wait`, on the 48 gated cores at 256×3072.
  - Markers per (core, RISC) max 34 at 256 and 58 at 2304, unchanged.
  - The patch moved `r_load_reserve` to reader line 93, where its 16-bit profiler hash collided with `w_store_wait` (writer:223). The device profiler throws on that, so the kernel-head comment gained two lines (with a note to re-run `perf_experiments/dev_tail/hashcheck.py <op dir>` after moving a zone). `hashcheck` is clean.
- **Raw code**: the semaphore wait and `noc_semaphore_inc` are plain dataflow API. No helper bypassed.

### Whole op, before → after (`r8_breakdown/ab_final.jsonl`, HEAD copy vs graduated op, same process, alternating, median of 3; all bit-exact)
| cell | HEAD µs | after µs | Δ |
|---|---|---|---|
| **2304×3072 wb fp32 (focus)** | 71.81 | 71.65 | −0.2% (off) |
| 2304×3072 wb 16-bit / noaff fp32 | 70.98 / 71.88 | 71.20 / 71.28 | +0.3% / −0.8% (off) |
| 2048×3072 wb fp32 / wb 16-bit | 65.96 / 64.50 | 64.60 / 64.89 | −2.1% / +0.6% (off) |
| **256×3072 wb fp32 (roofline)** | **11.76** | **10.92** | **−7.2%** (n = 6 recheck: 11.93 → 10.92, −8.5%) |
| 256×3072 wb 16-bit / noaff fp32 / noaff 16-bit | 12.19 / 11.33 / 11.58 | 11.51 / 10.84 / 10.97 | −5.6% / −4.3% / −5.3% |
| 192 / 224 / 288 / 320 ×3072 wb fp32 | 10.57 / 11.41 / 12.42 / 13.08 | 9.67 / 10.44 / 11.81 / 12.45 | −8.5% / −8.5% / −4.9% / −4.8% |
| 160 / 384 / 512 ×3072 wb fp32 | 9.57 / 14.99 / 17.92 | 9.32 / 14.42 / 17.14 | −2.7% / −3.8% / −4.3% |
| 640×3072 wb fp32 | 21.20 | 21.67 | +2.2% (n = 6 recheck +0.8%: noise) |
| 64 / 96 / 128 ×3072 wb fp32 (off: latency-bound) | 7.90 / 8.00 / 8.81 | 7.95 / 8.01 / 8.83 | +0.6% / +0.2% / +0.2% |
| 768×1024 / 512×1024 / 256×1024 wb fp32 | 11.27 / 9.14 / 7.50 | 10.34 / 8.93 / 7.55 | −8.3% / −2.3% / +0.8% (off) |
| 256×6144 wb fp32 / 256×2048 wb fp32 | 18.54 / 11.16 | 17.39 / 11.06 | −6.2% / −0.9% (off) |
| 1024×1024 noaff 16-bit (S 2, G 32) | 12.30 | 11.76 | −4.4% |
| 1024×3072 wb fp32 / 8192×256 wb fp32 | 33.36 / 22.44 | 33.57 / 22.61 | +0.6% / +0.7% (off) |
| 4096×2048 wb 16-bit / 3072×6144 wb fp32 | 85.97 / 193.6 | 86.20 / 191.9 | +0.3% / −0.9% (off) |
| 8192×3072 wb fp32 / 23040×1024 wb 16-bit | 259.6 / 246.1 | 254.8 / 246.0 | −1.9% / −0.1% (off) |

- **Guard-set no-regression: 0 of 32 cells slower beyond noise.** The largest increase is 640×3072 at +2.2% (n = 3); the n = 6 recheck is +0.8%.
- **Precision**: bit-exact against HEAD on every cell, since only the read issue order changed.
- **After-zones** (`zones_256x3072_1_1_after.txt`): BRISC span max 12.2 → 11.2 µs, and the 48 gated cores wait 2.1 µs p50 in `r_stagger_wait`. The focus (`zones_2304x3072_1_1_after.txt`) is unchanged, with BRISC max 71.7 µs.
- **Tests**:
  - Golden: 52 passed, 49 skipped, in both normal and `--dev` mode.
  - `--dev` on 10 stagger-on cells (256×3072 in all four modes, 224, 320, 384, 512 16-bit, 768×1024, 1024×1024 noaff 16-bit, 256×6144): 2 reps each, bit-exact against HEAD, no hang (`r8_breakdown/dev_grad.jsonl`). The agent also ran 13 more.
  - One unexplained non-dev failure was seen once with S1's prototype, never reproduced, and not seen with S1b.

### Leads for the next round
- **256×3072**: the remaining chain is the dev pass, which is SFPU-pipe-bound at ~112 ns/tile, plus the exchange and the DRAM write wall. A robust early pivot that passes precision would be worth ~6% (`dev_tail` `pc1`).
- **S1b constants**: these are board-measured. A 3-phase variant with endpoint rows kept out of the last phase (`read_stagger_v2` `v_t3x`) was better on S = 12, G = 8–10 (256 −10.7%, 288 −10.6%) but worse when endpoint rows fall in the middle phase. It was not run on the full sweep.
- **Focus**: still at the DRAM COPY floor.

### Helper bypasses — none
Nothing graduated this round bypasses a helper. The stagger handshake is plain dataflow semaphore API with no helper equivalent. A point-to-point one-shot release between two cores is not a gap in any helper, because no helper covers it; the existing raw rows (exchange, affine column mcast, SFPU stats) are unchanged by this round.

## Perf 3 (9th tournament)
- **Date**: 2026-11-17. Round 3 of 4 of the series that started with "Perf 1 (7th tournament)". Coordinator + 3 `perf-part-optimizer`s in parallel. Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. Precision contract the same in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off.
- **Focus**: `LOOSE_CASES` is empty, so no case is perf-flagged. Shapes were chosen by measured headroom (below). Only cells > 32 tiles were measured, and ~2–3% is noise. Precompile still fails here, so every run used `--no-precompile`.

### Measured breakdown (before; HEAD 7a20117a7cd; `perf_experiments/r9_breakdown/`)
- **Headline** (`baseline.jsonl`, one fresh run, µs): 2304×3072 wb fp32 71.55; 2048×3072 wb fp32 67.81; 256×3072 wb fp32 10.99, noaff fp32 10.63, wb 16-bit 11.55; 512×3072 17.05; 640×3072 21.77; 1024×3072 33.18; 128×3072 8.75; 256×1024 7.56; 1024×1024 noaff 16-bit 11.60; 8192×256 22.50.
- **Cumulative ablation** (`ablations.jsonl`, r8's `ablation_gates.patch`, applied temporarily and then reverted), all wb fp32, µs:

  | variant | 256×3072 | 512×3072 | 1024×3072 | 2304×3072 |
  |---|---|---|---|---|
  | full | 10.87 | 17.02 | 33.36 | 71.86 |
  | no normalize | 10.07 | 17.69 | 33.62 | 72.12 |
  | no write | 10.03 | 15.92 | 26.85 | 54.47 |
  | no normalize, no write | 8.28 | 13.42 | 21.00 | 42.57 |
  | DM + sync (no compute) | 8.93 | 17.26 | 33.77 | 73.01 |
  | read + sync | 5.27 | 9.54 | 17.82 | 36.76 |
  | write + sync | 8.54 | 13.54 | 22.52 | 45.47 |
  | compute + write (no read) | 9.82 | 15.26 | 24.12 | 53.04 |
  | sync only | 4.07 | 4.91 | 5.41 | 6.51 |

- **Roofline gate**: the measured DRAM COPY floor (`dram_ceiling`: 28.31 MB in 72.1 µs ≈ 393 GB/s). Against it, 2304×3072 is at 1.00×, 1024×3072 at 1.04× and 512×3072 at 1.07×; all three are DRAM-bound (DM + sync ≥ full), so there is no stage lever. 256×3072 is at 1.37×.
- **Zones** (`zones_256x3072_1_1.txt`, `phase_zones.py` for the split by phase): the wall is set by the 48 gated (phase-1) cores. Their median post-read chain, in µs from chip start: row lands at 5.23, then dev tail +1.05, publish +0.36, exchange issue + flags +0.68, `x_wait` (peer skew) +0.85, combine +0.22, normalize 1.87 (the store follows it), final barrier +0.46. The wall is ≈ 11.5 µs; ungated cores finish around 9.3.
- **Ranked**: (1) the one-row (core_h = 1) blockings' gated-phase chain, plus the work split that decides which cells get a one-row blocking; (2) nothing else: the multi-row cells are at the DRAM floor. Dev-chunk streaming was not re-floated: `STATS_CHUNK = 4` already streams it, and `dev_tail` `ck1` was flat. Normalize cost was not re-floated either: the 7th tournament's R2 was null, after four earlier normalize rounds.

### Portfolio and verdicts
| # | idea | verdict | before → after (whole op, same-process alternating, median of 3) | domain |
|---|---|---|---|---|
| T1 | `stagger_3phase`: three top-first stagger phases, chosen by one rule. Menu also covered group-synchronous release (`v2qs` full quorum, `v2qh` half quorum) and earlier release points (`v2r0` once reads are issued, `v2r1` once half the row has landed) | **WIN on the domain, NULL on 256×3072**. Recommended rule `g3`: 3 phases only where every phase holds ≥ m + 1 groups (m = the bandwidth-bound minimum) and no endpoint row is in the last phase; otherwise HEAD's 2-phase rule. The "+1 group" margin is fitted, not derived | Agent: 288×3072 wb fp32 11699 → 10969 (−6.2%), 320 −3.3%, 512 −3.5%, 640 −2.4%, 1024×1024 noaff 16-bit −3.0% (recheck −5.9%), 320 wb 16-bit −6.6%. 256×3072 keeps 2 phases. Group release +3.7%; release once issued +15.7%; release once half landed +6.4%: `x_wait` skew comes from DRAM contention, not release skew. Bit-exact | every one-row blocking. Same handshake as HEAD; the reader change is comment only |
| T2 | `exchange_split`: the idle reader issues payload writes + flush + flags for the even peer positions on its own NoC | **NULL** | Exchange issue + flags 0.66 → 0.41 µs, and publish → combine −0.22 µs, but 256×3072 wb fp32 10947 → 10909 (flat): the slowest core's store tail sets the wall. 256 noaff fp32 −3.1% (n = 3, separated); every other cell flat (2304 +0.2% on recheck). Bit-exact, `--dev` clean at S = 12/6/4/3/2 | not graduated: a second-RISC protocol (2 handoff semaphores, 16 of 16 at S = 12 wb + stagger) for one borderline cell |
| T3 | `worksplit_restagger`: forced-S sweep under the staggered kernels, 59 cells; refit `select_blocking` | **WIN** | Cost gets one stagger term, `− 0.5·bw` when core_h = 1, the stagger applies, and bw ≤ 16. 28 of 1388 offline picks change, and all 28 were A/B'd at 0.3–9.0% faster: 1024×1536 −8.9%, 992×1024 −8.8%, 1024×1024 wb −7.4%, 2048×512 −4.6%, 288×3072 noaff 16-bit −4.5%. Summed regret over 59 cells 73.4% → 23.2%; cells with > 3% regret 11 → 2. 16 guards flat | applies everywhere. Carve-out bw ≤ 16 is a **measured regression**: taking the one-row split with 32+-tile rows measured 2304×1024 noaff 16-bit +4.5%, 2048×2048 noaff fp32 +6.4%, 640×6144 wb fp32 +8.3% |

Breadcrumbs: `agent_logs/perf-coordinator_breadcrumbs.jsonl`. Artifact dirs: `perf_experiments/{r9_breakdown,stagger_3phase,exchange_split,worksplit_restagger}/`, each with its `graduate.patch`, packages and logs.

### Aggregation and what graduated
T1 and T3 do not conflict, and both are applied. T3 moves `_stagger_phases` out of the program descriptor into `layer_norm_adaln.py` as `stagger_phases`, and the descriptor passes the device-queried endpoint rows to `select_blocking(stagger_ep_rows=)`. T1's `g3` body replaces that function's body. The cost model's "stagger applies" term and the kernels' phase plan therefore come from **one predicate**.
- **Deleted**: the descriptor's private `_stagger_phases` and the `STAGGER_*` constants there (moved, not duplicated), and the descriptor's two-phase-only RT loop. It is replaced by a loop over consecutive phase pairs; a middle-phase core is both gated and a releaser.
- **Carve-outs**, each earned as stated:
  - `bw ≤ 16` on the stagger cost term. Measured regression: +4.5..+8.3% on the 32/64-tile-row cells above.
  - The 3-phase plan falls back to 2 phases unless every phase holds ≥ m + 1 groups and no endpoint row is in the last phase. Measured regression: 192×3072, with three 2-group phases, +3.7..5%. The endpoint-row condition is mechanism-derived, from the 8th tournament's S1b analysis.
  - The existing 2-phase bandwidth carve-out is unchanged.
- **Interaction measured** (`r9_breakdown/merged_picks.py`, offline over 1388 cells): 518 cells take T1's config, 8 take T3's, and **20 are new combinations** (T3's re-picked one-row wb splits now get 3 phases). All 20 were measured three ways (`ab_new.jsonl`; HEAD / T3-only / merged):
  - **Merged vs HEAD**: faster on all 20, −1.5 to −12.7%: 1024×1536 wb 16-bit −12.7%, 1536×1024 16-bit −11.1%, 1280×1536 fp32 −10.8%, 992×1536 16-bit −10.0%, 2048×512 16-bit −6.6%, 992×1024 fp32 −3.6%.
  - **Merged vs T3-only**: mixed, −7.7% to +4.8%. The n = 6 recheck (`ab_new_recheck.jsonl`): 992×1024 wb fp32 **+4.8%** (separated distributions), 992×1024 wb 16-bit +2.9%, 1024×1024 wb fp32 +2.5%, 1024×1024 wb 16-bit +0.6%, 2304×512 wb fp32 +0.9%.
  - **Not carved out.** The merged path is still 3.6% faster than HEAD there. No byte- or S-derived predicate separates the case: 1024×1024 noaff 16-bit, also S = 2, at a similar per-phase byte margin (1.25× vs 1.22× the minimum), wins with 3 phases. A guard would only fence the one shape measured. Recorded as a lead.
- **Instrumentation**: no new kernel path. Middle-phase cores use the existing `r_stagger_wait` zone (288×3072 after-zones: 72 gated cores across 2 gated phases). Markers per (core, RISC) max 34, and every RISC's zones reach its kernel end. `hashcheck` is ok (T1's reader head comment keeps its line count, so no zone moved).
- **Raw code**: none new. No helper bypassed.

### Whole op, before → after (`r9_breakdown/ab_guard.jsonl`, HEAD copy vs graduated op, same process, alternating, median of 3; µs)
| cell | HEAD | after | Δ | config change |
|---|---|---|---|---|
| 2304×3072 wb fp32 / wb 16-bit / noaff fp32 | 70.89 / 71.18 / 70.93 | 71.96 / 71.94 / 70.52 | +1.5 / +1.1 / −0.6% | none (noise) |
| 2048×3072 wb fp32 | 65.24 | 65.97 | +1.1% | none (noise) |
| 256×3072 wb fp32 / wb 16-bit / noaff fp32 / noaff 16-bit | 10.90 / 11.53 / 10.66 / 10.80 | 10.70 / 11.52 / 10.57 / 10.49 | −1.8 / −0.1 / −0.8 / −2.9% | noaff 16-bit S 12 → 6 |
| 128 / 192 / 224 ×3072 wb fp32 | 8.73 / 9.63 / 10.26 | 8.72 / 9.52 / 10.26 | 0.0 / −1.2 / 0.0% | none |
| **288×3072 wb fp32** | **11.66** | **11.20** | **−3.9%** | 3 phases |
| **320×3072 wb fp32 / noaff 16-bit** | **12.44 / 12.54** | **12.04 / 11.83** | **−3.2 / −5.6%** | 3 phases / S 12 → 6 |
| 384 / 512 / 640 ×3072 wb fp32 | 14.26 / 17.28 / 21.47 | 14.28 / 17.24 / 21.09 | +0.1 / −0.2 / −1.8% | 512, 640: 3 phases |
| 1024×3072 wb fp32 | 32.89 | 33.25 | +1.1% | none (noise) |
| 256×1024 / 768×1024 wb fp32 | 7.47 / 10.00 | 7.42 / 9.90 | −0.8 / −1.0% | 768: 3 phases |
| **1024×1024 noaff 16-bit** | **11.56** | **11.29** | **−2.4%** | 3 phases |
| 256×6144 wb fp32 / 8192×256 wb fp32 | 17.70 / 22.48 | 17.66 / 22.45 | −0.2 / −0.2% | none |
| 4096×2048 wb 16-bit / 3072×6144 wb fp32 / 8192×3072 wb fp32 | 86.42 / 195.6 / 256.5 | 86.67 / 193.3 / 255.8 | +0.3 / −1.2 / −0.3% | none |
| 20 combined cells (512 / 1024 / 1536 wide, H 992–3072, wb) | | | **−1.5 .. −12.7%** | see above |

- **Guard-set no-regression: 0 of 46 cells slower beyond noise.** The largest increase is 2304×3072 wb fp32 at +1.5%, a cell whose program is byte-identical to HEAD.
- **No change on the roofline cells**: the measured cells of `roofline_targets.md` (256 / 2048 / 2304 ×3072 wb fp32) keep HEAD's config and stay within noise. The gains are on one-row cells that are neither the roofline cells nor at the DRAM floor.
- **Precision**: cells with an unchanged S are bit-exact, since a phase change only reorders reads. Cells with a new S differ by summation order only: rel-RMS within 5e-5 of HEAD (guard set), and within 1e-4 on T3's sweep. PCC ≥ 0.99995.
- **Tests**:
  - Golden: 52 passed, 49 skipped.
  - `--dev` (watcher, NoC sanitizer, LLK asserts; `r9_breakdown/test_dev.py`, two calls per cell, call-to-call bit-stable): 14 cells pass. They cover 3 phases at S = 1 (2048×512, 3072×512), S = 2 (992 / 1024 ×1024, 1536×1024), S = 3 (1024 / 1280 ×1536), S = 6 (512 / 640 ×3072) and S = 12 (288×3072); the S 12 → 6 noaff 16-bit cells; and 256×3072. The agents ran another 10 (T1) and 4 (T3).
  - `tests/.../test_layer_norm_adaln.py::test_select_blocking_pinned`: 4 failed / 4 passed both at HEAD and after this round. The pinned S values predate earlier refits, so these failures are stale expectations this round did not change.

### Leads for the next round
- **3 vs 2 phases at S = 2 with affine** (992×1024 wb fp32 +4.8% vs 2-phase; noaff 16-bit at a similar margin wins): find the mechanism (zones per phase on 992×1024 wb vs noaff), then replace the fitted "+1 group" margin with a derived one.
- **T3's two open cells** (352 / 512 ×6144 wb): the bw = 32 one-row S6 split is 8.8% / 4.5% faster there, but the bw ≤ 16 cutoff excludes it, because 640×6144 regresses. Needs a mechanism, not a cutoff.
- **256×3072** is unchanged at ~10.7–10.9 µs (1.37× the COPY floor). Its wall is the gated cores' store tail. T2 showed that shortening the exchange alone does not move it. The remaining unbuilt route is still the 7th tournament's R4 lead: reader-side stores, gated on a chip-wide "all reads done" count.

### Helper bypasses — none
Nothing graduated this round bypasses a helper. Both changes are host-side (work-split cost term, stagger phase plan) plus a reader comment. The stagger handshake is HEAD's plain dataflow semaphore wait / NoC atomic increment, unchanged.

## Perf 4 (10th tournament)
- **Date**: 2026-11-17. Round 4 of 4 of the series that started with "Perf 1 (7th tournament)". Coordinator + 5 `perf-part-optimizer`s in parallel. Device: Blackhole, 12×10 grid, 1350 MHz. `SUPPORTED` unchanged. The precision contract was the same in every measurement: bf16, HiFi4, `fp32_dest_acc_en` as given, approx off, DEST sync mode unchanged.
- **Focus**: `LOOSE_CASES` is empty, so no case is perf-flagged. Shapes were chosen by measured headroom. Only cells > 32 tiles were measured, and ~2–3% is noise. Every run used `--no-precompile`.

### Measured breakdown (before; HEAD ad06bc7a7ab; `perf_experiments/r10_breakdown/`)
- **Headline** (`baseline.jsonl`, one fresh run, µs, all fp32 unless noted): 2304×3072 wb 71.32; 2048×3072 wb 65.64; 4096×2048 wb 16-bit 85.65; 1024×3072 32.96; 512×3072 16.55; 8192×256 22.76; 640×6144 42.71; 512×6144 34.86; 352×6144 25.75; 256×3072 wb 10.87 / noaff 10.69 / wb 16-bit 11.48; 288×3072 10.99; 128×3072 8.81; 256×1024 7.51; **992×1024 wb 13.88**; 1024×1024 noaff 16-bit 11.48.
- **Against the DRAM COPY floor** (~393 GB/s): every multi-row cell is at 1.00–1.07×, so there is no lever there. The headroom is in the low-H one-row cells: 256×3072 1.36×, 992×1024 1.35×, 288×3072 1.22×, 128×3072 2.2×.
- **Cumulative ablation** (`ablations.jsonl`, r8's `ablation_gates.patch`, applied temporarily and then reverted), wb fp32, µs:

  | variant | 256×3072 | 992×1024 | 352×6144 | 128×3072 |
  |---|---|---|---|---|
  | full | 11.02 | 13.74 | 25.64 | 8.74 |
  | no normalize | 9.81 | **11.25** | 23.92 | 7.36 |
  | no write | 10.04 | 13.70 | 23.73 | 8.48 |
  | no normalize, no write | 8.26 | 10.29 | 18.10 | 6.76 |
  | DM + sync (no compute) | 9.14 | 11.83 | 23.17 | 6.52 |
  | read + sync | 5.23 | 7.20 | 12.88 | 4.29 |
  | write + sync | 8.24 | 9.48 | 17.37 | 6.08 |
  | compute + write (no read) | 9.96 | 11.99 | 19.20 | 7.43 |
  | sync only | 4.07 | 4.59 | 6.50 | 4.10 |
- **Zones** (`zones_*_1_1.txt`): math-thread `c_normalize` takes 1.86 µs at 8 tiles per core (256×3072) and 3.21 µs at 16 tiles per core (992×1024, S = 2, 62 of 120 cores). It runs serially after combine. At 992×1024, `c_wbp_prep` ends after the exchange.
- **Ranked**:
  1. Normalize on one-row cells. Removing it alone saves 18% at 992×1024, 16% at 128×3072 and 11% at 256×3072.
  2. Per-core serial work on cells that leave the grid idle. 992×1024 runs on 62 cores because S had to divide both 12 and Wt.
  3. The fitted "+1 group" margin of the 3-phase stagger, open since Perf 3 (992×1024 wb is +4.8% with 3 phases).

### Portfolio and verdicts
| # | idea | verdict | before → after | domain |
|---|---|---|---|---|
| A | `normalize_sfpu_mac`: fp32-DEST normalize without the HiFi4 FPU multiply-accumulate. x goes into DEST, then SFPU load-macro multiply-adds; also tried per-strip algebraic fusion | **NULL at HEAD precision** | Bit-exact variants: v3 256×3072 noaff 10865 → 10703 (−1.5%), v2/v6a flat to slower (stage 992×1024 3203 → 3869 for v6a). The faster option **c1** (1-phase FPU x − p, then SFPU (D·rstd + c″)[·w + b]) is −2 to −4% (992×1024 wb −2.6% med3, 256 noaff −3.5%, 2048×3072 −2.8%), but at offset 0 its rel-RMS is 0.0017031 → 0.0017132 and only 36–41% of changed outputs are closer to float64. At offset 1000, 78–94% are closer. Recorded as a precision-trading option, **not graduated**. Fusion A = rstd·w, B = c′·w + b costs what it saves | n/a |
| B | `normalize_window_fixed`: cut normalize's fixed per-row / per-window cost. The c′ build moves into the first window acquire and covers only its 8 strips; a redundant reconfig and init are dropped; the row-stats wait moves to first use; 16-bit path reordered with no S-face zeroing | **WIN** | Stage: 256×3072 wb fp32 1864 → 1598 ns, 992×1024 3244 → 2904, wb 16-bit 1989 → 1591. Whole op (agent, median of 3): −2.6 to −3.9% on 7 focus cells, 256×1024 −5.5%, 1024×64 −6.3%, multi-row −0.4 to −1.9%. **Bit-identical** on 29 cells | everywhere, no exceptions |
| C | `prestats_affine_hoist`: precompute x·w in the exchange idle window | **NULL** (upper bound) | Deliberately wrong-output bound with no post-stats affine (ub_mac): 256×3072 −2.1% (noise). 992×1024 has no idle window (250 ns). y = (x·w)·rstd + (b + c′·w) keeps the operation count and adds cancellation (~2⁻¹¹·\|mean\|/σ) | n/a |
| D | `wide_groups`: use more of the grid on low-H cells. Built: **ragged slices**, so S need not divide Wt; the first Wt % S slices are one tile wider. Not built: non-row-aligned groups (measured bound 0) and S > grid_x (stand-in bound ≤ 7%) | **WIN** | Agent, median of 3: 992×1024 wb fp32 −13.4%, wb 16-bit −12.6%, 1024×1024 wb −12.9%, 256×4096 wb −25.3%, 256×2048 wb −15.8%, 8192×1024 wb −6.8%. Unchanged picks bit-exact | everywhere `select_blocking` picks a ragged S; no carve-out |
| E | `stagger_s2_margin`: the mechanism behind 3 vs 2 phases at S = 2 wb. A third phase delays the last phase's release by one more read wall (~1 µs). It only pays back when that phase is **store-bound**; with affine at 16-tile rows, normalize paces the output below DRAM bandwidth. New condition: `cores × 2048 / NORMALIZE_TILE_NS > 400 B/ns` (measured per-tile normalize ns). The "+1 group" margin stays: dropping it (sb_m0) measured +4.2 to +5.2% on 2304×256 / 576×1024 / 384×1536 | **WIN** | 22 of 1388 picks go from 3 to 2 phases. All 22 A/B'd: −1.2 to −7.5% (992×1024 wb fp32 −4.6%, 384×8192 wb −7.5%, 512×8192 −5.7%), 2048×512 +0.5% flat. Bit-exact | every one-row staggered blocking |

Breadcrumbs: `agent_logs/perf-coordinator_breadcrumbs.jsonl`. Artifact dirs: `perf_experiments/{r10_breakdown,normalize_sfpu_mac,normalize_window_fixed,prestats_affine_hoist,wide_groups,stagger_s2_margin}/`. Profiler CSVs over 500 KB are gzipped.

### Aggregation and what graduated
B, D and E do not conflict. B rewrites the normalize window, D the combine fold and the work split, and E the phase predicate. All three patches applied cleanly in order (B, D, E). A's c1 would have overlapped B's window body; it was not graduated on precision grounds.
- **B**: the op's one normalize path in both DEST modes. Deleted: the separate c′-build acquire and the 32-vector `nz_cprime`, replaced by `nz_cprime_strips`.
- **D**: `select_blocking` now accepts every S that divides grid_x and keeps ≥ `MIN_SLICE_TILES` per slice (new `slice_widths`); the cost model and footprint use the widest slice.
  - The descriptor compiles reader/writer/compute once per slice-width class. A grid column always holds one class.
  - The input and output CBs get one descriptor per class and are allocated last, so `cb_gathered_stats` and the affine landing CBs keep one address on every core. The first build split rows across a CB wrap on the narrow cores: PCC 0.88 at 8192×1024, found and fixed by the agent.
  - The combine weights each slot by n_j (`cf_fold_fp32_w`, `cf_fold_b16_w`, runtime per-slot counts). A CT flag keeps the even-slice path bit-identical. The old "S must divide Wt" restriction is deleted, not guarded.
- **E**: `stagger_phases(…, has_affine, fp32_dest_acc_en)` gains the store-bound condition. The cost model and the descriptor still share this one predicate.
- **Carve-outs**: none new. E keeps the existing "+1 group" margin, now backed by a measured regression (sb_m0, above), not just a fit.
- **Instrumentation**: no new zones were needed (the ragged path runs the same zoned stages). After-zones `zones_992x1024_1_1_after.txt`: 93 cores, max 34 markers per (core, RISC), zones reach kernel end; `c_normalize` math 3.21 → 2.05 µs. 256×3072: 1.86 → 1.56 µs. `hashcheck` ok.

### Whole op, before → after (`r10_breakdown/ab_merged.jsonl`; HEAD package copy vs merged op, same process, alternating, median of 3; µs)
| cell | HEAD | after | Δ | what changed |
|---|---|---|---|---|
| **992×1024 wb fp32 / wb 16-bit** | 13.74 / 13.76 | 11.63 / 12.11 | **−15.4 / −12.0%** | S 2 → 3 ragged, B |
| 992×1024 noaff fp32 / noaff 16-bit | 11.54 / 11.15 | 11.06 / 10.93 | −4.2 / −2.0% | S 2 → 3, B |
| 1024×1024 wb fp32 | 13.81 | 11.62 | −15.8% | ragged, B |
| **256×4096 wb fp32 / noaff fp32** | 18.81 / 14.29 | 13.57 / 13.18 | **−27.8 / −7.8%** | ragged |
| 256×2048 wb fp32 / noaff 16-bit | 11.31 / 9.26 | 9.25 / 8.64 | −18.3 / −6.7% | ragged |
| 512 / 1024 / 992 ×2048 wb fp32 | 14.12 / 25.91 / 25.24 | 12.27 / 24.05 / 23.34 | −13.1 / −7.2 / −7.5% | ragged |
| 1024×2048 noaff 16-bit | 22.06 | 21.89 | −0.8% | ragged (flat) |
| 8192×1024 wb fp32 / wb 16-bit | 93.50 / 90.74 | 89.02 / 87.96 | −4.8 / −3.1% | ragged |
| 2048×2048 wb fp32 | 46.08 | 43.84 | −4.9% | ragged |
| 480×2048 / 480×4096 / 512×4096 wb fp32 | 13.76 / 23.54 / 23.75 | 11.61 / 22.25 / 22.34 | −15.6 / −5.5 / −5.9% | E (2 phases) + D/B |
| 384×8192 wb fp32 / wb 16-bit / 512×8192 wb fp32 | 41.57 / 39.68 / 47.12 | 37.37 / 35.08 / 44.87 | −10.1 / −11.6 / −4.8% | E + B |
| 2048×512 wb fp32 | 13.66 | 12.87 | −5.8% | E, B |
| 256×3072 wb fp32 / wb 16-bit / noaff fp32 / noaff 16-bit | 10.90 / 11.71 / 10.53 / 10.74 | 10.68 / 11.16 / 10.43 / 10.49 | −2.1 / −4.7 / −0.9 / −2.4% | B only |
| 128×3072 / 256×1024 / 32×3072 / 1024×64 wb fp32 | 8.83 / 7.56 / 7.76 / 4.62 | 8.36 / 7.28 / 7.34 / 4.35 | −5.3 / −3.7 / −5.4 / −5.7% | B only |
| 288×3072 wb fp32 | 11.34 | 10.90 | −3.9% | B only |
| 1024×1024 noaff 16-bit | 11.38 | 11.64 | +2.3% → **n = 6 recheck −0.5%** | ragged S 2 → 3 |
| 352×6144 / 640×6144 wb fp32 | 25.31 / 42.27 | 25.17 / 42.81 | −0.5 / +1.3% (n = 6: 0.0%) | B only |
| 1024×3072 / 2048×3072 wb fp32 | 33.77 / 65.48 | 33.38 / 64.97 | −1.1 / −0.8% | B only |
| 2304×3072 wb fp32 / wb 16-bit / noaff fp32 | 72.06 / 71.05 / 70.61 | 72.03 / 71.60 / 70.91 | 0.0 / +0.8% (n = 6 −0.4%) / +0.4% | B only (DRAM floor) |
| 8192×256 wb / 4096×2048 wb 16-bit / 3072×6144 wb | 22.89 / 86.67 / 193.3 | 22.83 / 85.31 / 192.3 | −0.3 / −1.6 / −0.5% | B only |

- **Guard-set no-regression: 0 of 43 cells slower beyond noise.** The largest increase is 1024×1024 noaff 16-bit at +2.3% with n = 3; the n = 6 recheck is −0.5% (`ab_recheck.jsonl`).
- **Precision**: unchanged-S cells are bit-exact against HEAD. Ragged-S cells differ only by summation order; their rel-RMS vs float64 is equal or better (16-bit: 0.00352 → 0.00345 on 992×1024 noaff, 0.00342 → 0.00331 on 256×2048). Max ratio of op to HEAD rel-RMS over the 43 cells is 1.0008, and PCC ≥ 0.99987 everywhere.
- **Tests**:
  - Golden: 52 passed, 49 skipped.
  - `--dev` (`r10_breakdown/test_dev.py`, profiler off, two calls per cell, call-to-call bit-stable): 12 of 12 pass, covering ragged S in all four modes, E's 2-phase cells, 256×3072 wb fp32 and 16-bit, 1024×64 and 2304×3072. The agents ran 8 (B) and 5 (D) more.
  - `test_select_blocking_pinned`: 4 failed / 4 passed, the same stale expectations as at HEAD.
- **Not tested**: ragged S with batch > 1 and on the multi-block fallback path. Both are sized by the per-class CBs but neither ran, and the golden shapes never pick a ragged S.

### Leads
- E's `NORMALIZE_TILE_NS` were measured before B cut normalize by ~10%. Re-measure them on the new kernels; a few cells may flip to 3 phases.
- **128×3072 / 256×3072** still leave cores idle: S ≤ 12 and MIN_SLICE_TILES = 8. D's stand-in suggests 4-tile slices are cheap; re-fitting MIN_SLICE_TILES or groups spanning two grid rows (bound ≤ 7%) are the remaining routes.
- A's c1 (−2 to −4%) needs x − p to be exact in a 1-phase FPU subtract, e.g. a pivot rounded to a coarse grid. That is a stats-side change.
- Pack release without DEST clear could keep the S slot resident per DEST half (noaff only, ~90 ns/window; A's untested lead).

### Helper bypasses — none
Nothing graduated this round adds a helper bypass. B's edits stay inside the existing raw-LLK normalize path, whose justification is at the head of `kernels/layer_norm_adaln_compute.cpp` (a `normalize_window_fixed` paragraph was added there). D's `cf_fold_fp32_w` and `cf_fold_b16_w` are the existing raw combine folds (same justification: a runtime-count fold over CB tiles is not a chain element) with a runtime per-slot element count. E is host-only.
