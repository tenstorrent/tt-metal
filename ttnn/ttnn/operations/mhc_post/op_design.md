# Operation Design: mhc_post

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (eltwise mix + row reduction / projection + per-token SFPU scalar step + optional fabric all-to-all: a fused CCL op) |
| Goal | One device program per call (per chip). It folds the sublayer output F into the n=4 residual streams X and mixes them into X'. On request, the same pass also produces the next mHC step's stats row, its hc coefficients (Sinkhorn), and its collapsed input y. y is optionally RMS-normalized, and optionally delivered column-split to the TP group over the fabric. |
| Math | `X'_j = post_j·F + Σ_i comb[i][j]·X_i` (fp32 arithmetic, rounded once to X.dtype). Stats: `stats[:,0:24] = X'@W`, `stats[:,24] = Σ Xmix²`. hc: DeepSeek-V4 hc step on the stats row. `y = Σ_i P_i·X'_i` (bf16, rounded once). Norm: `y = bf16((y_b·rsqrt(Σy_b²/C+eps))·w)` and `stats[:,25] = Σ y_b²`. Feed: chip p's `D[q·T:(q+1)·T, :]` = chip q's y, column block p. |
| Mode | Hybrid. Dataflow and compute are written from scratch on the helper library. The hc step is a custom SFPU block operation. The feed transport is taken from the fabric examples. |
| References | `models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py` (ground truth); `.../reference/mhc/deepseek_sinkhorn_kernel.py` (hc order of operations); `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py` (composite being replaced); `eval/golden_tests/mhc_post/{feature_spec,helpers}.py`; `ttnn/ttnn/operations/examples/master.md` + `fabric_all_gather/`, `fabric_ring_exchange/`, `fabric_reduce_scatter/`; `.claude/references/generic_op_template/`; `ttnn/cpp/ttnn/kernel_lib/` |

Notation used below:
- n = 4 streams. NG = n(n+2) = 24 coefficients. C is the stream width.
- `tensor_c_tiles` = C/32.
- `tensor_t_tiles` = Σ over images of ceil(T_img/32). All leading dims fold into T, so this is `ceil(T_total/32)` where `T_total` = product of the leading dims × T.
- A **unit** is one token-tile row × one C-column tile. A unit reads 1 F tile and n X tiles, and writes n X' tiles (plus 1 y tile).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| compute_kernel_config | ComputeConfigDescriptor | no | `fp32_dest_acc_en` must be True, else UnsupportedAxisValue | `default_compute_kernel_config()` (HiFi4, fp32 DEST, approx off) | CT (math_fidelity, approx) |
| post_col | int | no | `0 ≤ post_col`, `post_col + n ≤ 32` | 0 | RT (reader) |
| comb_col | int | no | `0 ≤ comb_col`, `comb_col + n² ≤ 32` | 0 | RT (reader) |
| stats_weight W | Tensor | no | (1..,n·C,32) fp32 TILE DRAM | None | RT addr |
| collapse_pre P | Tensor | no | (...,T,n..32) fp32 TILE; needs W | None | RT addr |
| hc_scale | list[3] float | when collapse | len 3 | — | RT (root compute, bit-cast fp32) |
| hc_base | list[24] float | when collapse | len NG | — | RT (root compute) |
| hc_norm_width | float | no | > 0 | n·C | RT |
| hc_norm_eps, hc_eps | float | no | ≥ 0 | 1e-6 | RT |
| hc_sinkhorn_iters | int | no | ≥ 1 | 20 | RT (loop trip count) |
| output_tensor / stats_output / hc_output / y_output | Tensor | no | same shape, dtype, layout and memory config as the tensor the op would allocate, and not aliasing any input; anything else is refused | None | RT addr |
| max_grid | CoreCoord | no | 1..device grid | device compute grid | host (core set) |
| norm_weight w | Tensor | no | (1..,1,C) fp32 TILE; collapse only | None | RT addr |
| norm_eps | float | no | ≥ 0 | 1e-6 | RT |
| feed_output D | Tensor | no | per chip (...,G·T,C/G) bf16 TILE DRAM interleaved; collapse only | None | RT addr |
| feed_semaphores | list[2] int | with feed | L1 addresses | None | RT |
| feed_num_links | int | no | {1,2} | 1 | host (port count) |
| feed_cluster_axis | int | no | {0,1} | 1 | host |

The entry point canonicalizes and checks the arguments in this order:
1. `validate()` is the first line.
2. A clear `ValueError` is raised for: collapse_pre without stats_weight; hc_scale or hc_base missing or of the wrong length; norm_weight or feed_output without collapse_pre; a feed with missing semaphores, G < 2, or `C % (32·G) != 0`; a feed D of the wrong shape, dtype, layout or memory config; caller buffers that mismatch or alias an input.
3. Mode is derived: `collapse` if P is given, `stats` if only W is given, else `plain`.
4. `coef_source` is `packed` iff post and comb share a buffer address.

## Tensors

### Input

| Tensor | Shape | Dtype | Layout | Memory |
|--------|-------|-------|--------|--------|
| F | (..., T, C), rank 2–4 | fp32 / bf16 (`sublayer_dtype`) | TILE | DRAM interleaved |
| X | (..., T, n·C) | fp32 / bf16 (`dtype`) | TILE | DRAM interleaved |
| post / comb | (..., T, ≥post_col+n) / (..., T, ≥comb_col+n²), or one packed (..., T, NG) | fp32 | TILE | DRAM interleaved |
| W | (1.., n·C, 32) | fp32 | TILE | DRAM interleaved |
| P | (..., T, n..32) | fp32 | TILE | DRAM interleaved |
| w (norm) | (1.., 1, C) | fp32 | TILE | DRAM interleaved |

### Output

| Tensor | Shape | Dtype | Layout | Memory |
|--------|-------|-------|--------|--------|
| X' | = X.shape | = X.dtype | TILE | DRAM interleaved |
| stats | (..., T, 32) | fp32 | TILE | DRAM interleaved |
| hc | (..., T, NG) | fp32 | TILE | DRAM interleaved |
| y | (..., T, C) | bf16 | TILE | DRAM interleaved (feed: unspecified — R4 uses `y_output` as staging scratch) |
| D (feed, in place) | (..., G·T, C/G) per chip | bf16 | TILE | DRAM interleaved, same address on every chip |

---

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| token-tile row `t` (`tensor_t_tiles`) | **independent**: every output is per-token; padded rows never mix with real rows | `block_rows` | 1, a decision (see below) | host constant `BLOCK_ROWS` → CT arg | split into `R` row groups of contiguous token-tile rows; R is chosen by the split chooser | knob-turn |
| C-column tile `c` (`tensor_c_tiles`) | **independent** for X', y and the plain mode. **Dependent** for stats, the norm statistic and hc, which are sums over all C (and all n streams). | `block_cols` | `min(core_c_tiles, block_cols_cap)`: the whole per-core column range | host `choose_block_cols()` → CT arg | split into `Q` column ranges inside each row group; the cross-core combine is the dependent-axis scheme | the plain split is built in Phase 0. The combine (stats/collapse) is a scheme-change, designed here. |
| stream `i`/`j` (n = 4) | **dependent** inside a unit: X'_j needs all n X_i of the same (t, c) | `n` (fixed by contract) | n, the whole axis | `n = post.shape[-1]` → CT arg | not split: a core owns all n streams of its columns | none. Splitting the streams would re-read X n times. |
| projection column (32 cols of W/stats) | **dependent**: K = n·C contracted, output width 1 tile | 1 tile, the whole axis | 1 | contract | not split | none |
| W rows (K = n·C) | **reuse-shared**: does not vary with `t`, so every row group needs the same slice | — (follows `block_cols`) | the core's column slice, resident across its rows | follows `block_cols` | same split as `c` | W multicast down column groups: scheme-change, deferred |
| coefficients post/comb/P (per token row) | **reuse-shared** along `c`: one tile row feeds every column of that token row | the expansion is per `t`; one coefficient set per row | 1 row | follows `block_rows` | each core reads the coefficient tiles of its own rows | none. They are tiny: one 4 KB tile per row per tensor. |
| norm weight w | **reuse-shared** along `t` | follows `block_cols` | the core's column slice, resident | follows `block_cols` | same as `c` | none (C·4 B total) |
| feed group position `p` (G chips) | **independent** per destination block | `feed_block_cols = tensor_c_tiles / G` | contract | host | each worker's y tiles go to the chip that owns their column block | TRANSPORT and OVERLAP refinements |

**Why `block_rows = 1`.** Each token-tile row has its own coefficient set: 20 column-expanded fp32 tiles, plus P, rstd and the per-row stats partial. A block spanning more rows makes all of that per-row state resident × block_rows, but it amortizes nothing extra:
- The streams are read per tile either way.
- The chain's per-element inits recur per tile inside the mix regardless (heterogeneous SFPU elements).
- The row finalization (partial send) is per row by definition.

The block is therefore wide along `c` and one row tall along `t`. Rows are walked as `block_idx`. This is an explicit, reasoned decision, not a collapsed knob: `block_rows` stays a CT parameter, and a value > 1 is a legal later measurement.

**Intermediate stages re-tabled.** Each derived stage has its own axes:

| Stage data | Its axes | Character | Assignment |
|-----------|----------|-----------|------------|
| per-core stats partial (1 tile per (t, core)) | t; contributing cores | t independent, contributors dependent | combined by a fixed 2-level tree. Level-1 leaders and the level-2 root **rotate per token row** (`leader(t, g) = member (t mod fanin) of chunk g`; `root(t) = member (t mod Q)` of the row group). The summation is spread across all cores, so no core does every row's combine. |
| stats row → hc (24 values × 32 tokens per tile row) | t | independent | computed on `root(t)`. Roots rotate with t, so at T ≥ 32·Q every core computes hc for ~Ht/Q rows. At T=160 only 5 roots are busy, but each runs one ~µs SFPU step; there is nothing to spread inside one tile row (32 tokens = 32 SFPU lanes, one block operation). |
| rstd (norm) | t | independent | computed on `root(t)` in the same hc block operation, returned to the row group's Q members |
| y column blocks (feed) | p | independent | each worker forwards its own y tiles; the ports only move them |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_sublayer_tiles`, `cb_stream_tiles` | `in_depth` (units in flight) | 2 | the reader fetches unit k+1 (n+1 = 5 tile reads in flight, the catalog's ≥4 batch) while compute mixes unit k |
| `cb_mixed_out_tiles` | `out_depth` | 2 | the writer drains unit k while compute packs unit k+1 |
| `cb_coef_cols` | `coef_depth` (token rows) | 2 | the next row's coefficient expansion overlaps this row's mix |
| `cb_y_hold` (norm) | `norm_rows_in_flight` | 2 | the stall shadow: row t+1 mixes while row t waits for its rstd |
| `cb_partial_slots` (leader/root) | `fanin` slots × 1 | fanin | every contributor writes its own slot, so the fold order is fixed |
| port staging (feed) | `feed_chunks_in_flight` | 4 chunks of 7 tiles (the fabric_all_gather 112 KiB CB setting) | packets in flight per link |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity (fp32 DEST, half-sync → `DEST_AUTO_LIMIT` = 4 tiles) | live tiles inside one mix chain: accumulator + coefficient + operand = 3 slots | the chain uses D0..D2. Never hold all n X_i in DEST. | compile-time static_assert (chain.hpp:397-417) |
| unpack-to-dest COL broadcast reads **column 0** of faces 0/2 only | coefficient position inside its tile | the reader copies coefficient column k into column 0 of a dedicated page (`cb_coef_cols`) | the wrong coefficient is broadcast (silent) |
| `post_col + n ≤ 32`, `comb_col + n² ≤ 32` | coefficient columns inside one tile row | validated on the host | the reader reads past the tile row (wrong values) |
| matmul in1 width = 1 tile (W has 32 columns) | projection output width | fixed at 1 | — |
| L1 working set per core | `block_cols` | `block_cols_cap = floor(L1_budget_for_block / bytes_per_block_col)` (ledger expression) | L1 overflow / allocation failure |
| fabric packet payload (14 KiB + 64 B on BH, 7 KiB on WH) | feed chunk = `run_pages = payload // 2048` bf16 tiles | `feed_chunk_tiles = min(run_pages, …)`, read from `ttnn.get_tt_fabric_max_payload_size_bytes()` | the router rejects or truncates the packet |
| `PacketHeaderPool` ≤ 12 headers per RISC (fabric_ring_exchange `port.cpp:68`) | packets in flight per port | `feed_chunks_in_flight ≤ 8` | header pool exhausted (hang) |
| semaphore counter width (32 bit) | arrivals per call | per-call counts ≪ 2³¹ | n/a |

### Regimes

The minimum, stated at the DRAM boundary: F, X, the coefficients, W, P and w each cross DRAM once; X', stats, hc, and y (or D) each cross once. Feed: the own D block crosses DRAM once locally, and the G−1 remote blocks cross one fabric link each.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 plain-2D**: row groups × column ranges, no communication | **built (Phase 0)** | `mode == plain` | `block_rows (1) × block_cols × n` units | Minimum for F, X and X': each F/X tile is read once by the unit's owner and each X' tile is written once. **Above the minimum for the coefficients:** each of the Q cores that share a token row reads that row's coefficient tile, i.e. Q × 4 KB per row (0.4 MB at the plain T=160 split, 2.2 MB at R=1). A per-row-group broadcast is the successor (perf lamp). | One chain set-up per block instead of per unit, and one coefficient expansion per row amortized over `block_cols` units. CB handshakes are per unit (streamed). |
| **R2 fused-gather**: R1, plus a per-row stats partial per core, a fixed 2-level combine to `root(t)`, the hc step on the root, and y from the resident bf16 X' | **deferred**: mode ∉ Phase 0 SUPPORTED (owner-set baseline). Reachable: R1's block schedule already has the slots (`project_block`, `finalize_row`, `combine_row`). Splits and CBs are additive. | `mode ∈ {stats, collapse}` | as R1, plus the partial tile per (t, core) | DRAM: W crosses R times (once per row group). Each core reads its slice once and keeps it resident across its rows, so this is minimum when R = 1. Cross-core: per row, `Q` partial tiles (4 KB) to leaders, then `ceil(Q/fanin)` to the root. **X' is never re-read** (the projection and y use the in-L1 bf16 X'). | Bigger `block_cols` → fewer matmul interm reloads per row (one K-chain per block) and fewer partial sends (one per row per core). |
| **R2-norm**: R2, plus `Σy_b²` in partial column 25; the root returns rstd; members hold y_b until rstd arrives | deferred (norm ∉ Phase 0). Reachable: same tree, plus a reverse unicast of 128 B per member. | `norm == on` | as R2, plus `cb_y_hold` = `block_cols × norm_rows_in_flight` | DRAM: y crosses once (written after rstd). Cross-core adds Q × 128 B per row. | — |
| **R2-W-streamed**: W too large to keep resident is re-read per row | deferred, fallback. Predicate-guarded: R2's default is W-resident. | `n·core_c_tiles·4096 > W_resident_budget` | W streamed per block | W crosses `rows_per_core` times for that core's slice | — |
| **R3 W-mcast**: one reader per column group reads the W slice and mcasts it down the R row groups | deferred: W-resident R2 already reaches the W minimum when the chooser picks R = 1 (every shape where `tensor_c_tiles ≥ #workers`). R3 only pays where R > 1 (small C: 1792 at T ≥ 256; 2560 at T=1024+), saving `(R−1)·n·C·128 B` of DRAM. Structure kept: column ranges are identical across row groups, so the cores sharing a slice form a fixed `Mcast1D`/chain family (`mcast_host.hpp:303`). | `R > 1` | as R2 | saves `(R−1)` W reads; adds a NoC mcast of n·C·128 B per column group | — |
| **R4 feed-DRAM-staged**: workers write remote-destined y tiles to `y_output` (scratch) grouped by destination, then signal the ports; the ports read and send fused write+inc packets to the peer D | **deferred** (feed ∉ Phase 0). It is the **stepping stone**: the port kernel (fabric_all_gather's reader + sender structure) survives into R5 unchanged except for its source. | `feed == on` | y tiles of a worker's block, grouped by destination column block p | DRAM: +(G−1)/G·T·C·2 B written **and** read again (the staging) above the minimum. Fabric: (G−1)/G·T·C·2 B out per chip. | Bigger staging chunks → fewer port wake-ups |
| **R5 feed-L1-direct**: workers NoC-write remote y tiles straight into port L1 staging slots (credit semaphores); the ports send | deferred successor of R4 (OVERLAP + DRAM). Saves `2·(G−1)/G·T·C·2 B` of DRAM traffic, ≈ 2.4 MB at T=160, C=5120 (~13 % of the call's bytes). Positive reason not to build first: R4 reuses the measured transport and gives the per-chip breakdown that ranks R5 against the TRANSPORT items. | `feed == on` | as R4 | minimum DRAM; adds worker→port NoC traffic of the same bytes | — |
| **X1 tokens-only split** (each core owns whole token rows) | **rejected**: underfills. At T=160 there are 5 token-tile rows, so 5 of 110 cores. Superseded by R1/R2. | — | — | — | — |
| **X2 re-read X' from DRAM** for projection / collapse (a second and third pass) | **rejected: dead end.** It adds 2 × T·n·C·2 B of DRAM reads, and every good scheme instead keeps X' resident (R2). Forbidden by the contract. | — | — | — | — |
| **X3 separate programs** (mix, matmul, hc, collapse, norm, all_to_all) | **rejected**: violates one-dispatch | — | — | — | — |

**Regime selection (host, exact).** `mode` comes from the kwargs. `norm` = `norm_weight is not None`. `feed` = `feed_output is not None`. R2-W-streamed is chosen iff `n·max_core_c_tiles·4096 > W_RESIDENT_BUDGET` (ledger symbol). R3/R5 are not selectable until built. Regime-pinned tests:
- plain vs collapse on the same shape: test_regression's bit-identity of X'.
- A W-streamed pin: max_grid 3×2 at C=7168 forces a large `core_c_tiles`.

### Split chooser (core-assignment, exact)

**Worker set `L`.** These are the cores in the rectangle `(0,0)..(max_grid−1)` in row-major order, minus the **reserved port cores** whenever the device runs with an active fabric (`ttnn.get_fabric_config() != DISABLED`) and the mesh has ≥ 2 chips along `feed_cluster_axis`.
- The reserved set always has `2 × MAX_FEED_LINKS (=2)` = 4 cores. Its placement is a function of (mesh, cluster axis, rectangle) only: the free core nearest each link's Ethernet core (fabric_all_gather `_worker_nearest_noc1`), clamped into the rectangle.
- **Why reserve the ports even when the feed is off:** stats bits depend on how column partials are grouped, and the grouping depends on L. The feed contract requires the feed call's X', stats and hc to equal, bit for bit, the same call without the feed on the same mesh. So L must not depend on the feed kwargs.
- On a single device (fabric off), L is the whole rectangle.

**Candidates.** Enumerate `R ∈ 1..min(tensor_t_tiles, |L|)`. Set `Q = min(floor(|L|/R), tensor_c_tiles)`.
- Row group g is the members `L[g·Q : (g+1)·Q]`. It owns token-tile rows `[g·Ht/R, (g+1)·Ht/R)`, split ragged by ±1 via `ttnn.split_work_to_cores`-style arithmetic.
- Member q owns C-column tiles `[q·Ct/Q, (q+1)·Ct/Q)`, also ragged ±1.

**Score:** `est = max(total_dram_bytes(R) / DRAM_BW, max_core_bytes(R, Q) / CORE_BW)`, where:
- `unit_bytes = F_tile + n·X_tile + n·X'_tile (+ y_tile if mode == collapse … see note)`;
- `total_dram_bytes(R) = Ht·Ct·unit_bytes + has_W·R·n·C·128`;
- `max_core_bytes = ceil(Ht/R)·ceil(Ct/Q)·unit_bytes + has_W·n·ceil(Ct/Q)·4096`;
- `DRAM_BW` = 512e9 on BH and `CORE_BW` = 17.9e9, both host constants from `double_buffer` in the catalog.

Pick the minimum `est`. Ties go to the smaller R (fewer W reads, smaller combine fan-in).

**As implemented (Refinement 1).**
- With W, the per-core term is `rows·(ROW_NS_W + cols·UNIT_NS_W)`, fitted on device: the fused pass is compute-bound per core, not DRAM-fed.
- Only splits whose column range fits one block pass (`cols ≤ MAX_BLOCK_COLS_W`) are candidates when any exists.
- It still sees only (T, C, dtypes, has_W, |L|), so the stats-vs-collapse bit-identity holds.
- Combine:
  - root(t) = member `(t − t0) mod Q`, with its work deferred past the root's own rows;
  - the level-1 fold is lagged one row;
  - leader slot reuse is credited by a ready bitmask semaphore.
- In the W path, a ragged pass runs the full-block chains on zero-filled filler units.

**Note.** The score must use only `(T, C, X.dtype, F.dtype, has_W, |L|)`. It must not use collapse, norm or feed, because "stats bit-identical between stats and collapse" and "norm leaves stats columns ≠ 25 unchanged" need the same partial grouping. So `unit_bytes` always counts the y tile when `has_W`.

Phase 0 (plain, `has_W = 0`) makes `total` independent of R, so the tie-break term (`max_core_bytes`) picks the balanced split. **X' bits never depend on the split**: every X' element is computed by the same op sequence on the same inputs.

| Case | |L| | Chooser result | Units per core (max) |
|------|-----|---------------|---------------------|
| T=160, C=5120 plain (Ht=5, Ct=160) | 110 | R=5, Q=22 | 1 × 8 = 8 (ideal 7.3) |
| T=160, C=5120 collapse, fabric on | 106 | R=1, Q=106 (W read once wins on DRAM) | 5 × 2 = 10 |
| T=640, C=1792 (Ht=20, Ct=56) | 110 | R=2, Q=55 | 10 × 2 |
| T=1, C=7168 (Ht=1, Ct=224) | 110 | R=1, Q=110 | 1 × 3 |
| T=32, C=32 (Ct=1) | 110 | R=1, Q=1 (one unit) | 1 — the shape has one unit |

### Traffic ranking

| Candidate split | DRAM bytes (beyond minimum) | Cross-core | Verdict |
|-----------------|-----------------------------|------------|---------|
| C-columns only (R=1, Q=|L|) | **0**: W read once in total | stats: Ht·|L| partial tiles × 4 KB (2.2 MB at T=160 into a 2-level tree) | **cheapest traffic** for stats/collapse. The chooser picks it whenever `Ct ≥ |L|`. |
| 2-D (R>1) without W-mcast | +(R−1)·n·C·128 B (W re-read) | Ht·Q partials (fewer per row) | used only where R=1 cannot fill the grid (small Ct). Plain mode: 0 extra, so this is the balanced choice. |
| 2-D with W-mcast (R3) | 0 | + W mcast | deferred (R3) |
| tokens only | 0 | 0 | rejected: occupancy (X1) |
| dependent stream axis split | +(n−1)× X re-reads | — | rejected |

**Data-movement budget (chosen split, T=160, C=5120, bf16, collapse + norm, R=1):**
- DRAM: F 1.6 MB, X 6.6 MB, X' 6.6 MB, y 1.6 MB, W 2.6 MB, plus small tensors. That is ≈ 19 MB, the minimum.
- Cross-core: 2.2 MB of partials and 70 KB of rstd.

See `l1_ledger.md`.

### Block schedule

The logical schedule for one worker. Reader, compute and writer realize their parts asynchronously.

```cpp
expand_W_slice_once();                       // R2: W slice for my columns → cb_w_slice (resident)
for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // = my token-tile rows × col passes
    load_coefficients_row(block_idx);        // reader: post/comb (and P) tile(s) of row t → column-0 pages
    load_block(block_idx);                   // reader: F + n X tiles per unit, streamed (in_depth)
    mix_block(block_idx);                    // compute: X'_j for all units, pack → cb_mixed_out_tiles
                                             //   [R2] also → cb_mixed_bf16 (resident for the block),
                                             //   Xmix² L1-accumulated into cb_sumsq_acc
    store_block(block_idx);                  // writer: X' tiles → DRAM
    collapse_block(block_idx);               // [collapse] y_b = Σ P_i X'_i → cb_y_hold (norm) / cb_y_out
    project_block(block_idx);                // [R2] matmul_block(cb_mixed_bf16 × cb_w_slice) → partial
    finalize_row(block_idx);                 // [R2, last col pass of row t] SFPU row-reduce the acc tiles,
                                             //   writer merges col0s into partial cols 24/25, sends to leader
    norm_block(block_idx - lag);             // [norm] wait rstd(t), y = (y_b·rstd)·w → cb_y_out / feed
}
// on leader/root cores, interleaved by the dataflow kernels as slots fill:
combine_row(t);                              // fixed-order SFPU fold of fanin partial slots
hc_step_row(t);                              // [collapse, root] custom SFPU block op, 32 tokens = 32 lanes
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_coefficients_row` | 1 token-tile row: 1–2 raw coefficient tiles → 20 (24 with P) column-0 pages | the expanded pages, for the whole row (`coef_depth` rows) | one DRAM read per coefficient tensor per row (packed: one tile serves post, comb and P). 32 word copies per expanded page. |
| `load_block` | `block_cols` units of (1 + n) tiles | — (streamed, in_depth) | one barrier per unit (5 reads in flight) |
| `mix_block` | `block_cols × n` output tiles | coefficient pages of the row | chain set-up per output tile (heterogeneous chain: bcast-copy / copy / SFPU mul / SFPU add). No init outside the chain. |
| `store_block` | `block_cols × n` tiles | — | one write barrier per unit |
| `collapse_block` | `block_cols` y tiles | `cb_mixed_bf16` (n × block_cols) | per tile |
| `project_block` | in0: 1 × (n·block_cols) tiles; in1: (n·block_cols) × 1; out 1 tile | W slice (whole core), interm partial across col passes | one `mm_block_init` per block (re-init after the SFPU phase) |
| `finalize_row` | 1 tile (proj) + 1–2 acc tiles | — | once per token row per core |
| `combine_row` | fanin partial tiles → 1 | slot CB | once per row per leader/root. One NoC write + one semaphore inc per contributor. |
| `hc_step_row` | 1 permuted tile (32 tokens × 25–26 values) | — | once per token row, on root(t) only |
| `norm_block` | `block_cols` y tiles | `cb_y_hold` until rstd(t) lands | once per row |

**Stall shadow.** Two stages wait on peers: `norm_block` (rstd) and `combine_row` / `hc_step_row` (partials).
- Neither the next row's mix nor its projection depends on rstd(t). So the schedule lags `norm_block` by `norm_rows_in_flight − 1` rows: row t+1's `load_block`/`mix_block` run inside row t's rstd wait.
- y_b is algebraically independent of rstd, which is what lets the rounding to y_b happen before the wait. The norm only scales it.
- No floating-point order changes: y_b, rstd and w are applied in the contract's order. So the reorder needs no precision gate.
- The last row's round trip stays exposed. Nothing is independent of it.
- The roots run `combine_row`/`hc_step_row` for row t on their compute thread **between** their own blocks. The dataflow kernel signals the compute via a "slots full" CB push, so the hc for row t overlaps the root's mixing of later rows.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap: `in_depth = 2`, 1 unit per transfer | 5 tiles per barrier is at the catalog's batch floor. With the reader issue-bound, the reads may not saturate DRAM at 110 cores. | `in_depth` 3–4, or 2 units per transfer; split the reads across both RISCs (`split_reader`, catalog) |
| mix chain cost (9 SFPU tile-ops per output tile, X_i copied n times) | the SFPU may bind before DRAM: about 30 µs estimated at T=160 against the ~37 µs roofline | a fused SFPU MAD (`acc += coef·x`, one op instead of mul + add); `sfpu_tile_scope` does not apply (full tiles) |
| grid synchronization: combine fan-in Q = 106 per row at R=1 | partial traffic and fold time per root (fanin·~300 cycles) may dominate at small T | `fanin` 8 / 11 / 16; R=5 with W re-read (R2) vs R=1 |
| hc permutation on RISC-V (2 × 1024 word moves per row on root) | at T=4096, 128 rows spread over 110 roots ≈ 1–2 µs each, likely hidden | SFPU lane-shuffle in DEST (`SFPTRANSP`) |
| coefficient tile read by all Q cores of a row (reuse-shared along c) | at R=1, T=160 that is 530 × 4 KB reads ≈ 11 % of the call's bytes | one reader per row group reads the row's coefficient tile(s) and mcasts them to the group (`mcast_pipe`); or read only the needed face rows |
| `block_rows = 1` | per-row state is small; a 2-row block halves the coefficient-load latency exposure at the row seam | `block_rows = 2` |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| F, X DRAM → worker L1 | F.dtype / X.dtype tiles | reader (RISCV_1, NoC0) via `TensorAccessor`, one unit (1 + n pages) per barrier | X page of stream i, row t, col c = `t·(n·Ct) + i·Ct + c` |
| post/comb/P DRAM → L1 → column-0 pages | fp32 | reader reads tile `t` of each coefficient tensor (once if packed or shared) into `cb_coef_raw`, then copies column (`post_col+j`), (`comb_col+i·n+j`), (P col i) into column 0 of pages in `cb_coef_cols`: 16 words in face 0, 16 in face 2 | position (r, k) in a 32×32 tile = face `(r/16)·2 + k/16`, offset `(r%16)·16 + k%16`. Exact fp32 bits. |
| coefficient broadcast into DEST | fp32 → fp32 DEST | `UnaryBcast<Col>` via **unpack-to-dest** (bcast.h:46-56: fp32 CB → `unpack_to_dest=true`) | requires `unpack_to_dest_mode[cb_coef_cols] = UnpackToDestFp32`. Never srcB (19-bit). |
| F / X tiles into DEST | fp32: unpack-to-dest exact; bf16: srcA → fp32 DEST exact | `CopyTile` | `UnpackToDestFp32` on fp32 F/X CBs |
| mix | fp32 DEST | SFPU `MulBinary`, `AddBinary` (fp32 lane arithmetic) | the FPU never multiplies a coefficient (tf32 bias) |
| X' pack | X.dtype | packer (fp32 exact; bf16 = one rounding) | see Risks (packer rounding mode) |
| X' L1 → DRAM | X.dtype | writer (RISCV_0, NoC1) | |
| W DRAM → L1 (R2) | fp32 tiles | writer-side or reader, once per core, resident | in1 of the projection (W at 9-bit mantissa fits srcB exactly) |
| partial → leader → root (R2) | fp32 tile (4 KB) | worker writer: NoC unicast write into slot `q` of `cb_partial_slots` on the leader, then `noc_semaphore_inc` (local sem from `CreateSemaphore` / program semaphore) | slot-indexed, so the fold order is fixed |
| root stats → DRAM | fp32 | root writer | |
| stats tile → hc lane layout | fp32 | root dataflow permutes 26 columns × 32 tokens so value v occupies SFPU vector v with token t in lane t (a vector = 4 rows × 8 stride-2 columns of a face, catalog `sfpu_tile_scope`) | the inverse permutation goes to the hc / rstd outputs |
| rstd → members (norm) | 32 fp32 = 128 B | root dataflow unicasts into each member's `cb_rstd_col` column 0 + sem inc | |
| y → DRAM / D | bf16 | writer: own column block to local D rows `[p·T..]` (feed) or to y | |
| y remote blocks (feed, R4) | bf16 | worker writes them to `y_output` scratch and incs the port's local semaphore; the port reads 7-tile chunks and sends fused write+atomic-inc packets to the peer D (fabric_all_gather structure) | TRANSPORT from the examples |

### Feed (designed; deferred regime R4 → R5)

**TRANSPORT.** Taken from `fabric_all_gather` (balanced ring, relay-free on the 2-D fabric):
- **Ports.** One port per (direction, link): 2·L ports. They are the reserved cores of `L`, each next to its link's Ethernet core.
- **Host API.** `ttnn.setup_fabric_connection(src, dst, link, program, core)` and `ttnn.get_forwarding_link_indices`. The 2-D route comes from `fabric_set_unicast_route(hdr, chip_id, mesh_id)`, so the far chip (2 hops) is reached directly through the neighbour's router (fabric_ring_exchange: direct routing costs only the extra ready hop).
- **Kernel API.** `WorkerToFabricEdmSender` and `to_noc_fused_unicast_write_atomic_inc`. Equivalently, `ccl_helpers_dataflow.hpp`: `DirectConn` (:190), `UnicastWriteChannel::write_page` (:319-327).
- **Chunking.** Chunks are `payload // 2048` = 7 bf16 tiles. Arrival is incremented every 8 chunks and on the last chunk (more often costs 8–17%, AG report).
- **Far-block split.** The far chip's block goes half per direction, which leaves 1.5 blocks on the busiest link (AG `balance`: 1.26× at size).
- **Expected time.** About 29 µs at 1 link and ~18 µs at 2 links for 3 × 200 tiles out per chip (AG fit). Roofline: ~8 µs busiest link at 2 links.

**Handshakes.** These use the caller's semaphores, placed at the port cores' L1 addresses.
- *ready*: at start, each port sends a header-only atomic inc to the matching port of every destination chip. It then waits `ready ≥ #destinations` before its first data packet, and subtracts its count at the end. This fence means no chip writes into a peer's D before the peer has started this call.
- *arrival*: incoming packets increment the receiving port's arrival counter. The receiving port waits for the expected total (a closed form in T, C, G, L), then subtracts it (`noc_semaphore_inc(addr, 0u − n)` + `noc_async_atomic_barrier`). So the program on this chip ends only when D is complete, and both semaphores are back at 0.
- No global semaphore and no buffer is created per call (trace-safe).

**OVERLAP.** Ports are idle for most of the call, so they send as entries land:
1. Each worker processes its rows in row order and, after `norm_block(t)`, publishes its remote-block tiles of row t. In R4 it writes them to y_output and increments a port-local "landed" counter per (row, destination). The writer's DRAM write barrier comes before the inc.
2. The port waits per entry, not per call: it streams row t while workers compute row t+1.
3. The chain from last landing to output is then: last row's rstd → norm → publish → port send (≤ 1 chunk group) → arrival wait.
4. Launch skew between chips is absorbed by the ready fence, which only blocks the ports. The workers never wait on remote chips.
5. Expected exposure: the transport of the final row's blocks, plus the skew.

R5 replaces step 1's DRAM hop with a NoC write into port L1 slots under credit semaphores.

**As implemented (Refinement 3).** Step 1's "landed" signal is a per-worker count pushed into every port's
`cb_feed_poll` slot (inline write behind the y write barrier; the port zeroes its slots, then arms each worker on
`SEM_FEED_ROWS`, before any count may land), so the wave wait is a local scan. The port reader keeps
`FEED_READ_BATCH` staging chunks in flight per barrier. The lagged-root schedule (`NORM_ROOT_LAG` > 0: rstd within
`NORM_ROWS_IN_FLIGHT − 1` rows, publication inside the row loop) is built but parked at 0: at T160 / T1280 the root
chain (~25 µs) exceeds a row and the mid-loop hc on a root worker stalls the combine, so the deferred roots win and
step 3's chain still starts at the end of the row loop.

### Placement as a TARGET axis

`memory_layout` is not in TARGET: inputs are DRAM-interleaved only, so no shard scheme is designed. The physical realization of the R1 split would be BLOCK_SHARDED (rows × C-ranges), a knob-turn because each core's work is self-contained. Splitting along C for the stats is a scheme change (the combine in R2).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = 1 token-tile row × `block_cols` C-column tiles × n streams |
| Grid | worker list `L` = the max_grid rectangle (row-major) minus the reserved ports when the fabric is active. Kernel core sets are built as merged `CoreRangeSet`s (rectangles), never one range per core (dispatch check). |
| Per-core work | row group g = `L[g·Q:(g+1)·Q]` owns token-tile rows `[g·Ht/R, (g+1)·Ht/R)`. Member q owns column tiles `[q·Ct/Q, (q+1)·Ct/Q)`. Blocks = rows × `ceil(core_c_tiles / block_cols)` column passes. |
| Remainder | ragged ±1 on both axes (the first `Ht mod R` groups and the first `Ct mod Q` members take one more). `tensor_t_tiles = Σ_images ceil(T/32)` (T is the last-but-one dim; leading dims fold row-major into whole images, and each image is tile-padded independently — here T is the only padded dim). Cores with no work (|L| > R·Q) get no kernels. |
| Leaders / roots (R2) | chunk g of the row group = members `[g·fanin, (g+1)·fanin)`. `leader(t, g) = chunk member (t mod chunk size)`. `root(t)` = member `(t mod Q)`. |
| Runtime args | per-core: row range, column range, roles, slot indices. Common args: tensor addresses and hc scalars. Set `program.custom_program_hash` from the layout (ProgramDescriptor cache caveat, fabric_ring_exchange README:73-79). hc scalars, post_col/comb_col and norm_eps are **runtime** values, re-sent every call. |

Regime-pinned tests are required:
- plain/stats/collapse on the same shape (X' bit-identity);
- max_grid 11×10 and 3×2;
- the W-streamed predicate (3×2 grid, C=7168);
- feed vs no-feed on the mesh (bit-identity under the reserved-port worker set).

## Circular Buffers

Ledger: `l1_ledger.md`. Indices are any distinct values. The symbols are: `bc = block_cols`; `ind = in_depth`; `outd = out_depth`; `cd = coef_depth`; `nr = norm_rows_in_flight`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_sublayer_tiles` | 0 | F tile | `ind` | streams over c (1 tile per unit) | F.dtype | reader | compute | block |
| `cb_stream_tiles` | 1 | X tile | `n·ind` | spans streams (n), streams over c | X.dtype | reader | compute | block |
| `cb_coef_raw` | 2 | 4096 | 3 (post, comb, P; 1 when one packed tensor serves all) | raw tile row of the current token row | fp32 | reader | reader (scratch landing, never pushed to compute) | per row |
| `cb_coef_cols` | 3 | 4096 | `(n + n² [+ n if collapse])·cd` | spans the row's coefficients, streams over t | fp32 (`UnpackToDestFp32`) | reader | compute | per row |
| `cb_mixed_out_tiles` | 4 | X' tile | `n·outd` | spans n, streams over c | X.dtype | compute | writer | block |
| `cb_mixed_bf16` (R2) | 5 | 2048 | `n·bc` | spans n and the block's columns (matmul K, y) | bf16 | compute | compute | block |
| `cb_w_slice` (R2) | 6 | 4096 | `n·core_c_tiles` (W-resident) / `n·bc` (W-streamed) | resident across rows | fp32 | reader | compute | kernel |
| `cb_sumsq_acc` (R2) | 7 | 4096 | 1 | packer L1 accumulation over (n × cols) of a row | fp32 | compute | compute | per row |
| `cb_proj_interm` (R2) | 8 | 4096 | 1 | matmul partial across column passes | fp32 | compute | compute | per row |
| `cb_partial_out` (R2) | 9 | 4096 | 2 (proj + sumsq col) [+1 ysq] | the per-row partial and its reduce columns | fp32 | compute | writer | per row |
| `cb_partial_slots` (R2, leaders/roots) | 10 | 4096 | `fanin` | one slot per contributor | fp32 | writer (receives via remote NoC writes; it pushes once the semaphore count is reached) | compute | per row |
| `cb_combined` (R2) | 11 | 4096 | 1 | leader's folded partial (to root) / root's stats tile | fp32 | compute | writer | per row |
| `cb_hc_lanes_in` (collapse, root) | 12 | 4096 | 1 | permuted stats | fp32 (`UnpackToDestFp32`) | writer (permuter) | compute | per row |
| `cb_hc_lanes_out` (collapse, root) | 13 | 4096 | 1 | hc + rstd in lane layout | fp32 | compute | writer | per row |
| `cb_y_hold` (collapse) | 14 | 2048 | `bc·nr` (norm) / `bc` | y_b until rstd (norm) or until written | bf16 | compute | compute (norm) / writer (no norm: renamed `cb_y_out`) | per row |
| `cb_y_out` (collapse) | 15 | 2048 | `bc` | final y tiles | bf16 | compute | writer | block |
| `cb_ysq_acc` (norm) | 16 | 4096 | 1 | Σ y_b² over the row's cols | fp32 | compute | compute | per row |
| `cb_rstd_col` (norm) | 17 | 4096 | `nr` | rstd in column 0, per row | fp32 (`UnpackToDestFp32`) | reader (receives from root) | compute | per row |
| `cb_norm_w` (norm) | 18 | 4096 | `core_c_tiles` | w row 0 of my columns, resident | fp32 (`UnpackToDestFp32`) | reader | compute | kernel |

Phase 0 (plain) allocates only `cb_sublayer_tiles`, `cb_stream_tiles`, `cb_coef_raw`, `cb_coef_cols` (n + n² pages × cd) and `cb_mixed_out_tiles`.

`cb_coef_raw` is reader-private scratch. It is not a CB handshake: the reader reads and expands from it within one kernel, so it has one owner.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_coefficients_row` | 1 row | dataflow (TensorAccessor) | DRAM → `cb_coef_raw` | `cb_coef_cols` (n+n²[+n] pages) | expanded pages held for the whole row; popped by compute after the row's last unit |
| 2 | `load_block` | bc units | dataflow | DRAM | `cb_sublayer_tiles` (1/unit), `cb_stream_tiles` (n/unit) | — |
| 3 | `mix_block` | bc × n tiles | `eltwise_chain` per output tile j | `cb_coef_cols` (wait upfront, no pop; indexed: post_j = page j, comb_ij = page n + i·n + j), `cb_sublayer_tiles` / `cb_stream_tiles` (waited per unit, popped after the unit's n outputs) | `cb_mixed_out_tiles` (+ `cb_mixed_bf16`, `cb_sumsq_acc` L1-acc in R2) | F, X popped per unit |
| 4 | `store_block` | bc × n | dataflow | `cb_mixed_out_tiles` | DRAM | — |
| 5 | `collapse_block` | bc | `eltwise_chain` | `cb_coef_cols` P pages, `cb_mixed_bf16` (no pop) | `cb_y_hold` / `cb_y_out` (+ `cb_ysq_acc` L1-acc) | — |
| 6 | `project_block` | in0 1×(n·bc), in1 (n·bc)×1 | `matmul_block` | `cb_mixed_bf16` (pop), `cb_w_slice` (no pop: resident; In1BaseOffsetFn = column-pass offset) | `cb_proj_interm` (LastBlockTarget::Interm until the row's last pass) → `cb_partial_out` | — |
| 7 | `finalize_row` | 1–2 tiles | `eltwise_chain` (CopyTile + raw `sfpu_reduce` element) | `cb_sumsq_acc`, `cb_ysq_acc` | `cb_partial_out` | acc CBs reset (next row's first pack seeds) |
| 8 | `combine_row` | fanin tiles | `eltwise_chain` (CopyTile unpack-to-dest + `AddBinary`, fixed slot order) | `cb_partial_slots` | `cb_combined` | — |
| 9 | `hc_step_row` | 1 tile, 32 lanes | **custom SFPU block op** (raw sfpi) | `cb_hc_lanes_in` | `cb_hc_lanes_out` | — |
| 10 | `norm_block` | bc | `eltwise_chain` | `cb_y_hold` (pop), `cb_rstd_col` (pop per row), `cb_norm_w` (no pop) | `cb_y_out` | — |

**Mix chain per output tile j** (one dst-sync window, 3 DEST slots):
1. `UnaryBcast<Col, cb_coef_cols[post_j]> → D0`, `CopyTile<cb_sublayer_tiles> → D1`, then `MulBinary<D0,D1,D0>`.
2. For each i: `UnaryBcast<Col, cb_coef_cols[comb_ij]> → D1`, `CopyTile<cb_stream_tiles[i]> → D2`, `MulBinary<D1,D2,D1>`, `AddBinary<D0,D1,D0>`.
3. `PackTile<cb_mixed_out_tiles>(D0)`.
4. R2 only: `PackTile<cb_mixed_bf16>(D0)`, then `Square<D0>` and `PackTile<cb_sumsq_acc, L1Accumulation::Enabled>(D0)`.

The column-0 coefficient pages are picked with `TileAddressing::Offset` on an upfront / no-pop input.

**The hc block operation, in lane layout.** All values are fp32 SFPU, approximation off. v = value index (vector), and every lane is a token.
1. `s = rsqrt(stats[24]/hc_norm_width + hc_norm_eps)` (accurate rsqrt; the catalog Newton body is acceptable only at fp32 grade).
2. `z_v = stats[v]·s`.
3. pre and post:
   - `pre_k = sigmoid(a_pre·z_k + b_k) + hc_eps`
   - `post_k = 2·sigmoid(a_post·z_{n+k} + b_{n+k})`
4. `L_{jk} = a_res·z_{2n+jk} + b_{2n+jk}`.
5. Softmax per row j: subtract the row max, exp (exact), divide by the row sum, then add hc_eps.
6. Column step: divide by (column sum + hc_eps).
7. Repeat `iters−1` times: the row step, then the column step.
8. Norm only: `rstd = rsqrt(stats[25]/C + norm_eps)` in lane 25.

Divisions are `x·recip(d)` only where recip is correctly rounded. Otherwise use a true fp32 division (`DivBinary` semantics). The gate is 1e-5 relative to float64; the implementer verifies this on device. All 16 matrix entries plus the scratch row/column sums fit in DEST vectors of the single tile, so no L1 round trip happens inside the 20 iterations.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | raw_api | `compute_kernel_hw_startup` | tt_metal/hw/inc/api/compute (boot) | first statement of MAIN; one boot per stage | — | — | — |
| mix / collapse / norm / combine / finalize | helper | `eltwise_chain` | ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:533 | `IterationShape::one_tile()` per output tile inside the block op (the chain is heterogeneous, so inits cannot be hoisted; `InitReconfigOwner::Chain`) | see above | see above | `block_cols` drives the block walk |
| coefficient / rstd / w broadcast | helper | `UnaryBcast<BroadcastDim::Col / Row, input(cb, WaitPolicy::Upfront/None, PopPolicy::None, …, TileAddressing::Offset)>` | eltwise/broadcast/bcast.hpp:25; fp32 path bcast.h:37-56 (unpack-to-dest) | Col for coefficients/rstd, Row for w | `cb_coef_cols`, `cb_rstd_col`, `cb_norm_w` | DEST | — |
| copies | helper | `CopyTile<input(cb)>` | chain.hpp:483 | — | F, X, `cb_mixed_bf16`, slots | DEST | — |
| SFPU arithmetic | helper | `MulBinary`, `AddBinary` | eltwise/binary/sfpu/sfpu.inl:38,56 (decl basic.hpp:15-30) | D slots | — | — | — |
| square | helper | `Square` | eltwise/unary/misc.inl:67 | — | — | — | — |
| pack (+ L1 acc) | helper | `PackTile<output(cb, …, L1Accumulation::Enabled)>` | chain.hpp:504; enum :272 | — | — | out CBs | — |
| projection | helper | `matmul_block` + `MatmulBlockShape::of` | matmul_block_helpers.hpp:358, :160; `LastBlockTarget` :62 | `in0_block_k = n·block_cols`, `num_k_blocks` = column passes of the row; `LastBlockTarget::Interm` until the last pass; `packer_l1_acc` across passes; subblock = 1×1 (out is one tile) | `cb_mixed_bf16`, `cb_w_slice` | `cb_proj_interm` → `cb_partial_out` | `block_cols` (K per block). Subblocking: N/A (1 tile). |
| row reduce of acc | raw_api | `sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>` | tt_metal/hw/inc/api/compute/compute_kernel_api.h:742 (init :792) | in DEST after CopyTile | `cb_sumsq_acc` | `cb_partial_out` | — |
| hc step | raw_api | custom sfpi kernel (`calculate_*` style; `sfpi::dst_reg`, `vFloat`) | sfpi headers; precedent `ckernel_sfpu_reduce.h` | iters as RT arg | `cb_hc_lanes_in` | `cb_hc_lanes_out` | — |
| W mcast (R3) | helper | `SenderPipe` / `ReceiverPipe`, host `Mcast1D` | mcast/kernel/mcast_pipe.hpp:55,126; mcast/host/mcast_host.hpp:303 | per column group | — | — | — |
| feed transport | helper | `DirectConn`, `UnicastWriteChannel`, `FabricStream` | ccl/ccl_helpers_dataflow.hpp:190,319,295 | chunk = payload/2048 tiles | port staging | peer D | `feed_chunk_tiles`, `feed_chunks_in_flight` |

**Helpers considered and rejected (raw_api rows).**

| raw_api row | Helper considered | Mismatch (file:line) | What would close it |
|-------------|-------------------|----------------------|---------------------|
| row reduce of the acc tile | `reduce()` (reduce_helpers_compute.hpp:632) | Both algorithms (`ReduceAlgorithm` :156) are unsuitable. ReduceTile is the FPU matmul-with-scaler, which reads fp32 through srcA/srcB at tf32: the sum of squares would not be "exact fp32" (gate 1e-6). AccumulateViaAdd's cross-tile step is FPU `add_tiles` (:129), also tf32. Only its finalize is `sfpu_reduce`, and the cross-tile accumulation here is already done by the packer L1-acc. | a `reduce` option "SFPU-only within-tile finalize on a DEST tile" |
| sum of partials | `reduce()` / AccumulateViaAdd | the same FPU tf32 cross-tile add (:129) | SFPU accumulate mode |
| hc step | eltwise chain SFPU ops (`Exp`, `Sigmoid`, `Rsqrt`, `Recip` math.inl:29-62, activations.inl:31) | The chain works tile-elementwise. hc needs cross-value math per token: row/column sums of a 4×4 matrix stored in one row of 24 values. As tiles that is ~1100 full-tile SFPU ops per row, ≈ 120 µs. The lane layout plus one sfpi kernel does it in vectors. | a "per-lane scalar program" block helper |

**Coefficient multiply on the FPU is forbidden.** `BinaryFpu` with `BroadcastDim::Col` reads the coefficient through srcB at tf32, which biases every product by ~2⁻¹¹ (contract). The SFPU path is used for every coefficient product. The FPU is used only for the projection matmul (bf16 X' × 9-bit W: exact operands).

## Broadcast Verification

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| mix | coef × F / X_i (SFPU in DEST after bcast copy) | `cb_coef_cols` page: Col0 (32 token values) | F / X tile: All | Col (unpack-to-dest) |
| collapse | P_i × X'_i | `cb_coef_cols` P page: Col0 | `cb_mixed_bf16`: All | Col |
| norm | y_b × rstd | `cb_rstd_col`: Col0 | `cb_y_hold`: All | Col |
| norm | (·) × w | `cb_norm_w` tile: Row0 | DEST: All | Row (unpack-to-dest) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| fp32 CBs read through srcA/srcB at tf32 | without `unpack_to_dest_mode = UnpackToDestFp32`, `copy_tile`/`unary_bcast` of fp32 truncate to tf32. That gives a biased X' (bias gate) and an inexact sum of squares. | set `UnpackToDestFp32` for every fp32 CB the compute reads: F/X (fp32), `cb_coef_cols`, `cb_rstd_col`, `cb_norm_w`, `cb_partial_slots`, the acc CBs, `cb_hc_lanes_in`. Phase 0's first probe checks `post=1, comb=I` → X' bit-exact. |
| unpack-to-dest COL broadcast semantics | bcast.h:46 routes fp32 through `unpack_srca_to_dest_column` (llk_unpack_A.h:106-112), which is less exercised than the srcB path | the first probe pins it: column-0 page with distinct per-row values → DEST broadcast check (golden "cyclic" comb pins the orientation) |
| packer fp32→bf16 rounding mode | the bf16 X'/y bias gates allow one RNE rounding. A truncating packer biases by 2⁻⁹. | verify on the first bf16 run (refinement). If it truncates, round in DEST with an SFPU RNE typecast before the pack. |
| comb transposed | `X'_j` uses `comb[i][j]`, i.e. column `comb_col + i·n + j`. Easy to swap. | page mapping fixed in the realization table. golden `comb_mode="cyclic"` catches it. |
| packed coefficients | post, comb and P share one tensor: read the tile once and expand from it, never three reads | `cb_coef_raw` holds 1 page when the buffers alias (`coef_source == packed`) |
| bit-identity across modes / norm / feed | the stats grouping must not depend on mode, norm or feed | the chooser's inputs are listed and exclude them. Ports are reserved whenever the fabric is active (Split chooser). |
| determinism | arrival order of partials varies | slot-indexed receives, fixed fold order |
| padded token rows (T % 32) | garbage or NaN in padded rows of the stats/hc | every op is row-local (SFPU lanes = tokens; matmul rows independent; row reduce per row), so nothing crosses rows. Writes cover whole tiles (padding is part of the tile). |
| L1 accumulation reset | `cb_sumsq_acc` must restart each row | use `L1Accumulation::Enabled` (the first pack seeds) on the first unit of each row, then accumulate. Credit handled per Rule 3 (real push/wait between the pack-acc and finalize). |
| matmul after SFPU phases | unpacker/math state left by the SFPU chain | `mm_block_init` per `project_block` (intended once per block) |
| hc scalars cached | generic_op caches programs | scalars are runtime args rewritten each call; `custom_program_hash` covers the core layout only |
| dispatch cost | per-core CoreRanges → DISPATCH-BOUND | merge the core sets into rectangles; keep the per-core RT args minimal |
| feed semaphore leftovers | a trace replays calls; the semaphores must be 0 at entry | every wait subtracts what it consumed (fabric_all_gather pattern); exactly one ready per port per call |
| reserved ports cost cores in fabric-on, feed-off calls | 4 of 110 cores (3.6%) on a DRAM-bound op | accepted, for bit-identity. Measure in the perf pass. |

### Structural impossibilities

None beyond feature_spec.py's INVALID.

## Phase 0 scope (for the implementer)

Phase 0 builds R1 (plain) only:
- reader: coefficient expansion + F/X streaming;
- compute: the mix chain;
- writer: X';
- the split chooser with `has_W = 0` on the worker list;
- max_grid;
- `output_tensor`;
- validate().

The CB set is the Phase 0 subset above. Every knob (`block_cols`, `block_rows`, `in_depth`, `out_depth`, `coef_depth`, R/Q from the chooser) is a named host value passed as a CT or RT arg, never an inlined literal. SUPPORTED is exactly the owner-specified Phase 0 baseline.
