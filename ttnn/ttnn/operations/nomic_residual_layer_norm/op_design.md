# Operation Design: nomic_residual_layer_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (residual add + row LayerNorm + affine), one `ttnn.generic_op` program, cross-core exchange |
| Goal | `LayerNorm(x + residual) * gamma + beta` over H = 768 for every token of a (B, 1, S, 768) bf16 activation that stays in L1, at 32 <= M <= 128 tile rows, on the 11 x 10 Blackhole grid. It must beat the incumbent's 12.0 / 18.7 / 29.7 us (M = 32 / 64 / 128) by at least 1.85x. |
| Math | `y = x + r` (or `y = x`). `mean = S/768`, with `S = sum_j y_j` in fp32. `var = max(Q/768 - mean^2, 0)`, with `Q = sum_j y_j^2` in fp32. `rstd = rsqrt(var + eps)`. `out = (y - mean) * rstd * gamma + beta`, rounded to bf16. |
| Mode | Derivative. It learns from the incumbent's multicast exchange, but its statistics scaling and its per-row rsqrt are replaced. |
| References | `models/experimental/nomic_embed_text_v2_moe/docs/LAYER_NORM.md` §1-6, `docs/BLACKHOLE.md` §2-4 (cost constants); incumbent `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp`, `kernels/compute/layernorm_sharded.cpp`, `kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `sfpu_tile_scope`, `row_reduce_accumulate`, `reduce_accumulate`, `compute_fusion`, `compute_block_size`, `tensix_all_reduce`, `shared_input_reuse`; `eval/golden_tests/nomic_residual_layer_norm/feature_spec.py`. |

### What the design changes against the incumbent (LAYER_NORM.md §2: "6 us plus 2 us a tile row a core")

| Incumbent cost | This design |
|---|---|
| Two cross-core exchanges: E[x], then Var (`reader_mcast_sender_unary_sharded_ln.cpp:313-324`). | One exchange. The fp32 partials `(S, Q)` are all-gathered within each group of 3 cores, and every core combines them itself. |
| A full-tile `rsqrt_tile` and SFPU work per tile row, on every worker (`layernorm_sharded.cpp:544-558`). | Each core packs the statistics of all its tile rows into one tile, then runs mean/var/rstd once per packed tile, not once per tile row (§ Mechanism caps: 4 rows a packed tile). |
| 12 tile rows a core on 88 cores at M = 128. A block-sharded rectangle caps the grid at 11 row groups. | 4 tile rows x 8 hidden tiles a core on 96 cores at M = 128. The ND-sharded layout groups any 3 cores. |
| bf16 `winv` / `cinv` scalers (`layernorm_op_multi_core_sharded.cpp:274-278`). Mean and variance come out 1 + 2^-9 too large. | No scaler carries a statistic scale. 1/768 is applied as an fp32 SFPU scalar multiply. The FPU only sums, with exact 0/1 operands. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor (x) | ttnn.Tensor | yes | (B, 1, S, 768) bf16 TILE, L1, `memory_config() == input_memory_config(shape, device)`, 32 <= M <= 128 with M = B·ceil(S/32), and a Phase-0 plan exists (§ Work Distribution) | — | buffer address via the CB descriptor |
| gamma | ttnn.Tensor | yes | (1, 1, 1, 768) float32 TILE (DRAM interleaved in the model; any interleaved buffer is accepted) | — | RT: buffer address |
| beta | ttnn.Tensor | yes | (1, 1, 1, 768) bfloat16 TILE, interleaved | — | RT: buffer address |
| residual | ttnn.Tensor \| None | no | same shape, dtype and memory config as x | None | CT `HAS_RESIDUAL` |
| epsilon | float | no | > 0 | 1e-5 | RT: fp32 bit pattern |
| output_tensor | ttnn.Tensor \| None | no | Phase 0: None. Refinement R1: x itself (in place) | None | CB backing |
| compute_kernel_config | ttnn.ComputeConfigDescriptor \| None | no | fp32_dest_acc_en must be True | `default_compute_kernel_config()`: HiFi4, fp32_dest_acc_en=True, math_approx_mode=False | compute config. The op adds `UnpackToDestFp32` for the CBs listed in § Circular Buffers. |

Module exports (exact names):

| Export | Contract |
|---|---|
| `input_memory_config(input_shape, device)` | ND-sharded L1 config of § Work Distribution: shard [1, 1, rows_per_shard·32, 32·24/G], on the first `G·num_row_groups` cores of the grid (G = 3 or 2 per shape, Refinement 2) in row-major order, ROW_MAJOR orientation, round-robin. Raises SupportRefusal when no Phase-0 plan exists. |
| `output_memory_config(input_shape, device, input_memory_config)` | Returns `input_memory_config` unchanged. The output layout equals the input layout. |
| `default_compute_kernel_config()` | HiFi4, fp32_dest_acc_en=True, math_approx_mode=False |
| `circular_buffer_bytes(input_shape, residual)` | The closed form of `l1_ledger.md` (§ Total) evaluated for the plan of `input_shape`. `residual` does not change it: x, the residual and the output are io-backed. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | x and the residual are (B, 1, S, 768). gamma and beta are (1, 1, 1, 768). |
| Dtype | x and the residual are bf16. gamma is float32 (the plain weight, TF32-rounded by the model, never scaled by the op). beta is bf16. |
| Layout | TILE (all four) |
| Memory | x and the residual are ND-sharded L1 (`input_memory_config`), and the residual's config equals x's. gamma and beta are interleaved, normally DRAM. |

### Output

| Property | Value |
|----------|-------|
| Shape | = x's shape |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | `output_memory_config(...)` == x's ND-sharded L1 config. Each core writes only its own shard. |

## Blocking Model

The work-split decides everything else. The op's tile index space is (row_group, row, hidden). Batch and sequence fold into a tile-row axis inside each batch.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` (tile rows: B × ceil(S/32), sharded per batch) | **independent**: each token row's statistics and output involve only that row | `rows_per_shard` (rows a core owns) and `block_rows` (= block_extent[rows]) | `rows_per_shard` = the smallest r in {1..4} with `B·ceil(St/r) <= 36`, which is 1 / 2 / 4 at M = 32 / 64 / 128. `block_rows = rows_per_shard` (one block: the whole shard). | host plan `_plan()` (single source). It feeds the shard shape, the CT arg `block_rows` and `num_row_blocks = rows_per_shard / block_rows` (= 1). | Spread over `num_row_groups = B·ceil(St/rows_per_shard)` groups (up to 36): rows of the same group sit on the same 3 cores. | knob-turn: `num_row_blocks` 1 → 2 (perf lamp L2), multi-batch shards (R3) |
| `hidden` (24 tiles of 768) | **dependent**: mean and variance span the whole row, so a cut needs a combine | `hidden_tiles_per_core` (= block_extent[hidden] = 24 / HIDDEN_SPLIT) | 8 (HIDDEN_SPLIT = 3) | host constant `HIDDEN_SPLIT` in the op module → CT arg `hidden_tiles` | Cut in 3 across the 3 cores of a group. One fp32 all-gather of `(S, Q)` partials per block. | knob-turn: HIDDEN_SPLIT ∈ {1, 2, 3, 4, 6, 8}, since the combine exists (perf lamp L1). HIDDEN_SPLIT = 1 skips the exchange. |
| `stat_slot` (tile rows of a block whose statistics share one packed stats tile) | **independent** (a packing axis of the statistics data, re-run per the Step-1 rule for intermediate data) | `stat_slots` = 4 a packed tile (mechanism cap). `stat_tiles = ceil(block_rows / 4)`. | `stat_tiles` = 1 at every Phase-0 shape | CT arg `stat_tiles`, derived by `_plan()` | Each core finalizes its own rows' statistics redundantly with its 2 peers. There is no work to hand out: the cut would need a second round trip, which is rejected (regime `root_reduce`). | — |
| gamma / beta columns (24 tiles, row 0 used) | **reuse-shared**: they do not vary along `rows`, the axis split across groups, so all 32 row groups read the same third | `hidden_tiles_per_core` (same knob) | 8 tiles of row 0 per core | same | Each core reads its own third from DRAM (stepping stone). | scheme-change: one injector + multicast (regime `affine_mcast`, deferred) |

Every stage's own data was put through the same table. The packed statistics (`stat_slot` row) have nothing left to spread: each group computes its own rows' statistics, and the redundant finalize on the 3 peers is what keeps the scheme at one round trip. No stage runs on a designated core while the others wait. The writer's role in the exchange is communication only: every core sends one round, receives two, and computes on its own data.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_y` | `num_row_blocks` in flight | 1 block (the whole shard; T = block_rows·hidden_tiles pages) | Resident y. Every phase-B stage transforms it in place, so its depth is the block, not a stream window. |
| `cb_stats_gather` | rounds in flight | 1 round (G slots × 2·stat_tiles pages) | With `num_row_blocks = 2` (L2) this becomes 2 rounds, so block b+1's partials can land while block b is finalized. |
| `cb_stats_local` | rounds in flight | 1 | Same as above. |
| `cb_row_partials` / `cb_rowstats` | streaming FIFO of row tiles | `block_rows` pages | The S pass and then the Q pass reuse them in sequence (§ L1 ledger). |
| `cb_gamma`, `cb_beta` | none (persistent) | hidden_tiles | Read once a call |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Packed-stats placement without SFPU lane shifts. Column 0 of a row tile lands in lane 0 of the even-parity vectors, and only columns {0, 1, 16, 17} are reachable by vector copy (even/odd parity × face 0-2 / face 1-3). | rows a packed stats tile holds | `stat_slots = 4`. `stat_tiles = ceil(block_rows/4)`. Row k goes to tile k/4, slot k%4 → column {0, 1, 16, 17}[k%4]. | A fifth row overwrites slot 0's column. Wrong mean and rstd come out silently. |
| `VectorMode::C` scoping of the finalize (faces 0 and 2 only) | rows per packed tile when scoped | Scoped only if `block_rows <= 2` (slots in columns 0 and 1). Otherwise use `VectorMode::RC`. | Slots 2 and 3 (columns 16 and 17, faces 1 and 3) are never computed, so rstd stays garbage for those rows. |
| DEST capacity, fp32 half-sync: `DEST_AUTO_LIMIT` = 4 tiles (`dest_helpers.hpp`) | live tiles in finalize | 4 slots: D0 = S sum / mean, D1 = Q sum / var / rstd, D2 = incoming contributor / extracted mean column, D3 = temp / extracted rstd column | Overlapping slots corrupt the statistics. |
| Grid: 110 cores, groups of HIDDEN_SPLIT = 3 | `num_row_groups` | `num_row_groups <= floor(110/3) = 36`. `_plan()` raises its row count up to 4. | More shards than cores: ROUND_ROBIN places 2 shards on a core, which breaks the one-shard-per-core contract of every CB. |
| L1 budget, 256 KB of CBs beyond the io tensors (requirement) | `block_rows · hidden_tiles` | `rows_per_shard <= 4` (T <= 32, total 242 KB, l1_ledger). Shapes with no plan at r <= 4 are refused. | OOM at program creation, or a silent clash with the 256 KB the model keeps live. |
| HIDDEN_SPLIT must divide 24 | `hidden_tiles` | HIDDEN_SPLIT ∈ {1, 2, 3, 4, 6, 8, 12, 24} | Ragged hidden slices: shards no longer uniform. |
| `Semaphore`/flag wait is equality (BLACKHOLE.md §6) | exchange rounds | The Mcast channel's per-round counters are used. Never wait on a cumulative count that two increments could skip past. | Hang |
| DRAM read granularity 64 B, L1 / NoC 16 B (BLACKHOLE.md §4) | gamma/beta row reads | gamma: 64 B at page offsets 0 and 1024 (row 0 of faces 0 and 1, fp32). beta: 64 B at page offsets 0 and 512 (rows 0-1 of faces 0 and 1, bf16; row 0 used). | Misaligned reads return wrong rows. |
| AccumulateViaAdd reduce: inputs must be `BulkWaitBulkPop`/`WaitAndPopPerTile` (`reduce_helpers_compute.hpp:150-151`) | how row sums reach the collapse | The cross-tile sum is done by an eltwise chain over the resident `cb_y` (no pop), then the 1-tile-per-row collapse pops a scratch CB. | `cb_y` gets popped before phase B, and the normalize reads freed pages. |

### Regimes

The minimum is stated at the L1/NoC boundary, since no activation touches DRAM: each activation tensor stays in its own core's L1 (zero NoC bytes); each core reads the gamma/beta bytes it uses once; partial statistics cross the NoC once per peer.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `split3_exchange`: ND-sharded rows × hidden thirds, one fp32 all-gather of packed (S, Q) per block, redundant finalize | **built** | Phase 0 domain: op_layout placement, 32 <= M <= 128, and `_plan()` finds `rows_per_shard <= 4` (every INPUTS shape) | `block_rows × hidden_tiles` = rows_per_shard × 8 | x, r, out: 0 NoC bytes (zero-copy shards). Partials: 2·stat_tiles·4 KB to each of 2 peers. The structural minimum for one combine is 256·block_rows B of statistics; full fp32 tiles are 32× that (lamp L4). gamma/beta: 8 tiles' row 0 per core, re-read by all 32 row groups (successor `affine_mcast`). | One block per shard means one exchange round (rendezvous plus 3 rotating rounds), one finalize per 4 rows, one init/reconfig set per named op, and one compute_kernel_hw_startup a call. |
| `split_model_layout`: the same algorithm on the model's BLOCK_SHARDED COL_MAJOR 8 × C split layout (HIDDEN_SPLIT = 8, hidden_tiles = 3, block_rows = ceil(M/11)) | **deferred**: Phase 0 SUPPORTED is op_layout only. It is refinement R2 and the same kernels: a host plan read off the legacy ShardSpec, group = a grid column of 8, `stat_tiles = ceil(bh/4)` <= 3. Reachable because every kernel quantity is a CT/RT parameter of (G, hidden_tiles, block_rows, stat_tiles, group peers). | `x.memory_config() == model_layout(shape)` | bh × 3 | Same as built, but 7 peers and up to 3 stat tiles | Same |
| `whole_rows` (HIDDEN_SPLIT = 1, height-sharded, no exchange) | **deferred**: the cheapest-traffic split (no cross-core bytes), but `split3_exchange` already covers every shape it would serve at lower per-core FPU work (T_max 24 / 24 / 48 vs 8 / 16 / 32 at M = 32 / 64 / 128). The op is compute-bound, not traffic-bound. Reachable as HIDDEN_SPLIT = 1: kernels skip the exchange when G = 1. | — | rows_per_shard × 24 | Minimum: no exchange | — |
| `per_row_stats` (incumbent: mean/var/rsqrt as one SFPU chain per tile-row tile on every core) | **rejected**: superseded by packed stats. Per row it costs about 3.5k cycles (rsqrt 1064 plus about 10 full-tile SFPU passes at about 243), × 4 rows = about 10 us of the M = 128 budget. | — | — | — | — |
| `two_round_centered` (exchange the mean, then exchange Σ(y−mean)²: the incumbent's numerics) | **rejected**: a second round trip (1-2 us). Superseded by fp32 E[y²] − mean² with a clamp at 0, which meets the offset-row gate (§ Key Risks P2). | — | — | 2× rendezvous | — |
| `chan_local_centered` (one round: each core sends S and its locally-centred M2, and peers combine with Chan's formula) | **deferred**: numerically more stable for rows with mean >> std, at +1 col-bcast sub per tile (+37 cycles) and a local-mean step before Q. Positive reason: the E[y²] form's only extra error is TF32 truncation of y, which every FPU path shares. The estimated var error at a 3σ offset is ≤ 2e-3 relative, inside the 0.9999 PCC / 0.01 RMS offset gate. Reachable: only `row_square_sums_block` and `finalize_stats_block` change. | — | same | same | — |
| `root_reduce` (gather partials to a root, the root finalizes and multicasts mean/rstd back) | **rejected**: two serialized hops on the critical path, and the root idles its peers during the finalize. Superseded by the all-gather plus redundant finalize (one hop). | — | — | — | — |
| `block_sharded_rect` (legacy BLOCK_SHARDED rectangle; group count capped at 11 or 10) | **rejected**: superseded by ND sharding. At M = 128 it forces block_rows = 12 (T = 36 at G = 8, stat_tiles = 3). The 1135-candidate sweep (LAYER_NORM.md §5) mapped this space for the stock program. | — | 12 × 3 | — | — |
| `affine_mcast` (one injector reads row 0 of all 24 gamma and beta tiles, 4.5 KB, and multicasts to the grid) | **deferred**: the stepping stone is built (each core reads its 8 tiles' rows, 16 + 16 reads of 64 B, issued at kernel start and hidden behind phase A and the exchange). Positive reason: the reads finish before the first consumer (normalize) at every Phase-0 shape. The successor layers onto `cb_gamma`/`cb_beta` unchanged. | — | — | Stepping stone: 96 cores × 1.5 KB of DRAM reads. Minimum: 4.5 KB once. | — |

Only one built regime, so there is no selection predicate beyond the support gate. Plan-pinned tests are required instead: every `rows_per_shard` value 1, 2, 3, 4 and a partial last shard (§ Work Distribution).

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core bytes | Per-core FPU tiles (M=32/64/128) | Verdict |
|---|---|---|---|---|
| `whole_rows` (G = 1) | gamma/beta rows only | 0 | 24 / 24 / 48 | Cheapest traffic. Loses on compute by 1.5-3×. Deferred. |
| `split3_exchange` (G = 3) | gamma/beta rows only | 2 peers × 8 KB a core | 8 / 16 / 32 | **Chosen**. One rendezvous. Fewest tiles a core at all three focus shapes under the 110-core, ≤ 4 rows constraint. |
| G = 2 | same | 1 peer × 8 KB | 12 / 24 / 36 | More tiles a core, fewer peers |
| G = 4 | same | 3 peers × 8-16 KB | 12 / 18 / 30 (stat_tiles = 2 at M = 128) | Second finalize tile and more peers |
| G = 6, 8 | same | 5-7 peers | 8 / 15-16 / 30-32 | Exchange fan-in and rows a core grow (block_rows up to 10, stat_tiles 3) |
| Incumbent split, two exchanges | same | 2 rounds | 9 / 18 / 36 | Rejected above |

Split ranking and extent ranking are kept separate. The split (G = 3) is chosen on per-core FPU tiles and one rendezvous. The extent (block_rows = whole shard) is chosen on fixed per-block cost: one exchange round, one finalize, one init set.

### Block schedule

```cpp
// one block = the whole shard in Phase 0 (num_row_blocks_this_core == 1)
for (uint32_t block_idx = 0; block_idx < num_row_blocks_this_core; ++block_idx) {
    load_block(block_idx);              // reader: credits for the resident x / residual shard pages; gamma/beta reads once
    residual_add_block(block_idx);      // y = x + r -> cb_y (fp32). No residual: y is read straight from cb_x
    row_sums_block(block_idx);          // S_k: per-row elementwise sum over hidden tiles (DEST), collapse to col 0, fp32
    pack_sums_block(block_idx);         // S_k col-0 tiles -> packed S tile(s) (slot k%4)
    row_square_sums_block(block_idx);   // Q_k: sum y*y (DEST mul-accumulate), collapse to col 0, fp32
    pack_square_sums_block(block_idx);  // Q_k -> packed Q tile(s)
    exchange_block(block_idx);          // all-gather packed (S, Q) across the 3 cores of the group
    finalize_stats_block(block_idx);    // combine 3 contributors (fixed order), mean, var (clamp 0), rstd; extract per row
    normalize_block(block_idx);         // ((y - mean) * rstd) * gamma + beta -> cb_output (bf16)
    store_block(block_idx);             // writer: wait cb_output (the output is the shard itself)
}
```

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_block` | the shard: block_rows × hidden_tiles pages of x (and r); hidden_tiles gamma/beta tile rows | x, r shards (io), cb_gamma, cb_beta for the whole call | gamma/beta reads once a call, all issued before the reader waits on anything (stall shadow: they overlap phase A and the exchange). A single push of T credits each for cb_x and cb_residual. |
| `residual_add_block` | grid(block_rows, hidden_tiles) | cb_y fills to T pages | 1 init and 1 reconfig a block |
| `row_sums_block` | grid(block_rows, hidden_tiles/2) pairwise adds into one DEST tile per row, then (block_rows, 1) collapse | cb_y (no pop) | 1 init a pass. DEST accumulation per row (PerRow). One SFPU collapse per row. |
| `pack_sums_block` | block_rows col-0 tiles → stat_tiles packed tiles | packed S tile in DEST for the pass | 1 init. One vector copy group per row. |
| `row_square_sums_block` | grid(block_rows, hidden_tiles) mul-accumulate, then (block_rows, 1) collapse | cb_y (no pop) | As row_sums |
| `pack_square_sums_block` | as pack_sums | — | as pack_sums |
| `exchange_block` | 2·stat_tiles fp32 tiles a contributor | — | 1 rendezvous a block: G rotating rounds of one channel, plus a local copy of the own slot |
| `finalize_stats_block` | stat_tiles packed tiles × {S, Q}, 3 contributors | — | Once per packed tile, not per row. Per row: 2 column extractions plus packs. |
| `normalize_block` | grid(block_rows, hidden_tiles) × 4 FPU stages, in place on cb_y, last stage → cb_output | cb_y, cb_mean_col, cb_rstd_col, cb_gamma, cb_beta | 1 init and reconfig per stage a block. The 4 stages are separate chains with an L1 round trip each (catalog `compute_fusion`: DEST reuse loses for FPU consumers). |
| `store_block` | T output pages | — | One `cb_wait_front(cb_output, T)` a block |

Stall-shadow decision: the only wait is `exchange_block`. With one block nothing in the scheme is independent of the statistics. Every normalize stage reads mean or rstd, and lifting `* gamma` ahead of the wait (allowed because the row scale commutes with the column scale) does not remove an FPU op afterwards: it needs `mean ⊗ gamma`, which costs a col-bcast mul per tile. So the window is filled only by what is independent: the gamma/beta DRAM reads and the scaler preparation. The real shadow is the row-block pipeline: at `num_row_blocks = 2`, block 1's phase A runs inside block 0's exchange. It does not change floating-point order. This is perf lamp L2, a knob-turn: the loop above is already per block, and the exchange already carries a round index.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 grid synchronization: HIDDEN_SPLIT | G = 3 comes from the planning model (BLACKHOLE.md per-tile costs). G trades FPU tiles a core against exchange fan-in, and the user requires the layout to be chosen by measurement. | Sweep HIDDEN_SPLIT ∈ {2, 3, 4, 6} at 2x512, 4x512, 8x512 with the device profiler (DEVICE KERNEL DURATION). The layout follows `_plan()`, so the change is one constant. |
| L2 overlap: `num_row_blocks` | One block serializes phase A → exchange → finalize → normalize, and leaves compute idle for the exchange (about 1 us). | `num_row_blocks = 2` (block_rows = 2) at M = 128 and M = 64: exchange round 0 shadowed by phase A of block 1. It costs a second round and a double-buffered cb_stats_gather. |
| L3 finalize realization | The finalize sits on the critical path once per packed tile. A chain of full-tile SFPU passes costs about 3.7k cycles (2.7 us), which breaks the 6.5 us ceiling at M = 32. | The scoped realization (§ API Mapping) against the fused single-pass sfpi finalize over the used vectors only (catalog `sfpu_tile_scope`, `c_skip`). |
| L4 exchange payload | Full fp32 packed tiles (8 KB a contributor) carry 1/8 useful lanes. | Compact the payload to the useful face rows, or pack 16 rows a tile with lane rotates. Measure only if the exchange zone exceeds about 1 us. |
| L5 exchange transport | Rotating-sender rounds serialize the 3 contributors. | All 3 contributors write their peers at once (unicast plus one semaphore increment each) against the rotating `Mcast` channel. |
| L6 normalize fusion | 4 stages each round-trip T fp32 tiles through L1. The catalog says the L1 round trip beats DEST reuse for FPU consumers, but the pack/unpack of fp32 tiles is a quarter of phase B. | Precompute `a = rstd`, `b = −mean·rstd` (SFPU, in finalize), then 2 col-bcast stages plus 2 row-bcast stages, or col and row stages fused per tile in one chain. |

#### Planning cycle budget (BLACKHOLE.md §3 constants at 1350 MHz; a budget to check zones against, not a measurement)

| Term | Per unit | M = 32 (T=8, rows=1) | M = 64 (T=16, rows=2) | M = 128 (T=32, rows=4) |
|---|---|---|---|---|
| Phase A (add 41, pairwise sum about 20, mul-acc 85) | about 150 / tile | 1.2k | 2.4k | 4.8k |
| Collapse, pack and placement | about 260 / row | 0.3k | 0.5k | 1.0k |
| Exchange | about 1.3k / block | 1.3k | 1.3k | 1.3k |
| Finalize (scoped target) | about 1.2k / packed tile | 1.2k | 1.2k | 1.2k |
| Extract per row | about 260 / row | 0.3k | 0.5k | 1.0k |
| Normalize (sub 37, mul 87, mul 85, add 41) | about 250 / tile | 2.0k | 4.0k | 8.0k |
| Kernel fixed intercept | about 1k | 1.0k | 1.0k | 1.0k |
| **Total** | | **7.3k = 5.4 us** (accept 6.5) | **10.9k = 8.1 us** (accept 10.1) | **18.3k = 13.6 us** (accept 16.1) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x, residual shard → compute | bf16 tiles in place in L1 | `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy; pass `core_ranges` explicitly, since an ND tensor has no legacy `shard_spec()`, `tensor_utils.cpp:62`). The reader pushes T credits. | Never re-read over the NoC |
| gamma, beta → `cb_gamma`, `cb_beta` | fp32 / bf16 tile slots, row 0 valid | Reader, `TensorAccessor` page reads with a byte offset: tile `g·8 + j`, 64 B at offsets {0, 1024} (fp32) and {0, 512} (bf16). All 32 reads issued, one barrier, push hidden_tiles. | Issued first, before any wait |
| x + r → `cb_y` | fp32 (Float32 page, FPU reads truncate to TF32) | compute | Single source of y for the sums and the centring. This keeps the constant-row result exact (§ Key Risks P1). |
| row partials → `cb_row_partials` → `cb_rowstats` | fp32, UnpackToDestFp32 | compute | Statistics never pass through srcA/srcB |
| packed partials `cb_stats_local` → peers' `cb_stats_gather` slot c | fp32 tiles | writer: `Mcast` channel over consecutive groups of 3 in row-major core order, rotating sender (round r = group member r sends), `irregular_receiver_set_mode = ChainUnicast` (row-wrapping triples are irregular). Own slot: local L1 copy. Then push `2·stat_tiles·G` pages. | Group member index c = (core linear index) mod 3. It must equal the ND shard's hidden-slice index (§ Key Risks L1). |
| `cb_stats_gather` → finalize | fp32, UnpackToDestFp32 | compute: copy and SFPU add in contributor order 0, 1, 2 | Identical order on all 3 cores, so mean and rstd come out bit-identical across a row's thirds |
| mean, rstd columns → normalize | fp32 col-0 tiles | compute | FPU srcB reads them as TF32 |
| normalize → `cb_output` | bf16 | compute packs into the output shard (zero-copy) | Over x itself in R1 |

TARGET axes as regimes:

| TARGET value | Regime / cost | Class |
|---|---|---|
| placement `op_layout` | `split3_exchange` on the ND layout | built (Phase 0) |
| placement `model_layout` | `split_model_layout`: same kernels, plan from the legacy BLOCK_SHARDED COL_MAJOR spec (G = 8, hidden_tiles = 3, group = grid column, block_rows = bh) | knob-turn, refinement R2. The hidden cut is still across the dependent axis, but the combine already exists. |
| output_mode `new` | cb_output backed on a new tensor of the same config | built |
| output_mode `in_place` | cb_output backed on x. Legal because x is consumed by `residual_add_block` (or by normalize stage 1 when there is no residual) before any output tile of that block is packed, and in that stage tile t is read before tile t is written. | knob-turn, refinement R1 |
| residual_mode `residual` / `no_residual` | CT flag `HAS_RESIDUAL`. With no residual, `residual_add_block` is skipped, sums and squares read `cb_x` directly (bf16 is exact in TF32), and normalize stage 1 reads `cb_x` and writes `cb_y`. | built |
| seq_alignment `tile_aligned` / `seq_padded` | Padding rows inside a tile are rows like any other (row-independent statistics). Nothing is masked. | built |

### What the ops around the norm do with this layout

| Neighbour | Today | With this layout |
|---|---|---|
| out_proj, fc2 (pirotski_matmul), writing x | pack into 12 × 3 block-sharded shards | Pack into ND shards of rows_per_shard × 8 tiles, on the same 96 row-major cores. That is an output block of M rows_per_shard × N 8 tiles, with N = 768 cut in 3. Writes stay local. |
| routed_combine, writing x of MoE norm2 | writes the split layout through TensorAccessor | TensorAccessor supports ND sharding: change its output memory config. |
| ttnn.embedding (emb_ln input) | converted | Still converted. ND shards are uneven-per-batch; embedding needs even shards (LAYER_NORM.md §1.1). |
| QKV, fc1, router (readers of the output); the next norm (residual) | read the split shards | K = 768 is spread over 3 cores per row block. A matmul reads in0 through TensorAccessor (ND-aware) or gathers 3 shards per M block. The next norm takes it as its residual as is (same config). The planned QKV/head-split/rotary op reads it directly. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = block_rows tile rows × hidden_tiles (8) hidden tiles of one batch |
| Grid | the first `3·num_row_groups` cores of the 11 × 10 grid in row-major order (`ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)`). The ND shard grid is [B, 1, ceil(St/rows_per_shard), 3], distributed ROUND_ROBIN in ROW_MAJOR order, so linear shard `(b·RS + s)·3 + g` → core `(b·RS + s)·3 + g`, hidden slice g = core index mod 3. |
| Per-core work | one shard: rows_per_shard tile rows × 8 tiles of x (and r), one block in Phase 0 |
| Remainder | `St = ceil(S/32)` per batch (each image is padded on its own). `RS = ceil(St / rows_per_shard)`. The last shard of a batch is partial (`St mod rows_per_shard` valid rows). Its padding rows are allocated in the shard and processed at the nominal `block_rows`: CB counts stay nominal, results land only in shard padding. |
| Plan (single source `_plan(shape, device)`) | `HIDDEN_SPLIT = 3`; `max_groups = floor(110 / 3) = 36`; `rows_per_shard = min r ∈ {1,2,3,4} with B·ceil(St/r) <= max_groups`, or SupportRefusal; `block_rows = rows_per_shard`; `num_row_blocks = 1`; `stat_tiles = ceil(block_rows/4)`. |

Plan over INPUTS (all supported):

| shape | St | rows_per_shard | row groups | cores | T a core | partial last shard |
|---|---|---|---|---|---|---|
| 2x512, 1x1024, 2x500 | 16, 32, 16 | 1 | 32 | 96 | 8 | no |
| 3x352 | 11 | 1 | 33 | 99 | 8 | no |
| 4x288 | 9 | 1 | 36 | 108 | 8 | no |
| 3x400 | 13 | 2 | 21 | 63 | 16 | yes (1 of 2) |
| 4x352 | 11 | 2 | 24 | 72 | 16 | yes |
| 3x512, 4x512, 8x256 | 16, 16, 8 | 2 | 24, 32, 32 | 72, 96, 96 | 16 | no |
| 5x512, 6x512 | 16 | 3 | 30, 36 | 90, 108 | 24 | yes (1 of 3) |
| 8x352, 8x384 | 11, 12 | 3 | 32, 32 | 96, 96 | 24 | 8x352 yes |
| 7x512, 8x500, 8x512 | 16 | 4 | 28, 32, 32 | 84, 96, 96 | 32 | no |
| 11x352 | 11 | 4 | 33 | 99 | 32 | yes (3 of 4) |

The tests pin rows_per_shard ∈ {1, 2, 3, 4} and a partial shard: 2x512, 4x512, 5x512, 8x512, 3x400, 11x352.

## Circular Buffers

T = block_rows · hidden_tiles. P = stat_tiles. G = HIDDEN_SPLIT. Page sizes: fp32 tile 4096 B, bf16 tile 2048 B.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x` | 0 | 2048 | T (shard) | io-backed (x's shard). Spans rows and hidden. | bf16 | reader (credits) | compute | whole call |
| `cb_residual` | 1 | 2048 | T (shard) | io-backed. Absent when there is no residual. | bf16 | reader (credits) | compute | phase A |
| `cb_gamma` | 2 | 4096 | hidden_tiles | spans hidden. Row 0 valid. | Float32 | reader | compute | whole call |
| `cb_beta` | 3 | 2048 | hidden_tiles | spans hidden. Row 0 valid. | bf16 | reader | compute | whole call |
| `cb_reduce_scaler` | 4 | 2048 | 1 | constant 1.0 tile for the reduce helper API (exact in bf16; never a statistic scale) | bf16 | reader | compute | whole call |
| `cb_y` | 5 | 4096 | T | resident block, spans rows and hidden. Transformed in place by the normalize stages. | Float32 | compute | compute | phase A → normalize |
| `cb_row_partials` | 6 | 4096 | block_rows | one elementwise row-partial tile per row, streams over passes S then Q. Aliases `cb_rstd_col`. | Float32, UnpackToDestFp32 | compute | compute | row_sums / row_square_sums |
| `cb_rowstats` | 7 | 4096 | block_rows | one col-0 tile per row (S, then Q). Aliases `cb_mean_col`. | Float32, UnpackToDestFp32 | compute | compute | collapse → pack_* |
| `cb_stats_local` | 8 | 4096 | 2·P | packed S tiles then packed Q tiles | Float32 | compute | writer | pack_* → exchange |
| `cb_stats_gather` | 9 | 4096 | 2·P·G | G contributor slots × (S, Q); peers write slot c remotely | Float32, UnpackToDestFp32 | writer | compute | exchange → finalize |
| `cb_mean_col` | 10 | 4096 | block_rows | col-0 mean per row (memory shared with `cb_rowstats`) | Float32 | compute | compute | finalize → normalize |
| `cb_rstd_col` | 11 | 4096 | block_rows | col-0 rstd per row (memory shared with `cb_row_partials`) | Float32 | compute | compute | finalize → normalize |
| `cb_output` | 12 | 2048 | T (shard) | io-backed (output's shard, x's when in place) | bf16 | compute | writer | normalize → store |

The CBs with compute as both producer and consumer are intra-kernel scratch: one thread pushes and the same kernel's unpack pops, which is still one producer and one consumer. The aliases are two CB indices on one allocation (a `CBDescriptor` with two format descriptors), and their lifetimes are disjoint (l1_ledger).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | load_block | shard; hidden_tiles | dataflow (TensorAccessor, `prepare_reduce_scaler`) | — | cb_x (T), cb_residual (T), cb_gamma (8), cb_beta (8), cb_reduce_scaler (1) | all resident for the call |
| 2 | residual_add_block | grid(block_rows, 8) | `eltwise_chain` BinaryFpu Add | cb_x (T, pop), cb_residual (T, pop) | cb_y (T) | cb_y full |
| 3 | row_sums_block | grid(block_rows, 4) pairwise, PerRow DEST acc; then (block_rows, 1) collapse | `eltwise_chain` BinaryFpu Add (Strided bases 0 and 4, row stride 8, DestAccumulation::PerRow) + `reduce` AccumulateViaAdd | cb_y (T, waited by caller, no pop) | cb_row_partials (block_rows) → cb_rowstats (block_rows) | cb_y untouched |
| 4 | pack_sums_block | block_rows col-0 tiles → P packed tiles | custom block op (SFPU vector copy) | cb_rowstats (block_rows, pop) | cb_stats_local (P: packed S) | — |
| 5 | row_square_sums_block | grid(block_rows, 8) mul-acc PerRow; (block_rows, 1) collapse | `eltwise_chain` BinaryFpu Mul (cb_y, cb_y, Block, DestAccumulation::PerRow) + `reduce` AccumulateViaAdd | cb_y (no pop) | cb_row_partials → cb_rowstats | — |
| 6 | pack_square_sums_block | as 4 | custom block op | cb_rowstats | cb_stats_local (P: packed Q; 2P total) | cb_stats_local full |
| 7 | exchange_block | 2P tiles × G | `Mcast` channel (rotating sender) + `local_copy_helpers_dataflow` | cb_stats_local (2P, pop) | cb_stats_gather (2P·G) | — |
| 8 | finalize_stats_block | P packed tiles × {S, Q} × G | SFPU: copy (UnpackToDestFp32) + AddBinary / MulUnary / Square / SubBinary / Relu / AddUnary / Rsqrt (scoped, § API Mapping) + custom extract | cb_stats_gather (2P·G, pop) | cb_mean_col (block_rows), cb_rstd_col (block_rows) | — |
| 9 | normalize_block | grid(block_rows, 8) × 4 stages | `eltwise_chain` BinaryFpu Sub/Mul (BroadcastDim::Col), Mul/Add (BroadcastDim::Row) | cb_y (in place, pop/push per tile in stages 1-3, pop in stage 4), cb_mean_col (pop at end), cb_rstd_col (pop at end), cb_gamma / cb_beta (no pop) | cb_y (stages 1-3), cb_output (T, stage 4) | cb_output full |
| 10 | store_block | T | dataflow | cb_output (wait T, pop T) | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (required by every helper) | `compute_kernel_hw_startup(cb_x, cb_residual, cb_y)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h` | once, first statement of MAIN | — | — | — |
| residual_add | helper | `eltwise_chain` + `BinaryFpu<BinaryFpuOp::Add, input(cb_x, Upfront, AtEnd, Block), input(cb_residual, Upfront, AtEnd, Block)>` + `PackTile<output(cb_y)>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:568, 588, 620` | `IterationShape::grid(block_rows, hidden_tiles)` | cb_x, cb_residual | cb_y | block_rows, hidden_tiles |
| row_sums (cross-tile) | helper | `eltwise_chain` + `BinaryFpu<Add, input(cb_y, None, None, Block, …, Strided), input(cb_y, None, None, Block, …, Strided), Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_row_partials, PerOuter, PerOuter)>`. The pairwise-on-one-CB idiom is documented at `reduce_helpers_compute.hpp:211-235`. | `chain.hpp:308 (DestAccumulation), 568` | grid(block_rows, hidden_tiles/2); `StridedTileRange{0, hidden_tiles}` and `{hidden_tiles/2, hidden_tiles}` | cb_y | cb_row_partials | block_rows, hidden_tiles |
| row_sums (collapse) and row_square_sums (collapse) | helper | `reduce<SUM, REDUCE_ROW, cb_row_partials, cb_reduce_scaler, cb_rowstats, BulkWaitBulkPop, INPUT_AND_OUTPUT, Fast, AccumulateViaAdd, ReduceWithinTile::Collapse>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:654-681`; AccumulateViaAdd `:138-165` | `ReduceInputBlockShape::of(block_rows, 1)`. Collapse = `sfpu_reduce` in fp32 (`compute_kernel_api.h:797`), result column 0. | cb_row_partials | cb_rowstats | block_rows |
| row_square_sums (cross-tile) | helper | `eltwise_chain` + `BinaryFpu<Mul, input(cb_y,…,Block), input(cb_y,…,Block), Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_row_partials, PerOuter, PerOuter)>` | `chain.hpp:568` | grid(block_rows, hidden_tiles). Exact TF32 × TF32 products at HiFi4 into fp32 DEST. | cb_y | cb_row_partials | block_rows, hidden_tiles |
| pack_sums / pack_square_sums | raw_api (custom block op, about 16 SFPLOAD/SFPSTORE a row) | Copy `copy_tile` (UnpackToDestFp32) of row k into D2; custom sfpi copies D2's even-parity vectors of faces 0/2 into packed-tile D0 at slot k%4 (column 0: even, faces 0/2; 1: odd, faces 0/2; 16: even, faces 1/3; 17: odd, faces 1/3); pack D0 to cb_stats_local | model: `tt_metal/hw/inc/api/compute/copy_dest_values.h:33` (SFPU_BINARY_CALL dest→dest copy) | slot = k % 4, tile = k / 4 | cb_rowstats | cb_stats_local | block_rows, stat_tiles |
| exchange | helper | host `ttnn.Mcast(device, McastConfig{noc, irregular_receiver_set_mode=ChainUnicast}, core_set, group size G, McastRotatingSenderConfig{}, RowMajor)`; device `McastArgs<…>()`, `sender(noc).send(src, dst, bytes)` at round c, `receiver(noc).receive_and_forward(dst, bytes, round)` otherwise; own slot via `local_copy_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:98` (McastConfig), `:116` (McastRotatingSenderConfig); `mcast/kernel/mcast_args.hpp:19-53,180-181,217-220`; `local_copy_helpers_dataflow.hpp` | dst = gather base + c·2P·4096 | cb_stats_local | cb_stats_gather | stat_tiles, G |
| finalize: combine | helper | `eltwise_chain`: `CopyTile<input(cb_stats_gather, …, Offset)>` into D2 then `AddBinary<D0, D2, D0>` per contributor (fixed order c = 0..G-1), same into D1 for Q | `eltwise/binary/sfpu/basic.hpp:20`; `chain.hpp:554` | — | cb_stats_gather | DEST | G, stat_tiles |
| finalize: statistics | raw_api with vector-mode scoping (helper elements as fallback) | D0 ← S·(1/768): `mul_unary_tile(D0, bits(1/768f), vm)`. D3 ← D0 (`copy_dest_values`) then D3 ← D3² (`square_tile`). D1 ← Q·(1/768). D1 ← D1 − D3 (`sub_binary_tile`). D1 ← max(D1, 0) (`relu_tile`). D1 ← D1 + eps (`add_unary_tile`). D1 ← `rsqrt_tile(D1, vm)`, exact mode. `vm = VectorMode::C` when block_rows <= 2, else `RC`. | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:36,61`; `eltwise_unary/rsqrt.h:45`; `compute_kernel_api.h:284` (square_tile); `eltwise_binary_sfpu.h:131` (sub_binary_tile); `eltwise_unary/relu.h:31` | 1/768 and eps as fp32 bit patterns (CT `INV_HIDDEN_BITS` from host `1.0/768` in fp32; RT eps) | DEST | DEST | stat_tiles |
| finalize: extract | raw_api (custom, inverse of pack_*) | For row k: sfpi copy of slot k%4 of D0 (mean) → column 0 of D2 and of D1 (rstd) → D3; pack D2 → cb_mean_col, D3 → cb_rstd_col | as pack_* | — | DEST | cb_mean_col, cb_rstd_col | block_rows |
| normalize | helper | 4 × `eltwise_chain` over grid(block_rows, hidden_tiles): (1) `BinaryFpu<Sub, input(cb_y), input(cb_mean_col, BroadcastDim::Col, …, Col mapping)>` → `PackTile<cb_y>`; (2) `BinaryFpu<Mul, cb_y, input(cb_rstd_col, BroadcastDim::Col, …, Col)>` → cb_y; (3) `BinaryFpu<Mul, cb_y, input(cb_gamma, BroadcastDim::Row, Upfront, None, Row)>` → cb_y; (4) `BinaryFpu<Add, cb_y, input(cb_beta, BroadcastDim::Row, Upfront, None, Row)>` → `PackTile<cb_output>` | `chain.hpp:330 (BroadcastDim), 568, 588` | No residual: stage 1's A input is cb_x | cb_y, cb_x, cb_mean_col, cb_rstd_col, cb_gamma, cb_beta | cb_y, cb_output | block_rows, hidden_tiles |
| reader: scaler | helper | `prepare_reduce_scaler<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_reduce_scaler, 1.0f)` (pool-type-aware overload) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:70` | value 1.0 | — | cb_reduce_scaler | — |
| reader: gamma/beta | raw_api | `TensorAccessor` `get_noc_addr(page, offset)` + `noc_async_read` of 64 B, one barrier | `tech_reports/tensor_accessor/tensor_accessor.md` | pages g·hidden_tiles + j | — | cb_gamma, cb_beta | hidden_tiles |

**Helpers considered and rejected** (one per raw_api row):

| raw_api row | Candidate | File:Line of mismatch | Concrete reason |
|---|---|---|---|
| pack_* / extract | `copy_dest_values` / `CopyDest` | `copy_dest_values.h:33-43` (`VectorMode::RC`, same positions in and out) | It copies a whole tile position for position. Moving column 0 to column {1, 16, 17} needs a different source and destination vector offset, which neither helper exposes. Gap to close: a `copy_dest_vectors(src, dst, src_vec_offset, dst_vec_offset, mask)` block helper. |
| pack_* | FPU matmul with 0/1 selector tiles | (semantic) | It routes the fp32 statistic through srcA/srcB, truncating it to TF32 (2^-11), which violates "statistics fp32 end to end". |
| finalize: statistics | chain elements `MulUnary` / `Rsqrt` / `AddUnary` | `ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.inl:65` (`rsqrt_tile` called with the default `VectorMode::RC`); `scalar.hpp:26,32` (no vector-mode parameter) | Each element is a full-tile SFPU pass: about 3.7k cycles a packed tile, against a 1.2k budget, and it breaks the M = 32 ceiling. The compute API takes `vector_mode` (`rsqrt.h:45`, `binop_with_scalar.h:36,61`), so scoped calls are used. The binary steps (`AddBinary`, `SubBinary`) stay helpers. Bring-up may use the full chain first. |
| reader gamma/beta | `local_copy_helpers_dataflow`, a whole-page reader | — | Only 64 B of each page is needed. Whole-tile reads move 32× the bytes from banks that all 96 cores share (LAYER_NORM.md §1.3: tiled weights cost 0.48 ms a forward). |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| residual_add | Add | cb_x All | cb_residual All | None |
| row_sums | Add (DEST acc) | cb_y All | cb_y All | None |
| row_square_sums | Mul (DEST acc) | cb_y All | cb_y All | None |
| normalize 1 | Sub | cb_y All | cb_mean_col Col0 | Col |
| normalize 2 | Mul | cb_y All | cb_rstd_col Col0 | Col |
| normalize 3 | Mul | cb_y All | cb_gamma Row0 | Row |
| normalize 4 | Add | cb_y All | cb_beta Row0 | Row |

## Registry plan (for the implementer's SUPPORTED)

| Axis | Phase 0 SUPPORTED | Refinement |
|---|---|---|
| dtype | bfloat16 | — |
| fp32_dest_acc_en | True | — |
| layout | TILE_LAYOUT | — |
| placement | op_layout | R2: model_layout (knob-turn) |
| residual_mode | residual, no_residual | — |
| output_mode | new | R1: in_place (knob-turn: cb_output backed on x) |
| seq_alignment | tile_aligned, seq_padded | — |

The shape gate inside `validate()`: M in [32, 128], last dim 768, and `_plan()` succeeds. Otherwise SupportRefusal. R3 (knob-turn): multi-batch shards and a per-shape HIDDEN_SPLIT for shapes with more than 36 row groups at 4 rows. INPUT_TAGGERS: `tag_seq_alignment` → "tile_aligned" if S % 32 == 0 else "seq_padded". No index axes.

Structural impossibilities: none beyond feature_spec.py's empty INVALID.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| P1 constant row ≠ beta | Output = beta only if `y − mean` is exactly 0. That needs the mean to equal TF32(y) exactly, and every consumer to see y at the same precision. | Single fp32 `cb_y`, read only through the FPU (TF32 truncation, identical everywhere). The sums of identical TF32 values are exact (≤ 21 significant bits). The mean is `S × fl(1/768)` on the SFPU in fp32, which rounds back to c' (fl(1/768) = (1/768)(1+2^-25)). The variance is clamped at 0 before eps, so rstd stays finite. Never compute the mean from a recomputed `x + r`. |
| P2 E[y²] − mean² cancellation on offset rows | At a 3σ offset, E[y²] = 10σ². | Q is accumulated from exact TF32 products in fp32 DEST and collapsed on the SFPU in fp32. It is never packed through srcA. The remaining error is TF32 truncation of y, shared by every FPU path. Fallback: regime `chan_local_centered`. |
| P3 TF32 truncation bias in normalize | 4 in-place stages each re-read fp32 through srcA (truncation): about 0.1% systematic shrink of \|out\| | Inside the 0.005 RMS gate (PCC is unaffected). L6 fusion reduces the stage count. |
| P4 thirds of a row disagree | Each core finalizes its own copy. A different summation order changes the last bits of mean and rstd across the 3 hidden thirds. | Combine in contributor order 0..G−1 on every core, own slot included in its place (local copy into slot c), never "own first". |
| L1 shard-to-core order vs group order | The ND ROUND_ROBIN shard order, the `Mcast` group partition and the kernel's slice index g must all agree: core linear index i holds hidden slice i mod 3 of row group i / 3. | Build every one of them from one ordered core list (`_plan()`). Required test: an iota tensor read back per core (or a 3-slice-distinct input) to prove slice index = i mod 3. The probe of 2026-11-03 confirmed round trip and config equality only. |
| L2 `cb_descriptor_from_sharded_tensor` on ND tensors | It defaults `core_ranges` to `shard_spec()->grid`, an empty optional for ND (`tensor_utils.cpp:62`). | Always pass `core_ranges` = the plan's core set. The probe confirmed `total_size` = rows_per_shard·8·2048. |
| L3 peers writing my gather slot early | A fast peer finishes phase A and sends while I am still in phase A. | `cb_stats_gather` is never aliased (l1_ledger), so early writes land in dedicated memory. Its CB credit is pushed only after all rounds complete. |
| L4 irregular groups and hardware multicast | Triples that wrap a grid row are not rectangles. A rectangle multicast would write into other groups' gather slots. | `irregular_receiver_set_mode = ChainUnicast`. Required test: shapes with 99 and 108 cores (3x352, 4x288), where triples wrap. |
| L5 in-place streaming on cb_y | Stages 1-3 pop tile t and push its transform into the same slot. That is valid only because cb_y is exactly full (T pages, ring closed). | Push and pop counts stay nominal T per stage. Never size cb_y above T. |
| L6 partial shards | The last shard of a batch carries allocated padding rows with garbage. | Processed at nominal block_rows: statistics are per row and lane, so garbage stays in padding rows, which no consumer reads. Test 3x400 and 11x352. |
| L7 memory-config equality | The golden harness compares `x.memory_config() == input_memory_config(shape, device)` (axes.py `_placement`). | Construct the config once in `_plan()` and return the same object form. The probe confirmed `from_torch` keeps it equal (layout `ND_SHARDED`, no legacy spec). |
| L8 fp32 DEST, half-sync | DEST_AUTO_LIMIT = 4. The finalize uses exactly D0-D3. | Do not add a 5th live tile. The extraction reuses D2 and D3 after the combine. |
