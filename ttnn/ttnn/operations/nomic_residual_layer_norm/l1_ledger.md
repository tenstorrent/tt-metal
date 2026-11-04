# L1 ledger: nomic_residual_layer_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes come from `op_design.md`: `rows` (extent `block_rows`), `hidden` (extent `hidden_tiles`), `stat_slot` (Refinement 1: a "pair tile" carries 2 rows, row j's S in column slot j and its Q in slot 2 + j; `stat_tiles = ceil(block_rows/2)`). Two placements: op_layout (G picked per shape from `HIDDEN_SPLIT_CANDIDATES` = (3, 2) by measured cost, Refinement 2: G = 3 → hidden_tiles = 8, G = 2 → hidden_tiles = 12) and model_layout (G = 8, hidden_tiles = 3, rows_per_shard = bh = ceil(M/11)). fp32_dest_acc_en is on, so every intermediate that carries a statistic or y is Float32.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x` | `block_rows·hidden_tiles` (io: x's shard) | the same: the whole shard | {rows: spans → block_rows, hidden: spans → hidden_tiles, stat_slot: n/a (not a stats buffer)} | bf16 | reader (credits) | compute | call | io tensor, 0 bytes beyond the tensors. It cannot hold y: y needs fp32 (P1), and x's bf16 pages would round it. |
| `cb_residual` | `block_rows·hidden_tiles` (io) | the same | {rows: spans, hidden: spans, stat_slot: n/a} | bf16 | reader (credits) | compute | phase A | io tensor, 0 bytes. Dead after residual_add, but it is the caller's tensor and must not be overwritten. |
| `cb_output` | `rows_per_shard·hidden_tiles` (io: output's shard; x's own shard when in place) | packed block by block | {rows: spans, hidden: spans, stat_slot: n/a} | bf16 | compute | writer (credits only, retired once after the last exchange) | normalize → store | io tensor, 0 bytes. In place (Refinement 1) it is x's own memory: each block's x tiles are consumed (residual add / normalize stage 1) before stage 4 packs over them. |
| `cb_y` | `block_rows·hidden_tiles` | `block_rows·hidden_tiles`: y is read by both sum passes and then by normalize (lookahead: popped after phase A, recomputed from the resident x + r by normalize) | {rows: spans → block_rows, hidden: spans → hidden_tiles, stat_slot: streams (not stored)} | Float32 | compute | compute | phase A → normalize | Owns the normalize intermediates too: stages 1-3 transform it in place (pattern 2, real pop/push credit), so there is no separate intermediate. It cannot alias x or the residual (bf16, caller-owned, and live concurrently with residual_add), nor the output (bf16; in R1 the output is x). |
| `cb_gamma` | `hidden_tiles` | `hidden_tiles` (each re-read for every row of the block) | {rows: streams (broadcast over rows), hidden: spans → hidden_tiles, stat_slot: n/a} | Float32 (the model's TF32-rounded weight; never rounded to bf16) | reader | compute | call | Format differs from `cb_beta` (fp32 against bf16), and both are live for the whole normalize. Capacity is a full tile per hidden tile because the row-bcast unpacker reads a full tile slot; only row 0 is written. |
| `cb_beta` | `hidden_tiles` | `hidden_tiles` | {rows: streams, hidden: spans → hidden_tiles, stat_slot: n/a} | bf16 | reader | compute | call | As `cb_gamma`. |
| `cb_row_partials` | `2·block_rows` | `2·block_rows` (the block's per-row S partial tiles, then its Q partial tiles: the pair collapse needs a row's S and Q in DEST together) | {rows: spans → block_rows, hidden: streams (accumulated in DEST), stat_slot: streams} | Float32, UnpackToDestFp32 | compute | compute | row_sums + row_square_sums → one fused collapse+pack | **Shares its allocation with `cb_rstd_col`**: disjoint lifetimes (dead before the exchange; rstd is born in finalize; with lookahead phase A of block b+1 finishes before block b's finalize). Separate CB indices keep separate pointers, so the partials always walk full cycles (wrap-safe linear indexing). No peer writes into it. |
| `cb_stats_local` | `stat_tiles` (`2·stat_tiles` with lookahead) | the block's pair tiles until the writer has sent them; with lookahead also block b+1's | {rows: streams (packed), hidden: streams, stat_slot: spans → 2·stat_tiles} | Float32 | compute | writer | pack_stats → exchange | It cannot be my own slot of `cb_stats_gather`: that would give the gather CB two producers (compute packs, writer pushes), which breaks the ownership rule. The writer copies it locally into slot c instead. Own allocation (the Phase-0 alias `cb_stats_final` no longer exists, see notes). |
| `cb_stats_gather` | `stat_tiles·G` | `stat_tiles·G` (all contributors before the combine) | {rows: streams, hidden: spans → G contributors, stat_slot: spans → 2·stat_tiles} | Float32, UnpackToDestFp32 | writer | compute | exchange → finalize | **Never shared**: peers write into it remotely at any time after their own phase A, possibly while this core is still in phase A, so any alias that is live before this core's exchange would be corrupted (op_design Risk L3). Largest scratch term at G = 8 (64 KB at M = 128). |
| `cb_center_col` (was `cb_mean_col`) | `block_rows` | `block_rows` col-0 tiles: the row mean (`SEEDED_NORMALIZE` knob, default off: −mean·rstd, the stage-1 seed, then UnpackToDestFp32) | {rows: spans → block_rows, hidden: streams (col-bcast), stat_slot: streams (extracted)} | Float32 | compute | compute | finalize → normalize | Own allocation (`cb_rowstats`, its planned alias, no longer exists: the row collapse is fused into the packing in DEST). It cannot share with `cb_rstd_col`, because both are live through normalize. It cannot share with `cb_row_partials` either, because that already carries `cb_rstd_col`. |
| `cb_rstd_col` | `block_rows` | `block_rows` col-0 tiles | {rows: spans → block_rows, hidden: streams, stat_slot: streams} | Float32 | compute | compute | finalize → normalize | Shares with `cb_row_partials` (above). |

Capacity matches live set for every CB. `cb_y` is exactly full before each in-place stage, which closes the ring (op_design Risk L5).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `rows_per_shard` | op_layout: 1..4 (smallest r with `B·ceil(St/r) <= floor(110/G)`: 36 at G = 3, 55 at G = 2); model_layout: bh = ceil(M/11) in {3..12} | `_plan()`, else SupportRefusal (validate) |
| `block_rows` = largest divisor of `rows_per_shard` that is <= `MAX_BLOCK_ROWS` (default no cap) and whose CBs fit `CB_L1_BUDGET` (256 KB); a single-block result is split into `PIPELINE_BLOCK_ROWS` (2)-row blocks where that gives >= 2 blocks (Refinement 3) | op_layout: = rows_per_shard (one block), except 4-row shards (M 109..128) → 2 × 2; model_layout: 3 / 2 / 5 / 3 / 4 / 3 / 1 / 4 at bh 3 / 4 / 5 / 6 / 8 / 9 / 11 / 12 | `_block_rows()`; blocks are uniform (ring CBs re-walked with linear indices) |
| `hidden_tiles` = 24 / G | 8 (op_layout, G = 3: M 56-72 and M > 110), 12 (op_layout, G = 2: M <= 55 and 73-110), 3 (model_layout, G = 8) | G divides 24; op_layout also even (`_choose_hidden_split`) |
| `stat_tiles` = ceil(block_rows / 2) | 1..3 | `STAT_SLOTS = 2` rows a pair tile |
| lookahead | on iff `LOOKAHEAD` and `num_row_blocks > 1` (op_layout 4-row shards; model_layout bh 4 / 6 / 8 / 9 / 11 / 12) | `_lookahead()` |
| page sizes | fp32 tile 4096 B, bf16 tile 2048 B | TILE layout |

## Total (per core, beyond the io tensors)

```
L1 = 4096·block_rows·hidden_tiles            (cb_y)               -- scales with rows × hidden
   + 4096·hidden_tiles + 2048·hidden_tiles   (cb_gamma, cb_beta)  -- scales with hidden
   + 4096·2·block_rows                       (cb_row_partials ∪ cb_rstd_col)  -- scales with rows
   + 4096·block_rows                         (cb_center_col)                  -- scales with rows
   + 4096·stat_tiles·(2 if lookahead else 1) (cb_stats_local)                 -- scales with stat tiles
   + 4096·stat_tiles·G                       (cb_stats_gather)    -- scales with stat tiles × group size
```

| Placement, shape class | block_rows × blocks | L1 beyond io |
|---|---|---|
| op_layout M = 32..48 (2x512 .. 3x512), G = 2 | 1 × 1 | 48 + 72 + 8 + 4 + 4 + 8 = 144 KB |
| op_layout M = 80..110 (5x512 .. 8x384), G = 2 | 2 × 1 | 96 + 72 + 16 + 8 + 4 + 8 = 204 KB |
| op_layout M = 64 (4x512) | 2 × 1 | 64 + 48 + 16 + 8 + 4 + 12 = 152 KB |
| op_layout M = 109..128 (7x512, 8x512, 11x352), G = 3 | 2 × 2, lookahead (Refinement 3; was 4 × 1 at exactly 256 KB) | 64 + 48 + 16 + 8 + 8 + 12 = 156 KB |
| model_layout M = 32 (bh 3) | 3 × 1 | 36 + 18 + 24 + 12 + 8 + 64 = 162 KB |
| model_layout M = 34..44 (bh 4) | 2 × 2, lookahead (Refinement 3; was 4 × 1, 186 KB) | 24 + 18 + 16 + 8 + 8 + 32 = 106 KB |
| model_layout M = 48 (bh 5) | 5 × 1 | 60 + 18 + 40 + 20 + 12 + 96 = 246 KB |
| model_layout M = 128 (bh 12), budget point | 4 × 3, lookahead | 48 + 18 + 32 + 16 + 16 + 64 = **194 KB (198,656 B) <= 256 KB** |

`circular_buffer_bytes(input_shape, residual, device, placement)` returns this expression for `_plan(input_shape, placement)`; `test_nomic_residual_layer_norm_placements.py::test_cb_budget` checks every M in 32..128 on both placements. The io shards (x, residual, output; output = x in place) come on top, and they are the caller's.

## Data-movement budget (chosen split: `split3_exchange`, M = 128, 96 cores)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 0 | Resident L1 shard consumed zero-copy | 0 |
| residual | 0 | Same | 0 |
| output | 0 | Packed into the core's own shard | 0 |
| gamma | Useful bytes (row 0 of 24 tiles, 3 KB) read 32× (once per row group per third): 96 cores × 1 KB = 96 KB. The tensor's full 96 KB is never read whole. | Reuse-shared operand read per core (stepping stone). Only the 64-B face rows that carry row 0 are read. | 0 |
| beta | Row 0 (1.5 KB) read 32× = 96 cores × 512 B = 48 KB | Same | 0 |
| (S, Q) partials | 0 | Never leave L1 | Each core's 2·stat_tiles·4 KB = 8 KB reaches each of 2 peers. Implemented transport (`EXCHANGE_MODE`, default `serialized_chain`): one fixed-sender Mcast channel per member, handshaked, member-ordered rounds, chain unicast (sender → peer → peer): 16 KB of link traffic a core, about 1.5 MB chip-wide, in one rendezvous of G rounds |

Totals per call at M = 128: DRAM 144 KB (gamma/beta row reads only), NoC 1.5 MB (statistics), activations 0.
(Refinement 1: the (S, Q) payload is now `stat_tiles` pair tiles = 8 KB a contributor at op M = 128, unchanged.)

**model_layout, M = 128 (88 cores, 8 × 11 grid, 3 row blocks a core).** x / residual / output: 0 DRAM crossings
(zero-copy block shards). gamma / beta: each core reads row 0 of its 3 tiles (384 B + 192 B), 88 cores ≈ 50 KB.
Statistics: per block each core's 2 pair tiles (8 KB) reach its 7 column peers. The transport is the
`MODEL_EXCHANGE_MODE` default `serialized_chain`. On a 1 × 8 rectangle the Mcast helper multicasts, so it equals
`serialized_mcast` (measured identical). That is 8 handshaked rounds a block, 3 blocks, ≈ 2.1 MB of multicast payload
chip-wide a call. The lookahead schedule hides the rendezvous behind the next block's phase A (unpack-side gather wait
≈ 1k cycles a call at M = 128, against 18k without it).

> Cheapest-traffic split considered: `whole_rows` (HIDDEN_SPLIT = 1). It removes the 1.5 MB of statistics NoC traffic and the rendezvous, and moves gamma/beta reads to 3 KB + 1.5 KB a core (all 24 tiles' rows on each core). Implemented: `split3_exchange`. Deferred because a built regime already covers every shape it would serve at lower per-core compute: whole rows put 24 / 24 / 48 FPU tiles on a core against 8 / 16 / 32 at M = 32 / 64 / 128. At about 400 cycles a tile that is 2.4 to 4.8 us more than the exchange it saves, and the op is FPU-bound, not traffic-bound. It stays reachable as HIDDEN_SPLIT = 1 (the kernels skip `exchange_block` when G = 1).
>
> Cheapest traffic for the reuse-shared operand: `affine_mcast` (gamma/beta rows read once, 4.5 KB, and multicast). It saves 139 KB of DRAM reads at M = 128. Deferred because the stepping-stone reads finish behind phase A and the exchange at every Phase-0 shape. It layers onto `cb_gamma`/`cb_beta` unchanged.

## Implementation notes (ttnn-implementer, 2026-11-03)

- Removed vs the plan: `cb_reduce_scaler` (no reduce helper remains; its 1.0 tile had no consumer) and
  `cb_rowstats` (the row collapse runs in DEST inside the packing block op). Added: `cb_stats_final`, aliased
  onto `cb_stats_local`. Net: 2 KB less per core.
- `circular_buffer_bytes()` in the op module evaluates this table (`cb_bytes_per_core`).

## Verifier audit (2026-11-03)

- **Currency**: every declared CB has a row; sizes match `cb_bytes_per_core()` and the descriptor (240 KB at M = 128,
  re-derived from the code). No row without a live CB.
- **Capacity vs live set**: no unjustified over-capacity. `cb_gamma` / `cb_beta` hold a full tile slot per hidden
  tile for one valid row (32 KB + 16 KB for 1.5 KB of data): the row-bcast unpacker indexes tiles by page size and
  reads faces 0 and 1, so a page cannot shrink below the face-1 row (offset 1024 B + 64 B for fp32). Accepted.
  `cb_stats_gather` carries 1/8 useful lanes (perf lamp L4); harmless at G = 3 (24 KB), the largest scratch term at
  G = 8 (the `model_layout` plan), so it is folded into Refinement 1 (op_requirements.md).
- **Page format vs DEST**: fp32_dest_acc_en is on; every intermediate is Float32. 16-bit pages are io (`cb_x`,
  `cb_residual`, `cb_output`: output ABI) or a dataflow-only payload (`cb_beta`). No finding.
- **Disjoint lifetimes**: both aliases (`cb_row_partials`|`cb_rstd_col`, `cb_stats_local`|`cb_stats_final`) are
  justified; the second is race-free because the Mcast chain sender barriers its source (local-copy write barrier /
  `SourceL1Guard::Guard` flush) before the writer pushes `cb_stats_gather`.
- **Deferral reason for `affine_mcast` verified by measurement**: removing the gamma/beta reads entirely left the
  device kernel time unchanged (19.07 vs 19.04 us at 8x512, 8.28 vs 8.28 us at 2x512), though NCRISC dropped from
  ~7.1 us to ~0.4 us. The reads are fully shadowed; the op is compute-bound.
- **Block-size default**: sharded → one block = the whole shard, held. Turning `MAX_BLOCK_ROWS` to 2 is correct on
  every shape but measured slower (22.5 vs 19.0 us at 8x512): the compute loop runs block b's normalize before block
  b+1's phase A, so a second block only adds a second exchange round. The default stays.

## Refinement 1 notes (ttnn-implementer, 2026-11-03)

- **model_layout**: same CB inventory at G = 8, 3 hidden tiles; x / residual / output zero-copy on the block shards. A whole
  shard at bh = 12 would need ~522 KB (cb_y 144 KB, gather 192 KB), so the shard runs in uniform row blocks that fit
  the 256 KB budget (4 rows × 3 at M = 128). Prime bh = 11 (M = 112, 121) runs 11 one-row blocks. A ragged tail is
  not wrap-safe for the ring CBs, which several passes a block re-walk with linear indices.
- **Pair tiles** (both placements): a packed stats tile now carries 2 rows (S slots 0-1, Q slots 2-3) instead of 4 rows
  of S or of Q. The finalize extracts every row's mean / rstd in the same DEST window, so **`cb_stats_final` is deleted**
  together with its fp32 round trip. Gather bytes are unchanged (one tile per 2 rows either way). Cost: `cb_row_partials`
  doubles (S and Q partials both resident for the collapse): +16 KB at op M = 128, which lands exactly on the budget.
- **Lookahead** (multi-block plans only): `cb_stats_local` holds two blocks; `cb_y` stays one block because normalize
  recomputes y from the resident x + r. No new alias. Every lifetime claim above was re-derived for the lookahead order
  A0 A1 F0 N0 A2 F1 N1 ...

## Refinement 2 notes (ttnn-implementer, 2026-11-03)

- **Per-shape hidden split** (perf lamp L1): the op_layout plan now picks G ∈ `HIDDEN_SPLIT_CANDIDATES` = (3, 2) by the
  measured `SPLIT_COST_US` table. G = 2 plans (M <= 55: 1 row; M 73..110: 2 rows) keep the same CB inventory at 12 hidden
  tiles: 144 KB / 204 KB, one block. G = 3 plans are unchanged (M = 128 still lands on exactly 256 KB). No CB added.
  G = 4 and 6 were measured or found unbuildable (G = 6 runs out of semaphores for 6 chain-unicast channels) and are
  not candidates.
- **`cb_mean_col` → `cb_center_col`**: same allocation; it holds the mean (default) or −mean·rstd (the
  `SEEDED_NORMALIZE` knob, default off). Under the knob it is UnpackToDestFp32 (only its unary-broadcast seed reads it).
  The seeded normalize needs no new CB: it reads cb_y / x, cb_center_col, cb_rstd_col, cb_gamma and cb_beta as before,
  and stage 2 packs straight into cb_output.
- **Data-movement budget**: unchanged per tensor (x / residual / output 0 DRAM crossings; gamma / beta read once a core
  for its slice: 12 tiles at G = 2). The exchange ships one pair tile per 2 rows to G − 1 peers: fewer bytes at G = 2.

## Refinement 3 notes (ttnn-implementer, 2026-11-03)

- **Pipelined row blocks** (perf lamp L2): `PIPELINE_BLOCK_ROWS` = 2 splits a plan the L1 rule leaves as one block into
  2-row blocks where that gives at least two (op_layout 4-row shards, model bh 4), so the existing lookahead schedule
  runs. No CB added; every CB shrinks with block_rows, and `cb_stats_local` doubles to 2 blocks (pair tile × 2). The op
  layout's budget point moves off the 256 KB edge: M = 128 is 156 KB. The Phase-0 audit's "MAX_BLOCK_ROWS = 2 is
  slower" (22.5 vs 19.0 us) predates the lookahead schedule; with it the same blocks measure 16.9 vs 18.1 us.
- **Not adopted, for L1**: keeping phase A's y for normalize instead of recomputing it under lookahead would need
  `cb_y` two blocks deep plus a block-sized normalize intermediate: 156 + 64 + 64 = 284 KB at op M = 128 > 256 KB.
  The recompute stays; `FUSED_RECOMPUTE` (default off, measured slower) folds it into stage 1 with no new CB.
- **Exchange**: single-round op_layout plans use `concurrent_unicast` (the arrival semaphore, no channel handshake);
  `cb_stats_gather` is unchanged and still never aliased. Multi-round plans keep the serialized chain.
- **Data-movement budget**: unchanged per tensor. 4-row op shards now exchange in 2 rounds of one pair tile instead
  of 1 round of two: the same bytes, one more rendezvous, which the lookahead hides.
