# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`
→ Blocking Model): **row** (extent `block_row_tiles`) and **W** (extent = the core's whole share
`core_w_tiles`; RM pads it to `core_w_tiles_padded`). `rm_chunk_w_tiles` is a streaming quantum
along W, not a block extent. Page sizes: T_in / T_out / T_g = tile bytes (bytes) of the input /
output / gamma dtype (bf16 2048 B, fp32 4096 B, bfloat8_b 1088 B); T_acc = tile bytes of the DEST-width
intermediate format (`acc_dtype()`): Float32 (4096 B) when `fp32_dest_acc_en = True`, Float16_b (2048 B)
when False (Refinement 1) — with 16-bit DEST the value is rounded before `pack_tile()`, so a Float32 page
would carry no extra bits while doubling packer/unpacker/NoC bytes and L1.

## Inventory decisions (before any solve)

| Candidate buffer | Decision | Rule |
|---|---|---|
| x² scratch (block or chunk sized, fp32) | **eliminated** — pass 1 multiplies x·x with DEST accumulation per tile-row and packs one tile per row | Rule 3 pattern 4 (fold into the accumulator) |
| reduce scaler / 0-1 mask tile | **eliminated** — the reader zero-fills W padding in L1, and AccumulateViaAdd needs no scaler | Rule 1 |
| `x · inv_rms` intermediate before the gamma multiply | **eliminated** — one chain: `BinaryFpu` (bcast Col) → `DestReuseBinary` (gamma) → pack into `cb_out` | Rule 3 pattern 1 (pack into the destination) |
| second read of x for pass 2 | **eliminated** — cb_x stays resident across both passes | residency (one DRAM crossing) |
| `cb_out` for TILE path | kept as a small streaming window — the output is interleaved in DRAM, so there is no destination CB to pack into (zero-copy only for a sharded output, deferred) | Rule 3 pattern 1 not applicable |

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm` (RM only) | `rm_in_depth · rm_chunk_w_tiles` (= 2·chunk) | `rm_chunk_w_tiles` (one 32-stick × chunk piece being tilized) | `{row: streams → 1 tile-row, W: streams → rm_chunk_w_tiles}` | input dtype (Float16_b / Float32 — raw input, not an accumulation) | reader | compute | per (tile-row, chunk) | none: live concurrently with `cb_out_rm` (reader of block b+1 runs while the writer drains block b); capacity 2× live set = double buffering (reader one chunk ahead) |
| `cb_x` | TILE: `x_block_depth · block_row_tiles · slice_w_tiles`; RM: `block_row_tiles · core_w_tiles_padded` (x_block_depth = 1) | `block_rows_this · core_w_tiles` (whole block resident across pass 1 and pass 2); with lookahead two blocks | `{row: spans → block_row_tiles, W: spans → core_w_tiles (≤ slice_w_tiles)}` | input dtype (raw input) | reader (TILE) / compute tilize (RM) | compute | load → end of `normalize_block` | none: live through every compute phase of the block; capacity > live set only by `x_block_depth = 2` (next block in flight / lookahead) and, for the last W member, `slice_w_tiles − core_w_tiles` (one capacity value for all cores) |
| `cb_x_alt` (implementation, TILE only, `x_block_depth == 2`) | `block_row_tiles · x_row_pitch` | the other block of the depth-2 pair | as `cb_x` | input dtype | reader | compute | blocks of odd parity | none: this IS the second half of `cb_x`'s depth-2 budget, split into its own CB because indexed CB reads cannot wrap a ring (LLK `cb_access_within_bounds`) and the combine lookahead reads block b+1 while block b is resident. Footprint unchanged: `d·b·S_x·T_in` total. |
| `cb_gamma` | resident: `slice_w_tiles` (RM: padded); streamed (R4, `gamma_chunk = G > 0`): `GAMMA_STREAM_DEPTH · G` (= 2·G; RM: G = `rm_chunk_w_tiles`) | resident: `core_w_tiles`; streamed: one G-tile chunk (the next one in flight) | `{row: absent (gamma has no row extent; reused by every row), W: spans → core_w_tiles}` | gamma dtype (Float16_b / Float32 — raw weight); Perf 1: page = `T_g'` = [bfp exponent section +] faces 0/1 only (half a tile) — compute reads gamma ROW-broadcast, so row 0 of faces 0/1 is all it unpacks and the writer no longer expands | writer (R3; was reader) | compute | program | none: program lifetime, read by every block's `normalize_block` |
| `cb_sumsq_tiles` | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: reduced away (DEST accumulation)}` | T_acc format: Float32 / Float16_b per `fp32_dest_acc_en` (DEST accumulation result) | compute | compute | pass 1 → collapse | not with `cb_inv_rms` / `cb_partial_sumsq`: the collapse reduce reads this CB while writing those (concurrent). Same size and format as `cb_inv_rms`; in R2 their lifetimes are disjoint, but R1 overlaps them, and the saving is `block_row_tiles·4 KB` (< 1% of the footprint) — one design for both regimes kept |
| `cb_partial_sumsq` (k > 1) | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: reduced away}` | T_acc format | compute | writer | collapse → exchange (the gathered NoC payload follows `get_tile_size`) | not with `cb_sumsq_tiles` (concurrent in the collapse); not with `cb_gathered_sumsq` (peers write into that one at any time during the previous block) |
| `cb_gathered_sumsq` (k > 1) | symmetric: `gather_depth · w_split_cores · block_row_tiles` (`gather_depth` = 1 when every core runs one block, else 2 — Refinement 2); owner topology: `ceil(block_row_tiles / k) · k` | symmetric: `w_split_cores · block_rows_this` (one block); capacity 2× only for multi-block = mechanism cap (lockstep slot reuse). Owner: k partials per owned row, uniform count per member (lockstep) | `{row: spans → block_row_tiles, W: reduced to one partial per member → w_split_cores}` | T_acc format | writer (+ peer NoC writes landing in its reserved slot) | compute | exchange → combine | none: remote writes may land at any time during the previous block, so no other buffer may occupy these bytes |
| `cb_inv_rms` | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: absent (broadcast along W)}` | T_acc format (finalize result in DEST) | compute | compute | finalize → end of `normalize_block` | not with `cb_sumsq_tiles` (see that row) |
| `cb_out` | TILE: `out_depth · out_unit_tiles` (= 2·min(slice_w_tiles, 16)); RM: `block_row_tiles · rm_chunk_w_tiles` | TILE: ≥ 1 tile (streaming window); RM: `block_rows_this · rm_chunk_w_tiles` (pass 2 of one chunk column completes before untilize, same thread) | TILE `{row: streams → 1 tile, W: streams → 1 tile}`; RM `{row: spans → block_row_tiles, W: streams → rm_chunk_w_tiles}` | output dtype (terminal pack to the output format, not an intermediate) | compute | writer (TILE) / compute (RM) | per tile / per chunk | TILE: none, live with every other per-block CB; capacity > live set = double buffering for the writer. RM: not with `cb_x_rm` (concurrent across the reader/compute pipeline) |
| `cb_out_rm` (RM only) | `out_depth · rm_chunk_w_tiles` | `rm_chunk_w_tiles` | `{row: streams → 1 tile-row, W: streams → rm_chunk_w_tiles}` | output dtype | compute | writer | per (tile-row, chunk) | not with `cb_x_rm` (concurrent); capacity 2× = writer double buffering |

| `cb_inv_owned` (owner topology only — sharded k > 1, interleaved k > 8) | `ceil(block_row_tiles / k)` | `owned_rows_this` (≤ capacity) | `{row: spans → rows owned by this member (row r → member r % k), W: absent}` | T_acc format (finalize result) | compute (combine) | writer (returns each tile to all k members' `cb_inv_rms`) | combine → return | not with `cb_inv_rms`: remote owners write into `cb_inv_rms` while this member's own combine output is still being sent; members own different counts per block, so the writer reads it one tile at a time from the front (no linear multi-tile walk across the ring end) |
| `cb_tail_mask` (sharded TILE, W % 32 ≠ 0) | 1 | 1 (tail member only) | `{row: absent, W: the tail tile's valid columns}` | T_acc format (Float16_b / Float32 — the mask formats `prepare_reduce_mask` accepts; matches the masked operand) | reader (`prepare_reduce_mask`) | compute (pass 1 seeds each tail tile-row with `x_tail * mask` (bcast row 0) before the DEST accumulation — Perf 1; popped once at kernel end) | program | none: program lifetime, read on every block |
| `cb_ones` (Perf 1) | 1 | 1 | `{row: absent, W: absent}` (the 1.0 reduce scaler) | Float16_b | reader (`calculate_and_prepare_reduce_scaler`, after block 0 of x is published) | compute (FPU row collapse `reduce<…, ReduceTile>`) | program | none: program lifetime |

Semaphore `sem_partials_arrived` (k > 1): one 4-byte L1 word, program lifetime. Semaphore `sem_inv_arrived`
(owner topology only): one 4-byte word, program lifetime.

### Refinement 2 — WIDTH_SHARDED / BLOCK_SHARDED placement

- **`cb_x` (TILE)**: backed on the input shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy). Its
  capacity is the shard (`shard_tile_rows · shard_w_tiles` pages) and it is NOT in the CB budget: the shard is
  a tensor buffer allocated top-down, and `cb_l1_budget()` already measures the free region below it.
  `x_row_pitch` = shard width in tiles. The reader only publishes the valid tile-rows (one reserve/push); the
  caller's shard is never written (the W tail is masked in compute via `cb_tail_mask` + `cb_sumsq_acc`
  instead of the interleaved path's in-place zero fill). `cb_x_alt` is never declared (x_block_depth = 1).
- **`cb_out` (TILE)**: backed on the output shard. Compute packs straight into it, one shard row (`x_row_pitch`
  tiles) per tile-row; the writer only waits/pops per block. Padding tiles of a ragged last shard column are
  normalized as don't-care (their output lands in shard padding).
- **ROW_MAJOR shards**: shard widths are 16 B-granular (8 bf16 datums), not tile-aligned, so a stick shard is
  not a tilize-ready page layout. The reader copies each tile-row's valid stick pieces locally (own-L1 NoC
  reads — no accessor, no cross-core traffic) into the existing `cb_x_rm` staging, zero-fills to the tile edge
  (W tail + shard padding), and the writer copies the untilized pieces back into the local output shard.
  CB inventory = the interleaved RM rows (`cb_x_rm`, `cb_x`, `cb_out`, `cb_out_rm`).
- **Block**: the whole shard (`block_row_tiles` = shard tile-rows), sub-blocked along rows only when the
  CB plan (above, minus the shard-backed CBs) does not fit the free L1 left by the input + output shards.
- **Combine (sharded)**: owner topology for every k > 1 (`K_SYM_MAX_SHARDED = 1`). L1 per member:
  `ceil(b/k)·k` gathered + `ceil(b/k)` owned pages, i.e. ≈ `b + k` instead of the symmetric `k·b`
  (k = 128, b = 1: 128 pages, vs. 256 pages for the depth-2 symmetric all-gather).

### Refinement 4 — HEIGHT_SHARDED placement (k = 1) and the wide-W fallbacks

- **Placement**: one shard row per Tensix core in the shard grid's walk order (`_ordered_shard_cores`, ROW_MAJOR /
  COL_MAJOR, multi-range grids from `auto_shard_config`: full grid rows + a partial last row), k = 1, no combine
  CBs, no semaphores. TILE: `cb_x` / `cb_out` backed on the shards exactly as in R2 (zero-copy, not in the budget);
  RM: local stick copy into `cb_x_rm` + tilize, untilized pieces copied back into the local output shard.
- **Gamma**: every Tensix core needs the whole gamma, so the R3 k = 1 multicast (`GAMMA_MCAST_SHARDED_K1`) applies
  — one DRAM read of gamma per launch. Measured BH p150b, perf config (bf16, 16-bit DEST, HiFi2), per-core load →
  multicast: TILE (4096,4096) 92.6 → 54.8 µs, (8192,1024) 29.0 → 20.0 µs, (7136,736) 20.8 → 14.8 µs; RM 106.8 → 79.7,
  32.9 → 27.2, 24.2 → 19.7 µs; (99991,64) flat (2-tile gamma).
- **Wide W beside two resident shards** (HEIGHT holds whole rows, so no W split can relieve L1): the budget left
  after the input + output shards can be < the resident gamma (TILE W = 8192 bf16: 412 KB left, gamma 512 KB) or
  < the RM tilized row + gamma + three C-sized staging CBs. Regimes, in preference order (`_sharded_plan_candidates`):
  1. resident gamma, RM chunk cap 32 (default; every shape that fit before R4 keeps it);
  2. TILE: gamma streamed — `cb_gamma` = 2 chunks of the largest G ≤ 16 that fits; the writer re-streams gamma per
     block (whole block's chunks at once — the shard-backed `cb_out` never blocks compute);
  3. RM: per halved chunk cap C: resident gamma, then streamed gamma with G = C (pass 2 already walks C-tile
     windows). The writer prefetches `GAMMA_STREAM_DEPTH` chunks per block and refills chunk c + depth after
     draining chunk c's sticks — pushing all chunks first deadlocks (compute blocks on `cb_out_rm`, the writer on
     the gamma ring). The tilized x row (`cb_x`) stays resident: x is still read from the shard once per pass pair.
  Streamed gamma costs one gamma read per (block, Tensix core) instead of one multicast per launch; only cells
  that did not fit at all take it. Golden HEIGHT probe (843 shape × layout × dtype × gamma configs): 54 did not
  fit before the fallbacks, 0 after. 18 more are unbuildable by construction (fp32 TILE W = 8192: a 1 MB input
  shard + a 1 MB output shard exceed L1; the golden harness skips them).
- **Data movement (HEIGHT)**: x and out cross DRAM **0** times; gamma once per launch (multicast), or once per
  block per Tensix core when streamed. No cross-core x/partial traffic.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `block_row_tiles` | `1 ≤ block_row_tiles ≤ core_row_tiles_max` | `select_blocking()`: largest value with `footprint ≤ budget` (whole assignment at depth 1, else depth 2) |
| `slice_w_tiles` | `ceil(tensor_w_tiles / w_split_cores)`; `footprint(k, 1, 1) ≤ budget` | `k_fit` (smallest k ≤ K_CAP that fits); RuntimeError otherwise |
| `core_w_tiles` | `≤ slice_w_tiles`; ≥ 1 | k normalization `k = ceil(tensor_w_tiles / slice_w_tiles)` |
| `core_w_tiles_padded` | `ceil(core_w_tiles / rm_chunk_w_tiles) · rm_chunk_w_tiles < core_w_tiles + chunks` | balanced chunking |
| `w_split_cores` (k) | interleaved `1 ≤ k ≤ K_CAP = min(K_CAP_MAX, grid cores)` (Refinement 5; owner combine for k > 8 — its `cb_gathered_sumsq` = `ceil(b/k)·k` pages is checked against the budget per candidate k in `_choose_k`); sharded k = shard columns (≤ grid cores, up to 128 on BH) | clamp in `select_blocking()`; `select_sharded_blocking()` from the shard spec |
| `rm_chunk_w_tiles` | `≤ rm_chunk_cap` (32; sharded RM halves the cap 32→16→…→1 until one tile-row fits — R4) | balanced chunk formula `_rm_chunk(slice_w, cap)` |
| `gamma_chunk` (G) | 0 (resident) or `1 ≤ G ≤ 16` (TILE) / `G = rm_chunk_w_tiles` (RM) | sharded only: `_sharded_plan_candidates()` — first regime whose one-tile-row plan fits (resident gamma preferred) |
| `x_block_depth` | ∈ {1, 2} | 2 iff `num_blocks_this_core > 1` and it fits; RM: 1 |
| `out_depth`, `rm_in_depth` | 2 | constants (Phase 0) |
| `out_unit_tiles` | `min(slice_w_tiles, 16)` | constant cap |
| `budget` | free L1 between the allocator base and the lowest allocated L1 buffer (BH p150b, empty device: 1 461 120 B; L1-resident input/output tensors lower it) | `cb_l1_budget(device)` reads `ttnn.get_memory_view(device, L1)` on the host (verifier fix: was `get_max_worker_l1_unreserved_size()`, which starts at the kernel-config ring buffer and overshoots the CB region by ~69 KB on BH) |

## Total per-core footprint

```
footprint(k, b, d) =
    d · b · S_x · T_in                                  # cb_x          (scales with block_row_tiles, slice, depth)
  + has_gamma · S_x · T_g'                              # cb_gamma      (Perf 1: T_g' = faces 0/1 page ≈ T_g / 2)
  + b · T_acc                                             # cb_sumsq_tiles
  + b · T_acc                                             # cb_inv_rms
  + [k > 1] · (b · T_acc + 2 · k · b · T_acc)               # cb_partial_sumsq + cb_gathered_sumsq (scale with k·b)
  + TILE: 2 · min(slice_w_tiles, 16) · T_out            # cb_out (bounded constant)
  + RM:   2 · C · T_in + b · C · T_out + 2 · C · T_out  # cb_x_rm, cb_out, cb_out_rm (C = rm_chunk_w_tiles ≤ 32)
  + 2 KB                                                # cb_ones (Perf 1)
Perf 1 combine pipeline (sharded TILE owner combine, `pipe_nb` sub-blocks of s tile-rows; b := s): cb_inv_rms
min(3, pipe_nb) · s · T_acc, cb_partial_sumsq 2 · s · T_acc, cb_gathered_sumsq 2 · ceil(s / k) · k · T_acc — never
above the single-block plan (s ≤ ceil(crt / 2)).
where S_x = slice_w_tiles (TILE) or slice_w_tiles padded to a multiple of C (RM).
```

Dominant terms: `d·b·S_x·T_in` and `S_x·T_g`; both fall as `1/k`, which is how `k_fit` restores fit
for wide W. Worked check (BH, bf16, gamma bf16, TILE): W = 32768 (tensor_w_tiles 1024), 1 tile-row →
k = 8, S_x = 128: 256 KB + 256 KB + 8 KB + 72 KB (combine) + 64 KB ≈ 0.66 MB. W = 8192, 8192 rows
(tensor_row_tiles 256) → k = 1, b = 1, d = 1: 512 KB + 512 KB + 8 KB + 64 KB ≈ 1.1 MB.

## Data-movement budget

Chosen split: row-groups over the full grid × W split k (k = 1 in R1).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 | cb_x keeps the whole block (row × W share) resident across pass 1 and pass 2 | 0 |
| out | 1 | written once from cb_out / cb_out_rm | 0 |
| gamma | interleaved (R3, `GAMMA_MCAST`): 1 copy per multicast group — k = 1: one copy total; k > 1: one per grid column (`gpr·k` slices ≈ grid_x); per-core fallback (partial grid row of row-groups, k > 1): `G/k` copies. TILE gamma: 128 B per tile (row 0 only); RM: slice bytes | gamma is reuse-shared along the row split; the group's sender reads + expands its slice once and multicasts the expanded tiles; every core keeps it for the program | expanded slice (`core_w_tiles · T_g`) multicast once per group |
| sumsq partials (k > 1 only) | 0 | never leave L1 | symmetric (interleaved k ≤ 8): per block per group k² · block_rows_this partials; owner topology (sharded, interleaved k > 8): k · block_rows_this partials in + k · block_rows_this inv_rms tiles back. Payload per tile: faces 0 and 2 only (half a tile — column 0 lives there; `PARTIAL_FACES_ONLY`) |

Sharded (Refinement 2): x and out cross DRAM **0** times (resident shards, consumed and produced in place);
gamma is read once per member (its W slice). NoC: owner gather + return, 2 · k · tile-rows half-tiles per group.

Totals per launch: DRAM ≈ `2 · |x|` + `(G/k) · W · elem_bytes` (gamma, ≤ 1/32 of x per row-group);
NoC (R2 only) ≈ `k² · tensor_row_tiles · 4 KB`.

> Cheapest-traffic split considered: R1 (row-groups only, W resident) — on DRAM it ties with R2 for
> x and out and reads gamma k× more; R2 adds `k² · tensor_row_tiles · 4 KB` of NoC traffic. Implemented:
> both, selected by `select_blocking()`: R1 whenever k = 1 fits L1 and tensor_row_tiles > G/8 (it is
> then also the cheapest); R2 only where R1 cannot fill the grid (tensor_row_tiles ≤ G/8) or cannot keep
> the slice resident (W too wide), i.e. where the alternative to R2 is idle Tensix cores or a second DRAM
> crossing of x. The two-pass re-stream (+1 crossing of x) is `rejected`; gamma multicast
> (−(G/k − 1)·W bytes) was `deferred` because r ≤ 1/32 is in the catalog's measured-null band — **landed in
> Refinement 3**: the band is about bandwidth, but the per-core TILE-gamma load (2 × 64 B reads per tile on every
> Tensix core, before the first x read) was transaction-bound and serial: 131 µs of 771 µs at (8192,7168).
> Refinement 3 also replaced the `tensor_row_tiles ≤ G/8` occupancy predicate with a per-core cost model
> (`_choose_k`: min over k of busiest-group rows × (slice + COMBINE_TILE_COST)), so mid tile-row counts split too.

## Implementation notes (ttnn-implementer)

- (verifier) Single source of truth: every row's capacity above is produced by `cb_plan()` in
  `rms_norm_program_descriptor.py`; `footprint()` (the L1-fit solve) and the CB declarations both read it.

- `x_row_pitch` = `slice_w_tiles` (TILE) / `slice_w_tiles` padded to a multiple of `rm_chunk_w_tiles` (RM) is
  the page pitch of one tile-row in `cb_x` on EVERY core (one CB list program-wide, required because peers
  address `cb_gathered_sumsq` with their own write pointer). The ragged last W member pushes
  `x_row_pitch − core_w_tiles` (RM: `− chunks_this·C`) unwritten gap pages per tile-row; compute addresses
  `cb_x` with `Strided{base, x_row_pitch}`. Capacity column for `cb_x` = `block_row_tiles · x_row_pitch` per CB.
- `cb_out` (TILE) capacity `out_cb_pages = out_depth · out_unit_tiles`; the writer clamps each unit to the
  ring end (core_w_tiles need not be a multiple of out_unit_tiles).
- Data-movement budget unchanged: x and out cross DRAM once; gamma once per active Tensix core (its slice).
- (R3) `cb_gamma` producer moved from the reader to the writer (single producer still): the writer is idle until
  the first output tiles, so the x stream starts at once; compute waits for gamma before its first pass 2, not at
  kernel start. Interleaved: loaded by one multicast per W slice (`_gamma_mcast_groups`). No CB added or resized.


### Perf 1 — CB changes
- `cb_gamma` pages halve (faces 0/1 only; row-broadcast gamma), freeing ~S_x · T_g / 2 per core — the resident
  gamma now fits in more wide-W sharded plans before the streamed fallback is needed.
- `cb_sumsq_acc` (index 10) deleted: the masked sharded W tail is folded into pass 1's DEST accumulation.
- `cb_ones` (index 13, 1 bf16 page) added: the FPU row collapse's 1.0 scaler.
- Combine pipeline (sharded owner combine): `cb_inv_rms`, `cb_partial_sumsq`, `cb_gathered_sumsq` sized per
  sub-block (see the footprint note); two extra semaphores (ids 2, 3 — only on paths with no `ttnn.Mcast` attached).
