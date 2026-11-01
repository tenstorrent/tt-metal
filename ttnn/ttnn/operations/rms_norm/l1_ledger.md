# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`
→ Blocking Model): **row** (extent `block_row_tiles`) and **W** (extent = the core's whole share
`core_w_tiles`; RM pads it to `core_w_tiles_padded`). `rm_chunk_w_tiles` is a streaming quantum
along W, not a block extent. Page sizes: T_in / T_out / T_g = tile bytes (bytes) of the input /
output / gamma dtype (bf16 2048 B, fp32 4096 B); T32 = 4096 B (Float32 tile).
`fp32_dest_acc_en = True` in Phase 0, so every compute-to-compute intermediate is a Float32 page.

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
| `cb_gamma` | `slice_w_tiles` (RM: padded) | `core_w_tiles` | `{row: absent (gamma has no row extent; reused by every row), W: spans → core_w_tiles}` | gamma dtype (Float16_b / Float32 — raw weight) | reader | compute | program | none: program lifetime, read by every block's `normalize_block` |
| `cb_sumsq_tiles` | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: reduced away (DEST accumulation)}` | Float32 (fp32 DEST accumulation result) | compute | compute | pass 1 → collapse | not with `cb_inv_rms` / `cb_partial_sumsq`: the collapse reduce reads this CB while writing those (concurrent). Same size and format as `cb_inv_rms`; in R2 their lifetimes are disjoint, but R1 overlaps them, and the saving is `block_row_tiles·4 KB` (< 1% of the footprint) — one design for both regimes kept |
| `cb_partial_sumsq` (k > 1) | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: reduced away}` | Float32 | compute | writer | collapse → exchange | not with `cb_sumsq_tiles` (concurrent in the collapse); not with `cb_gathered_sumsq` (peers write into that one at any time during the previous block) |
| `cb_gathered_sumsq` (k > 1) | `2 · w_split_cores · block_row_tiles` | `w_split_cores · block_rows_this` (one block); capacity 2× = mechanism cap (lockstep slot reuse) | `{row: spans → block_row_tiles, W: reduced to one partial per member → w_split_cores}` | Float32 | writer (+ peer NoC writes landing in its reserved slot) | compute | exchange → combine | none: remote writes may land at any time during the previous block, so no other buffer may occupy these bytes |
| `cb_inv_rms` | `block_row_tiles` | `block_rows_this` | `{row: spans → block_row_tiles, W: absent (broadcast along W)}` | Float32 (finalize result in fp32 DEST) | compute | compute | finalize → end of `normalize_block` | not with `cb_sumsq_tiles` (see that row) |
| `cb_out` | TILE: `out_depth · out_unit_tiles` (= 2·min(slice_w_tiles, 16)); RM: `block_row_tiles · rm_chunk_w_tiles` | TILE: ≥ 1 tile (streaming window); RM: `block_rows_this · rm_chunk_w_tiles` (pass 2 of one chunk column completes before untilize, same thread) | TILE `{row: streams → 1 tile, W: streams → 1 tile}`; RM `{row: spans → block_row_tiles, W: streams → rm_chunk_w_tiles}` | output dtype (terminal pack to the output format, not an intermediate) | compute | writer (TILE) / compute (RM) | per tile / per chunk | TILE: none, live with every other per-block CB; capacity > live set = double buffering for the writer. RM: not with `cb_x_rm` (concurrent across the reader/compute pipeline) |
| `cb_out_rm` (RM only) | `out_depth · rm_chunk_w_tiles` | `rm_chunk_w_tiles` | `{row: streams → 1 tile-row, W: streams → rm_chunk_w_tiles}` | output dtype | compute | writer | per (tile-row, chunk) | not with `cb_x_rm` (concurrent); capacity 2× = writer double buffering |

Semaphore `sem_partials_arrived` (k > 1): one 4-byte L1 word, program lifetime.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `block_row_tiles` | `1 ≤ block_row_tiles ≤ core_row_tiles_max` | `select_blocking()`: largest value with `footprint ≤ budget` (whole assignment at depth 1, else depth 2) |
| `slice_w_tiles` | `ceil(tensor_w_tiles / w_split_cores)`; `footprint(k, 1, 1) ≤ budget` | `k_fit` (smallest k ≤ K_CAP that fits); RuntimeError otherwise |
| `core_w_tiles` | `≤ slice_w_tiles`; ≥ 1 | k normalization `k = ceil(tensor_w_tiles / slice_w_tiles)` |
| `core_w_tiles_padded` | `ceil(core_w_tiles / rm_chunk_w_tiles) · rm_chunk_w_tiles < core_w_tiles + chunks` | balanced chunking |
| `w_split_cores` (k) | `1 ≤ k ≤ K_CAP = min(8, grid_x)` | clamp in `select_blocking()` |
| `rm_chunk_w_tiles` | `≤ 32` | balanced chunk formula |
| `x_block_depth` | ∈ {1, 2} | 2 iff `num_blocks_this_core > 1` and it fits; RM: 1 |
| `out_depth`, `rm_in_depth` | 2 | constants (Phase 0) |
| `out_unit_tiles` | `min(slice_w_tiles, 16)` | constant cap |
| `budget` | free L1 between the allocator base and the lowest allocated L1 buffer (BH p150b, empty device: 1 461 120 B; L1-resident input/output tensors lower it) | `cb_l1_budget(device)` reads `ttnn.get_memory_view(device, L1)` on the host (verifier fix: was `get_max_worker_l1_unreserved_size()`, which starts at the kernel-config ring buffer and overshoots the CB region by ~69 KB on BH) |

## Total per-core footprint

```
footprint(k, b, d) =
    d · b · S_x · T_in                                  # cb_x          (scales with block_row_tiles, slice, depth)
  + has_gamma · S_x · T_g                               # cb_gamma      (scales with slice only)
  + b · T32                                             # cb_sumsq_tiles
  + b · T32                                             # cb_inv_rms
  + [k > 1] · (b · T32 + 2 · k · b · T32)               # cb_partial_sumsq + cb_gathered_sumsq (scale with k·b)
  + TILE: 2 · min(slice_w_tiles, 16) · T_out            # cb_out (bounded constant)
  + RM:   2 · C · T_in + b · C · T_out + 2 · C · T_out  # cb_x_rm, cb_out, cb_out_rm (C = rm_chunk_w_tiles ≤ 32)
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
| gamma | `G/k` copies of the W slice (RM: W·elem_bytes per group; TILE gamma: tensor_w_tiles · 128 B per group, row 0 only) | gamma is reuse-shared along the row split; each core reads its slice once and keeps it for the program | 0 |
| sumsq partials (k > 1 only) | 0 | never leave L1 | per block per group: k² · block_rows_this · 4 KB unicast + k² semaphore increments |

Totals per launch: DRAM ≈ `2 · |x|` + `(G/k) · W · elem_bytes` (gamma, ≤ 1/32 of x per row-group);
NoC (R2 only) ≈ `k² · tensor_row_tiles · 4 KB`.

> Cheapest-traffic split considered: R1 (row-groups only, W resident) — on DRAM it ties with R2 for
> x and out and reads gamma k× more; R2 adds `k² · tensor_row_tiles · 4 KB` of NoC traffic. Implemented:
> both, selected by `select_blocking()`: R1 whenever k = 1 fits L1 and tensor_row_tiles > G/8 (it is
> then also the cheapest); R2 only where R1 cannot fill the grid (tensor_row_tiles ≤ G/8) or cannot keep
> the slice resident (W too wide), i.e. where the alternative to R2 is idle Tensix cores or a second DRAM
> crossing of x. The two-pass re-stream (+1 crossing of x) is `rejected`; gamma multicast
> (−(G/k − 1)·W bytes) is `deferred` because r ≤ 1/32 is in the catalog's measured-null band.

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
