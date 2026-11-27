# L1 Ledger: g6_repeat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are those in `op_design.md`: `out_batch`, `out_tile_row`, `row_in_tile` and `out_tile_col`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_repeat_tiles` | `REPEAT_BUFFER_DEPTH * block_width_tiles` | `REPEAT_BUFFER_DEPTH * block_width_tiles`. Capacity equals the live set. One slot set is fully resident from the start of the block's load until its writes flush (all `block_extent` tiles: loads land before the replicate, and the replicate finishes before the stores). With depth > 1, each other slot set is in flight with an earlier or later block. | `{out_batch: streams → 1 (contract extent 1), out_tile_row: streams → 1 (one built tile serves every tile row, written tensor_out_ht_tiles times), row_in_tile: spans → whole tile (tile_h rows per slot; rows_to_fill ≤ tile_h), out_tile_col: spans → block_width_tiles}` | bfloat16 (`x.dtype`); a dataflow-only payload, never in DEST, so `fp32_dest_acc_en` does not apply | `g6_repeat_dataflow` (DRAM read + loopback copies) | `g6_repeat_dataflow` (NoC write source); the same thread, no handshake | load → replicate → store per block; the slot set is reused after `noc_async_writes_flushed` | The op's only buffer, so there is nothing to share with. It already absorbs the two buffers a naive design would have. **Input CB**: removed by packing into the destination; x's row-0 faces land in the out-tile slot at the offsets row 0 occupies in the out tile. **Replicate scratch**: removed by transforming in place; the copies read row 0 of the slot and write rows 1..R-1 of the same slot, with one thread ordered by a barrier. Folding into an accumulator does not apply (no compute). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `out_page_size` | 2048 B | bfloat16 TILE 32x32 (`validate()`: dtype and layout in SUPPORTED); from `x.buffer_aligned_page_size()`, which equals out's (same dtype, layout, tile and placement). Measured `device_l1_peak_bytes` at p1: 2048. |
| `REPEAT_BUFFER_DEPTH` | 1 (Phase 0) | host constant |
| `L1_BUDGET_BYTES` | 1347776 B | BB-05 (model-free L1 per core) |
| `block_width_tiles` | `1 <= block_width_tiles <= max_block_width_tiles = floor(L1_BUDGET_BYTES / (REPEAT_BUFFER_DEPTH * out_page_size))` = 658; also `<= ceil(tensor_out_wt_tiles / num_cores)` and `<= MAX_BLOCK_WIDTH_TILES` when set | host clamp in `g6_repeat_program_descriptor.py`. A core with more columns loops over `num_blocks_this_core` blocks, so no W is refused. |
| `tensor_out_ht_tiles` | 1 | `validate()`: `reps[2] < 32`, `shape[:3] == (1,1,1)`, `reps[0] == reps[1] == 1` |
| `rows_to_fill` | `1 <= rows_to_fill <= 31 < tile_h` | `validate()`: `1 <= reps[2] < 32` |
| `tensor_out_wt_tiles` | unbounded (W has no upper limit) | not in any capacity expression; it enters only the trip counts through `core_w_tiles` |

## Total per-core footprint

`L1_op = REPEAT_BUFFER_DEPTH * block_width_tiles * out_page_size  <=  L1_BUDGET_BYTES = 1347776 B`

- It scales linearly with `block_width_tiles` (the `out_tile_col` extent) and with `REPEAT_BUFFER_DEPTH`.
- No term scales with `R`, `W` or the tile-row count.
- p1 (W = 256, R = 16, 8 cores): `1 * 1 * 2048 = 2048 B` per core.
- The largest Phase 0 per-core footprint is `658 * 2048 = 1347584 B`, reached only when `tensor_out_wt_tiles >= 658 * num_cores`.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 per page, partial: `row0_faces_bytes` = 1024 of 2048 B | Each out tile column needs only row 0 of its own x tile, and row 0 lies in the upper faces. The read lands in the out slot and stays resident until the out page is built, so no second read happens (`row_in_tile` reuse is satisfied in L1). | none |
| out | 1 per page, write only, whole page (2048 B); never read | Each page is built completely in L1 before its single write. | none |
| (core-local) loopback copies | 0 DRAM | `2 * (rows_to_fill - 1) * face_row_bytes` per tile within one core's L1 (960 B at R = 16) | none (self-aimed) |

**p1 totals (W = 256, R = 16, 8 tiles):**

| Tier | Bytes | Transactions |
|------|-------|--------------|
| DRAM read | 8 x 1024 = 8192 B | 8 |
| DRAM write | 8 x 2048 = 16384 B | 8 |
| Cross-core NoC | 0 B | 0 |
| Core-local NoC loopback | 8 x 960 = 7680 B | 8 x 30 = 240 copies, 30 per core |

> Cheapest-traffic split considered: `out_tile_col` across cores — 0 cross-core bytes, x and out each crossing DRAM once. Implemented: `out_tile_col` across cores (regime 1). The implemented split **is** the cheapest. The only byte-level slack is x's 960 B of padding per page, which is kept in exchange for one read transaction per tile instead of two (perf lamp L3 in `op_design.md`).
