# L1 Ledger: g6_embedding

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`,
Blocking Model): `token_tile_rows` (extent `block_height_tiles`), `d_tiles` (extent
`block_width_tiles`), `table_rows` (data-addressed, no extent). `fp32_dest_acc_en = False` (16-bit
DEST). Inventory: three buffers — staged rows, tiles, index slice; no intermediate between phases
other than the tilize helper's own input/output pair.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_staging` (2048 B pages) | `ROWS_DEPTH_BLOCKS · block_height_tiles · block_width_tiles` = `2·bh·bw` | `2·bh·bw`: block k+1 being gathered while block k is tilized | `{token_tile_rows: spans → block_height_tiles (×2 depth), d_tiles: spans → block_width_tiles, table_rows: streams → the ≤ 32·bh rows the block's indices name}` | `UInt16` (bit-opaque bf16; 16-bit, matches 16-bit DEST — the width is a mechanism choice: `Float16_b` flushes subnormals/−0/NaN, probe_006) | reader | compute | whole kernel; refilled once per block | not with `cb_out_tiles`: concurrent lifetime (the tilize helper reads one while packing the other; `tilize_helpers.inl` static_assert input ≠ output); not with `cb_index_slots`: block k+1's index lands while block k's rows land (stall-shadow prefetch); capacity = live set (depth 2 is the overlap mechanism) |
| `cb_index_slots` (128 B pages) | `INDEX_SLOTS · block_height_tiles` = `bh` | `bh`: one block's index slices (block k's values are loaded into registers before block k+1's slices land in the same slot) | `{token_tile_rows: spans → block_height_tiles, d_tiles: streams → not stored (every piece uses the same slice), table_rows: streams → values only, no rows}` | `UInt32` (dataflow-only payload, never through DEST) | reader | reader (private scratch, no push/pop) | whole kernel; reused per block | not with `cb_row_staging`: concurrent (see above); an alias into the staging block's tail would need the slot dead before row reads land, which holds only for an ordered issue — saves `128·bh` B (≤ 2.4 KB) for a fragile ordering constraint: rejected; capacity = live set |
| `cb_out_tiles` (2048 B pages) | `TILES_DEPTH_ROWS · block_width_tiles` = `2·bw` | `2·bw`: tile row r packed while tile row r−1 is written | `{token_tile_rows: streams → window of 2 tile rows (the helper's push quantum is one tile row), d_tiles: spans → block_width_tiles, table_rows: streams → none}` | `UInt16` (bit-opaque bf16; output ABI is bf16 bits, written as raw 2048-B pages) | compute | writer | whole kernel; one tile row per push/pop | not with `cb_row_staging`: concurrent (tilize input/output); not with `cb_index_slots`: concurrent (reader runs ahead of the writer); capacity = live set (depth 2 is the overlap mechanism) |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `block_width_tiles` (bw) | `1 ≤ bw ≤ 164` | `select_plan()` admits only `piece_count` with `tensor_d_tiles / piece_count ≤ bw_cap = floor((1347776 − 128) / 8192) = 164`; `piece_count` divides `tensor_d_tiles`, so `bw` is an integer ≥ 1 |
| `block_height_tiles` (bh) | `1 ≤ bh ≤ floor((1347776 − 4096·bw) / (4096·bw + 128))` and `bh ≤ core_units_max` | `select_plan()`: `bh = ceil(core_units_max / ceil(core_units_max / block_height_l1_cap))` |
| `ROWS_DEPTH_BLOCKS`, `TILES_DEPTH_ROWS`, `INDEX_SLOTS` | 2, 2, 1 | host constants |
| `B`, `S`, `N`, `V`, `D` | unbounded by the contract (`S`, `N` ≥ 1; `D % 32 = 0`) | appear in no capacity expression — only in loop trip counts (`core_units`, `num_blocks_this_core`) and in `bw`, which is clamped |

## Total per-core footprint

`L1(bw, bh) = 2048·bw·(2·bh + 2) + 128·bh` bytes ≤ 1,347,776 (BB-05), checked on host
(`plan.l1_bytes_per_core`; violation → `ValueError` naming the bound).

| Term | Scales with |
|------|-------------|
| `4096·bw·bh` (`cb_row_staging`) | both block extents and `ROWS_DEPTH_BLOCKS` |
| `4096·bw` (`cb_out_tiles`) | `block_width_tiles` and `TILES_DEPTH_ROWS` only |
| `128·bh` (`cb_index_slots`) | `block_height_tiles` and `INDEX_SLOTS` only |

Largest INPUTS footprints: p1 1,311,616 B (bw 40, bh 7); p3 983,680 B (bw 40, bh 5); p2 902,400 B
(bw 20, bh 10). Smallest: 8,320 B (bw 1, bh 1).

## Data-movement budget

Chosen split: units (tile row × column piece), `piece_count` from `select_plan()`. `N` = number of
indices, `P` = `piece_count`, `T` = `tensor_tile_rows`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | `P` × `4N` B (p1: 1 × 262,144 B; p4: 5 × 15,376 B) | each unit reads its own tile row's 128-B slice; the `P` pieces of a tile row each read it (no broadcast in Phase 0 — R2 deferred) | none |
| `weight` | the `N` named rows once per index occurrence, `2·N·D` B in `P·N` transactions — never the table (p1: 167,772,160 B = 26 % of the 635,699,200-B table, 65,536 reads of 2560 B) | rows are not reused (repeats re-read: R4 deferred); no table residency (forbidden) | none |
| result | 1 × `2·T·32·D` B incl. tile padding rows (p1: 167,772,160 B; p4: 9,912,320 B for 9,840,640 logical) | each tile is packed once and written once | none |

Totals (p1): DRAM 335,806,464 B (read 168,034,304 B, write 167,772,160 B); cross-core NoC 0 B;
core-local L1 traffic: each staged row byte is written once by the NoC, read once by the unpacker,
each tile byte written once by the packer and read once by the NoC.

> Cheapest-traffic split considered: token tile rows only (`P = 1`) — delta of `P > 1` against it:
> `+4N·(P−1)` B of index reads (p4: +61,504 B) and ×`P` row transactions at equal row bytes;
> 0 B cross-core. Implemented: the same split, with `P > 1` selected by `select_plan()` only where
> it shortens the busiest core's path (one-tile-row and ragged shapes; `P = 1` for p1 and p3).
> The duplicated index traffic of `P > 1` is the `deferred` regime R2 (index broadcast), deferred
> because it is ≤ 6.25 % of a unit's bytes and ≤ 1/32 of its transactions while costing a
> semaphore round trip per block; `fetch_index_block` stays a separate block op so R2 replaces
> only its source.
