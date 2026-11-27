# L1 Ledger: g6_embedding

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`, Blocking Model):
`tile_rows` (token tile rows, extent `block_tile_rows`), `positions` (token positions in a tile row, fixed 32),
`width` (D tile columns, extent `block_width_tiles`), `table_rows` (gather-addressed, no extent knob).

Compute config: `fp32_dest_acc_en = False` (16-bit DEST); no value is ever computed.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_index_window` (128 B pages) | `index_window_tile_rows = min(INDEX_WINDOW_MAX_TILE_ROWS, max_core_tile_row_span)` | `index_window_tile_rows` — by decision: the window is filled by one transfer per image segment before the first block, so every slot is resident at once; a one-slot live set would put a DRAM round trip inside every block | `{tile_rows: spans → index_window_tile_rows, positions: spans → 32 per page, width: streams (the indices do not vary along D; one copy serves every width block of a tile row), table_rows: none (holds row ids, not rows)}` | UInt32 — dataflow-only payload, never passes through DEST (audit 2 mechanism reason) | reader | reader (scratch: reserved once, read in place, no push/pop) | whole kernel (one window per core on every production shape) | none: read by every `gather_rows_block` while `cb_gathered_rows` and `cb_output_tiles` stream → concurrent lifetime with both |
| `cb_gathered_rows` (2048 B pages) | `GATHER_DEPTH_BLOCKS · block_tile_rows · block_width_tiles` | `2 · block_tile_rows · block_width_tiles` — one block being gathered by the reader, one being tilized by compute; capacity == live set at `GATHER_DEPTH_BLOCKS = 2` (the depth is the pipelining decision) | `{tile_rows: spans → block_tile_rows (× depth), positions: spans → 32 sticks per unit, width: spans → block_width_tiles, table_rows: streams → ≤ 32·block_tile_rows rows per block, never resident beyond the block}` | UInt16 — 16-bit matches 16-bit DEST; UInt16 rather than Float16_b is the bit-exactness mechanism (integer tilize path) | reader | compute | per block | none: concurrent with `cb_output_tiles` (tilize reads this CB while packing into that one; in-place tilize is impossible — `tilize_helpers.inl:89-90` static_asserts input ≠ output, and a tile's output bytes overlap sticks that still hold every other tile of the row) and with `cb_index_window` |
| `cb_output_tiles` (2048 B pages) | `OUTPUT_DEPTH_BLOCKS · block_tile_rows · block_width_tiles` | `2 · block_tile_rows · block_width_tiles` — one block being packed by compute, one being written by the writer; capacity == live set at `OUTPUT_DEPTH_BLOCKS = 2` | `{tile_rows: spans → block_tile_rows (× depth), positions: spans → 32 rows per tile, width: spans → block_width_tiles, table_rows: none}` | UInt16 — 16-bit matches 16-bit DEST; bit-identical bytes of the bfloat16 result | compute | writer | per block | none: concurrent with `cb_gathered_rows` (pipelined tilize) and with `cb_index_window`; the writer cannot write from `cb_gathered_rows` because the result must be TILE |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `block_width_tiles` | `1 ≤ block_width_tiles ≤ 80`, divides `Dt` | `select_block_width` candidates are divisors of `Dt` ≤ `MAX_BLOCK_WIDTH_TILES = 80` |
| `block_tile_rows` | `= 1` (Phase 0); if raised: `block_tile_rows · block_width_tiles ≤ 80` and `= 1` whenever `width_blocks_per_row > 1` | host constant `BLOCK_TILE_ROWS` + the clamp in Mechanism caps |
| `GATHER_DEPTH_BLOCKS`, `OUTPUT_DEPTH_BLOCKS` | `= 2` (Phase 0); lamp LP6 measures 3–4 | host constants |
| `index_window_tile_rows` | `1 ≤ index_window_tile_rows ≤ 64` | `min(INDEX_WINDOW_MAX_TILE_ROWS = 64, max_core_tile_row_span)` |
| `max_core_tile_row_span` | unbounded in the contract (grows with `B·S / C`); ≤ 19 on every production shape at `C = 110` | clamped by `INDEX_WINDOW_MAX_TILE_ROWS`; refill per window beyond it |
| tile page bytes | 2048 | UInt16 32×32 tile (same bytes as bfloat16) |
| index page bytes | 128 | 32 uint32 indices per tile row |

## Total per-core footprint

```
footprint = 128 · index_window_tile_rows
          + 2048 · block_tile_rows · block_width_tiles · (GATHER_DEPTH_BLOCKS + OUTPUT_DEPTH_BLOCKS)
```

| Term | Scales with |
|------|-------------|
| `128 · index_window_tile_rows` | the per-core tile-row span, capped at 64 → ≤ 8,192 B |
| `2048 · block_tile_rows · block_width_tiles · (depths)` | block width (D knob) × block height (token knob) × the two depth knobs |

Bound: Phase 0 maximum over every legal input `128·64 + 2048·1·80·4 = 663,552 B ≤ 1,347,776 B` (BB-05), so no shape of the contract — golden or not — reaches the bound; `validate()` still evaluates the expression and raises `ValueError` naming 1,347,776 B if it ever exceeds it. Production maximum: 330,112 B (`[16, 4096] × [248320, 1280]`, `block_width_tiles = 40`, window 19).

## Data-movement budget

Formulas for the implemented split (R1, `width_blocks_per_row = Dt / block_width_tiles`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| index (`4·B·S` B) | once when `width_blocks_per_row = 1`; each tile row's slice `width_blocks_per_row` times otherwise | each core reads only the slice of the tile rows it owns, once, into `cb_index_window`; the D split gives a tile row `width_blocks_per_row` owners | 0 |
| weight (`V·2·D` B) | the named rows only: `B·S·2·D` B, each named row once per position (a repeated index reads its row again); never a whole-table pass | `gather_rows_block` reads exactly `rows_valid` row slices per unit; the width split cuts a row into `width_blocks_per_row` slices of `block_width_tiles·64` B, same bytes | 0 |
| output (`T·Dt·2048` B) | once | each tile written once from `cb_output_tiles`; padding rows of partial tile rows travel inside their tiles | 0 |

At the perf focus (`[16, 4096] × [248320, 1280]`, `block_width_tiles = 40`, `width_blocks_per_row = 1`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| index (262,144 B) | 1 → 262,144 B | 19-unit slices, one window fill per core | 0 |
| weight (635,699,200 B table) | 65,536 rows × 2,560 B = 167,772,160 B (26 % of the table) | rows named by the indices, full-row transfers | 0 |
| output (81,920 tiles) | 1 → 167,772,160 B | whole-tile writes | 0 |

Totals at p1: DRAM reads 168,034,304 B, DRAM writes 167,772,160 B, cross-core 0 B; core-local L1: each gathered byte read once by the tilize unpacker and each tile packed once.

> Cheapest-traffic split considered: token tile rows only (`width_blocks_per_row = 1`) — the minimum (index, named rows and output tiles each cross DRAM once). Implemented: R1 with `select_block_width`; it is the cheapest split on every shape where the selection returns `block_width_tiles = Dt` (p1, p3, and every `D = 64` call with `T ≥ 64`). On the other shapes the width split adds `(width_blocks_per_row − 1)` reads of each tile row's index slice (≤ 3.1 % of the call's DRAM bytes, ≤ 384 extra reads per call) and no table or output bytes. The split that restores the minimum under the width split is R3 (index broadcast per tile-row group), deferred because it replaces each duplicated ≤128-B read — issued before the core's first row read — with a sender read → multicast → semaphore chain that still contains a DRAM read on every receiver's critical path; R1 keeps it reachable (only `fill_index_window`'s source changes).
