# L1 ledger: g6_embedding

Schema: `.claude/references/l1-footprint-discipline.md`. Host model: ProgramSpec; the buffers are
DataflowBufferSpecs (`index_values`, `gathered_rows`, `output_tiles`), slot indices assigned by the runtime. Block axes (op_design.md, Blocking Model):
`tile_row` (extent `BLOCK_TILE_ROWS`), `col_tile` (extent `block_col_tiles`), `image` (folded into
`tile_row`), `table_row` (gathered, not a block axis). `fp32_dest_acc_en = False` (16-bit DEST).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_index_values` (`dfb::index_values`), 128 B pages | `INDEX_DEPTH · BLOCK_TILE_ROWS` | `2 · BLOCK_TILE_ROWS` slots: the current row-block's indices (read by the RISC while issuing row reads) + the next row-block's, prefetched before the current block's barrier | `{tile_row: spans → BLOCK_TILE_ROWS, streams → INDEX_DEPTH window; col_tile: neither (idx does not vary along D) → 1; image: folded into tile_row; table_row: n/a}` | UInt32 (index values; RISC-read, never through DEST) | reader (NoC read; `reserve_back`/`push_back` one slot per row block) | reader (self-loop DFB `dfb::index_values`: `wait_front` + RISC loads from the read pointer, `pop_front` after the row block's last column block has issued its reads); values valid after the reader's own read barrier | whole reader kernel | none: its prefetch lands while the same block's row reads fill `cb_gathered_rows` (concurrent), and its format differs. 256 B at Phase 0 |
| `cb_gathered_rows` (`dfb::gathered_rows`), 2048 B pages | `ROWS_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` | `BLOCK_TILE_ROWS · block_col_tiles` per pipeline stage; capacity is 2× for double buffering (reader fills block k+1 while compute tilizes block k) | `{tile_row: spans → BLOCK_TILE_ROWS, streams → ROWS_DEPTH; col_tile: spans → block_col_tiles; image: folded; table_row: n/a (≤ 32 gathered rows per tile-row fixed by the tile height)}` | UInt16 (bf16 bits as integers: keeps the tilize off the float fast path, so NaN payloads / -0.0 / subnormals survive; 16-bit = DEST width) | reader | compute (tilize helper) | whole kernel, block-pipelined | not with `cb_output_tiles`: tilize reads one while writing the other (concurrent) and the helper forbids in-place (`static_assert(input_dfb != output_dfb)`) |
| `cb_output_tiles` (`dfb::output_tiles`), 2048 B pages | `OUT_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` | `BLOCK_TILE_ROWS · block_col_tiles` per stage; 2× for double buffering (compute packs block k+1 while the writer drains block k) | `{tile_row: spans → BLOCK_TILE_ROWS, streams → OUT_DEPTH; col_tile: spans → block_col_tiles; image: folded; table_row: n/a}` | UInt16 (same bytes as a bf16 tile; 16-bit = DEST width) | compute | writer | whole kernel, block-pipelined | not with `cb_gathered_rows` (see above); the result is DRAM interleaved, so no zero-copy output CB exists |

Every push and pop on `cb_gathered_rows` / `cb_output_tiles` is `BLOCK_TILE_ROWS · block_col_tiles`
pages on the dataflow side and `block_col_tiles` pages per tile-row on the compute side, both
divisors of the capacity, so the ring closes. A ragged last row-block (possible only when
`BLOCK_TILE_ROWS > 1`) still pushes and pops the nominal count.

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `BLOCK_TILE_ROWS` | 1 (host constant) | op_design.md, `tile_row` axis |
| `INDEX_DEPTH`, `ROWS_DEPTH`, `OUT_DEPTH` | 2 each (host constants); `INDEX_DEPTH ≥ 2` asserted on host and in the reader | op_design.md, buffer-depth knobs |
| `block_col_tiles` | `≤ col_tile_cap = floor((1_347_776 − INDEX_DEPTH·BLOCK_TILE_ROWS·128) / ((ROWS_DEPTH + OUT_DEPTH)·BLOCK_TILE_ROWS·2048))` = 164 at Phase 0; a divisor of `tensor_col_tiles` | `select_plan`: `q_l1` clamp; regime predicate `col_blocks_per_row = max(q_l1, q_fill)` |

## Total per core

`L1 = INDEX_DEPTH · BLOCK_TILE_ROWS · 128 + (ROWS_DEPTH + OUT_DEPTH) · BLOCK_TILE_ROWS · block_col_tiles · 2048` B.

The first term scales with `BLOCK_TILE_ROWS` and `INDEX_DEPTH`; the second with
`BLOCK_TILE_ROWS`, `block_col_tiles` and the two depths. Largest production plan: `block_col_tiles
= 40` → 327 936 B (24 % of BB-05). `validate()` refuses a plan above 1 347 776 B (unreachable:
`block_col_tiles = 1` always fits).

## Data-movement budget (DRAM crossings per tensor)

| Tensor | `whole_row` | `column_blocked` |
|--------|-------------|------------------|
| idx | 1×: `4 · valid_rows(g)` B per tile-row, one read per tile-row | `col_blocks_per_row`× (each column block's core reads its tile-row's ≤ 128 B) |
| W | `2D` B per index (repeats read again), N reads of `2D` B | same bytes, `N · col_blocks_per_row` reads of `block_col_tiles · 64` B |
| out | 1× (tile-padded: `2D · 32 · tensor_tile_rows` B), one 2048 B write per tile | 1× |

No cross-core traffic in either regime.
