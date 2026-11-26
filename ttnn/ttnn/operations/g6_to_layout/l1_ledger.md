# L1 Ledger: g6_to_layout

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions:
`op_design.md` -> Blocking Model. Rows cover the built regime R1 (`untilize_interleaved`).

Block axes (from the Blocking Model): `n` (folded leading dims), `ht` (tile-rows of an image),
`wt` (tile-columns), `i` (row in tile), `c` (column in tile).

## Buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tiles` | `tiles_depth * block_segments * segment_width` pages of `tile_bytes_in` | `tiles_depth * block_segments * segment_width` at the steady-state peak: the block compute is untilizing plus the block the reader has landed and pushed. Capacity equals the live set; the `tiles_depth` factor is the double-buffer mechanism (read of block k+1 overlaps untilize of block k) | `{n: streams -> folded into the segment index (a block may span images), ht: spans -> block_segments, wt: spans -> segment_width, i: spans -> 32 (tile_h), c: spans -> 32 (tile_w)}` | `UInt16` — a dataflow payload whose bits must not be interpreted (MOVA2D preserves zero-exponent datums only for `UInt16` operands); 2 B per datum like bf16, so no extra bytes; `fp32_dest_acc_en` off, DEST 16-bit, no accumulation | reader | compute | whole program | none — concurrent with `cb_rows` for the whole pipelined program (the reader fills block k+1 here while the writer drains block k there); in place impossible: the untilize helper forbids `input == output` (`untilize_helpers.inl:153-154`); aliasing excluded by the pipelining |
| `cb_rows` | `rows_depth * block_segments * segment_width` pages of `tile_bytes_rows` | `rows_depth * block_segments * segment_width` at the steady-state peak: the block the writer is draining (its pages stay until writes-flushed) plus the block compute is packing. Capacity equals the live set; `rows_depth` is the double-buffer mechanism | `{n: streams -> folded into the segment index, ht: spans -> block_segments, wt: spans -> segment_width, i: spans -> 32 rows, c: spans -> 32 columns per tile column (64 B)}` | `UInt16` — same bits leave `pack_untilize` as entered; the writer emits raw bytes into the bf16 output | compute | writer | whole program | none — concurrent with `cb_tiles` (above); the writer cannot emit rows from tile order without 32-B fragmentation (the narrow-W exception is regime R6, deferred) |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `tile_bytes_in` | 2048 | R1 predicate: input dtype bf16 (R2 would make it 1088, `Bfp8_b`) |
| `tile_bytes_rows` | 2048 | every TILE input converts to a bf16 ROW_MAJOR output (Output rule) |
| `tiles_depth`, `rows_depth` | 2 each | Phase 0 constants (`Plan` fields) |
| `segment_width` | `1 <= segment_width <= min(tensor_w_tiles, seg_width_max)`, `seg_width_max = floor(B_cb / (tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows))` (164 for a DRAM output) | plan clamp (Work Distribution) |
| `block_segments` | `1 <= block_segments <= min(ceil(num_segments / cores_used), floor(B_cb / (segment_width * (tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows))))` | plan (Axes row `ht`) |
| `B_cb` | `1347776 - out_l1_bank_bytes - cb_align_slack` | BB-05: the model leaves 1347776 B per core, counted before the output exists |
| `out_l1_bank_bytes` | 0 for a DRAM output; `ceil(num_out_pages / num_l1_banks) * align(W*2, 16)` for an L1 output | output placement (`memory_config`, else the input's) |
| `num_l1_banks` | `ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks` | device |
| `cb_align_slack` | `2 * 64` B | one 64-B base alignment per CB (BH DRAM reads need local == NoC address mod 64) |

The host refuses (RuntimeError, before any dispatch) a call whose `B_cb` is below
`tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows` = 8192 B — only an L1 output of
roughly 1.3 MB per bank reaches it, which no golden INPUTS case does (the largest Phase 0 L1
output, case 2038 `(1, 32, 2560)`, is 163840 B in total).

## Total per-core footprint

```
F = block_segments * segment_width * (tiles_depth * tile_bytes_in + rows_depth * tile_bytes_rows)
    + out_l1_bank_bytes                                      (L1 output only)
  <= 1347776 - cb_align_slack
```

`F` is linear in `block_segments`, `segment_width`, `tiles_depth` and `rows_depth`; it scales
with no op dimension except through the clamps above (a narrow tensor gets a narrow
`segment_width`, a short assignment a small `block_segments`). The default takes `block_segments`
at its bound, so `F` sits at the budget whenever a core owns more segments than fit.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (TILE, bf16) | 1 x its live bytes at the 32-B face-row granule (a fully live tile is one 2048-B read; dead faces and tile-padding rows are never read) | each tile lands once in `cb_tiles`, is untilized once and popped; no pass re-reads it | 0 (an L1-interleaved input crosses the NoC once instead of DRAM) |
| output (ROW_MAJOR, bf16) | 1 x `N*H*W*2` B | each row segment is written once from `cb_rows` | 0 |

Totals for the flagged and boundary shapes:

| Shape | Input bytes read | Output bytes written | Write transactions |
|-------|------------------|----------------------|--------------------|
| `(16, 4096, 2560)` | 335,544,320 | 335,544,320 | 65,536 (one per row) |
| `(1, 2048, 2048, 3)` (p1) | 134,217,728 (faces 0 and 2 of 131072 tiles) | 25,165,824 | 4,194,304 (6 B each) |
| `(1, 1024, 2048, 128)` (p2 = p3) | 536,870,912 | 536,870,912 | 2,097,152 (256 B each) |
| `(1, 1024, 1024, 3)` (p4) | 33,554,432 | 6,291,456 | 1,048,576 |
| `(1, 131072)` | 262,144 (64 B of each of 4096 tiles) | 262,144 | one per segment |

Per tier: DRAM = input live bytes + output bytes; NoC between cores = 0; L1 = each byte written
and read once in each CB.

p1 against its target: the read term alone (134 MB at the 32-B granule) needs >= 262 us at the
~512 GB/s nominal DRAM peak, so the lead's 147.46 us target (1.35 x a roofline built on 25 MB of
logical input) is below the physical floor of any scheme that reads DRAM; the design reaches the
granule floor and reports the gap rather than chasing it.

> Cheapest-traffic split considered: tile-rows only (`num_col_chunks = 1`) — every candidate split
> moves the same bytes per tier (input live ranges once across DRAM, output once, 0 cross-core);
> they differ only in write transactions (`num_col_chunks` per output row). Implemented: the segment
> split with the smallest `num_col_chunks` that balances within `balance_slack`, which IS the
> cheapest split on every shape whose tile-rows fill the grid (all flagged performance cases); on
> shapes with fewer tile-rows than cores it adds `num_col_chunks - 1` writes per row and 0 bytes.
> Nothing deferred: no split moves fewer bytes.
