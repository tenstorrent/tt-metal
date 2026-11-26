# L1 Ledger: g6_to_layout (Phase 0, regime R1 `interleaved_untilize_2d`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_rows` (extent `block_tile_rows`), `tile_cols` (extent `block_tile_cols`), `tile_row_inner` (32, fixed by the tile), `face` (read unit inside a tile).

## Inventory

Two buffers, and no fewer is possible. Untilize cannot run in place: the helper asserts `input_dfb != output_dfb` (`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:153-154`), and the input and output orders of a tile differ. No intermediate exists between phases: the reader lands tiles directly in the compute's input, and the packer writes row-major straight into the buffer the writer drains (pattern 1, "pack into the destination"). There are no scalar or constant buffers. Nothing is aliased, because both buffers live for the whole kernel and are pipelined against each other (pattern 3 is closed by the pipelining decision).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `depth_in * block_tile_rows * block_tile_cols` (Phase 0: `2 * block_tile_rows * block_tile_cols`) | `2 * block_tile_rows * block_tile_cols`: one block landing (reader) plus one block being unpacked (compute) | `{tile_rows: spans → block_tile_rows, tile_cols: spans → block_tile_cols, tile_row_inner: spans → 32 (inside each tile page), face: spans → 4 (a page holds all faces; R1a leaves unread faces stale, it does not shrink the page)}` | UInt16 (opaque 16-bit = bfloat16 bits; DEST is 16-bit, `fp32_dest_acc_en` off — widths agree) | reader | compute | whole kernel; ring of `depth_in` blocks | cannot share with `cb_output_rows`: the lifetimes run at the same time (block k+1 lands while block k is packed), and the helper forbids in-place untilize. The extra block over a single block is `depth_in = 2`, which is double buffering |
| `cb_output_rows` | `depth_out * block_tile_rows * block_tile_cols` (Phase 0: `2 * block_tile_rows * block_tile_cols`) | `2 * block_tile_rows * block_tile_cols`: one block being packed (compute) plus one block being written out (writer) | `{tile_rows: spans → block_tile_rows, tile_cols: spans → block_tile_cols, tile_row_inner: spans → 32 (a unit's 32 rows, full width), face: streams (row-major content has no faces)}` | UInt16 (same reason; a ROW_MAJOR bfloat16 output is written with the same bits) | compute | writer | whole kernel; ring of `depth_out` blocks | cannot share with `cb_input_tiles` (same reason, concurrent). The extra block is double buffering (`depth_out = 2`). Writing straight from `cb_input_tiles` (no untilize) is regime R5, which is rejected |

Page size of both: `tile_bytes = 2048` (the untilize helper uses tile-sized pages for both, `untilize_helpers.hpp:112-113`).

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `depth_in`, `depth_out` | 2 each (Phase 0) | host constants `DEPTH_IN`, `DEPTH_OUT` |
| `tile_bytes` | 2048 (bfloat16 / UInt16); 1088 for a Bfp8_b input page (R4) | data format |
| `block_tile_cols` | `1 <= block_tile_cols <= max_block_cols`; a multiple of 8 when > 8 | `plan_blocking()`: DEST cap (`pack_untilize.h:102`) and L1 cap |
| `max_block_cols` | `floor_dest(floor(op_l1_budget / ((depth_in + depth_out) * tile_bytes)))`, which is 160 for a DRAM output | `plan_blocking()` |
| `block_tile_rows` | `1 <= block_tile_rows <= max_rows_fit = floor(op_l1_budget / ((depth_in + depth_out) * block_tile_cols * tile_bytes))`; target `floor(32 / block_tile_cols)` | `plan_blocking()` (`BLOCK_TILES_TARGET = 32`) |
| `op_l1_budget` | `1347776 - out_l1_bytes_per_core` | spec (## Shape handling, rule 12, BB-05) |
| `out_l1_bytes_per_core` | `ceil(num_out_pages / num_l1_banks) * align_up(2W, l1_alignment)` for an L1 output, else 0. Phase 0 production maximum: case 2038 gives `ceil(32 / banks) * 5120 = 5120` | output buffer, host |
| `tensor_tile_rows`, `tensor_tile_cols`, W, N | unbounded op dimensions; **no capacity expression uses them** | — |

## Total per-core footprint

```
L1_op = (depth_in + depth_out) * block_tile_rows * block_tile_cols * tile_bytes  [+ out_l1_bytes_per_core when the output is in L1]
      <= op_l1_budget + out_l1_bytes_per_core = 1347776     (asserted on host by plan_blocking())
```

Scaling: both terms scale with `block_tile_rows * block_tile_cols`, which is the block knob, and linearly with their own depth knob. Neither term scales with any op dimension. Production values with a DRAM output: p1 `4 * 32 * 1 * 2048 = 256 KiB`; p2 `4 * 8 * 4 * 2048 = 256 KiB`; Qwen `[16,4096,2560]` `4 * 1 * 80 * 2048 = 640 KiB`; `[1,131072]` `4 * 1 * 40 * 2048 = 320 KiB`; the worst case `block_tile_cols = 160` gives `4 * 160 * 2048 = 1.25 MiB`, which is at most 1347776.

## Data-movement budget

Named boundary: the memory of each tensor (DRAM, or L1 banks for an L1 tensor).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (TILE) | 1: each live tile read once. Tail tiles under R1a read 1/2 or 1/4 of the tile | no tile belongs to two units, so no core re-reads anything; the block extent does not change the count | none |
| `output` (ROW_MAJOR) | 1: each live row byte written once (one write per live row and chunk) | the whole row chunk is resident in `cb_output_rows` when it is written | none (an L1-interleaved input or output: the pages travel between L1 banks over the NoC; this is the tensor's own placement, counted as its one crossing) |

Totals per tier, as input + output bytes:
- p1: DRAM 134 MB read (faces 0 and 2 of 131072 tiles) + 4194304 row writes of 16 B, each into a 64 B slot. The DRAM write cost is one burst per row, which is structural.
- p2: 537 MB + 537 MB.
- Qwen `[16,4096,2560]`: 336 MB + 336 MB.
- Cross-core: 0 for every case.

> Cheapest-traffic split considered: tile rows only, with whole-row chunks. It saves 0 bytes per tier compared with the 2-D split, and it needs fewer, larger output writes. Implemented: the 2-D split (R1), which **is** the tile-rows-only split whenever `tensor_tile_rows >= num_cores` and the row fits L1 (`num_col_chunks = 1`). Width chunks appear only for occupancy or the L1 cap, and they move the same bytes. So the implemented split is the cheapest-traffic split.
