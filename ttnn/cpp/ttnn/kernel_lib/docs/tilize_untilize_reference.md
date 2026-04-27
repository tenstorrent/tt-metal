# Tilize/Untilize Reference (LLM)

## Helpers

Include: `#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"` and `untilize_helpers.hpp`.
Namespace: `compute_kernel_lib`.
MUST call `compute_kernel_hw_startup(cb_in, cb_out)` before any tilize/untilize.
Cannot operate in-place. cb_in != cb_out enforced via static_assert.

### compute_kernel_lib::tilize

Converts row-major data in cb_in to tile-format data in cb_out.

```cpp
template <uint32_t input_cb, uint32_t output_cb,
    InitUninitMode = InitAndUninit, WaitMode = WaitBlock,
    TilizeSpeedMode = Standard, uint32_t reconfig_from_cb = INVALID_CB>
void tilize(uint32_t block_width_tiles, uint32_t num_blocks,
    NonTileAlignedCBWaitConfig config = disabled());
```

- `block_width_tiles` (W_t): number of tiles across width. W_t = round_up(W, 32) / 32.
- `num_blocks`: number of tile-rows to process. For standard: H_aligned / 32.
- Each iteration: waits for input, reserves output, tilizes one tile-row (32 rows × W_t tiles), pushes output, pops input.

### compute_kernel_lib::untilize

Converts tile-format data in cb_in to row-major data in cb_out.

```cpp
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb,
    InitUninitMode = InitAndUninit, WaitMode = WaitBlock>
void untilize(uint32_t num_blocks);
```

- `block_width_tiles` (W_t): compile-time template param. Number of tiles across width.
- `num_blocks`: number of tile-rows to process.
- Auto-dispatches: pack_untilize (fast, if width fits DEST), block-based pack_untilize (wide integers), standard untilize (wide floats or WaitUpfront).

## Preferred Pattern: Both CBs Tile-Sized

**Always prefer tile-sized pages for both input and output CBs.** This is the simplest, most robust pattern. The reader kernel handles any padding from non-aligned source data before pushing to the CB.

Benefits:
- CB page accounting matches tile count directly.
- Helper's default Disabled mode works with no extra config.
- Avoids PCC issues seen with stick-sized output pages.
- Reader/compute/writer have symmetric page granularity.

### Host-side CB setup (program factory / descriptor)

```cpp
// Both CBs: page_size = tile_size, num_pages = W_t (or more for double-buffering)
create_cb(cb_in_idx,  program, core, tile_size, W_t, data_format);  // row-major data, tile-sized pages
create_cb(cb_out_idx, program, core, tile_size, W_t, data_format);  // tiled data, tile-sized pages
```

Why tile-sized pages for row-major cb_in: `W_t` tile-pages = `W_t × 32 × 32 × elem_size` bytes = `32 rows × W_aligned × elem_size` bytes. The byte count for one tile-row of row-major data equals exactly W_t tile-pages. Page accounting is consistent.

### Compute kernel

```cpp
compute_kernel_hw_startup(cb_in, cb_out);
compute_kernel_lib::tilize<cb_in, cb_out>(W_t, num_tile_rows);
```

### Reader kernel: fills cb_in with row-major data at tile-aligned stride

For each tile-row (32 rows of source data):
```cpp
cb_reserve_back(cb_in, W_t);
uint32_t base = get_write_ptr(cb_in);
uint32_t row_stride = W_aligned * elem_size;  // W_aligned = W_t * 32
uint32_t stick_bytes = W * elem_size;         // actual source width

for (uint32_t r = 0; r < num_real_rows; r++) {  // num_real_rows <= 32
    noc_async_read(get_noc_addr(stick_id + r, acc), base + r * row_stride, stick_bytes);
}
noc_async_read_barrier();
cb_push_back(cb_in, W_t);
```

If W is already tile-aligned (W == W_aligned): stick_bytes == row_stride, sticks are contiguous, no gaps.
If W is NOT tile-aligned: sticks placed at stride > stick_bytes, gap bytes (cols W..W_aligned-1) are L1 garbage.
If num_real_rows < 32 (last tile-row): rows num_real_rows..31 are L1 garbage.

## Row-Major vs Tiled Storage

Row-major [H, W]: H sticks of W elements. NO tile padding. Page = 1 stick.
Tiled [H, W]: round_up(H,32)/32 × round_up(W,32)/32 tiles. Padded to tile boundaries.

When tilizing row-major → tiled: source has NO padding. Reader creates all padding in CB via stick placement.

## Non-Aligned Dimensions

Given row-major [H, W] where H and/or W not multiples of 32:
- W_aligned = round_up(W, 32), H_aligned = round_up(H, 32)
- W_t = W_aligned / 32
- Total tiles = W_t × (H_aligned / 32)

### Width not aligned (W % 32 != 0)

Source sticks are W × elem_size bytes. CB row stride is W_aligned × elem_size bytes.
Reader MUST place sticks at this stride. Cannot read sticks contiguously into CB.
Cols W..W_aligned-1 in each row: don't-care (L1 garbage) unless downstream reads them.

### Height not aligned (H % 32 != 0)

Last tile-row has fewer than 32 real rows. Rows H_mod_32..31 in last tile-row: don't-care.
Helper handles this via TotalBatched mode (waits for fewer pages on last iteration).
Or reader simply writes fewer sticks in last iteration; tilize still reads 32 rows but garbage rows are harmless if unused.

### When padding must be zeroed

Zero if downstream reads padding positions: reductions (sum/mean across padded dim), softmax, comparisons/masks.
No zeroing needed if: element-wise ops followed by untilize (writer strips padding), matmul where padding × known-zero weights, broadcast ops where the required data is present
To zero: noc_memset CB region before reading sticks, or write zeros to gap bytes per row.

## NonTileAlignedCBWaitConfig Modes

Use these modes ONLY when the input CB has stick-sized pages instead of tile-sized pages. Prefer tile-sized pages when possible.

### Disabled (default)

Both CBs tile-sized. Wait/pop W_t tile-pages per iteration. Standard pattern.
```cpp
compute_kernel_lib::tilize<cb_in, cb_out>(W_t, num_tile_rows);
```

### PerIteration

Input CB has stick-sized pages. Wait/pop config.value stick-pages per iteration. Typically num_blocks=1.
Use when reader produces individual sticks and total stick count is known upfront.
```cpp
compute_kernel_lib::tilize<cb_in, cb_out>(W_t, 1,
    tilize_config::NonTileAlignedCBWaitConfig::per_iteration(total_sticks));
```
CB setup: cb_in page_size=stick_size, num_pages>=total_sticks. cb_out page_size=tile_size, num_pages>=W_t.

### TotalBatched

Input CB has stick-sized pages. Processes chunks of 32 rows, last chunk may be smaller.
Use when height is not tile-aligned and input CB uses stick pages.
NOTE: stick_size must be tile-width-aligned (W_aligned × elem_size). Only height is non-aligned in this mode.
```cpp
compute_kernel_lib::tilize<cb_in, cb_out>(W_t, H_aligned / 32,
    tilize_config::NonTileAlignedCBWaitConfig::total_batched(H));
```
CB setup: cb_in page_size=stick_size, num_pages>=H. cb_out page_size=tile_size, num_pages>=W_t*num_blocks.

## Untilize CB Setup

Always tile-sized pages for both CBs:
```cpp
create_cb(cb_in,  program, core, tile_size, W_t, data_format);  // tiled input
create_cb(cb_out, program, core, tile_size, W_t, data_format);  // row-major output, tile-sized pages
```
Using stick-sized pages for cb_out has caused PCC issues. Always use tile-sized.

Writer kernel strips padding:
- Width: writes only W × elem_size bytes per row, advances read pointer by W_aligned × elem_size.
- Height: program factory sets fewer blocks for last core so padding rows never written.

## InitUninitMode

`InitAndUninit` (default): standalone call, handles init and cleanup.
`InitOnly` / `UninitOnly` / `Neither`: for chaining multiple tilize/untilize calls without redundant init/uninit overhead.

## WaitMode

`WaitBlock` (default): cb_wait_front per iteration.
`WaitUpfront`: cb_wait_front for all data at start. Use when data is pre-loaded (e.g., sharded input, groupnorm pattern).
`NoWait`: caller already ensured data is available. Skip cb_wait_front entirely.

## TilizeSpeedMode (tilize only)

`Standard` (default): tilize_init/tilize_block/tilize_uninit.
`Fast`: fast_tilize variants. Requires 32x32 tiles + half-sync mode. Explicit opt-in.

## L1 Alignment

- L1 dest addresses: 16-byte aligned. CB base always aligned. Row stride always multiple of 64B (for bf16) → every row dest aligned.
- DRAM page alignment: 32 bytes.
- Row-major stick on device: W × elem_size bytes of data, page may be larger due to buffer alignment.

## Tile Internals

Tile = 32×32 elements. Stored as 4 contiguous 16×16 faces:
face0(r0-15,c0-15) → face1(r0-15,c16-31) → face2(r16-31,c0-15) → face3(r16-31,c16-31).
Tile size: 32×32×elem_size (bf16: 2048B, float32: 4096B).

## Concrete Examples

### [5, 37] bf16 — both non-aligned, tile-sized CB pages (preferred)

W_t=2, W_aligned=64. H=5, process 1 tile-row.
Host: create_cb(cb_in, ..., tile_size, 2, bf16); create_cb(cb_out, ..., tile_size, 2, bf16);
Reader: reserve 2 tile-pages (4096B). Read 5 sticks of 74B at stride 128B. Push 2 pages. Rows 5-31, cols 37-63: garbage.
Compute: `tilize<cb_in, cb_out>(2, 1);`
Output: 2 tiles. Real data in rows 0-4, cols 0-36. Rest garbage.

### [64, 128] bf16 — fully aligned

W_t=4, 2 tile-rows of 4 tiles each.
Host: create_cb(cb_in, ..., tile_size, 4, bf16); create_cb(cb_out, ..., tile_size, 4, bf16);
Reader: per tile-row, read 32 sticks × 256B contiguously. Push 4 pages.
Compute: `tilize<cb_in, cb_out>(4, 2);`

### [100, 64] bf16 — height non-aligned, width aligned

W_t=2, H_aligned=128. 4 tile-rows: 3×32 + 1×4 real rows.
Option A (preferred, tile-sized CB): Reader does 4 iterations. First 3: read 32 sticks at stride 128B. Fourth: read 4 sticks. Compute: `tilize<cb_in, cb_out>(2, 4);`
Option B (stick-sized CB): `tilize<cb_in, cb_out>(2, 4, NonTileAlignedCBWaitConfig::total_batched(100));`

### Untilize [5, 37] bf16

Input: 2 tiles in cb_in. Output: cb_out tile-sized pages.
Compute: `untilize<2, cb_in, cb_out>(1);`
Writer: reads 32 rows × 128B from cb_out, writes only 5 rows × 74B to DRAM.
