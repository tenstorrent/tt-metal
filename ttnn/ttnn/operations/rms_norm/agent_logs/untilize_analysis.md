# Untilize (Multi-Core) Implementation Analysis

## Analysis Focus

This analysis is produced with an **output_stage** focus for a downstream `rms_norm` operation (RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * gamma). The emphasis is on:

- The **untilize helper** signature and usage (compute kernel library)
- The **writer kernel** pattern (how RM sticks are written to DRAM)
- **Output CB sizing** (capacity, block size, buffering)
- **Stick extraction from tiles** (how tile data becomes row-major pages)

Reader kernel details, input CB configuration, and deep compute internals are de-emphasized.

## Overview

The untilize operation converts a tensor from TILE_LAYOUT (32x32 tiles, face-ordered) to ROW_MAJOR_LAYOUT (contiguous row sticks). It is a pure data-movement/reformatting operation with no arithmetic beyond the hardware-accelerated pack_untilize reordering.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

The operation uses three kernels:
1. **Reader**: reads tiled input pages from DRAM (interleaved) or signals sharded CB data
2. **Compute**: calls `compute_kernel_lib::untilize<>()` to reorder tile data into row-major sticks in the output CB
3. **Writer**: extracts row-major sticks from the output CB and writes them as RM pages to DRAM via TensorAccessor

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = `num_tiles_per_input_block` tiles wide, `tile_height` rows tall) |
| **Unit size** | `num_tiles_per_input_block` tiles (one full width of the tensor in tiles) |
| **Total units** | `num_tiles_per_col` (total tile-rows across the tensor height) |
| **Loop structure** | Outer: iterate over `num_input_blocks_to_process` blocks per core; Inner (writer): iterate over `tile_height` stick rows per block |

One "input block" is a horizontal strip spanning the tensor width in tiles and one tile height (32 rows). For interleaved input, `num_tiles_per_input_block = num_tiles_per_row` (the entire tensor width in tiles). For sharded input, it is `input_shard_width / tile_width`.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | N-dimensional (flattened to 2D: height x width) |
| **Dimension convention** | Height = product of all dims except last; Width = last dim |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face ordering) |
| **Memory layout** | INTERLEAVED or SHARDED (height, width, or block) |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT (contiguous row sticks) |
| **Memory layout** | INTERLEAVED or SHARDED (height, width, or block) |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformation

The core transformation is tile-to-row-major conversion. Inside a 32x32 tile, data is stored in 4 faces of 16x16 elements (face0, face1, face2, face3) in row-major face order. The untilize operation reorders this into 32 contiguous rows of 32 elements each. When the output CB contains `num_tiles_per_input_block` tiles worth of untilized data, it holds `tile_height` row-major sticks, each `num_tiles_per_input_block * tile_width` elements wide.

### Output Page Definition (Critical for rms_norm reference)

The output tensor pages are **row-major sticks**. Each page is one row of the output tensor:
- For interleaved output: `output_page_width = tensor_width` (full tensor row)
- For width/block-sharded output: `output_page_width = shard_width` (one shard's width)
- **output_stick_size** = `output_page_width * output_element_size` (bytes per page)

Pages are indexed linearly: `page_id = row_index * num_output_blocks_across_width + width_block_index`.

## Data Flow Pattern

### Stage 1: Reader (de-emphasized)
For interleaved input, tiles are read one-by-one from DRAM into CB c_0 using TensorAccessor. For sharded input, the CB is backed by the shard buffer directly and the reader just signals tile availability.

### Stage 2: Compute (untilize helper)
The compute kernel calls `compute_kernel_lib::untilize<>()` which:
1. **Waits** for `block_width_tiles` tiles in the input CB (one tile-row)
2. **Reserves** `block_width_tiles` slots in the output CB
3. Calls `pack_untilize_block<>()` which hardware-reorders face-format tiles into row-major sticks
4. **Pops** input tiles, **pushes** output tiles
5. Repeats for `num_blocks` (runtime arg: `num_input_blocks_to_process`)

After one iteration, the output CB contains `num_tiles_per_input_block` tiles of row-major data, which is equivalent to `tile_height` (32) contiguous sticks, each `num_tiles_per_input_block * tile_width` elements wide.

### Stage 3: Writer (primary focus)
The writer extracts sticks from the output CB and writes them as row-major pages to DRAM. This is the most complex kernel in the operation. Detailed analysis below.

## Untilize Helper Signature and Usage

### Helper Location
`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (declaration) and `untilize_helpers.inl` (implementation).

### Template Signature
```cpp
template <
    uint32_t block_width_tiles,     // Tiles per row (compile-time)
    uint32_t input_cb,              // Input CB index (tiled data)
    uint32_t output_cb,             // Output CB index (row-major data)
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
void untilize(uint32_t num_blocks);  // num_blocks = number of tile-rows (runtime)
```

### Invocation in untilize compute kernel
```cpp
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,   // = num_tiles_per_input_block (compile-time)
    src_cb_id,                 // = c_0
    out_cb_id,                 // = c_16
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,
    ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
    // per_core_block_cnt = num_input_blocks_to_process (runtime)
```

### Internal Dispatch Paths

The helper selects one of two paths at compile time based on `block_width_tiles` vs `DEST_AUTO_LIMIT`:

1. **Single-pass path** (`block_width_tiles <= DEST_AUTO_LIMIT`): Processes one full tile-row at a time through `pack_untilize_block<block_width_tiles, block_width_tiles>`. This is the fast, optimal path.

2. **Block-based path** (`block_width_tiles > DEST_AUTO_LIMIT`): Splits the tile-row into `num_sub_blocks` chunks, each fitting in DEST registers. Iterates: wait for sub-block, untilize sub-block, pop sub-block. Reserves the full row in the output CB upfront.

DEST_AUTO_LIMIT depends on sync mode and accumulation mode:
- Half-sync + bf16: 8 tiles
- Half-sync + fp32: 4 tiles
- Full-sync + bf16: 16 tiles
- Full-sync + fp32: 8 tiles

### CB Synchronization Within the Helper

For the default `WaitBlock` mode used by untilize:
- **Input CB**: `cb_wait_front(input_cb, block_width_tiles)` then `cb_pop_front(input_cb, block_width_tiles)` per block
- **Output CB**: `cb_reserve_back(output_cb, block_width_tiles)` then `cb_push_back(output_cb, block_width_tiles)` per block

The helper manages all CB synchronization internally. The caller only needs `compute_kernel_hw_startup()` before and the correct compile-time/runtime args.

### Key Usage Notes for rms_norm

For an rms_norm operation that wants to untilize its output:
- `block_width_tiles` = number of tiles along the last dimension (the dimension being normalized)
- `num_blocks` = number of tile-rows this core processes
- The untilize helper is self-contained: it does init, processing loop, and uninit
- If using untilize after other compute operations in the same kernel, use `InitUninitMode::InitOnly`/`UninitOnly` to avoid redundant init/uninit, or keep `InitAndUninit` if untilize is the only operation

## Circular Buffer Configuration

### CB c_0 (Input - de-emphasized)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Tiled input staging | See below | `num_tiles_per_input_block` tiles | Double (interleaved, multi-block) or Single (single-block/sharded) | Reader | Compute | Block |

Capacity logic:
- Sharded: `num_tiles_per_input_block * num_input_blocks_per_full_core` (entire shard at once)
- Interleaved, single block: `num_tiles_per_input_block` (single-buffered)
- Interleaved, multi-block: `num_tiles_per_input_block * 2` (double-buffered)

### CB c_16 (Output - primary focus)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_16 | cb_output | Row-major output staging | See below | `num_tiles_per_input_block` tiles | Double (multi-block) or Single (single-block) | Compute | Writer | Block |

**Output CB Capacity Logic** (program factory lines 149-163):
```
if (num_input_blocks_per_full_core == 1):
    output_cb_num_tiles = num_tiles_per_input_block          // Single-buffered
else:
    output_cb_num_tiles = num_tiles_per_input_block * 2      // Double-buffered
```

**Key insight**: The output CB is sized in units of `num_tiles_per_input_block` (one tile-row). When double-buffered, it holds 2 tile-rows, allowing the writer to drain one tile-row while compute fills the next.

**Data format**: `output_cb_data_format` (matches output tensor dtype). Page size = `output_single_tile_size` = `tt::tile_size(output_cb_data_format)`.

**What "one block" of output CB data looks like**: After `pack_untilize_block` processes one tile-row, the output CB contains `num_tiles_per_input_block` tiles of row-major data. Physically, this is `tile_height` (32) contiguous sticks laid out sequentially: stick 0 at offset 0, stick 1 at offset `num_tiles_per_input_block * tile_width * element_size`, etc. Each stick is `num_tiles_per_input_block * tile_width` elements wide.

## Writer Kernel Pattern (Primary Focus)

### File
`ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

### Architecture

The writer reads row-major sticks from CB c_16 and writes them as RM pages to DRAM via TensorAccessor. It handles the case where input blocks (from sharding) may not align with output pages (from different sharding configurations or interleaved output).

### Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size in bytes of one output page (row-major stick) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles per tile-row (width of input block in tiles) |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages per row (1 for interleaved, >1 for width/block sharded) |
| 5 | output_element_size | uint32_t | Bytes per element (2 for bf16, 4 for f32) |
| 6 | num_cols_per_input_block | uint32_t | `num_tiles_per_input_block * tile_width` (elements per input block row) |
| 7 | num_cols_per_output_block | uint32_t | `output_page_width` (elements per output page) |
| 8+ | TensorAccessor args | varies | Appended via `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-rows this core writes |
| 2 | height_wise_input_block_start_index | uint32_t | Starting tile-row index for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Actual (non-padded) columns in the input block |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column-index |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within the first output page |

### Writer Logic: Step by Step

The writer processes blocks in a loop over `num_input_blocks_to_process`. For each block:

1. **Wait for data**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- waits for one tile-row of untilized data in the output CB.

2. **Get base L1 address**: `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- the starting address of the row-major stick data in L1.

3. **Iterate over tile_height rows** (j = 0 to tile_height-1):
   - Compute `current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size` -- the L1 address of stick j within this block.
   - Compute `output_page_id = (block_height_index * tile_height + j) * num_output_blocks_across_width + width_wise_output_block_start_index` -- the DRAM page to write to.
   - **Inner loop**: Iterate while `num_input_cols_processed < num_unpadded_cols_per_input_block`:
     - Compute `num_cols_to_write = min(remaining_input_cols, remaining_output_block_cols)`
     - Get NOC address: `dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)`
     - Perform write: `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`
     - Advance pointers and page IDs

4. **Barrier and pop**: `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.

### TensorAccessor Usage in Writer

The writer creates a TensorAccessor on the device side:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The accessor's `get_noc_addr(page_id, byte_offset)` is used to:
- Map a logical page ID to the correct DRAM bank and offset
- Apply an optional byte offset within the page (for partial writes to width/block-sharded output pages)

This two-argument form (`page_id, offset`) is essential when a single input block spans multiple output pages or starts in the middle of an output page.

### Writer Pattern Summary for rms_norm Reference

For an rms_norm outputting row-major data to interleaved DRAM:
- `num_output_blocks_across_width = 1` (entire tensor width is one output page)
- `output_stick_size = tensor_width * element_size`
- Each page is one full row of the output tensor
- The writer simply writes `tile_height` sticks per block, one `noc_async_write` per stick
- Page ID = `row_index` (simple sequential)
- No partial-page logic needed (no width/block sharding concerns)
- The inner while-loop executes exactly once per stick (input block width == output block width)

### Simplified Writer Pattern (Interleaved Output)

When both input and output are interleaved (the common case for rms_norm), the writer simplifies to:
```
for each block (tile-row):
    cb_wait_front(output_cb, tiles_per_row)
    base_addr = get_read_ptr(output_cb)
    for j in 0..tile_height:
        l1_addr = base_addr + j * stick_size_bytes
        page_id = block_start_row + j
        noc_addr = tensor_accessor.get_noc_addr(page_id)
        noc_async_write(l1_addr, noc_addr, stick_size_bytes)
    noc_async_write_barrier()
    cb_pop_front(output_cb, tiles_per_row)
```

This is the pattern a rms_norm writer should follow when writing untilized output sticks.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 (interleaved, multi-block) | 2 * Wt tiles | Wt tiles | 2x | Double-buffered |
| c_0 (interleaved, single-block) | Wt tiles | Wt tiles | 1x | Single-buffered |
| c_0 (sharded) | full shard | Wt tiles | Nx | Multi-buffered (entire shard) |
| c_16 (multi-block) | 2 * Wt tiles | Wt tiles | 2x | Double-buffered |
| c_16 (single-block) | Wt tiles | Wt tiles | 1x | Single-buffered |

Where Wt = `num_tiles_per_input_block` (tiles along the width of one input block).

Double-buffering enables compute-writer overlap: while the writer drains one tile-row of sticks from the output CB, the compute kernel can fill the next tile-row.

## Index Calculations

### Page ID Calculation (Writer)

For each stick within a block, the output page ID is computed as:
```
num_rows_already_processed = block_height_index * tile_height + j
page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index
```

For interleaved output (`num_output_blocks_across_width = 1`, `width_wise_output_block_start_index = 0`):
```
page_id = block_height_index * tile_height + j
```
This is simply the global row index.

### L1 Read Address Calculation (Writer)

Within the output CB, sticks are laid out contiguously after untilize:
```
stick_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
```
Where `j` is the row within the tile-height (0 to 31).

### TensorAccessor Page-to-Bank Mapping

The TensorAccessor maps `page_id` to a physical DRAM bank and offset. For interleaved tensors, pages are distributed round-robin across DRAM banks. The accessor encapsulates this mapping so the kernel only needs to call `get_noc_addr(page_id)`.

## Memory Access Patterns

### Read Pattern (Writer reading from output CB)
- **Sequential within a block**: Sticks are read in order from L1 (row 0, row 1, ..., row 31)
- **Stride**: `num_cols_per_input_block * output_element_size` bytes between consecutive sticks
- **Access type**: L1 local read (same core)

### Write Pattern (Writer writing to DRAM)
- **Sequential by row**: Sticks are written in row order to DRAM
- **Interleaved access**: Page IDs increment sequentially, but physical bank assignments alternate (round-robin)
- **Transfer granularity**: One full stick per `noc_async_write` call
- **Barrier**: One `noc_async_write_barrier()` per block (after all `tile_height` sticks are issued)

### Key Pattern for rms_norm
The write pattern issues `tile_height` (32) NoC writes per block, each writing `output_stick_size` bytes. The barrier after the block ensures all writes complete before popping the CB. This is a standard and efficient pattern for writing RM output.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (full cores), `num_rows_per_cliff_core` (cliff core) |
| **Load balancing** | Near-equal; one optional cliff core handles remainder |

### Work Splitting (Interleaved)

`split_blocks_for_tilize(grid_size, num_tiles_per_col)` distributes tile-rows across cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- Full cores get `nblocks_per_core` tile-rows each
- One cliff core gets `num_tiles_per_col % nblocks_per_core` tile-rows (if non-zero)
- The writer's `height_wise_input_block_start_index` gives each core its starting tile-row

### Work Splitting (Sharded)

For sharded input, each core processes its local shard. The core grid comes from the shard spec. There is no cliff core. Cores are iterated in row-major or column-major order depending on `shard_spec.orientation`.

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Bytes per output page (stick) |
| 2 | tile_height | uint32_t | Tile height (32) |
| 3 | num_tiles_per_input_block | uint32_t | Tiles per tile-row (used for cb_wait_front/cb_pop_front) |
| 4 | num_output_blocks_across_width | uint32_t | Output pages per row (1 for interleaved) |
| 5 | output_element_size | uint32_t | Bytes per element |
| 6 | num_cols_per_input_block | uint32_t | Elements per input block row |
| 7 | num_cols_per_output_block | uint32_t | Elements per output page |
| 8+ | TensorAccessor compile-time args | varies | Bank mapping info for output buffer |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_input_blocks_to_process | uint32_t | Tile-rows to write |
| 2 | height_wise_input_block_start_index | uint32_t | First tile-row index |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Valid (non-padding) columns |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page column index |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Byte offset within first output page |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | = `num_tiles_per_input_block` |
| 1 | src_cb_id | uint32_t | Input CB index (c_0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | = `num_input_blocks_to_process` (tile-rows) |

## Kernel Implementations

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks / pack_untilize_variable_num_blocks | RISCV_2 (Unpack+Math+Pack) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | `compute_kernel_lib::untilize<>()` |

- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Selection logic**: Fast pack_untilize is used unless: `use_pack_untilize` is false, dtype is UINT16, or dtype is FLOAT32 with `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`.
- **Key logic**: Both kernels call the same `compute_kernel_lib::untilize<>()` helper. The "slow" vs "fast" distinction is historical; both use the unified helper which internally dispatches based on DEST capacity.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB c_16 (row-major sticks) | DRAM output buffer | Write RM sticks via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key logic**: Lambda `write_tiles_in_current_block` handles one tile-row. Iterates over `tile_height` rows, for each row computes page_id and writes via `noc_async_write`. Handles cross-block writes (input block spanning multiple output pages) via an inner while-loop. Uses `noc_async_write_barrier()` per block.

### Reader Kernel (de-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id (interleaved) | RISCV_0 | NOC0 | DRAM input buffer | CB c_0 | Read tiles via TensorAccessor |
| reader_unary_sharded (sharded) | RISCV_0 | NOC0 | L1 shard buffer | CB c_0 | Signal sharded CB availability |

## Implementation Notes

### Output CB Sizing for rms_norm

For a rms_norm operation that normalizes along the last dimension and produces row-major output:
- The output CB should hold untilized sticks (row-major data)
- **Block size**: `Wt` tiles (= last_dim_size / tile_width), representing one tile-row of RM sticks
- **Capacity**: `Wt * 2` tiles for double-buffering (if processing multiple blocks), or `Wt` for single-buffering (if processing exactly one block)
- **Data format**: matches output dtype
- The CB page size should be set to the tile size for the data format (even though the data is row-major sticks, the CB pages are still tile-sized because the compute kernel writes in tile-sized units via pack_untilize)

### Stick Extraction Pattern

After `pack_untilize_block` processes one tile-row:
- The output CB contains `num_tiles_per_input_block` tiles of data
- Physically, this is `tile_height` sticks, each `num_tiles_per_input_block * tile_width * element_size` bytes
- Sticks are contiguous: stick j starts at offset `j * num_tiles_per_input_block * tile_width * element_size` from the CB read pointer
- The writer reads these sticks and writes them as separate pages to DRAM

### TensorAccessor Pattern for Output Writes

Host-side setup:
```cpp
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```
This appends bank mapping info as compile-time args (default: all static, `ArgConfig::None`).

Device-side usage:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // CTA offset = first index after explicit CTAs
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
// ...
uint64_t noc_addr = s.get_noc_addr(page_id, byte_offset);
noc_async_write(l1_addr, noc_addr, num_bytes);
```

The `8` in `TensorAccessorArgs<8>()` is the compile-time args base index. It equals the number of explicit compile-time args passed before the TensorAccessor args were appended (indices 0-7 in this writer).

### Double Buffering Interaction

The output CB double-buffering enables overlap:
- **Compute** fills slot A with untilized sticks for block N
- **Writer** drains slot B (sticks from block N-1) to DRAM
- They swap roles each iteration via `push_back`/`pop_front` semantics

This is the standard producer-consumer pattern. The writer's `cb_wait_front` blocks until compute's `cb_push_back` makes data available. The compute's `cb_reserve_back` blocks until the writer's `cb_pop_front` frees space.

### Handling Padded vs Unpadded Columns

The `num_unpadded_cols_per_input_block` runtime argument allows the writer to skip writing padding data from the last shard in a row. For interleaved input, this equals `num_cols_per_input_block` (no padding). For sharded input with uneven width sharding, the last shard may have fewer valid columns.

## External Knowledge Sources

### DeepWiki Queries
DeepWiki was unavailable during this analysis (all queries returned errors). Analysis was conducted entirely from source code and in-repo documentation.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side `get_noc_addr` API
   **Key Information**: TensorAccessorArgs wraps buffer metadata; `get_noc_addr(page_id)` maps logical page to physical NOC address; optional byte offset parameter for partial-page writes

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile layout (face ordering), row-major layout, and interleaved page distribution
   **Key Information**: Tiles are 32x32 with 4 faces of 16x16; RM pages are one row each; interleaved pages are round-robin across banks

3. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding reader/compute/writer kernel model and circular buffer synchronization
   **Key Information**: Reader produces to CB, compute consumes/produces, writer consumes; CB ops: reserve_back/push_back (producer), wait_front/pop_front (consumer)

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the untilize compute helper API, dispatch paths, and CB synchronization
   **Key Information**: Unified `untilize<>()` template handles init/uninit, waiting, and block splitting; two paths based on DEST capacity; manages all CB ops internally

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST_AUTO_LIMIT and how it affects untilize block splitting
   **Key Information**: DEST capacity ranges from 4-16 tiles depending on sync mode and accumulation mode

6. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` work distribution
   **Key Information**: Distributes tile-rows across cores with full cores + optional cliff core; returns core ranges and blocks-per-core counts

7. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding `create_cb` helper for circular buffer creation
   **Key Information**: Wraps `CircularBufferConfig` creation; accepts optional buffer pointer for sharded CBs; sets page size and total size

8. **Source**: `tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
   **Reason**: Understanding host-side TensorAccessorArgs class
   **Key Information**: `append_to(compile_time_args)` appends accessor metadata as compile-time args; default `ArgConfig::None` means all args are compile-time
