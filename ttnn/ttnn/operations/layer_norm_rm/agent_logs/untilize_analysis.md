# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **TILE_LAYOUT** (32x32 tile format with faces) to **ROW_MAJOR_LAYOUT** (linear row-major sticks). This analysis focuses on the **multi-core interleaved** program factory variant (`UntilizeMultiCoreProgramFactory`), which reads tiled data from DRAM, untilizes it on compute cores, and writes row-major sticks back to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Role of this analysis**: Output-stage reference for `layer_norm_rm`. The focus is on how the untilize compute helper produces row-major data in the output CB, how the output CB is sized, and how the writer kernel extracts individual RM sticks and writes them to interleaved DRAM pages via TensorAccessor.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = all tiles across the width dimension for one row of tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width`, e.g., for a 128-wide tensor with 32-wide tiles: 4 tiles) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Outer loop: iterate over assigned blocks (tile-rows). Inner: compute untilizes one block; writer extracts `tile_height` (32) RM sticks from the block and writes each to DRAM. |

One "input block" is a horizontal strip of tiles spanning the full tensor width and one tile-height tall. The compute kernel processes one block at a time, and the writer extracts `tile_height` individual row-major sticks from each block.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D, flattened to 2D: `[tensor_height, tensor_width]` |
| **Dimension convention** | Height = `physical_volume / width`, Width = `padded_shape[-1]` |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED (or SHARDED -- this analysis focuses on interleaved) |
| **Buffer type** | DRAM (interleaved) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (pages = one full-width row each) |
| **Buffer type** | DRAM (interleaved) |
| **Data type** | Same as input |

### Layout Transformations

The core transformation is **tile-to-row-major conversion**:
1. Input tiles (32x32, stored as 4 faces of 16x16 in face-order) are read from DRAM into the input CB.
2. The compute kernel uses `pack_untilize` (hardware-accelerated) or `untilize` (software fallback) to convert tile data into row-major format in the output CB.
3. After compute, the output CB contains data arranged as: row 0 of all tiles concatenated, row 1 of all tiles concatenated, ..., row 31 of all tiles concatenated -- i.e., `tile_height` contiguous RM sticks, each `num_tiles_per_input_block * tile_width * element_size` bytes wide.
4. The writer reads these sticks from the output CB and writes them as individual RM pages to interleaved DRAM.

**Critical insight for downstream consumers**: After `pack_untilize_block` / `compute_kernel_lib::untilize`, the output CB already contains **row-major data**. The writer does NOT perform any tile-to-RM conversion -- it simply transfers contiguous byte ranges from L1 to DRAM.

---

## Data Flow Pattern

### Interleaved Input Path (focus of this analysis)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (`reader_unary_start_id.cpp`) | DRAM (interleaved tiles) | CB c_0 (input) | `cb_reserve_back(1)`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back(1)` -- one tile at a time |
| 2 | Compute (`untilize_variable_num_blocks.cpp` or `pack_untilize_variable_num_blocks.cpp`) | CB c_0 (input) | CB c_16 (output) | `cb_wait_front(block_width)`, `cb_reserve_back(block_width)`, `pack_untilize_block`, `cb_pop_front(block_width)`, `cb_push_back(block_width)` |
| 3 | Writer (`writer_unary_stick_layout_split_rows_multi_core.cpp`) | CB c_16 (output) | DRAM (interleaved RM sticks) | `cb_wait_front(block_width)`, `get_read_ptr`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front(block_width)` |

### Output CB Data After Compute (Key Detail)

After the compute kernel processes one block of `num_tiles_per_input_block` tiles:
- The output CB contains `tile_height` (32) concatenated RM sticks
- Each stick is `num_tiles_per_input_block * tile_width * element_size` bytes long
- The total block size in the output CB is `num_tiles_per_input_block * tile_size` bytes (where `tile_size = tile_height * tile_width * element_size`)
- Sticks are contiguous in memory: stick `j` starts at offset `j * num_cols_per_input_block * element_size` from the CB read pointer

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` (double-buffered) or `num_tiles_per_input_block` (single block) | `num_tiles_per_input_block` | Double (if >1 block/core) or Single (if 1 block/core) | Reader | Compute | Block |
| c_16 | cb_output | Output RM stick staging | `num_tiles_per_input_block * 2` (double-buffered) or `num_tiles_per_input_block` (single block) | `num_tiles_per_input_block` | Double (if >1 block/core) or Single (if 1 block/core) | Compute | Writer | Block |

### Output CB Sizing Logic (Focus Area)

From the program factory (lines 149-163):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Key sizing properties**:
- **Page size** = `output_single_tile_size` (one tile's worth of bytes, e.g., 2048 bytes for BF16 32x32 tiles)
- **Capacity** = `output_cb_num_tiles * output_single_tile_size` bytes
- **Block size for CB operations** = `num_tiles_per_input_block` tiles (one full-width row of tiles)
- **Double-buffering** is enabled when the core processes more than one block, allowing compute to fill one block while the writer drains another

**Physical interpretation**: The output CB holds either 1 or 2 complete RM-stick-blocks. Each block contains `tile_height` (32) RM sticks concatenated. Double-buffering allows the compute kernel to start untilizing the next block into one half of the CB while the writer is still draining the previous block from the other half.

**Relevance to layer_norm_rm**: For a new operation that does compute -> untilize -> write, the output CB should be sized at `Wt * 2` tiles (where `Wt = tensor_width / tile_width`) for double-buffering, with the CB page size set to the output tile size.

---

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 (input) | 2 * block_width (typical) | block_width | 2:1 | Double-buffered |
| c_16 (output) | 2 * block_width (typical) | block_width | 2:1 | Double-buffered |

Both CBs use the same double-buffering strategy: capacity = 2x block size when more than 1 block is processed per core. This allows reader/compute overlap and compute/writer overlap respectively.

---

## Index Calculations

### Writer's Stick-to-Page Mapping (Focus Area)

The writer kernel must map each RM stick from the output CB to the correct page in the output DRAM buffer. The key index calculations are:

**Per-block setup** (in the `write_tiles_in_current_block` lambda):
```
base_l1_read_addr = get_read_ptr(cb_id_out0)   // Start of CB data
```

**Per-stick within a block** (inner loop over `j` in `0..tile_height`):
```
current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
```
This computes the L1 address of stick `j` within the current block.

**Output page_id calculation**:
```
num_rows_already_processed = block_height_index * tile_height + j
num_pages_already_processed_in_previous_rows = num_rows_already_processed * num_output_blocks_across_width
output_page_id = num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index
```

For the **simple interleaved case** (no width/block sharding), `num_output_blocks_across_width = 1` and `width_wise_output_block_start_index = 0`, so:
```
output_page_id = (block_height_index * tile_height + j) * 1 + 0
               = global_row_index
```

This means each RM stick maps directly to one page in the interleaved output buffer, and the page_id equals the global row index.

### TensorAccessor Address Resolution

The writer constructs a `TensorAccessor` from compile-time args starting at index 8:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

To get the DRAM address for a write:
```cpp
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

`get_noc_addr(page_id, offset)` resolves the page_id to a physical bank and bank-offset (round-robin distribution across DRAM banks), then adds the byte offset within the page. For the simple interleaved case, `output_offset_within_page_in_bytes = 0` and the full stick is written in one `noc_async_write`.

---

## Memory Access Patterns

### Read Pattern (De-emphasized per focus)
- Reader reads tiles sequentially by tile_id from interleaved DRAM
- One tile per `noc_async_read` call with barrier after each tile

### Write Pattern (Focus Area)

**Ordering**: Row-major sequential. The writer processes blocks in order of ascending `height_wise_input_block_index`. Within each block, it writes sticks in order `j = 0, 1, ..., tile_height-1`.

**Access pattern per block**:
1. `cb_wait_front(num_tiles_per_input_block)` -- wait for compute to finish one block
2. `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- get CB read pointer
3. For each of the 32 sticks in the block:
   - Compute `current_l1_read_addr` (stride = `num_cols_per_input_block * element_size`)
   - Compute `output_page_id` (global row index for interleaved case)
   - `noc_async_write(current_l1_read_addr, dst_noc_addr, output_stick_size)` -- write one full stick
4. `noc_async_write_barrier()` -- wait for all 32 writes to complete
5. `cb_pop_front(num_tiles_per_input_block)` -- release the block

**Write granularity**: One complete RM stick per `noc_async_write` call. The stick size is `tensor_width * element_size` bytes for interleaved output.

**Barrier placement**: One `noc_async_write_barrier()` per block (after all 32 sticks are issued). This batches the barrier across `tile_height` writes for efficiency.

**L1 read stride**: The writer reads from the output CB with a stride of `num_cols_per_input_block * element_size` bytes between consecutive sticks. This is exactly the concatenated width of all tiles in the block.

### Width-Split Write Pattern (Sharded Output)

When the output is width-sharded or block-sharded, the writer may need to split a single input-block row across multiple output pages. The inner `while` loop handles this:
```
while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
    num_cols_to_write = min(remaining_input, remaining_output_page);
    noc_async_write(l1_addr, noc_addr, num_cols_to_write * element_size);
    // advance pointers...
}
```
This handles the case where a single RM stick from the CB spans multiple output pages. For the `layer_norm_rm` use case with interleaved output, this loop executes exactly once per stick (the entire stick fits in one page).

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores) |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` blocks (tile-rows) per full core |
| **Load balancing** | Near-equal distribution with one optional cliff core |
| **Remainder handling** | Last core (cliff core) processes `num_rows_per_cliff_core` blocks |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` function distributes `num_tiles_per_col` blocks across the available compute grid:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- If `num_tiles_per_col % nblocks_per_core != 0`, one cliff core handles the remainder

For interleaved input, each core gets a contiguous range of tile-rows. The `tile_start_index` is incremented by `num_tiles_per_input_block * num_input_blocks_per_full_core` for each core to give it a contiguous chunk of tiles.

---

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size of one RM output stick in bytes (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Height of one tile (32) |
| 3 | `num_tiles_per_input_block` | uint32_t | Number of tiles per block width (`tensor_width / tile_width`) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved, >1 for width/block sharded) |
| 5 | `output_element_size` | uint32_t | Size of one output element in bytes (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Number of columns per input block (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Number of columns per output page (`output_page_width`) |
| 8+ | TensorAccessor args | uint32_t[] | Bank distribution info for output buffer (appended by `TensorAccessorArgs(*dst_buffer).append_to(...)`) |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Base address of the output buffer in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-row blocks this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting block index (tile-row index) for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Number of valid (non-padding) columns in the input block |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page index in the width dimension (0 for interleaved) |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset into the first output page (0 for interleaved) |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Number of tiles per block width (`num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) this core processes |

---

## Kernel Implementations

### Compute Kernel: `compute_kernel_lib::untilize` helper

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (unpack/math/pack) | N/A | CB c_0 | CB c_16 | `pack_untilize_block` (hardware-accelerated) or `untilize_block` (software fallback) |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)

**Helper signature and usage** (from `untilize_helpers.hpp`):
```cpp
template <
    uint32_t block_width_tiles,   // Tiles per row (compile-time)
    uint32_t input_cb,            // Input CB index
    uint32_t output_cb,           // Output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);  // num_blocks = runtime
```

**Template parameters**:
- `block_width_tiles`: Must match `num_tiles_per_input_block`. This is the width of the tile-row in tiles. Determines whether block-based pack (splitting into DEST-sized sub-blocks) or single-pass pack is used.
- `input_cb` / `output_cb`: CB indices (c_0 and c_16 respectively).
- `InitUninitMode::InitAndUninit`: Default -- calls both `pack_untilize_init` and `pack_untilize_uninit`. For multi-phase usage in a fused kernel, use `InitOnly` / `Neither` / `UninitOnly` to avoid redundant init/uninit between phases.
- `WaitMode::WaitBlock`: Default -- the helper internally calls `cb_wait_front(input_cb, block_width_tiles)` per block. Use `WaitUpfront` if all input data is already available, or `NoWait` if the caller manages synchronization.
- `ReconfigureRegisterDatatypeMode::NoReconfigure`: Default -- no data format register reconfiguration. Use `UnpackAndPackReconfigure` when switching data formats mid-kernel (e.g., after a different compute phase that used different formats).

**Prerequisite**: Must call `compute_kernel_hw_startup(input_cb, output_cb)` before using the helper.

**Internal dispatch logic** (from `untilize_helpers.inl`):
- If `block_width_tiles <= DEST_AUTO_LIMIT` (typically 16 for half-DEST or 8 for full-DEST): uses single-pass `pack_untilize_block<block_width_tiles>` -- one shot per block.
- If `block_width_tiles > DEST_AUTO_LIMIT`: splits into sub-blocks. Finds the largest divisor of `block_width_tiles` that fits in DEST, processes `num_sub_blocks` sub-blocks per row. The input CB is consumed in sub-block-sized chunks while the output CB reservation covers the full block width.

**Key CB operations inside the helper** (single-pass path):
```
for each block r in 0..num_blocks:
    cb_wait_front(input_cb, block_width_tiles)      // Wait for reader
    cb_reserve_back(output_cb, block_width_tiles)    // Reserve output space
    pack_untilize_block(input_cb, 1, output_cb, 0)   // HW-accelerated untilize
    cb_pop_front(input_cb, block_width_tiles)        // Release input
    cb_push_back(output_cb, block_width_tiles)       // Signal writer
```

**Output data format**: After `pack_untilize_block`, the output CB contains `tile_height` (32) contiguous RM sticks, each `block_width_tiles * tile_width * element_size` bytes long. The hardware packer performs the tile-to-RM format conversion.

**Relevance to layer_norm_rm**: In a fused kernel that does tilize -> compute -> untilize, the untilize helper can be called as the final phase. Use `InitUninitMode::InitOnly`/`UninitOnly` if other compute phases precede it, and `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure` if the preceding phase used different data format registers. The `WaitMode::NoWait` mode is useful if the preceding compute phase pushes directly to the untilize input CB (which would be an intermediate CB).

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved RM) | `cb_wait_front`, `get_read_ptr`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front` |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic**:

The writer's core pattern is a nested loop:
1. **Outer loop** (`num_input_blocks_to_process` iterations): one iteration per tile-row block
2. **Per-block**: Wait for `num_tiles_per_input_block` tiles in the output CB
3. **Inner loop** (`tile_height` iterations): extract each RM stick
4. **Innermost while-loop**: handle potential page splits (for sharded output; executes once for interleaved)

**Stick extraction from tiles** (the critical detail):

After `cb_wait_front(cb_id_out0, num_tiles_per_input_block)`, the output CB read pointer points to a contiguous region containing `tile_height` RM sticks laid end-to-end. The writer extracts stick `j` at:
```
L1_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size
```

For interleaved output with `num_output_blocks_across_width = 1`:
```
page_id = (block_height_index * tile_height + j)   // = global row index
noc_addr = TensorAccessor.get_noc_addr(page_id, 0)
noc_async_write(L1_addr, noc_addr, output_stick_size)
```

The entire stick (`output_stick_size = tensor_width * element_size` bytes) is written in a single NoC transaction.

**TensorAccessor construction**:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // CTA offset 8 (after 8 explicit compile-time args)
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The `TensorAccessorArgs<8>()` on the device side reads compile-time arguments starting at index 8 to reconstruct the bank distribution information. The `TensorAccessor` then provides `get_noc_addr(page_id, offset)` which resolves the page to a physical DRAM bank (round-robin) and computes the physical NoC address.

### Reader Kernel (De-emphasized)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

Reads tiles sequentially from interleaved DRAM, one tile per NoC read, into the input CB. Uses `TensorAccessorArgs<1>()` for address resolution.

---

## Implementation Notes

### Fast vs. Slow Untilize Path Selection

The program factory selects between two compute kernel variants (lines 232-244):
- **Fast path** (`pack_untilize_variable_num_blocks.cpp`): Uses hardware-accelerated `pack_untilize`. Default for most data types.
- **Slow path** (`untilize_variable_num_blocks.cpp`): Used when:
  - `use_pack_untilize` is false (user override)
  - Data type is `UINT16`
  - Data type is `FLOAT32` AND `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`

Both paths now use the same `compute_kernel_lib::untilize` helper, which internally selects between `pack_untilize_block` (hardware path) and block-based sub-splitting.

### DST_ACCUM_MODE for 32-bit Types

When input dtype is INT32, UINT32, or FLOAT32, the define `DST_ACCUM_MODE=1` is set. This halves the effective DEST register capacity (from 16 to 8 tile slots) because 32-bit types occupy twice the space. The untilize helper accounts for this via `DEST_AUTO_LIMIT`.

### fp32_dest_acc_en

When enabled, the unpack stage uses `UnpackToDestFp32` mode for the input CB. This forces the DEST accumulator to operate in FP32 precision, which is important for numerical accuracy with FP32 inputs.

### Output Page Width and Sharding Interactions

For **interleaved output**, `output_page_width = tensor_width`, meaning each RM page is the full tensor width. The writer writes each stick as one contiguous NoC write.

For **width-sharded or block-sharded output**, `output_page_width` is the shard width, and `num_output_blocks_across_width > 1`. The writer's inner while-loop splits each stick across multiple output pages.

For the `layer_norm_rm` use case with interleaved output, the simple path applies: `output_page_width = tensor_width`, `num_output_blocks_across_width = 1`, `width_wise_output_block_start_index = 0`, `num_cols_already_processed_in_first_output_block = 0`.

### Reuse Pattern for layer_norm_rm

To adapt this output stage pattern for `layer_norm_rm`:
1. **Output CB**: Size at `Wt * 2` tiles (double-buffered) with CB index c_16, data format matching output dtype, page size = tile size
2. **Untilize helper call**: After the final compute phase, call `compute_kernel_lib::untilize<Wt, intermediate_cb, output_cb, ...>(num_blocks)` with appropriate `InitUninitMode` and `ReconfigureRegisterDatatypeMode` settings
3. **Writer kernel**: Reuse or adapt the writer pattern: wait for `Wt` tiles, extract `tile_height` sticks using `get_read_ptr + j * Wt * tile_width * elem_size`, write each stick via `TensorAccessor.get_noc_addr(page_id)`
4. **TensorAccessor setup**: Append `TensorAccessorArgs(*dst_buffer)` to writer compile-time args, pass `dst_addr` as runtime arg index 0

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation's writer kernel convert tile-format data to row-major sticks and write them to DRAM?"
   **Reason**: Needed to confirm whether the output CB contains tile-format or row-major data after the compute stage.
   **Key Findings**: After `pack_untilize` / `llk_pack`, the output CB already contains row-major data. The writer kernel's role is purely to transfer these RM sticks from L1 to DRAM -- it does NOT perform any format conversion.

2. **Query**: "What is TensorAccessorArgs and how is it used in tt-metal program factories?"
   **Reason**: Needed to understand the host-side setup and device-side consumption of tensor accessor arguments, particularly the template index mechanism.
   **Key Findings**: Host-side `TensorAccessorArgs(*buffer).append_to(compile_args)` serializes bank distribution info into the compile-time args vector. Device-side `TensorAccessorArgs<N>()` reads from compile-time args starting at index N. The `TensorAccessor(args, base_addr, page_size)` provides `get_noc_addr(page_id, offset)` for address resolution.

3. **Query**: "After pack_untilize_block writes data to the output circular buffer, what is the memory layout?"
   **Reason**: Critical to understand the exact byte layout in the output CB for the writer's stick extraction logic.
   **Key Findings**: Data is laid out as row 0 of all tiles concatenated, row 1 concatenated, ..., row 31 concatenated. This is the standard RM format where each "row" is `block_width_tiles * tile_width` elements wide.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page structure for interleaved output.
   **Key Information**: In RM layout, each row of the 2D tensor is one page. Pages are distributed round-robin across DRAM banks in interleaved mode. For a 64x64 tensor, there are 64 pages.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API for DRAM writes.
   **Key Information**: `get_noc_addr(page_id, offset)` resolves a page to its physical DRAM bank and address. The offset parameter allows writing to a byte offset within a page (used for sharded output splits).

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute helper's template parameters, dispatch logic, and CB interaction pattern.
   **Key Information**: The helper has 6 template parameters controlling block width, CB indices, init/uninit lifecycle, wait mode, and register reconfiguration. It auto-dispatches between single-pass and block-split paths based on DEST capacity. The `WaitMode::NoWait` and `InitUninitMode` enum allow integration into fused kernels.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding core distribution strategy.
   **Key Information**: `split_blocks_for_tilize(grid_size, nblocks)` divides blocks evenly across cores, with one optional cliff core for the remainder.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB with total size = `num_pages * page_size`. If `buffer != nullptr`, the CB is backed by a globally allocated buffer (used for sharded input).
