# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tiled-layout tensor data back to row-major (RM) format. It reads tiles from an input buffer, uses hardware-accelerated pack_untilize (or a slower software path) to extract rows from tiles, and writes the resulting RM sticks to an interleaved DRAM output buffer.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus of this analysis**: Output-stage reference for a new `layer_norm_rm` operation -- specifically, the untilize compute helper signature and usage, the writer kernel pattern for writing RM sticks to DRAM, output CB sizing, and stick extraction from tiles.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row across the width) |
| **Unit size** | `num_tiles_per_input_block` tiles (= `tensor_width / tile_width` for interleaved) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Each core processes `num_input_blocks_per_full_core` blocks sequentially |

One "block" represents one full-width tile row. For a tensor with width W and tile_width 32, each block is `W/32` tiles. Each such block contains `tile_height` (32) RM sticks.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[..., H, W]` (arbitrary batch dims) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | bfloat16, float32, int32, uint32, uint16 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (primary focus), or height/width/block-sharded |
| **Buffer type** | DRAM (interleaved) |
| **Data type** | Same as input (or converted via fp32_dest_acc_en) |

### Key Dimension Variables (Interleaved Case)

```
tensor_width  = a.padded_shape()[-1]           // total columns
tensor_height = a.physical_volume() / tensor_width  // total rows
tile_height   = 32   (standard tile)
tile_width    = 32   (standard tile)
num_tiles_per_row = tensor_width / tile_width   // tiles across width
num_tiles_per_col = tensor_height / tile_height // tile-rows down height
```

### Output Page Structure (Critical for Writer)

For interleaved output, the output is organized with **one page per RM stick row**:
- `output_page_width = tensor_width` (the entire tensor row is one output page)
- `output_stick_size = tensor_width * output_element_size` (bytes per page/stick)
- `num_output_blocks_across_width = 1` (single page per row for interleaved)
- `output_element_size = output.element_size()` (2 for bfloat16, 4 for float32)

The output buffer's TensorAccessor is initialized with `output_stick_size` as the page size. Each page_id corresponds to one RM row of the tensor.

---

## Data Flow Pattern

### Simplified Pipeline for Interleaved Input/Output

```
Stage 1: Reader reads tiles from DRAM -> CB_in (c_0)
   - Tiles read sequentially by tile_id: [start_id .. start_id + num_tiles)
   - One tile at a time: reserve_back(1), noc_async_read, barrier, push_back(1)

Stage 2: Compute untilizes tiles in CB_in -> CB_out (c_16)
   - Processes one block_width at a time (block = num_tiles_per_input_block tiles)
   - wait_front(block_width), reserve_back(block_width)
   - pack_untilize_block converts tiled data to RM in-place in CB_out
   - pop_front(block_width), push_back(block_width)
   - Repeats for num_blocks (= rows of tile-blocks assigned to this core)

Stage 3: Writer reads RM sticks from CB_out -> writes to DRAM
   - wait_front(num_tiles_per_input_block)
   - For each of tile_height (32) rows within the block:
     - Compute row address in CB_out
     - Write stick bytes to DRAM via TensorAccessor
   - noc_async_write_barrier()
   - pop_front(num_tiles_per_input_block)
```

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Page Size | Capacity (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-----------|-------------------|-----------|----------|----------|----------|
| c_0 (0) | cb_src0 | Input tile staging | `input_single_tile_size` | `num_tiles_per_input_block * 2` | Double (if >1 block) | Reader | Compute | Block |
| c_16 (16) | cb_output | Output RM staging | `output_single_tile_size` | `num_tiles_per_input_block * 2` | Double (if >1 block) | Compute | Writer | Block |

### CB Sizing Details

**Input CB (c_0)**:
- If only 1 block per core: capacity = `num_tiles_per_input_block` (single-buffered)
- If 2+ blocks per core: capacity = `num_tiles_per_input_block * 2` (double-buffered)
- Page size = `input_single_tile_size` = `tt::tile_size(input_cb_data_format)`

**Output CB (c_16)** -- the focus for output_stage reference:
- Same double-buffering logic as input CB
- Page size = `output_single_tile_size` = `tt::tile_size(output_cb_data_format)`
- **Critical**: The output CB page_size is tile-sized, NOT stick-sized. The pack_untilize hardware writes RM data into tile-sized pages. The writer then extracts individual sticks from these tile-sized pages.
- Capacity in bytes = `output_cb_num_tiles * output_single_tile_size`

### Output CB Memory Layout After Untilize

After `pack_untilize_block` writes to CB_out, the data in CB_out is in RM order but occupies `num_tiles_per_input_block` tile-sized pages. The layout within these pages is:

```
For a block of Wt tiles (Wt = num_tiles_per_input_block):
  Row 0: [tile0_row0 | tile1_row0 | ... | tileWt-1_row0]  = Wt * tile_width elements
  Row 1: [tile0_row1 | tile1_row1 | ... | tileWt-1_row1]  = Wt * tile_width elements
  ...
  Row 31: [tile0_row31 | tile1_row31 | ... | tileWt-1_row31]
```

The total number of contiguous bytes is `tile_height * num_tiles_per_input_block * tile_width * element_size`. The writer accesses this as `tile_height` rows, each of width `num_cols_per_input_block * output_element_size` bytes.

---

## Pipeline Pattern Summary

For the interleaved multi-block case (most common), both CBs are double-buffered:
- **CB c_0**: Double-buffered -- reader can fill block N+1 while compute processes block N
- **CB c_16**: Double-buffered -- compute can produce block N+1 while writer drains block N

For single-block-per-core cases, both are single-buffered (no overlap needed).

---

## Index Calculations

### Writer Index Mapping (Key for Output Stage)

The writer computes the output page_id (which RM row) and an optional byte offset within that page:

```c++
// For each block:
uint32_t block_height_index;  // which tile-row this block represents (global)

// For each row j within the tile-block (j = 0..tile_height-1):
uint32_t num_rows_already_processed = block_height_index * tile_height + j;
uint32_t output_page_id = num_rows_already_processed * num_output_blocks_across_width
                         + width_wise_output_block_start_index;
```

For interleaved output (the simple case):
- `num_output_blocks_across_width = 1`
- `width_wise_output_block_start_index = 0`
- `num_cols_already_processed_in_first_output_block = 0`
- Therefore: `output_page_id = num_rows_already_processed` (simply the row index)

The L1 read address for each row within the block:
```c++
uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);
uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
```

---

## Memory Access Patterns

### Read Pattern (Reader -> CB_in)
- Sequential tile reads by tile_id
- Each tile read: `noc_async_read` of `tile_bytes` from DRAM
- Barrier after each individual tile read (not batched)

### Write Pattern (CB_out -> DRAM) -- Primary Focus

For interleaved output, the writer performs:
1. `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- wait for a full block
2. For each of `tile_height` (32) rows:
   - Compute L1 source address: `base_l1_read_addr + j * row_bytes`
   - Compute DRAM dest: `s.get_noc_addr(output_page_id, offset_within_page)`
   - `noc_async_write(l1_addr, noc_addr, num_bytes_to_write)`
3. `noc_async_write_barrier()` -- wait for all 32 writes to complete
4. `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` -- release the block

**Write granularity**: One full RM stick per write (= `tensor_width * element_size` bytes for interleaved).

**Barrier placement**: One barrier per block (after all 32 rows), NOT per individual row write. This allows the NoC to pipeline the 32 writes.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-row blocks |
| **Load balancing** | Equal blocks per full core, last core may be "cliff" |
| **Remainder handling** | Cliff core gets `num_rows_per_cliff_core` blocks (< full) |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` function divides `num_tiles_per_col` tile-rows across available cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- Cliff core gets remainder: `num_tiles_per_col % nblocks_per_core`

---

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output RM stick (= `output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Height of tile (typically 32) |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per block width (= `tensor_width / tile_width`) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages across width (1 for interleaved) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bf16, 4 for f32) |
| 6 | `num_cols_per_input_block` | uint32_t | Columns per input block (= `num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Columns per output page (= `output_page_width`) |
| 8+ | TensorAccessorArgs | varies | Auto-appended by `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-row blocks this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting tile-row index (global) |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual data columns (may be < block width if uneven shard) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page column index (0 for interleaved) |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Byte offset into first output page (0 for interleaved) |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (= `num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to process |

---

## Kernel Implementations

### Compute Kernel: Untilize Helper

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (unpack/math/pack) | N/A | CB c_0 (tiled) | CB c_16 (RM) | pack_untilize |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (slow path) or `pack_untilize_variable_num_blocks.cpp` (fast path)

**Key Logic**:
Both kernels use the unified `compute_kernel_lib::untilize<>` helper:

```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);  // MUST call first
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,    // compile-time: tiles per block
    src_cb_id,                  // compile-time: input CB
    out_cb_id,                  // compile-time: output CB
    InitUninitMode::InitAndUninit,  // init + uninit lifecycle
    WaitMode::WaitBlock,            // wait for input per block
    ReconfigureRegisterDatatypeMode::NoReconfigure  // no reconfig
>(per_core_block_cnt);              // runtime: number of blocks
```

**Helper behavior** (from `untilize_helpers.inl`):
- Determines `DEST_AUTO_LIMIT` at compile time (4 for DST_ACCUM_MODE, 8 otherwise)
- If `block_width_tiles <= dest_limit`: single-pass pack_untilize per block
- If `block_width_tiles > dest_limit`: splits into sub-blocks fitting in DEST
- For each block: `cb_wait_front(input, tiles)`, `cb_reserve_back(output, tiles)`, `pack_untilize_block(...)`, `cb_pop_front(input, tiles)`, `cb_push_back(output, tiles)`

**Usage pattern for layer_norm_rm**: If the untilize is the final stage of a fused compute kernel, use `WaitMode::NoWait` (data is already in the CB from a prior compute stage) and `InitUninitMode` appropriately based on whether other operations precede/follow.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM) | DRAM (interleaved) | noc_async_write |

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

**Key Logic** (simplified for interleaved case):

```cpp
// Setup TensorAccessor from compile-time args starting at index 8
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);

// For each block assigned to this core:
for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
    cb_wait_front(cb_id_out0, num_tiles_per_input_block);
    uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

    // Extract tile_height (32) RM sticks from the block
    for (uint32_t j = 0; j < tile_height; ++j) {
        uint32_t l1_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;
        uint32_t row_index = block_start * tile_height + j;
        uint32_t page_id = row_index;  // for interleaved: page_id = row index

        uint64_t dst_noc_addr = s.get_noc_addr(page_id, 0);
        noc_async_write(l1_addr, dst_noc_addr, output_stick_size);
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, num_tiles_per_input_block);
}
```

**Key observations for reuse**:
1. `get_read_ptr(cb_id)` returns the L1 address of the front of the CB
2. Rows are laid out contiguously: row `j` starts at `base + j * row_stride_bytes`
3. Row stride = `num_cols_per_input_block * element_size` (NOT `tensor_width * element_size` -- these are the same for interleaved but differ for sharded)
4. The barrier is per-block (batches 32 writes), not per-row
5. `cb_pop_front` only after the barrier completes

### Reader Kernel (De-emphasized)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

Simple tile-by-tile reader: loops over `[start_page_id, start_page_id + num_tiles)`, reading one tile at a time with `noc_async_read` and individual barriers.

---

## Implementation Notes

### Output CB Page Size Is Tile-Sized, Not Stick-Sized

This is a critical design detail. The output CB (c_16) has `page_size = output_single_tile_size` (e.g., 2048 bytes for bfloat16 32x32 tiles). The `pack_untilize` hardware writes RM data into these tile-sized pages. The writer kernel then treats the CB memory as a flat buffer of RM rows, computing row offsets manually via `get_read_ptr()` + arithmetic. The CB page size being tile-sized is required because `pack_untilize_block` operates on tile-granularity pages.

### Writer Handles Sharded Output Width Splitting

The writer is designed to handle cases where input shards and output shards have different widths. The `while (num_input_cols_processed < num_unpadded_cols_per_input_block)` loop can split a single RM row across multiple output pages. For the interleaved case, this loop executes exactly once (the entire row is one page).

### TensorAccessor Setup for Writer

```cpp
// Host side (program factory):
std::vector<uint32_t> writer_compile_time_args = { cb_id, stick_size, tile_h, ... };
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

// Device side (writer kernel):
constexpr auto dst_args = TensorAccessorArgs<8>();  // CTA offset = 8 (after 8 explicit args)
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
// output_stick_size = page_size for the TensorAccessor
uint64_t addr = s.get_noc_addr(page_id, byte_offset_within_page);
```

The `TensorAccessorArgs<CTA_OFFSET>()` template parameter is the index in the compile-time args vector where the accessor's own arguments begin (after all manually-specified compile-time args).

### Compute Kernel Selection

Two compute paths exist:
1. **Fast path** (`pack_untilize`): Hardware-accelerated, ~80 cycles/tile. Used by default.
2. **Slow path** (software untilize): ~390 cycles/tile. Used when:
   - `use_pack_untilize` is false
   - Input dtype is UINT16
   - Input is FLOAT32 AND `num_tiles_per_input_block >= MAX_PACK_UNTILIZE_WIDTH`

Both paths use the same `compute_kernel_lib::untilize<>` helper, which auto-selects internally.

### DST_ACCUM_MODE Define

For INT32, UINT32, or FLOAT32 data types, the compute kernel defines `DST_ACCUM_MODE = 1`. This halves the DEST register capacity from 8 to 4 tiles per pass, requiring more sub-block iterations for wide tensors.

---

## Relevance to layer_norm_rm Output Stage

For a `layer_norm_rm` operation that fuses tilize -> compute -> untilize:

1. **Output CB (untilize destination)**: Use CB c_16 with `page_size = tile_size`, capacity = `Wt * 2` tiles (double-buffered) or `Wt` (single-buffered if only 1 block). Here `Wt = tensor_width / 32`.

2. **Compute kernel untilize call**: After the normalization compute produces results into an intermediate CB (in tile layout), call:
   ```cpp
   compute_kernel_lib::untilize<Wt, cb_compute_out, cb_untilize_out,
       InitUninitMode::..., WaitMode::NoWait, ...>(num_blocks);
   ```
   Use `WaitMode::NoWait` if the previous compute stage already produced data into the input CB. Use appropriate `InitUninitMode` based on lifecycle.

3. **Writer kernel**: Adapt the writer from `writer_unary_stick_layout_split_rows_multi_core.cpp`. For a simple interleaved output:
   - Wait for `Wt` tiles in the output CB
   - Extract 32 RM sticks, each of `tensor_width * element_size` bytes
   - Write each stick to DRAM via `TensorAccessor::get_noc_addr(row_index)`
   - Barrier after all 32 writes, then pop

4. **TensorAccessor page_size**: Initialize with `output_stick_size = tensor_width * element_size` (the full RM row width). Each `page_id` = one RM row.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal? Specifically, how are they used in writer kernels to compute NoC addresses for writing data to DRAM interleaved buffers?"
   **Reason**: The writer kernel uses TensorAccessor for all DRAM writes; needed to understand argument passing and address computation.
   **Key Findings**: TensorAccessorArgs appends configuration to compile-time args on the host. On the device, `TensorAccessor(args, base_addr, page_size)` reconstructs the accessor. `get_noc_addr(page_id, offset)` computes the physical NoC address for a given page and byte offset within it.

2. **Query**: "How does the untilize compute operation work in tt-metal? What does compute_kernel_lib::untilize do step by step?"
   **Reason**: Needed to understand the CB synchronization pattern and how RM data lands in the output CB.
   **Key Findings**: The helper auto-selects between single-pass and block-based pack_untilize based on DEST capacity. For each block: wait input, reserve output, pack_untilize, pop input, push output. Output data is contiguous RM rows within tile-sized CB pages.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor device-side API and page-based addressing
   **Key Information**: `get_noc_addr(page_id, offset=0)` for page-based writes; host setup via `TensorAccessorArgs(buffer).append_to(compile_args)`

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute_kernel_lib::untilize helper signature, template parameters, and internal logic
   **Key Information**: Template params: `<block_width_tiles, input_cb, output_cb, InitUninitMode, WaitMode, ReconfigMode>`. Runtime param: `num_blocks`. Requires `compute_kernel_hw_startup(input_cb, output_cb)` before first call. Handles DEST splitting internally.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility used to create circular buffers
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer=nullptr)`. Returns `{cb_index, cb_handle}`. If `buffer` is provided, the CB is globally allocated (for sharded inputs).

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` which determines core distribution
   **Key Information**: Divides `num_tiles_per_col` tile-row blocks across cores. Returns `{ncores, all_cores, full_cores, cliff_cores, blocks_per_full, blocks_per_cliff}`.
