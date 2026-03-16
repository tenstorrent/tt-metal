# Untilize Multi-Core Implementation Analysis (Output Stage Reference)

## Overview

The untilize operation converts tiled (32x32) tensor data back into row-major format. This analysis focuses on the **output stage** aspects relevant to a new `layer_norm_rm` operation: the compute-kernel untilize helper, the writer kernel pattern for writing row-major sticks to DRAM, and the output CB sizing strategy.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Scope**: This is an output_stage reference analysis. Reader kernel details, input CB configuration, and compute kernel internals are de-emphasized.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row (block) |
| **Unit size** | `num_tiles_per_input_block` tiles (= one tile row across the width) |
| **Total units** | `num_tiles_per_col` blocks (= `tensor_height / tile_height`) |
| **Loop structure** | Outer: iterate over blocks assigned to this core. Inner: writer iterates over `tile_height` rows within each block. |

A "block" here is one tile-row: `num_tiles_per_row` tiles spanning the full tensor width. The compute kernel processes one block at a time through the untilize helper, and the writer kernel extracts `tile_height` (32) row-major sticks per block.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | N-D (flattened to 2D: height x width) |
| **Dimension convention** | Last dim = width; all others collapsed into height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED (or SHARDED -- this analysis focuses on interleaved) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Configurable (bfloat16, float32, int32, uint32) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED (DRAM) -- other sharding modes supported |
| **Buffer type** | DRAM (interleaved) |
| **Data type** | Same as input (or configurable) |
| **Page size** | `output_page_width * element_size` bytes (one row-major stick) |

For interleaved output, `output_page_width = tensor_width` (the full row). Each page/stick is one row of the tensor. Pages are round-robin distributed across DRAM banks.

### Layout Transformations

The compute kernel converts tiled data in CB c_0 into row-major data in CB c_16. Each block of `num_tiles_per_input_block` tiles (one tile-row) produces `tile_height` (32) contiguous row-major sticks in the output CB.

---

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front` / `cb_pop_front` on c_0; `cb_reserve_back` / `cb_push_back` on c_16 |
| 3 | Writer | CB c_16 (output) | DRAM (interleaved) | `cb_wait_front`, `noc_async_write`, `cb_pop_front` |

**Key output-stage data flow**:
1. Compute produces `num_tiles_per_input_block` tiles worth of row-major data in c_16 per block
2. Writer waits for one block in c_16 (`cb_wait_front(cb_id_out0, num_tiles_per_input_block)`)
3. Writer reads L1 data linearly, extracting `tile_height` sticks
4. Each stick is written to DRAM via `noc_async_write` using TensorAccessor for address generation
5. After all sticks in the block are written, a write barrier is issued and the CB is popped

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles (tiled) | `num_tiles_per_input_block * 2` tiles (if multi-block) or `num_tiles_per_input_block` (if single-block) | `num_tiles_per_input_block` tiles | Single or Double | Reader | Compute | Block |
| c_16 | cb_output | Output sticks (row-major) | `num_tiles_per_input_block * 2` tiles (if multi-block) or `num_tiles_per_input_block` (if single-block) | `num_tiles_per_input_block` tiles | Single or Double | Compute | Writer | Block |

### Output CB Sizing (Critical for layer_norm_rm)

The output CB (c_16) sizing logic from the program factory (lines 148-162):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;      // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;  // Double-buffered
}
```

**Key insight**: The output CB capacity is expressed in tile units even though its content is row-major. The `output_single_tile_size` (`tt::tile_size(output_cb_data_format)`) determines the byte size per tile-equivalent page in the CB. This is because the untilize hardware operation produces data that fills exactly the same number of tile-sized pages in the output CB.

**For layer_norm_rm reference**: If your operation produces row-major output, the output CB should be sized as `Wt * output_tile_size` bytes per block (where `Wt = tensor_width / 32` is tiles per row), with optional double-buffering if multiple blocks are processed per core.

---

## Pipeline Pattern Summary

When `num_input_blocks_per_full_core > 1`:
- Both c_0 and c_16 are double-buffered (capacity = 2x block size)
- Reader can fill next block in c_0 while compute processes current block
- Compute can produce next block in c_16 while writer drains current block

When `num_input_blocks_per_full_core == 1`:
- Both c_0 and c_16 are single-buffered
- No overlap needed since only one block is processed

---

## The untilize Helper: Signature and Usage

### Header: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`

```cpp
template <
    uint32_t block_width_tiles,   // Tiles per row (compile-time)
    uint32_t input_cb,            // Input CB index (tiled data)
    uint32_t output_cb,           // Output CB index (row-major output)
    untilize_config::InitUninitMode init_uninit_mode = InitAndUninit,
    untilize_config::WaitMode wait_mode = WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure>
ALWI void untilize(uint32_t num_blocks);
```

### Template Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_width_tiles` | `uint32_t` (compile-time) | Number of tiles per row. Must be > 0. |
| `input_cb` | `uint32_t` (compile-time) | Input circular buffer index (0-31). Contains tiled data. |
| `output_cb` | `uint32_t` (compile-time) | Output circular buffer index (0-31). Receives row-major data. Must differ from `input_cb`. |
| `init_uninit_mode` | enum | Controls init/uninit lifecycle. Default: `InitAndUninit`. |
| `wait_mode` | enum | Input synchronization strategy. Default: `WaitBlock`. |
| `reconfig_mode` | enum | Register datatype reconfiguration. Default: `UnpackAndPackReconfigure`. |

### Runtime Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_blocks` | `uint32_t` | Number of tile-rows to process |

### Usage in Untilize Compute Kernel

From `untilize_variable_num_blocks.cpp`:

```cpp
#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);

    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);   // PREREQUISITE
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        src_cb_id,
        out_cb_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
```

### Internal Dispatch Logic

The helper selects between two paths at compile time based on `block_width_tiles` vs `DEST_AUTO_LIMIT`:

1. **Single-pass path** (`block_width_tiles <= DEST_AUTO_LIMIT`): Processes entire tile-row at once using `pack_untilize_block<block_width_tiles, block_width_tiles>`.
2. **Block-based path** (`block_width_tiles > DEST_AUTO_LIMIT`): Splits wide rows into sub-blocks. Finds the largest divisor of `block_width_tiles` that fits in DEST, processes sub-blocks sequentially.

For each block (tile-row), the processing loop is:
```
cb_wait_front(input_cb, block_width_tiles)    // Wait for tiled input
cb_reserve_back(output_cb, block_width_tiles)  // Reserve output space
pack_untilize_block(...)                        // Convert tile -> row-major
cb_pop_front(input_cb, block_width_tiles)       // Free input
cb_push_back(output_cb, block_width_tiles)      // Signal output ready
```

### Key Prerequisite

`compute_kernel_hw_startup(input_cb, output_cb)` must be called before using `untilize()`. This initializes the hardware for the specified CBs.

### InitUninitMode for Multi-Phase Kernels

When untilize is one of several operations in a kernel (as in layer_norm_rm where compute may do multiple things before untilizing), use the lifecycle modes:
- `InitOnly` for the first untilize call
- `Neither` for middle calls
- `UninitOnly` for the last call

Or simply use `InitAndUninit` for standalone calls.

---

## Writer Kernel: Row-Major Stick Writing Pattern

### File: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

This is the key pattern for writing untilized (row-major) data from L1 CBs to DRAM interleaved buffers.

### TensorAccessor Setup (Writer)

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();   // Start at compile-time arg index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

- `TensorAccessorArgs<N>()` on the device side: `N` is the compile-time argument offset where TensorAccessor args begin
- `TensorAccessor(args, base_addr, page_size)`: creates the accessor with the output buffer address and stick size
- `s.get_noc_addr(page_id, offset_in_bytes)`: returns the NoC address for a given stick (page) with optional byte offset within that stick

### Host-Side TensorAccessor Setup

From the program factory (lines 205-215):

```cpp
std::vector<uint32_t> writer_compile_time_args = {
    (uint32_t)output_cb_index,           // index 0
    (uint32_t)output_stick_size,         // index 1
    (uint32_t)tile_height,               // index 2
    (uint32_t)num_tiles_per_input_block, // index 3
    (uint32_t)output_num_blocks_across_width,  // index 4
    (uint32_t)output_element_size,       // index 5
    (uint32_t)num_cols_per_input_block,  // index 6
    (uint32_t)num_cols_per_output_block, // index 7
};
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

`TensorAccessorArgs(*dst_buffer).append_to(compile_time_args)` appends the TensorAccessor's compile-time arguments (bank coordinates, tensor shape info, etc.) to the existing compile-time args vector. On the device side, `TensorAccessorArgs<8>()` starts reading from index 8 (after the 8 explicit compile-time args).

### Writer Kernel Core Logic

The writer processes one block (tile-row) at a time:

```cpp
auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
    cb_wait_front(cb_id_out0, num_tiles_per_input_block);

    uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

    for (uint32_t j = 0; j < tile_height; ++j) {
        // Calculate L1 read address for this stick
        uint32_t current_l1_read_addr =
            base_l1_read_addr + j * num_cols_per_input_block * output_element_size;

        // Calculate output page (stick) ID
        uint32_t num_rows_already_processed = block_height_index * tile_height + j;
        uint32_t output_page_id =
            num_rows_already_processed * num_output_blocks_across_width
            + width_wise_output_block_start_index;

        // Write the stick to DRAM
        uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, offset);
        noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write);
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, num_tiles_per_input_block);
};
```

### Stick Extraction from Untilized CB Data

After the compute kernel untilizes a block, the output CB contains row-major data laid out as follows:
- `tile_height` (32) rows, each `num_cols_per_input_block` elements wide
- Row `j` starts at offset `j * num_cols_per_input_block * element_size` from `get_read_ptr(cb_id_out0)`
- Each row is a contiguous stick that can be written directly to DRAM

**For layer_norm_rm**: The same pattern applies. After your compute kernel produces row-major output in a CB, the writer reads sticks sequentially from L1 and writes them to DRAM using TensorAccessor.

### Simplified Writer Pattern (For Interleaved Output, No Width Splitting)

When the output is interleaved and there is only one output block across width (common case), the writer simplifies to:

```
For each block assigned to this core:
    cb_wait_front(output_cb, Wt)                   // Wait for untilized block
    base_addr = get_read_ptr(output_cb)
    For j = 0 to tile_height-1:
        l1_addr = base_addr + j * row_width_bytes
        page_id = (block_index * tile_height + j) * 1 + 0   // simplified
        noc_addr = tensor_accessor.get_noc_addr(page_id)
        noc_async_write(l1_addr, noc_addr, stick_size_bytes)
    noc_async_write_barrier()
    cb_pop_front(output_cb, Wt)
```

Where:
- `Wt` = `num_tiles_per_input_block` = tiles per row
- `row_width_bytes` = `tensor_width * element_size`
- `stick_size_bytes` = `tensor_width * element_size` = one full row
- `page_id` = the row index (stick index) in the output tensor

---

## Index Calculations

### Output Page (Stick) ID Calculation

For interleaved output with no width splitting (`num_output_blocks_across_width = 1`):

```
output_page_id = (block_height_index * tile_height + row_within_block)
```

This is simply the global row index of the tensor. Each row-major page (stick) is one full tensor row.

For cases with width splitting (width/block-sharded output):

```
output_page_id = row_index * num_output_blocks_across_width + width_block_index
```

### L1 Read Address Calculation

Within a block in the output CB:

```
l1_read_addr = get_read_ptr(output_cb) + row_within_block * num_cols_per_input_block * element_size
```

The output CB stores row-major data contiguously: 32 rows, each `num_cols_per_input_block` elements wide.

---

## Memory Access Patterns

### Read Pattern (De-emphasized)

Reader reads tiles sequentially from DRAM, one tile at a time, using TensorAccessor.

### Write Pattern (Critical for layer_norm_rm)

- **Pattern**: Sequential row writes
- **Granularity**: One stick (row) per `noc_async_write` call
- **Ordering**: Rows within a block are written sequentially (row 0, row 1, ..., row 31)
- **Barrier**: `noc_async_write_barrier()` after all 32 rows of a block are written
- **DRAM access**: Pages (sticks) are round-robin distributed across DRAM banks (interleaved)
- **Burst size**: One full stick (`tensor_width * element_size` bytes) per write

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` = ceil(num_tiles_per_col / nblocks_per_core) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) for full cores |
| **Load balancing** | Near-equal with optional cliff core for remainder |
| **Remainder handling** | Last core (cliff core) processes `num_rows_per_cliff_core` blocks |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` function divides tile-rows across cores:
- Each full core processes `nblocks_per_core` = ceil(num_tiles_per_col / grid_area) blocks
- If `num_tiles_per_col % nblocks_per_core != 0`, a cliff core handles the remainder
- Cores are assigned in row-major order across the compute grid

---

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output stick/page (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Tile height (32) -- rows per block |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per tile-row (Wt) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bfloat16, 4 for float32) |
| 6 | `num_cols_per_input_block` | uint32_t | Elements per row in input block (`Wt * 32`) |
| 7 | `num_cols_per_output_block` | uint32_t | Elements per output page/stick |
| 8+ | TensorAccessor args | uint32_t[] | Appended via `TensorAccessorArgs(*dst_buffer).append_to(...)` |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Starting tile-row index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Unpadded columns per block (for handling uneven sharding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page index across width |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset in first output page (for partial page writes) |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per tile-row (`num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of tile-rows to process |

---

## Kernel Implementations

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks | RISCV_2 (pack+math+unpack) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | `compute_kernel_lib::untilize<>()` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**: Calls `compute_kernel_hw_startup(src_cb_id, out_cb_id)` then the untilize helper template. The helper handles CB synchronization internally (wait/pop on input, reserve/push on output).

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB c_16 (row-major) | DRAM (interleaved) | `noc_async_write`, TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - Uses `TensorAccessorArgs<8>()` to create accessor starting after 8 explicit compile-time args
  - Iterates `tile_height` (32) rows per block, writing each stick to DRAM
  - Handles width splitting for sharded outputs (multiple output pages per row)
  - Issues `noc_async_write_barrier()` after each block and pops the CB

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (interleaved) | CB c_0 (tiled) | `noc_async_read`, TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

---

## Implementation Notes

### Output CB Uses Tile-Sized Pages Even for Row-Major Data

A critical design detail: the output CB is created with `output_single_tile_size` as the page size, not the stick size. This is because the `pack_untilize_block` hardware operation writes row-major data in tile-sized chunks. The untilized output occupies the same total bytes as the input tiles -- the data is just reordered from tile format to row-major. The writer kernel then reads this row-major data from the CB using raw L1 pointer arithmetic (`get_read_ptr` + row offset), NOT using CB page semantics.

### `create_cb` Utility

The `create_cb` utility from `ttnn/operations/cb_utils.hpp` provides a concise way to create circular buffers:

```cpp
auto [cb_index, cb_handle] = create_cb(
    tt::CBIndex::c_16,       // CB ID
    program,                  // Program
    compute_core_range,       // Core range
    output_single_tile_size,  // Page size in bytes
    output_cb_num_tiles,      // Number of pages
    output_cb_data_format);   // Data format
```

### DST_ACCUM_MODE Define

For INT32, UINT32, and FLOAT32 data types, the compute kernel defines `DST_ACCUM_MODE=1` to enable destination accumulator mode. This is set via `compute_kernel_defines["DST_ACCUM_MODE"] = "1"`.

### fp32_dest_acc_en

When enabled, `UnpackToDestFp32` mode is set for the input CB, and `fp32_dest_acc_en` is passed to `ComputeConfig`. This ensures the DEST register uses FP32 accumulation.

### Width Splitting for Sharded Outputs

The writer kernel supports cases where input and output may have different sharding patterns. The `num_output_blocks_across_width` and offset calculations handle writing from one sharded input block to potentially multiple output pages or vice versa. For the interleaved-output case (most relevant to layer_norm_rm), `num_output_blocks_across_width = 1` and no splitting occurs.

---

## Relevance to layer_norm_rm

### What to Reuse from This Pattern

1. **Untilize helper**: Use `compute_kernel_lib::untilize<Wt, input_cb, output_cb>()` in the compute kernel after your layer norm computation produces tiled results. Ensure `compute_kernel_hw_startup()` is called beforehand.

2. **Output CB sizing**: Size the untilize output CB as `Wt` tiles per block (where `Wt = tensor_width / 32`). Use double-buffering (`2 * Wt`) if multiple blocks per core.

3. **Writer kernel pattern**: After untilize, use the same stick-extraction pattern:
   - `cb_wait_front(output_cb, Wt)`
   - Read `tile_height` sticks from L1 using pointer arithmetic
   - Write each stick to DRAM via `noc_async_write` with TensorAccessor
   - `noc_async_write_barrier()` + `cb_pop_front(output_cb, Wt)`

4. **TensorAccessor setup**: On host, create `TensorAccessorArgs(*dst_buffer)` and `append_to(compile_time_args)`. On device, use `TensorAccessorArgs<N>()` where N is the offset after your explicit compile-time args.

### Key Differences for layer_norm_rm

- **Input is row-major**: Unlike untilize which reads tiled input, layer_norm_rm reads row-major input. So no reader-side untilize is needed.
- **Compute is different**: layer_norm_rm will tilize input, perform normalization, then untilize output. The untilize helper is used only as the final step.
- **Output sticks match input sticks**: Since both input and output are row-major, the output page size equals one full tensor row: `tensor_width * element_size`.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal kernels? What is the TensorAccessorArgs pattern?"
   **Reason**: Needed to understand how writer kernel generates DRAM addresses for output pages
   **Key Findings**: TensorAccessor maps logical page IDs to physical NoC addresses. TensorAccessorArgs on the host appends compile-time/runtime args. On device, `TensorAccessorArgs<N>()` reads from compile-time arg offset N.

2. **Query**: "How does the untilize operation work in tt-metal? How does pack_untilize_block convert tiled data to row-major?"
   **Reason**: Needed to understand the compute-side untilize mechanism
   **Key Findings**: `pack_untilize_block` reorders data from 32x32 tile format (with 16x16 faces) into contiguous row-major format. The output CB receives row-major data in tile-sized pages.

3. **Query**: "How does TensorAccessor::get_noc_addr(page_id, offset_within_page) work for writing row-major sticks?"
   **Reason**: Needed to understand the writer's address generation for partial page writes
   **Key Findings**: For row-major interleaved buffers, page_id is the stick (row) index. offset_within_page is a byte offset for writing to the middle of a page (used in width-splitting scenarios).

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host/device API
   **Key Information**: Host creates `TensorAccessorArgs(buffer)`, device uses `TensorAccessorArgs<offset>()`. `get_noc_addr(page_id)` resolves bank mapping automatically.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding output page structure for row-major tensors
   **Key Information**: For row-major layout, each tensor row is one page. Pages are round-robin distributed across DRAM banks in interleaved mode.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the untilize helper template signature and internal dispatch logic
   **Key Information**: Two dispatch paths based on block width vs DEST limit. Helper manages all CB synchronization internally. Prerequisite: `compute_kernel_hw_startup()`.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory
   **Key Information**: `create_cb(cb_id, program, core_range, page_size, num_pages, data_format)` returns `{cb_index, cb_handle}`.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` core distribution
   **Key Information**: Divides blocks across cores with optional cliff core for remainder. Returns structured decomposition with full/cliff core ranges.
