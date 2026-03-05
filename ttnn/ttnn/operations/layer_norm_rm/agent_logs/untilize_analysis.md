# Untilize (Multi-Core) Implementation Analysis

## Overview

The untilize operation converts tensor data from **TILE_LAYOUT** (32x32 tiles with face structure) to **ROW_MAJOR_LAYOUT** (contiguous row sticks). This analysis focuses on the **output stage** -- specifically how the writer kernel extracts row-major sticks from the output CB and writes them to DRAM, the output CB sizing strategy, and the untilize compute helper signature/usage. This analysis serves as an output-stage reference for a new `layer_norm_rm` operation that produces row-major interleaved output.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = one row of tiles across the tensor width) |
| **Unit size** | `num_tiles_per_input_block` tiles (i.e., all tiles in one tile-row for the core's assigned width) |
| **Total units** | `num_tiles_per_col` tile-rows distributed across cores |
| **Loop structure** | Outer loop over tile-rows assigned to core; compute untilizes one tile-row at a time; writer extracts `tile_height` (32) sticks per tile-row and writes each stick to DRAM |

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to `[tensor_height, tensor_width]`) |
| **Dimension convention** | Last dim = width; all outer dims collapsed into height |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED (or sharded -- analysis focuses on interleaved) |
| **Buffer type** | DRAM (typical) |
| **Data type** | BFLOAT16 (also supports FLOAT32, INT32, UINT32, UINT16) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) |
| **Data type** | Same as input |

**Key property for output-stage reuse**: In the interleaved case, one **output page = one full-width row** (i.e., `output_page_width = tensor_width`). The page size in bytes is `tensor_width * element_size`. Pages are distributed round-robin across DRAM banks.

### Layout Transformations

The compute kernel performs the tile-to-row-major conversion. The input CB holds data in tile layout (faces within tiles); the output CB holds data rearranged into contiguous sticks. The writer then copies these sticks from the output CB to DRAM pages.

## Data Flow Pattern (Output-Stage Focus)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved tiles) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 | CB c_16 (output) | `cb_wait_front` / `cb_pop_front` on c_0; `cb_reserve_back` / `cb_push_back` on c_16 |
| 3 | **Writer** | **CB c_16** | **DRAM (interleaved RM sticks)** | **`cb_wait_front`, `noc_async_write`, `cb_pop_front`** |

### Detailed Output-Stage Data Flow

1. Compute produces `num_tiles_per_input_block` tiles worth of row-major data into CB c_16 via `pack_untilize_block`. After one tile-row is untilized, it calls `cb_push_back(c_16, num_tiles_per_input_block)`.

2. Writer calls `cb_wait_front(c_16, num_tiles_per_input_block)` to synchronize.

3. Writer reads the output CB memory directly using `get_read_ptr(cb_id_out0)` to get the L1 base address of the untilized data.

4. The untilized data in the output CB is laid out as **`tile_height` (32) contiguous sticks**, each `num_cols_per_input_block * element_size` bytes wide. This is the key: the compute kernel has already rearranged tile data into consecutive row-major sticks.

5. Writer iterates over `tile_height` rows (j = 0..31), computing `current_l1_read_addr = base + j * num_cols_per_input_block * element_size` to find each stick.

6. For each stick, writer computes the DRAM destination using TensorAccessor: `s.get_noc_addr(output_page_id, offset_within_page)` and issues `noc_async_write`.

7. After all `tile_height` sticks are written, writer calls `noc_async_write_barrier()` then `cb_pop_front(c_16, num_tiles_per_input_block)` to release the CB space.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_input_block * 2` (interleaved, multi-block) or `num_tiles_per_input_block` (single-block) | `num_tiles_per_input_block` | Double (multi-block) / Single (single-block) | Reader | Compute | Block |
| **c_16** | **cb_output** | **Output RM stick staging** | **`num_tiles_per_input_block * 2` (multi-block) or `num_tiles_per_input_block` (single-block)** | **`num_tiles_per_input_block`** | **Double (multi-block) / Single (single-block)** | **Compute** | **Writer** | **Block** |

### Output CB Sizing Logic (Key for Reuse)

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Pattern**: The output CB is sized to hold one or two complete tile-rows. When a core processes multiple tile-rows (blocks), double-buffering allows compute to start on the next block while writer drains the current one. The block size is always `num_tiles_per_input_block` tiles, meaning the full width of data for one tile-row.

**Memory consumed**: `output_cb_num_tiles * output_single_tile_size` bytes, where `output_single_tile_size = tt::tile_size(output_cb_data_format)`. For BF16 with 32x32 tiles, each tile is 2048 bytes (32*32*2). For FLOAT32, each tile is 4096 bytes.

**Important**: The output CB is sized in units of **tiles** even though it holds row-major data. The `num_tiles_per_input_block` tiles worth of output space holds `tile_height` sticks, each `num_tiles_per_input_block * tile_width * element_size` bytes. The total capacity is `tile_height * num_tiles_per_input_block * tile_width * element_size` bytes per block (same as `num_tiles_per_input_block * tile_size`).

## Pipeline Pattern Summary

| Path | Input CB (c_0) | Output CB (c_16) | Overlap |
|------|----------------|-------------------|---------|
| Single-block per core | Single-buffered | Single-buffered | No overlap; sequential read-compute-write |
| Multi-block per core | Double-buffered | Double-buffered | Reader/Compute can overlap with Writer on adjacent blocks |

## Untilize Compute Helper Signature and Usage

The compute kernel uses the `compute_kernel_lib::untilize` helper from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`.

### Template Signature

```cpp
template <
    uint32_t block_width_tiles,    // Tiles per row (compile-time)
    uint32_t input_cb,             // Input CB index (tiled data)
    uint32_t output_cb,            // Output CB index (row-major output)
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);  // num_blocks = tile-rows to process (runtime)
```

### Key Parameters

- **`block_width_tiles`** (compile-time): Number of tiles per row. This is the width of one "block" of work. Must be known at compile time because it determines DEST register allocation and sub-block splitting.
- **`num_blocks`** (runtime): Number of tile-rows to process. This is the height dimension divided among cores.
- **`WaitMode::WaitBlock`** (default): The helper calls `cb_wait_front(input_cb, block_width_tiles)` before each block, then `cb_pop_front(input_cb, block_width_tiles)` after processing. This synchronizes with the reader on a per-block basis.
- **`WaitMode::WaitUpfront`**: Waits for ALL tiles upfront before processing any. Useful when data is already fully in the CB (e.g., when the CB is backed by a sharded buffer, or when a previous compute stage has already produced all data).
- **`WaitMode::NoWait`**: Caller manages synchronization externally. Useful when the untilize is called after another compute operation that already produced data into the CB.

### Internal Logic

The helper automatically selects between two paths:
1. **Single-pass pack_untilize** (when `block_width_tiles <= DEST_AUTO_LIMIT`): Processes the full tile-row in one DEST pass. Optimal path.
2. **Block-based pack_untilize** (when `block_width_tiles > DEST_AUTO_LIMIT`): Splits the tile-row into sub-blocks that fit in DEST registers, processing them sequentially.

For each block (tile-row):
```
cb_wait_front(input_cb, block_width_tiles)   // wait for tiled data
cb_reserve_back(output_cb, block_width_tiles) // reserve RM output space
pack_untilize_block(...)                      // hardware untilize: tile -> RM sticks
cb_pop_front(input_cb, block_width_tiles)     // release input
cb_push_back(output_cb, block_width_tiles)    // signal writer
```

### Usage in Untilize Operation

```cpp
// In the compute kernel (untilize_variable_num_blocks.cpp):
compute_kernel_hw_startup(src_cb_id, out_cb_id);  // REQUIRED before untilize
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,  // = num_tiles_per_input_block
    src_cb_id,                // = c_0
    out_cb_id,                // = c_16
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,
    ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
```

### Relevance for layer_norm_rm

For a new operation that needs to output row-major data, you have two options:
1. **Use `compute_kernel_lib::untilize`** as the final compute step, converting tile-format intermediate results to RM in the output CB. Then write RM sticks to DRAM using the same writer pattern.
2. **Produce RM data directly** in your compute kernel (if your computation naturally produces row-major results). In this case, you still need to size the output CB appropriately and use the same writer pattern.

If using option 1, the untilize helper with `WaitMode::NoWait` or `WaitMode::WaitUpfront` is appropriate since data is already in the CB from a previous compute step.

## Writer Kernel Pattern: RM Stick Writes to DRAM

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

This is the core pattern for writing row-major sticks to interleaved DRAM.

### TensorAccessor Setup

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // CT args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

- `TensorAccessorArgs<8>()`: Creates accessor args starting at compile-time arg index 8. The host appended TensorAccessorArgs to the writer compile-time args vector after the 8 explicit args (indices 0-7).
- `TensorAccessor(dst_args, dst_addr, output_stick_size)`: Creates the accessor with the buffer base address and page size. For interleaved RM output, `output_stick_size = output_page_width * element_size` (one full row).

### Stick Extraction from Output CB

After `cb_wait_front(cb_id_out0, num_tiles_per_input_block)`, the output CB contains `tile_height` (32) rows of untilized data. Each row is `num_cols_per_input_block * element_size` bytes wide.

```cpp
uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

for (uint32_t j = 0; j < tile_height; ++j) {
    uint32_t current_l1_read_addr = base_l1_read_addr
        + j * num_cols_per_input_block * output_element_size;
    // ... write this stick to DRAM
}
```

**Key insight**: The stride between sticks in the output CB is `num_cols_per_input_block * element_size`. This is NOT the output page size -- it is the width of data that this core's tile-row covers. In the simple interleaved case (no width sharding), `num_cols_per_input_block = tensor_width` and this equals the output page width.

### DRAM Page ID and Offset Calculation

For the **simple interleaved case** (no width/block sharding, which is the relevant case for layer_norm_rm):

- `output_num_blocks_across_width = 1` (one output page per row)
- `num_cols_per_output_block = tensor_width` (full row per page)
- `width_wise_output_block_start_index = 0`
- `num_cols_already_processed_in_first_output_block = 0`

The page ID calculation simplifies to:
```cpp
uint32_t output_page_id = (block_height_index * tile_height + j) * 1 + 0;
// i.e., output_page_id = absolute_row_index
```

And the write becomes:
```cpp
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, 0);  // no offset
noc_async_write(current_l1_read_addr, dst_noc_addr, output_stick_size);
```

This is a **one write per row** pattern: each row-major stick maps to exactly one DRAM page.

### Write Barrier and CB Release

```cpp
noc_async_write_barrier();                          // wait for all 32 writes
cb_pop_front(cb_id_out0, num_tiles_per_input_block); // release CB space
```

The barrier is placed after all `tile_height` (32) writes for one block, not after each individual write. This batches the NoC writes for efficiency.

### Simplified Writer Pattern for layer_norm_rm (Interleaved Output)

For a new operation producing row-major interleaved output, the writer pattern can be simplified to:

```cpp
// Compile-time args
constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);     // tensor_width * element_size
constexpr uint32_t num_sticks_per_block = get_compile_time_arg_val(2);  // e.g., tile_height=32
constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(3);
// ... TensorAccessorArgs at subsequent indices

// Runtime args
const uint32_t dst_addr = get_arg_val<uint32_t>(0);
const uint32_t num_blocks = get_arg_val<uint32_t>(1);
const uint32_t start_row = get_arg_val<uint32_t>(2);

constexpr auto dst_args = TensorAccessorArgs<N>();
const auto s = TensorAccessor(dst_args, dst_addr, stick_size);

uint32_t page_id = start_row;
for (uint32_t block = 0; block < num_blocks; ++block) {
    cb_wait_front(cb_id_out, num_tiles_per_block);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    for (uint32_t row = 0; row < num_sticks_per_block; ++row) {
        uint64_t noc_addr = s.get_noc_addr(page_id);
        noc_async_write(l1_read_addr, noc_addr, stick_size);
        l1_read_addr += stick_size;
        page_id++;
    }

    noc_async_write_barrier();
    cb_pop_front(cb_id_out, num_tiles_per_block);
}
```

## Index Calculations

### Output Page ID Mapping (Interleaved Case)

For the interleaved RM output case relevant to layer_norm_rm:

```
output_page_id = block_height_index * tile_height + row_within_block
```

Where:
- `block_height_index` = the tile-row index this core is currently processing (global, not local)
- `tile_height` = 32 (rows per tile-row)
- `row_within_block` = j in [0, tile_height)

Each `output_page_id` uniquely identifies one row of the output tensor. The TensorAccessor maps this page ID to a DRAM bank address using round-robin interleaving.

### L1 Read Address Calculation

```
current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * element_size
```

This is a simple stride calculation. The output CB contains a contiguous block of `tile_height` sticks, each `num_cols_per_input_block * element_size` bytes wide.

## Memory Access Patterns

### Read Pattern (De-emphasized)

Reader reads tiles sequentially from DRAM by tile page ID, one tile at a time, with `noc_async_read_barrier()` after each tile.

### Write Pattern (Key Focus)

- **Pattern**: Sequential row writes with write-barrier batching
- **Granularity**: One full row-major stick per `noc_async_write`
- **Write size**: `output_stick_size` bytes (= `tensor_width * element_size` for interleaved)
- **Ordering**: Rows are written in ascending order within each block (j = 0..31)
- **Barrier**: One `noc_async_write_barrier()` per block (after all 32 rows), not per row
- **DRAM access**: Round-robin across banks via TensorAccessor. Consecutive row pages land on different banks, providing good bank-level parallelism

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (computed by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) for full cores; `num_rows_per_cliff_core` for the single cliff core |
| **Load balancing** | Near-equal distribution; at most one cliff core with fewer blocks |

### Work Splitting via `split_blocks_for_tilize`

```cpp
auto [num_compute_cores, compute_core_range, full_compute_core_range,
      cliff_compute_core_range, num_rows_per_full_core, num_rows_per_cliff_core]
    = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);
```

- `num_tiles_per_col` = total tile-rows = `tensor_height / tile_height`
- Distributes tile-rows as evenly as possible across available cores
- At most one "cliff" core handles the remainder

## Arguments

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output row = `output_page_width * element_size` |
| 2 | `tile_height` | uint32_t | Rows per tile (32) = sticks extracted per block |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per tile-row = CB block size in tiles |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Elements per input block row = `num_tiles_per_input_block * tile_width` |
| 7 | `num_cols_per_output_block` | uint32_t | Elements per output page width |
| 8+ | TensorAccessorArgs | uint32_t(s) | Accessor config for output buffer (1 arg for interleaved) |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer DRAM base address |
| 1 | `num_input_blocks_to_process` | uint32_t | Tile-rows this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | First tile-row index for this core |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual data columns (excludes padding) |
| 4 | `width_wise_output_block_start_index` | uint32_t | First output page column index (0 for interleaved) |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Byte offset into first output page (0 for interleaved) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | = `num_tiles_per_input_block` (tiles per tile-row) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | = `num_input_blocks_to_process` (tile-rows for this core) |

## Kernel Implementations

### Writer Kernel (Primary Focus)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (RM sticks) | DRAM (interleaved RM pages) | `cb_wait_front`, `get_read_ptr`, `get_noc_addr`, `noc_async_write`, `noc_async_write_barrier`, `cb_pop_front` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**: Uses a lambda `write_tiles_in_current_block` that processes one tile-row at a time. For each tile-row, it iterates over `tile_height` rows, computing L1 source addresses with stride-based indexing and DRAM destination addresses via TensorAccessor. The `get_noc_addr(page_id, offset)` two-argument form enables writing to sub-page offsets when output pages are wider than input blocks (width/block sharding). For the interleaved case, offset is always 0.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 (tiled) | CB c_16 (RM) | `compute_kernel_hw_startup`, `compute_kernel_lib::untilize` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**: Single call to `compute_kernel_lib::untilize<block_width, cb_in, cb_out, InitAndUninit, WaitBlock, NoReconfigure>(num_blocks)`. The helper handles all CB synchronization internally.

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved tiles) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

## Implementation Notes

### Output CB Data Layout After Untilize

After the compute kernel untilizes one tile-row, the output CB contains data arranged as:
```
Row 0:  [col_0, col_1, ..., col_{W-1}]   (W = num_cols_per_input_block)
Row 1:  [col_0, col_1, ..., col_{W-1}]
...
Row 31: [col_0, col_1, ..., col_{W-1}]
```
Where each row is `W * element_size` bytes, and rows are contiguous in memory. The stride between rows equals `W * element_size`. This is the natural row-major layout, with the only caveat that W may include padding columns if the tensor width is not tile-aligned.

### Host-Side TensorAccessorArgs Append Pattern

The host code appends TensorAccessorArgs to the compile-time args vector:
```cpp
std::vector<uint32_t> writer_compile_time_args = { ... 8 explicit args ... };
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

On the device side, `TensorAccessorArgs<8>()` reads from compile-time arg index 8 onward. For interleaved buffers, this appends just 1 uint32_t (the ArgsConfig flags indicating DRAM interleaved layout).

### Double-Buffering Decision

The CB sizing uses a simple heuristic: if only one block is processed, single-buffer; otherwise double-buffer. This is conservative but effective. For layer_norm_rm, the same logic applies: if a core processes multiple output blocks (tile-rows worth of sticks), double-buffering the output CB enables compute-write overlap.

### FP32 Destination Accumulator

When `fp32_dest_acc_en` is true, the compute kernel unpacks to FP32 in DEST. This is relevant for layer_norm_rm which may want FP32 accumulation for numerical stability during mean/variance computation. The untilize operation sets `UnpackToDestMode::UnpackToDestFp32` on the input CB when this is enabled.

### `noc_async_write_barrier` Placement

The barrier is placed once per block (after all 32 stick writes), NOT per stick. This allows up to 32 outstanding NoC write transactions before stalling, improving throughput. For layer_norm_rm's writer, the same pattern should be followed.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal kernels? Specifically how does TensorAccessorArgs and TensorAccessor get_noc_addr work for mapping page IDs to NoC addresses for interleaved tensors?"
   **Reason**: Needed to understand the page-to-DRAM-address mapping used in the writer kernel.
   **Key Findings**: For interleaved tensors, TensorAccessorArgs appends just 1 compile-time arg (ArgsConfig flags). The `get_noc_addr(page_id, offset)` method maps a logical page ID to a physical NoC address using round-robin bank distribution. The two-argument form adds a byte offset within the page.

2. **Query**: "In the untilize operation, how does the compute kernel convert tile layout to row-major layout? What does pack_untilize vs standard untilize mean?"
   **Reason**: Needed to understand what data layout the writer kernel receives in the output CB.
   **Key Findings**: The hardware unpacker (`llk_unpack_untilize`) handles tile-to-row-major conversion. `pack_untilize` is an optimized path (80 cycles/tile vs 390 cycles/tile for standard untilize). The compute kernel orchestrates the data through the hardware pipeline.

3. **Query**: "What is the output data layout after the untilize compute kernel writes to the output circular buffer?"
   **Reason**: Critical for understanding the stick extraction pattern in the writer.
   **Key Findings**: The output CB contains contiguous sticks (rows) where each stick spans the full width of tiles in the block. For a row of N tiles, each stick is N*32 elements wide. Sticks are stored consecutively in memory (row 0, row 1, ..., row 31).

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host/device API and page addressing.
   **Key Information**: `get_noc_addr(page_id, offset)` provides page+offset addressing. For interleaved tensors, only 1 compile-time arg is needed. The accessor handles bank mapping automatically.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major layout page structure.
   **Key Information**: In row-major layout, each row is one page. Pages are distributed round-robin across DRAM banks in interleaved mode. Tiles are 32x32 with 16x16 faces.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the untilize compute helper API, template parameters, and internal dispatch logic.
   **Key Information**: The helper supports WaitBlock, WaitUpfront, and NoWait synchronization modes. It automatically splits wide rows into sub-blocks if they exceed DEST capacity. The `InitUninitMode` enum allows chaining multiple untilize calls without redundant init/uninit.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: `create_cb(cb_id, program, core_range, page_size, num_pages, data_format, buffer)` creates a circular buffer. The optional `buffer` parameter backs the CB with a pre-allocated buffer (used for sharded input CBs).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` work distribution.
   **Key Information**: Distributes `nblocks` across `grid_area` cores, computing `nblocks_per_core = ceil(nblocks / grid_area)` with at most one cliff core handling the remainder. Returns core ranges for full and cliff cores.
