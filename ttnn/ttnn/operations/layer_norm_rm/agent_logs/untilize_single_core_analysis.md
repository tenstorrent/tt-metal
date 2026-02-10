# Untilize Single-Core Implementation Analysis

## Overview

The **untilize single-core** operation converts tensor data from **tiled layout** (32x32 tiles with face-based sub-tile organization) back to **row-major (stick) layout** and writes the result to DRAM-interleaved memory. It runs entirely on a single Tensix core (core {0,0}) and is the simplest variant of the untilize operation, serving as a clear reference for how tile-to-row-major conversion and row-major DRAM writes work.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp`

**Kernels Used**:
- **Reader**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Compute** (fast path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`
- **Compute** (slow path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`
- **Writer**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles within one tile-row) |
| **Unit size** | `num_tiles_per_block` tiles (a divisor of the tile-row width that fits in L1) |
| **Total units** | `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| **Loop structure** | Outer: tile-rows across height; Middle: column groups (for sharded output); Inner: horizontal blocks per column-row |

A single "work unit" for the compute and writer kernels is one **block** of `num_tiles_per_block` tiles arranged horizontally. The reader processes tiles one at a time but pushes them in groups that align with the block size consumed by compute.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | N-dimensional (outer dims collapsed), last dim is width |
| **Dimension convention** | Generic (outer dims x width) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED (round-robin across DRAM banks) |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, or UINT16 |

The input tensor is stored as tiles in DRAM. Each tile is 32x32 elements, internally organized as four 16x16 faces in row-major order (face0, face1, face2, face3). The tile size in bytes depends on the data format (e.g., 2048 bytes for bfloat16 32x32 tiles).

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same logical shape as input |
| **Dimension convention** | Generic (outer dims x width) |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED (round-robin across DRAM banks) or WIDTH_SHARDED / BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 (depends on output memory config) |
| **Data type** | Same as input (BFLOAT16, FLOAT32, etc.) |

The output tensor is stored as row-major sticks. Each stick is one full row of the tensor (`output_stick_size = padded_shape[-1] * element_size`). For interleaved layout, sticks are distributed round-robin across DRAM banks. For sharded output, `num_columns_of_blocks` adjusts the stick indexing to account for shard boundaries.

**Page size for row-major output**: `output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks`, which simplifies to `padded_shape[-1] * element_size` for the non-sharded case. This is the size in bytes of one complete row of the output tensor.

### Layout Transformations

The core transformation is **untilize**: converting from tile layout (32x32 tiles with face sub-structure) to row-major layout (linear rows). The compute kernel performs this conversion in the DEST registers, reading tiled data from the input CB and writing row-major data to the output CB. The writer then takes row-major data from the output CB and writes it to DRAM as sticks.

## Data Flow Pattern

### Step 1: Reader reads tiles from DRAM into input CB
The reader kernel iterates over all `num_tiles` tiles sequentially by page ID. For each tile:
1. `cb_reserve_back(cb_id_in0, 1)` -- wait for space in input CB
2. `noc_async_read` from DRAM (using TensorAccessor to resolve bank/offset) into CB write pointer
3. `noc_async_read_barrier()` -- wait for DMA completion
4. `cb_push_back(cb_id_in0, 1)` -- make tile available to compute

### Step 2: Compute converts tiles from tile layout to row-major
The compute kernel processes tiles in blocks of `num_tiles_per_block`:
1. `cb_wait_front(input_cb, block_width_tiles)` -- wait for a full block of tiles
2. `cb_reserve_back(output_cb, block_width_tiles)` -- reserve output space
3. Perform untilize (either `pack_untilize_block` for fast path or `untilize_block` for slow path)
4. `cb_pop_front(input_cb, block_width_tiles)` -- free input tiles
5. `cb_push_back(output_cb, block_width_tiles)` -- make row-major data available to writer

After untilize, the output CB contains the data reorganized as contiguous rows. For a block of N tiles wide, the output CB holds `tile_height` (32) rows, each of width `N * TILE_WIDTH * element_size` bytes.

### Step 3: Writer writes row-major sticks to DRAM
The writer processes the output CB data block by block. For each block:
1. `cb_wait_front(cb_id_out0, num_tiles_per_output_block)` -- wait for untilized data
2. For each of the `tile_height` (32) rows within the block:
   - Compute the destination NOC address using `s.get_noc_addr(stick_id)`
   - Write `output_single_block_width_size` bytes from L1 to DRAM
   - Advance L1 read pointer and destination address
3. `noc_async_write_barrier()` -- ensure all writes complete
4. `cb_pop_front(cb_id_out0, num_tiles_per_output_block)` -- free CB space

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Tiled input staging | `num_tiles_per_block` tiles | 1 tile (reader pushes 1 at a time) | Single | Reader | Compute | Block |
| c_16 | cb_output | Row-major output staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

**Key observations**:
- Both CBs have identical capacity equal to `num_tiles_per_block` tiles.
- The input CB is filled tile-by-tile by the reader but consumed as a full block by compute.
- The output CB is filled as a full block by compute and consumed as a full block by the writer.
- Both are **single-buffered** (capacity equals block size for compute/writer). This means compute and writer cannot overlap with reader for the same block.
- The `num_tiles_per_block` is dynamically determined to be the largest divisor of `num_tiles_per_column_row` that fits within L1 constraints (`max_tiles_per_cb`).

**L1 Budget Calculation** (factory lines 66-70):
```
max_l1_size = (l1_size_per_core / 2) - base_allocator_addr
max_tiles_per_cb = max_l1_size / (input_tile_size + output_tile_size)
```
The factory conservatively uses half of L1 for CBs, and since there are 2 CBs (input + output), it divides by the sum of both tile sizes.

## Pipeline Pattern Summary

Since both CBs have capacity equal to the block size (single-buffered), there is **no overlap between pipeline stages within the same block**. The pipeline is strictly sequential per block:
1. Reader fills input CB with `num_tiles_per_block` tiles (one at a time)
2. Compute untilizes the block and pushes to output CB
3. Writer drains the output CB to DRAM

However, there can be some overlap between writing the current block and reading tiles for the next block, since the reader only needs 1 tile of CB space at a time while the writer frees the output CB atomically.

## Index Calculations

### Reader: Tile Index to DRAM Address
The reader uses `TensorAccessor` with the source buffer to map tile page IDs to NOC addresses:
```cpp
constexpr auto src_args = TensorAccessorArgs<1>();  // CTA offset 1 (after cb_id_in0)
const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
uint64_t noc_read_addr = s.get_noc_addr(page_id);
```
For interleaved DRAM, the TensorAccessor resolves to `InterleavedAddrGen<true>` which computes:
- `bank_id = page_id % num_dram_banks` (round-robin)
- `bank_offset = (page_id / num_dram_banks) * tile_bytes`
- `noc_addr = bank_noc_xy + bank_base_address + bank_offset`

### Writer: Stick Index to DRAM Address

The writer computes `stick_id` for each row of each block:
```cpp
uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
uint32_t stick_id = num_complete_rows_already_processed + j;
base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
```

Where:
- `i` = current tile-row index (0 to `num_blocks_across_height - 1`)
- `k` = row within the tile (0 to `tile_height - 1`, i.e., 0 to 31)
- `j` = column-of-blocks index (0 to `num_output_columns_of_blocks - 1`)

For the common non-sharded case (`num_output_columns_of_blocks = 1`), this simplifies to:
```
stick_id = i * tile_height + k
```
which is simply the global row index.

The TensorAccessor for the writer uses `output_stick_size` as the page size, so it maps each stick_id to the correct DRAM bank and offset for that row-major stick.

### Writer: Intra-block Address Advancement
Within a block, the writer writes `output_single_block_width_size` bytes per row per sub-block. After writing one sub-block, the base address for each row advances:
```cpp
base_dst_noc_addr[l] += output_single_block_width_size;
```
This handles the case where a tile-row is split into multiple horizontal blocks (`num_blocks_per_column_row > 1`). Each subsequent block writes to a position offset by the previous block's width.

## Memory Access Patterns

### Read Pattern
- **Ordering**: Sequential tile-by-tile, starting from `start_page_id` (always 0 in single-core)
- **Granularity**: One full tile per read (`tile_bytes`)
- **Access type**: DRAM interleaved reads via NoC0
- **Pattern**: Tiles are read in row-major tile order (left-to-right, top-to-bottom through the tile grid)
- **Barrier**: `noc_async_read_barrier()` after each tile (conservative, ensures completion before CB push)

### Write Pattern
- **Ordering**: Row-by-row within each block, blocks processed left-to-right then top-to-bottom
- **Granularity**: `output_single_block_width_size` bytes per write (= `num_tiles_per_block * TILE_WIDTH * element_size`)
- **Access type**: DRAM interleaved writes via NoC1
- **Pattern**: Within a block of `num_tiles_per_block` tiles, the writer issues `tile_height` (32) separate writes, one for each row. Each write covers the full width of the block.
- **Barrier**: `noc_async_write_barrier()` after all 32 rows of a block are issued
- **Stick targeting**: Each row targets a different stick_id (and potentially different DRAM bank), so writes scatter across DRAM banks

The key insight for **row-major output page sizing**: Each output page (stick) is `output_stick_size` bytes = `padded_shape[-1] * element_size`. The writer does NOT write full sticks at once if the tile-row is split into multiple blocks. Instead, it writes `output_single_block_width_size` bytes at a time and relies on address advancement to fill the full stick across multiple blocks.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core |
| **Grid dimensions** | 1x1 |
| **Total cores** | 1 |
| **Work per core** | All tiles in the tensor |
| **Load balancing** | N/A (single core) |

The core is always `{0, 0}`. All work is assigned to this single core. This is the simplest distribution strategy and serves as the fallback when multi-core is not requested or not beneficial.

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer index (c_0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Source buffer tensor accessor args (appended by `TensorAccessorArgs(*src0_buffer)`). For interleaved DRAM, this is a single uint32_t encoding `ArgsConfig::IsDram`. |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_tiles | uint32_t | Total number of tiles to read |
| 2 | start_page_id | uint32_t | Starting tile page ID (always 0 for single-core) |

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one row-major output stick in bytes (`padded_width * element_size`) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard 32x32 tiles) |
| 3 | num_blocks_across_height | uint32_t | Number of tile-rows in the tensor (`total_height / tile_height`) |
| 4 | num_output_columns_of_blocks | uint32_t | Number of column groups (1 for non-sharded, >1 for width/block sharded) |
| 5 | num_blocks_per_output_column_row | uint32_t | Number of horizontal blocks per column-row (`num_tiles_per_column_row / num_tiles_per_block`) |
| 6 | num_tiles_per_output_block | uint32_t | Tiles per block (= `num_tiles_per_block`) |
| 7 | output_single_block_width_size | uint32_t | Bytes per block-row written (`num_tiles_per_block * TILE_WIDTH * element_size`) |
| 8+ | TensorAccessorArgs | uint32_t[] | Destination buffer tensor accessor args (appended by `TensorAccessorArgs(*dst_buffer)`) |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (`num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height`) |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`num_tiles_per_block`) |
| 2 | src_cb_id | uint32_t | Input CB index (c_0) |
| 3 | out_cb_id | uint32_t | Output CB index (c_16) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (tiled) | CB c_0 | Read tiles via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Key Logic**: Simple sequential tile reader. Reads one tile at a time from DRAM using `TensorAccessor::get_noc_addr(page_id)`, performs `noc_async_read` into the CB, barriers after each tile, then pushes. No batching -- each tile is individually read and pushed. This is a generic reader reusable across many operations.

### Compute Kernel (Fast Path -- pack_untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize | Compute (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Hardware pack_untilize |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`
- **Key Logic**: Uses `compute_kernel_lib::untilize<>()` from `untilize_helpers.hpp`. This helper auto-dispatches based on block width vs DEST register capacity and data format:
  - **Single-pass pack_untilize**: When `block_width_tiles <= DEST_AUTO_LIMIT` (8 tiles for half-sync bfloat16). Uses hardware-accelerated `pack_untilize_block` which reads tiled data from DEST and packs to row-major in one shot.
  - **Block-based pack_untilize**: When `block_width_tiles > DEST_AUTO_LIMIT` AND integer data type. Splits the row into sub-blocks that each fit in DEST.
  - The `DST_ACCUM_MODE` define is set for INT32/UINT32/FLOAT32 data types, which halves the DEST capacity (from 8 to 4 in half-sync mode).

### Compute Kernel (Slow Path -- standard untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize | Compute (TRISC) | N/A | CB c_0 (tiled) | CB c_16 (row-major) | Software untilize via SFPU |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`
- **Key Logic**: Also uses `compute_kernel_lib::untilize<>()` but is selected when:
  - `use_pack_untilize` is false, OR
  - Data type is UINT16, OR
  - Data type is FLOAT32 AND `num_tiles_per_block >= MAX_PACK_UNTILIZE_WIDTH` (8)

  In this path, the helper always takes the standard `untilize_block` path which uses UNPACK + MATH datacopy + PACK to convert tile to row-major format (slower than hardware pack_untilize).

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_single_core | RISCV_1 | NOC1 | CB c_16 (row-major) | DRAM | Write sticks to DRAM |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`
- **Key Logic**: This is the critical kernel for understanding how row-major sticks are written to DRAM. The writer processes the output CB data in a three-level nested loop:

  **Outer loop** (`i` over `num_blocks_across_height`): Iterates over tile-rows. Each tile-row contains `tile_height` (32) rows of output data.

  **Middle loop** (`j` over `num_output_columns_of_blocks`): Handles column groups for sharded output. For interleaved (non-sharded) output, this is 1.

  **Address setup**: Before processing horizontal blocks, the writer pre-computes the destination NOC address for each of the 32 rows in the current tile-row:
  ```cpp
  for (uint32_t k = 0; k < tile_height; ++k) {
      stick_id = (i * tile_height + k) * num_output_columns_of_blocks + j;
      base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
  }
  ```
  This uses `TensorAccessor::get_noc_addr(stick_id)` where the page size is `output_stick_size` (one full row-major row). The stick_id maps to the correct DRAM bank and offset for that row.

  **Inner loop** (`k` over `num_blocks_per_output_column_row`): For each horizontal block within the tile-row, the `write_tiles_in_current_block()` lambda:
  1. Waits for `num_tiles_per_output_block` tiles in the output CB
  2. Gets the L1 read address from the CB
  3. For each of the 32 rows (`l` from 0 to `tile_height - 1`):
     - Writes `output_single_block_width_size` bytes from L1 to the pre-computed NOC address
     - Advances the L1 read pointer by `output_single_block_width_size`
     - Advances the base NOC address by `output_single_block_width_size` (to write the next horizontal chunk of the same row in the next block iteration)
  4. Issues `noc_async_write_barrier()`
  5. Pops the block from the CB

  **Data layout in output CB**: After untilize, the output CB contains data organized as `tile_height` rows, each of width `num_tiles_per_block * TILE_WIDTH * element_size`. The rows are contiguous in L1 memory. The writer reads this data row by row and scatters it to the appropriate DRAM locations.

## Implementation Notes

### Block Size Determination
The factory dynamically determines `num_tiles_per_block` to maximize L1 utilization while ensuring it divides evenly into the tile-row width:
```cpp
uint32_t num_tiles_per_block = num_tiles_per_column_row;
if (num_tiles_per_block > max_tiles_per_cb) {
    for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
        if (num_tiles_per_column_row % i == 0) {
            num_tiles_per_block = i;
            break;
        }
    }
}
```
This searches downward from the maximum allowable block size to find the largest divisor. The ideal case is when the entire tile-row width fits in one block (`num_blocks_per_column_row = 1`).

### Sharded Output Handling
When the output is WIDTH_SHARDED or BLOCK_SHARDED, `num_columns_of_blocks` is set to `padded_shape[-1] / output_shard_width`, which divides the output width into shard-width chunks. The stick_id calculation in the writer accounts for this by interleaving column group indices into the stick addressing. For INTERLEAVED output, `num_columns_of_blocks = 1`.

### FP32 Destination Accumulation
When `fp32_dest_acc_en` is true, the unpacker is configured with `UnpackToDestMode::UnpackToDestFp32`, and the `DST_ACCUM_MODE` define is set. This halves the DEST register capacity (from 16 to 8 tiles in full-sync, or 8 to 4 tiles in half-sync), which may affect the block-splitting logic in the compute kernel helper.

### MAX_PACK_UNTILIZE_WIDTH Constraint
The constant `MAX_PACK_UNTILIZE_WIDTH = 8` (defined in `ttnn/api/ttnn/common/constants.hpp`) limits when pack_untilize can be used for FLOAT32 data. When `num_tiles_per_block >= 8` and dtype is FLOAT32, the slow untilize path is forced because pack_untilize does not support wide blocks for floating-point types.

### Runtime Argument Override
The `override_runtime_arguments` function updates only the buffer addresses (runtime args index 0) for both reader and writer when the operation is re-run with new tensor allocations. All other parameters are baked into compile-time args and do not change between invocations with the same tensor shape/format.

### Relevance to layer_norm_rm Output Stage
For a layer_norm_rm operation that needs to write its final result in row-major format to DRAM:
1. **Use CB c_16 (or equivalent) as the output staging CB** with row-major data format and capacity of `num_tiles_per_block` tiles.
2. **The compute kernel must untilize the result** before pushing to the output CB. Use `compute_kernel_lib::untilize<>()` from `untilize_helpers.hpp` for this.
3. **The writer kernel writes sticks to DRAM** using TensorAccessor with page_size = `output_stick_size` (one full row). The stick_id is the global row index.
4. **output_single_block_width_size** = `num_tiles_per_block * 32 * element_size` determines the write granularity per row per block.
5. **The writer scatters writes across DRAM banks** -- each stick may land in a different bank due to round-robin interleaving.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in tt-metal? How are tiles converted back to row-major sticks?"
   **Reason**: Needed to understand the tile-to-stick conversion mechanism at the hardware level.
   **Key Findings**: The untilize occurs during the PACK stage of the compute pipeline. `pack_untilize` is the fast hardware-accelerated path that directly packs from DEST registers to row-major format. Standard `untilize` uses UNPACK + MATH datacopy + PACK and is slower. The PACK thread handles the actual format conversion.

2. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal kernels?"
   **Reason**: Both reader and writer kernels use TensorAccessor for address generation; needed to understand how it maps page IDs to NOC addresses.
   **Key Findings**: TensorAccessorArgs on the host packs tensor distribution info into compile-time args. On the device, TensorAccessor reconstructs this info and provides `get_noc_addr(page_id)` which handles bank selection (round-robin for interleaved) and offset calculation. For interleaved DRAM buffers, TensorAccessor reduces to `InterleavedAddrGen<true>`.

3. **Query**: "How does the writer kernel write row-major sticks to DRAM?"
   **Reason**: Critical for understanding the output stage pattern needed for layer_norm_rm.
   **Key Findings**: The writer pre-computes NOC addresses for all 32 rows of a tile-row, then writes `output_single_block_width_size` bytes per row per block. The stick_id formula `(i * tile_height + k) * num_columns + j` maps to global row indices. Addresses advance horizontally across blocks.

4. **Query**: "What is the create_cb utility and standard CB index conventions?"
   **Reason**: Needed to understand CB creation API and index naming conventions.
   **Key Findings**: `create_cb` is a convenience wrapper for `CircularBufferConfig`. `c_0` through `c_7` are conventionally for inputs, `c_16` through `c_23` for outputs, `c_24` through `c_31` for intermediates.

5. **Query**: "How is data organized in the output CB after untilize?"
   **Reason**: Needed to understand the memory layout the writer must interpret.
   **Key Findings**: After untilize, data in the output CB is in row-major format: `tile_height` rows, each of width `num_tiles_per_block * TILE_WIDTH * element_size`, laid out contiguously. The writer reads row by row and writes to DRAM.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand page definitions for row-major vs tiled tensors.
   **Key Information**: Row-major tensors use one row as one page. Tiled tensors use one 32x32 tile as one page. Interleaved layout distributes pages round-robin across DRAM banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Needed to understand TensorAccessor host/device setup pattern.
   **Key Information**: Host creates `TensorAccessorArgs(buffer)` and appends to compile-time args. Device creates `TensorAccessor(args, addr, page_size)` and uses `get_noc_addr(page_id)` for address generation.

3. **Source**: `ttnn/api/ttnn/common/constants.hpp`
   **Reason**: Needed to find the value of `MAX_PACK_UNTILIZE_WIDTH`.
   **Key Information**: `MAX_PACK_UNTILIZE_WIDTH = 8`, limiting pack_untilize usage for FLOAT32 with wide blocks.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Needed to understand the unified untilize compute helper used by both compute kernel variants.
   **Key Information**: The helper auto-dispatches between three paths (single-pass pack_untilize, block-based pack_untilize, standard untilize) based on block width vs DEST capacity and data format. Provides `InitUninitMode` and `WaitMode` configuration enums.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Needed to understand DEST register capacity detection.
   **Key Information**: DEST capacity depends on sync mode and accumulation mode: 16/8/8/4 tiles for full-sync-fp16/full-sync-fp32/half-sync-fp16/half-sync-fp32 respectively.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Needed to understand the `create_cb` helper function signature and behavior.
   **Key Information**: Wraps `CircularBufferConfig` creation. Takes CB index, program, core spec, page size, num pages, and data format. Returns tuple of (cb_index, cb_handle).
