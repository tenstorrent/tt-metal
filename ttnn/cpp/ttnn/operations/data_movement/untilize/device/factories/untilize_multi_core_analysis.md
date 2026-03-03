# Untilize Multi-Core Program Factory Implementation Analysis

## Overview

The **Untilize Multi-Core** program factory converts tensors from **tiled layout** (32x32 tiles with face structure) to **row-major layout** (contiguous rows of elements). This is the general-purpose multi-core untilize variant (`UntilizeMultiCoreProgramFactory`) that supports both interleaved and sharded input tensors, writing to interleaved or sharded output tensors.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Context for downstream usage**: This analysis serves as a reference for the **output stage** of a hybrid operation that reads RM sticks, tilizes them, performs compute, then untilizes back to RM. The key aspects are: how CB c_16 feeds the writer, how the writer maps untilized output sticks to row-major pages, and what compile-time/runtime arguments drive this process.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = one row of tiles spanning the input width) |
| **Unit size** | `num_tiles_per_input_block` tiles (tiles across one tile-row of the input or shard) |
| **Total units** | `num_tiles_per_col` tile-rows for interleaved; `num_input_blocks_per_full_core` per shard for sharded |
| **Loop structure** | Outer: iterate over assigned tile-rows (blocks). Inner (compute): process one block width of tiles through untilize. Inner (writer): iterate `tile_height` rows within each block, writing sticks to output pages. |

A "block" in this operation corresponds to one horizontal row of tiles. For example, if the tensor is 128 tiles wide and 4 tile-rows tall, there are 4 blocks, each containing 128 tiles. Each core processes a contiguous range of these tile-rows.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | N-dimensional (flattened to 2D: `tensor_height x tensor_width`) |
| **Dimension convention** | Last dim = width (C-contiguous) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with face sub-structure) |
| **Memory layout** | INTERLEAVED or SHARDED (height-sharded, width-sharded, or block-sharded) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

If sharded, the shard shape is `(input_shard_height, input_shard_width)` with shard grid and orientation from the shard spec.

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same volume as input |
| **Dimension convention** | Last dim = width |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks of contiguous elements) |
| **Memory layout** | INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (no type conversion) |

Output pages: For interleaved or height-sharded output, `output_page_width = tensor_width`. For width-sharded or block-sharded output, `output_page_width = shard_spec.shape[1]` (the shard's width). Each output page is one row-major stick of width `output_page_width`.

### Layout Transformation

The core transformation is **tile-to-row-major conversion**:
1. Input arrives as tiles (32x32 face-structured pages)
2. Compute kernel untilizes: rearranges data from tile order to row-major order within the output CB
3. Writer kernel extracts individual row-major sticks (of `tile_height` rows, each `num_cols_per_input_block` wide) and writes them to the correct output page locations

---

## Data Flow Pattern

### Interleaved Input Path

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader (`reader_unary_start_id.cpp`) | DRAM (src0_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` | Reads tiles one at a time from DRAM using TensorAccessor; sequential tile IDs starting at `start_page_id` |
| 2 | Compute (`untilize_variable_num_blocks.cpp` or `pack_untilize_variable_num_blocks.cpp`) | CB c_0 | CB c_16 | `cb_wait_front(c_0, block_width)`, `cb_pop_front(c_0, block_width)`, `cb_reserve_back(c_16, block_width)`, `cb_push_back(c_16, block_width)` | Untilizes one block (tile-row) at a time; converts tile layout to row-major in output CB |
| 3 | Writer (`writer_unary_stick_layout_split_rows_multi_core.cpp`) | CB c_16 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_16, num_tiles_per_input_block)`, `cb_pop_front(c_16, num_tiles_per_input_block)` | Reads untilized data from CB; writes row-major sticks to output pages via TensorAccessor |

### Sharded Input Path

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader (`reader_unary_sharded.cpp`) | L1 (shard, already in CB) | CB c_0 | `cb_push_back(c_0, num_tiles_per_core)` | No actual data movement; CB c_0 is backed by the shard buffer; just signals availability |
| 2 | Compute | CB c_0 | CB c_16 | Same as interleaved | Same untilize logic |
| 3 | Writer | CB c_16 | DRAM/L1 (dst_buffer) | Same as interleaved | Same stick-writing logic |

**Key insight for sharded input**: The input CB (`c_0`) is configured with `set_globally_allocated_address(*src0_buffer)`, meaning its L1 address points directly to the shard's memory. The reader kernel simply does `cb_push_back` to make tiles "available" in the CB without any NoC transfers. The CB capacity equals the entire shard (`num_tiles_per_input_block * num_input_blocks_per_full_core` tiles).

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 (`CBIndex::c_0`) | cb_src0 | Input tile staging | See below | `num_tiles_per_input_block` | See below | Reader | Compute | Block |
| c_16 (`CBIndex::c_16`) | cb_output | Output untilized staging | See below | `num_tiles_per_input_block` | See below | Compute | Writer | Block |

### CB c_0 Capacity Details

**Sharded input**: `num_tiles_per_input_block * num_input_blocks_per_full_core` tiles (entire shard). Single-buffered (capacity = total work). The CB is globally allocated to the shard buffer address.

**Interleaved input**:
- If `num_input_blocks_per_full_core == 1`: `num_tiles_per_input_block` tiles. **Single-buffered** (capacity = 1 block).
- If `num_input_blocks_per_full_core >= 2`: `num_tiles_per_input_block * 2` tiles. **Double-buffered** (capacity = 2 blocks). This allows the reader to prefetch the next block while compute processes the current one.

### CB c_16 Capacity Details

- If `num_input_blocks_per_full_core == 1`: `num_tiles_per_input_block` tiles. **Single-buffered**.
- If `num_input_blocks_per_full_core >= 2`: `num_tiles_per_input_block * 2` tiles. **Double-buffered**. This allows compute to produce the next block while the writer drains the current one.

### Data Formats

- CB c_0: `input_cb_data_format` = `datatype_to_dataformat_converter(a.dtype())`
- CB c_16: `output_cb_data_format` = `datatype_to_dataformat_converter(output.dtype())`

---

## Pipeline Pattern Summary

| Configuration | CB c_0 | CB c_16 | Pipeline Overlap |
|---------------|--------|---------|------------------|
| Single block per core | Single-buffered | Single-buffered | No overlap possible; purely sequential reader -> compute -> writer |
| Multiple blocks, interleaved | Double-buffered | Double-buffered | Reader/compute overlap on c_0; compute/writer overlap on c_16 |
| Sharded input | Single-buffered (entire shard) | Double-buffered (if multiple blocks) | No reader/compute overlap (reader instant); compute/writer overlap on c_16 |

---

## Index Calculations

### Reader (Interleaved): Tile Index Mapping

The reader uses `TensorAccessor` with the source buffer. Tiles are addressed by a linear `page_id` starting from `start_page_id` (runtime arg). The TensorAccessor resolves this to the correct DRAM bank and address via its internal bank-interleaving logic.

```
For core i (0-indexed):
  start_page_id = i * num_tiles_per_input_block * num_input_blocks_per_full_core
  end_page_id = start_page_id + num_tiles_per_input_block * num_input_blocks_to_process
```

Tiles are read sequentially: page_id increments from `start_page_id` to `end_page_id - 1`.

### Writer: Output Stick Addressing

The writer performs a more complex index calculation that maps untilized sticks to output pages. Key variables:

- `height_wise_input_block_start_index`: Which tile-row (globally) this core starts processing
- `width_wise_output_block_start_index`: Which output page column this core's data maps to
- `num_cols_already_processed_in_first_output_block`: Byte offset within the first output page

For each block at `block_height_index`:
```
For each row j in [0, tile_height):
  num_rows_already_processed = block_height_index * tile_height + j
  output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index
```

The writer iterates across columns within each row, potentially spanning multiple output pages if the input block width does not align with the output page width (common with width/block sharding).

### Uneven Shard Handling

For sharded inputs, the factory handles two types of unevenness:

1. **Width-wise**: The last shard in each row may have padding columns. `num_unpadded_cols_per_input_block` is computed as:
   ```
   num_cols_per_input_block - (round_up(tensor_width, input_shard_width) - tensor_width)
   ```
   This prevents writing garbage padding data.

2. **Height-wise**: The last shard in each column may have fewer valid tile-rows. `num_input_blocks_to_process` is reduced for the last height-wise shard:
   ```
   num_input_blocks_per_full_core - (round_up(tensor_height, input_shard_height) - tensor_height) / tile_height
   ```

---

## Memory Access Patterns

### Read Pattern (Interleaved Input)

- **Ordering**: Sequential by tile ID (row-major tile order within the tensor)
- **Granularity**: One tile at a time
- **Location**: DRAM banks (round-robin interleaved)
- **Barrier**: `noc_async_read_barrier()` after each tile (no batch prefetch within a single CB reserve)
- **Implication**: Each tile read involves a NoC0 read from DRAM to L1; the double-buffered CB c_0 hides some of this latency

### Read Pattern (Sharded Input)

- **Ordering**: N/A (data already in L1)
- **Granularity**: Entire shard signaled at once
- **Location**: L1 (shard buffer = CB c_0 address)
- **No NoC traffic**: Reader just signals CB availability

### Write Pattern

- **Ordering**: Row-major within each block; blocks processed sequentially
- **Granularity**: Variable-length stick writes (partial or full output pages)
- **Location**: Output buffer (DRAM or L1) via TensorAccessor
- **Pattern**: For each tile-row block, iterates over `tile_height` rows, writing contiguous column segments to output pages
- **Barrier**: `noc_async_write_barrier()` after each complete block (all `tile_height` rows written)

---

## Core Distribution Strategy

### Interleaved Input

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (blocks) per full core |
| **Cliff core** | 0 or 1 cliff core with `num_rows_per_cliff_core` tile-rows |
| **Load balancing** | Nearly equal; at most 1 cliff core with fewer blocks |

The `split_blocks_for_tilize(grid_size, num_tiles_per_col)` function computes:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- Cliff core gets `num_tiles_per_col % nblocks_per_core` remaining blocks

### Sharded Input

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Matches shard grid |
| **Grid dimensions** | From shard spec grid |
| **Total cores** | `shard_spec.grid.num_cores()` (may be reduced if more cores than shards) |
| **Work per core** | `input_shard_height / tile_height` tile-rows per core |
| **Cliff core** | No cliff core; unevenness handled per-core via runtime args |
| **Load balancing** | Equal (all cores have same shard size); last cores in row/column may skip padding |

For sharded input, the factory also handles the case where `num_compute_cores > num_shards`:
- Uses `BufferDistributionSpec` to determine which cores actually have data
- `ordered_cores_with_data` provides the exact core list
- Width-wise and height-wise block indexing uses `i / num_input_blocks_across_width` and `i % num_input_blocks_across_width`

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved: `reader_unary_start_id.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input circular buffer index (c_0) |
| 1+ | TensorAccessor args | uint32_t[] | `TensorAccessorArgs(*src0_buffer)` appended; encodes interleave config (is_dram flag) |

#### Reader Kernel (Sharded: `reader_unary_sharded.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input circular buffer index (c_0) |

#### Compute Kernel (both variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | `num_tiles_per_input_block` -- tiles per tile-row (block width in tiles) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

Additionally, `DST_ACCUM_MODE` is defined as `"1"` for INT32, UINT32, or FLOAT32 data types.

#### Writer Kernel (`writer_unary_stick_layout_split_rows_multi_core.cpp`)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output circular buffer index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size in bytes of one output page: `num_cols_per_output_block * output_element_size` |
| 2 | `tile_height` | uint32_t | Tile height (32 for standard tiles) |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per input block (tile-row width in tiles) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per row (1 for interleaved/height-sharded; >1 for width/block-sharded) |
| 5 | `output_element_size` | uint32_t | Bytes per output element (2 for bfloat16, 4 for float32, etc.) |
| 6 | `num_cols_per_input_block` | uint32_t | `num_tiles_per_input_block * tile_width` -- columns per input block |
| 7 | `num_cols_per_output_block` | uint32_t | `output_page_width` -- columns per output page |
| 8+ | TensorAccessor args | uint32_t[] | `TensorAccessorArgs(*dst_buffer)` appended |

### Runtime Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer DRAM address |
| 1 | `num_tiles` | uint32_t | Total tiles to read: `num_tiles_per_input_block * num_input_blocks_to_process` |
| 2 | `start_page_id` | uint32_t | First tile ID for this core |

#### Reader Kernel (Sharded)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `num_tiles_per_core` | uint32_t | Total tiles in shard: `num_tiles_per_input_block * num_input_blocks_to_process` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of tile-row blocks to process (`num_input_blocks_to_process`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer address |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-row blocks this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Global tile-row index where this core starts |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Columns of valid (non-padding) data in each input block |
| 4 | `width_wise_output_block_start_index` | uint32_t | First output page column index for this core |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within the first output page (for width/block sharding alignment) |

---

## Kernel Implementations

### Reader Kernel (Interleaved)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (interleaved tiles) | CB c_0 | Read tiles sequentially via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Key Logic**: Creates `TensorAccessor` from compile-time args at index 1+. Iterates from `start_page_id` to `start_page_id + num_tiles`, reading one tile per iteration. Uses `cb_reserve_back(c_0, 1)` / `cb_push_back(c_0, 1)` for single-tile flow control. Calls `noc_async_read_barrier()` after each tile (ensures tile is in L1 before signaling CB).

### Reader Kernel (Sharded)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_sharded | RISCV_0 | N/A | L1 shard (pre-loaded) | CB c_0 | Signal CB availability (no data movement) |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
- **Key Logic**: Single `cb_push_back(cb_id_in0, num_tiles_per_core)`. The CB is globally allocated to the shard buffer, so data is already in place. This kernel merely advances the CB write pointer so the compute kernel can consume tiles.

### Compute Kernel (Slow Path: Standard Untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| untilize_variable_num_blocks | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | `compute_kernel_lib::untilize` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**: Calls `compute_kernel_hw_startup(src_cb_id, out_cb_id)` to initialize unpack/math/pack threads. Then calls `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`. The helper function handles `cb_wait_front`/`cb_pop_front` on input and `cb_reserve_back`/`cb_push_back` on output internally.
- **Selection**: Used when `use_pack_untilize == false`, or `dtype == UINT16`, or (`dtype == FLOAT32` and `num_tiles_per_input_block >= 8`).

### Compute Kernel (Fast Path: Pack Untilize)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize_variable_num_blocks | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | `compute_kernel_lib::untilize` (with pack_untilize hw path) |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Key Logic**: Same interface as slow path. The `compute_kernel_lib::untilize` helper automatically selects `pack_untilize_block` (hardware-accelerated) vs standard untilize based on whether `block_width_tiles > DEST_AUTO_LIMIT`. If width exceeds DEST capacity, it splits into sub-blocks.
- **Selection**: Default (fast) path when `use_pack_untilize == true` and dtype is not UINT16 and not (FLOAT32 with width >= 8).
- **DEST limit handling**: `DST_ACCUM_MODE` define reduces DEST capacity (8->4 in half-sync for FP32/INT32). The helper uses `compute_num_blocks(block_width_tiles, DEST_AUTO_LIMIT)` to find the best sub-block decomposition.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_multi_core | RISCV_1 | NOC1 | CB c_16 | DRAM/L1 (output buffer) | Write row-major sticks to output pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  1. Outer loop: iterates over `num_input_blocks_to_process` blocks
  2. For each block: `cb_wait_front(c_16, num_tiles_per_input_block)` to wait for untilized data
  3. Inner loop: iterates over `tile_height` rows within the block
  4. For each row: computes `output_page_id` based on global row index and output block structure
  5. Innermost while loop: handles cases where one input block row spans multiple output pages (width/block sharding). Writes `min(remaining_input_cols, remaining_output_page_cols)` columns per iteration via `noc_async_write`.
  6. After all rows: `noc_async_write_barrier()` then `cb_pop_front(c_16, num_tiles_per_input_block)`
  7. Uses `TensorAccessor` constructed from compile-time args at index 8+ with `dst_addr` and `output_stick_size`

---

## Implementation Notes

### Compute Kernel Selection Logic

The program factory selects between two compute kernels (line 232-244):

```
if (!use_pack_untilize || dtype == UINT16 || (dtype == FLOAT32 && width >= MAX_PACK_UNTILIZE_WIDTH)):
    slow path: untilize_variable_num_blocks.cpp
else:
    fast path: pack_untilize_variable_num_blocks.cpp
```

Despite different kernel files, both call the same unified `compute_kernel_lib::untilize` helper. The differentiation is mostly historical; the helper handles all dispatch internally. The `DST_ACCUM_MODE` define is set for FLOAT32, INT32, and UINT32, and when this define is active the slow path forces `UnpackToDestMode::Default` (overriding `fp32_dest_acc_en` for unpack mode).

### Width/Block Sharding Output Handling

The writer kernel is designed to handle the general case where input blocks and output pages have different widths. This arises when:
- Input is width-sharded or block-sharded with a different shard width than the output
- Output is width-sharded or block-sharded

The writer computes `num_cols_already_processed_in_first_output_block` to handle partial writes to the first output page, then iterates through full output pages until all input columns are written.

### Override Runtime Arguments

The `override_runtime_arguments` method (line 424-459) supports program caching by updating only buffer addresses:
- For sharded input: updates CB c_0's globally allocated address via `UpdateDynamicCircularBufferAddress`
- For interleaved input: updates `src_addr` (runtime arg index 0) for the reader
- Always updates `dst_addr` (runtime arg index 0) for the writer

### Uneven Shard Support

The factory has extensive logic (lines 77-123, 296-323) to handle cases where:
1. The number of cores exceeds the number of shards (uses `BufferDistributionSpec` to identify cores with data)
2. The last shard in a row is narrower than full shards (width-wise unevenness)
3. The last shard in a column is shorter than full shards (height-wise unevenness)

### FP32 Destination Accumulation

When `fp32_dest_acc_en` is true, the unpack mode for CB c_0 is set to `UnpackToDestFp32`, which causes the unpacker to promote data to FP32 in the DEST registers. This halves the DEST capacity but preserves precision. The `DST_ACCUM_MODE` compile define propagates this to the compute kernel's JIT-generated headers.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in TTNN? What is pack_untilize vs standard untilize?"
   **Reason**: Needed to understand the distinction between the two compute paths and when hardware-accelerated pack_untilize is selected.
   **Key Findings**: `pack_untilize` uses the `llk_pack_untilize` hardware instruction for faster tile-to-RM conversion. Standard untilize uses unpack+copy+pack. Selection depends on `use_pack_untilize` flag, data type, and width. UINT16 always uses slow path. FLOAT32 with width >= 8 uses slow path.

2. **Query**: "How does TensorAccessor work in tt-metal kernels? What does TensorAccessorArgs do?"
   **Reason**: Both reader and writer use TensorAccessor for address resolution; needed to understand what compile-time args it appends.
   **Key Findings**: For interleaved tensors, `TensorAccessorArgs` appends just the `args_config` value (encodes is_dram flag). For sharded tensors, it appends rank, num_banks, tensor shape, shard shape, and bank coordinates. The device-side `TensorAccessor` resolves page_id to NoC address via bank-interleaving or shard-mapping logic.

3. **Query**: "What is split_blocks_for_tilize and how does it handle cliff cores?"
   **Reason**: The program factory uses this for interleaved work distribution.
   **Key Findings**: Computes `nblocks_per_core = ceil(nblocks / grid_area)` and `ncores = ceil(nblocks / nblocks_per_core)`. The last core may be a "cliff" core handling `nblocks % nblocks_per_core` remainder blocks. Returns separate `CoreRangeSet` for full cores and cliff core.

4. **Query**: "What is the CBIndex::c_0 and CBIndex::c_16 convention in tt-metal?"
   **Reason**: Needed to confirm the CB index convention used in the factory.
   **Key Findings**: c_0 is conventionally used for primary input, c_16 for primary output. Up to 32 CBs (c_0 through c_31). Indices c_0-c_7 are typically inputs, c_16-c_23 are typically outputs, c_24-c_31 are intermediates.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile layout, face structure, and how interleaved/sharded memory layouts work.
   **Key Information**: Tiles are 32x32 with 16x16 faces. Interleaved distributes pages round-robin across banks. Sharded places contiguous pages on specific cores. Row-major layout has one row = one page.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side usage.
   **Key Information**: `TensorAccessorArgs(buffer)` with default config makes all args compile-time. `append_to(compile_time_args)` adds them to the CTA vector. Device-side `TensorAccessor(args, addr, page_size)` constructs accessor from CTAs. `get_noc_addr(page_id)` resolves to bank address; `get_noc_addr(page_id, offset)` adds byte offset within page.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize helper API used by both compute kernels.
   **Key Information**: `compute_kernel_lib::untilize<block_width, in_cb, out_cb>(num_blocks)` is the unified API. Automatically selects pack_untilize (single-pass or block-based) based on DEST capacity. Handles CB wait/pop/reserve/push internally. Supports WaitBlock (default), WaitUpfront, and NoWait modes.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity calculation used in untilize path selection.
   **Key Information**: `DEST_AUTO_LIMIT` = 16 (SyncFull+16bit), 8 (SyncFull+32bit or SyncHalf+16bit), or 4 (SyncHalf+32bit). When `block_width_tiles > DEST_AUTO_LIMIT`, the helper splits into sub-blocks using `compute_num_blocks()`.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how `split_blocks_for_tilize` distributes work across cores.
   **Key Information**: Returns `BlockSplit{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff}`. Cliff core is always the last core and processes the remainder.
