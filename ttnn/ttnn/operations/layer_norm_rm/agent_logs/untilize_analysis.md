# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts tiled tensor data (32x32 tiles) back into row-major (RM) stick layout. It reads tiles from DRAM (or sharded L1), passes them through compute kernels that reorder elements from tile-order to row-order using hardware-accelerated pack_untilize, and writes the resulting RM sticks back to DRAM via a writer kernel that uses TensorAccessor for address translation.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus of this analysis**: Output stage patterns -- how untilized RM sticks are produced in the output CB, sized, and written to DRAM. This serves as a reference for layer_norm_rm's output stage.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = `num_tiles_per_row` tiles) |
| **Unit size** | `num_tiles_per_input_block` tiles (= tiles across one row of the tensor, or shard width / tile_width for sharded) |
| **Total units** | `num_tiles_per_col` blocks (= tensor_height / tile_height) |
| **Loop structure** | Outer: `num_input_blocks_to_process` (blocks per core), Inner: tile_height rows per block |

One "input block" corresponds to one tile-row -- a horizontal strip of tiles spanning the width assigned to this core. Processing one block means: (1) reading `num_tiles_per_input_block` tiles into the input CB, (2) untilizing them in compute to produce `tile_height` RM rows in the output CB, (3) writing those `tile_height` RM rows to DRAM.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (arbitrary rank, flattened to 2D) | Same logical shape |
| **Dimension convention** | Flattened to `tensor_height x tensor_width` | Flattened to `tensor_height x tensor_width` |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED (or sharded for width/block sharding) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) | DRAM (interleaved) or L1 (sharded) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same as input |

### Layout Transformation

The core transformation is TILE_LAYOUT to ROW_MAJOR_LAYOUT:
- Input: tiles are stored as 32x32 blocks with 16x16 faces (face0, face1, face2, face3 in row-major face order).
- Output: elements are stored as contiguous row-major sticks where each stick = one full row of the tensor width.
- The compute kernel's `pack_untilize` hardware instruction reorders elements from tile/face order into contiguous row-major order within the output CB.

### Output Page Convention (Critical for layer_norm_rm)

For interleaved output, the **output page** ("stick") is one full row of the tensor:
- `output_page_width = tensor_width` (line 190 of program factory)
- `output_stick_size = output_page_width * output_element_size`

For width/block sharded output, the page width equals the shard width:
- `output_page_width = shard_spec.shape[1]`

This means for interleaved RM output, the TensorAccessor page_id maps to row index, and each page is one complete tensor row.

---

## Data Flow Pattern

### Stage 1: Reader (de-emphasized)
Reads tiles from DRAM/L1 into input CB (c_0). For interleaved, uses TensorAccessor to read one tile at a time with `noc_async_read`. For sharded, the input CB is backed by the shard buffer directly (globally allocated address).

### Stage 2: Compute (untilize)
The compute kernel uses the `compute_kernel_lib::untilize` helper from `untilize_helpers.hpp`:

```
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,   // = num_tiles_per_input_block (tiles per row)
    src_cb_id,                 // = c_0 (input CB, tiled data)
    out_cb_id,                 // = c_16 (output CB, row-major data)
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,
    ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
```

The helper determines the implementation path at compile time:
- If `block_width_tiles <= DEST_AUTO_LIMIT` (8 for bf16, 4 for fp32): **single-pass pack_untilize** -- all tiles in the row fit in DEST at once.
- If `block_width_tiles > DEST_AUTO_LIMIT`: **block-based pack_untilize** -- splits the row into sub-blocks that each fit in DEST.

**Per-block loop** (single-pass path, the common case):
```
for each block (row of tiles):
    cb_wait_front(input_cb, block_width_tiles)     // wait for all tiles in this row
    cb_reserve_back(output_cb, block_width_tiles)   // reserve space in output CB
    pack_untilize_block(...)                        // HW-accelerated tile->RM conversion
    cb_pop_front(input_cb, block_width_tiles)       // release input tiles
    cb_push_back(output_cb, block_width_tiles)      // signal output data ready
```

**Output CB data layout after untilize**: After `pack_untilize_block` processes one block (one tile-row), the output CB contains `tile_height` (32) contiguous RM rows, each `num_tiles_per_input_block * tile_width` elements wide. The rows are stored consecutively. This is the data the writer kernel reads.

### Stage 3: Writer (output stage -- primary focus)
The writer kernel reads untilized RM rows from the output CB and writes them to DRAM using TensorAccessor.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | (see below) | `num_tiles_per_input_block` tiles | Single or Double | Reader | Compute | Block |
| c_16 | cb_output | Output RM staging | (see below) | `num_tiles_per_input_block` tiles | Single or Double | Compute | Writer | Block |

### Output CB Sizing (c_16) -- Key for layer_norm_rm

The output CB capacity is determined by the number of blocks the core processes (program factory lines 149-156):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;      // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;  // Double-buffered
}
```

**Sizing logic**:
- **Single-buffered** (capacity = 1 block): When each core processes only 1 block (1 tile-row), there is no overlap opportunity between compute and write, so no need for double buffering.
- **Double-buffered** (capacity = 2 blocks): When each core processes 2+ blocks, double buffering allows compute to fill one buffer slot while the writer drains the other.

**Block size** = `num_tiles_per_input_block` tiles. Each tile in the output CB is `output_single_tile_size` bytes (tile_size for the output data format).

**Physical size in bytes** = `output_cb_num_tiles * output_single_tile_size`.

Note: Even though the output data is row-major, the CB is still sized in terms of "tiles" because the compute kernel's pack_untilize writes data in tile-sized pages to the CB. The CB page_size is `output_single_tile_size` (the size of one tile in the output data format). What changes is the internal layout of data within those pages -- the elements are in RM order rather than tile order.

### Input CB Sizing (c_0) -- Brief

For interleaved: single-buffered if 1 block/core, double-buffered if 2+ blocks/core (same logic as output).
For sharded: the entire shard is the CB (`num_tiles_per_input_block * num_input_blocks_per_full_core` tiles), backed by the shard buffer's globally allocated address.

---

## Pipeline Pattern Summary

| Pattern | Condition | Overlap |
|---------|-----------|---------|
| Single-buffered | Core processes 1 block | No overlap: compute finishes, then writer runs |
| Double-buffered | Core processes 2+ blocks | Compute block N+1 overlaps with writer draining block N |

---

## Writer Kernel Deep Dive

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output page/stick in bytes = `output_page_width * element_size` |
| 2 | tile_height | uint32_t | Height of one tile (32 for standard tiles) |
| 3 | num_tiles_per_input_block | uint32_t | Number of tiles in one input block (tiles across assigned width) |
| 4 | num_output_blocks_across_width | uint32_t | Number of output pages that span the full tensor width (= tensor_width / output_page_width) |
| 5 | output_element_size | uint32_t | Size of one output element in bytes (2 for bf16, 4 for fp32) |
| 6 | num_cols_per_input_block | uint32_t | Number of columns (elements) in one input block = `num_tiles_per_input_block * tile_width` |
| 7 | num_cols_per_output_block | uint32_t | Number of columns (elements) in one output page = `output_page_width` |
| 8+ | TensorAccessorArgs | (variable) | TensorAccessor compile-time args for the output buffer |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_input_blocks_to_process | uint32_t | Number of tile-row blocks this core must write |
| 2 | height_wise_input_block_start_index | uint32_t | Starting block index (row of tiles) for this core |
| 3 | num_unpadded_cols_per_input_block | uint32_t | Actual valid columns (handles uneven shard padding) |
| 4 | width_wise_output_block_start_index | uint32_t | Starting output page index for width offset |
| 5 | num_cols_already_processed_in_first_output_block | uint32_t | Column offset within first output page (for mid-page writes) |

### Writer Kernel Logic (Stick Extraction from Tiles)

The writer kernel processes one block at a time in `write_tiles_in_current_block`:

1. **Wait for compute output**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- waits until compute has pushed one full block of untilized data.

2. **Get base L1 read address**: `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- this points to the start of the untilized data in the output CB.

3. **Iterate over `tile_height` rows**: For each of the 32 rows within the block:
   - Calculate L1 read address for this row: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`
   - This formula works because after untilize, data in the CB is laid out as contiguous RM rows, each `num_cols_per_input_block * element_size` bytes wide.

4. **For each row, iterate across output pages**:
   - Compute the output page_id: `(block_height_index * tile_height + j) * num_output_blocks_across_width + width_wise_output_block_start_index`
   - This maps (row_index, col_offset) to the interleaved page_id.
   - Get NoC address: `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` -- TensorAccessor translates page_id to physical DRAM bank + offset, with an optional byte offset within the page.
   - Write: `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`

5. **Handle width splitting**: When input blocks and output pages have different widths (e.g., sharded input, differently-sharded output), the writer uses a while loop to split one RM row across multiple output pages, advancing `output_page_id` as needed.

6. **Barrier and release**: `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` to release the CB space.

### Key Pattern: TensorAccessor for RM Stick Writing

The writer constructs a TensorAccessor on the device side:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

Key insight: The TensorAccessor is configured with `output_stick_size` as the page size. For interleaved output, this means each "page" maps to one full RM row of the tensor. The `get_noc_addr(page_id, byte_offset)` call with an optional byte offset allows writing to a specific column position within a page.

**For layer_norm_rm relevance**: When writing RM output to interleaved DRAM, the pattern is:
1. Configure TensorAccessor with `page_size = tensor_width * element_size` (one full RM row)
2. Use `page_id = row_index` to address each row
3. Use `noc_async_write(l1_addr, noc_addr, stick_size)` to write one complete row at a time

### Simplified Writer Pattern (Interleaved, Same Width)

For the common case where input_block_width == output_page_width (typical for interleaved-to-interleaved with no sharding), the writer simplifies to:

```
for each block i in [0, num_input_blocks_to_process):
    cb_wait_front(cb_out, num_tiles_per_input_block)
    base_l1_addr = get_read_ptr(cb_out)

    for each row j in [0, tile_height):
        l1_addr = base_l1_addr + j * num_cols_per_input_block * element_size
        row_index = (block_start + i) * tile_height + j
        page_id = row_index * 1 + 0    // one output page per row, no width offset
        noc_addr = accessor.get_noc_addr(page_id, 0)
        noc_async_write(l1_addr, noc_addr, stick_size)

    noc_async_write_barrier()
    cb_pop_front(cb_out, num_tiles_per_input_block)
```

This is the essential output-stage pattern for layer_norm_rm.

---

## Index Calculations

### Output Page ID Mapping

For each RM row within a block, the output page_id is computed as:

```
row_index = block_height_index * tile_height + j       // absolute row within tensor
pages_in_previous_rows = row_index * num_output_blocks_across_width
output_page_id = pages_in_previous_rows + width_wise_output_block_start_index
```

For interleaved output with no width sharding: `num_output_blocks_across_width = 1` and `width_wise_output_block_start_index = 0`, so `output_page_id = row_index`.

### L1 Read Address Within Output CB

After untilize, the output CB contains RM data laid out as:
```
Row 0:  [col_0, col_1, ..., col_{W-1}]   at offset 0
Row 1:  [col_0, col_1, ..., col_{W-1}]   at offset W * element_size
...
Row 31: [col_0, col_1, ..., col_{W-1}]   at offset 31 * W * element_size
```
where W = `num_cols_per_input_block` = `num_tiles_per_input_block * tile_width`.

The address for row j is: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`.

---

## Memory Access Patterns

### Read Pattern (Writer reads from L1 output CB)
- **Sequential within a block**: Reads rows 0..31 in order from contiguous L1 memory.
- **Stride**: `num_cols_per_input_block * element_size` bytes between consecutive rows.
- **Size per read**: `num_cols_per_output_block * element_size` bytes (one output page worth of data).

### Write Pattern (Writer writes to DRAM)
- **Row-sequential**: Writes consecutive RM rows to incrementing page_ids.
- **DRAM distribution**: Pages are interleaved across DRAM banks (round-robin by page_id via TensorAccessor).
- **One `noc_async_write` per output page per row**: For interleaved output with same width, this is one write per row.
- **Write barrier per block**: `noc_async_write_barrier()` is called once per block (every `tile_height` = 32 rows), not per row. This batches 32 write operations before waiting.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` (device compute grid) |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` blocks (tile-rows) per full core |
| **Load balancing** | Nearly equal, with optional cliff core for remainder |
| **Remainder handling** | Last core (cliff core) gets `num_rows_per_cliff_core` blocks |

The function `split_blocks_for_tilize(grid_size, num_tiles_per_col)` divides tile-rows across cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- Cliff core gets `num_tiles_per_col % nblocks_per_core` blocks (if non-zero)

For sharded input, core distribution follows the shard spec grid directly.

---

## Arguments Summary

### Compile-Time Arguments

#### Reader (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessorArgs | (variable) | Source buffer accessor args |

#### Compute

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_tile_cnt | uint32_t | Tiles per row (= num_tiles_per_input_block) |
| 1 | src_cb_id | uint32_t | Input CB index (c_0) |
| 2 | out_cb_id | uint32_t | Output CB index (c_16) |

#### Writer (see detailed table above in Writer section)

### Runtime Arguments

#### Reader (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to read for this core |
| 2 | start_page_id | uint32_t | First tile page_id to read |

#### Compute

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) to process |

#### Writer (see detailed table above in Writer section)

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM tiles | CB c_0 | Read tiles via TensorAccessor |
| compute | RISCV_2 | N/A | CB c_0 (tiled) | CB c_16 (RM) | pack_untilize: tile-to-RM conversion |
| writer | RISCV_1 | NOC1 | CB c_16 (RM) | DRAM sticks | Write RM rows via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Key Logic**: Simple tile-by-tile read loop using TensorAccessor. One tile per iteration with barrier per tile.

### Compute Kernel (Two Variants)
- **File (standard)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **File (fast pack)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Key Logic**: Both use `compute_kernel_lib::untilize<>()` helper. The helper determines the fastest path based on block width vs DEST capacity. The `compute_kernel_hw_startup(src_cb_id, out_cb_id)` call initializes srcA = srcB = input_cb. Standard path used for UINT16 or FLOAT32 with wide rows; fast pack path is the default.
- **Selection criteria** (program factory line 232-243): Standard untilize when `!use_pack_untilize || dtype==UINT16 || (dtype==FLOAT32 && width >= MAX_PACK_UNTILIZE_WIDTH)`.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**: Detailed above in the Writer Kernel Deep Dive section.

---

## Implementation Notes

### Output CB Uses Tile-Sized Pages Despite RM Data
Even though the output data is in row-major order, the output CB is configured with `page_size = output_single_tile_size` (the size of one tile in the output format). This is because the compute kernel's `pack_untilize` writes data to the CB in tile-sized chunks. The CB capacity is measured in tiles even though the data within those pages is RM-ordered. The writer kernel knows the actual RM layout and reads row-by-row using calculated offsets.

### TensorAccessor get_noc_addr with Byte Offset
The writer uses `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` -- the second parameter is a byte offset within the page. This is used when the writer needs to write to the middle of an output page (e.g., when input block width doesn't align with output page width due to sharding differences). For simple interleaved-to-interleaved cases, this offset is 0.

### Write Barrier Batching
The writer calls `noc_async_write_barrier()` once per block (after writing all `tile_height` = 32 rows), not per individual row. This allows up to 32 write operations to be in flight simultaneously, improving throughput.

### Double Buffering Decision
The double-buffering decision is identical for input and output CBs and depends solely on whether the core processes more than one block. Single block means no pipeline opportunity, so single-buffered saves L1 space. Two or more blocks enable pipelining between reader/compute/writer.

### Compute Kernel DST_ACCUM_MODE
For INT32, UINT32, and FLOAT32 data types, `DST_ACCUM_MODE` is defined as `1`. This halves the DEST register capacity (from 8 to 4 tiles for half-sync), which affects whether the block-based or single-pass untilize path is chosen.

### fp32_dest_acc_en
When `fp32_dest_acc_en` is true, the unpack mode is set to `UnpackToDestFp32` for the input CB, ensuring full precision during the untilize operation. This is separate from `DST_ACCUM_MODE`.

---

## Key Takeaways for layer_norm_rm Output Stage

1. **Output CB sizing**: Allocate the output CB with `num_tiles = Wt` (tiles per row) for single-buffered, or `Wt * 2` for double-buffered. Page size = `tile_size(output_data_format)`.

2. **Untilize helper usage**: Use `compute_kernel_lib::untilize<Wt, in_cb, out_cb>(num_blocks)` with appropriate InitUninitMode. If untilize follows other compute operations in the same kernel, use `InitOnly`/`Neither`/`UninitOnly` to avoid redundant init.

3. **Writer pattern for interleaved RM output**:
   - TensorAccessor with `page_size = tensor_width * element_size`
   - After `cb_wait_front(out_cb, Wt)`, read `tile_height` RM rows from CB
   - Row j address: `get_read_ptr(out_cb) + j * Wt * tile_width * element_size`
   - Write each row: `noc_async_write(l1_addr, accessor.get_noc_addr(row_page_id), stick_size)`
   - Barrier per block, then `cb_pop_front(out_cb, Wt)`

4. **CB index convention**: Input at c_0, output at c_16. Using c_16 (index 16) for compute output is a common TTNN convention that separates input and output CB namespaces.

5. **Compute kernel startup**: Always call `compute_kernel_hw_startup(in_cb, out_cb)` before using untilize helpers.

---

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps page_ids to DRAM bank addresses on the device side.
   **Key Information**: TensorAccessor is constructed with `TensorAccessorArgs<offset>()` on device, initialized with `(args, base_addr, page_size)`. `get_noc_addr(page_id)` translates to physical NoC address. For interleaved tensors, pages are distributed round-robin across DRAM banks. The second parameter to `get_noc_addr(page_id, byte_offset)` adds a byte offset within the page.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding page definitions for RM vs tiled layouts.
   **Key Information**: For RM layout, each page = one row of the 2D tensor. For tiled layout, each page = one 32x32 tile. Interleaved distribution maps pages round-robin to banks.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute helper API for untilize.
   **Key Information**: `compute_kernel_lib::untilize<block_width_tiles, input_cb, output_cb, ...>(num_blocks)` is the unified untilize entry point. It auto-selects between single-pass pack_untilize (when width fits DEST) and block-based pack_untilize (when width exceeds DEST). Template parameters control init/uninit lifecycle, wait mode, and register reconfiguration.

4. **Source**: `tt_metal/hw/inc/api/compute/pack_untilize.h`
   **Reason**: Understanding the hardware-level pack_untilize API.
   **Key Information**: `pack_untilize_block<block_ct_dim, full_ct_dim>(icb, block_rt_dim, ocb, block_c_index)` performs the actual tile-to-RM conversion. Block width is limited by DEST capacity: 8 tiles (bf16 half-sync), 4 tiles (fp32 half-sync). The `pack_untilize_dest` variant works when data is already in DEST from another operation (e.g., reduce).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how work (tile-rows) is distributed across cores.
   **Key Information**: `split_blocks_for_tilize(grid_size, num_tiles_per_col)` computes `nblocks_per_core = ceil(total_blocks / grid_area)`, then `ncores = ceil(total_blocks / nblocks_per_core)`. Supports a cliff core for remainder handling.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB. Total CB size = `num_pages * page_size`. If `buffer != nullptr`, sets globally allocated address (used for sharded input).

7. **Source**: `METALIUM_GUIDE.md`
   **Reason**: CB synchronization API reference.
   **Key Information**: `cb_reserve_back/cb_push_back` (producer side), `cb_wait_front/cb_pop_front` (consumer side). `get_read_ptr` returns address for consumer to read from; `get_write_ptr` returns address for producer to write to.
