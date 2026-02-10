# Untilize Single-Core Implementation Analysis

## Overview

The untilize single-core operation converts a tensor from **tiled layout** (32x32 tiles with 16x16 faces) to **row-major layout** (sequential rows/sticks). The operation runs entirely on a single Tensix core (0,0) and consists of three kernels: a tile reader, a compute kernel that performs the actual untilize transformation, and a stick-layout writer that writes row-major sticks to DRAM.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp`

**Key Design Insight**: Untilize is performed in the **compute kernel** (not dataflow). The PACK thread rearranges data from tile order (face-interleaved 32x32) into row-major order within the DEST registers, then writes the result to the output CB in row-major stick format. The writer kernel then writes these row-major sticks to DRAM.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles (one tile-row-wide block) |
| **Unit size** | `num_tiles_per_block` tiles (a horizontal slice of one tile row) |
| **Total units** | `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| **Loop structure** | Outer: height blocks (`num_blocks_across_height`) x column blocks (`num_columns_of_blocks`) x width blocks (`num_blocks_per_column_row`) |

A "block" is a horizontal group of tiles spanning part or all of one tile row. The block width is determined by L1 capacity: `num_tiles_per_block` is the largest divisor of the tile-row width that fits in available CB space. In the common non-sharded case, `num_columns_of_blocks = 1`, so the work is simply height-blocks times width-blocks.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, ..., H_padded, W_padded] (arbitrary rank, padded to tile alignment) |
| **Dimension convention** | Last dim is width (contiguous in memory) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED (pages distributed round-robin across DRAM banks) |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Last dim is width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT (each row is one page/stick) |
| **Memory layout** | INTERLEAVED (or WIDTH_SHARDED / BLOCK_SHARDED) |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input |

### Layout Transformations

The core transformation is tile-to-row-major conversion:
- **Input page**: One 32x32 tile (stored as 4 contiguous 16x16 faces: face0, face1, face2, face3)
- **Output page**: One row-major stick (one complete row of the tensor, width = `padded_shape[-1]`)
- The compute kernel rearranges face-interleaved data within each tile into sequential rows
- Each tile produces `tile_height` (32) output rows, each `tile_width` (32) elements wide
- Multiple tiles across a row produce a single wide output stick

### Page Size Calculations (Critical for layer_norm_rm reuse)

```
input_single_tile_size = tile_size(input_cb_data_format)
    -- For BFLOAT16: 32 * 32 * 2 = 2048 bytes (plus header = ~2080 bytes)
    -- For FLOAT32: 32 * 32 * 4 = 4096 bytes (plus header)

output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks
    -- Simplifies to: padded_shape[-1] * element_size (for non-sharded)
    -- Example: 1024-wide BFLOAT16 tensor -> stick_size = 1024 * 2 = 2048 bytes

output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size()
    -- This is the byte width of one block of untilized output
    -- Example: 4 tiles/block * 32 * 2 = 256 bytes
```

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations | Details |
|-------|--------|------------|-----------|---------------|---------|
| 1 | Reader | DRAM (tiled pages) | CB c_0 | reserve_back(1), push_back(1) | Reads one tile at a time via TensorAccessor; sequential page IDs from `start_page_id` to `start_page_id + num_tiles` |
| 2 | Compute | CB c_0 | CB c_16 | wait_front(block_tiles), pop_front(block_tiles), reserve_back(block_tiles), push_back(block_tiles) | Untilizes a block of tiles at once; converts tile layout to row-major in DEST registers |
| 3 | Writer | CB c_16 | DRAM (row-major sticks) | wait_front(block_tiles), pop_front(block_tiles) | Reads untilized data from CB in stick order; writes `tile_height` sticks per block, each `output_single_block_width_size` bytes wide |

### Data Flow Details

1. **Reader** reads tiles sequentially from DRAM using TensorAccessor. It reads one tile per iteration, pushing each tile into CB c_0. There is a `noc_async_read_barrier()` per tile (no pipelining of reads).

2. **Compute** waits for a full block of tiles (`num_tiles_per_block`), then performs untilize on the entire block. The untilize operation converts the tile-ordered data to row-major order within the DEST registers and packs the result into CB c_16. The output CB then contains `num_tiles_per_block` tiles worth of row-major data, organized as `tile_height` contiguous rows, each `num_tiles_per_block * tile_width` elements wide.

3. **Writer** waits for the compute to produce a full block, then writes row-major sticks to DRAM. For each block of tiles, it writes `tile_height` (32) sticks, each of width `output_single_block_width_size` bytes. The stick addresses are calculated using TensorAccessor with stick-level page IDs.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output row-major staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

**Capacity calculation**: Both CBs are sized to hold exactly one block (`num_tiles_per_block` tiles). The block size is the largest divisor of the tile-row width that fits in L1:

```
max_l1_size = (l1_size_per_core / 2) - allocator_base_addr
max_tiles_per_cb = max_l1_size / (input_tile_size + output_tile_size)
num_tiles_per_block = largest_divisor(num_tiles_per_column_row, max_tiles_per_cb)
```

Both CBs use single-buffering (capacity equals block size). This means compute must wait for the reader to fill the entire block before it can start, and the writer must wait for compute to finish the entire block before it can start. There is **no overlap** between reader, compute, and writer within a block.

## Pipeline Pattern Summary

- **Single-buffered** for both input and output CBs
- No reader/compute overlap: compute waits for `num_tiles_per_block` tiles
- No compute/writer overlap: writer waits for `num_tiles_per_block` tiles
- The reader itself is tile-by-tile with barrier per tile (no read pipelining)
- This is a simple sequential pipeline: read block -> compute block -> write block

## Index Calculations

### Reader Index Calculation (Tile Pages)

The reader uses a flat sequential page ID starting from `start_page_id`:
```
for page_id in [start_page_id, start_page_id + num_tiles):
    noc_addr = TensorAccessor.get_noc_addr(page_id)
```
Pages are tile-sized. TensorAccessor handles the mapping from flat page ID to the correct DRAM bank and offset within that bank (interleaved round-robin distribution).

### Writer Index Calculation (Row-Major Sticks)

The writer uses a more complex addressing scheme that maps tile-block coordinates to stick page IDs:

```cpp
// Outer loop: i over height blocks (num_blocks_across_height)
// Middle loop: j over column blocks (num_output_columns_of_blocks) -- 1 for non-sharded
// Inner: k over rows within a tile (tile_height = 32)
for i in [0, num_blocks_across_height):
    for j in [0, num_output_columns_of_blocks):
        for k in [0, tile_height):
            num_complete_rows = (i * tile_height + k) * num_output_columns_of_blocks
            stick_id = num_complete_rows + j
            base_dst_noc_addr[k] = TensorAccessor.get_noc_addr(stick_id)
        // Then write width blocks for these tile_height rows
        for k in [0, num_blocks_per_output_column_row):
            write_tiles_in_current_block()
```

The output TensorAccessor uses `output_stick_size` as the page size (one full row-major row). The stick_id formula accounts for sharding columns (`num_output_columns_of_blocks`), but for non-sharded output this simplifies to:
```
stick_id = i * tile_height + k  (row index)
```

### Write Pattern Within a Block

Within `write_tiles_in_current_block()`:
```cpp
cb_wait_front(cb_out, num_tiles_per_block);
l1_read_addr = get_read_ptr(cb_out);
for l in [0, tile_height):
    noc_async_write(l1_read_addr, base_dst_noc_addr[l], output_single_block_width_size);
    l1_read_addr += output_single_block_width_size;
    base_dst_noc_addr[l] += output_single_block_width_size;
noc_async_write_barrier();
cb_pop_front(cb_out, num_tiles_per_block);
```

The L1 read address advances by `output_single_block_width_size` for each row within the tile, reading `tile_height` consecutive row-segments from the CB. The destination DRAM addresses also advance by `output_single_block_width_size` for consecutive width blocks (multiple blocks within the same tile row).

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Sequential tile reads
- **Granularity**: One tile per NoC transaction
- **Access type**: `noc_async_read` with barrier per tile
- **Address generation**: TensorAccessor maps flat page IDs to interleaved DRAM bank addresses
- **Ordering**: Tiles read in storage order (row-major tile ordering: left-to-right, top-to-bottom)

### Write Pattern (Writer Kernel)

- **Pattern**: Strided row-major stick writes
- **Granularity**: One partial row (block-width) per NoC transaction
- **Access type**: `noc_async_write` with barrier per block (not per row)
- **Stride**: Within a block, `tile_height` consecutive writes at stride `output_single_block_width_size` in L1, mapping to different DRAM stick addresses
- **Key insight**: Each block write produces `tile_height` small writes (one per row in the tile), each `output_single_block_width_size` bytes. If the full width fits in one block, each write covers the entire output row width.

The writer accesses DRAM in a "tall-thin rectangle" pattern per block: it writes `tile_height` different rows, each getting a partial-width segment. Then for the next width-block, it writes the same `tile_height` rows at offset positions.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles in the tensor |
| **Load balancing** | N/A (single core) |

The single-core variant runs everything on core (0,0). This is the simplest case and is selected when the tensor is small or when multi-core overhead is not justified.

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input circular buffer ID (c_0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Serialized TensorAccessor configuration for src buffer (bank layout, shapes, coords) |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_tiles | uint32_t | Total number of tiles to read |
| 2 | start_page_id | uint32_t | Starting tile page ID (always 0 for single-core) |

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output circular buffer ID (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output row in bytes (`padded_width * element_size`) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_blocks_across_height | uint32_t | Number of tile rows in tensor height |
| 4 | num_output_columns_of_blocks | uint32_t | Number of column groups (1 for non-sharded, >1 for width/block sharded) |
| 5 | num_blocks_per_output_column_row | uint32_t | Number of width-blocks per column (width subdivision for L1 fitting) |
| 6 | num_tiles_per_output_block | uint32_t | Number of tiles per block (horizontal tile count per block) |
| 7 | output_single_block_width_size | uint32_t | Byte width of one block of output (`num_tiles_per_block * tile_width * element_size`) |
| 8+ | TensorAccessorArgs | uint32_t[] | Serialized TensorAccessor configuration for dst buffer |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (`num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height`) |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (`num_tiles_per_block`) |
| 2 | src_cb_id | uint32_t | Input CB ID (c_0) |
| 3 | out_cb_id | uint32_t | Output CB ID (c_16) |

### Compute Runtime Arguments

None. All compute parameters are compile-time.

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (tiled) | CB c_0 | Sequential tile read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **Key Logic**: Simple sequential tile reader. Creates TensorAccessor from compile-time args at offset 1 (offset 0 is CB ID). Loops from `start_page_id` to `start_page_id + num_tiles`, reading one tile per iteration. Each iteration: `cb_reserve_back(1)` -> `noc_async_read` -> `noc_async_read_barrier()` -> `cb_push_back(1)`. The barrier per tile means no read pipelining.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| pack_untilize (or untilize) | TRISC (UNPACK + MATH + PACK) | N/A | CB c_0 | CB c_16 | Tile-to-row-major conversion |

- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`
- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`
- **Key Logic**: Both kernels call `compute_kernel_hw_startup(src_cb, out_cb)` for hardware initialization, then invoke `compute_kernel_lib::untilize<block_tile_cnt, src_cb, out_cb>(block_cnt)`. The unified helper library automatically dispatches to the appropriate implementation:
  - **Pack untilize** (fast path): Used when `block_width_tiles <= DEST_limit`. The PACK thread directly rearranges tile data to row-major. This is hardware-accelerated.
  - **Block-based pack untilize**: Used when width exceeds DEST limit but data is integer type. Splits wide rows into DEST-sized sub-blocks.
  - **Standard untilize** (slow path): Used for wide non-integer types or when `use_pack_untilize` is false. Uses UNPACK + MATH threads.

**Path selection at host** (program factory lines 151-160):
- Slow path forced when: `use_pack_untilize == false`, dtype is `UINT16`, or (dtype is `FLOAT32` AND `num_tiles_per_block >= 8`)
- Fast path used otherwise (most common for BFLOAT16)

**DST_ACCUM_MODE define**: Set for INT32, UINT32, FLOAT32 data types. This halves the DEST register capacity (from 8 to 4 tiles in half-sync mode), limiting how many tiles can be processed at once.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_stick_layout_split_rows_single_core | RISCV_1 | NOC1 | CB c_16 | DRAM (row-major) | Strided stick writes |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`
- **Key Logic**: The writer implements the most important part for understanding row-major output:
  1. Creates TensorAccessor at compile-time arg offset 8 using `output_stick_size` as page size
  2. Maintains a `base_dst_noc_addr[tile_height]` array for pre-computing row addresses
  3. **Outer loop** over height blocks (`num_blocks_across_height`)
  4. **Middle loop** over column blocks (`num_output_columns_of_blocks`)
  5. For each (height, column) pair, computes DRAM addresses for all `tile_height` rows
  6. **Inner loop** over width blocks (`num_blocks_per_output_column_row`), calling `write_tiles_in_current_block()`
  7. Each block write: `cb_wait_front` -> write `tile_height` sticks of `output_single_block_width_size` bytes each -> `noc_async_write_barrier` -> `cb_pop_front`

**Stick ID calculation** (line 49-51):
```
stick_id = (i * tile_height + k) * num_output_columns_of_blocks + j
```
Where `i` = height block index, `k` = row within tile, `j` = column block index.

**Address progression**: After writing one block, `base_dst_noc_addr[k] += output_single_block_width_size` advances the DRAM write position horizontally within the same row.

## Implementation Notes

### Compute Kernel Selection Logic

The program factory selects between two compute kernels at compile time:

1. **`pack_untilize.cpp`** (fast): Default for BFLOAT16 and most types. Uses `DST_ACCUM_MODE` define for 32-bit types (INT32, UINT32, FLOAT32), which limits `max_bct` (maximum block column tiles) to 4 instead of 8.

2. **`untilize.cpp`** (slow): Fallback when pack_untilize is not supported:
   - UINT16 dtype (not supported by pack untilize hardware)
   - FLOAT32 with wide blocks (>= MAX_PACK_UNTILIZE_WIDTH = 8 tiles)
   - When `use_pack_untilize` attribute is false

### FP32 Destination Accumulation

When `fp32_dest_acc_en` is true, `UnpackToDestMode::UnpackToDestFp32` is set for the input CB. This uses 32-bit precision in DEST registers during untilize, important for FLOAT32 data to preserve precision. The slow path resets this to `Default` mode.

### Block Size Determination

The block size balances L1 usage with processing efficiency (lines 66-83):
```
max_l1_size = (l1_per_core / 2) - allocator_base
max_tiles_per_cb = max_l1_size / (input_tile_size + output_tile_size)
num_tiles_per_block = largest_divisor_of(num_tiles_per_column_row) <= max_tiles_per_cb
```
The division by 2 of L1 is conservative, ensuring the two CBs together do not exceed half of usable L1.

### Width/Block Sharding Support

For width-sharded or block-sharded output (`num_columns_of_blocks > 1`), the tensor width is split into multiple column groups. Each column group has its own set of width blocks. The stick IDs interleave column groups, so stick writes alternate between different shard destinations. For non-sharded output, `num_columns_of_blocks = 1` and this complexity disappears.

### Key Formulas Summary (for layer_norm_rm reuse)

```
num_tiles = physical_volume / tile_volume
num_blocks_across_height = physical_volume / padded_shape[-1] / tile_height
num_tiles_per_column_row = padded_shape[-1] / tile_width  (non-sharded)
output_stick_size = padded_shape[-1] * element_size
output_single_block_width_size = num_tiles_per_block * tile_width * element_size
num_total_sticks = physical_volume / padded_shape[-1]  (non-sharded)
```

### Override Runtime Arguments

The `override_runtime_arguments` method (lines 197-220) only updates buffer addresses (src and dst), confirming that all structural parameters are compile-time. This is important for program caching: the same compiled program can be reused with different tensor allocations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor and TensorAccessorArgs work in tt-metal kernels? How does get_noc_addr work with TensorAccessor for reading/writing pages to DRAM?"
   **Reason**: Needed to understand how the reader and writer kernels map page IDs to physical DRAM addresses.
   **Key Findings**: TensorAccessor abstracts interleaved bank distribution. `get_noc_addr(page_id)` returns a 64-bit NoC address combining bank XY coordinates with offset. Host-side `TensorAccessorArgs(*buffer).append_to(compile_args)` serializes bank layout info. Device-side `TensorAccessorArgs<offset>()` reconstructs from compile-time args. The page size (tile_bytes for input, stick_size for output) determines the granularity.

2. **Query**: "What is pack_untilize vs regular untilize in tt-metal compute kernels? How does pack_untilize_block work?"
   **Reason**: Needed to understand the two compute paths and what they do to tile data.
   **Key Findings**: Pack untilize is hardware-accelerated via the PACK thread - it converts tile-format data to row-major in DEST registers. Regular untilize uses UNPACK+MATH threads and is slower. Pack_untilize_block processes tiles through UNPACK->MATH->DST->PACK pipeline, where the PACK thread outputs row-major data. Block-based variant splits wide rows into DEST-sized sub-blocks for integer types.

3. **Query**: "In tt-metal untilize operation, how is the output stick size calculated for row-major output?"
   **Reason**: Critical for understanding page size calculations needed for the writer kernel and layer_norm_rm output stage.
   **Key Findings**: Output stick size = tensor_width * element_size (divided by sharding columns if applicable). This is the page size for the output TensorAccessor. Each stick corresponds to one complete row of the output tensor.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the physical layout of tiled vs row-major data.
   **Key Information**: Tiles are 32x32 with 16x16 faces stored contiguously (face0->face1->face2->face3). Row-major layout has one row per page. Interleaved memory distributes pages round-robin across DRAM banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API for address generation.
   **Key Information**: TensorAccessor handles page-to-bank mapping. `get_noc_addr(page_id)` returns NoC address for any page. Host-side TensorAccessorArgs serializes buffer layout into compile-time args. Supports both flat page IDs and ND coordinates.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize dispatch logic in compute kernels.
   **Key Information**: Three dispatch paths based on width vs DEST limit and data format: (1) single-pass pack_untilize for narrow widths, (2) block-based pack_untilize for wide integer types, (3) standard untilize for wide non-integer types or WaitUpfront mode. DEST limit is 8 tiles (half-sync, bfloat16) or 4 tiles (half-sync, fp32/int32).

4. **Source**: `tt_metal/include/compute_kernel_api/compute_kernel_hw_startup.h`
   **Reason**: Understanding hardware initialization required before untilize.
   **Key Information**: `compute_kernel_hw_startup(icb0, ocb)` must be called once at kernel start. It configures UNPACK, MATH, and PACK hardware via MMIO writes. Must be called before any operation-specific init functions. Cannot be called mid-kernel.

5. **Source**: `ttnn/api/ttnn/common/constants.hpp`
   **Reason**: Understanding MAX_PACK_UNTILIZE_WIDTH constant.
   **Key Information**: `MAX_PACK_UNTILIZE_WIDTH = 8` -- pack untilize does not support width > 8 tiles for FLOAT32 dtype.
