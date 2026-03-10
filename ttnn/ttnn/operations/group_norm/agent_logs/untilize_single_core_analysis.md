# Untilize Single Core Implementation Analysis

## Overview

The untilize single-core operation converts a tensor from **TILE_LAYOUT** (32x32 tiles with face-based internal storage) to **ROW_MAJOR_LAYOUT** (contiguous row sticks). It runs all three kernels (reader, compute, writer) on a single Tensix core at `(0, 0)`. The compute kernel transforms tiled data into row-major sticks in an output CB, and the writer kernel extracts those sticks and writes them to DRAM using TensorAccessor-based addressing.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp`

**Analysis focus**: Output stage -- writer kernel pattern, output CB sizing, and how row-major sticks are extracted from untilized tile data and written to DRAM. This analysis serves as an output-stage reference for a new group_norm operation producing RM output with shape `(N, 1, H*W, C)`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles (width-direction) |
| **Unit size** | `num_tiles_per_block` tiles (a horizontal slice of one tile row) |
| **Total units** | `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| **Loop structure** | Height blocks -> column blocks -> width blocks per column row |

One "block" is a horizontal group of tiles that fits within CB capacity. The compute kernel processes one block at a time (untilize it), and the writer kernel consumes the untilized row-major sticks from that block.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (e.g., [N, C, H, W]) |
| **Dimension convention** | Last dimension is width (contiguous in memory) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles, face-based internal storage) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Last dimension is width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT (one stick = one row of the 2D representation) |
| **Memory layout** | INTERLEAVED (default), WIDTH_SHARDED, or BLOCK_SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformation

The core transformation is TILE_LAYOUT -> ROW_MAJOR_LAYOUT. A 32x32 tile with face-based internal storage (face0, face1, face2, face3 each 16x16, stored in row-major order within the tile page) is converted to 32 contiguous row-major sticks. The `pack_untilize` hardware instruction or SFPU fallback performs this rearrangement in the compute kernel.

## Data Flow Pattern

### High-Level Pipeline

```
DRAM (tiles) --[Reader]--> CB_in (c_0) --[Compute: untilize]--> CB_out (c_16) --[Writer]--> DRAM (RM sticks)
```

### Detailed Step-by-Step Flow

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | DRAM (tile pages) | CB c_0 | `reserve_back(1)`, `push_back(1)` | Reads tiles one at a time from DRAM via TensorAccessor. Tiles are read sequentially by page_id. |
| 2 | Compute | CB c_0 | CB c_16 | `wait_front(block_w)`, `pop_front(block_w)`, `reserve_back(block_w)`, `push_back(block_w)` | Consumes `num_tiles_per_block` tiles from c_0, untilizes them into row-major sticks, pushes `num_tiles_per_block` tile-equivalents of RM data to c_16. |
| 3 | Writer | CB c_16 | DRAM (RM sticks) | `wait_front(block_w)`, `pop_front(block_w)` | Reads untilized row-major data from c_16, writes it to DRAM as row-major sticks using TensorAccessor. Each block produces `tile_height` (32) sticks of width `output_single_block_width_size` bytes. |

### Key Insight: Output CB Contains Row-Major Sticks

After the compute kernel's `pack_untilize_block` (or `untilize_block`) processes a block of `num_tiles_per_block` tiles, the output CB c_16 contains:

- **32 contiguous row-major sticks** (one per row of the tile height)
- Each stick is `num_tiles_per_block * TILE_WIDTH * element_size` bytes wide (`output_single_block_width_size`)
- Layout in CB: `row0_of_all_block_tiles, row1_of_all_block_tiles, ..., row31_of_all_block_tiles`

This is the fundamental data format that the writer kernel must handle.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output RM stick staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

### Output CB Sizing (Focus Area)

The output CB capacity is set to `num_tiles_per_block` tiles, where:

```cpp
uint32_t output_cb_num_tiles = num_tiles_per_block;  // line 97
```

The `num_tiles_per_block` value is determined by:
1. Start with `num_tiles_per_column_row` = full tile-width of tensor (or shard width for sharded outputs)
2. If this exceeds `max_tiles_per_cb` (L1 budget), find the largest divisor of `num_tiles_per_column_row` that fits
3. `max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size)` -- accounts for both CBs sharing L1

The output CB page size is `output_single_tile_size` (tile_size for the output data format), and the total CB capacity in bytes is `num_tiles_per_block * output_single_tile_size`.

**Important**: Even though the output CB holds row-major data after untilize, it is configured with tile-sized pages (`output_single_tile_size`). This works because `pack_untilize` writes row-major data that occupies the same total bytes as the tile-format data (32 rows x N*32 elements per row = N*1024 elements = N tiles worth of data).

### CB Sizing Formula for Group Norm Reference

For a new operation producing RM output with width `C`:
- `Ct = C / TILE_WIDTH` (tiles per row)
- Output CB needs capacity of at least `Ct` tiles (or a divisor of `Ct` if L1 is tight)
- Output CB page_size = `tile_size(output_data_format)`
- Output CB total bytes = `Ct * tile_size(output_data_format)`

## Pipeline Pattern Summary

Both CBs use **single-buffering** (capacity equals block size). This means compute and writer cannot overlap within a block -- the writer must fully drain c_16 before the compute kernel can produce the next block. Similarly, the reader and compute cannot overlap within a block for c_0.

The pipeline is: Read block -> Untilize block -> Write block (sequential per block).

## Index Calculations

### Writer Kernel Stick ID Computation (Critical for Output Stage)

The writer kernel computes stick IDs to map untilized rows to DRAM addresses. The stick_id calculation at line 49-50 of the writer kernel is:

```cpp
uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
uint32_t stick_id = num_complete_rows_already_processed + j;
```

Where:
- `i` = current height block index (0 to `num_blocks_across_height - 1`)
- `k` = current row within the tile height (0 to `tile_height - 1`, i.e., 0 to 31)
- `j` = current column-of-blocks index (0 to `num_output_columns_of_blocks - 1`)

For the **non-sharded case** (`num_output_columns_of_blocks = 1`), this simplifies to:
```
stick_id = i * tile_height + k
```
This is simply the sequential row index: row 0, row 1, ..., row (total_height - 1).

The TensorAccessor then maps this stick_id to a physical DRAM address:
```cpp
base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
```

### TensorAccessor Setup

Host side (program factory, line 131):
```cpp
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

Device side (writer kernel, lines 23-24):
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // starts at compile-time arg index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The TensorAccessor page size is `output_stick_size` = full row width in bytes:
```cpp
uint32_t output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks;
```
For a simple non-sharded tensor, this equals `padded_shape[-1] * element_size`.

### For Group Norm Reference

For an RM output with shape `(N, 1, H*W, C)`:
- `num_sticks = N * H * W` (total number of output rows)
- `stick_size = C * element_size` bytes (one full row of the C dimension)
- `stick_id` iterates from 0 to `num_sticks - 1` sequentially
- TensorAccessor page_size = `stick_size`

## Memory Access Patterns

### Read Pattern (De-emphasized)

Tiles are read sequentially from DRAM, one tile at a time, with a full `noc_async_read_barrier()` after each tile. Simple sequential access with TensorAccessor-based page addressing.

### Write Pattern (Focus Area)

The writer writes untilized data as row-major sticks to DRAM. The access pattern is structured as a 3-level nested loop:

```
for each height_block (i):           // tile-height-sized vertical strips
  for each column_of_blocks (j):     // sharding columns (1 for non-sharded)
    compute base_dst_noc_addr[0..31] // pre-compute DRAM addresses for all 32 rows
    for each width_block (k):        // horizontal blocks within the row
      wait for block in CB
      for each row (l = 0..tile_height-1):
        noc_async_write(l1_read_addr, base_dst_noc_addr[l], output_single_block_width_size)
        l1_read_addr += output_single_block_width_size   // advance through CB
        base_dst_noc_addr[l] += output_single_block_width_size  // advance within DRAM row
      noc_async_write_barrier()
      pop block from CB
```

**Key observations for the write pattern**:

1. **Row-interleaved writes**: For each block, the writer issues 32 separate `noc_async_write` calls -- one per row of the tile height. Each write is `output_single_block_width_size` bytes (a partial-width stick).

2. **Strided DRAM access**: Within a block, consecutive writes go to different DRAM rows (stride = `output_stick_size`). This is because each write targets a different row of the output tensor.

3. **Sequential within-row advancement**: When there are multiple width blocks (`num_blocks_per_column_row > 1`), the base addresses advance by `output_single_block_width_size` after each block, building up the full row progressively.

4. **Pre-computed addresses**: The base addresses for all 32 rows are computed once per height-block/column combination, then updated incrementally. This uses `TensorAccessor::get_noc_addr(stick_id)` for the initial address computation.

5. **Barrier per block**: `noc_async_write_barrier()` is called after all 32 rows of a block are issued, ensuring the CB can be freed.

### Write Pattern Diagram (Non-Sharded, Single Width Block)

For a tensor with `Ct` tiles wide fitting in one block:
```
CB c_16 layout after untilize:
  [row0: Ct*32 elements] [row1: Ct*32 elements] ... [row31: Ct*32 elements]

Writer issues 32 writes:
  Write row0 -> DRAM stick_id=i*32+0
  Write row1 -> DRAM stick_id=i*32+1
  ...
  Write row31 -> DRAM stick_id=i*32+31
```

Each write is `Ct * TILE_WIDTH * element_size` bytes = one complete output row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core |
| **Grid dimensions** | 1x1 |
| **Total cores** | 1 |
| **Work per core** | All tiles in the tensor |
| **Load balancing** | N/A (single core) |

The single-core variant runs everything on core `(0, 0)`. There is a separate multi-core program factory for parallelized untilize.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessorArgs | (varies) | Packed accessor parameters for source buffer (appended starting at index 1) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (height * columns * width_blocks) |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (`num_tiles_per_block`) |
| 2 | src_cb_id | uint32_t | Input CB index (c_0) |
| 3 | out_cb_id | uint32_t | Output CB index (c_16) |

#### Writer Kernel (Focus Area)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Full output row width in bytes = `padded_width * element_size` |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_blocks_across_height | uint32_t | Number of tile-height blocks vertically |
| 4 | num_output_columns_of_blocks | uint32_t | Number of sharding columns (1 for non-sharded) |
| 5 | num_blocks_per_output_column_row | uint32_t | Number of width blocks per column row |
| 6 | num_tiles_per_output_block | uint32_t | Tiles per block (for CB wait/pop) |
| 7 | output_single_block_width_size | uint32_t | Bytes per block-width per row = `num_tiles_per_block * TILE_WIDTH * element_size` |
| 8+ | TensorAccessorArgs | (varies) | Packed accessor parameters for destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_tiles | uint32_t | Total number of tiles to read |
| 2 | start_page_id | uint32_t | Starting tile page ID (always 0 for single-core) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (tiles) | CB c_0 | Read tiles via TensorAccessor |
| compute | RISCV_2 (unpack/math/pack) | N/A | CB c_0 | CB c_16 | pack_untilize or SFPU untilize |
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (RM sticks) | Write RM sticks via TensorAccessor + noc_async_write |

### Writer Kernel (Focus Area)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`

**Key Logic**:

1. **TensorAccessor construction**: Uses `TensorAccessorArgs<8>()` (compile-time args starting at index 8) with `output_stick_size` as the page size. This tells the accessor that each "page" in DRAM is one full RM row.

2. **`write_tiles_in_current_block` lambda**: This is the core write routine:
   - Waits for `num_tiles_per_output_block` tiles in the output CB
   - Gets the L1 read pointer from the CB
   - Loops over `tile_height` (32) rows, issuing one `noc_async_write` per row
   - Each write transfers `output_single_block_width_size` bytes (a partial or full row)
   - After all 32 writes, issues `noc_async_write_barrier()` and pops the block from the CB

3. **Address pre-computation**: Before processing width blocks, the writer computes `base_dst_noc_addr[k]` for all 32 rows using `s.get_noc_addr(stick_id)`. This means TensorAccessor::get_noc_addr is called 32 times per height-block/column combination, amortized across all width blocks.

4. **Incremental address update**: After writing a block, `base_dst_noc_addr[l] += output_single_block_width_size` advances the write position within each DRAM row. This handles the case where a row spans multiple blocks.

### Compute Kernel

**File** (fast path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`
**File** (slow path): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`

Both call `compute_kernel_lib::untilize<>()` from `untilize_helpers.hpp`. The fast path uses `pack_untilize_block` (hardware-accelerated), the slow path uses SFPU-based untilize. The slow path is selected when:
- `use_pack_untilize` is false, OR
- Data type is `UINT16`, OR
- Data type is `FLOAT32` and `num_tiles_per_block >= MAX_PACK_UNTILIZE_WIDTH`

The helper function handles:
- DEST register limit detection via `DEST_AUTO_LIMIT`
- Sub-blocking for wide rows that exceed DEST capacity
- CB synchronization (wait_front / pop_front / reserve_back / push_back)

### Reader Kernel (De-emphasized)

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`

Simple sequential tile reader: for each page_id, reads one tile from DRAM to CB c_0 using TensorAccessor. One tile at a time with full read barrier.

## Implementation Notes

### Critical Design Details for Group Norm Reference

1. **Output CB uses tile page size, not stick size**: The output CB is configured with `page_size = output_single_tile_size` even though it holds row-major data. This works because `pack_untilize` writes row-major data that occupies exactly the same bytes as the original tiles. The CB capacity in tiles is `num_tiles_per_block`, meaning it holds the RM equivalent of that many tiles.

2. **Writer writes partial rows when blocks are smaller than full width**: If L1 budget forces `num_tiles_per_block < Ct` (tiles per full row), the writer writes partial rows across multiple blocks. The `base_dst_noc_addr` is advanced by `output_single_block_width_size` after each block to stitch the partial rows together.

3. **TensorAccessor page_size for output is the full stick width**: Not the block width. The TensorAccessor computes DRAM addresses based on `stick_id` and `output_stick_size` (full row width). The writer then writes partial blocks within those stick addresses.

4. **The `output_single_block_width_size` is the write size per noc_async_write call**: This equals `num_tiles_per_block * TILE_WIDTH * element_size`. For a full-width block (entire row fits in one block), this equals `output_stick_size`.

5. **For the group_norm output stage**: If the output shape is `(N, 1, H*W, C)` in RM:
   - `num_sticks = N * H * W`
   - `stick_size = C * element_size` bytes
   - `Ct = C / 32` tiles per row
   - Output CB capacity = `Ct` tiles (if L1 allows), page_size = tile_size
   - Writer iterates height blocks, each producing 32 sticks
   - Each stick is written to DRAM with `noc_async_write(l1_addr, noc_addr, stick_size)`
   - TensorAccessor with page_size = stick_size handles interleaved bank addressing

6. **`compute_kernel_hw_startup(src_cb, out_cb)` is required**: Must be called at the top of any compute kernel before using `compute_kernel_lib::untilize<>()`. It sets up srcA=srcB=src_cb for the hardware.

7. **Sharding support via `num_output_columns_of_blocks`**: For WIDTH_SHARDED or BLOCK_SHARDED outputs, the tensor width is split into multiple columns of blocks. The stick_id calculation incorporates the column index. For non-sharded (INTERLEAVED), `num_output_columns_of_blocks = 1` and the logic simplifies.

### Untilize Helper Signature and Usage

The untilize helper from `untilize_helpers.hpp` has this signature:

```cpp
template <
    uint32_t block_width_tiles,    // Tiles per row in the block (compile-time)
    uint32_t input_cb,             // Input CB index
    uint32_t output_cb,            // Output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
void untilize(uint32_t num_blocks);  // num_blocks is runtime
```

For group_norm output stage, the relevant usage pattern would be:
```cpp
compute_kernel_hw_startup(cb_in, cb_out);
compute_kernel_lib::untilize<Ct, cb_in, cb_out>(num_height_blocks);
```

Or with WaitUpfront mode (noted as "GroupNorm pattern" in the helper docs):
```cpp
compute_kernel_lib::untilize<Ct, cb_in, cb_out,
    InitUninitMode::InitAndUninit,
    WaitMode::WaitUpfront>(num_rows);
```

The WaitUpfront mode waits for all input tiles at once before processing, which is useful when a preceding compute stage has already filled the CB.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize operation work in TT-Metal? Specifically, how does the compute kernel convert tiled data into row-major stick format, and how does the writer kernel write row-major sticks back to DRAM?"
   **Reason**: Needed to confirm the end-to-end data flow and the relationship between tile rows and output sticks.
   **Key Findings**: Each row of a 32x32 tile becomes one output stick. The writer processes tile_height (32) sticks per height block. pack_untilize uses DEST registers for hardware-accelerated conversion.

2. **Query**: "What is pack_untilize in TT-Metal? How does pack_untilize_block work vs regular untilize? What is the difference between the fast pack_untilize path and the slow SFPU untilize path?"
   **Reason**: Needed to understand the two compute paths and when each is selected.
   **Key Findings**: pack_untilize is ~5x faster (80 vs 390 cycles/tile). The slow path is used for UINT16 or when FLOAT32 with wide blocks. pack_untilize operates on data in DEST registers; regular untilize uses SFPU datacopy.

3. **Query**: "After pack_untilize writes to an output CB, what is the exact memory layout of that data?"
   **Reason**: Critical for understanding what the writer kernel reads from the CB.
   **Key Findings**: The CB contains tile_height (32) contiguous sticks, each of width `num_tiles_per_block * TILE_WIDTH` elements. Layout is row0_of_all_tiles, row1_of_all_tiles, ..., row31_of_all_tiles.

4. **Query**: "How does the output circular buffer (CB c_16) work in untilize operations?"
   **Reason**: Needed to confirm CB c_16 usage patterns and data layout.
   **Key Findings**: CB c_16 is the standard output CB for compute kernels. After pack_untilize, data is row-major within the CB. The CB is configured with tile page sizes even though the content is row-major.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps stick_id to DRAM addresses in the writer kernel.
   **Key Information**: TensorAccessor(args, base_addr, page_size) provides get_noc_addr(page_id) for address computation. Host-side uses TensorAccessorArgs(buffer).append_to() to inject compile-time args. Page_size for RM output is the full stick width.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major page structure and interleaved bank distribution.
   **Key Information**: In RM layout, each row of the 2D tensor representation is one page. Interleaved layout distributes pages round-robin across DRAM banks. Tile layout uses 32x32 tiles with 16x16 faces stored contiguously.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the compute_kernel_lib::untilize helper signature, template parameters, and dispatch logic.
   **Key Information**: block_width_tiles is compile-time, num_blocks is runtime. Automatically handles DEST limit splitting (sub-blocks). Supports WaitBlock (per-block sync), WaitUpfront (all-at-once), and NoWait (caller-managed). The GroupNorm pattern is explicitly called out in the documentation as a WaitUpfront use case.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST_AUTO_LIMIT used by untilize to determine sub-blocking.
   **Key Information**: DEST capacity: SyncHalf=8 tiles (16-bit), 4 tiles (32-bit). SyncFull=16 tiles (16-bit), 8 tiles (32-bit). The untilize helper automatically splits wide rows into sub-blocks that fit within DEST.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the create_cb helper used by the program factory.
   **Key Information**: `create_cb(cb_index, program, core, page_size, num_pages, data_format)` configures a CB with total size = num_pages * page_size. Returns the CB index and handle.
