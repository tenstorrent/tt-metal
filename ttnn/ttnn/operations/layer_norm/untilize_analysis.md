# Untilize (Single Core) Implementation Analysis

## Overview

The untilize operation converts tile-layout data (32x32 tiles) back to row-major (RM) stick layout. This is the single-core variant that runs on core (0,0). The operation reads tiles from DRAM, unpacks them into row-major format via the compute kernel, and the writer kernel writes the resulting RM sticks back to DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_single_core_program_factory.cpp`

**Focus of this analysis**: Output stage -- how untilized sticks are written to DRAM, output CB sizing, and the `compute_kernel_lib::untilize` helper signature/usage. This analysis serves as an output_stage reference for building a layer_norm operation.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (a horizontal strip of tiles) |
| **Unit size** | `num_tiles_per_block` tiles (a divisor of one tile-row width) |
| **Total units** | `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| **Loop structure** | For each tile-row height block, for each column group, for each block-width chunk: read block, untilize block, write `tile_height` sticks |

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, ..., H, W] (arbitrary rank) | Same logical shape |
| **Tensor layout** | TILE_LAYOUT | ROW_MAJOR |
| **Memory layout** | INTERLEAVED (or sharded for output) | INTERLEAVED (or WIDTH_SHARDED / BLOCK_SHARDED) |
| **Buffer type** | DRAM | DRAM (or L1 if sharded) |
| **Data type** | bfloat16 / float32 / int32 / uint32 / uint16 | Same as input (or converted) |

### Layout Transformations

The core transformation is tile-to-row-major. Each 32x32 tile is unpacked into 32 individual row-major sticks. The compute kernel performs this transformation in L1 via the pack_untilize hardware instruction, which rearranges tile data into contiguous row-major strips.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (tiled pages) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` (1 tile at a time) |
| 2 | Compute | CB c_0 (tiled) | CB c_16 (row-major) | `cb_wait_front(c_0, block_width)`, `pack_untilize_block`, `cb_pop_front(c_0)`, `cb_reserve_back(c_16)`, `cb_push_back(c_16, block_width)` |
| 3 | Writer | CB c_16 (row-major sticks) | DRAM (RM pages) | `cb_wait_front(c_16, block_width)`, `noc_async_write` per stick row, `cb_pop_front(c_16, block_width)` |

### Key Insight: Stick Extraction from Tiles

After the compute kernel untilizes a block of `num_tiles_per_block` tiles, the output CB contains data laid out as `tile_height` (32) rows, each of width `num_tiles_per_block * TILE_WIDTH * element_size` bytes. The writer must extract each of these 32 rows and write them to the correct stick position in DRAM. This is the central pattern of the output stage.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output RM staging | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

### Output CB Sizing (Key for Layer Norm Reference)

The output CB capacity equals `num_tiles_per_block` tiles, which is the number of tiles spanning one block of the row width. The sizing logic (lines 64-83 of the program factory) works as follows:

1. **Calculate max L1 budget**: `max_l1_size = (l1_size_per_core / 2) - base_allocator_addr`. This is half the available L1 minus reserved space.
2. **Max tiles per CB**: `max_tiles_per_cb = max_l1_size / (input_tile_size + output_tile_size)`. Since there are two CBs (input + output), total budget is split.
3. **Block width**: Start with `num_tiles_per_column_row` (full tile-row width). If it exceeds `max_tiles_per_cb`, find the largest divisor of the row width that fits.
4. **Both CBs get the same capacity**: `num_tiles_per_block` tiles each.

The output CB page size is `output_single_tile_size` (tile-sized pages), but the writer reads data as contiguous RM strips of `output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * element_size` bytes per stick-row.

## Pipeline Pattern Summary

Both CBs are single-buffered (capacity equals block size). The reader, compute, and writer process one block at a time in a lockstep pipeline. There is no double-buffering overlap between stages.

## Index Calculations

### Writer Stick ID Calculation (Critical for Output Stage)

The writer uses `TensorAccessor` to map stick IDs to NoC addresses. The stick ID computation (writer kernel lines 48-51):

```cpp
uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
uint32_t stick_id = num_complete_rows_already_processed + j;
base_dst_noc_addr[k] = s.get_noc_addr(stick_id);
```

Where:
- `i` = tile-row index (0..num_blocks_across_height-1)
- `k` = row within tile (0..tile_height-1)
- `j` = column-of-blocks index (0..num_output_columns_of_blocks-1, typically 1 for interleaved)

For the common interleaved case (`num_output_columns_of_blocks = 1`), `stick_id = i * tile_height + k`, which is simply the absolute row index. The `TensorAccessor` then maps this to the correct DRAM bank and address.

### TensorAccessor Pattern

**Host side** (program factory lines 130):
```cpp
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```
This appends all accessor parameters (rank, bank count, tensor shape, shard shape, bank coordinates) as compile-time args starting at index 8.

**Device side** (writer kernel lines 23-24):
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();  // compile-time args start at index 8
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```
The template parameter `8` is the compile-time arg offset where accessor args begin. The accessor is constructed with the base address and page size (stick size for RM output).

## Memory Access Patterns

### Read Pattern (Brief -- De-emphasized)
Reader reads tiles sequentially from DRAM, one tile at a time via `noc_async_read`.

### Write Pattern (Primary Focus)

The writer outputs row-major sticks to DRAM using a **strided row-extraction** pattern:

1. **Wait for a block** of untilized data in the output CB (`cb_wait_front(c_16, num_tiles_per_block)`).
2. **For each of `tile_height` (32) rows**: Write one contiguous strip of `output_single_block_width_size` bytes to the correct DRAM address via `noc_async_write`.
3. **Advance L1 read pointer** by `output_single_block_width_size` after each row (sequential L1 reads).
4. **Advance DRAM write addresses** by `output_single_block_width_size` after each row within the same block column (for multi-block-per-row cases).
5. **Barrier and pop**: `noc_async_write_barrier()` then `cb_pop_front`.

The L1 access is purely sequential (read pointer advances linearly through the CB). The DRAM writes target different stick addresses (different banks via TensorAccessor), so they are scattered across banks but each individual write is contiguous.

**Key pattern**: The `base_dst_noc_addr[tile_height]` array is precomputed for all 32 rows before writing the blocks in that tile-row group. After each block write, the base addresses are advanced by `output_single_block_width_size` so subsequent blocks in the same row append to the right.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core |
| **Grid dimensions** | 1x1 |
| **Total cores** | 1 |
| **Work per core** | All tiles |
| **Load balancing** | N/A (single core) |

## Arguments

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out0 | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one RM stick in bytes (`padded_width * element_size / num_columns_of_blocks`) |
| 2 | tile_height | uint32_t | Height of a tile (32 for standard tiles) |
| 3 | num_blocks_across_height | uint32_t | Number of tile-rows in height dimension |
| 4 | num_output_columns_of_blocks | uint32_t | Column groups (1 for interleaved, >1 for width/block sharded) |
| 5 | num_blocks_per_output_column_row | uint32_t | How many block-width chunks fit in one column row |
| 6 | num_tiles_per_output_block | uint32_t | Tiles per block (block width in tiles) |
| 7 | output_single_block_width_size | uint32_t | Bytes per stick-row within one block: `num_tiles_per_block * TILE_WIDTH * element_size` |
| 8+ | TensorAccessorArgs | varied | Bank mapping, shape, shard info for output buffer |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt (num_blocks) | uint32_t | Total blocks to process: `num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height` |
| 1 | per_core_block_tile_cnt (num_tiles_per_block) | uint32_t | Tiles per block (block width) |
| 2 | src_cb_id | uint32_t | Input CB index (c_0) |
| 3 | out_cb_id | uint32_t | Output CB index (c_16) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_start_id | RISCV_0 | NOC0 | DRAM (tiles) | CB c_0 | Read tiles one by one |
| pack_untilize (or untilize) | Compute | N/A | CB c_0 | CB c_16 | Untilize tile block to RM |
| writer_unary_stick_layout_split_rows_single_core | RISCV_1 | NOC1 | CB c_16 | DRAM (RM sticks) | Write RM sticks row by row |

### Writer Kernel: `writer_unary_stick_layout_split_rows_single_core.cpp`

**File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`

**Key Logic**:
- Uses a lambda `write_tiles_in_current_block()` that waits for the output CB, then loops over `tile_height` rows writing one strip per row.
- Outer loops: `num_blocks_across_height` (tile rows) x `num_output_columns_of_blocks` (sharding columns) x `num_blocks_per_output_column_row` (block chunks).
- Before each group of block writes, precomputes `base_dst_noc_addr[tile_height]` for all 32 rows using `TensorAccessor::get_noc_addr(stick_id)`.
- Each `noc_async_write` writes exactly `output_single_block_width_size` bytes -- one horizontal strip of one stick row within one block.
- Uses `noc_async_write_barrier()` after writing all 32 rows of one block before popping the CB.

### Compute Kernel: `compute_kernel_lib::untilize` Helper

**File (header)**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
**File (impl)**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`

**Signature**:
```cpp
template <
    uint32_t block_width_tiles,    // tiles per block (compile-time)
    uint32_t input_cb,             // input CB index
    uint32_t output_cb,            // output CB index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
void untilize(uint32_t num_blocks);  // runtime: number of blocks
```

**Usage in untilize operation**:
```cpp
compute_kernel_hw_startup(src_cb_id, out_cb_id);
compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt);
```

**Key behavior for layer_norm reference**:
- `block_width_tiles` must be a compile-time constant (template parameter).
- Two paths depending on whether `block_width_tiles > DEST_AUTO_LIMIT`:
  - **Single-pass** (fits in DEST): `cb_wait_front(input, block_width)` -> `cb_reserve_back(output, block_width)` -> `pack_untilize_block` -> `cb_pop_front(input)` -> `cb_push_back(output)`.
  - **Block-based** (wide rows): Splits into sub-blocks, processes each sub-block separately.
- **WaitMode::WaitUpfront**: Waits for ALL input tiles at once before processing. Useful when a prior compute stage has already produced all tiles into the CB (e.g., GroupNorm/LayerNorm pattern where compute writes everything, then untilize reads it all).
- **WaitMode::NoWait**: Caller manages CB synchronization externally (data already in CB).
- **InitUninitMode**: Controls whether `pack_untilize_init`/`pack_untilize_uninit` are called. For back-to-back untilize calls, use `InitOnly`/`Neither`/`UninitOnly` to avoid redundant re-initialization.
- Prerequisite: Must call `compute_kernel_hw_startup(input_cb, output_cb)` first.

## Implementation Notes

### Output CB Design for Layer Norm

For a layer_norm operation that wants to untilize its output:
1. The output CB (c_16) needs capacity of `num_tiles_per_block` tiles (the width of one block).
2. The writer kernel writes `tile_height` sticks per block, each of size `num_tiles_per_block * TILE_WIDTH * element_size`.
3. The stick_id for interleaved output is simply the absolute row index (0, 1, 2, ...).
4. `TensorAccessorArgs` must be appended to writer compile-time args on the host side, and reconstructed with `TensorAccessorArgs<offset>()` on the device side.

### Compute Kernel Selection

Two compute kernels exist:
- `pack_untilize.cpp`: Uses hardware `pack_untilize_block` instruction (fast path). Selected when `use_pack_untilize` is true and data type is not UINT16, and not (FLOAT32 with wide blocks).
- `untilize.cpp`: Software untilize (slow path). Fallback for unsupported types.

Both use the unified `compute_kernel_lib::untilize<>()` helper, which internally selects the appropriate hardware path based on block width vs DEST limit.

### DST_ACCUM_MODE Define

When the input type is INT32, UINT32, or FLOAT32, the program factory sets `DST_ACCUM_MODE=1` as a compute kernel define. In the pack_untilize kernel, this reduces `max_bct` from 8 to 4 (halving the max block column tiles due to 32-bit accumulator mode using more DEST register space).

### Sharding Support

The `num_output_columns_of_blocks` parameter handles WIDTH_SHARDED and BLOCK_SHARDED outputs. For interleaved output (the common case and likely layer_norm case), this is 1, simplifying the writer loop and stick ID calculation.

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup (`TensorAccessorArgs::append_to`) and device-side usage (`TensorAccessorArgs<offset>()`, `TensorAccessor::get_noc_addr`).
   **Key Information**: TensorAccessor maps logical page/stick IDs to physical NoC addresses across distributed DRAM banks. Host appends args to compile-time vector; device reconstructs with template offset parameter.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the `compute_kernel_lib::untilize` helper API that layer_norm would reuse.
   **Key Information**: Template-parameterized helper with WaitMode (WaitBlock/WaitUpfront/NoWait), InitUninitMode (for back-to-back calls), and automatic block splitting for wide rows exceeding DEST limit. Prerequisite is `compute_kernel_hw_startup(input_cb, output_cb)`.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: Convenience wrapper around `CircularBufferConfig` and `CreateCircularBuffer`. Takes CB index, program, core spec, page size, num pages, and data format. Returns tuple of (cb_index, CBHandle).

4. **Source**: `.claude/CLAUDE.md` (METALIUM architecture)
   **Reason**: Confirming Tensix core architecture (reader/compute/writer kernels, CB-based inter-kernel communication).
   **Key Information**: Reader on RISCV_0/NOC0, Writer on RISCV_1/NOC1, Compute on RISCV_2. CBs are the synchronization mechanism between kernels.
