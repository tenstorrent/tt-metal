# Tilize (Single Core) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) tensor into tiled (32x32) format on a single Tensix core. It reads contiguous sticks (rows) from DRAM, groups them into blocks of tiles, and the compute kernel rearranges the data from row-major into tile order (face-based layout). This analysis focuses on the **input stage**: how RM sticks are read from DRAM, input CB sizing, stick-to-tile batching, and core assignment.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp`

**Role focus**: input_stage -- reader kernel pattern, input CB sizing, stick-to-tile batching, work distribution.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles (horizontal slice of one tile-row) |
| **Unit size** | `num_tiles_per_block` tiles (ideally = `num_tiles_in_row`, the full tile-width of one row) |
| **Total units** | `num_tiles / num_tiles_per_block` = `(num_sticks / 32) * num_full_blocks_in_row` |
| **Loop structure** | Outer loop over tile-rows (groups of 32 sticks), inner loop over blocks within each tile-row |

One "work unit" is a horizontal block of `num_tiles_per_block` tiles, covering 32 sticks tall and `num_tiles_per_block * 32` elements wide. The reader fills the input CB with the raw RM data for one such block; the compute kernel tilizes it; the writer drains the output CB.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, ..., H, W] (flattened to 2D: num_sticks x W) | Same shape |
| **Dimension convention** | Last dim = W (stick width) | Last dim = W |
| **Tensor layout** | ROW_MAJOR | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (typical) | DRAM (typical) |
| **Data type** | BFLOAT16 or FLOAT32 | Configurable (output_dtype) |

### Input Tensor -- Row-Major Pages

In row-major layout, each page is one row ("stick") of width W. The page size in bytes is `W * element_size`. Pages are distributed round-robin across DRAM banks (interleaved layout). The reader accesses pages by stick ID using the TensorAccessor, which resolves the stick ID to the correct DRAM bank and address.

### Layout Transformation

The core transformation is ROW_MAJOR to TILE_LAYOUT. A group of 32 consecutive sticks (each W elements wide) is reorganized into `W/32` tiles. Each 32x32 tile is internally stored as four 16x16 faces in row-major face order (face0, face1, face2, face3). The compute kernel performs this rearrangement using the hardware tilize instruction.

## Data Flow Pattern

### Step-by-step (Input Stage Focus)

1. **Stick address resolution**: For each group of 32 sticks, the reader resolves all 32 DRAM NoC addresses upfront via `get_noc_addr(stick_id, s)` using TensorAccessor. These are cached in a local array `base_src_noc_addr[32]`.

2. **Block read loop**: For each block within the tile-row:
   - `cb_reserve_back(cb_id_in0, num_tiles_per_block)` -- reserve space for one block of tiles in the input CB.
   - For each of the 32 sticks: issue `noc_async_read` of `block_width_size` bytes from the stick's current DRAM address into L1 at the CB write pointer. The L1 write pointer advances by `block_width_size` per stick. The DRAM source address also advances by `block_width_size` (to read the next horizontal block of the same stick on the next iteration).
   - `noc_async_read_barrier()` -- wait for all 32 reads to complete.
   - `cb_push_back(cb_id_in0, num_tiles_per_block)` -- signal that data is available for compute.

3. **Compute tilize**: The compute kernel calls `cb_wait_front` on `cb_id_in0`, performs the tilize operation (converting RM data to tile format), writes results to `cb_id_out0` (CB 16), then `cb_pop_front` / `cb_push_back`.

4. **Writer**: Reads tiles from CB 16 and writes them to DRAM one tile at a time (de-emphasized).

### Key Insight: Stick-to-Tile Batching

The reader does NOT read one stick at a time into a stick-sized page. Instead, it reads partial rows from 32 sticks simultaneously, laying them out sequentially in L1 so that the data for `num_tiles_per_block` tiles is contiguous. This means the input CB page size is tile-sized, not stick-sized. The 32 partial-row reads (one per stick in the tile height) together populate `num_tiles_per_block` tiles worth of RM data in the CB.

The layout in the CB after the reader fills one block:
```
[stick_0 partial_row | stick_1 partial_row | ... | stick_31 partial_row]
 <-- block_width_size --> each
 Total = 32 * block_width_size = num_tiles_per_block * tile_size
```

## Circular Buffer Configuration (Input Stage Focus)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (0) | cb_id_in0 | Input staging (RM data for tilize) | num_tiles_per_block tiles | num_tiles_per_block tiles | Single | Reader | Compute | Block |
| c_16 | cb_id_out | Output staging (tilized data) | num_tiles_per_block tiles | num_tiles_per_block tiles | Single | Compute | Writer | Block |

### Input CB Sizing Strategy

The input CB capacity equals exactly one block: `num_tiles_per_block * input_single_tile_size` bytes. This is single-buffered (capacity == block size). The reader fills the entire block, barriers, pushes, then the compute kernel consumes it before the reader can fill the next block.

**Block size determination** (lines 53-69 of program factory):
- **Default (use_low_perf=false)**: Tries to make `num_tiles_per_block = num_tiles_in_row` (full width of the tensor in tiles). This maximizes throughput by processing an entire tile-row in one block.
- **L1 constraint**: The block must fit in available L1. `max_tiles = max_l1_size / (input_tile_size + output_tile_size)`. If the full row does not fit, the factory finds the largest divisor of `num_tiles_in_row` that fits.
- **Low-perf mode (use_low_perf=true)**: Falls back to `num_tiles_per_block = 1`, processing one tile at a time.

**Relevance to layer_norm**: For a 2D RM input to layer_norm, a similar strategy could be used for the input reader -- reading sticks from DRAM and batching them into tile-sized blocks. The key pattern is: resolve NoC addresses for 32 sticks, then read horizontal slices from all 32 sticks to fill tile-shaped CB regions.

## Pipeline Pattern Summary

Both CBs are single-buffered (capacity = block size). There is no overlap between reader and compute for the same block -- the reader must complete and push before compute can process, and compute must pop before the reader can reserve the next block. This is a **sequential pipeline** within each block.

## Index Calculations

### TensorAccessor for Stick Address Resolution

The reader uses `TensorAccessor` to map a linear stick ID to a physical DRAM NoC address:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args starting at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

The TensorAccessor encapsulates the interleaved bank mapping: given a page (stick) index, it computes which DRAM bank holds that page and the offset within the bank. The `TensorAccessorArgs` are appended to compile-time args on the host side (line 105) and decoded on the device side starting at compile-time arg index 1 (index 0 is `stick_size`).

### Stick ID to Block Mapping

- `stick_id` starts at `start_stick_id` (runtime arg index 8, typically 0 for single-core).
- Outer loop: `i` in `[0, num_sticks/32)` -- iterates over tile-rows.
- Each tile-row consumes 32 consecutive stick IDs.
- Inner loop: `j` in `[0, num_full_blocks_in_row)` -- iterates over horizontal blocks within the tile-row.
- The `base_src_noc_addr[k]` pointer advances by `block_width_size` after each block read, moving across the stick horizontally.

## Memory Access Patterns

### Read Pattern (Input Stage)

- **Pattern**: Strided reads across DRAM banks. For each block, 32 reads are issued, one per stick. Each read is `block_width_size` bytes (`num_tiles_per_block * TILE_WIDTH * element_size`).
- **Stride**: Sticks are stored in separate DRAM pages (interleaved round-robin). Consecutive sticks may reside in different banks, enabling bank-level parallelism.
- **Sequential within a stick**: Within a single stick, consecutive blocks are read left-to-right (the source address increments by `block_width_size`).
- **Batched NoC reads**: All 32 reads within a block are issued asynchronously before the barrier, enabling hardware-level pipelining of NoC transfers.
- **Barrier granularity**: One `noc_async_read_barrier()` per block (per 32 reads).

### Write Pattern (De-emphasized)

The writer drains output CB one tile at a time, writing tiles sequentially to DRAM using TensorAccessor-based addressing.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (1x1) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles in the tensor |
| **Load balancing** | N/A (single core) |

The core is either `(0,0)` by default or specified via `sub_core_grids` parameter. For multi-core tilize, see the separate `tilize_multi_core_interleaved_program_factory.cpp`.

## Arguments

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one stick (row) in bytes: `W * element_size` |
| 1+ | TensorAccessorArgs | multiple uint32_t | Bank mapping metadata for interleaved DRAM access (rank, num_banks, shapes, bank coords) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of source buffer in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks (rows) in the tensor |
| 2 | stick_size | uint32_t | Stick size in bytes (also passed as compile-time; unused in kernel but present in args array) |
| 3 | num_tiles_per_block | uint32_t | Number of tiles in one horizontal block |
| 4 | block_width_size | uint32_t | Width of one block in bytes: `num_tiles_per_block * 32 * element_size` |
| 5 | num_full_blocks_in_row | uint32_t | Number of full blocks per tile-row |
| 6 | num_leftover_tiles | uint32_t | Leftover tiles after full blocks (unused in kernel -- always 0 when row is divisible) |
| 7 | leftover_width_in_row | uint32_t | Leftover width in bytes (unused in kernel) |
| 8 | start_stick_id | uint32_t | Starting stick index (0 for single-core, varies for multi-core) |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process: `num_tiles / num_tiles_per_block` |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block: `num_tiles_per_block` |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 partial sticks per block via async NoC reads |
| Compute | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize (RM to tile format conversion) |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write tiles one at a time |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Pre-resolves 32 NoC addresses per tile-row before issuing any reads.
  - Uses a lambda `read_tiles` that reserves CB space, issues 32 async reads (one per stick in the tile height), barriers, and pushes.
  - The `base_src_noc_addr` array acts as 32 sliding pointers that advance horizontally across blocks.
  - The name "split_rows" refers to splitting each row into multiple blocks when the full row does not fit in L1.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**: Calls `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` which uses hardware tilize instructions to rearrange RM data into face-based tile format. Uses `fp32_dest_acc_en` when input dtype is FLOAT32.

### Writer Kernel (De-emphasized)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic tile writer. Reads page size from CB interface, writes one tile at a time to DRAM using TensorAccessor.

## Implementation Notes

### Relevance to Layer Norm Input Stage

For a layer_norm operation reading a 2D RM input tensor from DRAM:

1. **Stick-based reading pattern**: The tilize reader demonstrates how to read RM sticks from interleaved DRAM. Each stick is one page, addressed by stick ID via TensorAccessor. This same pattern applies to reading RM input rows for layer_norm.

2. **Batching 32 sticks for tile formation**: If layer_norm needs its input in tile format, this reader shows how to batch 32 sticks into tile-shaped CB regions. The key is issuing 32 NoC reads (one per stick) writing to consecutive L1 addresses, then barrier and push as a block of tiles.

3. **Block sizing for L1 fit**: The factory's approach of finding the largest divisor of `num_tiles_in_row` that fits in L1 is a reusable pattern. For layer_norm, the "row" is the normalization dimension (last dim), so the block size determines how many tiles along that dimension are processed at once.

4. **TensorAccessor for interleaved access**: The host-side `TensorAccessorArgs(*buffer).append_to(compile_time_args)` and device-side `TensorAccessor(args, addr, page_size)` pattern is the standard way to access interleaved tensors. The accessor handles bank resolution given a page index.

5. **Single-buffered CB limitation**: This implementation uses single-buffered CBs, meaning no overlap between reader and compute. For layer_norm performance, consider double-buffering the input CB (capacity = 2 * block_size) to overlap reading the next block while computing the current one.

## External Knowledge Sources

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding how RM and tiled layouts map to pages and memory banks.
   **Key Information**: In RM layout, each row is one page. In tiled layout, each 32x32 tile is one page. Interleaved layout distributes pages round-robin across DRAM banks. Tiles have internal face structure (4 x 16x16 faces).

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how the reader kernel resolves stick IDs to DRAM addresses.
   **Key Information**: TensorAccessor maps logical page indices to physical bank addresses. Host-side `TensorAccessorArgs(buffer)` generates compile-time or runtime args encoding bank count, tensor shape, shard shape, and bank coordinates. Device-side `TensorAccessor(args, base_addr, page_size)` + `get_noc_addr(page_id, accessor)` resolves addresses.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding the compute-side tilize API and its synchronization modes.
   **Key Information**: The tilize helper supports symmetric mode (both CBs have tile-sized pages, which is what this single-core factory uses) and asymmetric mode (input CB has row-sized pages). It supports WaitBlock (default, per-block synchronization), WaitUpfront, and NoWait modes. For this factory, symmetric WaitBlock mode is used.
