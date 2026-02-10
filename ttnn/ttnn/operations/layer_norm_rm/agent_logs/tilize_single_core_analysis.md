# Tilize Single-Core Implementation Analysis

## Overview

The **tilize single-core** operation converts a tensor from **row-major layout** to **tile layout** (32x32 tiles with 16x16 faces) using a single Tensix core. The input tensor is stored in DRAM as contiguous row-major "sticks" (one stick = one row of the innermost dimension). The output tensor is stored in DRAM in tile format.

This operation is particularly relevant as a reference for the **INPUT STAGE** of a layer_norm_rm operation, where row-major data arriving from DRAM must be tilized before compute operations can process it.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles (horizontally across one tile-row) |
| **Unit size** | `num_tiles_per_block` tiles (determined by L1 capacity; ideally a full tile-row) |
| **Total units** | `num_tiles / num_tiles_per_block` = `(num_sticks / 32) * num_full_blocks_in_row` |
| **Loop structure** | Outer loop: tile-rows (`num_sticks / 32`), Inner loop: blocks across the row (`num_full_blocks_in_row`) |

A "work unit" is a horizontal block of tiles spanning `num_tiles_per_block` tiles along the width dimension, across all 32 rows of a tile-row. The reader reads 32 sticks worth of that block, the compute tilizes those sticks into tiles, and the writer drains the output tiles. This repeats across all blocks in a row, then across all tile-rows.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, ..., H, W] (arbitrary rank, flattened to 2D) |
| **Dimension convention** | Last dim = W (stick width), all others collapsed into H (num_sticks) |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (or FLOAT32 with fp32_llk_acc) |

- **stick_s** = `padded_shape[-1]` (width in elements)
- **stick_size** = `stick_s * element_size` (width in bytes)
- **num_sticks** = `physical_volume / width` (total number of rows)
- Each page in the interleaved buffer corresponds to one stick (one row).

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile boundaries) |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Configurable via `output_dtype` (typically matches input) |

- Each page in the interleaved buffer corresponds to one tile.
- Total tiles = `physical_volume / TILE_HW` where `TILE_HW = 32 * 32 = 1024`.

### Layout Transformations

The core transformation is **tilize**: row-major sticks are rearranged into 32x32 tiles with internal 16x16 face structure. This is performed by the compute kernel's hardware unpack unit (TRISC_UNPACK thread), not by explicit data copying in software. The unpack hardware reads row-major data from the input CB and writes it in tile format to destination registers, from which it is packed into the output CB.

## Data Flow Pattern

### Stage-by-Stage Flow

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | DRAM (interleaved, row-major sticks) | CB c_0 | `cb_reserve_back`, `cb_push_back` | Reads 32 sticks per tile-row, writing `block_width_size` bytes per stick into L1 in a strided pattern |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` | Hardware tilize: unpack reads row-major data, rearranges into tile format, packs to output CB |
| 3 | Writer | CB c_16 | DRAM (interleaved, tile pages) | `cb_wait_front`, `cb_pop_front` | Writes one tile at a time to DRAM |

### Detailed Reader Data Movement Pattern

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`) implements a **split-rows** reading pattern that is critical to understand for the layer_norm_rm input stage:

1. **Outer loop**: iterates over tile-rows (`num_sticks / tile_height` iterations, where `tile_height = 32`)
2. **Address pre-computation**: For each tile-row, the reader pre-computes NoC addresses for all 32 sticks using `get_noc_addr(stick_id, s)` where `s` is a TensorAccessor. This maps each stick_id to the physical DRAM bank and offset via round-robin interleaving.
3. **Inner loop**: iterates over blocks within the tile-row (`num_full_blocks_in_row` iterations)
4. **Block read (`read_tiles` lambda)**: For each block:
   - `cb_reserve_back(cb_id_in0, num_tiles)` -- reserve space for `num_tiles_per_block` tiles
   - For each of the 32 sticks: `noc_async_read(src_noc_addr, l1_write_addr, width_size)` reads `block_width_size` bytes from the stick
   - L1 write address advances by `width_size` after each stick, creating a column of 32 partial-sticks stacked contiguously in the CB
   - The base NoC addresses advance by `width_size` to point to the next block segment of each stick
   - `noc_async_read_barrier()` -- wait for all 32 reads to complete
   - `cb_push_back(cb_id_in0, num_tiles)` -- signal data is ready

**Key insight**: The reader does NOT read one full stick at a time. It reads `block_width_size` bytes from each of the 32 sticks, laying them out contiguously in L1. This creates exactly the memory layout that the hardware tilize unit expects: 32 rows of `num_tiles_per_block * TILE_WIDTH` elements, packed sequentially. The tilize hardware then reorganizes this into proper tile format with 16x16 faces.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (row-major sticks) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_out | Output staging (tilized tiles) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

**Capacity calculation**: Both CBs are sized to hold exactly one block (`num_tiles_per_block` tiles). This means:
- `cb_in0` capacity = `num_tiles_per_block * input_single_tile_size` bytes
- `cb_out` capacity = `num_tiles_per_block * output_single_tile_size` bytes

**Single-buffered**: Capacity equals block size (1x), meaning the reader must wait for the compute kernel to consume the current block before writing the next one. Similarly, the compute kernel must wait for the writer to drain the output before producing more tiles. This creates a sequential pipeline with no overlap between stages within a block.

**Note on input CB data format**: Although the input data is row-major, the CB is configured with `input_cb_data_format` derived from the tensor's data type and sized in tiles (`input_single_tile_size`). The hardware tilize unit interprets the CB contents differently: it reads the data as 32 rows of raw elements and rearranges them into tile format. The "tile size" allocation ensures sufficient space for `TILE_HEIGHT * block_width_size` bytes.

## Pipeline Pattern Summary

Both CBs are **single-buffered** (capacity = block size). This means:
- Reader fills `cb_in0` with one block, then waits for compute to consume it.
- Compute tilizes `cb_in0` into `cb_out`, then waits for writer to drain `cb_out`.
- Writer drains `cb_out` one tile at a time, then signals done.

There is **no overlap** between reader/compute or compute/writer within a block. The pipeline is strictly sequential: Read -> Compute -> Write -> repeat.

**Implication for layer_norm_rm**: If using tilize as an input stage, the single-buffering means the compute kernel for the actual normalization operation cannot start until the tilize block is fully complete. Double-buffering the tilize output CB would allow overlap.

## Index Calculations

### Reader Index Mapping

The reader uses a `TensorAccessor` to map logical stick IDs to physical DRAM addresses:

1. **stick_id** is a linear index from 0 to `num_sticks - 1`, starting at `start_stick_id` (runtime arg, typically 0 for single-core).
2. `get_noc_addr(stick_id, s)` calls into the TensorAccessor which:
   - Computes `bank_offset_index = stick_id / num_banks` (which page slot within a bank)
   - Computes `bank_index = stick_id % num_banks` (which DRAM bank)
   - Returns a 64-bit NoC address combining the bank's NoC X-Y coordinates with the physical byte offset within that bank
3. The TensorAccessor is constructed from compile-time args (appended via `TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)`) and the runtime `src_addr` (buffer base address) and `stick_size` (page size).

### Writer Index Mapping

The writer uses a simpler linear tile index. For tile `i` (from `start_id` to `start_id + num_tiles`):
- `noc_async_write_page(i, s, l1_read_addr)` uses TensorAccessor `s` to map tile `i` to the correct DRAM bank and offset.

### How Block-Width Addressing Works

Within a tile-row, the reader tracks progress along the width dimension through the `base_src_noc_addr` array:

```
Initial state:  base_src_noc_addr[k] = start of stick k in DRAM
After block j:  base_src_noc_addr[k] += j * block_width_size
```

This is a pointer-advance pattern: after reading the first block of width `block_width_size` from each stick, the addresses slide right by `block_width_size` bytes to point to the next block segment.

## Memory Access Patterns

### Read Pattern

- **Pattern type**: Strided reads across 32 sticks, sequential within each stick segment
- **Access granularity**: `block_width_size` bytes per stick per block (= `num_tiles_per_block * TILE_WIDTH * element_size`)
- **Stride**: One full `stick_size` between consecutive stick reads in DRAM (sticks are in different DRAM banks due to interleaving)
- **L1 write pattern**: Sequential within CB -- 32 segments of `block_width_size` bytes each, packed contiguously
- **Barrier**: `noc_async_read_barrier()` after all 32 reads per block (batch barrier, not per-read)

### Write Pattern

- **Pattern type**: Sequential, one tile at a time
- **Access granularity**: `output_single_tile_size` bytes per tile
- **Ordering**: Linear tile ID order (tile 0, 1, 2, ...)
- **Synchronization**: `noc_async_writes_flushed()` after each tile (not a full barrier -- allows pipelining of NoC writes)
- **Final barrier**: `noc_async_write_barrier()` at the end of all writes

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (1x1) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles (entire tensor) |
| **Load balancing** | N/A (single core) |
| **Core selection** | `sub_core_grids[0]` if provided, else `(0, 0)` |

This is a single-core implementation. All work is assigned to one core. The `sub_core_grids` parameter allows the caller to specify which physical core to use, which is useful for avoiding conflicts with other operations running on other cores.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one complete stick in bytes (`width * element_size`) |
| 1+ | TensorAccessor args | uint32_t[] | Bank mapping, shape, and distribution info for src buffer (appended by `TensorAccessorArgs(*src0_buffer).append_to(...)`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output circular buffer ID (c_16) |
| 1+ | TensorAccessor args | uint32_t[] | Bank mapping, shape, and distribution info for dst buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (`num_tiles / num_tiles_per_block`) |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles in each block (`num_tiles_per_block`) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM base address |
| 1 | num_sticks | uint32_t | Total number of sticks (rows) in the tensor |
| 2 | stick_size | uint32_t | Size of one stick in bytes (also passed as compile-time; used for validation) |
| 3 | num_tiles_per_block | uint32_t | Tiles per horizontal block |
| 4 | block_width_size | uint32_t | Block width in bytes (`num_tiles_per_block * TILE_WIDTH * element_size`) |
| 5 | num_full_blocks_in_row | uint32_t | Number of full blocks across one tile-row |
| 6 | num_leftover_tiles | uint32_t | Leftover tiles if width not divisible by block size (unused in current kernel) |
| 7 | leftover_width_in_row | uint32_t | Leftover width in bytes (unused in current kernel) |
| 8 | start_stick_id | uint32_t | Starting stick index (0 for single-core) |

**Note**: Runtime args 2, 6, and 7 are declared in the factory but not actually used in the kernel code. The kernel reads args at indices 0, 1, 3, 4, 5, and 8. This suggests the argument layout is shared with a multi-core variant that may use these fields.

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM base address |
| 1 | num_pages | uint32_t | Total number of output tiles to write |
| 2 | start_id | uint32_t | Starting tile index (0 for single-core) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (row-major sticks) | CB c_0 | Read 32 sticks x block_width per block |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Pre-computes 32 NoC addresses at the start of each tile-row
  - Uses a `read_tiles` lambda that reads `block_width_size` bytes from each of the 32 sticks into contiguous L1 locations
  - Advances all 32 base addresses by `block_width_size` after each block to slide the read window across the row
  - Uses `noc_async_read_barrier()` to ensure all 32 reads complete before signaling the CB
  - The data layout in the CB after reading is: 32 partial-rows packed contiguously (row 0 segment, row 1 segment, ..., row 31 segment)

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (UNPACK+MATH+PACK) | N/A | CB c_0 | CB c_16 | Hardware tilize |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize UNPACK/MATH/PACK threads
  - Delegates to `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` from `tilize_helpers.hpp`
  - The helper function calls `tilize_init()` once (configures UNPACK for tilize mode), then loops calling `tilize_block()` per block
  - `tilize_block()` internally: calls `llk_unpack_tilize_block()` (UNPACK thread reads row-major and rearranges to tile format), then per-tile: `llk_math_eltwise_unary_datacopy()` (MATH thread copies to dst), `llk_pack()` (PACK thread writes to output CB)
  - The actual row-to-tile rearrangement is done by the **UNPACK hardware** -- it is not a software memcpy. The UNPACK unit reads elements from the row-major layout in the input CB and places them into the correct positions within 16x16 faces in the destination registers.
  - Uses default `InitAndUninit` mode, `WaitBlock` synchronization, and `Standard` speed mode

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tile pages) | Write one tile at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**:
  - Generic writer shared across many operations
  - Reads page size from CB interface: `get_local_cb_interface(cb_id_out).fifo_page_size`
  - Per-tile loop: `cb_wait_front(1)` -> `get_read_ptr()` -> `noc_async_write_page(i, s, l1_read_addr)` -> `noc_async_writes_flushed()` -> `cb_pop_front(1)`
  - Uses `noc_async_writes_flushed()` (not barrier) between tiles, allowing write pipelining
  - Final `noc_async_write_barrier()` ensures all writes complete before kernel exits
  - Supports `#ifdef OUT_SHARDED` (not used in this single-core interleaved path) and `#ifdef BACKWARDS` (not used here)

## Implementation Notes

### Block Size Selection Strategy (use_low_perf flag)

The program factory has an important optimization for block size selection (lines 53-69):

1. **Default mode** (`use_low_perf = false`): Maximizes `num_tiles_per_block` to process as many tiles per block as possible, limited by L1 capacity. The calculation is:
   - `max_l1_size` = half of L1 minus allocator base address
   - `max_tiles` = `max_l1_size / (input_tile_size + output_tile_size)` (both CBs must fit)
   - If an entire tile-row fits (`num_tiles_in_row <= max_tiles`), use the full row as one block
   - Otherwise, find the largest divisor of `num_tiles_in_row` that fits in L1
2. **Low-perf mode** (`use_low_perf = true`): Uses `num_tiles_per_block = 1`, processing one tile at a time (minimum memory, maximum overhead)

The constraint that `num_tiles_in_row` must be divisible by `num_tiles_per_block` ensures clean block boundaries -- no partial blocks at the end of a row.

### Leftover Tiles (Unused Code Path)

The factory computes `num_leftover_tiles` and `leftover_width_in_row`, and passes them as runtime args (indices 6 and 7), but the reader kernel does not use them. The kernel loop only processes `num_full_blocks_in_row` blocks. This suggests the leftover handling may have been removed or is handled by padding the tensor to tile boundaries before calling the operation.

### fp32 Accumulation

When the input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in the compute config. This enables 32-bit accumulation in the destination registers during the datacopy operation, preventing precision loss.

### Override Runtime Arguments

The `override_runtime_arguments` method (lines 145-165) updates only `src_addr` (reader arg 0) and `dst_addr` (writer arg 0) for program caching. This allows the same compiled program to be reused with different input/output buffer allocations, as long as the tensor shape and configuration remain the same.

### Relevance to layer_norm_rm Input Stage

For a layer_norm_rm operation that needs to read row-major data from DRAM and tilize it:

1. **The reader pattern is reusable**: The split-rows reading strategy (read 32 sticks at a time, block-width at a time) is the canonical way to prepare row-major data for tilization.
2. **CB sizing matters**: Single-buffering creates a bottleneck. For layer_norm_rm, double-buffering the tilize input/output CBs would allow overlapping tilize with the normalization compute.
3. **Block width optimization**: The block size selection logic (finding the largest divisor of tile-row width that fits in L1) is directly applicable.
4. **TensorAccessor usage**: The pattern of passing TensorAccessor args as compile-time args and the buffer address as runtime args is the standard approach for DRAM access.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? Specifically, how does the compute kernel tilize_block function rearrange row-major data into tile format (32x32 tiles with 16x16 faces)?"
   **Reason**: Needed to understand the internal mechanics of the tilize_block hardware operation.
   **Key Findings**: The tilize operation is executed by the TRISC_UNPACK thread. `tilize_block` calls `llk_unpack_tilize_block` which reads row-major data and rearranges it into 32x32 tiles with 16x16 faces. The MATH thread then does a datacopy to destination registers, and the PACK thread writes to the output CB. The face structure (16x16) exists because the matrix engine natively multiplies 16x16 matrices.

2. **Query**: "How does the reader kernel read row-major sticks from DRAM for the tilize operation? What is the split rows pattern?"
   **Reason**: Needed to confirm the split-rows reading strategy for the input stage reference.
   **Key Findings**: The reader reads 32 consecutive sticks (one tile-height worth) and arranges them into blocks. `get_noc_addr` maps stick IDs to physical DRAM NoC addresses. `noc_async_read` performs non-blocking transfers. The split-rows pattern processes data in blocks of 32 rows (tile height).

3. **Query**: "What is the get_noc_addr function used with TensorAccessor in tt-metal kernels? How does it translate a stick_id (page_id) into a physical DRAM NOC address for interleaved buffers?"
   **Reason**: Needed to understand the index calculation mechanism for DRAM access.
   **Key Findings**: For interleaved buffers, `get_noc_addr` computes bank_offset_index (which slot within a bank), bank_index (which bank via round-robin), calculates the physical address within the bank using page_size and base offsets, retrieves NoC X-Y coordinates for the bank, and combines them into a 64-bit NoC address.

4. **Query**: "What does compute_kernel_hw_startup do in tt-metal?"
   **Reason**: Needed to understand the initialization requirement before tilize operations.
   **Key Findings**: `compute_kernel_hw_startup` initializes the three-thread compute engine (UNPACK, MATH, PACK). It configures CB IDs, data formats, and DST register synchronization. Must be called exactly once at kernel start before any compute operations.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps logical page IDs to physical DRAM addresses.
   **Key Information**: TensorAccessor provides `get_noc_addr(page_id)` for address calculation. Constructed from compile-time args via `TensorAccessorArgs<base_idx>()`. Supports interleaved and sharded buffers.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major vs tile layout and page definitions.
   **Key Information**: In row-major layout, each row is one page. In tile layout, each 32x32 tile is one page. Tiles contain 16x16 faces in row-major order (face0->face1->face2->face3). Interleaved memory distributes pages round-robin across banks.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel library API for tilize.
   **Key Information**: Unified `tilize<input_cb, output_cb>(block_width_tiles, num_blocks)` function with configurable init/uninit mode, wait mode, speed mode, and non-tile-aligned CB support. Standard mode uses `tilize_init` + `tilize_block` loop + `tilize_uninit`.

4. **Source**: `tt_metal/include/compute_kernel_api/tilize.h`
   **Reason**: Understanding the low-level tilize_block implementation.
   **Key Information**: `tilize_block` calls `llk_unpack_tilize_block` (UNPACK), then per-tile: `llk_math_eltwise_unary_datacopy` (MATH datacopy to DST) + `llk_pack` (PACK to output CB). Three-thread pipeline with proper synchronization via `llk_math_wait_for_dest_available` and `llk_packer_wait_for_math_done`.
