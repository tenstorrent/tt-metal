# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts tensor data from **row-major layout** to **tiled layout** (32x32 tiles). The input is a row-major interleaved tensor in DRAM; the output is a tiled interleaved tensor. This analysis focuses on the interleaved multi-core program factory variant.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Focus**: Reader kernel pattern (how RM sticks are read from DRAM), input CB sizing and page format, stick-to-tile batching, work distribution and core assignment strategy.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles, TILE_HEIGHT rows x full tensor width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = total tile rows across all batches/height |
| **Loop structure** | Outer: block loop (groups of 32 sticks), Inner: width blocks within a row |

A **block** represents one horizontal strip of TILE_HEIGHT (32) rows spanning the full tensor width. Each block produces `ntiles_per_block` output tiles. The total number of blocks equals the number of tile-rows across all batch/height dimensions.

---

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (flattened to 2D: outer dims x inner dim) |
| **Dimension convention** | Last dim = width (stick length), outer dims collapsed |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) |
| **Data type** | BFLOAT16 or FLOAT32 |

- **Page**: One row-major stick = one row of the flattened 2D tensor.
- **Page size in bytes**: `padded_shape[-1] * element_size()` (the `block_size_nbytes` variable).
- **Stick count**: `physical_volume / padded_shape[-1]` = number of rows.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (configurable via `output_mem_config`) |
| **Data type** | Same as input (or configurable via `output_dtype`) |

### Layout Transformation

Row-major sticks are read into CB c_0, where the compute kernel's tilize LLK reorders the data from sequential row-major order into the 32x32 tile format with 16x16 faces (face0, face1, face2, face3 in row-major face order within each tile).

---

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (RM sticks via TensorAccessor) | CB c_0 | `cb_reserve_back(c_0, ntiles_per_block)`, `cb_push_back(c_0, ntiles_per_block)` |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front(c_0, ntiles_per_block)`, `cb_pop_front(c_0, ntiles_per_block)`, `cb_reserve_back(c_16, ntiles_per_block)`, `cb_push_back(c_16, ntiles_per_block)` |
| 3 | Writer | CB c_16 | DRAM (tiles via TensorAccessor) | `cb_wait_front(c_16, 1)`, `cb_pop_front(c_16, 1)` |

### Detailed Reader Data Flow (Primary Focus)

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`) operates as follows:

1. **Outer loop**: Iterates over blocks. Each block is `TILE_HEIGHT` (32) consecutive sticks. Loop count = `num_sticks / tile_height`.
2. **NOC address resolution**: For each block, resolves all 32 stick addresses upfront into `base_src_noc_addr[32]` using `get_noc_addr(stick_id, s)` via TensorAccessor. This pre-resolves which DRAM bank each stick lives in.
3. **Inner loop**: For each width block within a row (controlled by `num_full_blocks_in_row`, which is always 1 in this factory), calls the `read_tiles` lambda.
4. **`read_tiles` lambda**:
   - `cb_reserve_back(c_0, ntiles_per_block)` -- reserves space for all tiles in one tile-row.
   - Gets `l1_write_addr` from CB write pointer.
   - Iterates over all 32 sticks: issues `noc_async_read(src_noc_addr, l1_write_addr, width_size)` for each stick.
   - Each stick is read as a contiguous `width_size` byte transfer (= `block_size_nbytes` = full row width in bytes).
   - `l1_write_addr` advances by `width_size` after each stick read, packing 32 sticks contiguously in L1.
   - `noc_async_read_barrier()` -- waits for all 32 reads to complete.
   - `cb_push_back(c_0, ntiles_per_block)` -- signals that `ntiles_per_block` tiles worth of row-major data are ready.

**Key insight for layer_norm_rm**: The reader batches exactly 32 sticks (one tile-height) per CB push. The CB page size is `input_single_tile_size` (tile-sized), but the reader writes raw row-major bytes contiguously. The tilize compute kernel then reinterprets this data during the unpack phase. The capacity is `ntiles_per_block` tiles, meaning one full tile-row fits in the CB at a time.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-------------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (RM sticks written as tile-sized pages) | `ntiles_per_block` | `ntiles_per_block` | Single | Reader | Compute | Block |
| c_16 | cb_out0 | Output staging (tilized tiles) | `ntiles_per_block` | `ntiles_per_block` | Single | Compute | Writer | Block |

### Input CB (c_0) Sizing Details

- **Page size**: `input_single_tile_size` = `tt::tile_size(input_cb_data_format)` -- this is the size of one 32x32 tile in the given data format (e.g., 2048 bytes for BFLOAT16).
- **Num pages**: `ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH`.
- **Total capacity**: `ntiles_per_block * input_single_tile_size` bytes.
- **What is actually written**: 32 consecutive RM sticks, each `block_size_nbytes` wide. The total bytes written = `32 * padded_shape[-1] * element_size` = `32 * block_size_nbytes`. This equals `ntiles_per_block * tile_size` because `ntiles_per_block * TILE_WIDTH * element_size * TILE_HEIGHT = padded_shape[-1] * element_size * 32`.
- **Data format in CB**: Row-major bytes, but the CB is configured with tile data format. The tilize unpack LLK handles the reinterpretation.

### Input CB Sizing for Layer Norm RM Reference

For a new operation that reads RM sticks and tilizes them, the input CB should be sized to hold one full tile-row of sticks:
- `num_pages = Wt` (tiles along width = `padded_shape[-1] / TILE_WIDTH`)
- `page_size = tile_size(data_format)` (size of one tile)
- This gives capacity for 32 sticks x full_width, which is exactly what the tilize compute kernel expects per block.

---

## Pipeline Pattern Summary

Both CBs (c_0 and c_16) are **single-buffered**: capacity equals block size (`ntiles_per_block`). This means the reader and compute cannot overlap on the same block -- the reader must finish pushing before compute can start, and compute must finish popping before the reader can start the next block.

---

## Memory Access Patterns

### Read Pattern (Primary Focus)

- **Pattern**: Strided DRAM reads. Each of the 32 sticks in a block may reside in different DRAM banks (interleaved round-robin by stick/page index).
- **Granularity**: One full stick per `noc_async_read` call. Each read is `block_size_nbytes` = `padded_shape[-1] * element_size` bytes.
- **Address resolution**: `get_noc_addr(stick_id, s)` via TensorAccessor maps logical stick index to physical DRAM bank + offset. The stick_id increments sequentially (`stick_id++`), but physical addresses jump between banks.
- **L1 write pattern**: Sequential. Sticks are written contiguously in L1 starting from CB write pointer, advancing by `width_size` per stick.
- **Barrier**: `noc_async_read_barrier()` after all 32 sticks, ensuring the entire tile-row is in L1 before signaling the compute kernel.
- **Pre-resolved addresses**: The reader pre-computes all 32 NOC addresses before issuing any reads, storing them in `base_src_noc_addr[32]`. This avoids interleaving address computation with DMA.

### Write Pattern (De-emphasized)

The writer reads one tile at a time from CB c_16 and writes it to DRAM via TensorAccessor, incrementing the tile page ID sequentially.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of available cores) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` (full compute grid) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (last core may have `nblocks_per_core_cliff`) |
| **Load balancing** | Nearly equal, with optional cliff core for remainder |

### Work Splitting Details

The function `ttnn::split_blocks_for_tilize(available_grid, nblocks)` performs:

1. **Compute nblocks_per_core** = `ceil(nblocks / grid_area)` where `grid_area` = number of available cores.
2. **Compute ncores** = `ceil(nblocks / nblocks_per_core)` -- actual cores used (may be less than grid_area).
3. **Cliff handling**: `nblocks_per_core_cliff = nblocks % nblocks_per_core`. If non-zero, the last core gets fewer blocks.
4. **Core enumeration**: Cores are enumerated linearly from the `available_grid` CoreRangeSet using `corerange_to_cores()`.

### Per-Core Runtime Args Derivation

For each full core `i`:
- `row_start_id = i * nblocks_per_core * TILE_HEIGHT` -- first stick index for this core.
- `num_sticks = nblocks_per_core * TILE_HEIGHT` -- total sticks to read.
- `tile_start_id = i * nblocks_per_core * ntiles_per_block` -- first output tile index.

For the cliff core (last core, if present):
- Same pattern but with `nblocks_per_core_cliff` instead of `nblocks_per_core`.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one RM stick in bytes (`block_size_nbytes` = `padded_shape[-1] * element_size`) |
| 1+ | TensorAccessorArgs | varies | Compile-time accessor args for src buffer (bank layout, shapes, coordinates) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) this core processes |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block`) |

Note: Cliff cores get a separate kernel instance with `nblocks_per_core_cliff` as arg 0.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks to read (`nblocks_per_core * TILE_HEIGHT`) |
| 2 | block_size_nbytes | uint32_t | Width of each stick in bytes (same as CT arg 0, also passed at runtime) |
| 3 | num_tiles_per_block | uint32_t | Tiles per tile-row (`ntiles_per_block`) |
| 4 | block_width_size | uint32_t | Same as block_size_nbytes (width of full block in bytes) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 (full width processed as single block) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial width blocks in interleaved variant) |
| 7 | leftover_width_in_row | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | First stick index for this core (`row_start_id`) |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block via TensorAccessor, batch into CB |
| Compute | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | Tilize RM data to tile format via `compute_kernel_lib::tilize` |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write tiles one-at-a-time via TensorAccessor |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Uses `TensorAccessorArgs<1>()` (compile-time args starting at index 1) to create a TensorAccessor with page_size = `stick_size`.
  - Pre-resolves 32 NOC addresses per block into a local array before issuing any reads.
  - Each stick is read as a single contiguous DMA transfer of `width_size` bytes.
  - The `read_tiles` lambda handles the actual DMA: reserve CB space, read 32 sticks, barrier, push.
  - `num_full_blocks_in_row` is always 1 in this factory (the entire width is one block), so the inner loop executes once per tile-row.
  - The reader does NOT perform any data reordering -- it writes sticks sequentially in L1. The tilize hardware unpack handles the RM-to-tile conversion.

### Compute Kernel

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize hardware with input/output CBs.
  - Calls `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(ntiles_per_block, nblocks)`.
  - **WaitBlock mode**: For each block, the compute kernel calls `cb_wait_front(c_0, ntiles_per_block)` to wait for the reader to push one complete tile-row, then processes it, then pops.
  - **Symmetric CB pages**: Both input and output CBs have tile-sized pages, so `total_input_pages` is omitted (symmetric mode). The tilize unpack LLK reads 32 RM rows from the input CB and produces `ntiles_per_block` tiles.
  - The `fp32_dest_acc_en` flag is set when input dtype is FLOAT32, enabling FP32 accumulation in the destination register.

### Writer Kernel (De-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Generic tile writer: waits for one tile, reads from CB, writes to DRAM via `noc_async_write_page`, pops.

---

## Implementation Notes

### Stick-to-Tile Batching (Key for Layer Norm RM)

The fundamental pattern is: **read 32 RM sticks (one tile-height) at full tensor width, push as `ntiles_per_block` tile-pages**. The CB is configured with tile-sized pages but the reader writes raw row-major data. The tilize unpack LLK handles the actual data reordering during the compute phase.

This means for a layer_norm_rm operation:
1. **Reader**: Read `TILE_HEIGHT` (32) sticks per CB push, each stick being `padded_shape[-1] * element_size` bytes.
2. **Input CB sizing**: `ntiles_per_block` pages of `tile_size` each -- holds exactly one tile-row of RM data.
3. **After tilize**: Data is in tile format in CB c_16 (or equivalent), ready for compute operations (e.g., reduce for mean/variance).

### TensorAccessor Pattern for RM Reads

The TensorAccessor is created with `page_size = stick_size` (one RM row). On the host side:
```cpp
std::vector<uint32_t> reader_ct_args = {block_size_nbytes};  // CT arg 0 = stick size
TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);  // CT args 1+ = accessor
```
On the device side:
```cpp
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto src_tensor_args = TensorAccessorArgs<1>();     // starts after CT arg 0
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```
The `get_noc_addr(stick_id, s)` call maps a logical stick index to the physical DRAM bank address, accounting for interleaved round-robin distribution.

### Width Always Processed as One Block

In the interleaved variant, `num_full_blocks_in_row` is hardcoded to 1 and `num_leftover_tiles` is 0. This means the entire tensor width is read as a single contiguous chunk per stick. This simplifies the reader but requires the full row width to fit in the CB.

### Override Runtime Arguments

The `override_runtime_arguments` method only updates `src_addr` (index 0) and `dst_addr` (index 0) when the same program is reused with different tensor buffers. All other runtime args remain unchanged.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor and TensorAccessorArgs work in tt-metal kernels? How does get_noc_addr work with TensorAccessor for reading interleaved row-major data from DRAM?"
   **Reason**: Needed to understand how stick indices map to physical DRAM addresses in the reader kernel.
   **Key Findings**: TensorAccessor maps logical page indices to physical bank addresses. Created on host with `TensorAccessorArgs(buffer)`, passed as compile-time args, then reconstructed on device. `get_noc_addr(page_id, accessor)` returns a 64-bit NOC address encoding the bank and offset.

2. **Query**: "How does the tilize operation work in tt-metal? What is the relationship between row-major sticks and tiles in the tilize process?"
   **Reason**: Needed to understand how RM data in the input CB gets converted to tile format.
   **Key Findings**: Tilize converts sequential RM rows into 32x32 tiles with 16x16 face substructure. The LLK unpacker handles the reordering during `tilize_block`. Each tile consists of 4 faces: face0 (rows 0-15, cols 0-15), face1 (rows 0-15, cols 16-31), face2 (rows 16-31, cols 0-15), face3 (rows 16-31, cols 16-31).

3. **Query**: "What does the create_cb utility function do in ttnn/operations/cb_utils.hpp?"
   **Reason**: Needed to understand CB creation pattern used in the program factory.
   **Key Findings**: DeepWiki did not have this file. Read the source directly: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a `CircularBufferConfig` with `num_pages * page_size` total capacity, sets page size for the CB, and calls `CreateCircularBuffer`.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM vs tiled page representation and interleaved memory layout.
   **Key Information**: In RM layout, each row is one page. In interleaved layout, pages are distributed round-robin across DRAM banks. Tiles are 32x32 with 16x16 faces.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding host-side and device-side TensorAccessor setup.
   **Key Information**: Host creates `TensorAccessorArgs(buffer)` and appends to compile-time args. Device reconstructs with `TensorAccessorArgs<offset>()` and creates accessor with `TensorAccessor(args, addr, page_size)`.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory.
   **Key Information**: Wraps `CircularBufferConfig` and `CreateCircularBuffer`. Total buffer size = `num_pages * page_size`. Can bind multiple CB indices to the same physical buffer.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize helper behavior, especially CB synchronization.
   **Key Information**: In `WaitBlock` mode (used here), each block iteration does `cb_wait_front(input, block_width_tiles)` then `cb_pop_front(input, block_width_tiles)`. In symmetric mode (no `total_input_pages`), input and output pages are both tile-sized. The helper automatically selects `fast_tilize` when conditions are met (32x32 tiles, Float32/Float16_b, half-sync dest).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` function and core distribution.
   **Key Information**: Computes `nblocks_per_core = ceil(nblocks / grid_area)`, then `ncores = ceil(nblocks / nblocks_per_core)`. Cliff core gets `nblocks % nblocks_per_core` blocks. Returns `BlockSplit` with core ranges and block counts.
