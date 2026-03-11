# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) tensor into tiled (TILE_LAYOUT) format. It reads contiguous rows ("sticks") from DRAM, packs groups of 32 sticks into tiles in the compute kernel, and writes tiled output back to DRAM. This analysis focuses on the **interleaved multi-core** program factory variant.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Kernels used**:
- **Reader**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Compute**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (uses `compute_kernel_lib::tilize` helper from `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`)
- **Writer**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (generic tile writer, de-emphasized)

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (= one tile row: 32 sticks covering the full padded width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / 32`) |
| **Total units** | `nblocks = physical_volume / TILE_HW / ntiles_per_block` (equivalently, total tile rows) |
| **Loop structure** | Outer: blocks (tile rows), Inner: full-width stick reads per block |

A **block** represents one horizontal tile row: 32 consecutive sticks spanning the full padded tensor width. The number of tiles in one block equals `padded_shape[-1] / TILE_WIDTH`. The total number of blocks equals the total number of tile rows across all batches and height dimensions.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [..., H, W] (any rank, flattened to 2D for pages) |
| **Dimension convention** | Last dim = W (stick width) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |

**Page definition for RM interleaved**: Each page is one stick (one row) of width `padded_shape[-1]`. The page size in bytes is `padded_shape[-1] * element_size()`. Sticks are distributed round-robin across DRAM banks by the interleaved allocator. Stick 0 goes to bank 0, stick 1 to bank 1, etc., wrapping around.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input (or specified via `output_dtype`) |

### Layout Transformation

The tilize operation performs the core transformation: ROW_MAJOR_LAYOUT to TILE_LAYOUT. 32 consecutive sticks of width W are packed into `W/32` tiles of 32x32 elements each. No resharding or explicit untilize occurs.

## Data Flow Pattern

### Step-by-step flow (emphasis on reader/input stage)

1. **Reader resolves stick addresses**: For each block of 32 sticks, the reader computes the NOC addresses of all 32 sticks using `get_noc_addr(stick_id, s)` where `s` is a TensorAccessor. The addresses are cached in a local array `base_src_noc_addr[32]`.

2. **Reader reads full-width sticks into CB c_0**: The reader calls `cb_reserve_back(cb_id_in0, ntiles_per_block)` to reserve space for one full tile row. Then it reads all 32 sticks at `block_width_size` bytes each into the CB in a tight loop, issuing `noc_async_read()` for each stick. After barrier, it calls `cb_push_back(cb_id_in0, ntiles_per_block)`.

3. **Compute tilizes the block**: The compute kernel waits on `ntiles_per_block` pages in CB c_0, performs the tilize transform (packing 32 rows into tile format), and pushes `ntiles_per_block` tiles to CB c_16.

4. **Writer writes tiles to DRAM**: (de-emphasized) The writer reads tiles one at a time from CB c_16 and writes them to DRAM using tile-level TensorAccessor addressing.

### Key insight: Reader CB page size = tile size, not stick size

Although the reader reads raw sticks, CB c_0 is configured with **tile-sized pages** (`input_single_tile_size = tt::tile_size(input_cb_data_format)`). The reader fills the CB with raw row-major data that happens to occupy exactly `ntiles_per_block` tile-sized pages per block of 32 sticks. This works because:
- One block = 32 sticks x `padded_shape[-1]` elements
- One tile = 32 rows x 32 cols = 1024 elements
- `ntiles_per_block` tiles x 1024 elements = 32 x `padded_shape[-1]` elements

The data in CB c_0 is **row-major layout** (not yet tiled). The compute kernel's tilize_block/fast_tilize_block operation reinterprets this row-major data and produces tiled output in CB c_16.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (RM sticks) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Reader | Compute | Block |
| c_16 | cb_out0 | Output staging (tiled) | ntiles_per_block tiles | ntiles_per_block tiles | Single | Compute | Writer | Block |

Both CBs are created with `create_cb(cb_id, program, all_cores, single_tile_size, ntiles_per_block, data_format)`, meaning:
- **Total capacity** = `ntiles_per_block * single_tile_size` bytes
- **Page size** = `single_tile_size` (one tile)
- **Number of pages** = `ntiles_per_block`

Since capacity equals block size for both CBs, they are **single-buffered**. The reader must complete writing all sticks for a block before the compute kernel can begin processing, and the compute kernel must finish before the reader can begin the next block.

## Pipeline Pattern Summary

Both CB c_0 and CB c_16 are single-buffered (capacity = 1 block). This means no reader/compute overlap or compute/writer overlap within a single core. Each block flows through the reader->compute->writer pipeline sequentially.

## Index Calculations

### Stick-to-NOC-address mapping (Reader)

The reader uses TensorAccessor for address resolution:

```
stick_id = start_stick_id  // set per-core via runtime args
for each block of 32 sticks:
    for j in 0..31:
        base_src_noc_addr[j] = get_noc_addr(stick_id, s)  // TensorAccessor resolves bank + offset
        stick_id++
    // Then read width_size bytes from each of the 32 addresses
```

`get_noc_addr(page_id, accessor)` resolves the page_id (= stick_id for RM interleaved) to a 64-bit NOC address encoding the DRAM bank coordinates (x,y) and the offset within that bank. For interleaved layout, `page_id % num_banks` determines the bank, and `page_id / num_banks * page_size` determines the offset within the bank (plus the buffer base address).

### Host-side stick ID assignment

Each core starts reading at `row_start_id`:
- Full cores: `row_start_id += TILE_HEIGHT * nblocks_per_core` (advancing by 32 sticks per block)
- Cliff core: starts where the last full core left off

## Memory Access Patterns

### Read Pattern (Reader kernel)

The reader reads sticks in a **grouped sequential** pattern:
1. For each block: resolve 32 consecutive stick addresses
2. Read all 32 sticks in order, each read covering `block_width_size` bytes (full padded row width)
3. Sticks within a block are consecutive in logical order but distributed across different DRAM banks due to interleaving

**Key characteristic**: Each `noc_async_read()` call reads one full stick (`padded_shape[-1] * element_size` bytes). For a tensor with W=1024 and bfloat16 dtype, each read is 2048 bytes. All 32 reads for a block are issued before `noc_async_read_barrier()` is called, allowing the NoC to pipeline the transfers.

The `base_src_noc_addr[k] += width_size` line in the `read_tiles` lambda advances each stick's read position. However, in this interleaved factory, `num_full_blocks_in_row` is always 1 (set in runtime args), so the inner loop over `j < num_full_blocks_in_row` executes only once. The full stick width is read in a single call per stick.

### Write Pattern (Writer kernel - de-emphasized)

The writer reads one tile at a time from CB c_16 and writes it to DRAM using `noc_async_write_page()`. Sequential tile IDs starting from `tile_start_id`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (cliff core gets `nblocks_per_core_cliff`) |
| **Load balancing** | Near-equal with optional cliff core |

### How `split_blocks_for_tilize` works

From `work_split_tilize.hpp`:
1. `nblocks_per_core = ceil(nblocks / grid_area)` -- even distribution target
2. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for last core
4. If `nblocks_per_core_cliff > 0`, the last core is a "cliff" core with fewer blocks
5. All other cores get exactly `nblocks_per_core` blocks

The cores are enumerated in row-major order from the available grid. The program factory uses `corerange_to_cores(available_grid)` to linearize the 2D grid into a 1D list.

### Per-core work assignment

For core `i` (0-indexed):
- **Sticks processed**: `nblocks_per_core * TILE_HEIGHT` (= nblocks_per_core * 32 sticks)
- **Tiles produced**: `ntiles_per_block * nblocks_per_core`
- **Starting stick**: `row_start_id = i * TILE_HEIGHT * nblocks_per_core`
- **Starting output tile**: `tile_start_id = i * ntiles_per_block * nblocks_per_core`

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Full stick width in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank mapping info for the source RM buffer (appended by `TensorAccessorArgs(*src0_buffer).append_to()`) |

The TensorAccessor compile-time args encode the interleaved buffer's bank topology: rank, number of banks, tensor shape in pages, and bank coordinates. On the device, `TensorAccessorArgs<1>()` reconstructs these starting at compile-time arg index 1 (index 0 is `stick_size`).

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks this core processes (`nblocks_per_core` or `nblocks_per_core_cliff`) |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block = padded_shape[-1] / 32`) |

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank mapping info for destination tiled buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks for this core: `nblocks * TILE_HEIGHT` |
| 2 | (unused/block_size_nbytes) | uint32_t | Stick size in bytes (redundant with CT arg) |
| 3 | num_tiles_per_block | uint32_t | Tiles per block: `ntiles_per_block` |
| 4 | block_width_size | uint32_t | Stick size in bytes: `block_size_nbytes` |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for interleaved (full width read in one pass) |
| 6 | num_leftover_tiles | uint32_t | Always 0 for interleaved |
| 7 | leftover_width | uint32_t | Always 0 for interleaved |
| 8 | start_stick_id | uint32_t | First stick ID for this core: `row_start_id` |

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles for this core |
| 2 | start_id | uint32_t | First tile ID for this core |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block, push as ntiles_per_block pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  1. Constructs TensorAccessor from compile-time args at index 1: `constexpr auto src_tensor_args = TensorAccessorArgs<1>(); const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);`
  2. Outer loop iterates `num_sticks / tile_height` times (= number of blocks for this core)
  3. Per block: resolves 32 stick NOC addresses into `base_src_noc_addr[32]` array
  4. Calls `read_tiles(num_tiles_per_block, block_width_size)` which:
     - Reserves `num_tiles` pages in CB c_0
     - Issues 32 `noc_async_read()` calls (one per stick, each reading `width_size` bytes)
     - Waits for barrier
     - Pushes `num_tiles` pages to CB c_0
  5. Stick reads fill the CB with raw row-major data; the CB page_size is tile_size but content is RM

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block (row-major to tile format) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  1. Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize hardware
  2. Delegates to `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_tile_cnt, per_core_block_cnt)`
  3. The tilize helper (from `tilize_helpers.inl`) loops `num_blocks` times:
     - `cb_wait_front(input_cb, block_width_tiles)` -- waits for reader to push one block
     - `cb_reserve_back(output_cb, block_width_tiles)` -- reserves output space
     - Calls `tilize_block()` or `fast_tilize_block()` (auto-selected at compile time based on tile size, format, sync mode)
     - `cb_push_back(output_cb, block_width_tiles)` -- signals writer
     - `cb_pop_front(input_cb, block_width_tiles)` -- frees input CB space
  4. The tilize operation reads 32 rows of RM data and packs them into proper tile format

### Writer Kernel (de-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Single-tile-at-a-time writer using TensorAccessor. Generic, not tilize-specific.

## Implementation Notes

1. **Single-buffered CBs**: Both input and output CBs hold exactly one block. This means the pipeline does not overlap reader/compute or compute/writer. For a layer_norm_rm operation that needs to overlap tilize with subsequent compute, larger CB capacities (double-buffering) would be beneficial.

2. **Full-width reads**: The interleaved variant always reads the entire stick width in one `noc_async_read()` call per stick. `num_full_blocks_in_row` is hardcoded to 1, and leftover tiles/width are 0. This contrasts with sharded or block variants that may split wide rows.

3. **32-stick batching**: The reader always reads exactly 32 sticks (TILE_HEIGHT) per block, which is the natural grouping for tilize. This is a fundamental constraint: tilize requires groups of 32 rows to form complete tiles.

4. **TensorAccessor for RM stick addressing**: The reader uses `get_noc_addr(stick_id, tensor_accessor)` where `stick_id` is the logical stick index. For interleaved layout, the TensorAccessor handles the page-to-bank mapping (round-robin across DRAM banks). The stick_size is provided as a compile-time constant.

5. **Fast tilize auto-selection**: The compute kernel automatically uses `fast_tilize_block()` when conditions are met: 32x32 tiles, Float32 or Float16_b format, and half-sync dest mode. This is determined at compile time via `can_use_fast_tilize<>()`.

6. **Cliff core handling**: The compute kernel is compiled with different compile-time args for the cliff core (`nblocks_per_core_cliff` vs `nblocks_per_core`). Two separate `CreateKernel` calls handle this, one for full cores and one for the cliff core range.

7. **`fp32_llk_acc`**: When the input dtype is FLOAT32, the compute config enables `fp32_dest_acc_en`, which uses 32-bit accumulation in the FPU destination register.

8. **Relevance for layer_norm_rm**: This tilize reader pattern is directly applicable as the input stage for a layer_norm_rm operation. The key reusable patterns are:
   - TensorAccessor for RM stick reads from interleaved DRAM
   - 32-stick batching into tile-sized CB pages
   - `split_blocks_for_tilize` for 1D work distribution
   - The `compute_kernel_lib::tilize` helper for the tilize compute phase
   - CB c_0 sized to hold one full tile row (`ntiles_per_block` pages of tile_size)

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessorArgs work on the host side when passed as compile-time arguments vs runtime arguments?"
   **Reason**: Needed to understand the host-to-device argument passing pattern used in the reader kernel setup.
   **Key Findings**: `TensorAccessorArgs(*buffer).append_to(ct_args)` serializes the buffer's bank topology (rank, num_banks, tensor shape, shard shape, bank coords) into a uint32_t vector as compile-time arguments. On the device, `TensorAccessorArgs<INDEX>()` reconstructs from compile-time args starting at offset INDEX. The template parameter is the starting index into the compile-time argument list.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor resolves page_id to NOC addresses
   **Key Information**: `get_noc_addr(page_id)` maps logical page to physical bank+offset. For interleaved tensors, pages are distributed round-robin across banks. The accessor supports both compile-time and runtime argument configurations.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page definition
   **Key Information**: For row-major tensors, each row of the innermost 2D slice is one page. A 64x64 RM tensor has 64 pages. Pages are stored consecutively within a buffer and distributed across banks in interleaved mode.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution function
   **Key Information**: Uses `compute_ncores()` to calculate even distribution of blocks across cores, with optional cliff core for remainder. Returns `BlockSplit` struct with core ranges and per-core block counts.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute tilize helper API and its CB synchronization pattern
   **Key Information**: `compute_kernel_lib::tilize<>()` handles the complete tilize loop with configurable init/uninit modes, wait modes, and register reconfiguration. In WaitBlock mode (used here), it calls `cb_wait_front` per block. Automatically selects fast_tilize when hardware supports it.

5. **Source**: `METALIUM_GUIDE.md` (CB section)
   **Reason**: Confirming circular buffer synchronization semantics
   **Key Information**: `cb_reserve_back`/`cb_push_back` are producer-side (reader for input CB, compute for output CB). `cb_wait_front`/`cb_pop_front` are consumer-side (compute for input CB, writer for output CB). These form the producer-consumer synchronization between kernels.

6. **Source**: `tt_metal/api/tt-metalium/constants.hpp`
   **Reason**: Confirming tile dimension constants
   **Key Information**: TILE_HEIGHT = 32, TILE_WIDTH = 32, TILE_HW = 1024.
