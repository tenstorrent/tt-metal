# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The **tilize** operation converts a tensor from **row-major layout** to **tile layout** (32x32 tiles). It reads row-major "sticks" (contiguous rows) from DRAM, groups 32 sticks into a tile-height batch, and uses the compute kernel's hardware tilize unit to reorder data into tile format (four 16x16 faces). The output is written back to DRAM in tile layout.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

This analysis is focused on the **input stage** patterns: how RM sticks are read from DRAM, input CB sizing and page format, stick-to-tile batching, and work distribution. Writer details are de-emphasized.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles, 1 tile-row high) |
| **Unit size** | `ntiles_per_block` tiles = one full row of tiles (width of tensor / TILE_WIDTH) |
| **Total units** | `nblocks` = total tiles / ntiles_per_block = (padded_H / TILE_HEIGHT) where H is the total height of the 2D flattened tensor |
| **Loop structure** | Outer: iterate over blocks (each block = 32 sticks). Inner: iterate over full blocks in a row (always 1 in this factory). |

A **block** corresponds to TILE_HEIGHT (32) consecutive row-major sticks spanning the full tensor width. Each block produces `ntiles_per_block` output tiles. The reader reads 32 sticks, the compute tilizes them into `ntiles_per_block` tiles, and the writer drains those tiles.

---

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (e.g., [N, C, H, W]) |
| **Dimension convention** | Last dim is contiguous width; all outer dims flattened to height |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |

**Key property**: `padded_shape[-1]` must be divisible by TILE_WIDTH (32). The physical volume must be divisible by TILE_HW (1024).

A row-major page = one stick = one row of the flattened 2D tensor = `padded_shape[-1]` elements.

**Page size in bytes**: `padded_shape[-1] * element_size()` (the `block_size_nbytes` variable in the program factory).

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input |

### Layout Transformations

The sole transformation is **tilize**: row-major sticks are rearranged into 32x32 tiles with four 16x16 faces. No padding, resharding, or data type conversion occurs within this operation.

---

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations | Details |
|-------|--------|------------|-----------|---------------|---------|
| 1 | Reader | DRAM (interleaved sticks) | CB c_0 | `cb_reserve_back`, `cb_push_back` | Reads 32 sticks worth of data (one tile-row-width), pushes `ntiles_per_block` pages |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` | Tilizes one block (converts RM to tile format) |
| 3 | Writer | CB c_16 | DRAM (interleaved tiles) | `cb_wait_front`, `cb_pop_front` | Writes tiles one-at-a-time to interleaved DRAM |

### Reader Kernel Data Flow (Emphasized)

The reader kernel (`reader_unary_stick_layout_split_rows_interleaved.cpp`) implements a two-level loop:

1. **Outer loop**: Iterates over blocks (`num_sticks / tile_height` iterations). Each block = 32 sticks.
2. **Per block setup**: Resolves NoC addresses for all 32 sticks by calling `get_noc_addr(stick_id, s)` for each of the 32 sticks, storing results in `base_src_noc_addr[32]`.
3. **Inner loop**: For each "full block in row" (always 1 in this factory since the width is not split):
   - `cb_reserve_back(c_0, ntiles_per_block)` -- reserves space for all tiles in one row
   - Reads data from 32 sticks by issuing `noc_async_read` for each stick, advancing the L1 write address by `width_size` after each stick
   - `noc_async_read_barrier()` -- waits for all 32 reads to complete
   - `cb_push_back(c_0, ntiles_per_block)` -- signals data is ready

**Critical insight for layer_norm_rm**: The reader batches exactly 32 sticks (one tile height) per CB push. Each push writes `ntiles_per_block * tile_size` bytes into the CB. This is the natural unit for converting RM to tile format.

### Stick-to-Tile Batching Pattern

The reader packs 32 rows of width `block_width_size` bytes into the input CB in a single reservation. The CB page size is set to `input_single_tile_size` (the size of one tile in the input data format), but the number of pages is `ntiles_per_block`. So one `cb_reserve_back` / `cb_push_back` cycle fills `ntiles_per_block` tile-sized slots with raw row-major data covering 32 sticks x full_width.

The key layout inside the CB after the reader pushes is: 32 contiguous rows of `padded_shape[-1]` elements each, occupying the same byte footprint as `ntiles_per_block` tiles. The compute kernel's tilize unpack hardware reads this RM data and reorders it into tile format.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (RM sticks) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tilized tiles) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**CB c_0 details (input -- emphasized)**:
- Page size: `input_single_tile_size` = `tt::tile_size(input_cb_data_format)` (e.g., 2048 bytes for BFloat16 32x32)
- Number of pages: `ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH`
- Total capacity: `ntiles_per_block * input_single_tile_size` bytes
- This exactly fits 32 rows of width `padded_shape[-1]` because `32 * padded_shape[-1] * element_size = ntiles_per_block * tile_size` for supported data types
- Created via `create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, ntiles_per_block, input_cb_data_format)`

**CB c_16 details (output -- de-emphasized)**:
- Page size: `output_single_tile_size`
- Number of pages: `ntiles_per_block`
- Single-buffered, same capacity pattern as input

### Input CB Page Format

The input CB is configured with tile-sized pages (`input_single_tile_size`), but the reader writes raw row-major sticks into it. The 32 sticks are packed contiguously. The tilize unpack hardware in the compute kernel knows how to interpret this RM layout and convert it to tile format. This is the "symmetric" tilize mode where input and output CB page sizes match (both tile-sized).

---

## Pipeline Pattern Summary

Both CBs are **single-buffered** (capacity = block size = `ntiles_per_block` tiles). This means the reader must complete writing all 32 sticks before compute can start, and compute must complete one block before the reader can push the next block. There is no overlap between consecutive blocks.

---

## Index Calculations

### Stick ID to NoC Address Mapping

The reader uses `TensorAccessor` for address generation:

```
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time arg at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```

For each stick, `get_noc_addr(stick_id, s)` translates the logical stick ID to a 64-bit NoC address. For interleaved DRAM:
1. The `stick_id` determines which DRAM bank and offset within that bank (round-robin distribution)
2. The `InterleavedAddrGen` inside TensorAccessor computes: `bank_index = stick_id % num_banks`, `bank_offset = (stick_id / num_banks) * aligned_page_size`
3. The NoC XY coordinates for the target bank are looked up, and combined with the address to form the 64-bit NoC address

### Block-to-Stick Mapping

Each core processes `nblocks_per_core` blocks. The starting stick for core `i` is:
- `start_stick_id = i * nblocks_per_core * TILE_HEIGHT`

Within each block, sticks are numbered sequentially: `stick_id, stick_id+1, ..., stick_id+31`.

### Tile ID Tracking

For the writer (briefly): `tile_start_id = i * ntiles_per_block * nblocks_per_core`. Tiles are written sequentially starting from this ID.

---

## Memory Access Patterns

### Read Pattern (Emphasized)

**Pattern**: Strided reads across DRAM banks, 32 sticks at a time.

For each block:
1. Resolve NoC addresses for 32 consecutive stick IDs
2. Issue 32 `noc_async_read` calls, each reading `block_width_size` bytes (one full stick)
3. Reads go to consecutive L1 addresses (packed contiguously in the CB)
4. A single `noc_async_read_barrier` waits for all 32 reads

Since sticks are interleaved across DRAM banks in round-robin, consecutive stick IDs typically target different banks, which enables bank-level parallelism in the reads.

**Read granularity**: One stick = `padded_shape[-1] * element_size` bytes. For a 1024-wide BFloat16 tensor, this is 2048 bytes per stick.

**L1 write pattern**: Sequential. Each stick's data is written at offset `k * width_size` within the CB, where k is the stick index within the block (0..31).

### Write Pattern (De-emphasized)

The writer reads one tile at a time from CB c_16 and writes it to interleaved DRAM using `noc_async_write_page`. Sequential tile IDs, one `cb_wait_front` / `cb_pop_front` per tile.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core enumeration from available grid) |
| **Grid dimensions** | Up to device `compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (last core may get `nblocks_per_core_cliff`) |
| **Load balancing** | Near-equal with cliff: most cores get `ceil(nblocks/grid_area)` blocks, last core gets remainder |

### Work Splitting Details

The function `split_blocks_for_tilize(available_grid, nblocks)` computes:
1. `nblocks_per_core = ceil(nblocks / grid_area)` -- base blocks per core
2. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- blocks for the last (cliff) core
4. If `nblocks_per_core_cliff > 0`, one "cliff" core handles the remainder

Cores are enumerated linearly from the `CoreRangeSet` using `corerange_to_cores()`. The first `ncores - has_cliff` cores get `nblocks_per_core` blocks each. The last core (if cliff exists) gets `nblocks_per_core_cliff` blocks.

### Runtime Args Assignment Loop

The program factory iterates over cores and assigns:
- `row_start_id`: incremented by `TILE_HEIGHT * nblocks_per_core` per core
- `tile_start_id`: incremented by `ntiles_per_block * nblocks_per_core` per core

This creates contiguous, non-overlapping block ranges per core.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size (block_size_nbytes) | uint32_t | Size of one row-major stick in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessorArgs | multiple uint32_t | Appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` -- encodes buffer type, bank count, interleaving params |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) this core processes: `nblocks_per_core` or `nblocks_per_core_cliff` |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block: `ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH` |

#### Writer Kernel (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | multiple uint32_t | For the destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total sticks this core reads: `nblocks_per_core * TILE_HEIGHT` |
| 2 | block_size_nbytes | uint32_t | Byte width of one stick (same as compile-time arg 0, passed redundantly) |
| 3 | ntiles_per_block | uint32_t | Tiles per block row |
| 4 | block_width_size | uint32_t | Byte width for read (same as block_size_nbytes in this factory) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 (width is not split across blocks) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial tile handling in this factory) |
| 7 | leftover_width_in_row | uint32_t | Always 0 |
| 8 | start_stick_id | uint32_t | First stick ID for this core: `row_start_id` |

#### Writer Kernel (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address |
| 1 | num_tiles | uint32_t | Total tiles this core writes |
| 2 | start_id | uint32_t | First tile ID for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 (BRISC) | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 sticks per block, batch into CB |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` with compile-time args for address generation
  - Pre-computes NoC addresses for all 32 sticks in a block before issuing reads (`base_src_noc_addr[32]` array)
  - The `read_tiles` lambda does: `cb_reserve_back` -> 32x `noc_async_read` -> `noc_async_read_barrier` -> `cb_push_back`
  - The inner loop supports splitting the width into multiple blocks (`num_full_blocks_in_row`), but in this interleaved factory, it is always 1

**Relevance to layer_norm_rm**: This reader pattern demonstrates how to read RM sticks from interleaved DRAM and pack them into a CB. For layer_norm_rm, a similar pattern can be used to read input sticks, but the CB page size should match the stick size (RM page = one row) rather than tile size, since layer_norm operates on rows and produces RM output.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (TRISC) | N/A | CB c_0 | CB c_16 | Hardware tilize (RM to tile) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize unpack/math/pack hardware
  - Calls `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_tile_cnt, per_core_block_cnt)`
  - The helper iterates `per_core_block_cnt` times, each iteration: `cb_wait_front(c_0, block_width_tiles)` -> `cb_reserve_back(c_16, block_width_tiles)` -> `tilize_block` (or `fast_tilize_block`) -> `cb_push_back(c_16)` -> `cb_pop_front(c_0)`
  - Uses symmetric mode: both input and output CBs have tile-sized pages

### Writer Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 (NCRISC) | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write tiles one at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard tile-by-tile writer using TensorAccessor. One `cb_wait_front` / `cb_pop_front` per tile.

---

## Implementation Notes

### CB Sizing is Matched to Tile-Width Batching
The input CB capacity of `ntiles_per_block` tile-sized pages exactly holds 32 rows of the full tensor width. This is because `ntiles_per_block * tile_size = (W/32) * (32*32*elem_size) = W * 32 * elem_size = 32 sticks`. This identity is fundamental to the symmetric tilize pattern.

### Reader Reads Sticks But CB is Tile-Sized
Even though the reader reads individual sticks (rows), the CB is configured with tile-sized pages. The compute kernel's tilize unpack hardware knows to interpret the contiguous RM data in the CB and rearrange it into tile format. This is the standard "symmetric" tilize mode documented in `tilize_helpers.hpp`.

### Width is Never Split in This Factory
The reader supports splitting the width into multiple sub-blocks (`num_full_blocks_in_row`), but in the interleaved factory, it is always set to 1 with 0 leftover tiles. The entire row width is read in one shot.

### FP32 Accumulation
When the input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in the compute config, enabling FP32 accumulation in the destination register.

### Cliff Core Handling
The cliff core uses identical reader and writer kernel binaries but receives different runtime args (`nblocks_per_core_cliff` instead of `nblocks_per_core`). The compute kernel for cliff cores is compiled separately with `compute_args_cliff`.

### Override Runtime Arguments
The `override_runtime_arguments` method only updates buffer addresses (index 0) for both reader and writer. This enables program caching -- the compiled kernels can be reused with different tensor allocations.

---

## Key Patterns for layer_norm_rm Reuse

### Reader Pattern: RM Stick Reading from Interleaved DRAM
1. **TensorAccessor setup**: `TensorAccessorArgs(*buffer).append_to(ct_args)` on host; `TensorAccessorArgs<offset>()` + `TensorAccessor(args, addr, page_size)` on device
2. **Stick iteration**: Outer loop over groups of sticks, resolve NoC addresses via `get_noc_addr(stick_id, accessor)`
3. **Batch read**: Reserve CB space, issue `noc_async_read` per stick, barrier, push
4. **CB page sizing**: For RM-in/RM-out operations, the CB page size should match the stick size (not tile size), since no tilize is needed

### Work Distribution Pattern: 1D Block Split
1. Define total work units (e.g., total rows or row-groups)
2. Use `ceil(total_units / grid_area)` for units-per-core
3. Handle cliff core for remainder
4. Assign contiguous ranges of work to each core via runtime args

### CB Configuration Pattern
1. Use `create_cb(cb_index, program, core_set, page_size, num_pages, data_format)` from `cb_utils.hpp`
2. For double-buffering, set `num_pages = 2 * block_size` to allow overlap

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal kernels? How does get_noc_addr work with TensorAccessor to translate a stick_id to a NoC address for interleaved DRAM buffers?"
   **Reason**: Needed to understand the address generation mechanism used in the reader kernel
   **Key Findings**: TensorAccessorArgs packs buffer layout info into compile-time args on the host. On the device, TensorAccessor uses InterleavedAddrGen to compute bank_index = stick_id % num_banks and bank_offset from the stick_id, then combines with NoC XY coordinates to form a 64-bit NoC address. The `get_noc_addr(id, accessor)` function is the standard way to resolve logical page/stick IDs to physical addresses.

2. **Query**: "How does the circular buffer API work in tt-metal kernels? Specifically cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front."
   **Reason**: Needed to understand the synchronization semantics between reader, compute, and writer kernels
   **Key Findings**: `cb_reserve_back` blocks until space is available (producer side). `cb_push_back` marks data as ready for consumer. `cb_wait_front` blocks until data is available (consumer side). `cb_pop_front` frees space for producer. These form a producer-consumer handshake. Within compute kernels, three TRISC threads (unpack/math/pack) also synchronize via `tile_regs_acquire/commit/wait/release`.

3. **Query**: "What does the tilize operation do in tt-metal? How does the compute kernel tilize helper work?"
   **Reason**: Needed to understand the RM-to-tile conversion process
   **Key Findings**: Tilize converts row-major data to 32x32 tile format (with four 16x16 faces). The `tilize_block` function uses LLK unpack (`llk_unpack_tilize_block`) to read RM data from the input CB and reorder it into tiles, then packs to the output CB. A `fast_tilize` path exists for 32x32 tiles with Float32/Float16_b formats when not in full-sync dest mode.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM vs tile layout and interleaved memory distribution
   **Key Information**: RM layout: each row = one page. Tiled layout: each 32x32 tile = one page. Interleaved: pages distributed round-robin across DRAM banks. Tiles contain four 16x16 faces stored contiguously.

2. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` function used for core distribution
   **Key Information**: `compute_ncores` computes `nblocks_per_core = ceil(nblocks / grid_area)` and `ncores = ceil(nblocks / nblocks_per_core)`. The cliff core handles `nblocks % nblocks_per_core` remaining blocks. Returns `BlockSplit` struct with core ranges.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory
   **Key Information**: `create_cb` creates a `CircularBufferConfig` with `num_pages * page_size` total size, sets the page size per CB index, and calls `CreateCircularBuffer`. Optionally accepts a globally allocated buffer for sharded tensors.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel library's tilize function
   **Key Information**: The `compute_kernel_lib::tilize` template handles init/uninit lifecycle, wait mode, and register reconfiguration. In symmetric mode (default), both CBs have tile-sized pages. The main loop: `cb_wait_front(input, block_tiles)` -> `cb_reserve_back(output, block_tiles)` -> `tilize_block` -> `cb_push_back(output)` -> `cb_pop_front(input)`. Auto-selects `fast_tilize_block` when possible.

5. **Source**: `tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
   **Reason**: Understanding the host-side TensorAccessorArgs API
   **Key Information**: `append_to(compile_time_args)` appends buffer layout metadata to compile-time args vector. Constructed from a `Buffer&` or `Buffer*`. Also supports a two-arg `append_to` that splits between compile-time and common runtime args.
