# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize multi-core interleaved** operation converts a tensor from **row-major (stick) layout** to **tiled layout** (32x32 tiles), distributing the work across multiple Tensix cores. The input tensor is stored in interleaved memory (pages round-robin distributed across DRAM or L1 banks), and the output is also stored in interleaved memory but in tile format. This is the foundational "input stage" for any pipeline that receives row-major data and needs to process it through the Tensix compute engine, which natively operates on tiles.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (a group of tiles spanning one tile-row of the tensor's width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = number of tile-height (32-row) strips across all batch/height dimensions |
| **Loop structure** | Outer loop: iterate over blocks (each block = 32 rows of the full width). Inner: reader reads all sticks for 32 rows across the full width; compute tilizes the block; writer writes the resulting tiles. |

A **block** represents one horizontal strip of 32 rows spanning the full padded width of the tensor. The number of tiles in a block equals `padded_shape[-1] / 32` (i.e., `ntiles_per_block`). The total number of blocks equals the total number of tile-rows: `physical_volume / TILE_HW / ntiles_per_block`.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (treated as flattened 2D: outer dims x inner dim) |
| **Dimension convention** | Last dimension is contiguous in memory (row-major) |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 (determined by input memory config) |
| **Data type** | BFLOAT16 or FLOAT32 (fp32_llk_acc enabled for FLOAT32) |

**Page definition**: Each page (stick) is one row of the flattened 2D representation. Page size = `padded_shape[-1] * element_size()` bytes. Pages are distributed round-robin across memory banks.

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (padded to tile alignment) |
| **Dimension convention** | Same outer dims; inner two dims tiled to 32x32 |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 16x16 faces) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 (determined by output memory config) |
| **Data type** | Same as input, or as specified by `output_dtype` |

**Page definition**: Each page is one 32x32 tile. Tile size = `tile_size(output_cb_data_format)` bytes. Tiles are distributed round-robin across memory banks.

### Layout Transformations

The core transformation is **tilize**: 32 consecutive row-major sticks (each of width `padded_shape[-1]` elements) are reorganized into `ntiles_per_block` tiles (each 32x32 elements, stored as four 16x16 faces). The reader gathers the sticks into L1 in row-major order; the compute kernel's `tilize_block` / `fast_tilize_block` HW instruction reorders them into tiled format.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations | Details |
|-------|--------|------------|-----------|---------------|---------|
| 1 | Reader | DRAM/L1 (interleaved sticks) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `cb_push_back` | Reads 32 sticks (rows) at a time, each of width `block_width_size` bytes. Uses TensorAccessor to resolve stick_id to NoC address. |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front`, `tilize_block`/`fast_tilize_block`, `cb_pop_front`, `cb_push_back` | Unpacks row-major data from c_0, tilizes into 32x32 tiles, packs into c_16. |
| 3 | Writer | CB c_16 | DRAM/L1 (interleaved tiles) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` | Writes one tile at a time using TensorAccessor. Sequential tile IDs starting from `start_id`. |

### Detailed Data Flow

1. **Reader** iterates over tile-height groups (32 rows per group) assigned to this core:
   - For each group, resolves the NoC address for each of the 32 sticks using `get_noc_addr(stick_id, s)` (TensorAccessor-based lookup).
   - Reserves `ntiles_per_block` pages in CB c_0.
   - Issues 32 `noc_async_read` calls (one per stick), each reading `block_width_size` bytes (the full row width). The writes to L1 are sequential within the CB, packing all 32 rows contiguously.
   - Waits for all reads (`noc_async_read_barrier`), then pushes `ntiles_per_block` pages to c_0.

2. **Compute** processes blocks in a loop (`per_core_block_cnt` iterations):
   - Waits for `ntiles_per_block` pages in CB c_0 (the 32 rows of row-major data).
   - Reserves `ntiles_per_block` pages in CB c_16.
   - Calls `tilize_block` or `fast_tilize_block` to transform the row-major data into tiles in-place in the register file, then packs to c_16.
   - Pops the consumed pages from c_0 and pushes produced tiles to c_16.

3. **Writer** iterates over tiles in the output:
   - Waits for 1 tile in CB c_16.
   - Reads the tile from c_16 and writes it to the output buffer using `noc_async_write_page(tile_id, s, l1_read_addr)`.
   - Pops the consumed tile from c_16.
   - After all tiles: `noc_async_write_barrier`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (row-major sticks) | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | cb_out0 | Output staging (tilized tiles) | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

**Key observations**:

- **CB c_0** capacity = `ntiles_per_block` pages of `input_single_tile_size`. This holds exactly one block's worth of row-major data (32 rows x full width). The page size is set to `input_single_tile_size` (tile-sized pages), and the number of pages equals `ntiles_per_block`. This means the total CB size = `ntiles_per_block * input_single_tile_size`, which corresponds to `padded_shape[-1] / 32 * tile_size` bytes -- exactly the amount needed for 32 full-width rows of data.

- **CB c_16** capacity = `ntiles_per_block` pages of `output_single_tile_size`. This holds exactly one block's worth of output tiles.

- Both CBs are **single-buffered** (capacity == block size). This means the reader must complete writing an entire block before compute can process it, and compute must complete before the writer can drain it. There is no overlap within a single block.

## Pipeline Pattern Summary

Both CBs have capacity equal to block size (`ntiles_per_block` pages), making them **single-buffered**. This means:

- **Reader-Compute overlap**: Not possible within a block. The reader fills the entire CB c_0, then compute consumes all of it. However, since both CBs are sized for exactly one block, once compute finishes consuming c_0 and produces to c_16, the reader can begin filling c_0 for the next block while the writer drains c_16.
- **Compute-Writer overlap**: Limited. The writer processes one tile at a time, so partial overlap is possible after compute pushes tiles. But since capacity = block size, compute produces the entire block before the next input can arrive.

For a hybrid operation reusing this pattern, **doubling CB capacity** (to `2 * ntiles_per_block`) would enable double-buffering and allow reader/compute overlap between consecutive blocks.

## Index Calculations

### Reader: Stick ID to NoC Address

The reader uses `TensorAccessor` to map `stick_id` to a NoC address:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time arg offset = 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

- `stick_id` starts at `start_stick_id` (a runtime arg) and increments by 1 per row.
- `TensorAccessor` resolves `stick_id` -> `(bank_index, bank_offset)` -> `noc_xy + local_address` using interleaved round-robin distribution.
- Page size for the accessor = `stick_size` (= `padded_shape[-1] * element_size()`, the compile-time arg at index 0).

### Writer: Tile ID to NoC Address

The writer uses `TensorAccessor` to map tile IDs to NoC addresses:

```cpp
constexpr auto dst_args = TensorAccessorArgs<1>();  // compile-time arg offset = 1
const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);
// ...
noc_async_write_page(i, s, l1_read_addr);
```

- `i` starts at `start_id` (a runtime arg) and increments by 1 per tile.
- Page size is derived from the CB interface: `get_local_cb_interface(cb_id_out).fifo_page_size`.

### Block-to-Stick/Tile Mapping

For each core processing `N` blocks:
- **Stick range**: `[row_start_id, row_start_id + N * TILE_HEIGHT)` (each block = 32 rows)
- **Tile range**: `[tile_start_id, tile_start_id + N * ntiles_per_block)` (each block = `ntiles_per_block` tiles)

## Memory Access Patterns

### Read Pattern

- **Pattern**: Strided reads with sequential sticks within each block.
- **Detail**: For each block, the reader resolves 32 stick addresses (which may land on different DRAM banks due to interleaving). Within the `read_tiles` lambda, it reads `block_width_size` bytes from each of the 32 sticks sequentially. The writes to L1 are contiguous (sequential addresses in the CB).
- **Observation**: The 32 NoC addresses are pre-computed into `base_src_noc_addr[32]` before the read loop. After reading a block's worth, `base_src_noc_addr[k] += width_size` for potential multi-block-per-row scenarios (not used in the interleaved factory since `num_full_blocks_in_row = 1`).
- **DRAM access**: Round-robin across banks. Adjacent sticks may reside on different banks, providing bank-level parallelism.

### Write Pattern

- **Pattern**: Sequential tile writes, one tile at a time.
- **Detail**: The writer processes tiles sequentially from `start_id` to `start_id + num_pages - 1`. Each tile is written via `noc_async_write_page`, which resolves the tile_id to a bank and offset. The write uses `noc_async_writes_flushed()` after each tile to allow CB pop, enabling the compute kernel to continue producing tiles.
- **DRAM access**: Round-robin across banks based on tile_id. Sequential tile IDs go to different banks.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from available 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores (or custom `sub_core_grids`) |
| **Total cores** | `ncores` (computed by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (each block = `ntiles_per_block` tiles = 32 rows) |
| **Load balancing** | Equal distribution with optional cliff core |
| **Remainder handling** | Last core may get `nblocks_per_core_cliff` blocks (fewer than others) |

### Distribution Details

The `split_blocks_for_tilize` function (from `work_split_tilize.hpp`) performs the following:

1. Computes `nblocks_per_core = ceil(nblocks / grid_area)`.
2. Computes actual `ncores = ceil(nblocks / nblocks_per_core)`.
3. If `nblocks % nblocks_per_core != 0`, the last core is a "cliff" core with `nblocks_per_core_cliff = nblocks % nblocks_per_core` blocks.
4. Returns `core_range` (full cores) and `core_range_cliff` (cliff core, if any).

Cores are enumerated from the `available_grid` in row-major order (using `corerange_to_cores`). The runtime args loop assigns contiguous ranges of sticks and tiles to each core:

- Core `i` starts at `row_start_id = i * nblocks_per_core * TILE_HEIGHT` and processes `nblocks_per_core * TILE_HEIGHT` sticks.
- Core `i` starts at `tile_start_id = i * nblocks_per_core * ntiles_per_block` and produces `nblocks_per_core * ntiles_per_block` tiles.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` (aka `block_size_nbytes`) | uint32_t | Size of one row in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessor args | uint32_t[] | Interleaved buffer metadata (rank, num_banks, bank coords, etc.) appended by `TensorAccessorArgs(*src0_buffer).append_to()` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks this core processes (`nblocks_per_core` or `nblocks_per_core_cliff`) |
| 1 | `per_core_block_tile_cnt` | uint32_t | Number of tiles per block (`ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH`) |

**Note**: Cliff cores get a separate kernel instance with `nblocks_per_core_cliff` for arg 0. The `fp32_dest_acc_en` flag is set in `ComputeConfig` when the input dtype is FLOAT32.

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out` | uint32_t | Output circular buffer index (`c_16`, value 16) |
| 1+ | TensorAccessor args | uint32_t[] | Interleaved buffer metadata for the output buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | `num_sticks` | uint32_t | Total sticks (rows) to read = `nblocks_per_core * TILE_HEIGHT` |
| 2 | `block_size_nbytes` | uint32_t | Size of one full-width row in bytes (same as CT arg 0; redundant for flexibility) |
| 3 | `num_tiles_per_block` | uint32_t | Tiles per block (`ntiles_per_block`) |
| 4 | `block_width_size` | uint32_t | Width of a block in bytes (= `block_size_nbytes` since `num_full_blocks_in_row = 1`) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for this factory (one block spans the full width) |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 (no partial blocks in interleaved variant) |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 |
| 8 | `start_stick_id` | uint32_t | First stick (row) index for this core = `row_start_id` |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | `num_pages` | uint32_t | Total tiles to write = `ntiles_per_block * nblocks_per_core` |
| 2 | `start_id` | uint32_t | First tile index for this core = `tile_start_id` |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM/L1 interleaved sticks | CB c_0 | Read 32 rows per block, NoC async read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Pre-computes NoC addresses for all 32 sticks in a tile-height group before issuing reads.
  - Uses a `read_tiles` lambda that reserves CB space, reads all 32 rows of a block width, waits for completion, then pushes.
  - The inner loop `for (j = 0; j < num_full_blocks_in_row; j++)` allows for multi-block rows, but in this factory `num_full_blocks_in_row = 1` (the entire width is one block).
  - This means each `cb_reserve_back / cb_push_back` cycle handles `ntiles_per_block` pages covering all tiles in one row of tiles.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | tilize_block / fast_tilize_block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16)` to initialize UNPACK, MATH, and PACK hardware units with the correct CB formats.
  - Delegates to `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` from `tilize_helpers.hpp`.
  - The helper library automatically selects `fast_tilize` (hardware-accelerated path) when conditions are met: 32x32 tiles, Float32 or Float16_b format, and half-sync dest mode.
  - The main loop processes `per_core_block_cnt` blocks. Per block: `cb_wait_front(c_0, block_width_tiles)`, `cb_reserve_back(c_16, block_width_tiles)`, `tilize_block(c_0, block_width_tiles, c_16)`, then push/pop.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM/L1 interleaved tiles | Write tiles one at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**:
  - This is a **generic writer** shared across many operations. It writes `num_pages` tiles starting from `start_id`.
  - Processes tiles one at a time: `cb_wait_front(c_16, 1)`, read L1 pointer, `noc_async_write_page(tile_id, s, l1_addr)`, `noc_async_writes_flushed()`, `cb_pop_front(c_16, 1)`.
  - The per-tile `noc_async_writes_flushed()` (not a full barrier) ensures the write is in-flight before popping the CB, allowing the next tile to be produced while the previous write completes.
  - Supports `OUT_SHARDED` and `BACKWARDS` modes via preprocessor defines, but neither is used in this factory.

## Implementation Notes

### Why `num_full_blocks_in_row` is Always 1

In the interleaved variant, the full padded width of the tensor fits in a single block. The reader reads all `padded_shape[-1]` elements of each stick in one shot. The `num_full_blocks_in_row` runtime arg is set to 1, and `leftover` args are 0. This simplifies the reader logic but means the CB must be large enough to hold `ntiles_per_block` tile-sized pages (one full tile-row).

### TensorAccessor Pattern

Both reader and writer use the `TensorAccessor` pattern:
- **Host side**: `TensorAccessorArgs(buffer).append_to(compile_time_args)` serializes buffer metadata (rank, bank count, bank coordinates, etc.) into compile-time args. Default config = `ArgConfig::None` (all compile-time).
- **Device side**: `TensorAccessorArgs<offset>()` + `TensorAccessor(args, base_addr, page_size)` reconstructs the accessor from compile-time args. The `get_noc_addr(page_id, accessor)` function resolves page IDs to physical addresses.

### Override Runtime Arguments (Caching)

The `override_runtime_arguments` method updates only `src_addr` (arg 0) and `dst_addr` (arg 0) when the program is reused with different buffer addresses. All other args remain unchanged since they depend on tensor shape, which is fixed for cached programs.

### Fast Tilize Selection

The compute kernel library automatically selects `fast_tilize_block` over `tilize_block` at compile time when:
1. Tile dimensions are 32x32 (checked via `pack_tile_r_dim` / `pack_tile_c_dim`).
2. Data format is Float32 or Float16_b (checked via `unpack_src_format`).
3. Destination is in half-sync mode (not full sync).

This is transparent to the program factory -- it only sets `fp32_dest_acc_en` in `ComputeConfig`.

### Relevance to Hybrid Operations (Layer Norm RM Pipeline)

For a hybrid operation that reads RM sticks, tilizes, computes (e.g., layer norm), then untilizes back to RM:

1. **This tilize pattern is the input stage**. The reader kernel (`reader_unary_stick_layout_split_rows_interleaved`) and the CB c_0 configuration serve as the template for reading RM sticks into a CB.
2. **CB c_0 sizing**: Must hold `ntiles_per_block` pages of `input_single_tile_size`. For a fused operation, this CB would feed into the tilize step of the compute kernel.
3. **CB c_16 sizing**: After tilize, tiles in c_16 can be consumed by subsequent compute stages (e.g., reduce for mean/variance computation) rather than being written out.
4. **Work distribution**: The 1D block-based distribution (blocks = tile-height strips) is directly applicable. Each core processes a contiguous set of tile-rows.
5. **Key reusable components**:
   - `split_blocks_for_tilize` for work distribution
   - `reader_unary_stick_layout_split_rows_interleaved` reader pattern for RM input
   - `compute_kernel_lib::tilize<>()` helper for the tilize compute step
   - `TensorAccessor` pattern for both reader and writer
6. **Modification needed for fusion**: Instead of writing tilized tiles to DRAM (writer), the tiles would remain in L1 CBs and flow into the next compute stage. The writer would be replaced with an untilize + RM writer at the end of the pipeline.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? Specifically, how does it convert row-major data to tiled format?"
   **Reason**: Needed to understand the fundamental transformation being performed by this operation.
   **Key Findings**: Tilize reorders row-major elements into 32x32 tiles (each subdivided into 16x16 faces). The LLK functions `llk_unpack_tilize` and `llk_unpack_tilize_block` handle the actual data reordering in hardware. Padding is applied if dimensions are not tile-aligned.

2. **Query**: "What are the CBIndex constants c_0 and c_16 in tt-metal circular buffer conventions?"
   **Reason**: Needed to understand the CB naming convention used in the program factory.
   **Key Findings**: c_0 is conventionally used for input operands, c_16 for output operands. The hardware supports 32 CBs; the specific indices are arbitrary but conventionally standardized. Intermediate buffers often use c_24+.

3. **Query**: "How does the split_blocks_for_tilize function work for 1D work distribution?"
   **Reason**: Needed to understand the work splitting strategy used in the program factory.
   **Key Findings**: Divides total blocks across cores with `ceil(nblocks / grid_area)` blocks per core. If there is a remainder, the last core becomes a "cliff" core with fewer blocks. Returns separate CoreRangeSets for full cores and cliff core.

4. **Query**: "What does compute_kernel_hw_startup do in tt-metal compute kernels?"
   **Reason**: The compute kernel calls this before tilize, needed to understand its role.
   **Key Findings**: Initializes UNPACK, MATH, and PACK hardware units via MMIO writes. Must be called exactly once at the start of a compute kernel. Configures data formats and sync modes for all three compute TRISCs.

5. **Query**: "In tt-metal reader kernels, what is the get_noc_addr function with TensorAccessor?"
   **Reason**: The reader uses `get_noc_addr(stick_id, s)` to resolve addresses.
   **Key Findings**: For interleaved buffers, it computes `bank_index` and `bank_offset` from the page ID, retrieves the NoC x-y coordinates for that bank, and constructs a 64-bit NoC address combining coordinates and local address.

6. **Query**: "What does noc_async_write_page do in tt-metal writer kernels?"
   **Reason**: The writer uses `noc_async_write_page` instead of raw `noc_async_write`.
   **Key Findings**: `noc_async_write_page` is a higher-level API that uses the TensorAccessor/AddrGen to resolve the destination address internally, while `noc_async_write` requires explicit address computation. The page version simplifies tile-by-tile writes.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor usage pattern for both host-side setup and device-side access.
   **Key Information**: Host side uses `TensorAccessorArgs(buffer).append_to(compile_time_args)` to serialize; device side reconstructs with `TensorAccessorArgs<offset>()` + `TensorAccessor(args, addr, page_size)`. For interleaved buffers, `get_noc_addr(page_id)` resolves to physical address.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the distinction between row-major pages (sticks) and tiled pages.
   **Key Information**: Row-major layout: each row = one page. Tiled layout: each 32x32 tile = one page (with faces). Interleaved memory distributes pages round-robin across banks.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel library used by the tilize compute kernel.
   **Key Information**: The `tilize<>()` template function handles init/uninit lifecycle, fast_tilize selection, symmetric vs asymmetric CB pages, and configurable wait modes. Default mode: `WaitBlock` (wait per block), `InitAndUninit` (full lifecycle).

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize(CoreRangeSet, nblocks)` computes per-core block counts and returns core range sets. Uses `compute_ncores` for the basic division.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility used to create circular buffers.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, format)` creates a `CircularBufferConfig` with total size = `num_pages * page_size` and sets the page size for the specified CB index.
