# Tilize Multi-Core Interleaved Implementation Analysis

## Overview

The **tilize** operation converts a tensor from row-major layout to tiled (32x32) layout. The input tensor is stored as flat rows ("sticks") in interleaved DRAM; the output tensor is stored as tiles in interleaved DRAM. This analysis focuses on the **input stage**: how row-major sticks are read from DRAM, how they are batched into groups of 32 for tile formation, the input circular buffer sizing strategy, and the core work distribution model.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role focus**: input_stage -- reader kernel pattern, input CB sizing, stick-to-tile batching, work distribution.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (a horizontal strip of tiles spanning the full tensor width at tile height = 32 rows) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`) |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = total tile rows in the tensor |
| **Loop structure** | Outer: iterate over blocks (groups of 32 sticks). Inner: iterate over full-width tile columns per block. |

A single **block** corresponds to one horizontal strip of 32 rows (one tile height) spanning the full width of the tensor. Each block contains `ntiles_per_block` tiles. The total number of blocks equals the number of tile rows: `physical_volume / TILE_HW / ntiles_per_block`.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (flattened to 2D internally) |
| **Dimension convention** | Last dim = width (stick length); all other dims collapsed to height |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |

**Page definition for row-major interleaved**: Each page is one row (one "stick") of the flattened 2D tensor. The page size in bytes is `padded_shape[-1] * element_size()`. Pages are distributed round-robin across DRAM banks.

### Output Tensor (de-emphasized)

| Property | Value |
|----------|-------|
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (or converted) |

### Layout Transformation

The core transformation is ROW_MAJOR to TILE_LAYOUT. The reader gathers 32 consecutive row-major sticks and writes them as a contiguous block into the input CB. The compute kernel's `tilize_block` (or `fast_tilize_block`) rearranges this row-major block into tiles with face-interleaved ordering (four 16x16 faces per 32x32 tile).

---

## Data Flow Pattern (Input-Stage Focus)

### Step 1: Reader Reads 32 Sticks from DRAM

For each block (group of 32 rows):

1. **Resolve NoC addresses**: For each of the 32 sticks in the block, the reader calls `get_noc_addr(stick_id, s)` using the TensorAccessor. This maps the logical stick ID to a physical DRAM bank address via round-robin page distribution. The 32 addresses are cached in a local `base_src_noc_addr[32]` array.

2. **Reserve CB space**: `cb_reserve_back(cb_id_in0, ntiles_per_block)` -- reserves enough pages in the input CB for one full-width tile row. This blocks if the compute kernel has not yet consumed previous data.

3. **Read row-major data**: For each of the 32 sticks, the reader issues `noc_async_read(src_noc_addr, l1_write_addr, width_size)` where `width_size = block_width_size` (full stick width in bytes). The L1 write address advances by `width_size` per stick, placing all 32 sticks contiguously in the CB.

4. **Barrier + push**: `noc_async_read_barrier()` waits for all 32 reads to complete, then `cb_push_back(cb_id_in0, ntiles_per_block)` signals the compute kernel that one block of data is ready.

### Step 2: Compute Tilizes the Block (brief)

The compute kernel calls `tilize_block` or `fast_tilize_block`, which reads from CB c_0 (row-major) and writes to CB c_16 (tiled format).

### Step 3: Writer Writes Tiles to DRAM (de-emphasized)

The writer reads tiles one at a time from CB c_16 and writes them to DRAM.

---

## Reader Kernel Pattern: Detailed Analysis

**Kernel file**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

### Key Design: "Split Rows" Pattern

The name "split_rows" refers to the fact that this reader can split the reading of a full-width row across multiple CB pushes when there are multiple tile-column blocks per row. However, in this program factory, `num_full_blocks_in_row` is always set to 1, meaning the entire width is read in a single CB transaction.

### Stick-to-Tile Batching

The reader groups exactly **32 sticks** (one `TILE_HEIGHT`) per iteration of the outer loop. This is the fundamental unit of input for tile formation:

```
for (i = 0; i < num_sticks / tile_height; i++) {     // outer: tile-rows
    for (j = 0; j < tile_height; j++) {                // resolve 32 addresses
        base_src_noc_addr[j] = get_noc_addr(stick_id, s);
        stick_id++;
    }
    for (j = 0; j < num_full_blocks_in_row; j++) {    // inner: column blocks (always 1 here)
        read_tiles(ntiles_per_block, block_width_size);
    }
}
```

### CB Push Granularity

Each `read_tiles` call reserves and pushes `ntiles_per_block` pages (tiles) at once. This means the compute kernel receives a complete row of tiles in each CB transaction.

### Memory Access Pattern: Reads

- **Pattern**: Strided -- the reader reads 32 sticks where each stick is a separate DRAM page. Due to interleaved round-robin distribution, consecutive sticks may reside on different DRAM banks, enabling bank-level parallelism.
- **Width**: Each read is one full stick width (`block_width_size` = `padded_shape[-1] * element_size` bytes).
- **Address pre-computation**: All 32 NoC addresses are resolved upfront before any reads begin, stored in `base_src_noc_addr[32]`. This separation of address computation from data transfer is an optimization.
- **Barrier**: A single `noc_async_read_barrier()` after all 32 reads ensures the entire block is in L1 before signaling the compute kernel.

### TensorAccessor Usage

The reader uses `TensorAccessor` initialized with compile-time arguments (via `TensorAccessorArgs<1>()`). The `1` indicates the compile-time argument offset starts at index 1 (after the `stick_size` arg at index 0). At runtime, only `src_addr` (the buffer base address) is passed. The accessor handles the mapping from stick_id to the correct DRAM bank and offset within that bank.

---

## Circular Buffer Configuration (Input-Stage Focus)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input staging (RM sticks) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_out0 | Output staging (tiled) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

### Input CB Sizing Rationale (c_0)

The input CB (c_0) is allocated with:
- **Page size**: `input_single_tile_size` = `tt::tile_size(input_cb_data_format)` -- the size of one tile in the input data format.
- **Number of pages**: `ntiles_per_block` = `padded_shape[-1] / TILE_WIDTH` -- the number of tiles that span the full width.
- **Total capacity**: `ntiles_per_block * input_single_tile_size` bytes.

**Important nuance**: Even though the input data is row-major, the CB page size is set to `tile_size` (e.g., 2048 bytes for bfloat16 32x32 tiles). This works because the reader writes 32 sticks contiguously (each stick = `padded_shape[-1] * element_size` bytes), and `32 * padded_shape[-1] * element_size` = `ntiles_per_block * tile_size`. The CB capacity exactly fits one full block of 32 rows.

**Why single-buffered**: Both capacity and block size equal `ntiles_per_block` tiles. The reader must wait for the compute kernel to consume the entire block before writing the next one. There is no overlap between reading block N+1 and computing block N.

### Page Format in Input CB

The input CB holds data in **row-major order**: 32 contiguous sticks, each of width `padded_shape[-1]` elements. The compute kernel's tilize function understands this layout and rearranges it into tiled format during the tilize_block operation.

---

## Pipeline Pattern Summary

Both CBs are single-buffered (capacity = block_size), so:
- Reader and compute alternate: reader fills a block, then compute processes it.
- Compute and writer alternate similarly.
- No pipelining overlap between stages within a single core.

(Detailed pipeline analysis out of scope per instructions.)

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent, e.g., 8x8 = 64 cores) |
| **Total cores** | `ncores` = ceil(`nblocks` / `nblocks_per_core`) |
| **Work per core** | `nblocks_per_core` blocks (except cliff core) |
| **Load balancing** | Equal division with single cliff core for remainder |

### Work Splitting: `split_blocks_for_tilize`

The function `ttnn::split_blocks_for_tilize(available_grid, nblocks)` performs the following:

1. **Compute `nblocks_per_core`**: `ceil(nblocks / grid_area)` -- distributes blocks as evenly as possible.
2. **Compute `ncores`**: `ceil(nblocks / nblocks_per_core)` -- only activate as many cores as needed.
3. **Compute `nblocks_per_core_cliff`**: `nblocks % nblocks_per_core` -- the last core gets fewer blocks if there is a remainder.
4. **Partition into core ranges**:
   - `core_range`: all cores except the cliff core (if any), each processing `nblocks_per_core` blocks.
   - `core_range_cliff`: the single cliff core (if `nblocks_per_core_cliff > 0`), processing the remainder.

### Per-Core Work Assignment

Each core is assigned a contiguous range of blocks (tile rows). The runtime arguments encode:
- `row_start_id`: the first stick (row) this core reads from.
- `num_sticks`: `nblocks_per_core * TILE_HEIGHT` (or cliff equivalent).
- `tile_start_id`: the first output tile this core writes.

Cores process consecutive blocks with no overlap. Core 0 starts at row 0, Core 1 starts at row `nblocks_per_core * 32`, etc.

### Concrete Example

For a tensor with shape `[1, 1, 256, 512]` (bfloat16):
- `ntiles_per_block` = 512 / 32 = 16 tiles per block
- `ntiles` = 256 * 512 / (32 * 32) = 128 tiles
- `nblocks` = 128 / 16 = 8 blocks (= 8 tile rows = 256 sticks / 32 per block)
- On a 64-core grid: `nblocks_per_core` = ceil(8/64) = 1, `ncores` = 8, `cliff` = 0
- Each of the 8 cores processes one block of 32 sticks.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one row-major stick in bytes (`padded_shape[-1] * element_size`) |
| 1+ | TensorAccessor args | uint32_t[] | Packed TensorAccessor parameters (bank mapping, distribution spec) appended via `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile rows) this core processes |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (`ntiles_per_block`) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_sticks | uint32_t | Total sticks (rows) this core reads (`nblocks * TILE_HEIGHT`) |
| 2 | block_width_size | uint32_t | Stick width in bytes (same as compile-time `stick_size` for full blocks) |
| 3 | num_tiles_per_block | uint32_t | Tiles per block (width) |
| 4 | block_width_size | uint32_t | Width of data to read per stick (same as index 2 for full blocks) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 in this factory (full-width blocks) |
| 6 | num_leftover_tiles | uint32_t | Always 0 (no partial blocks) |
| 7 | leftover_width | uint32_t | Always 0 (no partial width) |
| 8 | start_stick_id | uint32_t | First stick (row) ID this core should read |

Note: Args at indices 2/4 and 5/6/7 suggest the reader kernel supports partial-width blocks (splitting a row across multiple CB pushes), but this factory always uses full-width with no leftovers.

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 sticks per block, batch into CB |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Pre-computes all 32 NoC addresses before issuing reads (address array pattern).
  - Issues 32 `noc_async_read` calls per block, one per stick, each reading `width_size` bytes.
  - Single barrier after all 32 reads, then single push of `ntiles_per_block` pages.
  - The `num_full_blocks_in_row` inner loop allows for width partitioning but is always 1 here.

### Compute Kernel (brief)

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- Calls `compute_kernel_lib::tilize<c_0, c_16>` with WaitBlock mode.
- Per block: waits for `ntiles_per_block` pages in c_0, reserves c_16, performs tilize_block, pushes c_16, pops c_0.

### Writer Kernel (de-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Generic tile writer: reads one tile at a time from output CB, writes to DRAM via TensorAccessor.

---

## Implementation Notes

### Key Patterns Relevant for layer_norm_rm

1. **Stick-level reading from interleaved DRAM**: The reader demonstrates how to use TensorAccessor with `get_noc_addr(stick_id, s)` where `stick_id` is the page ID for a row-major tensor. For layer_norm_rm, which also reads row-major data, this same pattern applies -- each row is a separate DRAM page.

2. **Batching 32 sticks for tile-height processing**: The reader groups exactly 32 sticks before any processing. For layer_norm_rm, the reader could similarly batch sticks, though layer norm processes each row independently (mean/variance across the row width), so the batching quantum might differ.

3. **Full-width CB allocation**: The input CB is sized to hold one complete row of tiles (`ntiles_per_block` tiles). This ensures the compute kernel has the full width available for processing. For layer_norm_rm, where the reduction is along the width dimension, having the full width in the CB at once is essential.

4. **Single-buffered approach**: Both CBs are single-buffered. For higher throughput, a new operation could double-buffer the input CB (allocate `2 * ntiles_per_block` pages) so the reader can fill the next block while compute processes the current one.

5. **TensorAccessor for interleaved access**: The pattern of creating `TensorAccessorArgs(*buffer)` on host and `TensorAccessor(args, addr, page_size)` in the kernel is the standard approach for interleaved tensor access. The page_size for row-major tensors is the stick size (row width in bytes).

6. **Work distribution via `split_blocks_for_tilize`**: The 1D block-splitting approach with a cliff core is simple and effective. For layer_norm_rm, a similar strategy can distribute rows (or groups of rows) across cores.

7. **Compute kernel compile-time args for block dimensions**: The compute kernel receives its workload dimensions as compile-time args (block count, tiles per block). This means different cores can have different kernels compiled (e.g., cliff core with fewer blocks). The same pattern works for layer_norm_rm's compute kernel.

### Input Data Size Constraint

The CB capacity must hold at least `ntiles_per_block * tile_size` bytes. For a tensor width of W elements in bfloat16, this is `(W/32) * 2048 = W * 64` bytes. For example, W=4096 requires 256 KB of CB space for the input alone, which is a significant fraction of the 1.5 MB L1 SRAM. Layer_norm_rm should account for this when allocating multiple CBs (input, mean, variance, gamma, beta, output).

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in TTNN? Specifically, how does it convert row-major tensor data into tiled format?"
   **Reason**: Needed to understand the end-to-end tilize flow and the relationship between sticks and tiles.
   **Key Findings**: Tilize reorganizes row-major sticks into 32x32 tiles with face-interleaved ordering. Padding is auto-applied if dimensions are not tile-aligned. The compute kernel performs the actual data rearrangement using `tilize_block` or `fast_tilize_block`.

2. **Query**: "How does get_noc_addr work with TensorAccessor for interleaved row-major tensors?"
   **Reason**: Needed to understand the page-to-bank mapping for the reader kernel.
   **Key Findings**: TensorAccessor maps page_id to bank_index via round-robin (modulo num_banks), computes the offset within that bank, then combines with the bank's NoC coordinates to form a 64-bit NoC address. For row-major tensors, each stick is one page.

3. **Query**: "What does cb_reserve_back do in tt-metal dataflow kernels?"
   **Reason**: Needed to understand reader-compute synchronization.
   **Key Findings**: `cb_reserve_back` is a blocking call that waits until the consumer has freed enough space in the CB. The `pages_received` and `pages_acked` counters coordinate producer/consumer. This is the primary synchronization mechanism between reader and compute kernels.

4. **Query**: "What is the split_blocks_for_tilize function and how does it distribute blocks?"
   **Reason**: Needed to understand the core distribution strategy.
   **Key Findings**: Computes `nblocks_per_core = ceil(nblocks / grid_area)`, assigns equal blocks to all cores except the last ("cliff") core which handles the remainder. Returns separate CoreRangeSet objects for regular and cliff cores.

5. **Query**: "In TT-Metal, what is a 'block' in the context of the tilize operation?"
   **Reason**: Needed to clarify the block/tile/row terminology.
   **Key Findings**: A block is a horizontal strip of `ntiles_per_block` tiles (one tile-height = 32 rows wide). `nblocks` = total tile rows. The block is the unit of work assigned to cores and the unit of CB transactions.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major page layout and interleaved memory distribution.
   **Key Information**: In row-major layout, each row is one page. In interleaved mode, pages are distributed round-robin across DRAM banks. For tiled layout, each 32x32 tile is one page.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API for address generation.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` creates an accessor on the device side. `get_noc_addr(page_id)` returns a 64-bit NoC address. Args can be compile-time or runtime. `TensorAccessorArgs<base_idx>()` in the kernel reads compile-time args starting at the given offset.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper.
   **Key Information**: `create_cb(cb_index, program, cores, page_size, num_pages, data_format)` creates a CircularBufferConfig with `num_pages * page_size` total capacity, sets the page_size per CB, and calls `CreateCircularBuffer`.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute-side tilize mechanism.
   **Key Information**: The symmetric mode (no `total_input_pages`) treats both CBs as tile-sized pages. Per block: `cb_wait_front(input, block_width_tiles)`, `cb_reserve_back(output, block_width_tiles)`, `tilize_block(input, width, output)`, `cb_push_back(output)`, `cb_pop_front(input)`. This confirms the reader's push granularity of `ntiles_per_block` tiles matches the compute's wait granularity.
