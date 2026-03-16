# Tilize (Multi-Core Default) Implementation Analysis

## Overview

The tilize operation converts row-major (RM) tensor data into tiled format (32x32 tiles). The "multi-core default" variant handles interleaved DRAM inputs, distributes work across multiple Tensix cores using a 1D block-splitting strategy, and uses TensorAccessor for page-level DRAM reads.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_default_program_factory.cpp`

**Focus**: This analysis emphasizes the **reader kernel pattern** (how RM sticks are read from DRAM), input CB sizing and page format, stick-to-tile batching, and work distribution. Writer and output-side details are de-emphasized per the input_stage reference role.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of the tensor width) |
| **Unit size** | `ntiles_per_block` tiles = `ceil(logical_width / 32)` tiles |
| **Total units** | `nblocks` = `ceil(total_output_tiles / ntiles_per_block)` = total tile-rows across all batches/height |
| **Loop structure** | Outer: iterate over `num_rows / TILE_HEIGHT` tile-rows; Inner: iterate over `num_full_blocks_in_row` (always 1 for non-sharded default case) |

A **block** corresponds to one horizontal strip of 32 rows (TILE_HEIGHT) spanning the full logical width. Reading one block means reading 32 consecutive RM sticks (pages) and producing `ntiles_per_block` tiles of output.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D, flattened to 2D for memory: [outer_dims, logical_width] |
| **Dimension convention** | Last dim = width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page definition** | One page = one row of width `logical_width` (for non-sharded interleaved). Pages are distributed round-robin across DRAM banks. |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same logical shape as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input dtype (or specified `output_dtype`) |

### Layout Transformations

- Input is ROW_MAJOR, output is TILE_LAYOUT. The compute kernel performs the tilize transformation.
- Width is padded to the nearest multiple of TILE_WIDTH (32) via the tile count: `ntiles_per_block = ceil(logical_width / 32)`.
- Height is implicitly tile-aligned because each block reads exactly TILE_HEIGHT (32) sticks.

---

## Data Flow Pattern (Input Stage Focus)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | **Reader** | DRAM (interleaved RM pages) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back` |
| 2 | Compute | CB c_0 (RM data) | CB c_16 (tiled data) | `cb_wait_front`, `tilize_block`, `cb_pop_front`, `cb_push_back` |
| 3 | Writer | CB c_16 | DRAM (interleaved tiles) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

### Reader Kernel Data Flow (Detailed)

The reader kernel (`reader_unary_stick_layout_split_rows_multicore.cpp`) reads row-major sticks from DRAM into the input CB. For each tile-row (a group of 32 RM sticks):

1. **Reserve CB space**: `cb_reserve_back(cb_id_in0, num_tiles_per_block)` -- reserves enough CB space for one block worth of tiles.
2. **Read 32 sticks**: Iterates `k = 0..TILE_HEIGHT-1` (32 iterations), issuing `noc_async_read` for each stick.
   - For non-sharded case: `num_pages_in_row = 1`, so the inner loop over `l` runs once per stick.
   - Uses `TensorAccessor::get_noc_addr(page_id)` to translate the logical page ID to a physical NoC address (handles round-robin DRAM bank distribution).
   - Each read transfers `block_width_size` bytes (= page_size = `logical_width * element_size`) into L1 at the current CB write pointer.
3. **Barrier**: `noc_async_read_barrier()` -- waits for all 32 stick reads to complete.
4. **Push**: `cb_push_back(cb_id_in0, num_tiles_per_block)` -- makes the data available to the compute kernel.

**Key insight for layer_norm_rm**: The reader reads 32 consecutive sticks (one tile-height worth) in a single `cb_reserve_back/cb_push_back` pair. The CB is sized to hold exactly `ntiles_per_block` tiles. This means the compute kernel receives a full tile-width strip of 32 rows at once.

### Stick-to-Tile Batching

- **32 sticks = 1 tile-row push**: The reader always reads exactly TILE_HEIGHT (32) sticks per block, regardless of the actual number of tiles in the width. This is because tilize needs 32 rows of data to form one row of tiles.
- The CB page size is `input_single_tile_size` (tile-sized pages), but the reader writes raw stick data contiguously. The compute kernel's tilize operation reinterprets this contiguous RM data as tile-format data.
- **The CB capacity is `ntiles_per_block` tiles**, meaning the entire width of the tensor (rounded up to tile boundaries) for 32 rows fits in one CB push.

---

## Circular Buffer Configuration (Input Stage Focus)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input RM sticks (to be tilized) | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output tiled data | `ntiles_per_block` tiles | `ntiles_per_block` tiles | Single | Compute | Writer | Block |

**CB c_0 Details**:
- Page size: `input_single_tile_size` = `tile_size(input_cb_data_format)` (e.g., 2048 bytes for BFLOAT16 with 32x32 tiles)
- Total size: `ntiles_per_block * input_single_tile_size`
- The CB is created with tile-sized pages even though the reader writes stick-format data. The tilize compute operation handles the format conversion.
- **Single-buffered**: capacity equals block size (`ntiles_per_block` = `ntiles_per_block`), so the reader and compute alternate rather than overlap.

---

## Pipeline Pattern Summary

Both CBs (c_0 and c_16) have capacity equal to their block size, making them **single-buffered**. The reader must complete writing 32 sticks (one block) before the compute kernel can begin processing, and the compute must finish before the reader can fill the CB again.

---

## Index Calculations

### Page ID Tracking

The reader kernel uses a simple linear page ID scheme:
- `start_page_id`: Set per-core by runtime args (the global page ID where this core starts reading).
- For each tile-row iteration: `page_id += tile_height * num_pages_in_row` advances past the 32 rows just read.
- For non-sharded interleaved: `num_pages_in_row = 1`, each page is one full tensor row. So page ID simply advances by 32 per tile-row.

### TensorAccessor Mapping

`TensorAccessor(src_tensor_args, src_addr, page_size)` is constructed with:
- `src_tensor_args`: Compile-time `TensorAccessorArgs<3>()` unpacked from the host-side `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)`.
- Given a page_id, `s.get_noc_addr(page_id)` returns the 64-bit NoC address by:
  1. Computing `bank_offset_index` from page_id (which bank slot).
  2. Computing `bank_index` from page_id (which DRAM bank, via round-robin).
  3. Combining bank base address + offset to get the local address.
  4. Looking up NoC XY coordinates for that bank.
  5. Constructing the full NoC address.

---

## Memory Access Patterns

### Read Pattern

- **Sequential with stride**: The reader iterates row-by-row through 32 consecutive RM pages (sticks). Each `noc_async_read` reads one full row (`block_width_size` bytes). Pages are logically sequential but physically scattered across DRAM banks (round-robin interleaving).
- **Burst of 32 reads + barrier**: All 32 stick reads are issued in a burst, then a single `noc_async_read_barrier()` ensures all complete. This amortizes NoC latency.
- **L1 write pattern**: Sticks are written contiguously in L1 starting at the CB write pointer. Successive sticks are appended at `l1_write_addr += width_size`.

### Write Pattern (De-emphasized)

The writer kernel reads one tile at a time from CB c_16 and writes to DRAM using TensorAccessor for address translation. Tiles are written sequentially by tile ID.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D |
| **Grid dimensions** | Uses `device->compute_with_storage_grid_size()` to get max available grid; work distributed linearly across cores |
| **Total cores** | `ncores` = `ceil(nblocks / nblocks_per_core)` (capped by available grid area) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows) for full cores; `nblocks_per_core_cliff` for the last (cliff) core |
| **Load balancing** | Near-equal: `nblocks_per_core = ceil(nblocks / grid_area)`. Last core gets remainder (`nblocks % nblocks_per_core`). |

### Work Splitting Detail (`split_blocks_for_tilize`)

1. Compute `grid_area` = total cores in available grid.
2. `nblocks_per_core = ceil(nblocks / grid_area)`.
3. `ncores = ceil(nblocks / nblocks_per_core)`.
4. `nblocks_per_core_cliff = nblocks % nblocks_per_core` (0 means no cliff core).
5. Full cores get `nblocks_per_core` blocks each. The last core (if cliff) gets the remainder.
6. Returns separate `CoreRangeSet` for full cores and cliff core, allowing different compute compile-time args.

### Per-Core State

Each core is assigned via runtime args:
- **Reader**: `page_start_id` = first RM page for this core; `num_rows` = `nblocks_per_core * TILE_HEIGHT` sticks to read.
- **Writer**: `tile_start_id` = first output tile for this core; `ntiles` = `ntiles_per_block * nblocks_per_core` tiles to write.
- Page/tile IDs advance sequentially across cores: core i starts where core i-1 ends.

---

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | page_size | uint32_t | Aligned page size of RM input buffer (= `aligned_page_size` from buffer) |
| 1 | num_pages_in_row | uint32_t | Number of pages per tensor row (1 for non-sharded interleaved) |
| 2 | size_of_valid_data_in_last_page_in_row | uint32_t | Bytes of valid data in last page (= `page_size` for non-sharded) |
| 3+ | TensorAccessorArgs | uint32_t[] | TensorAccessor configuration (bank layout, addressing mode) appended by `TensorAccessorArgs(*src0_buffer).append_to()` |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of source buffer in DRAM |
| 1 | num_rows | uint32_t | Total RM rows to read for this core (`nblocks_per_core * 32`) |
| 2 | page_size | uint32_t | Size of one RM page in bytes (redundant with CT arg but needed for sharded path compatibility) |
| 3 | num_tiles_per_block | uint32_t | Tiles per block = `ntiles_per_block` |
| 4 | block_width_size | uint32_t | Byte width of a block = `page_size` |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for non-sharded interleaved |
| 6 | num_leftover_tiles | uint32_t | Always 0 for default program factory |
| 7 | leftover_width_in_row | uint32_t | Always 0 for default program factory |
| 8 | start_page_id | uint32_t | First RM page ID for this core |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) for this core |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block = `ntiles_per_block` |

(Compute has no runtime arguments.)

### Writer Compile-Time Arguments (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB index (c_16 = 16) |
| 1+ | TensorAccessorArgs | uint32_t[] | Destination buffer accessor config |

### Writer Runtime Arguments (De-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Total tiles to write for this core |
| 2 | start_id | uint32_t | First output tile ID for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_multicore | RISCV_0 (BRISC) | NOC0 | DRAM (interleaved RM pages) | CB c_0 | Read 32 RM sticks per block into CB |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp`
- **Key Logic**:
  - Uses a lambda `read_tiles` that encapsulates the full block-read pattern: reserve CB, read 32 sticks, barrier, push.
  - TensorAccessor handles the page-to-NoC-address translation, abstracting DRAM bank interleaving.
  - Inner loop over `num_pages_in_row` is relevant only for ND-sharded tensors; for interleaved, it runs once.
  - The outer loop runs `num_rows / tile_height` iterations, each producing one block of `num_tiles_per_block` tiles in the CB.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize.cpp | RISCV_2 (TRISC) | N/A | CB c_0 | CB c_16 | Tilize RM data to tile format |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to configure unpack/pack for these CBs.
  - Delegates to `compute_kernel_lib::tilize<ntiles_per_block, c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(nblocks_per_core)`.
  - The tilize library function iterates over `num_blocks` blocks, each time waiting for `block_width_tiles` pages in c_0, then calling `tilize_block` or `fast_tilize_block`, pushing tiles to c_16, and popping input.
  - Fast tilize is auto-selected at compile time for BFLOAT16 + 32x32 tiles + half-sync dest mode.

### Writer Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 (NCRISC) | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write tiles one at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

---

## Implementation Notes

### Key Patterns for layer_norm_rm Reference

1. **RM Stick Reading Pattern**: The reader reads 32 consecutive RM sticks (one tile-height) per CB push. For a layer_norm_rm operation that also takes RM input, this same pattern of reading TILE_HEIGHT sticks, accumulating them in a CB, and then processing can be reused. However, layer_norm_rm may not need tilize -- it may process sticks directly if the compute is done in RM format.

2. **TensorAccessor Usage**: The host creates `TensorAccessorArgs(*buffer)` and appends to compile-time args. The kernel constructs `TensorAccessor(args, addr, page_size)` and calls `get_noc_addr(page_id)`. This is the standard pattern for reading interleaved DRAM tensors.

3. **CB Sizing for RM Input**: The input CB is sized as `ntiles_per_block * tile_size`, which holds exactly one full-width strip of 32 rows. For layer_norm_rm, the CB size should hold enough data for the normalization width (which is the last dimension).

4. **Page = Row for RM Interleaved**: One page equals one full row of the tensor (`logical_width * element_size` bytes). Page IDs are sequential from row 0 to row N-1.

5. **Work Distribution**: The 1D `split_blocks_for_tilize` function divides tile-row blocks across cores. For layer_norm_rm, a similar strategy could divide rows or groups of rows across cores.

6. **Separate Compute Kernel Variants**: The program factory creates two compute kernels when there's a cliff core -- one with `nblocks_per_core` and one with `nblocks_per_core_cliff` compile-time args, applied to different CoreRangeSets. This is necessary because compile-time args are baked into the kernel binary.

7. **FP32 Accumulation**: When `input dtype == FLOAT32`, `fp32_dest_acc_en` is set and `UnpackToDestMode::UnpackToDestFp32` is configured for CB c_0. This ensures lossless FP32 processing through the compute pipeline.

8. **Single-Buffered CBs**: Both CBs have capacity = block size, meaning no overlap between reader and compute. For better performance, a layer_norm_rm implementation could consider double-buffering the input CB.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal kernels? What is TensorAccessorArgs and how does get_noc_addr work with it?"
   **Reason**: The reader kernel uses `TensorAccessor` and `TensorAccessorArgs` to translate page IDs to physical DRAM NoC addresses. Understanding this mapping is essential for the reader pattern.
   **Key Findings**: TensorAccessor abstracts round-robin DRAM bank distribution. `get_noc_addr(page_id)` computes bank index via modulo, offset within bank via division, looks up bank's NoC XY coordinates, and constructs a 64-bit NoC address. `TensorAccessorArgs` packs buffer configuration (rank, bank count, shapes) into compile-time/runtime uint32_t vectors.

2. **Query**: "What are circular buffers in tt-metal? How do cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front work?"
   **Reason**: The reader-compute-writer pipeline is synchronized entirely through CB operations. Understanding these primitives is required to explain the data flow.
   **Key Findings**: CBs are L1-backed FIFO queues. `cb_reserve_back` blocks until space is available (producer side). `cb_push_back` signals data is ready. `cb_wait_front` blocks until data is available (consumer side). `cb_pop_front` frees consumed space. Capacity = num_pages * page_size determines buffering depth.

3. **Query**: "For row-major interleaved tensors stored in DRAM, what is a page?"
   **Reason**: The reader kernel reads RM pages from DRAM. Understanding page granularity is critical for the input_stage reference.
   **Key Findings**: For RM interleaved tensors, one page = one row of the tensor (width * element_size bytes). Pages are distributed round-robin across DRAM banks. Page ID i maps to bank (i % num_banks).

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to confirm page definitions for RM and tiled layouts.
   **Key Information**: RM tensors have one page per row. Tiled tensors have one page per tile. N-D tensors are flattened to 2D (outer dims collapsed). Interleaved layout distributes pages round-robin across banks.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: The compute kernel delegates to `compute_kernel_lib::tilize<>()`. Understanding the library's block processing loop reveals the compute-side CB interaction pattern.
   **Key Information**: The tilize library processes `num_blocks` iterations. In WaitBlock mode, each iteration does `cb_wait_front(input_cb, block_width_tiles)`, then `tilize_block` or `fast_tilize_block`, then `cb_push_back(output_cb, block_width_tiles)` and `cb_pop_front(input_cb, block_width_tiles)`. Fast tilize is auto-selected for BFLOAT16 + 32x32 + half-sync.

3. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: The `split_blocks_for_tilize` function determines how blocks are distributed across cores.
   **Key Information**: `nblocks_per_core = ceil(nblocks / grid_area)`. `ncores = ceil(nblocks / nblocks_per_core)`. Cliff core gets `nblocks % nblocks_per_core`. Returns separate CoreRangeSets for full cores and cliff core.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: The `create_cb` helper is used to create circular buffers.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with total size = `num_pages * page_size`. Sets page size per CB. Returns (cb_id, handle).
