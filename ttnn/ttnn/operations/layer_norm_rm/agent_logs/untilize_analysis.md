# Untilize Multi-Core Implementation Analysis

## Overview

The untilize operation converts tensor data from tiled layout (32x32 tiles with face-based internal ordering) to row-major layout (contiguous rows/"sticks"). This analysis covers the `UntilizeMultiCoreProgramFactory`, which is one of 8 program factory variants selected based on sharding configuration and core-grid constraints.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Analysis focus**: Output stage -- writer kernel pattern (how RM sticks are written to DRAM), output CB sizing, stick extraction from tiles. Reader kernel and compute internals are summarized but de-emphasized.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row of the tensor) |
| **Unit size** | `num_tiles_per_input_block` tiles (one tile-row width of the tensor) |
| **Total units** | `num_tiles_per_col` blocks (total tile-rows in the tensor) |
| **Loop structure** | Outer loop: blocks assigned to this core. Inner loop (writer): `tile_height` rows per block, each row iterated column-wise across output pages. |

A "block" here corresponds to one tile-row -- a horizontal strip of tiles spanning the tensor width (or shard width). For a tensor that is `H x W` in elements with tile dimensions `tile_height x tile_width`, there are `H / tile_height` blocks total and `W / tile_width` tiles per block.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (squeezed to 2D: `tensor_height x tensor_width`) | Same logical shape |
| **Dimension convention** | Last dim = W (inner), outer dims collapsed to height | Same |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles with 4 faces of 16x16) | ROW_MAJOR_LAYOUT (contiguous rows) |
| **Memory layout** | INTERLEAVED or SHARDED (height/width/block) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, UINT16, UINT32, INT32 | Same as input |

### Layout Transformations

The compute kernel performs the core transformation: it reads tiles from the input CB (tile-ordered data with face-based internal layout) and writes row-major data to the output CB. Each tile of `tile_height x tile_width` elements is linearized into `tile_height` contiguous rows of `tile_width` elements. For a block of `num_tiles_per_input_block` tiles, the output in the CB is a contiguous region of `tile_height` rows, each `num_tiles_per_input_block * tile_width` elements wide.

### Output Page Structure

The output tensor uses row-major layout, so each **page** corresponds to one row (stick) of the tensor's output "block width". For interleaved output, a page = one full tensor row of width `tensor_width`. For width-sharded or block-sharded output, a page = one shard row of width `output_page_width` (the shard's column count). This means a single tile-row block produces `tile_height` output pages, each of size `output_stick_size = output_page_width * output_element_size` bytes.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) or L1 (sharded) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front`, `pack_untilize_block` (or `untilize_block`), `cb_pop_front`, `cb_push_back` |
| 3 | Writer | CB c_16 (output) | DRAM (interleaved) or L1 (sharded) | `cb_wait_front`, `noc_async_write`, `cb_pop_front` |

### Step-by-step flow (focus: compute-to-writer-to-DRAM)

1. **Compute** produces one block of untilized data per iteration: `num_tiles_per_input_block` tiles are consumed from CB c_0 and `num_tiles_per_input_block` tiles worth of row-major data are pushed to CB c_16. The output CB now contains `tile_height` contiguous rows, each `num_cols_per_input_block` elements wide.

2. **Writer** calls `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` to wait for the block.

3. **Writer** obtains `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- the L1 address of the first byte of the untilized block.

4. **Writer** iterates `j = 0..tile_height-1` over each row within the block:
   - Computes `current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size` to address each row within the block's contiguous row-major output.
   - Computes the `output_page_id` for this row based on the block's height index and the width-wise output block index.
   - Iterates column-wise across the row, writing chunks to output pages. For each chunk:
     - Determines `num_cols_to_write = min(remaining input cols, remaining output page cols)`
     - Gets the DRAM NoC address via `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)`
     - Calls `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`
     - Advances the L1 read pointer and output page pointer

5. **Writer** calls `noc_async_write_barrier()` after all rows of the block, then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | Interleaved: `num_tiles_per_input_block * 2` tiles (or 1x if single block). Sharded: entire shard. | `num_tiles_per_input_block` tiles | See below | Reader | Compute | Block |
| c_16 | cb_output | Output row-major staging | `num_tiles_per_input_block * 2` tiles (or 1x if single block) | `num_tiles_per_input_block` tiles | See below | Compute | Writer | Block |

### Output CB Sizing (Key for output_stage reference)

The output CB (c_16) capacity is determined by `output_cb_num_tiles`:

```
if (num_input_blocks_per_full_core == 1):
    output_cb_num_tiles = num_tiles_per_input_block          // Single-buffered
else:
    output_cb_num_tiles = num_tiles_per_input_block * 2      // Double-buffered
```

**Key insight for downstream use**: One "tile" in the output CB is `output_single_tile_size` bytes (the tile size for the output data format). Even though the data inside is row-major, the CB capacity is still measured in tile-sized pages. Each block produces `num_tiles_per_input_block` tiles worth of output data, arranged as `tile_height` contiguous rows of `num_cols_per_input_block` elements.

The actual byte capacity of the output CB is `output_cb_num_tiles * output_single_tile_size`.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type |
|----|----------|------------|----------------|
| c_0 (input, interleaved, multi-block) | 2x block | 1x block | Double-buffered |
| c_0 (input, interleaved, single-block) | 1x block | 1x block | Single-buffered |
| c_0 (input, sharded) | entire shard | entire shard | Single-buffered (all data resident) |
| c_16 (output, multi-block) | 2x block | 1x block | Double-buffered |
| c_16 (output, single-block) | 1x block | 1x block | Single-buffered |

Double-buffering on the output CB allows the compute kernel to produce the next block of untilized rows while the writer is still writing the previous block to DRAM.

## Index Calculations

### Writer Index Mapping (Critical for output_stage reference)

The writer must map from untilized block data in L1 to the correct output pages in DRAM. The key variables are:

1. **`height_wise_input_block_start_index`**: The global block-row index where this core starts processing. For core `i`: `(i / num_input_blocks_across_width) * num_input_blocks_per_full_core`.

2. **`width_wise_output_block_start_index`**: Which output page column this core writes to. Computed as `input_block_global_col_index / num_cols_per_output_block`.

3. **`output_page_id`**: For row `j` within block at height index `block_height_index`:
   ```
   num_rows_already_processed = block_height_index * tile_height + j
   output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index
   ```
   This maps (block_height_index, row_within_block, width_block_index) to a flat page ID in the output tensor.

4. **`output_offset_within_page_in_bytes`**: Non-zero only for the first output page per row, when the input block does not start at a page boundary (relevant for width/block sharding). Equals `num_cols_already_processed_in_first_output_block * output_element_size`.

5. **L1 read offset for row `j`**: `base_l1_read_addr + j * num_cols_per_input_block * output_element_size`. This is a direct linear offset into the contiguous row-major data produced by the compute kernel.

### TensorAccessor Usage

The writer creates a `TensorAccessor` from compile-time args starting at index 8:
```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
```

The `TensorAccessor` maps `(page_id, offset_within_page)` to a physical NoC address in DRAM, handling the round-robin interleaved bank distribution. `output_stick_size` is the page size (one full output row).

## Memory Access Patterns

### Read Pattern (de-emphasized)

The reader reads tiles sequentially by tile ID from interleaved DRAM, one tile at a time with barriers per tile. For sharded input, data is already in L1 via the globally-allocated CB.

### Write Pattern (Key for output_stage reference)

**Pattern**: Row-by-row within each block, column-wise across output pages within each row.

For each block:
- **Outer loop**: `tile_height` iterations (one per element-row within the tile-row)
- **Inner loop**: Column-wise iteration across output pages. For interleaved output with no width sharding, this is a single write per row (entire row = one page). For width/block-sharded output, the row may span multiple output pages.

**Write granularity**: Each `noc_async_write` transfers `num_cols_to_write * output_element_size` bytes, which is at most one output page width.

**Barriers**: `noc_async_write_barrier()` is called once per block (after all `tile_height` rows), NOT per row. This means all row writes within a block are batched before the barrier.

**Access pattern for interleaved output (common case)**:
- Each row write is to a different DRAM bank (round-robin interleaving by page)
- Rows within a block are to consecutive page IDs
- Blocks are to consecutive groups of `tile_height` pages
- This creates a stride pattern: pages `[start, start + tile_height)` per block, with blocks advancing by `tile_height`

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Derived from `device->compute_with_storage_grid_size()` (interleaved) or shard spec grid (sharded) |
| **Total cores** | `num_compute_cores` from `split_blocks_for_tilize` |
| **Work per core** | `num_input_blocks_per_full_core` blocks (tile-rows) for full cores; `num_input_blocks_per_cliff_core` for cliff core |
| **Load balancing** | Near-equal with one cliff core handling the remainder |

### Work Splitting (Interleaved Input)

`split_blocks_for_tilize(grid_size, num_tiles_per_col)` divides the total tile-rows across available cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- Full cores get `nblocks_per_core` blocks each
- One cliff core (if remainder exists) gets `num_tiles_per_col % nblocks_per_core` blocks

### Work Splitting (Sharded Input)

Each shard maps to one core. The `num_input_blocks_per_full_core = shard_height / tile_height`. The last height-wise shard may process fewer blocks if the tensor height is not evenly divisible by shard height.

## Arguments

### Compile-Time Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Bytes per output page/stick: `output_page_width * output_element_size` |
| 2 | `tile_height` | uint32_t | Height of one tile in elements (typically 32) |
| 3 | `num_tiles_per_input_block` | uint32_t | Number of tiles per block (tiles per row) |
| 4 | `num_output_blocks_across_width` | uint32_t | Number of output pages per tensor row (1 for interleaved, >1 for width/block sharded output) |
| 5 | `output_element_size` | uint32_t | Bytes per output element (2 for BF16, 4 for FP32) |
| 6 | `num_cols_per_input_block` | uint32_t | Elements per input block row: `num_tiles_per_input_block * tile_width` |
| 7 | `num_cols_per_output_block` | uint32_t | Elements per output page: `output_page_width` |
| 8+ | TensorAccessor args | various | Bank mapping info for the output buffer via `TensorAccessorArgs(*dst_buffer)` |

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_in0` | uint32_t | Input CB index (c_0) |
| 1+ | TensorAccessor args | various | Bank mapping info for the input buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | `num_tiles_per_input_block` -- tiles per block |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of blocks (tile-rows) this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Global block-row index where this core starts |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Actual (non-padding) columns to write per block row |
| 4 | `width_wise_output_block_start_index` | uint32_t | Which output page column this core starts writing at |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Column offset within the first output page (for partial-page writes) |

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Input buffer base address |
| 1 | `num_tiles_to_read` | uint32_t | Total tiles to read on this core |
| 2 | `tile_start_index` | uint32_t | First tile ID to read |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to untilize (= `num_input_blocks_to_process`) |

## Kernel Implementations

### Writer: `writer_unary_stick_layout_split_rows_multi_core.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (L1) | DRAM (interleaved) | Write RM sticks from untilized output CB to DRAM pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - The `write_tiles_in_current_block` lambda encapsulates the per-block write logic.
  - For each block, waits for `num_tiles_per_input_block` tiles in the output CB.
  - Gets `base_l1_read_addr` from `get_read_ptr(cb_id_out0)` -- this points to the contiguous row-major data.
  - For each of the `tile_height` rows, computes the L1 offset as `j * num_cols_per_input_block * output_element_size`.
  - The inner while loop handles the case where one input block row spans multiple output pages (width/block sharding) by splitting each row write across page boundaries.
  - Uses `TensorAccessor::get_noc_addr(page_id, offset_within_page)` for sub-page addressing.
  - Calls `noc_async_write_barrier()` once per block (not per row), then `cb_pop_front`.
  - The `num_unpadded_cols_per_input_block` runtime arg handles padding: only unpadded columns are written, even if the CB contains a full tile-width of data.

### Compute: `untilize_variable_num_blocks.cpp` / `pack_untilize_variable_num_blocks.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (unpack + math + pack) | N/A | CB c_0 | CB c_16 | Untilize: tile-to-RM conversion |

- **File (slow path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **File (fast path)**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Both call `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt)`.
  - The unified `untilize` helper (in `untilize_helpers.inl`) handles two paths:
    - **Single-pass pack untilize**: When `block_width_tiles <= DEST_AUTO_LIMIT` (typically 8 tiles). Reads entire block into DEST, pack-untilizes in one shot (80 cycles).
    - **Block-based pack untilize**: When `block_width_tiles > DEST_AUTO_LIMIT`. Splits the block into sub-blocks of width `sub_block_width <= DEST_AUTO_LIMIT`, processing each sub-block sequentially (390 cycles for the slow path).
  - The slow path is selected when `use_pack_untilize` is false, data type is UINT16, or data type is FLOAT32 with width >= `MAX_PACK_UNTILIZE_WIDTH`.
  - CB synchronization in the compute kernel: For WaitBlock mode (default), `cb_wait_front(input_cb, block_width_tiles)` waits for one block of input tiles, then `cb_reserve_back(output_cb, block_width_tiles)` reserves output space.

### Reader: `reader_unary_start_id.cpp` (de-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved) | CB c_0 | Read tiles sequentially by ID |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- Reads tiles one at a time in a simple loop from `start_page_id` to `start_page_id + num_tiles`.

## Implementation Notes

### Output Stage Design Patterns (Key for layer_norm_rm reference)

1. **Stick-by-stick writing from untilized CB**: The writer reads row-major data from the output CB using `get_read_ptr` and computes byte offsets for each row. This pattern is directly reusable for any operation that produces row-major output in a CB and needs to write it to interleaved DRAM.

2. **Page ID calculation for RM output**: `output_page_id = row_index * num_pages_per_row + column_page_index`. For interleaved output where one page = one tensor row, this simplifies to `output_page_id = global_row_index`.

3. **Sub-page writes via offset**: The `get_noc_addr(page_id, offset)` API enables writing to arbitrary byte offsets within a page. This is used when input blocks are narrower than output pages (width/block sharding scenarios).

4. **Barrier per block, not per row**: All `tile_height` row writes within a block share a single `noc_async_write_barrier()`, amortizing barrier overhead. This is a performance optimization -- as long as write addresses do not overlap, NoC writes can be pipelined.

5. **Padding handling via `num_unpadded_cols_per_input_block`**: The writer only writes the unpadded portion of each row, even though the compute kernel produces a full tile-width of output. This is a runtime arg, not compile-time, because padding depends on tensor dimensions.

6. **Output CB sizing for double-buffering**: The output CB holds `2 * num_tiles_per_input_block` tiles when the core processes more than one block, enabling compute-writer overlap. For a new operation, the architect should size the output CB to hold at least 2 blocks worth of RM output data to enable this pipelining.

### Compute Path Selection

The program factory selects between slow and fast untilize at program creation time based on `use_pack_untilize`, data type, and tile-row width. The `DST_ACCUM_MODE` define is set for INT32, UINT32, and FLOAT32 data types, which halves the DEST register capacity (from 8 to 4 tiles max per sub-block).

### `untilize_helpers.hpp` Signature and Usage

The `compute_kernel_lib::untilize` helper has the following signature:

```cpp
template <
    uint32_t block_width_tiles,   // tiles per row (compile-time)
    uint32_t input_cb,            // input circular buffer index
    uint32_t output_cb,           // output circular buffer index
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitBlock,
    ReconfigureRegisterDatatypeMode reconfig_mode = NoReconfigure>
ALWI void untilize(uint32_t num_blocks);  // num_blocks = runtime
```

For use in a new operation that needs an untilize stage:
- Set `block_width_tiles` to the number of tiles per row of the region being untilized.
- Use `WaitBlock` (default) for block-by-block synchronization with the reader.
- Use `WaitUpfront` if all tiles are already available (e.g., from a sharded CB).
- Use `InitOnly`/`Neither`/`UninitOnly` modes to chain multiple untilize calls without redundant init/uninit overhead.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal kernels? What is TensorAccessorArgs and how does get_noc_addr work for mapping page IDs to physical DRAM addresses?"
   **Reason**: Needed to understand how the writer kernel maps output page IDs to physical DRAM bank addresses.
   **Key Findings**: `TensorAccessor` abstracts interleaved bank distribution. Created on device from `TensorAccessorArgs<offset>()` with base address and page size. `get_noc_addr(page_id)` maps logical page to physical bank+offset. `get_noc_addr(page_id, byte_offset)` supports sub-page writes. Zero-cost construction if rank is static.

2. **Query**: "How does the untilize operation work in tt-metal? What is the difference between pack_untilize (fast path) and standard untilize (slow path)?"
   **Reason**: Needed to understand the compute kernel's tile-to-RM conversion mechanism.
   **Key Findings**: Fast path (`pack_untilize`) directly converts from DEST to RM in ~80 cycles when width <= DEST limit (8 tiles). Slow path goes through unpack -> datacopy -> pack in ~390 cycles. The program factory selects between them at kernel creation time.

3. **Query**: "What is the split_blocks_for_tilize function? How does it distribute tile rows across cores? What is a cliff core?"
   **Reason**: Needed to understand how work is distributed across cores.
   **Key Findings**: `split_blocks_for_tilize(grid_size, num_blocks)` divides blocks equally, with one "cliff core" handling the remainder. Returns `BlockSplit` with `core_range` (full cores), `cliff_core_range`, `nblocks_per_core`, and `nblocks_per_core_cliff`.

4. **Query**: "What is the get_noc_addr overload that takes a page_id and an offset_within_page parameter?"
   **Reason**: The writer kernel uses `s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` for sub-page writes.
   **Key Findings**: Signature is `get_noc_addr(page_id, offset = 0, noc = noc_index)`. The offset is added to the computed bank address, enabling writes to arbitrary byte positions within a page. Essential for width/block sharding where input blocks may not align with output page boundaries.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the difference between tiled and row-major page structures.
   **Key Information**: In row-major layout, each page = one tensor row. In tiled layout, each page = one 32x32 tile (with 4 faces of 16x16 internally). Interleaved distribution places pages round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side and device-side setup.
   **Key Information**: Host-side `TensorAccessorArgs(buffer)` extracts bank mapping. Device-side `TensorAccessorArgs<CTA_OFFSET>()` reads compile-time args. `TensorAccessor(args, addr, page_size)` creates the accessor. `get_noc_addr(page_id, offset)` is the primary API for address resolution.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the unified untilize compute helper API.
   **Key Information**: Template-parametrized function with compile-time block width, CB indices, and mode enums. Automatically selects single-pass vs block-based untilize based on DEST capacity. Provides `WaitBlock`, `WaitUpfront`, and `NoWait` synchronization modes.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding `split_blocks_for_tilize` work distribution.
   **Key Information**: `compute_ncores(grid_area, nblocks)` computes `nblocks_per_core` and total `ncores`. The cliff core gets `nblocks % nblocks_per_core` blocks. Returns `BlockSplit` struct with core ranges.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used for CB allocation.
   **Key Information**: `create_cb(cb_id, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB. If `buffer != nullptr`, CB is globally allocated (backed by the buffer's memory, used for sharded inputs). Returns `(cb_index, cb_handle)`.
