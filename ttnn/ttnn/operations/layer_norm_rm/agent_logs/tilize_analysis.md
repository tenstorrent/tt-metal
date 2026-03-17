# Tilize (Multi-Core Default) Implementation Analysis

## Overview

The **tilize** operation converts a row-major (RM) tensor on device into tiled (TILE_LAYOUT) format. The input tensor has one page per stick (row), and the output tensor has one page per 32x32 tile. The compute kernel performs the actual data reordering from row-major sticks into the hardware-native tile format (four 16x16 faces per tile).

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_default_program_factory.cpp`

**Focus of this analysis**: Reader kernel pattern, input CB sizing, stick-to-tile batching, and core assignment strategy, for use as an input-stage reference when building a `layer_norm_rm` operation that reads RM sticks from DRAM.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (= one tile-row of the tensor) |
| **Unit size** | `ntiles_per_block` tiles = `ceil(logical_width / TILE_WIDTH)` tiles |
| **Total units** | `nblocks` = `ceil(total_output_tiles / ntiles_per_block)` = number of tile-rows |
| **Loop structure** | Outer loop over tile-rows assigned to this core; inner loop over full blocks in each row (always 1 in the interleaved non-sharded case) |

A "block" is a horizontal strip of tiles spanning the full width of the tensor at tile height (32 rows). Each block contains `ntiles_per_block` output tiles. Blocks are the unit of work distribution across cores.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D, e.g. `[N, C, H, W]` |
| **Dimension convention** | Last dim is width (stick length) |
| **Tensor layout** | `ROW_MAJOR_LAYOUT` |
| **Memory layout** | `INTERLEAVED` (default path; sharded also supported) |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page definition** | 1 page = 1 stick (one row of width W) |
| **Page size** | `logical_width * element_size` bytes |
| **Aligned page size** | page_size rounded up for DRAM alignment |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (may be padded to tile alignment) |
| **Tensor layout** | `TILE_LAYOUT` |
| **Memory layout** | `INTERLEAVED` |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or specified `output_dtype`) |
| **Page definition** | 1 page = 1 tile (32x32 elements) |

### Key Insight for layer_norm_rm

For a row-major interleaved input tensor:
- **page_size** = `width * element_size_bytes` (one stick)
- **aligned_page_size** = page_size rounded up for HW alignment
- Pages are distributed round-robin across DRAM banks
- `TensorAccessor.get_noc_addr(page_id)` maps a linear page ID to the correct bank and offset

---

## Data Flow Pattern (Reader Focus)

### Step-by-Step Flow

1. **Reader kernel** reads row-major sticks from DRAM into `CB c_0` (input CB).
   - Reserves space for one block's worth of tiles (`ntiles_per_block` tiles).
   - Reads 32 sticks (= `TILE_HEIGHT` rows) one at a time via `noc_async_read`, writing them contiguously into CB space.
   - Issues `noc_async_read_barrier()` then `cb_push_back` to signal data is ready.

2. **Compute kernel** reads row-major data from `CB c_0` and tilizes it into `CB c_16` (output CB).
   - Uses `tilize_block()` / `fast_tilize_block()` to reorder row-major bytes into tile format (four 16x16 faces).
   - Operates one block at a time: `cb_wait_front` on input, `cb_reserve_back` on output, process, `cb_push_back` output, `cb_pop_front` input.

3. **Writer kernel** reads tiled data from `CB c_16` and writes tiles to DRAM.

### Reader `read_tiles` Lambda (the critical pattern)

```cpp
auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
    cb_reserve_back(cb_id_in0, num_tiles);         // Reserve ntiles_per_block tiles of space
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t k = 0; k < tile_height; k++) {   // 32 rows per tile-row
        for (uint32_t l = 0; l < num_pages_in_row; l++) {  // 1 for non-sharded
            uint64_t src_noc_addr = s.get_noc_addr(page_id);
            page_id++;
            uint32_t width_size = (l == num_pages_in_row - 1)
                ? size_of_valid_data_in_last_page_in_row : block_width_size;
            noc_async_read(src_noc_addr, l1_write_addr, width_size);
            l1_write_addr += width_size;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, num_tiles);            // Signal: block is ready
};
```

**Key observations for layer_norm_rm**:
- The reader reserves `ntiles_per_block` pages of CB space but fills it with **32 sequential sticks** (not tiles).
- The CB page size is set to `input_single_tile_size` (tile size in bytes), but the reader treats the CB as flat memory and writes sticks contiguously, end-to-end. The CB does not enforce any tile structure -- it is just a byte buffer.
- For non-sharded interleaved tensors: `num_pages_in_row = 1`, `block_width_size = page_size`, so the inner loop over `l` executes once per stick.
- After 32 sticks are written, the reader calls `cb_push_back(cb_id_in0, ntiles_per_block)`. This means the compute kernel sees `ntiles_per_block` "pages" of tile-sized data containing the 32 raw sticks.

### Stick-to-Tile Batching

- **32 sticks** are read per `cb_reserve_back` / `cb_push_back` cycle.
- These 32 sticks fill exactly one tile-row (one block of `ntiles_per_block` tiles in height dimension).
- The total bytes written per batch = `32 * page_size` = `32 * width * element_size`.
- The CB capacity is `ntiles_per_block * tile_size` bytes, which equals `ceil(width/32) * 32 * 32 * element_size`. For a width that is tile-aligned, this equals exactly `32 * width * element_size`, matching the 32-stick batch.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Total Size | Buffering | Producer | Consumer |
|-------|------|---------|-------------------|-----------|------------|-----------|----------|----------|
| `c_0` | cb_input | Input staging (RM sticks) | `ntiles_per_block` | `input_single_tile_size` | `ntiles_per_block * tile_size` | Single | Reader | Compute |
| `c_16` | cb_output | Output staging (tiles) | `ntiles_per_block` | `output_single_tile_size` | `ntiles_per_block * tile_size` | Single | Compute | Writer |

### Input CB (c_0) Details -- Critical for layer_norm_rm

- **Capacity**: `ntiles_per_block` pages, where each page = `tile_size` bytes (e.g., 2048 bytes for BF16 32x32 tiles).
- **Actual usage**: The reader writes 32 sticks into this space. The total bytes = `32 * width * element_size`. When width is tile-aligned (divisible by 32), this exactly equals `ntiles_per_block * tile_size`.
- **Buffering**: Single-buffered. The capacity equals exactly one block (one tile-row). The reader writes one block, the compute consumes it, then the reader writes the next.
- **The CB page size is tile_size even though sticks are being stored**. This is because the compute kernel's `tilize_block` expects to consume `ntiles_per_block` "tiles" from the CB. The CB bookkeeping counts in pages of `tile_size`, but the actual data layout in the buffer is 32 sequential sticks.

### Implication for layer_norm_rm

For a reader that reads RM sticks and feeds them to a compute kernel that tilizes internally:
- Set CB page_size = tile_size (not stick size)
- Set CB num_pages = `Wt` (tiles per row) for single buffering, or `2 * Wt` for double buffering
- Reader fills `32 * W * element_size` bytes per `reserve_back` / `push_back` cycle
- `cb_reserve_back(cb, Wt)` reserves space for `Wt` tile-pages = one tile-row of raw stick data
- `cb_push_back(cb, Wt)` signals that one tile-row is ready for the compute kernel

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Sequential stick reads with per-stick NoC transactions.
- **Granularity**: One stick (one page = one full tensor row) per `noc_async_read` call.
- **Addressing**: `TensorAccessor.get_noc_addr(page_id)` maps sequential page IDs to interleaved DRAM banks (round-robin).
- **Batch size**: 32 sticks are read before a barrier, then pushed to CB.
- **Stride**: `page_id` increments by 1 for each stick within a tile-row. Between tile-rows (blocks), `page_id += tile_height * num_pages_in_row` (= 32 for non-sharded).
- **No reuse**: Each stick is read exactly once.

### Write Pattern (Writer Kernel -- brief)

- Tiles are written one-at-a-time from the output CB to DRAM using `noc_async_write_page`.
- Sequential tile IDs starting from `tile_start_id`.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from available 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` = `ceil(nblocks / nblocks_per_core)` |
| **Work per core** | `nblocks_per_core` blocks (tile-rows) per full core |
| **Cliff core** | Last core may get `nblocks_per_core_cliff = nblocks % nblocks_per_core` blocks |
| **Load balancing** | Nearly equal: `nblocks_per_core = ceil(nblocks / grid_area)`, remainder on cliff |

### Work Splitting Function: `split_blocks_for_tilize`

```cpp
auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(available_grid, nblocks);
```

This function:
1. Computes `nblocks_per_core = ceil(nblocks / grid_area)`.
2. Computes `ncores = ceil(nblocks / nblocks_per_core)`.
3. If `nblocks % nblocks_per_core != 0`, the last core is a "cliff" core with fewer blocks.
4. Returns separate `CoreRangeSet` objects for full cores and cliff core.

### Per-Core Runtime Args Setup

For each core `i` (out of `ncores_full` full cores):
- `page_start_id = i * nblocks_per_core * TILE_HEIGHT * num_pages_in_row`
  - This is the first RM page (stick) this core should read.
- `num_rows = nblocks_per_core * TILE_HEIGHT`
  - Total number of sticks this core reads (number of blocks * 32 rows/block).
- `tile_start_id = i * ntiles_per_block * nblocks_per_core`
  - First output tile ID for the writer.

---

## Arguments

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `page_size` | uint32_t | Aligned page size of input buffer (= aligned stick size) |
| 1 | `num_pages_in_row` | uint32_t | Pages per tensor row (1 for non-sharded, >1 for ND-sharded) |
| 2 | `size_of_valid_data_in_last_page_in_row` | uint32_t | Bytes of valid data in last page (handles padding for sharded) |
| 3+ | TensorAccessor args | multiple uint32_t | Bank mapping, memory type, tensor shape info (appended by `TensorAccessorArgs`) |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Base address of input buffer |
| 1 | `num_rows` | uint32_t | Total sticks to read = `nblocks_per_core * 32` |
| 2 | `page_size` | uint32_t | Raw page size (stick size in bytes) |
| 3 | `num_tiles_per_block` | uint32_t | `ntiles_per_block` = `ceil(width / 32)` |
| 4 | `block_width_size` | uint32_t | `page_size` (bytes per stick for NoC read) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for interleaved non-sharded |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 in this factory |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 in this factory |
| 8 | `start_page_id` | uint32_t | First page (stick) index for this core |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | `nblocks_per_core` (or `nblocks_per_core_cliff` for cliff core) |
| 1 | `per_core_block_tile_cnt` | uint32_t | `ntiles_per_block` = tiles per block |

---

## Kernel Implementations

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp` |
| **Core** | RISCV_0 (data movement) |
| **NOC** | NOC0 (reader default) |
| **Input** | DRAM (interleaved RM pages) |
| **Output** | CB c_0 (input staging) |
| **Operations** | `cb_reserve_back`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back` |

**Key Logic**:
- Creates a `TensorAccessor` from compile-time args (index 3+) and runtime `src_addr` + `page_size`.
- Outer loop: `num_rows / tile_height` iterations (= number of blocks for this core).
- Inner loop per block: reads 32 sticks via individual `noc_async_read` calls.
- Each stick read uses `s.get_noc_addr(page_id)` for bank-aware addressing.
- Barrier after all 32 sticks, then `cb_push_back` to signal compute.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` |
| **Core** | Compute threads (unpack + math + pack) |
| **Input** | CB c_0 |
| **Output** | CB c_16 |
| **Operations** | `compute_kernel_lib::tilize<>` helper |

**Key Logic**:
- Uses `compute_kernel_hw_startup(c_0, c_16)` for HW initialization.
- Calls `compute_kernel_lib::tilize<ntiles_per_block, c_0, c_16>(nblocks_per_core)`.
- The helper handles: `cb_wait_front` on c_0, `cb_reserve_back` on c_16, `tilize_block` / `fast_tilize_block`, `cb_push_back` on c_16, `cb_pop_front` on c_0.
- Processes `nblocks_per_core` blocks, each with `ntiles_per_block` tiles.

### Writer Kernel (brief)

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Core** | RISCV_1 (data movement) |
| **NOC** | NOC1 (writer default) |
| **Input** | CB c_16 |
| **Output** | DRAM (interleaved tile pages) |

---

## Implementation Notes

### TensorAccessor Pattern for RM Interleaved Reads

The `TensorAccessorArgs` is constructed on the host from the input buffer:
```cpp
TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
```
This appends all bank mapping information (number of banks, bank coordinates, tensor shape in pages, etc.) as compile-time arguments. The default `ArgConfig::None` makes everything compile-time.

On the device, the accessor is constructed:
```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<3>();  // CT args start at index 3
const auto s = TensorAccessor(src_tensor_args, src_addr, page_size);
```

Then `s.get_noc_addr(page_id)` maps a linear page ID to the physical NoC address of that page in the correct interleaved DRAM bank.

### CB Sizing Rationale

The input CB capacity of `ntiles_per_block` tile-pages is exactly enough for 32 sticks when width is tile-aligned. This is because:
- `ntiles_per_block = ceil(W / 32)`
- CB total bytes = `ntiles_per_block * tile_size` = `ceil(W/32) * 32 * 32 * elem_size`
- 32 sticks = `32 * W * elem_size`
- When `W % 32 == 0`: `ceil(W/32) * 32 = W`, so both equal `32 * W * elem_size`.

### FP32 Handling

When `a.dtype() == DataType::FLOAT32`:
- `fp32_llk_acc` is set to true
- Compute config enables `.fp32_dest_acc_en = true`
- Unpack mode for CB c_0 is set to `UnpackToDestFp32` for lossless tilize (standard path, not fast_tilize)

### Single-Buffering Observation

Both input and output CBs are single-buffered (capacity = 1 block). This means:
- Reader writes one block, then waits for compute to consume it before writing the next.
- Compute writes one block of output, then waits for writer to consume it.
- No overlap between reader and compute, or between compute and writer.
- For a new operation, double-buffering (capacity = 2 * ntiles_per_block) would allow overlap.

---

## Design Patterns Applicable to layer_norm_rm

### Pattern 1: Reading RM Sticks into a Tile-Sized CB

The tilize reader demonstrates the canonical pattern for reading row-major sticks from DRAM interleaved storage into a CB that will be consumed by a compute kernel expecting tile-sized pages:

1. **CB setup**: `create_cb(c_0, program, cores, tile_size, Wt, data_format)` -- page_size = tile_size, num_pages = Wt (tiles per row).
2. **Reader**: `cb_reserve_back(c_0, Wt)` then read 32 sticks into the reserved space as flat bytes, then `cb_push_back(c_0, Wt)`.
3. **Compute**: `cb_wait_front(c_0, Wt)` consumes Wt "tile-pages" that actually contain 32 row-major sticks, then tilizes.

### Pattern 2: TensorAccessor for Interleaved Page Reads

```cpp
// Host side:
TensorAccessorArgs(*buffer).append_to(ct_args);

// Kernel side:
constexpr auto args = TensorAccessorArgs<CT_ARG_START_INDEX>();
const auto accessor = TensorAccessor(args, base_addr, page_size);
uint64_t noc_addr = accessor.get_noc_addr(page_id);
noc_async_read(noc_addr, l1_write_addr, read_size);
```

### Pattern 3: Block-Based Work Distribution

```cpp
auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
    ttnn::split_blocks_for_tilize(available_grid, nblocks);
```

Each core gets a contiguous range of blocks. Runtime args encode the start page/tile ID and count for each core.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in TTNN? Specifically, how does it convert row-major sticks to tiled format?"
   **Reason**: Needed to understand the fundamental block/tile-row relationship.
   **Key Findings**: A block = one tile-row; nblocks = total tile-rows. Reader writes sticks, compute reorders to tiles.

2. **Query**: "What is the create_cb utility function and how does CircularBufferConfig work?"
   **Reason**: Needed to understand CB capacity calculation from page_size and num_pages.
   **Key Findings**: CB total_size = page_size * num_pages. The `create_cb` helper wraps `CircularBufferConfig` construction and `CreateCircularBuffer`.

3. **Query**: "For a row-major interleaved tensor, what does page mean? How is page_size calculated?"
   **Reason**: Critical for understanding how the reader addresses DRAM.
   **Key Findings**: For RM tensors, 1 page = 1 row (stick). page_size = width * element_size. aligned_page_size is page_size rounded up for HW alignment.

4. **Query**: "How does cb_reserve_back and cb_push_back work? For the tilize reader, how does the CB handle non-tile-sized pages?"
   **Reason**: Needed to understand how the reader writes sticks into a tile-sized CB.
   **Key Findings**: CB is just a byte buffer. The reader writes sticks contiguously; the CB doesn't enforce tile structure. cb_reserve_back/cb_push_back count in pages (set by page_size config), but the actual data layout is determined by the kernel.

5. **Query**: "In tilize, the input CB is created with tile_size as page size but the reader writes RM sticks. How does this work?"
   **Reason**: This was the most critical question for understanding the input stage.
   **Key Findings**: The CB is agnostic to data layout. Reader fills it with raw row-major sticks. The compute kernel's `tilize_block` reads row-major data from the input CB and reorders it into tile format in the output CB. The CB page_size=tile_size is just for bookkeeping to match the compute kernel's expectation of consuming `ntiles_per_block` "pages".

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor API for reading interleaved pages.
   **Key Information**: `TensorAccessorArgs` on host appends bank mapping to compile-time args. On device, `TensorAccessor(args, addr, page_size)` + `get_noc_addr(page_id)` provides bank-aware addressing.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page definition and interleaved layout.
   **Key Information**: RM layout: 1 page = 1 row. Interleaved: pages round-robin across banks. Tile: 32x32 with 16x16 faces.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize helper.
   **Key Information**: `compute_kernel_lib::tilize<block_width_tiles, input_cb, output_cb>(num_blocks)` handles the full tilize loop with CB synchronization. Supports fast_tilize for BF16 with 32x32 tiles.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility.
   **Key Information**: Wraps `CircularBufferConfig(num_pages * page_size, data_format_spec)` with `.set_page_size(cb, page_size)` and calls `CreateCircularBuffer`.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution.
   **Key Information**: Computes `nblocks_per_core = ceil(nblocks / grid_area)`, `ncores = ceil(nblocks / nblocks_per_core)`, cliff core gets remainder.
