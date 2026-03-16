# Tilize (Multi-Core Default) Implementation Analysis -- Input Stage Reference

## Overview

The **tilize** operation converts a row-major tensor into tiled (32x32) format. This analysis focuses on the **reader kernel pattern**, **input CB sizing**, **stick-to-tile batching**, and **core distribution strategy** as an input_stage reference for a new `layer_norm_rm` operation that reads row-major interleaved data from DRAM.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_default_program_factory.cpp`

### Relevance to layer_norm_rm

The tilize reader kernel demonstrates the canonical pattern for reading row-major interleaved sticks from DRAM into L1 circular buffers. The `layer_norm_rm` operation needs a similar reader: fetch RM sticks from DRAM, accumulate enough sticks to form a tile-height batch (32 rows), and push them to a CB for compute consumption. The tilize operation's 1D block-based core distribution and TensorAccessor usage are directly applicable.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles spanning the tensor width at tile_height=32 rows) |
| **Unit size** | `ntiles_per_block` tiles (= `ceil(logical_width / TILE_WIDTH)`) |
| **Total units** | `nblocks` = `ceil(total_output_tiles / ntiles_per_block)` = number of tile-rows |
| **Loop structure** | Outer: iterate over blocks (tile-rows). Inner: iterate over full blocks in a row (always 1 for non-sharded interleaved). Per block: 32 stick reads. |

A **block** represents one tile-row: 32 consecutive sticks of width `logical_width`, which when tilized produce `ntiles_per_block` output tiles. The total number of blocks equals the number of tile-rows in the tensor.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | `[N, C, H, W]` (arbitrary ND, last dim = W) | `[N, C, H, W]` (same shape, padded to tile alignment) |
| **Dimension convention** | NCHW (last dim is row width) | NCHW |
| **Tensor layout** | ROW_MAJOR | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16 / FLOAT32 | BFLOAT16 / FLOAT32 (matches input) |

### Key Layout Properties for Reader

- **Page = one stick**: For row-major interleaved tensors, each page is one full row (width `W`) of the tensor. `page_size = W * element_size`. Pages are distributed round-robin across DRAM banks.
- **Page ID sequencing**: Pages are numbered sequentially from row 0. Page `i` corresponds to the i-th row of the flattened tensor.
- **aligned_page_size**: The page size aligned to hardware requirements (may include padding for alignment). Used in TensorAccessor address calculations.

---

## Data Flow Pattern (Reader Focus)

### Step-by-step flow:

1. **Reader reserves CB space**: `cb_reserve_back(cb_id_in0, ntiles_per_block)` -- reserves space for one block's worth of tile-sized pages in the input CB.

2. **Reader fetches 32 sticks**: Inner loop runs `k = 0..31` (TILE_HEIGHT iterations). For each stick:
   - Computes NoC address via `TensorAccessor::get_noc_addr(page_id)`
   - Issues `noc_async_read(src_noc_addr, l1_write_addr, width_size)` to copy the stick into L1
   - Advances `l1_write_addr` by `width_size` (the row width in bytes)
   - Increments `page_id`

3. **Reader completes transfer**: `noc_async_read_barrier()` -- waits for all 32 stick reads to finish.

4. **Reader pushes to CB**: `cb_push_back(cb_id_in0, ntiles_per_block)` -- signals that one block of data is ready for compute.

5. **Compute consumes**: `cb_wait_front(input_cb, ntiles_per_block)` then `tilize_block` converts 32 row-major sticks into `ntiles_per_block` tiles.

### Critical Insight: CB Page Size vs. Data Written

The input CB (c_0) is configured with `page_size = input_single_tile_size` (tile-sized pages, e.g., 2048 bytes for bfloat16 32x32), and `num_pages = ntiles_per_block`. The reader writes 32 sticks of width `W` bytes each, for a total of `32 * W` bytes. This equals `ntiles_per_block * tile_size` because `ntiles_per_block = ceil(W / 32)` and each tile is `32 * 32 * element_size`. The CB capacity in bytes matches the raw stick data volume for one tile-row.

The `cb_reserve_back` / `cb_push_back` calls use `ntiles_per_block` as the page count, but the reader writes raw stick bytes sequentially. The compute kernel (tilize_block via `llk_unpack_tilize_block`) reinterprets this sequential stick data as tiles during the tilization process.

### Data Flow Table

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved pages) | CB c_0 | `cb_reserve_back`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back` |
| 2 | Compute | CB c_0 | CB c_16 | `cb_wait_front`, `tilize_block`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` |
| 3 | Writer | CB c_16 | DRAM (interleaved tiles) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Total Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | RM sticks -> tilize input | `ntiles_per_block` | `input_single_tile_size` | `ntiles_per_block * input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | cb_output | Tilized tile output | `ntiles_per_block` | `output_single_tile_size` | `ntiles_per_block * output_single_tile_size` | Single | Compute | Writer | Block |

### CB Sizing Rationale (Input Stage Focus)

- **c_0 capacity = ntiles_per_block**: Holds exactly one block (one tile-row, i.e., 32 sticks). This is single-buffered: the reader fills the entire block before the compute kernel can consume it.
- **Page size = tile size**: Even though the reader writes raw RM sticks, the CB is configured with tile-sized pages because the compute kernel (tilize) consumes in tile units. The total byte capacity matches regardless: `ntiles_per_block * tile_size = 32 * W_bytes` (for standard 32x32 tiles with the same data type).

### Implications for layer_norm_rm

For `layer_norm_rm`, the input CB should similarly hold enough data for one processing unit. Since layer_norm operates on complete rows, the natural unit is one or more complete rows. The CB can be configured with:
- Page size = stick_size (row width in bytes) if compute consumes row-by-row, OR
- Page size = tile_size if the data will be tilized for compute

---

## Pipeline Pattern Summary

Both CBs are **single-buffered** (capacity equals one block). There is no overlap between reader and compute within a single block; the reader must complete writing all 32 sticks before compute can begin tilizing. Overlap occurs only between consecutive blocks: compute processes block N while reader fetches block N+1 (but with single-buffered CBs, this cannot happen in this configuration).

**Note**: Detailed pipeline simulation is out of scope for this analysis.

---

## Index Calculations

### Page ID Mapping (Reader)

The reader uses `TensorAccessor` to map page IDs to physical DRAM addresses:

```
page_id = start_page_id                    // Set per-core by host
for each block (i = 0..num_blocks-1):
    for each stick in block (k = 0..31):   // TILE_HEIGHT = 32
        noc_addr = TensorAccessor.get_noc_addr(page_id)
        page_id++
    page_id advances by tile_height * num_pages_in_row (32 for interleaved)
```

For the **interleaved non-sharded** case (the primary focus for `layer_norm_rm`):
- `num_pages_in_row = 1` (each tensor row is one page)
- `size_of_valid_data_in_last_page_in_row = page_size` (no partial pages)
- The inner `l` loop (over pages within a row) executes exactly once
- `page_id` increments by 1 per stick, and by `32 * 1 = 32` per block

### Host-Side Page Start Calculation

```cpp
page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
// For interleaved: page_start_id += 32 * nblocks_per_core
```

Each core starts reading from its assigned page offset. Pages are contiguous in the flattened tensor.

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Sequential stick reads with per-stick NoC transactions
- **Granularity**: One stick (one full row of width W) per `noc_async_read` call
- **Ordering**: Sticks are read in row-major order (row 0, row 1, ..., row 31) within each block
- **Barrier**: `noc_async_read_barrier()` after all 32 sticks in a block, not per-stick
- **DRAM access**: Each stick may land on a different DRAM bank (round-robin interleaving), so 32 stick reads may hit up to 12 different DRAM banks (Wormhole) -- good for bandwidth utilization
- **L1 write pattern**: Sticks are written contiguously into CB L1 space, advancing `l1_write_addr` by `width_size` each iteration

### Key Pattern for layer_norm_rm Reuse

The read pattern of "32 sequential sticks, barrier, push" is the fundamental tile-height-batched read pattern. For `layer_norm_rm`, which processes rows individually or in small groups, a simpler pattern may suffice: read fewer sticks (possibly just 1 row at a time) if row-level processing is desired, or batch 32 rows if tilize-then-compute is the strategy.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear core assignment from available grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` (device compute grid) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows) per full core |
| **Load balancing** | Near-equal, with optional cliff core |
| **Remainder handling** | Last core gets `nblocks_per_core_cliff` blocks if `nblocks % nblocks_per_core != 0` |

### Work Splitting Details (`split_blocks_for_tilize`)

```
nblocks_per_core = ceil(nblocks / grid_area)
ncores = ceil(nblocks / nblocks_per_core)
nblocks_per_core_cliff = nblocks % nblocks_per_core   // 0 if evenly divisible
```

- Cores are enumerated via `corerange_to_cores(available_grid)`, which linearizes the 2D grid into a 1D sequence.
- Full cores (indices 0..ncores_full-1) each process `nblocks_per_core` blocks.
- The cliff core (index ncores_full, if it exists) processes `nblocks_per_core_cliff` blocks.
- Two separate compute kernel variants are compiled: one for full cores and one for cliff cores, differing only in `per_core_block_cnt`.

### Relevance to layer_norm_rm

This 1D distribution pattern maps directly: each core processes a contiguous range of tensor rows. For `layer_norm_rm`, where each row is independently normalized, this is ideal. The work unit could be rows (or groups of rows) rather than tile-rows.

---

## Arguments

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `page_size` | uint32_t | Aligned page size in bytes (= stick size for RM interleaved) |
| 1 | `num_pages_in_row` | uint32_t | Pages per tensor row (1 for interleaved, >1 for ND-sharded) |
| 2 | `size_of_valid_data_in_last_page_in_row` | uint32_t | Valid bytes in last page (= page_size for interleaved) |
| 3+ | TensorAccessorArgs | uint32_t[] | Compile-time tensor accessor parameters (rank, bank info, shapes, coords) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Base address of source buffer in DRAM |
| 1 | `num_rows` | uint32_t | Total rows to read = `nblocks_per_core * TILE_HEIGHT` |
| 2 | `page_size` | uint32_t | Page size (duplicated from CT for flexibility) |
| 3 | `num_tiles_per_block` | uint32_t | Tiles per block = `ntiles_per_block` |
| 4 | `block_width_size` | uint32_t | Width of one block in bytes = `page_size` |
| 5 | `num_full_blocks_in_row` | uint32_t | Full blocks per row (1 for interleaved) |
| 6 | `num_leftover_tiles` | uint32_t | Leftover tiles (0 for interleaved) |
| 7 | `leftover_width_in_row` | uint32_t | Leftover width (0 for interleaved) |
| 8 | `start_page_id` | uint32_t | Starting page ID for this core's work |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) for this core |
| 1 | `per_core_block_tile_cnt` | uint32_t | Tiles per block = `ntiles_per_block` |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_multicore | RISCV_0 | NOC0 | DRAM (RM pages) | CB c_0 | Read 32 sticks per block |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` initialized from compile-time args at index 3 (`TensorAccessorArgs<3>()`)
  - The `read_tiles` lambda encapsulates the per-block read pattern: reserve CB, read 32 sticks, barrier, push
  - For interleaved tensors, the inner `num_pages_in_row` loop is trivially 1 iteration
  - `l1_write_addr` is obtained from `get_write_ptr(cb_id_in0)` and advanced by `width_size` per stick
  - Page IDs advance sequentially; `page_id` is incremented by 1 per stick within the inner loop

### Compute Kernel (Brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block (RM -> tiled) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**: Calls `compute_kernel_lib::tilize<ntiles_per_block, c_0, c_16>(nblocks)` which auto-selects fast_tilize when possible. Each block: `cb_wait_front(c_0)`, `tilize_block`, `cb_pop_front(c_0)`, `cb_push_back(c_16)`.

### Writer Kernel (Brief, De-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Writes tilized output pages one tile at a time using TensorAccessor.

---

## Implementation Notes

### TensorAccessor Pattern

The tilize reader demonstrates the standard TensorAccessor integration pattern:
1. **Host side**: `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` appends accessor compile-time args after the kernel's own CT args.
2. **Device side**: `constexpr auto src_tensor_args = TensorAccessorArgs<3>()` reconstructs from CT args starting at index 3. `TensorAccessor(src_tensor_args, src_addr, page_size)` creates the accessor with the runtime base address and page size.
3. **Usage**: `s.get_noc_addr(page_id)` returns the 64-bit NoC address for any page, handling bank interleaving transparently.

### Stick-to-Tile Batching Insight

The reader always groups exactly `TILE_HEIGHT = 32` sticks per CB push. This is because the compute kernel (tilize) processes data in tile-height units. The reader's `num_rows` runtime arg is always a multiple of 32 (`nblocks_per_core * TILE_HEIGHT`).

For `layer_norm_rm`, if the compute stage operates on individual rows (not tiles), the reader could push fewer sticks per CB cycle. But if tilized compute is used, the 32-stick batching pattern should be preserved.

### Sub-Core Grids

The program factory accepts `sub_core_grids` to restrict which cores are used. This supports running alongside other operations that reserve part of the compute grid.

### FP32 Accumulation

When `a.dtype() == DataType::FLOAT32`, the compute kernel is configured with `fp32_dest_acc_en = true` and `UnpackToDestMode::UnpackToDestFp32` for lossless float32 tilization.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does TensorAccessor work in tt-metal kernels? What does TensorAccessorArgs do and how does get_noc_addr work for interleaved row-major tensors?"
   **Reason**: Understanding how the reader kernel translates page IDs to physical DRAM addresses is fundamental to the read pattern.
   **Key Findings**: TensorAccessor maps page_id -> bank_id + bank_offset -> NoC XY + physical address. For interleaved tensors, pages are distributed round-robin across banks. TensorAccessorArgs packs configuration as compile-time args on the host and reconstructs on device.

2. **Query**: "How does split_blocks_for_tilize work for 1D core distribution?"
   **Reason**: Understanding work distribution is essential for the core assignment strategy.
   **Key Findings**: Computes `nblocks_per_core = ceil(nblocks / grid_area)`, then `ncores = ceil(nblocks / nblocks_per_core)`. Last core may be a "cliff" core with fewer blocks. Returns BlockSplit with core_range and core_range_cliff.

3. **Query**: "What is the relationship between page_size and stick_size for row-major interleaved tensors?"
   **Reason**: Needed to confirm that each page is one full row for RM interleaved tensors.
   **Key Findings**: For row-major layout, each row = one page. `page_size = W * element_size`. Number of pages = total rows = product of all dims except the last.

4. **Query**: "How does cb_reserve_back and cb_push_back work with row-major data in the tilize context?"
   **Reason**: The input CB uses tile-sized pages but receives raw RM sticks; needed to understand how this works.
   **Key Findings**: CB operations use page counts. The reader writes raw stick bytes sequentially into the CB's L1 space. The total bytes (32 sticks * width) equals `ntiles_per_block * tile_size`. The compute kernel reinterprets this sequential data as tiles via `llk_unpack_tilize_block`.

5. **Query**: "How does the input CB work when configured with tile-sized pages but receiving row-major stick data?"
   **Reason**: Deeper understanding of the CB page_size vs actual data relationship.
   **Key Findings**: The CB is a raw memory region; page_size determines how the CB tracks space. The reader fills it with contiguous stick data. Compute's tilize_block reorders sticks into tile format during unpacking. The byte volumes match because `ntiles_per_block * tile_size = 32 * stick_size`.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host-side setup and device-side usage.
   **Key Information**: Detailed API for `TensorAccessorArgs` construction, compile-time vs runtime arg configuration, and `get_noc_addr` usage patterns.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major page structure and interleaved memory layout.
   **Key Information**: RM layout: each row = one page. Interleaved: pages distributed round-robin across banks. Tiled layout: each 32x32 tile = one page. N-dimensional tensors are stored as 2D: outer dims squeezed, inner dim preserved.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding compute kernel's tilize function template and its CB interaction.
   **Key Information**: `tilize<block_width_tiles, input_cb, output_cb>()` with WaitBlock mode: per-block `cb_wait_front`, `tilize_block`, `cb_pop_front`, `cb_push_back`. Supports symmetric (tile-sized input pages) and asymmetric (row-sized input pages) modes. Auto-selects fast_tilize for bfloat16 32x32.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the 1D block splitting algorithm.
   **Key Information**: `compute_ncores` calculates even distribution with cliff. `split_blocks_for_tilize(CoreRangeSet)` returns `BlockSplit` with separate `core_range` and `core_range_cliff` sets.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper.
   **Key Information**: Wraps `CircularBufferConfig(num_pages * page_size, ...)` and `set_page_size(cb, page_size)`. Returns `(cb_index, cb_handle)`.
