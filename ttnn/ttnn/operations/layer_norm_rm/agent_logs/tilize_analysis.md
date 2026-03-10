# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) tensor into a tiled tensor. It reads contiguous "sticks" (rows) from DRAM, groups them 32 at a time into the input CB, and the compute kernel rearranges the data into 32x32 tiles. This analysis covers the interleaved multi-core variant.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role focus**: input_stage -- reader kernel pattern, input CB sizing, stick-to-tile batching, core distribution.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a horizontal strip of tiles spanning the full tensor width, 32 rows tall) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / 32`) |
| **Total units** | `nblocks = physical_volume / TILE_HW / ntiles_per_block` (equivalently, `padded_height / 32`) |
| **Loop structure** | Outer: iterate over blocks (each block = 32 sticks). Inner (reader): read 32 sticks, then push one block of tiles to CB. |

A "block" is one horizontal strip of tiles spanning the full tensor width and 32 rows (one TILE_HEIGHT) tall. Each block contains `ntiles_per_block = W / 32` tiles, where W is the padded width of the last dimension. The total number of blocks equals the padded height divided by 32.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[..., H, W]` (arbitrary outer dims, H and W padded to tile multiples) |
| **Dimension convention** | Last dim is contiguous in memory |
| **Tensor layout** | ROW_MAJOR (each page = one row / "stick") |
| **Memory layout** | INTERLEAVED (pages round-robin across DRAM banks) |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 or FLOAT32 |

- **Page definition**: One row-major page = one stick = `W * element_size` bytes.
- **Page count**: Total number of sticks = `physical_volume / W`.
- **Bank distribution**: Sticks are interleaved round-robin across DRAM banks.

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT (each page = one 32x32 tile) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input |

### Layout Transformation

The entire purpose of this operation is the RM-to-tile conversion. No data values change; only the memory arrangement changes. The compute kernel uses the hardware tilize LLK to rearrange 32 consecutive sticks into tiles.

---

## Data Flow Pattern

### Step-by-Step Flow

1. **Reader** resolves NoC addresses for 32 consecutive sticks (one tile-height of rows).
2. **Reader** calls `cb_reserve_back(c_0, ntiles_per_block)` to reserve space in the input CB.
3. **Reader** issues 32 `noc_async_read` calls, one per stick, each reading `block_width_size` bytes (= full row width). The L1 write pointer advances by `block_width_size` after each stick.
4. **Reader** calls `noc_async_read_barrier()` then `cb_push_back(c_0, ntiles_per_block)` to signal the data is ready.
5. **Compute** calls `cb_wait_front(c_0, ntiles_per_block)` to wait for the reader's block.
6. **Compute** calls `tilize_block` (or `fast_tilize_block`) to rearrange the row-major data into tiled format in the output CB (`c_16`).
7. **Compute** calls `cb_push_back(c_16, ntiles_per_block)` and `cb_pop_front(c_0, ntiles_per_block)`.
8. **Writer** waits on `c_16` one tile at a time, writes each tile to DRAM via NoC.
9. Repeat for all blocks assigned to this core.

### Data Flow Table

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (RM sticks) | CB c_0 | reserve_back(ntiles_per_block), push_back(ntiles_per_block) |
| 2 | Compute | CB c_0 | CB c_16 | wait_front(ntiles_per_block), pop_front(ntiles_per_block), reserve_back(ntiles_per_block), push_back(ntiles_per_block) |
| 3 | Writer | CB c_16 | DRAM (tiles) | wait_front(1), pop_front(1) per tile |

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Page Size | Num Pages | Total Capacity | Buffering | Producer | Consumer |
|-------|------|---------|-----------|-----------|----------------|-----------|----------|----------|
| c_0 | input | RM sticks staged as tile-equivalent blocks | `input_single_tile_size` | `ntiles_per_block` | `ntiles_per_block * tile_size` | Single | Reader | Compute |
| c_16 | output | Tilized output tiles | `output_single_tile_size` | `ntiles_per_block` | `ntiles_per_block * tile_size` | Single | Compute | Writer |

### Critical Insight: CB c_0 Page Size vs Actual Data Written

The input CB c_0 is configured with `page_size = input_single_tile_size` (e.g., 2048 bytes for bfloat16) and `num_pages = ntiles_per_block`. This means the CB can hold `ntiles_per_block * tile_size` bytes total.

However, the **reader writes row-major sticks**, not tiles. The key mathematical identity that makes this work:

```
CB capacity = ntiles_per_block * tile_size
            = (W / 32) * (32 * 32 * element_size)
            = W * 32 * element_size
            = 32 * stick_size
            = TILE_HEIGHT sticks * stick_size
```

So the byte capacity of `ntiles_per_block` tiles **exactly equals** 32 sticks of width W. The reader fills the CB with 32 raw sticks laid out contiguously, and the compute kernel's tilize LLK interprets this same memory region as tile data by rearranging it during unpack.

The `cb_reserve_back` / `cb_push_back` calls use `ntiles_per_block` as the page count. The CB tracks capacity in units of `page_size`, so reserving `ntiles_per_block` pages means reserving `ntiles_per_block * tile_size` bytes -- which is exactly the 32 sticks the reader will write.

### Reader Write Pattern into CB c_0

```
L1 address:  [stick_0][stick_1]...[stick_31]
             |<----------- ntiles_per_block * tile_size bytes ----------->|
             |<----------- 32 * W * elem_size bytes --------------------->|
```

Each stick is written at offset `k * block_width_size` from the CB write pointer, where k ranges from 0 to 31. This is a contiguous fill: 32 sticks packed end-to-end.

---

## Reader Kernel Deep Dive

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

### Architecture

The reader uses a simple but effective pattern:
1. Pre-compute NoC addresses for all 32 sticks in a group
2. Read all sticks into the CB as a batch
3. Repeat for each block assigned to this core

### TensorAccessor Usage

```cpp
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```

- The TensorAccessor is constructed at compile-time arg index 1 (the host calls `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` after the first compile-time arg `block_size_nbytes`).
- `stick_size` is the page size for address calculation (one full row in bytes).
- `get_noc_addr(stick_id, s)` maps a logical stick ID to the physical NoC address in the appropriate DRAM bank via interleaved round-robin mapping.

### Read Pattern

```cpp
uint64_t base_src_noc_addr[tile_height];  // 32 addresses cached

// For each group of 32 sticks:
for (uint32_t j = 0; j < tile_height; j++) {
    base_src_noc_addr[j] = get_noc_addr(stick_id, s);
    stick_id++;
}

// Then read all sticks into CB:
auto read_tiles = [&](num_tiles, width_size) {
    cb_reserve_back(cb_id_in0, num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t k = 0; k < tile_height; k++) {
        noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size);
        l1_write_addr += width_size;
        base_src_noc_addr[k] += width_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, num_tiles);
};
```

Key observations:
- **Address caching**: All 32 NoC addresses are resolved before any reads begin. This separates address computation from data transfer.
- **Sequential stick reads**: Sticks are read in order (stick 0 through stick 31) for each block. Each `noc_async_read` reads `width_size` bytes (the full row width).
- **`base_src_noc_addr[k] += width_size`**: This line advances the cached NoC address for potential multi-block-per-row scenarios. In this interleaved variant, `num_full_blocks_in_row` is always 1 (the entire row width is one block), so this advance is not exercised across iterations.
- **Barrier after all 32 reads**: A single `noc_async_read_barrier()` is called after all 32 stick reads are issued. This ensures all DMA transfers complete before signaling the compute kernel.

### Read Access Pattern

- **Pattern type**: Strided across DRAM banks (each stick may reside in a different bank due to interleaving).
- **Within a stick**: Sequential / contiguous (the entire stick is one NoC read).
- **Across sticks**: 32 consecutive stick IDs are read, which means 32 consecutive pages from the interleaved buffer. These hit different DRAM banks in round-robin fashion, providing bank-level parallelism.

### Stick-to-Tile Batching Summary

| Property | Value |
|----------|-------|
| **Sticks per CB push** | 32 (= TILE_HEIGHT) |
| **CB pages per push** | `ntiles_per_block` (= W / TILE_WIDTH) |
| **Bytes per push** | `32 * W * element_size` |
| **Reads per push** | 32 (one per stick) |
| **Barrier** | One per push (after all 32 reads) |

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` (full compute grid) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (or `nblocks_per_core_cliff` for the last core) |
| **Load balancing** | Equal blocks per core, with one optional "cliff" core for remainder |

### Work Split Algorithm (`split_blocks_for_tilize`)

The function `ttnn::split_blocks_for_tilize(available_grid, nblocks)` performs 1D partitioning:

1. Compute grid area = number of available cores.
2. `nblocks_per_core = ceil(nblocks / grid_area)` -- tries to give each core equal work.
3. `ncores = ceil(nblocks / nblocks_per_core)` -- only use as many cores as needed.
4. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for last core.
5. If `nblocks_per_core_cliff > 0`, the last core is a "cliff" core with fewer blocks.

The cores are enumerated from the available grid in linearized order. Full cores get `nblocks_per_core` blocks; the optional cliff core gets `nblocks_per_core_cliff` blocks.

### Runtime Args Per Core (Reader)

Each core receives:
- `src_addr`: Source buffer base address
- `num_sticks`: `nblocks_per_core * TILE_HEIGHT` (total sticks this core reads)
- `block_size_nbytes`: Full row width in bytes (= `padded_shape[-1] * element_size`)
- `ntiles_per_block`: Tiles per row (= `padded_shape[-1] / 32`)
- `block_width_size`: Same as `block_size_nbytes` (full row width)
- `num_full_blocks_in_row`: Always 1 (entire width in one block)
- `num_leftover_tiles`: 0 (no partial tile columns)
- `leftover_width_in_row`: 0 (no partial width)
- `start_stick_id`: Starting stick index for this core (= `row_start_id`)

The `row_start_id` advances by `TILE_HEIGHT * nblocks_per_core` between cores, so each core processes a contiguous range of sticks.

---

## Arguments

### Compile-Time Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` | uint32_t | Full row width in bytes (`block_size_nbytes`) |
| 1+ | TensorAccessor args | (multiple) | Bank mapping info for source buffer (appended via `TensorAccessorArgs(*src0_buffer).append_to()`) |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer address in DRAM |
| 1 | `num_sticks` | uint32_t | Total sticks this core reads (`nblocks * TILE_HEIGHT`) |
| 2 | `block_size_nbytes` | uint32_t | Full row width in bytes |
| 3 | `ntiles_per_block` | uint32_t | Tiles per row (`W / 32`) |
| 4 | `block_width_size` | uint32_t | Same as block_size_nbytes (row width bytes) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 (full width per block) |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 |
| 8 | `start_stick_id` | uint32_t | First stick index for this core |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks this core processes |
| 1 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (`ntiles_per_block`) |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block via noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**: Pre-computes 32 NoC addresses, then issues 32 contiguous stick reads into CB c_0. Single barrier per block. Uses TensorAccessor for interleaved bank address resolution.
- **Loop structure**: Outer loop over `num_sticks / tile_height` blocks. Inner loops: (1) resolve 32 addresses, (2) read sticks via `read_tiles` lambda.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block (LLK) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**: Calls `compute_kernel_lib::tilize<c_0, c_16>` with `WaitBlock` mode. For each block, waits for `ntiles_per_block` pages in c_0, rearranges row-major data into tile format, pushes `ntiles_per_block` tiles to c_16. Supports automatic `fast_tilize` path for bfloat16/float32 with 32x32 tiles and half-sync mode.

### Writer Kernel (Brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write one tile at a time via noc_async_write_page |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic tile writer. Reads page_size from CB interface, writes tiles one at a time using TensorAccessor.

---

## Implementation Notes

### Why `num_full_blocks_in_row` is Always 1

The reader kernel supports a "split rows" pattern where a single row could be divided into multiple horizontal blocks (useful when the row is very wide). In this interleaved factory, the entire row width is treated as one block (`block_width_size = padded_shape[-1] * element_size`, `num_full_blocks_in_row = 1`). The leftover tile and leftover width args are both 0.

This simplifies the reader to: for each group of 32 sticks, make one `read_tiles` call that reads the full width.

### CB Sizing is Single-Buffered

Both CBs are sized at exactly `ntiles_per_block` pages (one block). This means the reader, compute, and writer cannot overlap across blocks -- each must complete its block before the next can start. For a layer_norm_rm operation that chains tilize into compute, this is an important constraint: the tilize input CB should be sized to hold exactly one tile-row's worth of sticks.

### FP32 Accumulation

When the input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in the compute config, enabling FP32 accumulation in the destination registers.

### `sub_core_grids` Support

The factory accepts an optional `sub_core_grids` parameter from the operation attributes, allowing the caller to restrict which cores are used. If not provided, the full compute-with-storage grid is used.

---

## Reuse Guidance for layer_norm_rm Input Stage

For an RM-input layer normalization operation that needs to tilize on the fly:

1. **Reader pattern to reuse**: The `reader_unary_stick_layout_split_rows_interleaved.cpp` pattern is directly applicable. Use TensorAccessor with stick_size as page_size. Read 32 sticks at a time into a CB sized at `ntiles_per_block` tile-sized pages.

2. **Input CB sizing rule**: `num_pages = ntiles_per_block = W / 32`, `page_size = tile_size(dtype)`. This gives exactly 32 sticks of capacity.

3. **Work unit**: One block = 32 sticks = one tile-row. Distribute tile-rows across cores.

4. **Core distribution**: Use `split_blocks_for_tilize(grid, nblocks)` or equivalent 1D block splitting.

5. **TensorAccessor setup**: `TensorAccessorArgs(*src_buffer).append_to(ct_args)` on host. `TensorAccessorArgs<offset>()` + `TensorAccessor(args, addr, stick_size)` in kernel.

6. **Key identity**: `ntiles_per_block * tile_size == 32 * stick_size`. This is what makes tile-page-counted CBs work for row-major stick input.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in TTNN? What is the relationship between row-major sticks and tiles?"
   **Reason**: Needed to confirm the fundamental mechanics of stick-to-tile conversion.
   **Key Findings**: 32 consecutive RM sticks of width W are collected and rearranged into tiles. The tilize LLK handles the unpack from row-major to tiled format. Within tiles, data is organized into 16x16 faces.

2. **Query**: "What is a block in tilize? How does split_blocks_for_tilize partition work?"
   **Reason**: Needed to understand the work unit and distribution strategy.
   **Key Findings**: A block = a collection of tiles processed together (one tile-row). `split_blocks_for_tilize` calculates `nblocks_per_core = ceil(nblocks / grid_area)` and optionally creates a cliff core for the remainder.

3. **Query**: "How does get_noc_addr(stick_id, tensor_accessor) work for interleaved RM tensors?"
   **Reason**: Needed to understand how stick IDs map to physical DRAM addresses.
   **Key Findings**: TensorAccessor uses interleaved round-robin bank mapping. `get_noc_addr` resolves bank index and offset within bank, then combines with NoC coordinates.

4. **Query**: "CB reserve_back/push_back page count semantics for tilize with stick input"
   **Reason**: Critical to understand how tile-page-counted CBs work when filled with RM sticks.
   **Key Findings**: The CB tracks capacity in page-size units (tile_size). The identity `ntiles_per_block * tile_size == 32 * stick_size` ensures byte-level equivalence. Verified analytically: `(W/32) * (32*32*elem_size) = 32 * (W*elem_size)`.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor host setup and device-side usage.
   **Key Information**: Host uses `TensorAccessorArgs(buffer).append_to(ct_args)`. Device constructs with `TensorAccessorArgs<offset>()` + `TensorAccessor(args, addr, page_size)`. `get_noc_addr(page_id)` for address resolution.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM page definition and interleaved bank mapping.
   **Key Information**: In RM layout, each row = one page. In interleaved memory layout, pages are round-robin distributed across banks. In tiled layout, each page = one 32x32 tile.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize helper API.
   **Key Information**: `compute_kernel_lib::tilize<input_cb, output_cb>` supports symmetric mode (both CBs tile-sized pages) and asymmetric mode. WaitBlock mode waits per-block. Auto-selects fast_tilize for bfloat16/float32 with 32x32 tiles.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the block-to-core distribution algorithm.
   **Key Information**: 1D split: `nblocks_per_core = ceil(nblocks / grid_area)`, `ncores = ceil(nblocks / nblocks_per_core)`, cliff core gets remainder. Returns `BlockSplit` with core ranges.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with total size = `num_pages * page_size`, setting `page_size` for the given CB index.
