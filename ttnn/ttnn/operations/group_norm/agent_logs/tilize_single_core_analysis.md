# Tilize Single Core Implementation Analysis

## Overview

The tilize single-core operation converts a row-major (RM) tensor into tile layout (32x32 tiles). It reads contiguous rows ("sticks") from DRAM, packs them into tile-sized CB pages, and uses the hardware tilize LLK to reorder the data into the face-interleaved tile format. The entire operation runs on a single Tensix core.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp`

**Intended downstream use**: This analysis serves as an `input_stage` reference for a new group_norm operation that needs to read RM sticks from DRAM and tilize them before compute. The group_norm input has shape `(N, 1, H*W, C)` in ROW_MAJOR_LAYOUT.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles (horizontal strip within a tile-row) |
| **Unit size** | `num_tiles_per_block` tiles (up to a full tile-row width) |
| **Total units** | `num_tiles / num_tiles_per_block` = `(num_sticks / 32) * num_full_blocks_in_row` |
| **Loop structure** | Outer: iterate over tile-rows (`num_sticks / 32`). Inner: iterate over blocks within each tile-row (`num_full_blocks_in_row`). |

A "tile-row" is a group of 32 consecutive RM sticks. Within each tile-row, the width is split into blocks of `num_tiles_per_block` tiles. Each block is one work unit processed through the reader-compute-writer pipeline.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | `[N, ..., H, W]` (arbitrary, flattened to 2D) | Same logical shape |
| **Dimension convention** | Last dim = W (stick width) | Last dim = W |
| **Tensor layout** | ROW_MAJOR_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Key Dimensional Quantities (from program factory, lines 45-51)

| Symbol | Formula | Meaning |
|--------|---------|---------|
| `stick_s` | `padded_shape[-1]` | Width of one RM stick in elements |
| `num_sticks` | `physical_volume / stick_s` | Total number of RM sticks |
| `stick_size` | `stick_s * element_size` | Size of one stick in bytes |
| `num_tiles_in_row` | `stick_s / TILE_WIDTH` (= `stick_s / 32`) | Tiles along the width dimension |
| `num_tiles` | `physical_volume / TILE_HW` (= `physical_volume / 1024`) | Total output tiles |

### Layout Transformations

The operation performs a single transformation: ROW_MAJOR to TILE_LAYOUT. The reader reads RM sticks and packs 32 of them contiguously into tile-sized CB pages. The compute kernel then uses the hardware tilize LLK to reorder these 32x32 element blocks from row-major ordering into the face-interleaved tile format (face0, face1, face2, face3 each 16x16).

---

## Data Flow Pattern

### Step-by-step Flow

```
DRAM (RM sticks)
    |
    | [Reader: noc_async_read per stick, 32 sticks batched per tile-row]
    v
CB c_0 (input, tile-sized pages containing 32 packed RM sticks)
    |
    | [Compute: tilize_block / fast_tilize_block]
    v
CB c_16 (output, tile-sized pages in TILE_LAYOUT)
    |
    | [Writer: noc_async_write per tile page]
    v
DRAM (tiled output)
```

### Reader Kernel: Detailed RM Stick Reading Pattern

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

The reader kernel name "split_rows" describes its core strategy: it reads tile-width segments of sticks rather than full sticks, assembling tile-sized blocks from horizontal strips across 32 consecutive rows.

#### The `read_tiles` Lambda (lines 27-38)

This is the central data movement primitive. For a block of `num_tiles` tiles:

1. **`cb_reserve_back(cb_id_in0, num_tiles)`** -- Reserve space for `num_tiles` tile-sized pages in the input CB.
2. **`get_write_ptr(cb_id_in0)`** -- Get the L1 address where data should be written.
3. **Inner loop over 32 sticks** (`k = 0..tile_height-1`):
   - Read `width_size` bytes from `base_src_noc_addr[k]` (one stick's horizontal segment) into L1.
   - `noc_async_read(src_noc_addr, l1_write_addr, width_size)` -- Asynchronous DRAM-to-L1 transfer.
   - Advance `l1_write_addr` by `width_size` (pack sticks contiguously in L1).
   - Advance `base_src_noc_addr[k]` by `width_size` (move to next block within same stick).
4. **`noc_async_read_barrier()`** -- Wait for all 32 reads to complete.
5. **`cb_push_back(cb_id_in0, num_tiles)`** -- Signal that data is available for compute.

#### Outer Loop Structure (lines 40-51)

```
for each tile_row (i = 0 .. num_sticks/32 - 1):
    // Phase 1: Resolve DRAM addresses for 32 consecutive sticks
    for j = 0..31:
        base_src_noc_addr[j] = get_noc_addr(stick_id, s)
        stick_id++

    // Phase 2: Read blocks horizontally across the tile-row
    for j = 0 .. num_full_blocks_in_row - 1:
        read_tiles(num_tiles_per_block, block_width_size)
```

**Critical pattern**: The 32 base addresses are resolved once per tile-row, then reused (with advancing offsets) across all horizontal blocks. The `base_src_noc_addr[k] += width_size` inside `read_tiles` advances each stick's read pointer horizontally, so consecutive calls to `read_tiles` read successive horizontal segments of the same 32 sticks.

#### Stick-to-Tile Batching: How 32 RM Sticks Become Tiles

The reader packs exactly **32 RM sticks** (each `block_width_size` bytes wide) contiguously into a single CB reservation. This creates a memory layout of `[32 rows x block_width_elements]` in L1, which is precisely the raw data for `num_tiles_per_block` tiles (each 32 rows x 32 columns).

```
CB page layout in L1 after read_tiles(num_tiles_per_block, block_width_size):

  stick[0]:   [col_0..col_31] [col_32..col_63] ... (block_width_size bytes)
  stick[1]:   [col_0..col_31] [col_32..col_63] ...
  ...
  stick[31]:  [col_0..col_31] [col_32..col_63] ...

  Total size: 32 * block_width_size = num_tiles_per_block * tile_size
```

**CRITICAL**: The input CB page_size is set to `input_single_tile_size` (tile-sized), NOT `stick_size`. This is essential because `tilize_init` / `state_configure` reads face/tile dimensions from the input CB's metadata. With stick-sized pages, the hardware would get incorrect tile dimensions and only process 16 of 32 rows, leaving faces 2-3 empty. The reader must pack 32 RM sticks contiguously into tile-sized pages. (Reference: user memory note from 2026-02-27.)

#### TensorAccessor Usage

The reader creates a `TensorAccessor` to map logical stick IDs to physical DRAM NoC addresses:

```cpp
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args starting at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```

The `page_size` parameter to TensorAccessor is `stick_size` (one full RM row in bytes), because the interleaved memory layout distributes pages at stick granularity. `get_noc_addr(stick_id, s)` returns the 64-bit NoC address for the start of stick `stick_id` in DRAM.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging (RM sticks packed as tile-sized pages) | `num_tiles_per_block` | `num_tiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | output | Output staging (tiled data) | `num_tiles_per_block` | `num_tiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### Input CB Sizing Details (lines 76-82)

```cpp
uint32_t src0_cb_index = 0;                          // CB c_0
uint32_t num_input_tiles = num_tiles_per_block;       // capacity = exactly one block
auto src0_cb_config = CircularBufferConfig(
    num_input_tiles * input_single_tile_size,          // total bytes
    {{src0_cb_index, input_cb_data_format}})
    .set_page_size(src0_cb_index, input_single_tile_size);  // page = one tile
```

Key observations:
- **Capacity equals block size**: Both CBs are single-buffered (capacity = 1 block). No overlap between reader and compute is possible.
- **Page size = tile size**: Even though the reader writes RM data, the page size is tile-sized. This is the required contract for `tilize_init`.
- **Total CB bytes**: `num_tiles_per_block * tile_size`. For a tensor with 64 columns (2 tiles wide) using bfloat16: `2 * 2048 = 4096 bytes`.

### Block Size Calculation (lines 53-69)

The program factory tries to maximize `num_tiles_per_block` up to a full row width, constrained by L1 capacity:

```cpp
uint32_t max_l1_size = (l1_size_per_core / 2) - base_allocator_addr;
uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);

if (num_tiles_in_row <= max_tiles) {
    num_tiles_per_block = num_tiles_in_row;    // entire row fits
} else {
    // Find largest divisor of num_tiles_in_row that fits
    for (n_t = max_tiles; n_t > 0; n_t--)
        if (num_tiles_in_row % n_t == 0) { num_tiles_per_block = n_t; break; }
}
```

The constraint `num_tiles_in_row % num_tiles_per_block == 0` ensures clean block boundaries with no leftover tiles. The `use_low_perf` flag forces `num_tiles_per_block = 1` for minimal memory usage.

---

## Pipeline Pattern Summary

Both CBs are **single-buffered** (capacity = block size). The pipeline is strictly sequential within each block:
1. Reader fills CB c_0 (1 block)
2. Compute reads CB c_0, writes CB c_16 (1 block)
3. Writer drains CB c_16 (1 block)

No reader-compute overlap is possible. This is acceptable for a single-core implementation where simplicity is prioritized.

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Strided reads -- 32 non-contiguous DRAM locations per block, each reading `block_width_size` bytes.
- **Stride**: Each of the 32 sticks is at a different DRAM page (interleaved across banks). Within a tile-row, the 32 sticks are logically consecutive but physically distributed across DRAM banks via round-robin interleaving.
- **Transfer size**: `block_width_size = num_tiles_per_block * TILE_WIDTH * element_size` bytes per stick segment. For 2 tiles of bfloat16: `2 * 32 * 2 = 128 bytes`.
- **Reads per block**: Exactly 32 `noc_async_read` calls (one per stick in the tile-row).
- **Address resolution**: One `get_noc_addr` per stick per tile-row (32 calls), cached in `base_src_noc_addr[]` and advanced via `+= width_size` for subsequent blocks.

### Write Pattern (Writer Kernel)

- **Pattern**: Sequential tile writes. One tile at a time, incrementing tile ID.
- **Transfer size**: `output_single_tile_size` bytes per tile.
- **Uses**: `noc_async_write_page(i, s, l1_read_addr)` with TensorAccessor for address mapping.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (0,0) or specified via `sub_core_grids` |
| **Grid dimensions** | 1x1 |
| **Total cores** | 1 |
| **Work per core** | All tiles in the tensor |
| **Load balancing** | N/A (single core) |

The single-core variant assigns all work to one core. The `sub_core_grids` parameter allows the caller to specify which physical core to use, defaulting to core (0,0).

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` | uint32_t | Size of one complete RM stick in bytes (`width * element_size`) |
| 1+ | TensorAccessorArgs | (multiple) | Interleaved buffer metadata (rank, num_banks, tensor_shape, bank_coords) appended by `TensorAccessorArgs(*src0_buffer).append_to()` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to process = `num_tiles / num_tiles_per_block` |
| 1 | `per_core_block_tile_cnt` | uint32_t | Tiles per block = `num_tiles_per_block` |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `output_cb_index` | uint32_t | CB index for output = `c_16` |
| 1+ | TensorAccessorArgs | (multiple) | Interleaved buffer metadata for destination |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer DRAM address |
| 1 | `num_sticks` | uint32_t | Total number of RM sticks |
| 2 | `stick_size` | uint32_t | Stick size in bytes (duplicated from compile-time for flexibility) |
| 3 | `num_tiles_per_block` | uint32_t | Tiles per horizontal block |
| 4 | `block_width_size` | uint32_t | Bytes per stick segment per block (`num_tiles_per_block * 32 * element_size`) |
| 5 | `num_full_blocks_in_row` | uint32_t | Number of complete blocks per tile-row |
| 6 | `num_leftover_tiles` | uint32_t | Leftover tiles (unused in current kernel -- always 0 due to divisibility constraint) |
| 7 | `leftover_width_in_row` | uint32_t | Leftover width bytes (unused -- always 0) |
| 8 | `start_stick_id` | uint32_t | Starting stick index (0 for single-core) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer DRAM address |
| 1 | `num_tiles` | uint32_t | Total tiles to write |
| 2 | `start_id` | uint32_t | Starting tile ID (0 for single-core) |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per tile-row, pack into tile-sized pages |
| Compute | RISCV_2 | N/A | CB c_0 | CB c_16 | `compute_kernel_lib::tilize` (row-major to tile reorder) |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM | Write tiles sequentially |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Uses a local array `base_src_noc_addr[32]` to cache the starting DRAM addresses for 32 sticks in a tile-row.
  - The `read_tiles` lambda performs 32 parallel async reads (one per stick), each reading `width_size` bytes.
  - After each `read_tiles` call, `base_src_noc_addr[k] += width_size` advances the horizontal read position, enabling the next block to read the next tile-width segment of the same 32 sticks.
  - Each `cb_reserve_back` + `cb_push_back` pair signals exactly `num_tiles_per_block` tile-sized pages.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` for hardware initialization.
  - Delegates to `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>`.
  - The helper internally calls `cb_wait_front(c_0, block_width_tiles)` per block, then `tilize_block` (or `fast_tilize_block`), then `cb_push_back(c_16, block_width_tiles)` and `cb_pop_front(c_0, block_width_tiles)`.
  - `fast_tilize` is selected at compile time when tiles are 32x32, format is Float32 or Float16_b, and half-sync dest mode is active.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic single-page writer. Waits for one tile, writes it to DRAM via TensorAccessor, pops it. Loops over all tiles sequentially.

---

## Implementation Notes

### Critical: Input CB Page Size Must Be Tile-Sized
Even though the reader writes row-major data, the input CB `page_size` is set to `input_single_tile_size` (not `stick_size`). This is because `tilize_init` (called by the compute helper) reads face and tile dimensions from the CB's metadata. If the page size were stick-sized, the hardware would derive incorrect tile geometry, resulting in only 16 of 32 rows being processed. The reader compensates by packing exactly 32 sticks into each tile-sized page.

### Block Size Optimization
The program factory maximizes `num_tiles_per_block` subject to L1 capacity. This minimizes the number of `cb_reserve_back`/`cb_push_back` cycles and amortizes the overhead of the 32 DRAM address resolutions per tile-row. The divisibility constraint (`num_tiles_in_row % num_tiles_per_block == 0`) avoids partial-block handling.

### Leftover Tile Handling
The reader kernel's runtime args include `num_leftover_tiles` and `leftover_width_in_row`, but the kernel does not use them (the loop only processes `num_full_blocks_in_row`). This is because the program factory enforces divisibility, making leftovers always zero.

### FP32 Accumulation
When the input dtype is FLOAT32, `fp32_dest_acc_en` is set to true in the compute config, enabling full-precision accumulation in the destination register.

### Relevance to Group Norm Input Stage
For a group_norm operation reading RM input with shape `(N, 1, H*W, C)`:
- The stick width would be `C` (the channel dimension), with `stick_size = C * element_size`.
- `num_sticks = N * H * W`.
- The reader pattern of "resolve 32 stick addresses, then read horizontal blocks" maps directly to reading 32 spatial positions at a time.
- The CB page size must be tile-sized (not stick-sized) if using `compute_kernel_lib::tilize`.
- The `TensorAccessor` pattern with `stick_size` as the page size for DRAM address resolution is reusable.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor work in dataflow kernels? Specifically, how does TensorAccessorArgs and get_noc_addr work for reading interleaved tensors from DRAM?"
   **Reason**: Needed to understand the address resolution mechanism used in the reader kernel.
   **Key Findings**: TensorAccessor maps logical page/stick IDs to physical NoC addresses in DRAM, abstracting the round-robin bank interleaving. `TensorAccessorArgs` packs tensor metadata (rank, banks, shape) as compile-time or runtime args. The `stick_size` parameter defines the page granularity for address calculation.

2. **Query**: "What does the tilize operation do in TTNN? How does it convert row-major data into tile layout?"
   **Reason**: Needed to understand the fundamental transformation being performed.
   **Key Findings**: Tilize reorders elements from consecutive RM rows into 32x32 blocks, further subdivided into 16x16 faces. The `llk_unpack_tilize_block` function performs the actual data reordering. 32 consecutive sticks are combined to form one tile-row of tiles.

3. **Query**: "How do circular buffers work in TT-Metal kernels? Specifically cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front?"
   **Reason**: Needed to verify the producer-consumer synchronization semantics used in all three kernels.
   **Key Findings**: `cb_reserve_back` blocks until space is available (producer). `cb_push_back` signals data availability to consumers. `cb_wait_front` blocks until data is available (consumer). `cb_pop_front` frees space for the producer. These form the fundamental inter-kernel synchronization mechanism.

4. **Query**: "Why must the input CB page_size be tile_page_size rather than stick_size for tilize?"
   **Reason**: Needed to verify the critical CB page size constraint mentioned in user memory.
   **Key Findings**: `tilize_init` and the LLK configuration read tile dimensions from CB metadata (`pack_tile_r_dim`, `pack_tile_c_dim`). These are set during CB creation based on page_size. Stick-sized pages cause incorrect tile geometry, resulting in partial processing (only 16 of 32 rows).

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand RM vs tiled layout, page definitions, and interleaved memory distribution.
   **Key Information**: RM layout: each row = one page. Tiled layout: each 32x32 tile = one page. Interleaved: pages distributed round-robin across banks. Tiles have 4 faces of 16x16 each, stored contiguously in row-major face order.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Needed to understand host-side TensorAccessorArgs setup and device-side usage pattern.
   **Key Information**: Host creates `TensorAccessorArgs(buffer)` and appends to compile-time args. Device creates `TensorAccessorArgs<offset>()` then `TensorAccessor(args, addr, page_size)`. The `get_noc_addr(page_id, accessor)` function handles bank mapping. Page size for RM tensors = stick_size (full row width in bytes).

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Needed to understand the compute helper's CB synchronization and block processing.
   **Key Information**: The helper's `WaitBlock` mode calls `cb_wait_front(input_cb, block_width_tiles)` per block. It calls `cb_pop_front(input_cb, block_width_tiles)` after each block. The `fast_tilize` path is auto-selected for 32x32 tiles with Float32/Float16_b format. Asymmetric mode (row-sized input pages) is supported via `total_input_pages` parameter but NOT used in this program factory.
