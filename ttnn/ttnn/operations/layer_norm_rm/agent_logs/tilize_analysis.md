# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) interleaved tensor into tiled (TILE_LAYOUT) interleaved format. Each 32x32 block of RM elements is reorganized into the hardware-native tile format (four 16x16 faces in row-major face order). This analysis covers the **multi-core interleaved** program factory variant.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Role in this document**: This analysis is an **input_stage reference** for a new `layer_norm_rm` operation. The focus is on the reader kernel pattern, input CB sizing, stick-to-tile batching, and work distribution -- the parts most directly reusable when building a fused RM-input operation.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (one tile-row = one row of tiles along the width dimension) |
| **Unit size** | `ntiles_per_block` tiles, where `ntiles_per_block = padded_shape[-1] / TILE_WIDTH` |
| **Total units** | `nblocks = physical_volume / TILE_HW / ntiles_per_block` (equivalently, total tile-rows in the tensor) |
| **Loop structure** | Outer loop over tile-rows (groups of 32 sticks), inner loop over width blocks |

A **block** here is one complete row of tiles -- all tiles spanning the width dimension at a particular 32-row boundary. For a tensor with shape `[..., H, W]` padded to tile alignment, there are `H / 32` blocks, each containing `W / 32` tiles.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[N, C, H, W]` (arbitrary rank, flattened to 2D for memory) |
| **Dimension convention** | Last dim = W (width, contiguous in memory) |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page definition** | One page = one stick = one row of width W |
| **Page size** | `padded_shape[-1] * element_size()` bytes (e.g., W * 2 for bfloat16) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input |
| **Page definition** | One page = one tile = 32x32 elements |
| **Page size** | `tile_size(data_format)` bytes (e.g., 2048 for bfloat16) |

### Layout Transformation

The operation performs an RM-to-tile format conversion. The reader kernel reads 32 consecutive sticks (rows) at a time, packing them contiguously into a CB with tile-sized pages. The compute kernel then applies `tilize_block` which rearranges the 32x32 element blocks into face-ordered tile format (face0, face1, face2, face3 -- each 16x16).

---

## Data Flow Pattern

### Stage 1: Reader (DRAM -> CB c_0)

1. For each group of 32 sticks (one tile-height):
   a. Resolve 32 NoC addresses (one per stick) via `get_noc_addr(stick_id, tensor_accessor)`
   b. For each width-block in the row:
      - `cb_reserve_back(c_0, ntiles_per_block)` -- reserve space for all tiles in one tile-row
      - Read 32 sticks of `block_width_size` bytes each into the CB contiguously
      - `noc_async_read_barrier()` -- wait for all 32 reads to complete
      - `cb_push_back(c_0, ntiles_per_block)` -- signal data is ready

**Key insight for reuse**: The reader packs 32 RM sticks contiguously into tile-sized CB pages. The CB page_size is `input_single_tile_size` (tile size), NOT stick_size. This is critical because `tilize_init()` reads face/tile dimensions from the input CB's metadata -- with stick-sized pages, the hardware gets wrong tile dimensions and only processes 16 of 32 rows. (See MEMORY.md note on this requirement.)

### Stage 2: Compute (CB c_0 -> CB c_16)

1. `compute_kernel_hw_startup(c_0, c_16)` -- initialize unpack/pack hardware
2. `tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(ntiles_per_block, nblocks_per_core)`
   - For each of `nblocks_per_core` blocks:
     - `cb_wait_front(c_0, ntiles_per_block)` -- wait for reader
     - `cb_reserve_back(c_16, ntiles_per_block)` -- reserve output space
     - `tilize_block(c_0, ntiles_per_block, c_16)` -- unpack RM -> pack tiled
     - `cb_push_back(c_16, ntiles_per_block)` / `cb_pop_front(c_0, ntiles_per_block)`

### Stage 3: Writer (CB c_16 -> DRAM)

Writes one tile at a time from CB c_16 to output DRAM using `noc_async_write_page`.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| `c_0` | Input CB | Staging RM sticks in tile-sized pages | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| `c_16` | Output CB | Staging tilized output tiles | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### Critical CB Sizing Details for Input CB (c_0)

- **Capacity**: `ntiles_per_block * input_single_tile_size` bytes total
- **Page size**: `input_single_tile_size` (e.g., 2048 bytes for bfloat16, 4096 for float32)
- **Pages**: `ntiles_per_block` (one per tile in a tile-row)
- **Buffering**: Single-buffered -- capacity equals one block, so reader and compute cannot overlap on the same block. The reader must complete the entire block of 32 sticks before compute can start.

**Why page_size = tile_size (not stick_size)**: The `state_configure<SRCA, PACK>()` inside `tilize_init` reads face and tile dimensions from the CB's metadata. If the CB has stick-sized pages (e.g., 64 bytes for a 32-element bfloat16 row), the hardware gets wrong tile dimensions and only processes 16 of 32 rows, leaving faces 2-3 empty. The reader must pack 32 RM sticks contiguously into one tile-sized page so that the compute kernel sees correct geometry.

---

## Reader Kernel Deep Dive

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`

### How RM Sticks Are Read from DRAM

The reader uses a **batch-address, sweep-width** pattern:

```
for each tile_row (group of 32 sticks):
    // Phase 1: Resolve all 32 NoC addresses
    for j in 0..31:
        base_src_noc_addr[j] = get_noc_addr(stick_id++, tensor_accessor)

    // Phase 2: For each width-block in the row
    for j in 0..num_full_blocks_in_row:
        cb_reserve_back(c_0, ntiles_per_block)
        l1_write_addr = get_write_ptr(c_0)
        for k in 0..31:
            noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size)
            l1_write_addr += width_size
            base_src_noc_addr[k] += width_size  // advance to next width-block
        noc_async_read_barrier()
        cb_push_back(c_0, ntiles_per_block)
```

**Key observations**:

1. **Address pre-computation**: All 32 stick addresses are resolved up front per tile-row. This is efficient because `get_noc_addr` involves a bank lookup via TensorAccessor (page_id -> bank_id + offset).

2. **Contiguous packing into CB**: The 32 reads write contiguously into CB space (`l1_write_addr += width_size` for each stick). After all 32, the CB contains the sticks packed in row order -- exactly the format the tilize compute kernel expects.

3. **Width-block advancement**: `base_src_noc_addr[k] += width_size` advances each stick's read pointer for the next width-block. This handles tensors wider than one tile-row worth of width by splitting into multiple passes.

4. **Full-width reads**: In the interleaved variant, `num_full_blocks_in_row = 1` and `block_width_size = padded_shape[-1] * element_size()`, meaning the entire width is read as one block. The split-rows capability exists for the block variant.

### Stick-to-Tile Batching

- **Sticks per CB push**: 32 (one TILE_HEIGHT worth of rows)
- **Tiles per CB push**: `ntiles_per_block` (all tiles spanning the width)
- **Bytes per push**: `32 * block_width_size` = `32 * padded_shape[-1] * element_size()`
- **This equals**: `ntiles_per_block * tile_size`, confirming the CB capacity is exactly filled by one push

### TensorAccessor Usage

The reader uses `TensorAccessorArgs` for address resolution:

- **Host side**: `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` appends the accessor config (and for non-sharded buffers, just the args_config flags) to compile-time args starting at index 1 (index 0 is `block_size_nbytes`).
- **Device side**: `constexpr auto src_tensor_args = TensorAccessorArgs<1>()` decodes from compile-time arg index 1. Then `TensorAccessor(src_tensor_args, src_addr, stick_size)` creates the accessor with the buffer base address and page size = stick_size.
- **Page ID**: `stick_id` is the linear index of the stick (row) within the RM tensor. The accessor maps this to the correct DRAM bank and offset.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (computed by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows), cliff core gets `nblocks_per_core_cliff` |
| **Load balancing** | Near-equal, with at most one cliff core getting the remainder |

### Work Splitting Algorithm (`split_blocks_for_tilize`)

1. `nblocks_per_core = ceil(nblocks / grid_area)` -- target blocks per core
2. `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for last core (0 if evenly divisible)
4. If `nblocks_per_core_cliff > 0`, the last core is a "cliff" core with fewer blocks

### Per-Core Offset Tracking

The host iterates over cores and tracks:
- `tile_start_id`: starting output tile index (for writer)
- `row_start_id`: starting stick index (for reader)

Each full core advances by `TILE_HEIGHT * nblocks_per_core` sticks and `ntiles_per_block * nblocks_per_core` tiles.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` (block_size_nbytes) | uint32_t | Size of one RM stick in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessorArgs | uint32_t[] | Accessor config for source buffer (bank mapping, DRAM flag, etc.) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) this core processes |
| 1 | `per_core_block_tile_cnt` | uint32_t | Number of tiles per block (`ntiles_per_block`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out` | uint32_t | Output CB index (`c_16`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Accessor config for destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer DRAM address |
| 1 | `num_sticks` | uint32_t | Total sticks to read: `nblocks_per_core * TILE_HEIGHT` |
| 2 | `block_size_nbytes` | uint32_t | Width of a stick in bytes (same as CT arg 0 -- redundant in this variant) |
| 3 | `ntiles_per_block` | uint32_t | Tiles per tile-row |
| 4 | `block_width_size` | uint32_t | Width of a block in bytes (same as block_size_nbytes for full-width) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for interleaved variant |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 for interleaved variant |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 for interleaved variant |
| 8 | `start_stick_id` | uint32_t | First stick index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer DRAM address |
| 1 | `num_pages` | uint32_t | Total tiles to write: `ntiles_per_block * nblocks_per_core` |
| 2 | `start_id` | uint32_t | First output tile index for this core |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks -> pack into tile-sized pages |
| Compute | TRISC (unpack+math+pack) | N/A | CB c_0 | CB c_16 | tilize_block (RM -> tiled format conversion) |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write tiles one-by-one |

### Reader: `reader_unary_stick_layout_split_rows_interleaved.cpp`

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**: Pre-computes 32 NoC addresses per tile-row, then sweeps across width blocks reading 32 sticks per block. For the interleaved variant, there is always exactly 1 full block in the row (the entire width). The `read_tiles` lambda encapsulates the pattern of reserve -> read 32 sticks -> barrier -> push.

### Compute: `tilize.cpp`

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**: Uses `compute_kernel_lib::tilize<c_0, c_16>()` with WaitBlock mode. The helper automatically selects `fast_tilize` when conditions are met (32x32 tiles, Float32 or Float16_b format, half-sync DEST mode). Falls back to standard `tilize_block` otherwise. Each block processes `ntiles_per_block` tiles.

### Writer: `writer_unary_interleaved_start_id.cpp`

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic tile writer. Iterates from `start_id` to `start_id + num_pages`, writing one tile at a time using `noc_async_write_page`. Gets page_size from CB interface.

---

## Pipeline Pattern Summary

Both CBs are single-buffered (capacity = 1 block = `ntiles_per_block` tiles). This means:
- The reader fills the entire input CB with one block of 32 sticks before the compute kernel can start.
- The compute kernel produces one block of tiles into the output CB before the writer can start.
- There is no overlap between reader and compute on the same block, and no overlap between compute and writer on the same block.

However, because the writer processes tiles one-by-one while the compute processes blocks, the writer from block N can overlap with compute on block N+1 once the compute finishes pushing block N+1 and the writer has popped enough tiles.

---

## Implementation Notes

### Reuse Guidance for layer_norm_rm Input Stage

1. **Reader kernel pattern is directly reusable**: The batch-address / sweep-width pattern of reading 32 RM sticks into tile-sized CB pages can be reused or adapted for any operation that takes RM interleaved input and needs tilized data for compute.

2. **CB page_size must be tile_size**: This is the most critical gotcha. Even though the input is RM, the CB holding data for `tilize_block` must have `page_size = tile_size`, not `stick_size`. The reader physically packs 32 sticks contiguously into each tile-sized page.

3. **Work unit = tile-row**: The natural work granularity is one tile-row (32 sticks x full width). For layer_norm_rm, where normalization is along the last dimension (width), each tile-row of 32 sticks could be one work unit -- tilize it, compute norm, untilize, and write back.

4. **TensorAccessor for interleaved RM**: The accessor is created with `page_size = stick_size` (one RM row). The `get_noc_addr(stick_id, accessor)` call maps stick index to the correct DRAM bank + offset. This is the standard pattern for reading interleaved RM data.

5. **Width must be tile-aligned**: `padded_shape[-1]` must be divisible by TILE_WIDTH (32). The `ntiles_per_block = padded_shape[-1] / TILE_WIDTH` calculation assumes this. For layer_norm_rm, the input width must be padded to a multiple of 32.

6. **`split_blocks_for_tilize` for 1D distribution**: This utility divides `nblocks` evenly across available cores with at most one cliff core. It returns `BlockSplit{ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff}`.

7. **Compute kernel has separate compile-time args for full vs cliff cores**: The compute kernel is created twice -- once for the full core range with `{nblocks_per_core, ntiles_per_block}` and once for the cliff range with `{nblocks_per_core_cliff, ntiles_per_block}`. This is because `per_core_block_cnt` is a compile-time arg.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? How does the reader kernel read RM sticks and pack them for the compute kernel?"
   **Reason**: Needed to confirm the end-to-end tilize data flow and the relationship between sticks, tiles, and blocks.
   **Key Findings**: Confirmed that the reader reads RM sticks into CBs, the compute kernel uses a 3-thread pipeline (unpack/math/pack) for the RM->tile conversion, and blocks are groups of tiles processed together. The term "stick" is an internal concept for one row of RM data.

2. **Query**: "What is the create_cb utility function and how does CircularBufferConfig work?"
   **Reason**: Needed to understand CB creation helper and the relationship between page_size, num_pages, and total capacity.
   **Key Findings**: `create_cb(cb_id, program, cores, page_size, num_pages, format)` creates a CB with total size = `page_size * num_pages`. Each push/pop operates on one page. For tilize, page_size = tile_size ensures tile-aligned CB operations.

3. **Query**: "How does TensorAccessorArgs work on the host side (append_to) and device side (template decode)?"
   **Reason**: Needed to understand the host-device argument passing mechanism for tensor address resolution.
   **Key Findings**: For non-sharded buffers, `append_to` adds the args_config uint32_t (flags for DRAM, sharded, runtime config). On the device, `TensorAccessorArgs<CTA_OFFSET>()` reads the config from `get_compile_time_arg_val(CTA_OFFSET)` and computes offsets for subsequent args. The TensorAccessor then maps page_id to bank_id + offset for NoC address calculation.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand RM page definition (one page = one row/stick) and interleaved distribution (round-robin across banks).
   **Key Information**: In RM layout, each row is one page. In tiled layout, each 32x32 tile is one page. Interleaved layout distributes pages round-robin across DRAM banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Needed to understand how `get_noc_addr(page_id, accessor)` maps logical stick IDs to physical DRAM addresses.
   **Key Information**: TensorAccessor maps page_id to (bank_id, offset) and then to a 64-bit NoC address. Supports both compile-time and runtime argument configurations.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Needed to understand the compute-side tilize library API and its CB requirements.
   **Key Information**: The `tilize<input_cb, output_cb>()` function automatically selects `fast_tilize` when hardware supports it (32x32 tiles, Float32/Float16_b, half-sync DEST). In WaitBlock mode, it does `cb_wait_front` on the input CB per block. The asymmetric mode (total_input_pages parameter) is available for when input CB has row-sized pages, but the standard tilize program factory uses the symmetric mode with tile-sized input pages.

4. **Source**: User memory note (MEMORY.md) on tilize CB page size requirement
   **Reason**: Critical correctness constraint discovered in prior work.
   **Key Information**: The input CB for `tilize_block` MUST have `page_size = tile_page_size`. With stick-sized pages, `state_configure<SRCA, PACK>()` gets wrong tile dimensions and only processes 16 of 32 rows, leaving faces 2-3 empty.
