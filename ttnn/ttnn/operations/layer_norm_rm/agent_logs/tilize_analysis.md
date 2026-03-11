# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) interleaved tensor into a tiled (TILE_LAYOUT) interleaved tensor. It reads contiguous RM sticks from DRAM, groups them into tile-height batches of 32 sticks, and uses the hardware tilize unit to reformat the data into 32x32 tiles (each internally organized as four 16x16 faces).

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Analysis focus**: Reader kernel pattern (how RM sticks are read from DRAM), input CB sizing and page format, stick-to-tile batching (how many sticks per CB push), work distribution unit and core assignment strategy. This analysis serves as an input_stage reference for a `layer_norm_rm` operation.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (= one tile-row of tiles) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / 32` tiles horizontally) |
| **Total units** | `nblocks` = `physical_volume / TILE_HW / ntiles_per_block` = total number of tile-rows |
| **Loop structure** | Outer: iterate over tile-rows (blocks). Inner: for each block, read 32 sticks, tilize into `ntiles_per_block` tiles |

A **block** represents one horizontal row of tiles, which is the natural work unit because tilization requires grouping exactly 32 consecutive sticks (one tile-height) before the data can be rearranged into tiles. Each block produces `ntiles_per_block` output tiles.

---

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [W, Z, Y, X] (any rank; outer dims flattened) |
| **Dimension convention** | Inner dim X is the row width |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page granularity** | One stick = one row of the innermost dimension |
| **Page size (bytes)** | `padded_shape[-1] * element_size()` |

For a row-major interleaved tensor, each page is one stick (one row of width X). Pages are distributed round-robin across DRAM banks. The stick_id is the linear index of the row, starting from 0.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile alignment) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input |
| **Page granularity** | One tile = 32x32 elements |

### Layout Transformation

The tilize operation performs:
- **Input**: RM sticks (page = one row of width X, stored contiguously per row)
- **Output**: Tiled format (page = one 32x32 tile, stored as four 16x16 faces: face0, face1, face2, face3)

No padding is applied by this program factory variant (padding is handled by a separate padded tilize variant). The input tensor's height must already be a multiple of 32 and width a multiple of 32.

---

## Data Flow Pattern (Reader-Focused)

### Step-by-Step Flow

1. **Reader kernel** reads 32 consecutive RM sticks from DRAM into input CB (c_0), one tile-row at a time.
2. **Compute kernel** waits for the sticks in the input CB, performs hardware tilize to convert RM data to tiled format, writes tiles to output CB (c_16).
3. **Writer kernel** writes tiles from output CB to DRAM.

### Detailed Reader Flow

For each block (tile-row of tiles):

```
for each block i in [0, num_sticks/32):
    // Phase 1: Resolve DRAM addresses for 32 sticks
    for j in [0, 32):
        base_src_noc_addr[j] = get_noc_addr(stick_id, tensor_accessor)
        stick_id++

    // Phase 2: Read all 32 sticks into CB as one unit
    cb_reserve_back(c_0, ntiles_per_block)  // reserve space for all tiles in the row
    l1_write_addr = get_write_ptr(c_0)
    for k in [0, 32):
        noc_async_read(base_src_noc_addr[k], l1_write_addr, block_width_size)
        l1_write_addr += block_width_size
        // base_src_noc_addr[k] += block_width_size  (for multi-block-in-row, not used here)
    noc_async_read_barrier()
    cb_push_back(c_0, ntiles_per_block)
```

**Key insight**: The reader reserves `ntiles_per_block` tiles in the CB, but physically writes 32 full-width sticks contiguously into that space. The data lands in the CB in row-major order (stick 0, stick 1, ..., stick 31). The compute kernel's tilize hardware then reinterprets this contiguous row-major data as tiles.

### Critical Pattern: Stick Batching Into CB

The reader batches exactly **32 sticks** per CB push. This is because:
- A tile has height 32 (TILE_HEIGHT = 32)
- The tilize hardware unit expects 32 consecutive rows of data to produce one row of tiles
- The CB page size is set to `input_single_tile_size` (tile-sized pages), but the reader writes 32 sticks contiguously, which occupies `ntiles_per_block` tile-worth of space

The **CB reservation count** is `ntiles_per_block` (number of tiles in one horizontal row), and the data written is `32 * stick_width_bytes = 32 * padded_shape[-1] * element_size`. This works because `ntiles_per_block * tile_size_bytes = ntiles_per_block * (32 * 32 * element_size) = 32 * (ntiles_per_block * 32 * element_size) = 32 * stick_width_bytes`.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| c_0 | input | Input staging (RM sticks) | `ntiles_per_block` tiles | `input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | output | Output staging (tiled) | `ntiles_per_block` tiles | `output_single_tile_size` | Single | Compute | Writer | Block |

### Input CB (c_0) Deep Dive

- **Capacity**: `ntiles_per_block` pages, where each page is `input_single_tile_size` bytes (e.g., 2048 bytes for BFLOAT16 32x32 tiles).
- **Total capacity bytes**: `ntiles_per_block * input_single_tile_size`
- **How reader uses it**: Reserves `ntiles_per_block` pages, then writes 32 full-width sticks (= `32 * block_width_size` bytes) into the reserved space. The math works out because `ntiles_per_block * tile_size = (width/32) * (32*32*elem_size) = width * 32 * elem_size = 32 * stick_size`.
- **Buffering**: Single-buffered. The capacity equals the block size (`ntiles_per_block`), so reader and compute cannot overlap on different blocks. However, within a block the reader issues all 32 async reads before the barrier, exploiting NoC-level parallelism.
- **Data format in CB**: The page_size is tile-sized for compute kernel compatibility, but the reader writes raw RM stick data. The compute kernel's tilize unit handles the RM-to-tile reformatting.

### Key Takeaway for layer_norm_rm

For a new operation that reads RM sticks, the input CB should be sized to hold one tile-row worth of data. The CB page_size should match tile size for the compute kernel, even though the reader writes stick-by-stick. The `create_cb` call with `ntiles_per_block` pages and tile-sized page_size is the pattern to follow.

---

## Pipeline Pattern Summary

Both CBs are single-buffered (capacity = block size = `ntiles_per_block`). This means:
- Reader and compute do NOT overlap across blocks (reader must finish a full block before compute can start)
- Within a block, the reader issues 32 async reads overlapping NoC transfers
- Compute and writer similarly do not overlap across blocks

---

## Index Calculations

### Stick-to-DRAM Address Mapping

The reader uses `TensorAccessor` to map `stick_id` to a DRAM NOC address:

```cpp
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

The `get_noc_addr(stick_id, s)` call translates to `s.get_noc_addr(stick_id)`, which:
1. Computes `bank_id = stick_id % num_banks` (round-robin interleaving)
2. Computes `offset_within_bank = (stick_id / num_banks) * page_size + bank_base_address`
3. Returns a 64-bit NOC address encoding the bank's (x, y) coordinates and the offset

### Stick ID Assignment Per Core

Each core gets a contiguous range of sticks:
- Full cores: `start_stick_id = TILE_HEIGHT * nblocks_per_core * core_index`, processes `nblocks_per_core * TILE_HEIGHT` sticks
- Cliff core (last core): starts where the previous core left off, processes `nblocks_per_core_cliff * TILE_HEIGHT` sticks

This ensures each core processes a contiguous chunk of the tensor's rows.

---

## Memory Access Patterns

### Read Pattern

- **Pattern**: Sequential stick reads within a 32-stick group, with stride between sticks being `block_width_size` bytes in L1 write space
- **DRAM access**: Each stick is on a potentially different DRAM bank (due to interleaved round-robin). For a 32-stick group, up to 32 different banks may be accessed. This provides good DRAM bandwidth utilization.
- **NoC transfers**: All 32 reads in a group are issued as `noc_async_read` before a single barrier. This allows maximum NoC and DRAM parallelism.
- **L1 layout**: Sticks are written contiguously in L1: stick0 at offset 0, stick1 at offset `block_width_size`, ..., stick31 at offset `31 * block_width_size`. This is exactly the row-major format the tilize hardware expects.

### Write Pattern (de-emphasized)

Writer writes one tile at a time to DRAM using sequential tile IDs and `noc_async_write_page`.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear list of cores) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` cores |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows), cliff core gets `nblocks_per_core_cliff` |
| **Load balancing** | Near-equal with at most one cliff core |

### Work Splitting Algorithm (`split_blocks_for_tilize`)

```
nblocks_per_core = ceil(nblocks / grid_area)
ncores = ceil(nblocks / nblocks_per_core)
nblocks_per_core_cliff = nblocks % nblocks_per_core  (0 means no cliff)
```

- All cores except possibly the last one get exactly `nblocks_per_core` blocks.
- The last core (if `nblocks_per_core_cliff > 0`) gets the remainder, which is always less than `nblocks_per_core`.
- Cores are assigned from the available grid in linear order using `corerange_to_cores()`.
- The compute kernel is compiled separately for full cores vs. cliff core (different `per_core_block_cnt` compile-time arg).

### Example

For a tensor [1, 1, 128, 64]:
- ntiles_per_block = 64/32 = 2 tiles per block
- nblocks = 128/32 = 4 blocks
- With 4 available cores: each core gets 1 block (32 sticks -> 2 tiles)

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one RM stick in bytes (`padded_shape[-1] * element_size`) |
| 1+ | TensorAccessorArgs | multiple uint32_t | Encodes bank layout, interleaving info for DRAM access (appended by `TensorAccessorArgs(*src0_buffer).append_to(ct_args)`) |

The `TensorAccessorArgs` with default `ArgConfig::None` means ALL accessor parameters (rank, num_banks, tensor_shape, shard_shape, bank_coords) are compile-time. This makes `get_noc_addr` calls very efficient (no runtime parameter fetching).

On the device side, the accessor is constructed as:
```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // starts at CT arg index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tile-rows) this core processes |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (`ntiles_per_block`) |

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | multiple uint32_t | DRAM bank info for output buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total sticks to read (`nblocks * TILE_HEIGHT`) |
| 2 | block_size_nbytes | uint32_t | (unused in kernel -- arg index 2 is not read) |
| 3 | ntiles_per_block | uint32_t | Tiles per block / row of tiles |
| 4 | block_width_size | uint32_t | Width of one stick in bytes (same as stick_size) |
| 5 | num_full_blocks_in_row | uint32_t | Always 1 for this variant |
| 6 | num_leftover_tiles | uint32_t | Always 0 for this variant |
| 7 | leftover_width | uint32_t | Always 0 for this variant |
| 8 | start_stick_id | uint32_t | First stick ID for this core |

Note: Args 5-7 support a "split row" mode (where a row is wider than the CB can hold), but in this interleaved variant they are fixed to 1/0/0 because the entire row width fits in one block.

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_tiles | uint32_t | Total tiles this core writes |
| 2 | start_tile_id | uint32_t | First tile ID for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per block via noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Outer loop: iterates over `num_sticks / 32` blocks (tile-height groups).
  - For each block, first resolves all 32 DRAM addresses into `base_src_noc_addr[32]` array.
  - Then calls `read_tiles(ntiles_per_block, block_width_size)` which:
    1. `cb_reserve_back(c_0, ntiles_per_block)` -- reserves space for one tile-row
    2. Issues 32 `noc_async_read` calls, each reading one stick of `block_width_size` bytes
    3. `noc_async_read_barrier()` -- waits for all 32 reads to complete
    4. `cb_push_back(c_0, ntiles_per_block)` -- signals data is ready for compute
  - The `num_full_blocks_in_row` loop (always 1 here) allows for future support of splitting wide rows across multiple CB fills.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize_block (HW tilize unit) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize srcA/srcB and output CB.
  - Calls `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(ntiles_per_block, nblocks_per_core)`.
  - The tilize helper's main loop (per block):
    1. `cb_wait_front(c_0, ntiles_per_block)` -- waits for reader to fill input CB
    2. `cb_reserve_back(c_16, ntiles_per_block)` -- reserves output CB space
    3. `tilize_block(c_0, ntiles_per_block, c_16)` or `fast_tilize_block(...)` -- hardware tilize
    4. `cb_push_back(c_16, ntiles_per_block)` + `cb_pop_front(c_0, ntiles_per_block)`
  - The `fast_tilize` path is automatically selected at compile time when tile dims are 32x32 and data format is Float32 or Float16_b (bfloat16), which covers all typical cases.

### Writer Kernel (de-emphasized)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- Generic tile writer: waits for one tile in output CB, writes to DRAM, pops.

---

## Implementation Notes

### Why the Reader Pre-computes All 32 NOC Addresses

The reader resolves all 32 stick addresses into the `base_src_noc_addr[32]` array before issuing any reads. This is because:
1. Each stick may reside on a different DRAM bank (interleaved round-robin), so the NOC address computation involves bank ID lookup and offset calculation.
2. Pre-computing avoids interleaving address computation with NoC read commands, reducing latency.
3. The array is stack-allocated (32 entries of uint64_t = 256 bytes) which fits in L1.

### TensorAccessor Pattern for Interleaved RM Tensors

The `TensorAccessorArgs` on the host side serializes all bank metadata (rank, num_banks, tensor_shape, shard_shape, bank_coords) into compile-time arguments. On the device, `TensorAccessorArgs<1>()` reconstructs the metadata from compile-time args starting at index 1 (index 0 is `stick_size`). The `TensorAccessor(args, src_addr, stick_size)` then provides `get_noc_addr(stick_id)` to map any stick index to its DRAM NOC address.

This pattern is the recommended way to access interleaved tensor pages in new kernels.

### The "Block" Concept Clarification

In this operation, a "block" has a dual meaning:
1. **From the compute perspective**: A block = `ntiles_per_block` tiles arranged horizontally (one tile-row).
2. **From the reader perspective**: A block = 32 consecutive RM sticks (one tile-height worth of rows).

These are equivalent because 32 sticks of width W produce exactly `W/32` tiles side by side.

### Relevance to layer_norm_rm

For a `layer_norm_rm` operation that reads RM sticks, tilizes in-kernel, computes, then untilizes:
1. **Reader pattern to reuse**: Read 32 sticks at a time, use `cb_reserve_back(cb, ntiles_per_block)`, write sticks contiguously, then `cb_push_back(cb, ntiles_per_block)`.
2. **Input CB sizing**: `ntiles_per_block` pages at tile size = space for 32 full-width sticks.
3. **TensorAccessor setup**: Use `TensorAccessorArgs(*buffer).append_to(ct_args)` on host, `TensorAccessorArgs<offset>()` + `TensorAccessor(args, addr, stick_size)` in kernel.
4. **Work distribution**: Use `split_blocks_for_tilize()` or a similar 1D block-splitting strategy to distribute tile-rows across cores.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? What is the relationship between row-major sticks and tiles during tilization? How does the reader kernel batch 32 sticks into a tile-height group for the compute kernel?"
   **Reason**: Needed to understand the fundamental mechanism of how RM data is converted to tiles, specifically the relationship between sticks and tiles.
   **Key Findings**: Tilize is a three-phase operation (unpack, math/datacopy, pack). 32 RM sticks are grouped to form the height of a tile. Each tile is internally structured as four 16x16 faces. The `tilize_block` hardware operation handles the RM-to-tile reformatting.

2. **Query**: "How does TensorAccessorArgs work on the host side and the device side?"
   **Reason**: Needed to understand how the reader kernel resolves stick_id to DRAM addresses via the TensorAccessor API.
   **Key Findings**: Query failed. Information was obtained from reading `tensor_accessor_args.hpp`, `tensor_accessor.h`, and the tensor_accessor tech report instead.

3. **Query**: "In tt-metal, when a row-major interleaved tensor is stored in DRAM, what is the page granularity?"
   **Reason**: Needed to confirm that each page is one RM stick for interleaved RM tensors.
   **Key Findings**: Query failed. Confirmed from `tech_reports/tensor_layouts/tensor_layouts.md` Section 3.1: "Each row of a 2D tensor corresponds to a single page."

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand tensor page granularity for RM vs tiled layouts, and interleaved memory distribution.
   **Key Information**: RM layout: page = one row. Tiled layout: page = one tile. Interleaved: pages distributed round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Needed to understand TensorAccessor host-side setup and device-side usage for DRAM address resolution.
   **Key Information**: Host: `TensorAccessorArgs(buffer)` serializes bank metadata. Device: `TensorAccessor(args, addr, page_size)` provides `get_noc_addr(page_id)`. All-compile-time config is default and most efficient.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Needed to understand the compute kernel's tilize implementation, especially the `WaitBlock` synchronization pattern.
   **Key Information**: The tilize helper waits for `ntiles_per_block` input pages per block, calls `tilize_block` or `fast_tilize_block`, then pops. Fast tilize is auto-selected for 32x32 tiles with Float32/Float16_b format.

4. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Needed to understand how blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize` divides blocks evenly, with at most one cliff core getting the remainder. Returns ncores, core ranges, and blocks per core.

5. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Needed to understand the `create_cb` helper used for circular buffer creation.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CB with `num_pages * page_size` total capacity.

6. **Source**: `METALIUM_GUIDE.md` (CB sections)
   **Reason**: Needed to verify CB API semantics (reserve_back, push_back, wait_front, pop_front).
   **Key Information**: Confirmed reader uses reserve_back/push_back, compute uses wait_front/pop_front + reserve_back/push_back, writer uses wait_front/pop_front. These are the standard synchronization primitives.

7. **Source**: `tt_metal/hw/inc/api/dataflow/dataflow_api.h`
   **Reason**: Needed to understand `noc_async_read_page` and `noc_async_read` semantics.
   **Key Information**: `noc_async_read(noc_addr, l1_addr, size)` initiates an async DMA read from a DRAM bank to local L1. `noc_async_read_barrier()` waits for all pending reads to complete.

8. **Source**: `tt_metal/hw/inc/api/tensor/tensor_accessor.h`
   **Reason**: Needed to understand how `get_noc_addr(page_id)` works on the TensorAccessor.
   **Key Information**: TensorAccessor computes bank_id from page_id using interleaving, gets bank NOC coordinates, and computes the offset within the bank. Returns a 64-bit NOC address suitable for `noc_async_read`.
