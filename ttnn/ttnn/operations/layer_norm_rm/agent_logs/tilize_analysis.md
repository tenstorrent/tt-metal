# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The tilize operation converts a row-major (RM) tensor into tile layout (32x32 tiles). The input tensor is stored in DRAM as interleaved row-major sticks; the output is written in DRAM as interleaved tiles. This analysis focuses on the **reader kernel pattern**, **input CB sizing**, **stick-to-tile batching**, and **core distribution strategy**, as these are the aspects most relevant to a new `layer_norm_rm` operation that reads RM interleaved input.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (one "block" = one row of tiles = `ntiles_per_block` tiles along the width) |
| **Unit size** | `ntiles_per_block` tiles (equivalently, 32 RM sticks each of width `padded_shape[-1]`) |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` = number of tile-rows in the tensor |
| **Loop structure** | Outer loop over tile-rows (blocks), inner loop reads 32 sticks per block |

A **block** corresponds to one tile-row: 32 consecutive row-major sticks spanning the full tensor width. Each block produces `ntiles_per_block` output tiles (the tiles that fit across the width at tile height 32).

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (e.g. [N, C, H, W]) |
| **Dimension convention** | Last dim is width (contiguous in memory) |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16 or FLOAT32 |

**Page definition for RM interleaved**: One page = one stick = one full row of the flattened 2D representation. Page size = `padded_shape[-1] * element_size()` bytes. Pages are distributed round-robin across DRAM banks.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input (padded to tile alignment) |
| **Dimension convention** | Same |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input (or specified `output_dtype`) |

### Layout Transformations

The operation converts ROW_MAJOR to TILE_LAYOUT. The reader loads raw RM sticks into the input CB. The compute kernel's hardware unpacker performs the actual data rearrangement from row-major order into 32x32 tiles with 16x16 face sub-structure. The writer emits tiles (one tile page at a time) to DRAM.

## Data Flow Pattern

### Step-by-step flow

1. **Reader** resolves NoC addresses for 32 consecutive sticks (one tile-height) using `TensorAccessor::get_noc_addr(stick_id)`.
2. **Reader** calls `cb_reserve_back(c_0, ntiles_per_block)` to reserve space for one full row of tiles in the input CB.
3. **Reader** issues 32 `noc_async_read()` calls, one per stick, each reading `block_width_size` bytes (= full stick width). The 32 sticks are written contiguously into the CB starting at `get_write_ptr(c_0)`.
4. **Reader** calls `noc_async_read_barrier()` then `cb_push_back(c_0, ntiles_per_block)` to signal data is ready.
5. **Compute** calls `cb_wait_front(c_0, ntiles_per_block)` to wait for one block of input.
6. **Compute** calls `tilize_block(c_0, ntiles_per_block, c_16)` which uses the hardware unpacker to rearrange the 32 RM sticks into `ntiles_per_block` tiles in the output CB.
7. **Compute** calls `cb_push_back(c_16, ntiles_per_block)` and `cb_pop_front(c_0, ntiles_per_block)`.
8. **Writer** waits on `c_16`, reads one tile at a time, writes to DRAM via `noc_async_write_page()`.
9. Steps 1-8 repeat for each block assigned to this core.

### Key insight for layer_norm_rm

The reader pattern here -- batching 32 sticks (one tile-height) at a time and pushing them as `ntiles_per_block` CB pages -- is specific to the tilize format conversion. For a `layer_norm_rm` operation that stays in RM format, the reader would instead read individual sticks (or groups of sticks) as RM pages. The TensorAccessor usage and `get_noc_addr(stick_id)` pattern is directly transferable.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (RM sticks) | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tiles) | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### CB sizing details (focus: input CB c_0)

The input CB is configured with:
- **num_pages** = `ntiles_per_block` (tiles across the width)
- **page_size** = `input_single_tile_size` = `tt::tile_size(input_cb_data_format)` (e.g. 2048 bytes for BFLOAT16 32x32 tiles)
- **Total capacity** = `ntiles_per_block * input_single_tile_size` bytes

**Critical observation**: The CB page size is set to `input_single_tile_size` (tile-sized), but the reader writes raw RM sticks into it. This works because:
- One block of 32 sticks x full_width = `ntiles_per_block` tiles worth of data in bytes
- `32 sticks * stick_width_bytes = ntiles_per_block * tile_size_bytes` (since `stick_width = ntiles_per_block * TILE_WIDTH * element_size` and `tile_size = TILE_HW * element_size` for 32x32 tiles)
- The CB is used as a raw byte buffer; the "page" abstraction is just for capacity accounting

The reader writes sticks contiguously from `get_write_ptr(c_0)` and the tilize hardware unpacker reads them back in tile order.

### Buffering classification

Both CBs are **single-buffered** (capacity equals one block). The reader must wait for the compute to consume the current block before it can write the next one. There is no overlap between reading block N+1 and computing block N.

## Pipeline Pattern Summary

Single-buffered for both input and output CBs. No read/compute overlap. This is a straightforward sequential pipeline: read block -> compute block -> write block. For a new operation, double-buffering (capacity = 2 * block_size) would allow the reader to prefetch the next block while compute processes the current one.

## Index Calculations

### Stick ID to NoC Address (Reader)

The reader uses `TensorAccessor` to map stick IDs to physical DRAM addresses:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args starting at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

`get_noc_addr(stick_id, s)` internally:
1. Computes `bank_id = stick_id % num_banks` (round-robin interleaving)
2. Computes `bank_offset = (stick_id / num_banks) * aligned_page_size + bank_base_address`
3. Looks up physical (x,y) coordinates of the DRAM bank
4. Returns a 64-bit NoC address encoding (x,y) in upper bits and address in lower bits

### Stick ID progression

Each core starts at `row_start_id = sum of sticks for all prior cores`:
- Full cores: `row_start_id += TILE_HEIGHT * nblocks_per_core` per core
- The reader loops: `stick_id` starts at `start_stick_id` and increments by 1 for each stick, processing 32 sticks (one tile-height) per outer loop iteration

### Host-side setup of TensorAccessor

```cpp
std::vector<uint32_t> reader_ct_args = {block_size_nbytes};  // CT arg 0: stick_size
TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);  // CT args 1+: accessor metadata
```

The `TensorAccessorArgs` with default `ArgConfig::None` puts all accessor parameters (rank, num_banks, tensor_shape, shard_shape, bank_coords) into compile-time args. This makes the on-device `TensorAccessor` construction zero-cost (everything is constexpr).

## Memory Access Patterns

### Read Pattern (Reader Kernel)

1. **Stick grouping**: 32 consecutive sticks are read per outer iteration (one tile-height worth)
2. **Address pre-computation**: All 32 NoC addresses are resolved upfront into `base_src_noc_addr[32]` before any reads are issued
3. **Sequential within group**: The 32 reads proceed in stick order (stick 0, stick 1, ..., stick 31), each reading `block_width_size` bytes
4. **Stride pattern**: Within a group, each stick is at a different DRAM bank (due to interleaving). The sticks are logically contiguous rows but physically distributed across banks.
5. **Partial width advancing**: After reading all 32 sticks for one tile-column group, `base_src_noc_addr[k] += width_size` advances each stick's read pointer. However in the interleaved factory, `num_full_blocks_in_row = 1` so the full stick width is read in one shot.
6. **Barrier**: `noc_async_read_barrier()` after all 32 reads ensures data is in L1 before signaling the CB.

**Key pattern**: The reader reads one complete tile-row (32 sticks x full width) per CB push. This is a "batch of sticks" pattern.

### Write Pattern (de-emphasized)

The writer reads one tile at a time from the output CB and writes it to DRAM using `noc_async_write_page()`, sequentially by tile ID.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_size.x * grid_size.y` (full compute grid) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (tile-rows), cliff core gets `nblocks_per_core_cliff` |
| **Load balancing** | Near-equal, with optional cliff core for remainder |

### Work splitting algorithm (`split_blocks_for_tilize`)

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)` -- target blocks per core
2. Compute `ncores = ceil(nblocks / nblocks_per_core)` -- actual cores needed
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder for last core (0 means no cliff)
4. First `ncores - has_cliff` cores each get `nblocks_per_core` blocks
5. Last core (if cliff) gets `nblocks_per_core_cliff` blocks

Cores are assigned linearly from the `available_grid` using `corerange_to_cores()`.

### Per-core runtime args setup

For each full core `i`:
- `num_sticks = nblocks_per_core * TILE_HEIGHT` (total sticks to process)
- `row_start_id = TILE_HEIGHT * nblocks_per_core * i` (first stick ID)
- `tile_start_id = ntiles_per_block * nblocks_per_core * i` (first output tile ID)

## Arguments

### Compile-Time Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` (block_size_nbytes) | uint32_t | Size of one RM stick in bytes: `padded_shape[-1] * element_size()` |
| 1+ | TensorAccessor args | uint32_t[] | Accessor metadata (rank, num_banks, tensor_shape, bank_coords) for interleaved DRAM buffer |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Base address of the source buffer in DRAM |
| 1 | `num_sticks` | uint32_t | Total number of sticks this core processes (`nblocks * TILE_HEIGHT`) |
| 2 | `block_size_nbytes` | uint32_t | Stick size in bytes (same as CT arg 0, passed redundantly) |
| 3 | `num_tiles_per_block` | uint32_t | Number of tiles across the width (`ntiles_per_block`) |
| 4 | `block_width_size` | uint32_t | Width of one full block in bytes (= stick size for full-width blocks) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for interleaved (full row read in one shot) |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 for interleaved |
| 7 | `leftover_width` | uint32_t | Always 0 for interleaved |
| 8 | `start_stick_id` | uint32_t | First stick ID for this core's work range |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks (tile-rows) this core processes |
| 1 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (`ntiles_per_block`) |

### Compile-Time Arguments (Writer) (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out` | uint32_t | Output CB index (c_16) |
| 1+ | TensorAccessor args | uint32_t[] | Accessor metadata for output DRAM buffer |

### Runtime Arguments (Writer) (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address |
| 1 | `num_tiles` | uint32_t | Total tiles for this core |
| 2 | `start_tile_id` | uint32_t | First output tile ID |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 sticks per block, push as ntiles_per_block pages |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Creates `TensorAccessor` from compile-time args (index 1+) with `src_addr` and `stick_size`
  - Pre-resolves 32 NoC addresses into `base_src_noc_addr[32]` array before issuing reads
  - Uses a lambda `read_tiles()` that: reserves CB space, issues 32 `noc_async_read()` calls (one per stick), barriers, then pushes
  - Outer loop: `num_sticks / tile_height` iterations (= number of blocks)
  - Inner loop structure: (1) resolve 32 addresses, (2) for each block-in-row call `read_tiles()`
  - For interleaved factory, `num_full_blocks_in_row = 1`, so the inner block loop runs once per tile-row

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 | CB c_16 | tilize (RM to tile format conversion) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Uses `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_tile_cnt, per_core_block_cnt)`
  - The helper handles the full loop: for each block, waits on input CB, reserves output CB, calls hardware `tilize_block()`, pushes output, pops input
  - Automatically selects `fast_tilize` when conditions are met (32x32 tiles, Float32 or Float16_b format, half-sync dest mode)

### Writer Kernel (de-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write one tile at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

## Implementation Notes

### Relevance to layer_norm_rm

The following patterns from tilize are directly applicable to a `layer_norm_rm` reader:

1. **TensorAccessor for RM interleaved input**: The host-side setup (`TensorAccessorArgs(*buffer).append_to(ct_args)`) and device-side usage (`TensorAccessor(args, addr, page_size)` + `get_noc_addr(page_id)`) is the standard pattern for reading interleaved RM sticks from DRAM.

2. **Stick-based page model**: For RM interleaved tensors, one page = one stick = one row of the flattened 2D view. The page size is `width * element_size`. The TensorAccessor maps `stick_id` -> `(bank, offset)` using round-robin interleaving.

3. **Batching 32 sticks**: The tilize reader batches 32 sticks (one tile-height) because the tilize compute kernel needs exactly 32 sticks to form tiles. For `layer_norm_rm`, the natural batch unit is one stick (one row) since layer norm operates along the last dimension (width) of each row independently.

4. **CB page size vs stick size**: In tilize, the CB page size is set to `tile_size` even though RM sticks are written -- this only works because the byte counts align. For `layer_norm_rm`, the input CB page size should match the actual stick size (or tile size if tilizing internally).

5. **Work splitting by blocks of rows**: `split_blocks_for_tilize` distributes "blocks" (groups of 32 rows) across cores. For `layer_norm_rm`, work can be split by individual rows or groups of rows since each row's normalization is independent.

6. **Single-buffered CBs**: This factory uses single-buffering. For better performance in a new operation, consider double-buffering the input CB to overlap reading with compute.

### Edge cases

- **fp32_llk_acc**: When input dtype is FLOAT32, the compute kernel enables FP32 destination accumulation (`fp32_dest_acc_en`).
- **Cliff core handling**: The cliff core gets different compile-time args for compute (`nblocks_per_core_cliff` instead of `nblocks_per_core`), meaning two separate `CreateKernel` calls are needed if there is a cliff.
- **Sub-core grids**: The operation supports optional sub-core grids (a subset of the full compute grid) passed via `TilizeParams::sub_core_grids`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the TensorAccessor work for interleaved row-major tensors? How does get_noc_addr(page_id) map a page (stick) ID to a physical DRAM address?"
   **Reason**: Needed to understand the TensorAccessor's address resolution for RM interleaved tensors, which is the core mechanism the reader kernel uses.
   **Key Findings**: The mapping is round-robin: `bank_id = page_id % num_banks`, `bank_offset = (page_id / num_banks) * aligned_page_size + bank_base_address`. The 64-bit NoC address encodes physical (x,y) bank coordinates in upper bits and the address in lower bits. Construction is zero-cost when all args are compile-time.

2. **Query**: "How does cb_reserve_back and cb_push_back work? What is the relationship between num_tiles and byte size?"
   **Reason**: Needed to understand how the CB translates "number of pages" into bytes for the reader's reserve/push calls.
   **Key Findings**: The CB stores `fifo_page_size` as metadata. When `cb_reserve_back(cb, num_pages)` is called, it computes `num_pages * fifo_page_size` to determine bytes. The terms "tiles" and "pages" are interchangeable in the CB API -- they both refer to one unit of the CB's configured page size.

3. **Query**: "How does the tilize compute kernel handle the format conversion from RM sticks to tiles?"
   **Reason**: Needed to understand how the CB can have tile-sized pages yet receive RM stick data.
   **Key Findings**: The conversion happens in the hardware unpacker. `tilize_block()` calls `llk_unpack_tilize_block()` which reads RM data from the input CB and rearranges it into tile format. The math unit (with tilize-enable flag) performs a datacopy. The CB is used as a raw byte buffer -- the page size just needs to account for the same total bytes.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding host-side and device-side TensorAccessor setup
   **Key Information**: `TensorAccessorArgs(buffer)` extracts all metadata from the buffer. `append_to(ct_args)` appends compile-time args. Device-side `TensorAccessorArgs<base_idx>()` reconstructs from compile-time args. `get_noc_addr(page_id)` is the primary API for address resolution.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM vs tiled page definitions and interleaved memory layout
   **Key Information**: For RM layout, each row = one page. For tiled layout, each 32x32 tile = one page. Interleaved distributes pages round-robin across banks. Allocation always starts at bank 0.

3. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the three-kernel architecture and CB synchronization model
   **Key Information**: Reader (RISCV_0/NOC0), Compute (RISCV_2), Writer (RISCV_1/NOC1). CBs are producer-consumer queues in L1 SRAM. Reader writes into CB, compute waits/reads/writes, writer waits/reads.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute kernel's tilize helper library
   **Key Information**: `compute_kernel_lib::tilize<>()` is a templated helper that handles init, the main block loop (wait_front, reserve_back, tilize_block, push_back, pop_front), and uninit. Supports symmetric (same CB page sizes) and asymmetric (different page sizes) modes. Auto-selects fast_tilize when hardware conditions are met.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the 1D block-splitting algorithm for core distribution
   **Key Information**: `split_blocks_for_tilize(grid, nblocks)` returns `{ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff}`. Uses `ceil(nblocks/grid_area)` for blocks per core, last core handles remainder.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function
   **Key Information**: `create_cb(cb_index, program, cores, page_size, num_pages, data_format)` creates a CircularBufferConfig with `num_pages * page_size` total size and sets page_size for the given CB index.
