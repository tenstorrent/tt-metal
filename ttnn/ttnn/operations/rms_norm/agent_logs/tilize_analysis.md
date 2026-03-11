# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The **tilize** operation converts tensor data from **row-major (RM) layout** to **tile layout** (32x32 tiles). The input is a row-major tensor stored in interleaved DRAM; the output is the same tensor rearranged into tiled format, also in interleaved DRAM. This is one of the fundamental data movement primitives in TT-Metal, enabling row-major data from host or other operations to be consumed by tile-based compute engines.

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Focus**: This analysis emphasizes the **input stage** -- the reader kernel pattern, input CB sizing, stick-to-tile batching, and work distribution -- as a reference for building an rms_norm operation that reads RM sticks from DRAM.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (a group of 32 consecutive sticks spanning the full padded width) |
| **Unit size** | `ntiles_per_block` tiles (= `padded_shape[-1] / TILE_WIDTH`), equivalent to one row of tiles along the width dimension |
| **Total units** | `nblocks` = `ceil(ntiles / ntiles_per_block)` = total tile rows in the tensor |
| **Loop structure** | Outer loop over tile-height groups of sticks (32 sticks = 1 block), inner loop over width blocks (always 1 for this factory) |

A **block** is defined as 32 consecutive row-major sticks (one tile-height) covering the full padded width. Each block produces `ntiles_per_block` output tiles. The reader pushes one full-width block at a time; the compute kernel tilizes one full-width block at a time.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D, but treated as 2D: `[num_sticks, padded_width]` |
| **Dimension convention** | Last dim is the contiguous stick dimension |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

The input is a row-major tensor. Each **page** (the unit of interleaved distribution) is one stick: one complete row of the flattened 2D representation. The page size equals `padded_shape[-1] * element_size`.

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Tiled (32x32 tiles) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (BFLOAT16 or FLOAT32) |

### Layout Transformation

The entire purpose of this operation is the RM-to-tile conversion. The reader kernel loads 32 sticks at a time into the input CB. The compute kernel's `tilize_block` / `fast_tilize_block` hardware instruction rearranges these 32 contiguous sticks into tile format (32x32 tiles with face decomposition). The writer then writes the resulting tiles to DRAM.

---

## Data Flow Pattern

### Step-by-step flow (per block)

1. **Reader (RISCV_0, NoC0)**: For each block of 32 sticks:
   a. Pre-compute NoC addresses for all 32 sticks using `get_noc_addr(stick_id, tensor_accessor)` -- one address per stick, stored in `base_src_noc_addr[32]`.
   b. Call `cb_reserve_back(c_0, ntiles_per_block)` to reserve the entire block in the input CB.
   c. For each of the 32 sticks: issue `noc_async_read(src_noc_addr, l1_write_addr, width_size)` to DMA one full stick from DRAM into L1, advancing the L1 write pointer by `width_size` after each stick.
   d. Call `noc_async_read_barrier()` to wait for all 32 reads to complete.
   e. Call `cb_push_back(c_0, ntiles_per_block)` to signal the data is ready.

2. **Compute (RISCV_2, FPU/SFPU)**: For each block:
   a. Call `cb_wait_front(c_0, ntiles_per_block)` to wait for reader data.
   b. Call `cb_reserve_back(c_16, ntiles_per_block)` to reserve output space.
   c. Call `tilize_block(c_0, ntiles_per_block, c_16)` or `fast_tilize_block(...)` to convert the 32 RM sticks into tiles.
   d. Call `cb_push_back(c_16, ntiles_per_block)` and `cb_pop_front(c_0, ntiles_per_block)`.

3. **Writer (RISCV_1, NoC1)**: For each tile (one at a time):
   a. Call `cb_wait_front(c_16, 1)`.
   b. Write one tile to DRAM via `noc_async_write_page(tile_id, tensor_accessor, l1_read_addr)`.
   c. Call `cb_pop_front(c_16, 1)`.

### Key insight for rms_norm reuse

The **reader pattern** is the critical piece: it reads RM sticks in groups of 32 (tile-height), assembling a full-width row of tile data in L1. The `base_src_noc_addr[32]` array pre-computes all 32 NoC addresses before issuing reads, enabling efficient DMA pipelining. Each stick is read as a single contiguous `noc_async_read` of `width_size` bytes.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| `c_0` (CBIndex::c_0) | Input CB | Input staging (RM sticks arranged as pseudo-tiles) | `ntiles_per_block` tiles | `input_single_tile_size` | Single | Reader | Compute | Block |
| `c_16` (CBIndex::c_16) | Output CB | Output staging (tilized tiles) | `ntiles_per_block` tiles | `output_single_tile_size` | Single | Compute | Writer | Block |

### Input CB Sizing Details (c_0)

- **num_pages**: `ntiles_per_block` = `padded_shape[-1] / 32`
- **page_size**: `input_single_tile_size` = `tt::tile_size(input_cb_data_format)` (e.g., 2048 bytes for BFLOAT16, 4096 bytes for FLOAT32)
- **Total capacity**: `ntiles_per_block * input_single_tile_size` bytes
- **Buffering**: Single-buffered (capacity == block size). The reader fills the entire CB with one block of 32 sticks, the compute consumes the entire block, then the reader refills. No overlap between reader and compute for consecutive blocks.

**Why tile-sized pages for RM data?** Even though the input is row-major, the CB is configured with tile-sized pages. This is because the tilize hardware instruction expects to read from a CB that contains enough data to form tiles. The reader writes 32 sticks sequentially into the CB space, and the hardware tilize logic reinterprets this contiguous block of `32 * padded_width` elements as tiles. The page_size in the CB config must be tile-sized so the compute kernel's `cb_wait_front` / `cb_pop_front` correctly tracks the number of tile-equivalents.

### Stick-to-Tile Batching

- **Sticks per CB push**: 32 (one tile-height worth of sticks)
- **Tiles per CB push**: `ntiles_per_block` = `padded_width / 32`
- **How it works**: The reader writes `32 * padded_width * element_size` bytes contiguously into the CB. This represents `32 * (padded_width / 32)` = `32 * ntiles_per_block` datum rows, which the tilize instruction rearranges into `ntiles_per_block` tiles of 32x32 each.
- **Reserve/push granularity**: The reader does `cb_reserve_back(c_0, ntiles_per_block)` and `cb_push_back(c_0, ntiles_per_block)` once per block (not once per stick).

---

## Pipeline Pattern Summary

Both CBs are **single-buffered**: capacity equals one full block (`ntiles_per_block` pages). This means:
- Reader and compute **cannot overlap** on consecutive blocks -- the reader must wait until compute finishes consuming the current block before writing the next.
- Compute and writer **can partially overlap** since the writer processes one tile at a time from `c_16` while compute may have already pushed multiple tiles.

---

## Index Calculations

### Reader: Stick ID to NoC Address

The reader uses `TensorAccessor` to map stick IDs to DRAM NoC addresses:

```c++
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// For each stick:
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

- **stick_size**: compile-time arg 0 (`block_size_nbytes` = `padded_shape[-1] * element_size`). This is the page size for the RM tensor in DRAM.
- **TensorAccessorArgs<1>()**: Starts reading compile-time args at index 1 (index 0 is `stick_size`). The TensorAccessor args encode the interleaved buffer's bank distribution information (rank, num_banks, tensor shape, bank coordinates).
- **get_noc_addr(stick_id, s)**: Maps `stick_id` to the physical bank and offset within that bank, then computes the 64-bit NoC address. For interleaved buffers, this involves `bank_id = stick_id % num_banks`, `offset = (stick_id / num_banks) * page_size`, then translating to the physical NoC coordinates of that bank.

### Reader: L1 Write Address Progression

Within a single block of 32 sticks, the reader writes sticks contiguously:
```c++
uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
for (uint32_t k = 0; k < 32; k++) {
    noc_async_read(base_src_noc_addr[k], l1_write_addr, width_size);
    l1_write_addr += width_size;
}
```
After all 32 sticks are written, the CB contains `32 * width_size` bytes of contiguous RM data, which the tilize instruction can process.

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)

- **Pattern**: Strided across DRAM banks (interleaved), sequential within each block.
- **Per block**: 32 individual `noc_async_read` calls, one per stick. Each reads `padded_width * element_size` bytes.
- **Stick ordering**: Sequential stick IDs within a block (`stick_id`, `stick_id+1`, ..., `stick_id+31`). Since the tensor is interleaved, consecutive sticks map to different DRAM banks in round-robin fashion, naturally distributing reads across banks.
- **NoC address pre-computation**: All 32 addresses are computed before any reads are issued. This is a deliberate optimization -- the address computation loop and the read loop are separated, allowing the read loop to issue all 32 DMAs back-to-back before the barrier.

### Write Pattern (Writer Kernel -- de-emphasized)

- **Pattern**: Sequential tile writes, one tile at a time via `noc_async_write_page`.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear enumeration of cores from a potentially 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size()` (e.g., 8x8 = 64 cores) |
| **Total cores** | `ncores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `nblocks_per_core` blocks (each block = 32 sticks x full width) |
| **Load balancing** | Near-equal, with optional cliff core |
| **Remainder handling** | Last core is a "cliff core" with `nblocks_per_core_cliff` blocks |

### Work Splitting Algorithm (`split_blocks_for_tilize`)

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)`.
2. Compute `ncores = ceil(nblocks / nblocks_per_core)`.
3. Compute `nblocks_per_core_cliff = nblocks % nblocks_per_core`.
4. If `nblocks_per_core_cliff > 0`, the last core processes fewer blocks.
5. `core_range` contains all full-work cores; `core_range_cliff` contains the single cliff core (if any).

### Per-Core Assignment (Runtime Args)

Each core receives:
- `row_start_id`: The global stick index where this core begins reading.
- `num_sticks`: `nblocks_per_core * TILE_HEIGHT` (or cliff variant) -- total sticks this core reads.
- `tile_start_id`: The global tile index where this core begins writing.

The host loop increments these across cores:
```
tile_start_id += ntiles_per_block * nblocks_per_core;
row_start_id += TILE_HEIGHT * nblocks_per_core;
```

### Core Assignment Strategy (Key for rms_norm)

The cores from the available grid are linearized via `corerange_to_cores(available_grid)`. Work is assigned sequentially: core 0 gets the first `nblocks_per_core` blocks, core 1 gets the next, etc. The cliff core (if any) gets the remainder. This is a simple 1D strip decomposition along the tile-height dimension.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` | uint32_t | Size of one RM stick in bytes (`padded_shape[-1] * element_size`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank distribution info for the source buffer (appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)`) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks assigned to this core (`nblocks_per_core` or `nblocks_per_core_cliff`) |
| 1 | `per_core_block_tile_cnt` | uint32_t | Tiles per block (`ntiles_per_block` = `padded_shape[-1] / 32`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `output_cb_index` | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank distribution info for the destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer DRAM address |
| 1 | `num_sticks` | uint32_t | Total sticks for this core (`nblocks_per_core * 32`) |
| 2 | `block_size_nbytes` | uint32_t | Size of one stick in bytes (same as CT arg 0; passed redundantly for legacy reasons) |
| 3 | `ntiles_per_block` | uint32_t | Number of tiles per block (= width in tiles) |
| 4 | `block_width_size` | uint32_t | Same as `block_size_nbytes` |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for this factory (entire width processed in one block) |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 (no partial width blocks) |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 |
| 8 | `row_start_id` | uint32_t | Global stick index where this core starts reading |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer DRAM address |
| 1 | `num_tiles` | uint32_t | Total tiles for this core (`ntiles_per_block * nblocks_per_core`) |
| 2 | `tile_start_id` | uint32_t | Global tile index where this core starts writing |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 sticks per block via noc_async_read |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Outer loop: iterates over `num_sticks / 32` blocks (each block = 32 sticks = one tile height).
  - Inner address computation: For each block, computes all 32 NoC addresses into `base_src_noc_addr[32]` array.
  - Inner read via `read_tiles` lambda: reserves `ntiles_per_block` pages in CB, then issues 32 `noc_async_read` calls (one per stick) writing contiguously into the CB, calls `noc_async_read_barrier()`, then pushes.
  - The `num_full_blocks_in_row` inner loop (always 1 here) allows this kernel to be reused in scenarios where the width is split across multiple blocks, but for the interleaved factory the full width is always one block.
  - **Width advance**: After each stick read, `base_src_noc_addr[k] += width_size` advances the source address for any subsequent width blocks. Since `num_full_blocks_in_row = 1`, this advance is never used again within the same group of 32 sticks.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize.cpp | RISCV_2 (Unpack + Math + Pack) | N/A | CB c_0 | CB c_16 | tilize_block / fast_tilize_block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize hardware for the tilize operation.
  - Calls `compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, NoReconfigure>(ntiles_per_block, nblocks_per_core)`.
  - The helper library (`tilize_helpers.inl`) handles the main loop: for each block, it waits on `ntiles_per_block` input pages, reserves output space, calls the hardware tilize instruction, and pushes results.
  - **Fast tilize**: Automatically selected at compile time when: (1) tiles are 32x32, (2) half-sync dest mode, (3) Float32 or Float16_b format. Otherwise falls back to regular `tilize_block`.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM (interleaved tiles) | Write one tile at a time via noc_async_write_page |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple tile-by-tile writer. Shared across many operations (not tilize-specific).

---

## Implementation Notes

### Reader Pattern: Batch-32-Stick Reading

The defining characteristic of this reader is that it reads exactly 32 sticks per block, regardless of the tensor's logical structure. The number 32 comes from `TILE_HEIGHT` -- it takes 32 rows of data to form one row of tiles. This pattern is directly applicable to any operation that needs to:
1. Read RM data from DRAM
2. Feed it to a tilize compute stage

For **rms_norm with in-kernel tilize**: The same 32-stick batching approach applies. The reader would read 32 sticks at a time into a CB, the compute kernel would tilize and then perform the rms_norm computation on the resulting tiles.

### NoC Address Pre-computation

The reader pre-computes all 32 NoC addresses before issuing any reads. This is important because `get_noc_addr` involves bank index calculation and coordinate lookup, which takes cycles. By batching all address computations, the actual DMA issue loop is tight and can overlap reads more efficiently.

### TensorAccessor Pattern

The `TensorAccessor` is constructed from compile-time args (starting at index 1) and runtime args (the buffer address). The key call is:
```c++
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```
Where `1` is the starting CT arg index (index 0 is `stick_size`). The TensorAccessor encapsulates all bank distribution logic, making the kernel portable across different memory configurations.

### Single-Buffered Design Implications

The single-buffered CB design means reader and compute alternate strictly: reader fills, compute processes, reader fills again. For a high-throughput operation like rms_norm, consider **double-buffering** the input CB (setting `num_pages = 2 * ntiles_per_block`) to allow the reader to fill the next block while compute processes the current one.

### Cliff Core Handling

The program factory creates **two separate compute kernel instances** with different compile-time args: one for full cores (`nblocks_per_core`) and one for the cliff core (`nblocks_per_core_cliff`). This avoids runtime branching in the compute kernel. The reader and writer use the same kernel binary but receive different runtime args.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the tilize operation and how does noc_async_read work in tt-metal dataflow kernels?"
   **Reason**: Needed to understand the hardware-level tilize mechanism and async read semantics.
   **Key Findings**: Tilize converts RM to tile format using hardware instructions (tilize_block/fast_tilize_block). noc_async_read is a non-blocking DMA that requires noc_async_read_barrier for synchronization. Multiple reads can be issued before a single barrier.

2. **Query**: "How does TensorAccessorArgs work in tt-metal?"
   **Reason**: Needed to understand how buffer bank distribution is passed to kernels.
   **Key Findings**: DeepWiki returned error; information obtained from `tech_reports/tensor_accessor/tensor_accessor.md` instead.

3. **Query**: "How do circular buffers work in tt-metal?"
   **Reason**: Needed to understand cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front protocol.
   **Key Findings**: DeepWiki returned error; information obtained from METALIUM_GUIDE.md and direct code analysis.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Core architecture understanding -- Tensix core structure, reader/compute/writer kernel model, circular buffer coordination.
   **Key Information**: Three kernel types coordinate through CBs in SRAM. Reader writes data into CB and signals availability. Compute waits for data, processes, writes results to another CB. Writer waits for results and writes to DRAM.

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding RM vs tiled layout, page definitions, interleaved memory layout.
   **Key Information**: RM layout: one page = one row. Tiled layout: one page = one 32x32 tile (with face decomposition). Interleaved: pages distributed round-robin across DRAM banks.

3. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessorArgs maps page/stick IDs to NoC addresses.
   **Key Information**: Host-side `TensorAccessorArgs(buffer)` extracts bank distribution from the Buffer object and packs it as compile-time args. Kernel-side `TensorAccessor(args, base_addr, page_size)` reconstructs the distribution and provides `get_noc_addr(page_id)` to compute 64-bit NoC addresses.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute-side tilize helper library.
   **Key Information**: The `compute_kernel_lib::tilize<>()` template provides a clean interface with configurable init/uninit modes, wait modes, and register reconfiguration. It automatically selects `fast_tilize_block` when hardware supports it (32x32 tiles, half-sync, Float32/Float16_b).

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding the `split_blocks_for_tilize` work distribution algorithm.
   **Key Information**: Simple 1D block distribution: `nblocks_per_core = ceil(nblocks / grid_area)`, with a cliff core handling the remainder. Returns CoreRangeSets for full and cliff cores.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper function.
   **Key Information**: `create_cb(cb_id, program, cores, page_size, num_pages, data_format)` creates a CircularBuffer with `total_size = num_pages * page_size`, sets page size, and returns the handle.
