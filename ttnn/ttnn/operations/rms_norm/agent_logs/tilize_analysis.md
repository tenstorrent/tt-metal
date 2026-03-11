# Tilize (Multi-Core Interleaved) Implementation Analysis

## Overview

The **tilize** operation converts a tensor from **row-major layout** to **tiled layout** (32x32 tiles). It reads contiguous row-major sticks from DRAM, groups them into blocks of 32 sticks (one tile height), and uses the compute unit's `tilize_block` / `fast_tilize_block` hardware instruction to rearrange data into tile format. The output tensor has the same logical shape but is stored in TILE_LAYOUT with pages organized as 32x32 tiles (with internal 16x16 face decomposition).

**Program factory path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_interleaved_program_factory.cpp`

**Focus of this analysis**: Input stage -- reader kernel pattern, input CB sizing, stick-to-tile batching, and work distribution strategy. This is intended as a reference for building RMS norm's reader that must similarly read row-major sticks from DRAM.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block (group of tile-rows) |
| **Unit size** | 1 block = `ntiles_per_block` tiles across the width, spanning `TILE_HEIGHT` (32) sticks vertically |
| **Total units** | `nblocks = ceil(ntiles / ntiles_per_block)` where `ntiles = physical_volume / TILE_HW` |
| **Loop structure** | Outer loop: blocks assigned to core. Inner: read 32 sticks, tilize, write tiles. |

A **block** represents one horizontal row of tiles across the full tensor width. Specifically, one block corresponds to 32 consecutive row-major sticks (one tile-height) across the entire width dimension. The number of tiles per block equals `padded_shape[-1] / TILE_WIDTH`. The total number of blocks equals the total tile count divided by tiles-per-block.

**Relevance to RMS norm**: For an operation that processes along the last dimension, the natural work unit is similarly a group of rows -- specifically, the set of rows that constitute one tile-height (32 sticks). RMS norm would read these sticks, compute the root mean square across the last dimension, then normalize.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D, padded to tile alignment on last 2 dims |
| **Dimension convention** | Last dim = width (contiguous in memory), second-to-last = height |
| **Tensor layout** | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (interleaved across banks) |
| **Data type** | BFLOAT16 or FLOAT32 |
| **Page definition** | One page = one row-major stick = `padded_shape[-1]` elements = `padded_shape[-1] * element_size` bytes |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |
| **Page definition** | One page = one tile = `tile_size(data_format)` bytes |

### Layout Transformations

The tilize operation itself IS the layout transformation: ROW_MAJOR -> TILE_LAYOUT. The compute kernel's `tilize_block` instruction reads 32 sticks from the input CB (row-major data) and writes tiles to the output CB (tiled format with 16x16 face decomposition).

---

## Data Flow Pattern

### Step-by-step flow (input-stage focus)

1. **Reader kernel** iterates over its assigned blocks. For each block:
   a. **Resolve 32 stick addresses**: For sticks `[stick_id, stick_id+1, ..., stick_id+31]`, call `get_noc_addr(stick_id, tensor_accessor)` to translate each logical stick index into a physical DRAM bank NoC address. These addresses are cached in a local `base_src_noc_addr[32]` array.
   b. **Reserve CB space**: Call `cb_reserve_back(cb_id_in0, ntiles_per_block)` to wait for `ntiles_per_block` pages of free space in the input CB.
   c. **DMA all 32 sticks into CB**: For each of the 32 sticks, issue `noc_async_read(src_noc_addr, l1_write_addr, width_size)` to transfer the entire stick width into the CB. The L1 write pointer advances by `width_size` (= `block_size_nbytes` = full row width in bytes) after each stick.
   d. **Barrier**: `noc_async_read_barrier()` ensures all 32 stick reads are complete.
   e. **Push to CB**: `cb_push_back(cb_id_in0, ntiles_per_block)` signals the compute kernel that `ntiles_per_block` tiles worth of row-major data are ready.

2. **Compute kernel** waits for block data, calls `tilize_block` to convert row-major sticks into tiles, and pushes tiles to output CB.

3. **Writer kernel** waits for tiles in output CB, writes them to DRAM one tile at a time.

### Key insight for RMS norm reader

The reader packs **32 sticks at once** into the CB before signaling. The CB page size is `input_single_tile_size` (tile-sized), but the actual data written is row-major: 32 sticks laid out contiguously. The `ntiles_per_block` pages reserved correspond to the number of tiles that can be formed from those 32 sticks across the width. This is because the compute's tilize hardware reads the CB as row-major and reorders into tiles.

For RMS norm, the reader would similarly want to read sticks (one full row per stick), but the CB page sizing would differ because RMS norm's compute does element-wise operations on tiles, not tilize operations.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (pages) | Page Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-----------|-----------|----------|----------|----------|
| `c_0` | cb_in0 | Input staging (row-major sticks packed as tile-equivalent pages) | `ntiles_per_block` | `input_single_tile_size` | Single | Reader | Compute | Block |
| `c_16` | cb_out0 | Output staging (tiled data) | `ntiles_per_block` | `output_single_tile_size` | Single | Compute | Writer | Block |

### Input CB sizing rationale (critical for RMS norm reference)

- **Page size**: `input_single_tile_size = tile_size(input_cb_data_format)`. For BFLOAT16, this is `32 * 32 * 2 = 2048 bytes`. For FLOAT32, this is `32 * 32 * 4 = 4096 bytes`.
- **Number of pages**: `ntiles_per_block = padded_shape[-1] / TILE_WIDTH`. This equals the number of tiles across one row of tiles (the full width).
- **Total CB capacity in bytes**: `ntiles_per_block * input_single_tile_size`. This is exactly enough to hold one full block (32 sticks x full width) of row-major data, since `32 sticks * width_bytes = 32 * padded_shape[-1] * elem_size = ntiles_per_block * TILE_WIDTH * 32 * elem_size = ntiles_per_block * tile_size`.

The CB is **single-buffered** (capacity = 1 block). The reader fills one full block, the compute consumes it entirely, then the reader fills the next block. There is no overlap between blocks.

### Why tile-sized pages for row-major data?

Although the input data is row-major sticks, the CB is configured with tile-sized pages. This works because:
- The total bytes in `ntiles_per_block` tile-sized pages equals the total bytes in 32 full-width sticks
- The compute kernel's `tilize_block` knows to interpret the CB data as row-major and rearranges it into tiles
- The page count `ntiles_per_block` tells the compute how many output tiles to produce

---

## Index Calculations

### Stick-to-address mapping

The reader uses `TensorAccessor` to map logical stick IDs to physical DRAM addresses:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args starting at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

// For each stick in the block:
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

`TensorAccessorArgs` is constructed from the source buffer on the host side and appended to compile-time args. On the device, `TensorAccessor` uses these args plus the base address and page size to compute the NoC address for any page (stick) index. For interleaved buffers, this involves:
1. Computing which DRAM bank holds the page: `bank_id = stick_id % num_banks`
2. Computing the offset within that bank: `offset = (stick_id / num_banks) * stick_size`
3. Translating the bank ID to a NoC coordinate and combining with the offset

### Stick ID progression

Each core starts at `start_stick_id = row_start_id` (set per core via runtime args). The stick ID advances sequentially: for each block, 32 consecutive sticks are processed. After all blocks on one core, the next core picks up where it left off.

---

## Memory Access Patterns

### Read Pattern (input-stage focus)

- **Pattern**: Strided burst -- 32 sequential sticks are read, each from a potentially different DRAM bank (due to interleaved layout where each stick is a separate page in round-robin bank assignment).
- **Access order**: Sticks are read in sequential order (stick_id, stick_id+1, ..., stick_id+31). Within each stick, the full width is read as a contiguous DMA transfer.
- **NoC usage**: NOC0 (reader kernel). All 32 `noc_async_read` calls are issued before a single barrier, enabling the NoC to pipeline the transfers.
- **Burst size**: Each read transfers `block_size_nbytes = padded_shape[-1] * element_size` bytes (one full row width). For a tensor with width 1024 in BFLOAT16, this is 2048 bytes per stick.

### Write Pattern (de-emphasized)

- The writer reads tiles sequentially from the output CB and writes them one-at-a-time to DRAM using `noc_async_write_page`.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linear across available cores) |
| **Grid dimensions** | Up to full compute grid (e.g., 8x8 = 64 cores on Wormhole) |
| **Total cores** | `ncores` from `split_blocks_for_tilize()` |
| **Work per core** | `nblocks_per_core` blocks (cliff core gets `nblocks_per_core_cliff`) |
| **Load balancing** | Near-equal with cliff remainder |

### Work splitting algorithm (`split_blocks_for_tilize`)

1. Compute `nblocks_per_core = ceil(nblocks / grid_area)` -- try to use all available cores.
2. Compute `ncores = ceil(nblocks / nblocks_per_core)` -- actual number of cores needed.
3. `nblocks_per_core_cliff = nblocks % nblocks_per_core` -- remainder blocks for the last core.
4. If `nblocks_per_core_cliff > 0`, the last core is a "cliff" core handling fewer blocks.
5. All other cores (non-cliff) handle exactly `nblocks_per_core` blocks each.

### Runtime args per core

Each core receives:
- Its starting stick ID: `row_start_id = sum of (TILE_HEIGHT * nblocks_per_core) for all preceding cores`
- Its starting tile ID (for the writer): `tile_start_id = sum of (ntiles_per_block * nblocks_per_core) for all preceding cores`
- The number of sticks to process: `nblocks_per_core * TILE_HEIGHT` (or `nblocks_per_core_cliff * TILE_HEIGHT` for cliff)

**Relevance to RMS norm**: This 1D block distribution pattern maps directly to distributing rows across cores. For RMS norm operating on rows, the same approach applies: divide total row-blocks evenly, with an optional cliff core for the remainder.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stick_size` | uint32_t | Size of one row-major stick in bytes (`padded_shape[-1] * element_size`) |
| 1+ | TensorAccessorArgs | varies | Appended by `TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args)` -- encodes DRAM bank layout info for address resolution |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks this core processes (`nblocks_per_core` or `nblocks_per_core_cliff`) |
| 1 | `per_core_block_tile_cnt` | uint32_t | Number of tiles per block (`ntiles_per_block`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out` | uint32_t | Output CB index (c_16) |
| 1+ | TensorAccessorArgs | varies | Appended for output buffer address resolution |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `src_addr` | uint32_t | Source buffer base address in DRAM |
| 1 | `num_sticks` | uint32_t | Total sticks for this core (`nblocks * TILE_HEIGHT`) |
| 2 | `block_size_nbytes` | uint32_t | Full row width in bytes |
| 3 | `num_tiles_per_block` | uint32_t | Tiles per block (`ntiles_per_block`) |
| 4 | `block_width_size` | uint32_t | Same as `block_size_nbytes` (used in read_tiles lambda) |
| 5 | `num_full_blocks_in_row` | uint32_t | Always 1 for this factory (one full-width block per tile-row) |
| 6 | `num_leftover_tiles` | uint32_t | Always 0 (no partial blocks) |
| 7 | `leftover_width_in_row` | uint32_t | Always 0 |
| 8 | `start_stick_id` | uint32_t | First stick index for this core |

#### Writer Kernel (de-emphasized)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Destination buffer base address |
| 1 | `num_tiles` | uint32_t | Total tiles for this core |
| 2 | `start_id` | uint32_t | First tile index for this core |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_stick_layout_split_rows_interleaved | RISCV_0 | NOC0 | DRAM (interleaved RM sticks) | CB c_0 | Read 32 sticks per block via NoC DMA |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - Outer loop iterates `num_sticks / tile_height` times (= number of blocks).
  - For each block: resolves NoC addresses for 32 consecutive sticks into `base_src_noc_addr[32]` array.
  - Inner loop calls `read_tiles(num_tiles_per_block, block_width_size)` which:
    1. `cb_reserve_back(cb_id_in0, num_tiles_per_block)` -- waits for CB space.
    2. Issues 32 `noc_async_read` calls (one per stick, full width each).
    3. `noc_async_read_barrier()` -- waits for all DMA transfers.
    4. `cb_push_back(cb_id_in0, num_tiles_per_block)` -- signals data ready.
  - The `num_full_blocks_in_row` loop (runtime arg index 5) controls how many horizontal sub-blocks exist. In this interleaved factory, it is always 1, meaning the entire width is read in one shot.
  - **Address caching optimization**: The `base_src_noc_addr[k]` values are incremented by `width_size` after each sub-block read, supporting the general case where a wide row is split into multiple sub-blocks. For the interleaved factory (always 1 sub-block), this increment is never used across iterations.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| tilize.cpp | RISCV_2 (unpack/math/pack) | N/A | CB c_0 | CB c_16 | tilize_block or fast_tilize_block |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**: Calls `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)` which loops over blocks, waiting for input, performing `tilize_block`/`fast_tilize_block`, and pushing output.

### Writer Kernel (de-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_16 | DRAM | Write tiles one-at-a-time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

---

## Pipeline Pattern Summary

Both CBs (c_0 and c_16) are single-buffered: capacity equals one block. This means:
- The reader fills one block, then the compute processes it. No overlap between consecutive blocks for the same kernel pair.
- The compute produces one block of tiles, then the writer drains it.
- However, the reader CAN start filling the next block while the writer is draining the previous block's output, since c_0 and c_16 are independent CBs.

---

## Implementation Notes

### Key design patterns relevant to RMS norm

1. **Stick-based reading with TensorAccessor**: The reader demonstrates the standard pattern for reading interleaved row-major data: construct `TensorAccessor` from compile-time args, call `get_noc_addr(page_id, accessor)` to get physical addresses, issue `noc_async_read` for each page.

2. **32-stick batching**: The reader always reads 32 sticks at once (one tile height). This is fundamental because the compute kernel expects tile-height groups. For RMS norm, if the input is already in TILE_LAYOUT, this pattern would not apply directly; instead, tiles would be read directly.

3. **CB sizing = one block = full width**: The input CB holds exactly one block (all tiles across the width for 32 rows). This means the entire width is available for the compute kernel at once. For RMS norm which reduces along the last dimension, this is exactly the right access pattern: the compute needs all width elements of a row to compute the mean square.

4. **Single-buffered simplicity**: No double-buffering is used. For a new operation like RMS norm, starting with single-buffered CBs simplifies correctness before optimizing for throughput.

5. **1D work distribution with cliff**: The `split_blocks_for_tilize` function provides a clean pattern for dividing row-blocks across cores with an optional cliff core. This same utility could be reused or adapted for RMS norm.

6. **TensorAccessorArgs pattern**: On the host, `TensorAccessorArgs(*buffer).append_to(compile_time_args)` serializes buffer metadata. On the device, `TensorAccessorArgs<start_index>()` deserializes it. This is the standard way to pass buffer addressing information to kernels.

### Edge cases

- **FP32 accumulation**: When input dtype is FLOAT32, `fp32_dest_acc_en = true` is set on the compute kernel. This ensures the FPU destination register uses FP32 precision.
- **Sub-core grids**: The operation supports an optional `sub_core_grids` parameter that restricts which cores are used, useful for multi-operation pipelines.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do circular buffers work in tt-metal kernels? What do cb_reserve_back, cb_push_back, get_write_ptr do?"
   **Reason**: Needed to understand the producer-consumer synchronization between reader and compute kernels.
   **Key Findings**: `cb_reserve_back` blocks until free pages are available (backpressure mechanism). `cb_push_back` increments `pages_received` counter. `cb_wait_front` on the consumer side blocks until data is available. The free space formula is `free_pages = capacity - (pages_received - pages_acked)`. 16-bit counters handle wraparound correctly.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding the relationship between row-major sticks, tile pages, and interleaved bank distribution.
   **Key Information**: In row-major layout, each row = one page. In tiled layout, each 32x32 tile = one page. Interleaved layout distributes pages round-robin across DRAM banks. Tiles have internal 16x16 face decomposition.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor translates stick IDs to DRAM bank addresses.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` constructed on device. `get_noc_addr(page_id)` returns 64-bit NoC address. TensorAccessorArgs can be split between compile-time and runtime args. Default (ArgConfig::None) puts everything in compile-time args.

3. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the three-kernel architecture and NoC data movement patterns.
   **Key Information**: Reader on RISCV_0 uses NOC0, writer on RISCV_1 uses NOC1. Compute kernel is compiled for 3 TRISC cores (unpack/math/pack). Circular buffers are the synchronization mechanism between kernels.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding what the compute kernel's `tilize` helper does internally.
   **Key Information**: The helper loops over blocks, for each: `cb_wait_front` on input, `cb_reserve_back` on output, calls `tilize_block` or `fast_tilize_block` (auto-selected at compile time based on tile size and data format), then `cb_push_back` output and `cb_pop_front` input. Fast tilize requires 32x32 tiles and Float32/Float16_b format.

5. **Source**: `ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp`
   **Reason**: Understanding how blocks are distributed across cores.
   **Key Information**: `split_blocks_for_tilize(grid, nblocks)` computes `nblocks_per_core = ceil(nblocks / grid_area)`, `ncores = ceil(nblocks / nblocks_per_core)`, `nblocks_per_core_cliff = nblocks % nblocks_per_core`. Returns `BlockSplit` struct with core ranges for full and cliff cores.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper.
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` creates a circular buffer with total size = `num_pages * page_size`, configured with the given data format and page size.
