# Tilize Single-Core Implementation Analysis

## Overview

The tilize single-core operation converts a tensor from **row-major (RM) layout** to **tiled layout** (32x32 tiles with 16x16 faces). This is a fundamental data transformation required before any compute operation on Tenstorrent hardware, since the Tensix compute engines operate natively on 32x32 tiles.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp`

The operation reads row-major "sticks" (rows) from DRAM interleaved memory, arranges them into a tile-compatible layout in L1 via the "split-rows" reader pattern, passes them through the compute tilize hardware (UNPACK thread reorders row-major into face-based tile format), and writes the resulting tiles back to DRAM interleaved memory.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (group of tiles spanning a tile-row's width) |
| **Unit size** | `num_tiles_per_block` tiles (dynamically chosen, up to `num_tiles_in_row`) |
| **Total units** | `num_tiles / num_tiles_per_block` blocks |
| **Loop structure** | Outer loop: tile-rows (`num_sticks / 32`). Inner loop: width blocks (`num_full_blocks_in_row`). Each iteration of the inner loop processes one block of `num_tiles_per_block` tiles. |

A "block" consists of `num_tiles_per_block` tiles arranged horizontally across one tile-row (32 sticks high). The reader fills the input CB with the row-major data for one block, the compute kernel tilizes it, and the writer drains the output CB.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary N-D (flattened to 2D: `num_sticks x stick_s`) |
| **Dimension convention** | Last dim = width (stick_s); all outer dims collapsed into num_sticks |
| **Tensor layout** | ROW_MAJOR_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

- **Page definition**: One row-major page = one stick = `stick_s` elements = `stick_size` bytes.
- `stick_s = padded_shape[-1]` (the innermost dimension width).
- `num_sticks = physical_volume / stick_s`.
- Pages are distributed round-robin across DRAM banks.

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same logical shape as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (or `output_dtype` from params) |

- **Page definition**: One tiled page = one 32x32 tile = `tile_size(output_cb_data_format)` bytes.
- Tiles are stored with 16x16 faces in row-major face order (face0, face1, face2, face3).
- `num_tiles = physical_volume / TILE_HW` total tiles.

### Layout Transformations

The core transformation is **row-major to tiled**:

1. **Input**: Data is stored as contiguous rows ("sticks"). Each stick is one page in DRAM.
2. **Reader stage**: The reader gathers 32 consecutive sticks and reads `block_width_size` bytes from each, placing them contiguously in L1. This creates a 2D block of shape `[32 rows x block_width_elements]` in row-major order within the input CB.
3. **Compute stage**: The `tilize_block` hardware operation reads the row-major data from the input CB and reorders elements into tiles with 16x16 faces, writing results to the output CB.
4. **Output**: Each tile page in the output CB is a 32x32 tile stored as four 16x16 faces in face-row-major order.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved, RM sticks) | CB 0 (input) | `cb_reserve_back(cb_0, num_tiles_per_block)`, `noc_async_read` x32 sticks, `noc_async_read_barrier`, `cb_push_back(cb_0, num_tiles_per_block)` |
| 2 | Compute | CB 0 (input) | CB 16 (output) | `cb_wait_front(cb_0, num_tiles_per_block)`, `tilize_block(cb_0, block_width_tiles, cb_16)`, `cb_push_back(cb_16, block_width_tiles)`, `cb_pop_front(cb_0, num_tiles_per_block)` |
| 3 | Writer | CB 16 (output) | DRAM (interleaved, tiles) | `cb_wait_front(cb_16, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(cb_16, 1)` |

### Detailed Reader Data Flow (Split-Rows Pattern)

This is the most important pattern for understanding how RM data is prepared for tilization:

```
For each tile-row (group of 32 consecutive sticks):
  1. Resolve base NoC addresses for all 32 sticks via TensorAccessor
     - base_src_noc_addr[k] = get_noc_addr(stick_id + k, s)   for k in [0..31]

  2. For each width block j in [0..num_full_blocks_in_row):
     a. cb_reserve_back(CB_0, num_tiles_per_block)
     b. l1_write_addr = get_write_ptr(CB_0)
     c. For each of 32 sticks (k in [0..31]):
        - noc_async_read(base_src_noc_addr[k], l1_write_addr, block_width_size)
        - l1_write_addr += block_width_size
        - base_src_noc_addr[k] += block_width_size   (advance to next segment of same stick)
     d. noc_async_read_barrier()
     e. cb_push_back(CB_0, num_tiles_per_block)
```

The key insight is that `base_src_noc_addr[k]` is advanced by `block_width_size` after each block read. This means for the next width block, the reader continues reading from where it left off within the same stick. The data in L1 after step (c) is arranged as:

```
L1 layout in CB 0 after read_tiles:
  [stick_0_segment]  (block_width_size bytes)
  [stick_1_segment]  (block_width_size bytes)
  ...
  [stick_31_segment] (block_width_size bytes)
```

This is exactly the row-major representation of a `[32 x block_width_elements]` sub-matrix, which is what `tilize_block` expects as input.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (CB 0) | src0 | Input staging (RM sticks arranged for tilize) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 (CB 16) | output | Output staging (tilized tiles) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

**Capacity calculation**:
- CB 0 capacity = `num_tiles_per_block * input_single_tile_size` bytes
- CB 16 capacity = `num_tiles_per_block * output_single_tile_size` bytes
- `input_single_tile_size = tile_size(input_cb_data_format)` (e.g., 2048 bytes for bfloat16 32x32 tiles)
- `output_single_tile_size = tile_size(output_cb_data_format)`

**Block size selection logic** (lines 53-69 of program factory):
1. If `use_low_perf` is true: `num_tiles_per_block = 1` (minimum, lowest performance).
2. Otherwise, compute `max_tiles` that fit in half the available L1 (shared between 2 CBs):
   - `max_l1_size = (l1_size_per_core / 2) - base_allocator_addr`
   - `max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size)`
3. If `num_tiles_in_row <= max_tiles`: block = entire row (`num_tiles_per_block = num_tiles_in_row`).
4. Otherwise, find the largest factor of `num_tiles_in_row` that fits: iterate from `max_tiles` down to 1 and pick the first divisor.

This ensures blocks evenly divide the tile row width, which is necessary because the reader reads exact `block_width_size` segments.

## Pipeline Pattern Summary

Both CB 0 and CB 16 are **single-buffered** (capacity equals block size). This means:
- The reader and compute kernel cannot overlap: the reader must fully populate CB 0 before compute can start, and compute must fully drain CB 0 before the reader can start filling the next block.
- Similarly, compute and writer have limited overlap. However, the writer drains one tile at a time, so partial overlap is possible within a block.

This is the simplest pipeline pattern -- suitable for a single-core operation where throughput optimization is secondary to correctness and simplicity.

## Index Calculations

### Stick-to-NoC Address Mapping (Reader)

The reader uses `TensorAccessor` to map logical stick IDs to physical NoC addresses:

```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // compile-time args start at index 1
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
// ...
base_src_noc_addr[j] = get_noc_addr(stick_id, s);
```

- `stick_id` is a linear index into the flattened tensor (0-based).
- `TensorAccessor` encodes the bank distribution (round-robin across DRAM banks) and computes: bank_id = stick_id % num_banks, offset_within_bank = (stick_id / num_banks) * stick_size + base_address.
- `get_noc_addr` returns a 64-bit NoC address encoding both the bank's physical NoC coordinates and the offset.

### Tile-to-NoC Address Mapping (Writer)

The writer uses the same `TensorAccessor` pattern:

```cpp
constexpr auto dst_args = TensorAccessorArgs<1>();  // compile-time args start at index 1
const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);
// ...
noc_async_write_page(i, s, l1_read_addr);
```

- `i` is a linear tile index (0-based, sequential).
- The tile index maps to a bank and offset via the same round-robin distribution as sticks, but with tile-sized pages.

### Width Block Calculations (Host)

```
stick_s          = padded_shape[-1]                    // elements per stick
num_sticks       = physical_volume / stick_s           // total sticks
stick_size       = stick_s * element_size              // bytes per stick
num_tiles_in_row = stick_s / TILE_WIDTH                // tiles per row (= stick_s / 32)
block_width_size = num_tiles_per_block * TILE_WIDTH * element_size  // bytes per block per stick
num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block
```

## Memory Access Patterns

### Read Pattern

**Pattern type**: Strided gather across DRAM banks, then sequential within each stick segment.

For each tile-row:
1. The reader resolves 32 NoC addresses (one per stick) -- these may target different DRAM banks since sticks are interleaved round-robin.
2. For each width block, it issues 32 `noc_async_read` calls, each reading `block_width_size` contiguous bytes from a different DRAM bank/offset.
3. These reads land sequentially in L1 (each read appends after the previous).
4. After all 32 reads complete (barrier), the CB is pushed.

**Access characteristics**:
- 32 reads per block, each from a potentially different DRAM bank
- Each read is contiguous within a stick (sequential within that stick's memory)
- Cross-stick accesses are strided (different banks, different base addresses)
- NoC read barrier after each block ensures all 32 reads complete before compute starts

### Write Pattern

**Pattern type**: Sequential, one-tile-at-a-time to interleaved DRAM.

The writer processes tiles in order (tile 0, 1, 2, ..., num_tiles-1):
1. Wait for 1 tile in output CB.
2. Write that tile to DRAM via `noc_async_write_page`.
3. Flush the write (`noc_async_writes_flushed` -- ensures departure, not completion).
4. Pop the tile from CB.
5. Final `noc_async_write_barrier` after all tiles ensures completion.

**Access characteristics**:
- One tile written per iteration
- Sequential tile indices map round-robin to DRAM banks
- Uses `noc_async_writes_flushed` (lighter than full barrier) per tile, with final barrier at end

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (0D) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles (`num_tiles`) |
| **Load balancing** | N/A (single core) |

The single-core variant uses core (0,0) by default, or an optionally specified core from `sub_core_grids`. All work is assigned to this one core. This is the simplest distribution strategy, intended for small tensors or as a fallback.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one stick in bytes (`stick_s * element_size`) |
| 1+ | TensorAccessorArgs | (multiple) | Bank distribution metadata for source buffer (appended via `TensorAccessorArgs(*src0_buffer).append_to(...)`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (= 16, i.e., `tt::CBIndex::c_16`) |
| 1+ | TensorAccessorArgs | (multiple) | Bank distribution metadata for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (`num_tiles / num_tiles_per_block`) |
| 1 | per_core_block_tile_cnt | uint32_t | Number of tiles per block (`num_tiles_per_block`) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of row-major sticks |
| 2 | stick_size | uint32_t | Bytes per stick (also passed as compile-time, used here for runtime reference) |
| 3 | num_tiles_per_block | uint32_t | Tiles per width block |
| 4 | block_width_size | uint32_t | Bytes per block segment per stick (`num_tiles_per_block * TILE_WIDTH * element_size`) |
| 5 | num_full_blocks_in_row | uint32_t | Number of full blocks across one tile-row width |
| 6 | num_leftover_tiles | uint32_t | Remaining tiles after full blocks (currently unused in kernel -- see note) |
| 7 | leftover_width_in_row | uint32_t | Bytes for leftover tiles (currently unused in kernel -- see note) |
| 8 | row_start_id | uint32_t | Starting stick ID (always 0 for single-core) |

**Note on args 6-7**: The reader kernel code does NOT use `num_leftover_tiles` or `leftover_width_in_row`. The block size selection logic on the host ensures `num_tiles_in_row % num_tiles_per_block == 0`, making leftovers zero. These args appear to be reserved for potential future use or are vestigial from a more general version.

**Note on arg 2**: `stick_size` is passed as both a runtime arg (index 2) and a compile-time arg (index 0). The kernel only uses the compile-time version (`get_compile_time_arg_val(0)`), but `get_arg_val<uint32_t>(2)` is never called in the current kernel code. The runtime arg slot exists but is skipped.

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_tiles | uint32_t | Total number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index (always 0 for single-core) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (RM sticks, interleaved) | CB 0 | Read 32 sticks, gather block_width_size from each |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp`
- **Key Logic**:
  - **TensorAccessor construction**: `TensorAccessorArgs<1>()` starts reading compile-time args from index 1 (index 0 is `stick_size`). Creates accessor with `src_addr` and `stick_size`.
  - **Split-rows pattern**: Outer loop iterates over tile-rows (`num_sticks / 32`). For each tile-row, resolves 32 NoC addresses and stores in `base_src_noc_addr[32]`. Inner loop iterates over width blocks.
  - **read_tiles lambda**: Reserves CB space, reads `width_size` bytes from each of 32 sticks (using cached+advancing NoC addresses), barriers, then pushes CB. The advancing of `base_src_noc_addr[k] += width_size` is critical -- it moves the read pointer within each stick for the next width block.
  - **No leftover handling**: The kernel only processes `num_full_blocks_in_row` blocks; the host ensures no leftovers exist.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Compute | RISCV_2 (UNPACK + MATH + PACK) | N/A | CB 0 | CB 16 | tilize_block (row-major to tile format conversion) |

- **File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_16)` to initialize UNPACK/MATH/PACK threads.
  - Delegates to `compute_kernel_lib::tilize<c_0, c_16>(per_core_block_tile_cnt, per_core_block_cnt)`.
  - Default template parameters: `InitAndUninit`, `WaitBlock`, `Standard` speed mode.
  - The `tilize` helper (from `tilize_helpers.inl`) runs:
    1. `tilize_init(c_0, block_width_tiles, c_16)` -- configures unpacker for row-major input.
    2. For each of `per_core_block_cnt` blocks:
       - `cb_wait_front(c_0, block_width_tiles)` -- wait for reader to fill CB 0.
       - `cb_reserve_back(c_16, block_width_tiles)` -- ensure output CB has space.
       - `tilize_block(c_0, block_width_tiles, c_16)` -- hardware-accelerated tilization. UNPACK thread reads row-major data from CB 0 and reorders into 16x16 faces; MATH copies to DST registers; PACK writes tiled data to CB 16.
       - `cb_push_back(c_16, block_width_tiles)` -- make tiles available to writer.
       - `cb_pop_front(c_0, block_width_tiles)` -- free input CB space for reader.
    3. `tilize_uninit(c_0, c_16)` -- cleanup.
  - If `fp32_dest_acc_en` is true (input is FLOAT32), the compute engine uses FP32 accumulation in destination registers.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Writer | RISCV_1 | NOC1 | CB 16 | DRAM (tiles, interleaved) | Write one tile at a time |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**:
  - Generic writer reused across many operations.
  - Gets `page_bytes` from the CB interface (`get_local_cb_interface(cb_id_out).fifo_page_size`).
  - Iterates from `start_id` to `start_id + num_pages`, writing one tile per iteration.
  - Per tile: `cb_wait_front(cb_out, 1)` -> `noc_async_write_page(i, s, l1_read_addr)` -> `noc_async_writes_flushed()` -> `cb_pop_front(cb_out, 1)`.
  - Final `noc_async_write_barrier()` ensures all writes complete.
  - Uses `noc_async_writes_flushed` (not full barrier) per tile for performance -- this only ensures the write leaves the core, not that it arrives at DRAM. The final barrier guarantees all writes have landed.

## Implementation Notes

### Block Size Optimization
The program factory dynamically selects `num_tiles_per_block` to maximize throughput while fitting within L1 constraints. The ideal case is `num_tiles_per_block = num_tiles_in_row` (entire tile-row as one block), which minimizes the number of CB reserve/push/wait/pop cycles. The constraint is that 2 CBs (input + output) must fit in half the L1 minus the allocator base address.

### Split-Rows vs Stick-at-a-Time Reading
The split-rows pattern is more efficient than reading one stick at a time because:
1. It batches 32 `noc_async_read` calls before issuing a single barrier, amortizing barrier overhead.
2. The 32 sticks correspond exactly to one tile-height, so after all 32 reads complete, the data is immediately ready for `tilize_block`.
3. Width-block iteration with advancing `base_src_noc_addr` avoids re-resolving NoC addresses for the same sticks.

### FP32 Support
When the input dtype is FLOAT32, the program factory sets `fp32_dest_acc_en = true` in the compute config. This enables FP32 accumulation in the destination registers, which is necessary for lossless tilization of FP32 data. The `input_single_tile_size` and `output_single_tile_size` will be larger (4096 bytes for FP32 vs 2048 bytes for BFLOAT16), affecting the block size calculation.

### Program Caching
The `override_runtime_arguments` method enables program caching. When the same operation is called with different tensors (but same shapes/dtypes), only the buffer addresses (runtime args 0 for both reader and writer) need updating. The program structure, CBs, and compile-time args remain the same.

### Relevance to layer_norm_rm Input Stage
For a layer_norm_rm operation that needs to tilize its row-major input:
- The split-rows reader pattern can be reused or adapted directly.
- The same CB 0 configuration works for feeding tilized input to downstream compute.
- The block size optimization logic applies to any tilize stage.
- Key difference: in layer_norm_rm, the tilized output feeds into compute (not back to DRAM), so the writer kernel would not be needed for this stage -- the output CB would feed directly into the layer norm compute kernel instead.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in tt-metal? Specifically, how does it convert row-major data into tiled format (32x32 tiles with 16x16 faces)?"
   **Reason**: Needed to understand the fundamental transformation performed by the tilize compute kernel, particularly the face-based internal tile layout.
   **Key Findings**: Tilize is performed by TRISC_UNPACK which reads row-major data from CB and reorders into 16x16 faces. Three threads cooperate: UNPACK (reorder), MATH (datacopy to DST registers), PACK (write to output CB). Tiles have face-row-major order: face0->face1->face2->face3.

2. **Query**: "How does TensorAccessor and TensorAccessorArgs work in tt-metal? How are they used to map logical tensor indices to physical memory addresses across interleaved DRAM banks?"
   **Reason**: The reader and writer both use TensorAccessor for address resolution. Needed to understand how stick_id/tile_id maps to physical NoC addresses.
   **Key Findings**: TensorAccessorArgs configures compile-time vs runtime args on the host. On device, TensorAccessor maps logical page IDs to 64-bit NoC addresses encoding bank coordinates and offsets. get_noc_addr(page_id, accessor) handles the round-robin bank distribution transparently.

3. **Query**: "How does the reader_unary_stick_layout_split_rows_interleaved kernel work for tilize?"
   **Reason**: The split-rows reader is the most complex kernel in this operation and the primary focus of the analysis.
   **Key Findings**: Reads 32 consecutive sticks, caches base NoC addresses, then iterates across width reading block_width_size bytes per stick. This constructs tile-height blocks in L1 suitable for tilize_block consumption.

4. **Query**: "What does tilize_block do internally?"
   **Reason**: Needed to understand the hardware-level transformation inside the compute kernel.
   **Key Findings**: UNPACK thread reads row-major from input CB and reorders into faces. MATH thread copies to DST registers via datacopy. PACK thread writes tiled data to output CB. Input CB must contain row-major data arranged as [32 rows x N columns].

5. **Query**: "What is noc_async_write_page vs noc_async_write_tile? What is noc_async_writes_flushed vs noc_async_write_barrier?"
   **Reason**: Writer kernel uses noc_async_write_page and noc_async_writes_flushed; needed to understand the synchronization semantics.
   **Key Findings**: noc_async_write_page is the generic replacement for deprecated noc_async_write_tile. noc_async_writes_flushed only ensures writes depart the core (lighter). noc_async_write_barrier waits for writes to complete at destination (heavier, used at end).

6. **Query**: "What is compute_kernel_hw_startup in tt-metal?"
   **Reason**: The compute kernel calls this before tilize. Needed to understand what hardware initialization is required.
   **Key Findings**: Initializes UNPACK/MATH/PACK threads with CB IDs and data formats. Configures DST register synchronization. Must be called exactly once before any tilize_init/tilize_block calls.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Needed to understand RM vs tiled layout, page definitions, interleaved bank distribution.
   **Key Information**: RM layout: each row = one page. Tiled layout: each 32x32 tile = one page with 16x16 faces. Interleaved: pages distributed round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: TensorAccessor is used in both reader and writer kernels for address resolution.
   **Key Information**: Host-side TensorAccessorArgs configures compile-time vs runtime args. Device-side TensorAccessor provides get_noc_addr(page_id) for physical address mapping. noc_async_read_page/noc_async_write_page integrate directly.

3. **Source**: `METALIUM_GUIDE.md` (CB pattern sections)
   **Reason**: Needed to verify standard CB API usage patterns (reserve_back, push_back, wait_front, pop_front).
   **Key Information**: Confirmed standard producer pattern (reserve_back -> write -> push_back) and consumer pattern (wait_front -> read -> pop_front).

4. **Source**: `tt_metal/api/tt-metalium/constants.hpp`
   **Reason**: Needed to verify tile dimension constants.
   **Key Information**: TILE_HEIGHT=32, TILE_WIDTH=32, TILE_HW=1024, FACE_HEIGHT=16, FACE_WIDTH=16.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: The compute kernel delegates to this library. Needed to understand the full tilize flow including init/uninit and CB synchronization.
   **Key Information**: Default mode: InitAndUninit + WaitBlock + Standard speed. Flow: tilize_init -> loop(wait_front, reserve_back, tilize_block, push_back, pop_front) -> tilize_uninit. Supports fast mode, DT reconfiguration, and non-tile-aligned CB configs.
