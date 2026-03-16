# Tilize (Single Core) Implementation Analysis

## Overview
The tilize operation converts a tensor from **row-major (RM) layout** to **tiled (32x32) layout**. The input tensor is stored as contiguous rows ("sticks") in DRAM, and the output is a set of 32x32 tiles in DRAM. This single-core variant runs all three kernels (reader, compute, writer) on a single Tensix core.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_single_core_program_factory.cpp`

**Focus**: This analysis emphasizes the **input stage** -- how RM sticks are read from DRAM into L1 circular buffers, the input CB sizing strategy, and stick-to-tile batching. Writer details are de-emphasized.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | block (a group of tiles forming part of a tile-row) |
| **Unit size** | `num_tiles_per_block` tiles (up to the full tile-width of the tensor) |
| **Total units** | `num_tiles / num_tiles_per_block` = total tile-rows x blocks-per-row |
| **Loop structure** | Reader: outer loop over tile-rows (num_sticks / 32), inner loop over blocks per row. Compute: single loop over total blocks. |

A "block" is `num_tiles_per_block` contiguous tiles along the width dimension. The reader fills one block of tile-space (32 sticks x block_width_size bytes each) into the input CB, then the compute kernel tilizes that block into output tiles.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary N-D (treated as 2D: num_sticks x width) |
| **Dimension convention** | Last dimension = stick width; all others collapsed into num_sticks |
| **Tensor layout** | ROW_MAJOR (sticks) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, via TensorAccessor) |
| **Data type** | BFLOAT16 (also supports FLOAT32 with fp32_llk_acc) |

**Key dimensions derived from input**:
- `width = a.padded_shape()[-1]` -- the innermost dimension in elements
- `stick_size = width * element_size` -- one row in bytes
- `num_sticks = physical_volume / width` -- total number of rows
- `num_tiles_in_row = width / TILE_WIDTH` -- tiles along width (TILE_WIDTH = 32)

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same logical shape as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (or specified output_dtype) |

### Layout Transformations
The entire operation IS the layout transformation: ROW_MAJOR -> TILE_LAYOUT. No separate tilize/untilize helpers are needed outside the compute kernel; the compute kernel itself performs the tilization via `compute_kernel_lib::tilize`.

---

## Data Flow Pattern

### Step-by-step Flow

1. **Reader kernel** reads 32 contiguous sticks (one tile-height) from DRAM into L1 via NoC async reads. For each group of 32 sticks, it iterates over blocks along the width:
   - `cb_reserve_back(cb_id_in0, num_tiles_per_block)` -- reserves space for one block of tiles in the input CB
   - For each of the 32 sticks in the group, reads `block_width_size` bytes from DRAM using `noc_async_read`
   - `noc_async_read_barrier()` -- waits for all 32 reads to complete
   - `cb_push_back(cb_id_in0, num_tiles_per_block)` -- signals data is ready

2. **Compute kernel** consumes blocks from CB c_0, tilizes them, and writes tiled output to CB c_16:
   - `cb_wait_front(input_cb, num_tiles_per_block)` -- waits for reader's push
   - Calls `tilize_block` (or `fast_tilize_block`) to convert row-major sticks into tiled format
   - `cb_push_back(output_cb, num_tiles_per_block)` -- signals tiled output ready
   - `cb_pop_front(input_cb, num_tiles_per_block)` -- frees input CB space

3. **Writer kernel** (de-emphasized) writes tiles from CB c_16 to DRAM one tile at a time.

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (RM sticks) | CB c_0 | reserve_back, push_back |
| 2 | Compute | CB c_0 | CB c_16 | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | CB c_16 | DRAM (tiles) | wait_front, pop_front |

---

## Circular Buffer Configuration

### Input CB (Focus of This Analysis)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_id_in0 | Input staging (RM sticks arranged in tile-shaped blocks) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Reader | Compute | Block |
| c_16 | cb_output | Output staging (tiled data) | `num_tiles_per_block` tiles | `num_tiles_per_block` tiles | Single | Compute | Writer | Block |

**Key insight**: Both CBs are **single-buffered** -- capacity equals block size. This means the reader and compute cannot overlap: the reader must finish filling the entire block before compute begins, and compute must finish consuming it before the reader can fill the next block.

### Input CB Sizing Strategy (Critical for rms_norm Reference)

The input CB page size is `input_single_tile_size` (the byte size of one tile in the input data format, e.g., 2048 bytes for bfloat16 32x32). The CB capacity is `num_tiles_per_block * input_single_tile_size` bytes.

**`num_tiles_per_block` determination** (program_factory.cpp lines 53-69):

1. **Low-perf mode** (`use_low_perf = true`): `num_tiles_per_block = 1`. Minimizes L1 usage.
2. **Normal mode**: The factory tries to maximize `num_tiles_per_block` up to the full width (`num_tiles_in_row`):
   - Computes `max_l1_size` = half of L1 minus allocator base address
   - Computes `max_tiles` = `max_l1_size / (input_tile_size + output_tile_size)` (accounting for both CBs)
   - If `num_tiles_in_row <= max_tiles`: uses the full width as the block (`num_tiles_per_block = num_tiles_in_row`), meaning one block = one complete tile-row
   - Otherwise: finds the largest divisor of `num_tiles_in_row` that fits in `max_tiles`, scanning downward from `max_tiles`

**Implication for rms_norm**: When reading RM input for in-kernel tilize, the CB capacity should be sized to hold enough tiles-worth of sticks (at least one tile-row width if L1 permits). The divisibility constraint (`num_tiles_in_row % num_tiles_per_block == 0`) ensures clean block boundaries.

---

## Pipeline Pattern Summary

Both CBs use single-buffering (capacity = block size). There is **no overlap** between reader and compute within a single block. The pipeline is strictly sequential: read block -> tilize block -> write block. This is acceptable for a single-core implementation where all kernels share the same core's resources.

---

## Reader Kernel Deep Dive (Primary Focus)

**File**: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_singlecore.cpp`

### How RM Sticks Are Read from DRAM

The reader must arrange row-major sticks into a memory layout that the compute kernel's tilize hardware can consume. The key insight is that **32 consecutive sticks** (one tile-height) must be interleaved in L1 to form tile-shaped blocks.

#### Address Resolution
```cpp
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);
```
- `TensorAccessorArgs<1>()`: template parameter `1` is the **compile-time argument offset** (index 0 is `stick_size`, so TensorAccessor args start at index 1)
- `TensorAccessor` maps a linear stick ID to a physical NoC address, handling bank interleaving automatically
- `get_noc_addr(stick_id, s)` returns a 64-bit NoC address for stick `stick_id`

#### The "Split Rows" Reading Pattern

The outer loop processes groups of 32 sticks (one tile-height):

```cpp
for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
    // Phase 1: Resolve NoC addresses for all 32 sticks
    for (uint32_t j = 0; j < tile_height; j++) {
        base_src_noc_addr[j] = get_noc_addr(stick_id, s);
        stick_id++;
    }
    // Phase 2: Read blocks across the width
    for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
        read_tiles(num_tiles_per_block, block_width_size);
    }
}
```

**Phase 1** pre-computes the base NoC addresses for all 32 sticks in the current tile-row. This is done once per tile-row, avoiding redundant address calculations.

**Phase 2** iterates over width blocks. Each `read_tiles` call:
1. Reserves `num_tiles_per_block` pages in CB c_0
2. Gets the current L1 write pointer
3. For each of the 32 sticks: reads `block_width_size` bytes from DRAM at `base_src_noc_addr[k]` into L1, advancing both the L1 pointer and the source NoC address by `block_width_size`
4. Calls `noc_async_read_barrier()` to wait for all 32 reads
5. Pushes the block to CB c_0

#### Memory Layout in L1 After Read

After one `read_tiles` call, the L1 buffer contains:
```
[stick_0, block_width_size bytes]
[stick_1, block_width_size bytes]
...
[stick_31, block_width_size bytes]
```

This is 32 rows x `block_width_size` bytes per row = 32 rows x (`num_tiles_per_block` * 32 elements * element_size) -- exactly the data needed for `num_tiles_per_block` tiles arranged in a row. The hardware tilize unit (unpack) reads this row-major arrangement and produces tiled output.

#### Stick-to-Tile Batching

- **Sticks per CB push**: Always 32 (one tile-height worth of sticks)
- **Tiles per CB push**: `num_tiles_per_block` (width of one block in tiles)
- **Bytes per stick read**: `block_width_size = num_tiles_per_block * TILE_WIDTH * element_size`
- **Total bytes per CB push**: `32 * block_width_size = num_tiles_per_block * TILE_HW * element_size = num_tiles_per_block * tile_size`

This confirms that each CB push fills exactly `num_tiles_per_block` tile-sized pages worth of L1 space, matching the CB capacity.

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Full stick width in bytes (width * element_size) |
| 1+ | TensorAccessor args | varies | Bank mapping info for source buffer (appended by TensorAccessorArgs) |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_sticks | uint32_t | Total number of sticks (rows) in the tensor |
| 2 | stick_size | uint32_t | Stick width in bytes (redundant with CT arg, used differently) |
| 3 | num_tiles_per_block | uint32_t | Tiles per block along width |
| 4 | block_width_size | uint32_t | Bytes per stick per block (num_tiles_per_block * 32 * elem_size) |
| 5 | num_full_blocks_in_row | uint32_t | Number of full blocks per tile-row |
| 6 | num_leftover_tiles | uint32_t | Leftover tiles if width not divisible (unused in kernel -- always 0 due to divisibility constraint) |
| 7 | leftover_width_in_row | uint32_t | Leftover width in bytes (unused) |
| 8 | start_stick_id | uint32_t | Starting stick index (always 0 for single-core) |

**Note**: Runtime args at indices 2, 6, 7 are declared in the host but the kernel only uses indices 0, 1, 3, 4, 5, 8. The kernel reads `stick_size` from compile-time arg 0 instead.

---

## Compute Kernel Summary

**File**: `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`

The compute kernel uses the `compute_kernel_lib::tilize` helper:

```cpp
compute_kernel_lib::tilize<
    per_core_block_tile_cnt,     // block_width_tiles = num_tiles_per_block
    tt::CBIndex::c_0,            // input CB (row-major data)
    tt::CBIndex::c_16,           // output CB (tiled data)
    InitUninitMode::InitAndUninit,
    WaitMode::WaitBlock,         // wait for each block individually
    ReconfigureRegisterDatatypeMode::NoReconfigure
>(per_core_block_cnt);           // num_blocks = total_tiles / tiles_per_block
```

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total number of blocks to process (num_tiles / num_tiles_per_block) |
| 1 | per_core_block_tile_cnt | uint32_t | Tiles per block (num_tiles_per_block) |

### Tilize Helper Internals (from tilize_helpers.inl)

The `tilize` helper's main loop:
1. **WaitBlock mode**: calls `cb_wait_front(input_cb, block_width_tiles)` per block
2. Reserves output space: `cb_reserve_back(output_cb, block_width_tiles)`
3. Calls `tilize_block` or `fast_tilize_block` depending on compile-time conditions:
   - **fast_tilize**: Used for bfloat16 with 32x32 tiles and half-sync dest mode
   - **standard tilize**: Used for float32 or other formats
4. Pushes output: `cb_push_back(output_cb, block_width_tiles)`
5. Pops input: `cb_pop_front(input_cb, block_width_tiles)`

**FP32 support**: When `a.dtype() == DataType::FLOAT32`, the program factory sets `fp32_dest_acc_en = true` and configures `UnpackToDestMode::UnpackToDestFp32` for CB c_0. This enables lossless float32 tilization.

---

## Memory Access Patterns

### Read Pattern (Reader Kernel)
- **Pattern**: Strided reads of 32 sticks per tile-row group
- **Stride**: Each stick is a separate DRAM page (interleaved across banks). Within a tile-row, sticks are logically consecutive but may reside in different banks.
- **Width traversal**: Sequential across the row in `block_width_size`-byte chunks
- **NoC transactions per block**: 32 `noc_async_read` calls (one per stick, reading `block_width_size` bytes each)
- **Barrier**: One `noc_async_read_barrier` per block (batches all 32 reads)

### Write Pattern (Writer Kernel -- De-emphasized)
- Sequential tile-by-tile writes to interleaved DRAM, one tile per `noc_async_write_page` call.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Single core (1x1) |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 |
| **Work per core** | All tiles (entire tensor) |
| **Load balancing** | N/A (single core) |

The core is either `{0,0}` (default) or specified via `sub_core_grids` parameter. All work -- reading, tilizing, writing -- happens on this single core.

---

## Kernel Implementations Summary

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM (RM sticks) | CB c_0 | Read 32 sticks per tile-row, split into width-blocks |
| Compute | RISCV_2 (unpack+math+pack) | N/A | CB c_0 | CB c_16 | tilize_block / fast_tilize_block |
| Writer | RISCV_1 | NOC1 | CB c_16 | DRAM (tiles) | Write tiles one at a time |

---

## Implementation Notes

### Design Choices Relevant to rms_norm

1. **Stick-as-page model**: The reader treats each row-major row as a "page" for TensorAccessor addressing. `stick_size` is passed as the page size to `TensorAccessor(src_tensor_args, src_addr, stick_size)`. For rms_norm with RM input, the same pattern applies: each input row is one stick/page.

2. **Pre-computed base addresses**: The reader pre-computes all 32 NoC addresses before any reads. This amortizes the address computation cost. For rms_norm, a similar pattern would help if reading 32 sticks at once for tilize.

3. **Block width sizing**: The factory dynamically sizes blocks to maximize L1 usage while respecting divisibility. For rms_norm, the block width should be chosen based on the final dimension (which is the reduction dimension), considering that the entire row's worth of tiles may be needed for the reduction.

4. **CB page size = tile size**: Even though the reader writes sticks (row-major data), the CB page size is set to `input_single_tile_size` (tile size in bytes). The tilize hardware expects this: it reads 32 stick-fragments from the CB that collectively form tile(s) of data. The CB "pages" are tile-sized units, but the actual data written is 32 rows of `block_width_size` bytes = `num_tiles_per_block * tile_size` bytes total.

5. **No leftover handling in the kernel**: Although the program factory computes `num_leftover_tiles` and `leftover_width_in_row`, the reader kernel does not use them (it only processes `num_full_blocks_in_row` blocks). The divisibility constraint on `num_tiles_per_block` ensures there are no leftovers.

6. **TensorAccessor usage pattern**: Host appends accessor args after index 0 of compile-time args. Device side uses `TensorAccessorArgs<1>()` where `1` is the CTA offset (skipping the `stick_size` arg at index 0). This pattern is reusable: place custom compile-time args first, then append TensorAccessor args, and use the appropriate offset in the kernel.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work in TT-Metal? Specifically, how does the reader kernel read row-major sticks from DRAM and arrange them for the compute kernel to convert to tiled format?"
   **Reason**: Needed to understand the fundamental concept of sticks vs tiles and how the reader/compute split works.
   **Key Findings**: Sticks are individual rows in RM layout; tiles are 32x32 blocks. The reader places sticks into L1 CBs, the compute kernel's unpack hardware rearranges them into tiled format. `tilize_block` handles the actual format conversion.

2. **Query**: "How does TensorAccessorArgs work with compile-time arguments? When you call TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args), what arguments get appended?"
   **Reason**: Needed to understand the host-to-device argument passing mechanism for tensor addressing.
   **Key Findings**: `TensorAccessorArgs.append_to()` appends bank mapping metadata (rank, num_banks, shapes, coordinates). The device-side `TensorAccessorArgs<N>()` template parameter N is the CTA (compile-time argument) offset where these args begin. Together with a base address and page size, they construct a `TensorAccessor` that maps page IDs to NoC addresses.

3. **Query**: "How does cb_reserve_back and cb_push_back work in TT-Metal circular buffers?"
   **Reason**: Needed to understand the producer-consumer synchronization model between reader and compute kernels.
   **Key Findings**: `cb_reserve_back` blocks until enough free space is available (producer side). `cb_push_back` increments the pages_received counter making data visible to consumers. `cb_wait_front` on the consumer side blocks until enough pages are available. `cb_pop_front` frees space for the producer.

4. **Query**: "What is the noc_async_read function and how does get_noc_addr work with it?"
   **Reason**: Needed to understand the DRAM reading mechanism used by the reader kernel.
   **Key Findings**: `noc_async_read(src_noc_addr, l1_write_addr, size)` initiates an async DMA transfer from a NoC address to local L1. `get_noc_addr(page_id, accessor)` uses TensorAccessor to compute the physical NoC address for a given page (stick). `noc_async_read_barrier` blocks until all outstanding reads complete.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding how RM and tiled layouts are represented in memory.
   **Key Information**: RM layout: each row is one page. Tiled layout: each 32x32 block is one page. Interleaved memory distributes pages round-robin across banks.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how the reader kernel addresses sticks in DRAM.
   **Key Information**: TensorAccessor abstracts bank-interleaved addressing. On device, `TensorAccessorArgs<CTA_OFFSET>()` reconstructs from compile-time args. `get_noc_addr(page_id)` returns the full NoC address.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `tilize_helpers.inl`
   **Reason**: Understanding the compute helper's internal implementation.
   **Key Information**: The helper supports symmetric (tile-page) and asymmetric (row-page) CB modes. WaitBlock mode waits per block. Automatically selects fast_tilize for bfloat16/32x32/half-sync. The helper manages init/uninit lifecycle and register reconfiguration.
