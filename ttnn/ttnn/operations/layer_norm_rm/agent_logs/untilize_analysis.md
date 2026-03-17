# Untilize (Multi-Core) Implementation Analysis

## Overview

The **untilize** operation converts a tensor from tiled layout (32x32 tiles, face-major storage) to row-major layout (contiguous sticks). It reads tiled data from DRAM (interleaved) or L1 (sharded), converts the layout in compute via `pack_untilize_block`, and writes contiguous row-major sticks to the output buffer.

**Program factory**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_program_factory.cpp`

**Focus of this analysis**: Output stage -- how the compute kernel produces row-major sticks in the output CB, how the writer kernel extracts and writes those sticks to DRAM, output CB sizing, and the `untilize` helper signature/usage. This analysis serves as an output_stage reference for a new `layer_norm_rm` operation.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row (one row of tiles across the width) |
| **Unit size** | `num_tiles_per_input_block` tiles (= tensor_width / tile_width tiles) |
| **Total units** | `num_tiles_per_col` tile-rows (= tensor_height / tile_height) |
| **Loop structure** | Outer loop: `num_input_blocks_to_process` tile-rows per core; inner: implicit in untilize helper (32 stick-rows per tile-row) |

One "input block" corresponds to one tile-row: a horizontal strip of `num_tiles_per_input_block` tiles that together span the full tensor width (for interleaved) or the shard width (for sharded). Each such block produces `tile_height` (32) row-major sticks when untilized.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (flattened to 2D: H_total x W) | [N, C, H, W] (flattened to 2D: H_total x W) |
| **Dimension convention** | Last dim = W (contiguous) | Last dim = W (contiguous) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | ROW_MAJOR_LAYOUT (sticks) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) | DRAM (interleaved) or L1 (sharded) |
| **Data type** | bfloat16, float32, int32, uint32 | Same as input |

### Output Tensor Details

For the interleaved output case (primary focus for `layer_norm_rm`):
- **Page width** = `tensor_width` (one full row is one page)
- **Page size** = `tensor_width * element_size` bytes
- **Number of pages** = `tensor_height` (one page per stick/row)
- Pages are distributed round-robin across DRAM banks

For width-sharded or block-sharded output:
- **Page width** = `shard_spec.shape[1]` (shard width, may be less than tensor width)
- Multiple output pages may be needed per row of the tensor

### Layout Transformation

The core transformation is: **Tiled -> Row-Major**. The compute kernel's `pack_untilize_block` reads tiles from the input CB (in face-major tile format) and writes row-major sticks to the output CB. After processing one tile-row of `num_tiles_per_input_block` tiles:
- The output CB contains `tile_height` (32) contiguous sticks
- Each stick has width `num_tiles_per_input_block * tile_width` elements
- Stick layout: `[T0.row0 | T1.row0 | ... | Tn.row0]`, then `[T0.row1 | T1.row1 | ... | Tn.row1]`, etc.

---

## Data Flow Pattern

### High-Level Flow (Output Stage Focus)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (tiles) | CB c_0 (input) | `cb_reserve_back`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_16 (output) | `cb_wait_front`, `cb_pop_front` on c_0; `cb_reserve_back`, `cb_push_back` on c_16 |
| 3 | Writer | CB c_16 (output) | DRAM (sticks) | `cb_wait_front`, `cb_pop_front` on c_16 |

### Detailed Output Stage Flow

1. **Compute produces one block of sticks**: The `untilize` helper calls `pack_untilize_block` which reads `num_tiles_per_input_block` tiles from CB c_0, converts them, and writes `num_tiles_per_input_block` tiles-worth of row-major data into CB c_16. After `cb_push_back(c_16, num_tiles_per_input_block)`, the output CB contains `tile_height` (32) sticks of width `num_tiles_per_input_block * tile_width` elements.

2. **Writer waits for one block**: `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- the writer waits until a full block (one tile-row worth) is available in the output CB.

3. **Writer extracts sticks from the output CB**: The writer reads the base L1 address of the output CB via `get_read_ptr(cb_id_out0)`, then iterates over `tile_height` rows. For each row, it computes `current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size`, giving the start address of stick `j` in the CB.

4. **Writer writes sticks to DRAM**: For each stick, the writer uses `TensorAccessor::get_noc_addr(output_page_id, output_offset_within_page_in_bytes)` to get the destination NoC address, then issues `noc_async_write(current_l1_read_addr, dst_noc_addr, num_bytes_to_write)`.

5. **Writer issues barrier and pops**: After all `tile_height` sticks are written from one block, the writer calls `noc_async_write_barrier()` then `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` to release the CB space.

6. **Loop repeats**: The writer processes `num_input_blocks_to_process` blocks sequentially.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | 2 * `num_tiles_per_input_block` tiles (interleaved, multi-block); 1 * `num_tiles_per_input_block` tiles (single-block) | `num_tiles_per_input_block` tiles | Double (multi-block) or Single (single-block) | Reader | Compute | Block |
| c_16 | cb_output | Output stick staging | 2 * `num_tiles_per_input_block` tiles (multi-block); 1 * `num_tiles_per_input_block` tiles (single-block) | `num_tiles_per_input_block` tiles | Double (multi-block) or Single (single-block) | Compute | Writer | Block |

### Output CB Sizing (Primary Focus)

The output CB sizing logic (lines 148-162 of program factory):

```cpp
uint32_t output_cb_num_tiles;
if (num_input_blocks_per_full_core == 1) {
    output_cb_num_tiles = num_tiles_per_input_block;       // Single-buffered
} else {
    output_cb_num_tiles = num_tiles_per_input_block * 2;   // Double-buffered
}
```

**Key insight for `layer_norm_rm`**: The output CB capacity is always a multiple of `num_tiles_per_input_block` (the number of tiles across the width). When double-buffered, the CB holds 2 blocks, allowing the writer to drain one block while compute fills the next. The block size is `num_tiles_per_input_block * output_single_tile_size` bytes.

**What the output CB actually contains**: Despite being sized in "tiles", the output CB holds row-major sticks after untilize. Each "tile" of output CB space holds `tile_height * tile_width * element_size` bytes of row-major data. One block of `num_tiles_per_input_block` tiles in the output CB stores exactly `tile_height` sticks of width `num_tiles_per_input_block * tile_width` elements.

---

## Pipeline Pattern Summary

When `num_input_blocks_per_full_core > 1` (the common case):
- **Input CB c_0**: Double-buffered (capacity = 2 * block_size). Reader can fill block N+1 while compute processes block N.
- **Output CB c_16**: Double-buffered (capacity = 2 * block_size). Compute can fill block N+1 while writer drains block N.
- **Three-stage pipeline**: Reader -> Compute -> Writer, with double-buffering enabling overlap between adjacent stages.

When `num_input_blocks_per_full_core == 1`:
- Both CBs are single-buffered. No pipeline overlap is possible since only one block is processed.

---

## Index Calculations

### Output Page ID Calculation (Writer Kernel)

The writer calculates which output page (stick) to write to using:

```
output_page_id = num_rows_already_processed * num_output_blocks_across_width + width_wise_output_block_start_index
```

Where:
- `num_rows_already_processed = block_height_index * tile_height + j` (j is the row within the tile-row, 0..31)
- `num_output_blocks_across_width`: Number of output pages per row (1 for interleaved, potentially > 1 for width/block sharded)
- `width_wise_output_block_start_index`: Which output page column this core starts at

For the **interleaved output** case (focus for `layer_norm_rm`):
- `num_output_blocks_across_width = 1` (one page per row)
- `output_page_id = num_rows_already_processed * 1 + 0 = num_rows_already_processed`
- This means page_id directly equals the row index -- page 0 = row 0, page 1 = row 1, etc.

### TensorAccessor Address Resolution

The writer uses `TensorAccessor` for DRAM address calculation:

```cpp
constexpr auto dst_args = TensorAccessorArgs<8>();
const auto s = TensorAccessor(dst_args, dst_addr, output_stick_size);
// ...
uint64_t dst_noc_addr = s.get_noc_addr(output_page_id, output_offset_within_page_in_bytes);
```

- `TensorAccessorArgs<8>()`: Compile-time args start at index 8 (after the 8 explicit compile-time args)
- `TensorAccessor(args, base_addr, page_size)`: Creates accessor with custom page size = `output_stick_size`
- `get_noc_addr(page_id, offset)`: Returns the NoC address for the given page with an optional byte offset within the page

The `offset` parameter is critical for handling width/block sharded outputs where one input block's data may span multiple output pages or start mid-page. For interleaved output, `output_offset_within_page_in_bytes = 0` and `num_cols_already_processed_in_first_output_block = 0`.

---

## Memory Access Patterns

### Read Pattern (Reader -- De-emphasized)
- Tiles read sequentially by tile ID from DRAM, one tile at a time
- Uses `TensorAccessor::get_noc_addr(page_id)` for tile address resolution

### Write Pattern (Writer -- Primary Focus)

**For interleaved output (the common `layer_norm_rm` case)**:

1. Per tile-row block: 32 sequential stick writes to DRAM
2. Each stick is a full tensor row (`tensor_width * element_size` bytes)
3. Sticks are written to consecutive page IDs (row indices)
4. Within each stick: single contiguous `noc_async_write` of `output_stick_size` bytes
5. The writer's inner loop (`while (num_input_cols_processed < ...)`) executes exactly once per row since `num_unpadded_cols_per_input_block == num_cols_per_output_block` for interleaved
6. Pattern: **Sequential page writes, one `noc_async_write` per row, barrier per block**

**For width/block sharded output**:

- Input block data may span multiple output pages
- The inner while-loop may execute multiple times per row, writing partial sticks to different output pages
- The `output_offset_within_page_in_bytes` mechanism handles writing to mid-page positions

### NoC Write Details

- Uses `noc_async_write` (asynchronous, NOC1)
- Write barrier (`noc_async_write_barrier()`) is issued once per block (after all 32 rows are dispatched)
- This batches 32 async writes before a single barrier, which is efficient

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized) |
| **Grid dimensions** | Up to `device->compute_with_storage_grid_size()` |
| **Total cores** | `num_compute_cores` (determined by `split_blocks_for_tilize`) |
| **Work per core** | `num_rows_per_full_core` tile-rows (= `num_input_blocks_per_full_core`) |
| **Load balancing** | Near-equal with optional cliff core |

### Work Splitting (Interleaved Case)

The function `split_blocks_for_tilize(grid_size, num_tiles_per_col)` divides tile-rows across cores:
- `nblocks_per_core = ceil(num_tiles_per_col / grid_area)`
- `ncores = ceil(num_tiles_per_col / nblocks_per_core)`
- If `num_tiles_per_col % nblocks_per_core != 0`, a **cliff core** gets the remainder
- Full cores: each processes `num_rows_per_full_core` tile-rows
- Cliff core: processes `num_rows_per_cliff_core` tile-rows (< full)
- Cliff core only exists for interleaved input; sharded input never has a cliff core

### Work Splitting (Sharded Case)

- `num_compute_cores = shard_spec.grid.num_cores()`
- Each core processes its local shard
- No cliff core; uneven shards are handled by per-core runtime args (`num_input_blocks_to_process`, `num_unpadded_cols_per_input_block`)

---

## Arguments

### Compile-Time Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `cb_id_out0` | uint32_t | Output CB index (c_16) |
| 1 | `output_stick_size` | uint32_t | Size of one output page/stick in bytes (`output_page_width * element_size`) |
| 2 | `tile_height` | uint32_t | Tile height (32 for standard tiles) -- number of sticks per block |
| 3 | `num_tiles_per_input_block` | uint32_t | Tiles per block width (= tensor_width / tile_width for interleaved) |
| 4 | `num_output_blocks_across_width` | uint32_t | Output pages per tensor row (1 for interleaved) |
| 5 | `output_element_size` | uint32_t | Bytes per element (2 for bfloat16, 4 for float32) |
| 6 | `num_cols_per_input_block` | uint32_t | Columns in one input block (`num_tiles_per_input_block * tile_width`) |
| 7 | `num_cols_per_output_block` | uint32_t | Columns in one output page (`output_page_width`) |
| 8+ | TensorAccessorArgs | varies | Compile-time args for output TensorAccessor (bank coords, shapes, etc.) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_tile_cnt` | uint32_t | Tiles per block width (`num_tiles_per_input_block`) |
| 1 | `src_cb_id` | uint32_t | Input CB index (c_0) |
| 2 | `out_cb_id` | uint32_t | Output CB index (c_16) |

### Runtime Arguments

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `dst_addr` | uint32_t | Output buffer base address in DRAM |
| 1 | `num_input_blocks_to_process` | uint32_t | Number of tile-rows this core processes |
| 2 | `height_wise_input_block_start_index` | uint32_t | Global tile-row index where this core starts |
| 3 | `num_unpadded_cols_per_input_block` | uint32_t | Valid columns (handles uneven sharding; = `num_cols_per_input_block` for interleaved) |
| 4 | `width_wise_output_block_start_index` | uint32_t | Starting output page column (0 for interleaved) |
| 5 | `num_cols_already_processed_in_first_output_block` | uint32_t | Byte offset into first output page (0 for interleaved) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | uint32_t | Number of tile-rows (blocks) this core processes |

---

## Kernel Implementations

### Writer Kernel (Primary Focus)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_16 (row-major sticks) | DRAM (interleaved pages) | Extract sticks from CB, write via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- **Key Logic**:
  - The `write_tiles_in_current_block` lambda processes one tile-row block:
    1. `cb_wait_front(cb_id_out0, num_tiles_per_input_block)` -- wait for compute to produce one block
    2. `base_l1_read_addr = get_read_ptr(cb_id_out0)` -- get L1 address of first stick in block
    3. Loop over `tile_height` rows (j = 0..31):
       - Compute L1 read address for this stick: `base + j * num_cols_per_input_block * element_size`
       - Compute output page_id and within-page offset
       - Inner while-loop writes stick data to one or more output pages
       - Each write: `noc_async_write(l1_addr, noc_addr, num_bytes)`
    4. `noc_async_write_barrier()` -- wait for all 32 writes to complete
    5. `cb_pop_front(cb_id_out0, num_tiles_per_input_block)` -- release CB space
  - The main loop calls `write_tiles_in_current_block` for each tile-row block
  - The writer handles uneven sharding (partial last shard width-wise) via `num_unpadded_cols_per_input_block`

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (unpack+math+pack) | N/A | CB c_0 (tiles) | CB c_16 (row-major sticks) | `pack_untilize_block` |

- **File**: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(src_cb_id, out_cb_id)` for hardware initialization
  - Calls `compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id, InitAndUninit, WaitBlock, NoReconfigure>(per_core_block_cnt)`
  - The `untilize` helper (from `untilize_helpers.hpp`/`.inl`) handles:
    - DEST register capacity detection via `DEST_AUTO_LIMIT`
    - If `block_width_tiles <= DEST_AUTO_LIMIT`: single-pass `pack_untilize_block` (optimal path)
    - If `block_width_tiles > DEST_AUTO_LIMIT`: block-based path that splits into sub-blocks
    - `WaitBlock` mode: waits for input tiles per block via `cb_wait_front`
    - Automatically handles `cb_reserve_back`/`cb_push_back` on the output CB
    - Automatically handles `cb_wait_front`/`cb_pop_front` on the input CB

### Reader Kernel (De-emphasized)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (tiles) | CB c_0 | Read tiles via TensorAccessor |

- **File** (interleaved): `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp`
- **File** (sharded): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

---

## Untilize Helper Signature and Usage (Key Reference)

### Header

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
```

### Full Template Signature

```cpp
template <
    uint32_t block_width_tiles,    // Number of tiles per row (MUST be compile-time)
    uint32_t input_cb,             // Input CB index (tiled data)
    uint32_t output_cb,            // Output CB index (row-major output, must differ from input_cb)
    untilize_config::InitUninitMode init_uninit_mode = InitAndUninit,
    untilize_config::WaitMode wait_mode = WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure>
void untilize(uint32_t num_blocks);   // num_blocks = runtime, number of tile-rows to process
```

### Prerequisite

```cpp
compute_kernel_hw_startup(input_cb, output_cb);  // MUST be called before untilize()
```

### Configuration Enums

**InitUninitMode** (for chaining multiple untilize calls):
- `InitAndUninit` -- Default, standalone call
- `InitOnly` -- First of multiple back-to-back calls
- `Neither` -- Middle calls
- `UninitOnly` -- Last call

**WaitMode**:
- `WaitBlock` -- Default, waits for input per block (one tile-row at a time)
- `WaitUpfront` -- Waits for all tiles upfront before processing
- `NoWait` -- Caller manages synchronization

**ReconfigureRegisterDatatypeMode**:
- `NoReconfigure` -- No data format reconfiguration
- `UnpackReconfigure` -- Reconfigure unpack registers
- `PackReconfigure` -- Reconfigure pack registers
- `UnpackAndPackReconfigure` -- Default, reconfigure both

### Usage Example (as seen in untilize operation)

```cpp
// In compute kernel:
compute_kernel_hw_startup(src_cb_id, out_cb_id);
compute_kernel_lib::untilize<
    per_core_block_tile_cnt,       // e.g., Wt = tensor_width / 32
    src_cb_id,                     // c_0
    out_cb_id,                     // c_16
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
```

### For `layer_norm_rm` Integration

When using untilize as the output stage of a fused kernel:
- Use `WaitMode::NoWait` or `WaitMode::WaitBlock` depending on whether the preceding compute stage pushes data into the untilize input CB block-by-block
- Use `InitUninitMode::InitOnly` / `Neither` / `UninitOnly` to avoid redundant init/uninit when chaining with other compute operations (tilize, reduce, etc.)
- Use `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure` (or appropriate subset) when switching data formats between compute stages
- The output CB MUST have capacity >= `block_width_tiles` tiles
- The input CB MUST have capacity >= `block_width_tiles` tiles (or `sub_block_width` if block-based path)

---

## Implementation Notes

### Output CB Data Interpretation

The output CB is configured with tile-sized pages (`output_single_tile_size`), but after untilize the data inside is row-major. The writer kernel does NOT use tile-based addressing on the output CB. Instead, it:
1. Gets the raw L1 pointer via `get_read_ptr(cb_id_out0)`
2. Manually computes stick offsets: `base + row * width_bytes`
3. Writes raw bytes to DRAM

This is important: the CB's "tile" page size is used for space accounting (reserve/push/wait/pop operate in tile units), but the actual data is row-major sticks.

### Width Splitting Between Input and Output Pages

The writer's inner while-loop handles the case where input block width and output page width differ (e.g., input is height-sharded with wide shards but output is width-sharded with narrow pages). The loop writes partial sticks to consecutive output pages, handling:
- `num_cols_already_processed_in_first_output_block`: byte offset into the first output page
- `num_cols_remaining_in_current_output_block`: bytes left to write in current output page
- `num_unpadded_cols_per_input_block`: skip trailing garbage in uneven last shard

For interleaved-to-interleaved (the `layer_norm_rm` case), these complications vanish: one stick = one page = one write.

### FP32 Destination Accumulation

When `fp32_dest_acc_en` is true (for float32/int32/uint32 types), the compute kernel is configured with `UnpackToDestMode::UnpackToDestFp32` on the input CB. This uses FP32 precision in the DEST accumulator during untilize. The `DST_ACCUM_MODE` define is also set for INT32/UINT32/FLOAT32 data types.

### Barrier Placement

The writer issues `noc_async_write_barrier()` once per block (after all `tile_height` = 32 writes), not per individual write. This allows 32 asynchronous writes to be pipelined before synchronizing. This is an important efficiency pattern to replicate: batch writes, barrier once.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the untilize compute operation work? What does pack_untilize_block do?"
   **Reason**: Needed to understand the transformation from tiled to row-major format in the compute kernel.
   **Key Findings**: `pack_untilize_block` reads tiled data from the input CB, processes it through the FPU/SFPU pack hardware, and writes row-major data to the output CB. It iterates over sub-rows within tiles to produce contiguous sticks. The fast-path (pack_untilize) is preferred over the slow-path (unpack_untilize).

2. **Query**: "How does split_blocks_for_tilize work in tt-metal?"
   **Reason**: Needed to understand how tile-rows are distributed across cores and what the cliff core concept means.
   **Key Findings**: The function divides `num_tiles_per_col` tile-rows evenly across available cores. If the division is uneven, a single "cliff core" receives the remainder. Full cores are grouped in `core_range`, the cliff core (if any) in `cliff_core_range`. This is a 1D work split across the height dimension.

3. **Query**: "After pack_untilize_block processes a row of tiles, what is the exact data layout in the output CB?"
   **Reason**: Needed to confirm the stick layout for the writer kernel's memory access pattern.
   **Key Findings**: For a tile-row of N tiles (each 32x32), the output CB contains 32 contiguous sticks of width N*32 elements. Stick j contains `[T0.rowj | T1.rowj | ... | Tn.rowj]`. This matches the writer's assumption of `current_l1_read_addr = base + j * width_bytes`.

### Documentation References

1. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding row-major page structure for the output tensor.
   **Key Information**: In row-major layout, each row of the 2D tensor is one page. Pages are interleaved round-robin across DRAM banks. For a tensor with width W and element size E, page size = W * E bytes.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how the writer resolves page addresses for DRAM writes.
   **Key Information**: `TensorAccessor(args, base_addr, page_size)` creates an accessor with custom page size. `get_noc_addr(page_id, offset)` returns the NoC address for a page with an optional byte offset within the page. Host-side: `TensorAccessorArgs(buffer).append_to(compile_time_args)` appends accessor configuration to compile-time args.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `untilize_helpers.inl`
   **Reason**: Understanding the untilize helper API, template parameters, and dispatch logic.
   **Key Information**: The helper selects between single-pass (block fits in DEST) and block-based (splitting into sub-blocks) paths at compile time based on `DEST_AUTO_LIMIT`. It manages all CB operations (wait/pop/reserve/push) internally. The `WaitMode`, `InitUninitMode`, and `ReconfigureRegisterDatatypeMode` enums provide fine-grained control for fused kernels.

4. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used to configure circular buffers.
   **Key Information**: `create_cb(cb_index, program, core_range, page_size, num_pages, data_format, buffer)` creates a CB with total size = `num_pages * page_size`. If `buffer` is non-null, the CB is backed by a globally-allocated buffer (used for sharded input).
