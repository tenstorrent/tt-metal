# Moreh Norm W (Width Reduction) Implementation Analysis

## Overview

The `moreh_norm_w` operation computes the p-norm of a tensor along the **width (W) dimension**, reducing each row of tiles across the W dimension to a single output tile. It supports several norm variants depending on the value of `p`:

- **p = 0**: Counts non-zero elements (L0 "norm") using SUM reduction of `ne(x, 0)` results
- **p = -infinity**: Computes the minimum absolute value using negation + MAX reduction + negation
- **Other p values (default path)**: Computes `max(|x|)` across the row using MAX reduction (this is the L-infinity norm path for the "ord_other" variant)

**Program factory**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp`

This operation is relevant as a reference for LayerNorm because it demonstrates the complete pattern of width-dimension reduction on Tenstorrent hardware: reading tiles row-by-row, accumulating across tiles in the W dimension, and performing a final tile-internal reduce-row to collapse 32 columns into a single scalar per row.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | 1 tile-row = Wt tiles (all tiles in one row of the tiled tensor) |
| **Total units** | `num_units = (physical_volume / H / W) * Ht` -- i.e., the total number of tile-rows across all batches and height tiles |
| **Loop structure** | Outer loop: `num_rows_per_core` tile-rows; Inner loop: `Wt` tiles per row |

A "work unit" is one tile-row: the set of `Wt` tiles that share the same batch and height-tile index. Each tile-row is reduced to exactly one output tile.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [..., H, W] (arbitrary rank) |
| **Dimension convention** | Last dim = W, second-to-last = H |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 (detected at runtime via `is_dram()`) |
| **Data type** | Any (converted to DataFormat; intermediate uses Float32 if `fp32_dest_acc_en`) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [..., H, 1] (W dimension reduced) |
| **Dimension convention** | Same as input with W collapsed |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformations

No explicit tilize/untilize is performed. Both input and output are in TILE_LAYOUT. The reduction produces one output tile per tile-row, where only the first column of the output tile contains meaningful values (due to REDUCE_ROW collapsing all 32 columns into column 0).

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Reader | DRAM/L1 | CB c_1 (one) | `fill_cb_with_value` -- one-time fill of scaler tile |
| 1b | Reader | N/A | CB c_2 (mask_w) | `generate_mask_w` -- one-time generation if needed |
| 2 | Reader | DRAM/L1 | CB c_0 (input) | `cb_reserve_back(1)`, `noc_async_read_tile`, `cb_push_back(1)` -- one tile at a time, Wt times per row |
| 3 | Compute | CB c_0 | CB c_24 (val) | Apply f(x): abs/ne/mask, then pack to intermediate |
| 4 | Compute | CB c_24 | CB c_25 (cal) | Accumulate across tiles: copy (first tile) or max/add (subsequent tiles) |
| 5 | Compute | CB c_25 + CB c_1 | CB c_26 (reduce) | `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_ROW>` -- reduces 32 columns within the accumulated tile to a single column |
| 6 | Compute | CB c_26 | CB c_16 (output) | Copy (with optional negation for MINUS_INF), pack to output CB |
| 7 | Writer | CB c_16 | DRAM/L1 | `cb_wait_front(1)`, `noc_async_write_tile`, `cb_pop_front(1)` -- one tile per row |

### Detailed Flow per Tile-Row

1. **Reader** streams `Wt` input tiles one at a time into `c_0`.
2. **Compute** processes each tile through the element-wise function f(x):
   - For p=0: `unary_ne_tile(dst0, 0)` -- compares each element to zero
   - For general case: `abs_tile(dst0)` -- takes absolute value
   - For p=-inf: additionally `negative_tile(dst0)` -- negates
   - If last tile in row and width is not tile-aligned: applies `mask_tile` to zero out padding
3. **Compute** accumulates tile results in `c_25`:
   - First tile (col_idx=0): directly copies f(x) into `c_25`
   - Subsequent tiles: applies the reduction op (add for p=0, max for others) between current f(x) and accumulated value
4. After all `Wt` tiles are accumulated, **Compute** calls `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_ROW>` on the accumulated tile. This collapses the 32 columns within the tile into column 0, using the scaler tile from `c_1`.
5. The reduced result is copied (with optional negation for MINUS_INF) to the output CB `c_16`.
6. **Writer** writes one output tile per tile-row.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 1 tile | 1 tile | Single | Reader | Compute | Block (per tile) |
| c_1 | cb_one | Scaler tile (all 1.0f) | 1 tile | 1 tile | Single | Reader | Compute | Program (filled once, consumed at end) |
| c_2 | cb_mask_w | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program (generated once if needed) |
| c_16 | cb_output | Output tile staging | 1 tile | 1 tile | Single | Compute | Writer | Block (per row) |
| c_24 | cb_val (im0) | f(x) intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile) |
| c_25 | cb_cal (im1) | Cross-tile accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row (accumulates across Wt tiles) |
| c_26 | cb_reduce (im2) | Post-reduce result | 1 tile | 1 tile | Single | Compute | Compute | Block (per row) |

**Notes on data formats**: CBs c_24, c_25, c_26 use `intermed_data_format` which is Float32 when `fp32_dest_acc_en` is true, otherwise matches the input data format. All other CBs use the input data format.

## Pipeline Pattern Summary

All circular buffers have capacity = block size = 1 tile, meaning **all CBs are single-buffered**. There is no double-buffering overlap between reader and compute. The reader issues `noc_async_read_barrier()` after every tile read, creating a fully synchronous read-per-tile pattern. Similarly, the writer issues `noc_async_write_barrier()` after every tile write.

This means the pipeline is fully serialized: Reader waits for DRAM read to complete before pushing, compute waits for reader, writer waits for compute. No read/compute overlap occurs.

## Index Calculations

### Input Tile Indexing (Reader)

Tiles are accessed via a flat tile index using `TensorAccessor`:

```
tile_idx = start_tile_idx + row_idx * Wt + col_idx
```

Where:
- `start_tile_idx = tile_offset` (cumulative offset from all previous cores)
- `row_idx` iterates over `num_rows_per_core` tile-rows
- `col_idx` iterates over `Wt` tiles in the width dimension

The `TensorAccessor` maps this flat tile index to the actual physical DRAM address, handling bank interleaving internally.

### Output Tile Indexing (Writer)

```
tile_idx = start_tile_idx + row_idx
```

Where `start_tile_idx = tile_offset / Wt` -- the writer divides the input tile offset by Wt because the output has Wt fewer tiles per row (reduced). Each tile-row in the input produces exactly one output tile.

### Width Masking

When `origin_w % 32 != 0`, the last tile in each row contains padding elements. A mask tile is generated with:
- `mask_w = origin_w % 32` active columns
- Active columns get value 1.0, padding columns get 0.0
- Applied via `mask_tile(dst0, dst1)` which zeros out elements where mask is 0
- For MINUS_INF variant, `mask_posinf_tile` is used instead (sets masked elements to +inf so they don't affect the min)

## Memory Access Patterns

### Read Pattern

- **Pattern**: Sequential tile reads with stride
- **Ordering**: Row-major within each tile-row (iterate col_idx from 0 to Wt-1), then advance to next tile-row
- **Granularity**: One tile per NoC read transaction
- **Synchronization**: Full barrier after each tile (`noc_async_read_barrier()`)
- **Memory type**: DRAM (or L1) interleaved, accessed via TensorAccessor

### Write Pattern

- **Pattern**: Sequential tile writes, one per tile-row
- **Ordering**: One output tile per completed tile-row
- **Granularity**: One tile per NoC write transaction
- **Synchronization**: Full barrier after each tile (`noc_async_write_barrier()`)
- **Memory type**: DRAM (or L1) interleaved, accessed via TensorAccessor

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (column-major linearization of 2D grid) |
| **Grid dimensions** | Up to `grid.x * grid.y` from `compute_with_storage_grid_size()` |
| **Total cores** | `num_cores_to_be_used` (at most total available, at most `num_units`) |
| **Work per core** | `num_units_per_core_group_1` or `num_units_per_core_group_2` tile-rows |
| **Load balancing** | Two-group split via `split_work_to_cores()` |

### Work Splitting Details

The `split_work_to_cores` function divides `num_units` tile-rows across available cores:
- **core_group_1**: Gets `num_units_per_core_group_1` tile-rows (the larger share if uneven)
- **core_group_2**: Gets `num_units_per_core_group_2` tile-rows (one fewer, or empty if evenly divisible)

Core linearization: `core = {i / num_cores_y, i % num_cores_y}` -- cores are enumerated column-major (y varies fastest).

Tile offset tracking: Each core receives a `tile_offset` that is the cumulative sum of all tiles assigned to prior cores. This offset advances by `num_units_per_core * Wt` per core (since each tile-row contains Wt tiles).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Input buffer accessor metadata (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Output buffer accessor metadata |

#### Compute Kernel

The compute kernel has no explicit compile-time arguments. However, it receives **preprocessor defines** that control behavior:

| Define | Value | Description |
|--------|-------|-------------|
| `REDUCE_DIM` | `ReduceDim::REDUCE_ROW` | Always reduces along the row (width) dimension |
| `REDUCE_OP` | `PoolType::SUM` or `PoolType::MAX` | SUM for p=0, MAX otherwise |
| `IS_ZERO` | `1` (optional) | Present when p=0; triggers ne-zero counting path |
| `MINUS_INF` | `1` (optional) | Present when p=-infinity; triggers negation path |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer DRAM/L1 address |
| 1 | input_is_dram | uint32_t | 1 if input is in DRAM, 0 if L1 |
| 2 | num_rows_per_core | uint32_t | Number of tile-rows assigned to this core |
| 3 | Wt | uint32_t | Number of tiles in width dimension |
| 4 | tile_offset | uint32_t | Starting flat tile index for this core |
| 5 | origin_w | uint32_t | Original (unpadded) width of the tensor |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer DRAM/L1 address |
| 1 | output_is_dram | uint32_t | 1 if output is in DRAM, 0 if L1 |
| 2 | num_rows_per_core | uint32_t | Number of tile-rows (= output tiles) for this core |
| 3 | Wt | uint32_t | Width in tiles (used to compute output tile index) |
| 4 | tile_offset | uint32_t | Starting flat tile index (input-space, divided by Wt for output indexing) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Number of tile-rows to process |
| 1 | Wt | uint32_t | Number of tiles in width dimension |
| 2 | origin_w | uint32_t | Original (unpadded) width for masking |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_norm_w | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0, c_1, c_2 | Read input tiles, fill scaler, generate mask |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/reader_moreh_norm_w.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` with compile-time args at index 0 for address translation
  - Fills `cb_one` with 1.0f using `fill_cb_with_value` (used as scaler for reduce)
  - Conditionally generates width mask via `generate_mask_w` when `origin_w % 32 != 0`
  - Streams tiles one at a time with full read barrier per tile (no pipelining)
  - The `input_l1_write_ptr` is obtained once via `get_write_ptr(cb_id_input)` and reused (safe because CB capacity is 1 tile)

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| moreh_norm_w_kernel | TRISC (RISCV_2,3,4) | N/A | CB c_0, c_1, c_2 | CB c_16 | Element-wise transform, cross-tile accumulation, intra-tile reduce |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`
- **Key Logic**:
  - **Phase 1 - Element-wise f(x)**: For each input tile, applies the norm-specific transformation:
    - `IS_ZERO`: `unary_ne_tile` (compare != 0, producing 1 for non-zero elements)
    - Default: `abs_tile` (absolute value)
    - `MINUS_INF`: `abs_tile` followed by `negative_tile` (negate so MAX finds minimum)
    - Width masking applied to the last tile per row
  - **Phase 2 - Cross-tile accumulation**: Accumulates results across `Wt` tiles in `cb_cal`:
    - First tile: direct copy
    - Subsequent tiles: `add_tiles` (for IS_ZERO) or `binary_max_tile` (for default/MINUS_INF)
    - This uses the `c_25` CB as a ping-pong accumulator (pop old value, push new)
  - **Phase 3 - Intra-tile reduce**: Calls `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_ROW>` with `ReduceInputBlockShape::single()` (1x1x1) on the accumulated tile. This uses the hardware reduce instruction to collapse 32 columns into column 0 within the tile.
  - **Phase 4 - Output**: Copies the reduced tile to output CB, with optional negation for MINUS_INF to restore the sign.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_moreh_norm_w | NCRISC (RISCV_1) | NOC1 | CB c_16 | DRAM/L1 | Write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/writer_moreh_norm_w.cpp`
- **Key Logic**:
  - Uses `TensorAccessor` with compile-time args at index 0
  - Output tile index: `tile_offset / Wt + row_idx` -- converts input tile offset to output tile index
  - Writes one tile per tile-row with full write barrier
  - The `output_l1_read_addr` is obtained once via `get_read_ptr(cb_id_output)`

## Reduction Strategy Across Width Dimension

The width reduction uses a **two-phase approach**:

### Phase 1: Cross-Tile Accumulation (Manual)

The compute kernel manually accumulates across `Wt` tiles using a loop. For each tile-row:
1. Apply f(x) to each tile (abs, ne, or neg-abs depending on p)
2. Accumulate using the appropriate binary operation (add or max) into `cb_cal`
3. After all Wt tiles, `cb_cal` contains one tile where each element holds the accumulated value across all tiles that share the same (row, col_within_tile) position

### Phase 2: Intra-Tile Reduce (Hardware-Assisted)

The `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_ROW>` call performs the final reduction within the accumulated tile:
- Uses the hardware reduce instruction to collapse all 32 columns in each row to column 0
- The scaler tile (all 1.0f from `cb_one`) is used as the multiplication factor during reduction
- Called with `ReduceInputBlockShape::single()` since only one tile needs to be reduced

This two-phase approach is necessary because the hardware reduce instruction operates **within a single tile** (collapsing 32 columns), while the operation needs to reduce across **multiple tiles** in the width dimension.

## Implementation Notes

1. **Single-buffered pipeline**: All CBs have capacity of 1 tile with full barriers, meaning no overlap between read/compute/write. This is a simple but not performance-optimal approach.

2. **Accumulator ping-pong pattern**: The `cb_cal` buffer is used as both input and output during accumulation (pop old, compute new, push new). This works because CB capacity is 1 and operations are fully serialized.

3. **Program caching**: The `override_runtime_arguments` method only updates buffer addresses (args[0]), enabling efficient re-execution when only tensor addresses change but shapes remain the same.

4. **Compute defines control behavior**: The three variants (IS_ZERO, MINUS_INF, default) are selected at program creation time via preprocessor defines, resulting in three different compiled compute kernels. This avoids runtime branching in the hot loop.

5. **Width masking correctness**: For MINUS_INF, `mask_posinf_tile` is used instead of `mask_tile` to set padding elements to +infinity (which after negation becomes -infinity, ensuring padding does not affect the max result).

6. **The `ReduceInputBlockShape::single()` call**: This tells the reduce helper that the input is a single tile (1 row, 1 col, 1 batch). The cross-tile accumulation has already been done manually, so the hardware reduce only needs to handle the intra-tile column reduction.

## External Knowledge Sources

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the three-kernel programming model (reader/compute/writer), Tensix core architecture, and circular buffer semantics
   **Key Information**: Each Tensix core has 5 RISC-V CPUs, reader uses NOC0, writer uses NOC1, circular buffers provide inter-kernel synchronization via reserve_back/push_back/wait_front/pop_front

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps flat tile indices to physical DRAM bank addresses
   **Key Information**: TensorAccessor handles interleaved bank mapping through compile-time args that encode buffer layout metadata

3. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding TILE_LAYOUT and interleaved memory organization
   **Key Information**: In TILE_LAYOUT, tiles are stored in row-major order; interleaved layout distributes pages across DRAM banks for bandwidth

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the `compute_kernel_lib::reduce` helper function behavior
   **Key Information**: REDUCE_ROW collapses the W dimension (32 columns) within tiles; uses ReduceInputBlockShape to specify input dimensions; the scaler CB must be pre-filled before calling reduce

5. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding helper functions used in the reader kernel
   **Key Information**: `fill_cb_with_value` fills a CB tile with a constant; `generate_mask_w` creates a binary mask for partial width tiles; `Scalar` is a float/uint32_t union for type-punning

6. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding compute helper functions like `copy_tile_init_with_dt`, `pack_tile_with_dt`, `mask_tile`
   **Key Information**: `_with_dt` variants handle data format reconfiguration when FP32_DEST_ACC_EN is defined; `mask_tile` zeros elements where mask is 0; `binary_max_tile` computes element-wise max between two DST registers

7. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding core work distribution
   **Key Information**: `split_work_to_cores` divides N work units across a grid, returning two core groups (one possibly with one fewer unit) for load balancing
