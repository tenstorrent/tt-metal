# Softmax (General W-Small Variant) Implementation Analysis

## Overview

This analysis covers the **SoftmaxProgramFactoryGeneralWSmall** variant of the TTNN softmax operation, which implements numerically stable softmax along the **last dimension (W)** of a tilized tensor. This variant is selected when the total circular buffer memory required to hold an entire row of tiles (Wt tiles) fits within the L1 memory threshold (512KB).

This variant is the most relevant reference for the **row_standardize** operation because both operations share a common pattern:
1. Row-wise reduction (softmax: max, sum; row_standardize: sum for mean, sum-of-squares for variance)
2. Element-wise subtraction of a per-row scalar (softmax: x - max; row_standardize: x - mean)
3. Element-wise final normalization (softmax: exp(x-max) / sum; row_standardize: (x-mean) * rsqrt(var+eps))

**Program Factory Path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`

**Kernel Paths** (all from `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`):
- Reader: `reader_moreh_softmax_w.cpp`
- Compute: `moreh_softmax_w.cpp`
- Writer: `writer_moreh_softmax_w.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | row (tile-row) |
| **Unit size** | Wt tiles (one complete row of tiles in the W dimension) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop: N kernel-rows assigned to this core. Inner: Wt tiles per row. |

One "work unit" is a single tile-row: the Wt tiles that span the full W dimension for one height-tile-row. The compute kernel processes `N` such rows per core, where `N = num_tiles_per_core_group`.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary rank, last two dims are H, W | Same as input |
| **Dimension convention** | [..., H, W] | [..., H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (typical) | DRAM (typical) |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
- No explicit tilize/untilize within the kernel pipeline.
- Input must already be in TILE_LAYOUT before reaching this factory.
- The host-side `softmax()` function will auto-tilize with -inf padding if needed before dispatching.

## Data Flow Pattern

The operation processes one tile-row (Wt tiles) at a time through a multi-stage compute pipeline:

### Stage 1: Read Input Row
**Reader** reads Wt tiles from DRAM into `cb_in0` (c_0) via `noc_async_read_tile`. All Wt tiles for one row are read as a batch, then pushed to the compute kernel.

### Stage 2: Find Row Maximum (Numerical Stability)
**Compute** performs `REDUCE_ROW` with `PoolType::MAX` across the Wt input tiles.
- For Wt == 1: Mask the single tile, then reduce it to get max.
- For Wt > 1: Reduce first (Wt-1) tiles with `WaitUpfrontNoPop` policy (tiles stay in CB for reuse), then mask the last tile and accumulate into the running max.
- Result: 1 tile in `cb_max` (c_26) containing the per-row maximum as a column vector.

### Stage 3: Subtract Maximum (x - max)
**Compute** subtracts `cb_max` from each of the Wt input tiles using `sub_tiles_bcast<BroadcastType::COL>`. The COL broadcast replicates the column-vector max across all columns of each tile. Result stored in `cb_x_m_max` (c_27).

### Stage 4: Exponentiate (exp(x - max))
**Compute** applies `exp_tile` to each of the Wt tiles from `cb_x_m_max`. The last tile is additionally masked (to zero out padding elements). Result stored in `cb_exps` (c_24).

### Stage 5: Sum Reduction and Reciprocal
**Compute** performs `REDUCE_ROW` with `PoolType::SUM` across the Wt exp tiles, using `WaitUpfrontNoPop` policy (exp tiles remain for the final multiply). A `recip_tile` post-reduce operation is applied inline via lambda to compute `1/sum(exp)`. Result: 1 tile in `cb_recipsumexps` (c_25).

### Stage 6: Final Normalization (exp / sum)
**Compute** multiplies each of the Wt exp tiles by the reciprocal sum using `mul_tiles_bcast_cols`. Again, COL broadcast replicates the per-row reciprocal across all columns. Result stored in `cb_out0` (c_16).

### Stage 7: Write Output Row
**Writer** reads Wt tiles from `cb_out0` and writes them to DRAM via `noc_async_write_tile`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tile row | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_mask | Width mask tile (handles padding) | 1 tile | 1 tile | Single | Reader (generated once) | Compute | Program |
| c_2 | cb_bcast_scaler | Reduce scaler (all 1.0) | 1 tile | 1 tile | Single | Reader (generated once) | Compute | Program |
| c_16 | cb_out0 | Output tile row | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row maximum | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary (masked tile for max reduction) | 1 tile | 1 tile | Single | Compute | Compute | Row |

**Data Format Notes**:
- c_0, c_1, c_2, c_16: Use the input tensor's data format (BFLOAT16 or FLOAT32).
- c_24, c_25, c_26, c_27, c_28: Use `intermed_data_format` which is FLOAT32 if `fp32_dest_acc_en` is set, otherwise matches input format. This ensures intermediate accumulations maintain precision.

## Pipeline Pattern Summary

All circular buffers have capacity equal to their block size (single-buffered). This means the reader, compute, and writer operate sequentially on each row: the reader fills c_0 for a row, compute processes all stages for that row, then the writer drains c_16. There is no overlap between reader and compute for consecutive rows since the reader must wait for compute to free c_0. This is acceptable because the compute pipeline for softmax is multi-stage and dominates execution time.

## Index Calculations

### Reader Index Mapping
- `tile_offset`: Starting tile index for this core (cumulative across cores).
- `curr_tile`: Incremented linearly: for each of N rows, reads Wt tiles sequentially starting at `tile_offset + row * Wt`.
- Uses `TensorAccessor` with compile-time args from `TensorAccessorArgs` for bank-interleaved addressing.

### Compute Index Mapping
- No explicit index calculations. The compute kernel consumes tiles from CBs in FIFO order.
- The Wt tiles in a CB represent one tile-row (all W-dimension tiles for a single height position).
- Within the row, tiles are indexed by `w` from 0 to Wt-1 for broadcast operations.

### Writer Index Mapping
- Mirrors reader: `tile_id` starts at `tile_offset`, increments through N * Wt tiles.

## Memory Access Patterns

### Read Pattern
- **Sequential tile reads**: For each row, reads Wt tiles in order from DRAM.
- All Wt tiles for a row are read in one burst (`cb_reserve_back(cb_in, Wt)` then Wt individual `noc_async_read_tile` calls followed by one barrier).
- Tiles are laid out contiguously in the tile-linearized order of the tensor.

### Write Pattern
- **Sequential tile writes**: For each row, writes Wt tiles in order to DRAM.
- All Wt tiles for a row are written in one burst (Wt individual `noc_async_write_tile` calls followed by one barrier).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (traversed as 1D) |
| **Grid dimensions** | device's full compute_with_storage_grid_size |
| **Total cores** | Determined by `split_work_to_cores_wt_core_range` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` rows |
| **Load balancing** | Two core groups: group 1 gets `ceil(total/cores)` rows, group 2 gets `floor(total/cores)` rows |

The work unit count is `num_kernel_rows = (physical_volume / H / W) * Ht`, which is the total number of tile-rows across all batches/channels. These are divided across cores using `split_work_to_cores_wt_core_range`, which creates two CoreRangeSets:
- **core_group_1**: Gets `num_tiles_per_core_group_1` rows (ceiling division).
- **core_group_2**: Gets `num_tiles_per_core_group_2` rows (floor division, possibly 0 if evenly divisible).

Core iteration order: `core = {i / core_h, i % core_h}` -- column-major traversal.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input is FLOAT32, 0 for BFLOAT16 |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank-interleaved addressing parameters for input tensor |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows to process on this core |
| 1 | Wt | uint32_t | Number of tiles in the W dimension (tiles per row) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Bank-interleaved addressing parameters for output tensor |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles per row (W dimension) |
| 4 | scaler | uint32_t | Scaler value as reinterpreted float bits (1.0f) |
| 5 | mask_w | uint32_t | Number of valid elements in last W tile (for masking padding) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles per row (W dimension) |

### Compile-Time Defines
| Define | Value | Description |
|--------|-------|-------------|
| SOFTMAX | 1 | Enables softmax path (skip negation before exp). Without this, computes softmin. |
| FP32_DEST_ACC_EN | 1 (conditional) | Enables FP32 accumulation in DST registers for higher precision |

## Kernel Implementations

### Reader: reader_moreh_softmax_w.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input tensor) | cb_in0, cb_mask, cb_scaler | Read tiles, generate mask and scaler |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`
- **Key Logic**:
  - **One-time setup**: Generates `cb_scaler` (all 1.0 values) and `cb_mask` (width mask for padding) using `generate_bcast_scaler` and `generate_mask_w` from `moreh_common.hpp`. These are persistent for the entire program.
  - **Per-row loop**: Reserves Wt tiles in cb_in0, reads Wt tiles sequentially via `noc_async_read_tile`, issues one barrier, then pushes all Wt tiles.
  - The `scaler` value is passed as `1.0f` reinterpreted as uint32_t, then used by `generate_bcast_scaler` which writes it into a tile format (filling the first 16 elements of each face-row).

### Compute: moreh_softmax_w.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | cb_in0, cb_mask, cb_scaler | cb_out0 | MAX reduce, SUB bcast, EXP, SUM reduce, RECIP, MUL bcast |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- **Key Logic**:

  **Initialization**:
  - `binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0)` -- initializes unpack/pack hardware.
  - Waits once for `cb_mask` and `cb_bcast_scaler` (persistent tiles, never popped).

  **Phase 1 - Row MAX**:
  - Uses `compute_kernel_lib::reduce<MAX, REDUCE_ROW>` from `reduce_helpers_compute.hpp`.
  - For Wt > 1: First reduces (Wt-1) tiles with `WaitUpfrontNoPop` policy (tiles remain in cb_in0 for reuse in Phase 2). Then masks the last tile into `cb_tmp`, and accumulates into `cb_max` using `Accumulate::at(cb_max, 1)`.
  - For Wt == 1: Masks the single tile, then reduces it.
  - Result: `cb_max` holds the per-row maximum as a column vector tile.

  **Phase 2 - Subtract MAX (x - max)**:
  - `cb_reserve_back(cb_x_m_max, Wt)` -- reserves output space for the full row.
  - Waits for `cb_in0` (Wt tiles) and `cb_max` (1 tile).
  - For each of Wt tiles: `sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0)` subtracts the max column-vector from each tile.
  - After all Wt tiles: pops cb_max (1 tile), pops cb_in0 (Wt tiles), pushes cb_x_m_max (Wt tiles).

  **Phase 3 - Exponentiate**:
  - `cb_reserve_back(cb_exps, Wt)` -- reserves output space.
  - For each of Wt tiles from cb_x_m_max: copies to DST, applies `exp_tile`. For the last tile (w == Wt-1), also applies mask_tile to zero out padding.
  - **Note**: When `SOFTMAX` is NOT defined (i.e., softmin mode), `negative_tile` is applied before exp.
  - Pushes cb_exps (Wt tiles). cb_x_m_max is NOT popped yet (still needed for LOG variant).

  **Phase 4 - Sum Reduction with Reciprocal**:
  - `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` over cb_exps.
  - Uses `WaitUpfrontNoPop` so exp tiles remain in CB for the final multiplication in Phase 5.
  - Post-reduce lambda: `recip_tile_init(); recip_tile(dst_idx);` computes 1/sum inline.
  - Result: `cb_recipsumexps` holds 1/sum(exp) as a column vector tile.

  **Phase 5 - Final Normalization**:
  - `cb_reserve_back(cb_out0, Wt)` -- reserves output space.
  - For each of Wt tiles: `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)` multiplies each exp tile by the reciprocal sum, broadcasting the column vector.
  - After all Wt tiles: pops cb_recipsumexps, cb_x_m_max, pushes cb_out0. Also pops cb_exps.

### Writer: writer_moreh_softmax_w.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | cb_out0 | DRAM (output tensor) | Write tiles |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`
- **Key Logic**:
  - Per-row loop: Waits for Wt tiles in cb_out0, writes them sequentially to DRAM via `noc_async_write_tile`, issues barrier, then pops Wt tiles.
  - Uses `TensorAccessor` with compile-time `TensorAccessorArgs` for bank-interleaved addressing.

## Implementation Notes

### Scaler Tile Format
The `generate_bcast_scaler` function creates a tile where the first 16 elements of each face-row (4 faces per tile) contain the scaler value, and the rest are zero. For BFLOAT16, the float value is reinterpreted and the upper 16 bits are used (`u >> 16`). For FLOAT32, the full 32-bit value is used. This format is required by the hardware reduce operations which expect the scaler in this specific layout.

### Mask Tile for Padding
The `generate_mask_w` function creates a tile where columns up to `mask_w` contain 1.0 and columns beyond contain 0.0. This is used in two places:
1. Before the MAX reduction, to mask the last tile so padding values do not affect the max.
2. After exponentiation of the last tile, to zero out exp values for padding positions.

### Width Mask Calculation
```cpp
uint32_t mask_w = input_tensor.logical_shape()[-1] % tt::constants::TILE_WIDTH;
if (mask_w == 0) mask_w = tt::constants::TILE_WIDTH;  // Full tile, no masking needed
```

### L1 Memory Budget Check
The `w_small` variant is only selected if the total CB memory fits within L1. The calculation considers:
- 3*Wt tiles for data format (input, output, x-max for non-intermed)
- 3*1 tiles for data format (mask, scaler, and one more)
- 3*Wt + 3*1 tiles for intermediate format (exps, reduce, max, x-max, tmp)
If total exceeds `L1_512KB`, the `w_large` variant is used instead, which processes tiles in blocks.

### Conditional Compilation (SOFTMAX vs LOG)
The compute kernel supports two modes via preprocessor defines:
- **`#define SOFTMAX`**: Standard softmax path: `exp(x-max) * (1/sum(exp))`
- **`#define LOG`**: Log-softmax path: `(x - max) - log(sum(exp))`
The `w_small` factory always defines `SOFTMAX=1`.

### FP32 Destination Accumulation
When `fp32_dest_acc_en` is true:
- Intermediate CBs use Float32 data format (even if input is BFLOAT16)
- The `FP32_DEST_ACC_EN` define is set, which triggers `pack_reconfig_data_format` and `reconfig_data_format_srca` calls in the `_with_dt` helper functions from `moreh_common.hpp`
- This is mandatory when input dtype is FLOAT32

### Broadcast Semantics (Critical for Row Standardize)
- **`BroadcastType::COL`**: The second operand (B) is treated as a column vector. `B[h]` is broadcast across all `w` positions. Used when the reduced result represents per-row values (e.g., row max, row mean).
- This is the key pattern for row_standardize: after computing mean and variance (both per-row column vectors via REDUCE_ROW), they are broadcast back to the full tile using COL broadcast for subtraction and multiplication.

### reduce_helpers_compute.hpp Library Usage
The compute kernel uses the `compute_kernel_lib::reduce<>` template function which encapsulates:
- DST register management (`tile_regs_acquire/commit/wait/release`)
- `reduce_init/reduce_uninit` hardware initialization
- CB synchronization (`cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`)
- `pack_tile` for writing results to output CB
- Accumulation across multiple calls (via `Accumulate` parameter)
- Post-reduce operations (via lambda, e.g., `recip_tile`, `log_tile`)

Key policies used:
- **`WaitUpfrontNoPop`**: Waits for all tiles, processes them with indexed access, does NOT pop. Tiles remain available for subsequent operations. Used in softmax for both MAX reduction (tiles reused for subtraction) and SUM reduction (exp tiles reused for final multiply).
- **`BulkWaitBulkPop`**: Waits for all tiles, processes, then pops all. Used in LOG variant.

## Relevance to Row Standardize

Row standardize follows a similar pattern but with different operations:

| Softmax Stage | Row Standardize Equivalent |
|---------------|--------------------------|
| MAX reduce along row | SUM reduce along row (for mean) |
| x - max | x - mean |
| exp(x - max) | (x - mean)^2 (for variance) |
| SUM reduce of exp | SUM reduce of (x-mean)^2 (for variance) |
| 1/sum (recip) | rsqrt(var + eps) |
| exp * (1/sum) | (x - mean) * rsqrt(var + eps) |

Key differences:
1. **Scaler value**: Softmax uses 1.0 as the reduce scaler. Row standardize needs `1/W` (or more precisely, `1/logical_W`) as the reduce scaler to compute mean = sum/W and variance = sum_sq/W.
2. **No exp operation**: Row standardize replaces exp with squaring (for variance computation).
3. **rsqrt instead of recip**: Row standardize applies `rsqrt_tile(var + eps)` instead of `recip_tile(sum)`.
4. **Epsilon addition**: Before rsqrt, need to add epsilon to the variance tile.
5. **Two reductions**: Row standardize needs two reductions (sum for mean, sum of squared diffs for variance) vs softmax's two reductions (max, sum of exps).

The CB layout pattern is directly transferable: use `c_0` for input, intermediate CBs (c_24-c_28) for mean, variance, x-mean, and the normalization factor, `c_16` for output.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the softmax operation work in the general (moreh) softmax implementation? Specifically for the W dimension variant (softmax_w_small), what is the compute pipeline for numerically stable softmax?"
   **Reason**: Needed to confirm the overall pipeline structure and understand the relationship between the factory and kernel files.
   **Key Findings**: Confirmed the 5-stage pipeline (max, subtract, exp, sum, divide), CB usage pattern, and that the w_small variant is selected based on L1 memory availability.

2. **Query**: "What are the BroadcastType::COL and BroadcastType::ROW operations in TT-Metal compute kernels? How do sub_tiles_bcast<BroadcastType::COL> and mul_tiles_bcast_cols work?"
   **Reason**: Needed to understand how per-row reduction results (column vectors) are broadcast back to full tiles for element-wise operations.
   **Key Findings**: BroadcastType::COL treats operand B as a column vector where B[h] is replicated across all w positions. The hardware unpack stage handles the replication. This is the key mechanism for applying per-row statistics (mean, variance) back to full tiles.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the reduce helper library used by the compute kernel.
   **Key Information**: The `reduce<>` template handles all reduce dimensions with multiple input policies. `WaitUpfrontNoPop` is the key policy for softmax patterns where tiles need to persist for reuse. Post-reduce lambdas allow inline operations like `recip_tile` and `log_tile`.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding the helper functions used in the compute kernel (sub_bcast_cols_init_short_with_dt, mul_bcast_cols_init_short_with_dt, pack_tile_with_dt, copy_tile_init_with_dt).
   **Key Information**: These `_with_dt` variants handle FP32 destination accumulation by conditionally calling `reconfig_data_format` and `pack_reconfig_data_format` when FP32_DEST_ACC_EN is defined. Essential pattern for supporting both BFLOAT16 and FLOAT32 inputs.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding the reader-side helper functions (generate_bcast_scaler, generate_mask_w).
   **Key Information**: `generate_bcast_scaler` fills specific positions in a tile (first 16 elements of each face-row across 4 faces) with the scaler value. For BFLOAT16, it extracts upper 16 bits of the float. `generate_mask_w` creates a binary mask tile with 1.0 for valid columns and 0.0 for padding.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp`
   **Reason**: Understanding program factory selection logic and L1 memory budget calculation.
   **Key Information**: The w_small variant is selected when `dim == rank - 1` and the total CB memory fits within 512KB of L1. The memory calculation accounts for all 9 CBs with their respective tile counts and data formats.

5. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp`
   **Reason**: Understanding the moreh helper APIs (CreateCircularBuffer, CreateComputeKernel, split_work_to_cores_wt_core_range).
   **Key Information**: `CircularBufferArg{buffer_index, num_tiles, data_format}` specifies each CB. `ComputeKernelArg` bundles core_spec with compile_args. `split_work_to_cores_wt_core_range` returns two core groups with balanced work distribution.
