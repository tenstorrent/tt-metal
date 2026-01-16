# Layernorm (Interleaved Multi-Core) Implementation Analysis

## Overview

This analysis documents the layernorm operation implementation for interleaved (non-sharded) tensors in TTNN. Layernorm normalizes each row of the input tensor independently, computing per-row mean and variance, then applying the transformation: `output = (x - mean) / sqrt(variance + eps) * gamma + beta`.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp`

**Key Features**:
- Row-wise statistics (mean, variance) computed independently per row
- Gamma (scaling) and beta (bias) broadcast across rows (applied per-column)
- Support for optional fused pre-add (residual connection): `x = a + b`
- Two norm types: LAYERNORM (with mean subtraction) and RMSNORM (without mean subtraction)
- Optional Welford algorithm for numerically stable variance computation
- Large tensor support for tensors exceeding L1 capacity

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row |
| **Unit size** | Wt tiles (all tiles along width dimension) |
| **Total units** | NCHt = NC * Ht (number of channel-height tile rows) |
| **Loop structure** | Outer: NCHt rows assigned across cores; Inner: Wt tiles per row processed in blocks |

A single work unit is **one tile row** (Wt tiles). Each core processes `num_tile_rows_per_core` complete rows. This is critical because layernorm computes row-wise statistics - all Wt tiles in a row must be available to compute the mean and variance for that row.

## Tensor Format and Layout

### Input Tensor (a)

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (batch, channel, height, width) |
| **Dimension convention** | Last dimension (W) is normalized |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16, BFLOAT8_B, or FLOAT32 |

### Optional Residual Input (b) - Fused Pre-Add

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input (a) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |
| **Purpose** | Fused residual connection: x = a + b |

### Gamma Tensor (weight)

| Property | Value |
|----------|-------|
| **Logical shape** | [1, 1, 1, W] (1D along last dimension) |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT16_B |
| **Broadcast pattern** | Broadcast across rows (multiply each column) |

### Beta Tensor (bias)

| Property | Value |
|----------|-------|
| **Logical shape** | [1, 1, 1, W] (1D along last dimension) |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT16_B |
| **Broadcast pattern** | Broadcast across rows (add to each column) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

## Data Flow Pattern

The layernorm kernel follows a three-phase data flow for each row:

### Phase 1: Statistics Computation (Mean and Variance)

```
[DRAM] Input tiles (a) --> [CB_in0] --> [Compute: Row-wise Sum] --> [CB_ex] E[x] (mean)
                                    --> [Compute: x - E[x]] --> [CB_xmm] centered values
                                    --> [Compute: (x-E[x])^2] --> [CB_xmm2]
                                    --> [Compute: Row-wise Sum] --> [CB_ex2] Var[x]
```

### Phase 2: Normalization

```
[CB_ex2] Var[x] + [CB_eps] epsilon --> [Compute: add + rsqrt] --> [CB_ex2pe] 1/sqrt(Var+eps)
[CB_xmm] (x-E[x]) * [CB_ex2pe] --> [CB_fusion or CB_out] normalized values
```

### Phase 3: Affine Transform (Gamma/Beta Application)

```
[CB_fusion] normalized * [CB_gamma] --> [CB_fusion or CB_out] scaled
[CB_fusion] scaled + [CB_beta] --> [CB_out] --> [DRAM] Output
```

### Detailed Step-by-Step Flow

1. **Reader**: Reads input tiles (a) from DRAM into CB_in0 in blocks of `blk` tiles
2. **Reader** (if fuse_pre_add): Reads residual tiles (b) from DRAM into CB_in1
3. **Compute** (if fuse_pre_add): Adds a + b, stores result in CB_x (CB_c_23)
4. **Compute** (if !RMSNORM): Computes row-wise mean E[x] using reduce_tile, stores in CB_ex (CB_c_18)
5. **Compute** (if !RMSNORM): Subtracts mean from input: (x - E[x]), stores in CB_xmm (CB_c_24)
6. **Compute**: Squares centered values: (x - E[x])^2, stores in CB_xmm2 (CB_c_20)
7. **Compute**: Computes variance Var[x] using row-wise reduction, stores in CB_ex2 (CB_c_19)
8. **Compute**: Adds epsilon: Var[x] + eps
9. **Compute**: Computes rsqrt: 1/sqrt(Var[x] + eps), stores in CB_ex2pe (CB_c_21)
10. **Compute**: Normalizes: (x - E[x]) * rsqrt, stores in CB_fusion (CB_c_22) or CB_out
11. **Reader** (first row only): Reads gamma tiles into CB_gamma (CB_c_5)
12. **Reader** (first row only): Reads beta tiles into CB_beta (CB_c_6)
13. **Compute** (if gamma): Multiplies with gamma using row broadcast, stores in CB_fusion or CB_out
14. **Compute** (if beta): Adds beta using row broadcast, stores in CB_out (CB_c_16)
15. **Writer**: Writes output tiles from CB_out to DRAM

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (a) | Wt tiles (or 2*blk if fuse_pre_add) | blk | Multi | Reader | Compute | Row |
| c_1 | cb_in1 | Residual input (b) | 2*blk tiles | blk | Double | Reader | Compute | Row |
| c_2 | cb_scaler | Reduce scaler (1.0 tile) | 2 tiles | 1 | Single | Reader | Compute | Program |
| c_3 | cb_eps | Epsilon scalar tile | 2 tiles | 1 | Single | Reader | Compute | Program |
| c_5 | cb_gamma | Gamma weights | Wt tiles | blk | Multi | Reader | Compute | Program |
| c_6 | cb_beta | Beta bias | Wt tiles | blk | Multi | Reader | Compute | Program |
| c_16 | cb_out | Output tiles | 2*blk tiles | blk | Double | Compute | Writer | Block |
| c_18 | cb_ex | E[x] mean tile | 2 tiles | 1 | Single | Compute | Compute | Row |
| c_19 | cb_ex2 | Var[x] variance tile | 2 tiles | 1 | Single | Compute | Compute | Row |
| c_20 | cb_xmm2 | (x-E[x])^2 tiles | Wt tiles | blk | Multi | Compute | Compute | Row |
| c_21 | cb_ex2pe | 1/sqrt(Var+eps) tile | 8 tiles | 1 | Multi | Compute | Compute | Row |
| c_22 | cb_fusion | Intermediate for gamma/beta | 2*blk tiles | blk | Double | Compute | Compute | Block |
| c_23 | cb_x | x=a+b result (fuse_pre_add only) | Wt tiles | blk | Multi | Compute | Compute | Row |
| c_24 | cb_xmm | x - E[x] centered values | Wt tiles | blk | Multi | Compute | Compute | Row |

### CB Sizing Notes

- **Block size (blk)**: 4 tiles for fp32_dest_acc_en=true, 8 tiles otherwise
- **Wt_next_block_up**: Wt rounded up to nearest multiple of blk
- **Large tensor mode**: Reduces buffer sizes to fit in L1 (56 or 112 tiles max)
- **cb_data_format**: Float32 if fp32_dest_acc_en, else Float16_b for intermediates
- **Gamma/beta CBs**: Read once on first row, reused for all NCHt rows (never popped)

## Pipeline Pattern Summary

| Pattern | CBs | Description |
|---------|-----|-------------|
| **Double-buffered I/O** | cb_in1, cb_out, cb_fusion | Capacity = 2x block size enables overlap |
| **Row-persistent** | cb_gamma, cb_beta | Loaded once, reused across all rows |
| **Row-scoped** | cb_in0, cb_xmm, cb_xmm2, cb_x | Entire row's tiles held simultaneously |
| **Scalar** | cb_scaler, cb_eps, cb_ex, cb_ex2, cb_ex2pe | Single-tile results for row statistics |

The key insight is that row-wise reduction requires all Wt tiles to be available before computing statistics. This necessitates large CB capacities (Wt tiles) for input and intermediate buffers.

## Index Calculations

### Tile Offset Computation (Program Factory)

```cpp
// Linear tile indexing across all rows
uint32_t tile_offset = curr_row * Wt;  // Starting tile for this core's rows
```

### Reader Kernel Tile Access

```cpp
// For each row iteration
uint32_t offs = ncht * Wt;  // Row offset
// For each block within row
const auto total_offset = offs + block.start() + tile_offset;
noc_async_read_tile(total_offset + r, addr, l1_write_addr);  // r is local index within block
```

### Gamma/Beta Tile Access

```cpp
// Gamma and beta are 1D tensors (1,1,1,Wt), indexed by column tile only
// Read once on ncht==0, then reused
block.start()  // Column tile index within row
```

### TensorAccessor Usage

The implementation uses `TensorAccessor` with `TensorAccessorArgs` for compile-time configuration:
- `src0_args`: Input tensor accessor
- `src1_args`: Residual tensor accessor (if fuse_pre_add)
- `gamma_args`: Gamma tensor accessor
- `beta_args`: Beta tensor accessor
- `dst_args`: Output tensor accessor

## Memory Access Patterns

### Read Pattern

| Data | Pattern | Description |
|------|---------|-------------|
| Input (a) | Sequential tiles | Wt tiles per row, processed in blk-sized blocks |
| Residual (b) | Sequential tiles | Same pattern as input, parallel read |
| Gamma | Sequential, once | Wt tiles, read on first row only |
| Beta | Sequential, once | Wt tiles, read on first row only |
| Scaler | Generated | Created in L1, filled with 1.0 values |
| Epsilon | Generated | Created in L1, column broadcast scalar |

### Write Pattern

| Data | Pattern | Description |
|------|---------|-------------|
| Output | Sequential tiles | Wt tiles per row, written in blk-sized blocks |

### Access Characteristics

- **DRAM reads**: Coalesced, tile-sized transactions
- **DRAM writes**: Coalesced, tile-sized transactions with barrier
- **L1 accesses**: Direct pointer manipulation for scalar tiles
- **NoC barrier**: After each block read/write for synchronization

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized row-major) |
| **Grid dimensions** | Up to device's compute_with_storage_grid_size |
| **Total cores** | min(grid_size.x * grid_size.y, num_tile_rows) |
| **Work per core** | num_tile_rows_per_core rows (each = Wt tiles) |
| **Load balancing** | split_work_to_cores creates 2 groups for remainder handling |

### Work Distribution Algorithm

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);
```

- **core_group_1**: Cores with `num_tile_rows_per_core_group_1` rows
- **core_group_2**: Cores with `num_tile_rows_per_core_group_2` rows (handles remainder)
- Rows are contiguous: each core processes sequential rows

## Arguments

### Compile-Time Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | blk | uint32_t | Block size (4 or 8) |
| 1 | use_welford | uint32_t | Whether to use Welford algorithm |
| 2+ | src0_args | TensorAccessorArgs | Input tensor accessor config |
| N+ | src1_args | TensorAccessorArgs | Residual tensor accessor config |
| M+ | gamma_args | TensorAccessorArgs | Gamma tensor accessor config |
| P+ | beta_args | TensorAccessorArgs | Beta tensor accessor config |
| Last | tile_size/stick_size | uint32_t | Tile size or RM stick size |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor DRAM address |
| 1 | NCHt | uint32_t | Number of tile rows for this core |
| 2 | Wt | uint32_t | Tiles per row (width in tiles) |
| 3 | tile_offset | uint32_t | Starting tile index |
| 4 | packed_one_value | uint32_t | Packed bfloat16 1.0 for scaler |
| 5 | eps | uint32_t | Epsilon value (bit_cast float) |
| 6 | gamma_addr | uint32_t | Gamma tensor DRAM address |
| 7 | beta_addr | uint32_t | Beta tensor DRAM address |
| 8 | b_addr | uint32_t | Residual tensor DRAM address |
| 9 | W | uint32_t | Logical width (for partial tile handling) |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | blk | uint32_t | Block size (4 or 8) |
| 2 | do_gamma | uint32_t | Apply gamma scaling |
| 3 | do_beta | uint32_t | Apply beta bias |
| 4 | fp32_dest_acc_en | uint32_t | FP32 accumulation enabled |
| 5 | float32_reduction | uint32_t | Use FP32 for reduce |
| 6 | legacy_rsqrt | uint32_t | Use legacy rsqrt implementation |
| 7 | W | uint32_t | Logical width |

### Runtime Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | NCHt | uint32_t | Number of tile rows for this core |

### Compile-Time Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | blk | uint32_t | Block size (4 or 8) |
| 1+ | dst_args | TensorAccessorArgs | Output tensor accessor config |

### Runtime Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor DRAM address |
| 1 | Wt | uint32_t | Tiles per row |
| 2 | num_tile_rows | uint32_t | Number of rows for this core |
| 3 | tile_offset | uint32_t | Starting tile index |

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (a, b, gamma, beta) | cb_in0, cb_in1, cb_gamma, cb_beta, cb_scaler, cb_eps | Read tiles, generate scalers |

**Key Logic**:
1. Generates reduce scaler tile (all 1.0s) into cb_scaler using `generate_reduce_scaler`
2. Generates partial reduce scaler if W not divisible by TILE_WIDTH
3. Generates epsilon scalar tile into cb_eps using `generate_bcast_col_scalar`
4. For each row (NCHt iterations):
   - Reads input tiles in blocks using `read_block_to_cb` helper
   - If fuse_pre_add: Also reads residual tiles
5. On first row only: Reads gamma and beta tiles (they persist for all rows)

**CB Interactions**:
- `cb_reserve_back(cb_id, full_block_size)` before reading block
- `noc_async_read_tile` for each tile
- `noc_async_read_barrier` after block
- `cb_push_back(cb_id, full_block_size)` to signal compute

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | cb_in0, cb_in1, cb_gamma, cb_beta, cb_scaler, cb_eps | cb_out | Mean, variance, normalize, scale, bias |

**Key Logic**:

1. **Pre-add (if FUSE_PRE_ADD)**:
   ```cpp
   add_tiles(cb_in, cb_inb, i, i, i);  // x = a + b
   pack_tile(i, cb_x);
   ```

2. **Mean computation (if !RMSNORM)**:
   ```cpp
   row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(
       cb_x, cb_scaler, cb_ex, W, Wt, blk);
   ```
   - Uses `reduce_tile` to sum across row
   - Scales by 1/W in epilogue using `scale_dest`

3. **Center values (if !RMSNORM)**:
   ```cpp
   sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);  // x - E[x]
   pack_tile(i, cb_xmm);
   ```

4. **Squared differences**:
   ```cpp
   mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i);  // (x - E[x])^2
   pack_tile(i, cb_xmm2);
   ```

5. **Variance**:
   ```cpp
   row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(
       cb_xmm2, cb_scaler, cb_ex2, W, Wt, blk);
   ```

6. **Reciprocal sqrt**:
   ```cpp
   add_tiles(cb_ex2, cb_eps, 0, 0, dst0);  // Var + eps
   rsqrt_tile<LEGACY_RSQRT>(dst0);          // 1/sqrt(Var + eps)
   pack_tile(dst0, cb_ex2pe);
   ```

7. **Normalize**:
   ```cpp
   mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, global_i, 0, i);  // (x-E[x]) * rsqrt
   pack_tile(i, cb_im_or_out);
   ```

8. **Apply gamma (if do_gamma)**:
   ```cpp
   mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, global_i, i);  // * gamma
   pack_tile(i, cb_outg);
   ```

9. **Apply beta (if do_beta)**:
   ```cpp
   add_tiles_bcast_rows(cb_fusion, cb_beta, i, global_i, i);  // + beta
   pack_tile(i, cb_out);
   ```

**Broadcast Semantics**:
- `bcast_cols`: B is a single-column tile, broadcast to all columns of A
  - Used for mean and rsqrt (scalar per row, applied to all columns)
- `bcast_rows`: B is a single-row tile, broadcast to all rows of A
  - Used for gamma and beta (vector per column, applied to all rows)

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | cb_out | DRAM | Write tiles |

**Key Logic**:
1. For each row:
   - For each block in row:
     - `cb_wait_front(cb_out, full_block_size)` - wait for compute
     - Write tiles using `noc_async_write_tile`
     - `noc_async_write_barrier`
     - `cb_pop_front(cb_out, full_block_size)` - free buffer

## Implementation Notes

### Block Processing Pattern

The implementation uses a `BlockedRange` utility class for structured iteration:
```cpp
for (auto block : generic::blocks(Wt, blk)) {
    // block.start() - global starting tile index
    // block.size() - actual tiles in this block (may be < blk for last block)
    // block.full_block_size() - blk (for CB sync alignment)
    // block.local() - range [0, size) for local iteration
    // block.to_global(i) - converts local index to global
}
```

This ensures CB operations use aligned sizes while handling partial last blocks correctly.

### Gamma/Beta Reuse Pattern

Gamma and beta are read only once (on first row) and persist in their CBs:
```cpp
if (ncht == 0) {
    // Read gamma and beta tiles
}
// Compute uses them for all rows - never cb_pop_front
```

This is efficient because gamma and beta are 1D tensors that apply identically to every row.

### Partial Tile Handling

For tensors where W is not divisible by TILE_WIDTH (32):
1. Reader generates a second scaler tile in cb_scaler with zeros in unused columns
2. Compute uses `scaler_tile_idx = (is_last_tile && partial) ? 1 : 0`
3. This ensures reduction only sums valid elements

### Large Tensor Mode

When CBs don't fit in L1:
1. `large_tensor_needed = true` triggers reduced buffer sizes
2. Uses different kernel variants that process in smaller chunks
3. May require multiple passes over input data

### Conditional Compilation

The kernel uses preprocessor defines for variant selection:
- `FUSE_PRE_ADD`: Enable residual addition
- `FUSE_GAMMA`: Enable gamma scaling
- `FUSE_BETA`: Enable beta bias
- `RMSNORM`: Skip mean computation (RMS normalization)

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What are broadcast operations (bcast_cols, bcast_rows) in tt-metal compute kernels and how do they work?"
   **Reason**: Understanding how gamma and beta are applied across tiles
   **Key Findings**:
   - `bcast_cols`: B is single-column tile, result C[h,w] = A[h,w] op B[w]
   - `bcast_rows`: B is single-row tile, result C[h,w] = A[h,w] op B[h]
   - Implemented via `llk_unpack_AB` with broadcast type and `llk_math_eltwise_binary`
   - Initialization functions (`add_bcast_rows_init_short`, etc.) configure hardware

2. **Query**: "How does layernorm normalization work in TTNN? What is the algorithm for computing row-wise mean and variance?"
   **Reason**: Verifying the mathematical algorithm and understanding Welford variant
   **Key Findings**:
   - Standard: mean via sum/N, variance via E[(x-mean)^2]
   - Welford: Iterative update of running mean and M2 for numerical stability
   - `numeric::row_wise_mean` function used for both mean and variance computation
   - Scale by 1/N applied as epilogue via `scale_dest`

3. **Query**: "What is the rsqrt_tile function in tt-metal compute kernels and how is it used?"
   **Reason**: Understanding the inverse square root operation for normalization
   **Key Findings**:
   - Performs element-wise 1/sqrt(x) on destination register tile
   - Must call `rsqrt_tile_init<LEGACY_RSQRT>()` before use
   - Used in all normalization kernels after variance + epsilon addition
   - `LEGACY_RSQRT` template parameter controls approximation method

4. **Query**: "How do circular buffers in tt-metal work? What is the relationship between capacity, block size, and double buffering?"
   **Reason**: Understanding CB sizing and buffering strategies
   **Key Findings**:
   - Capacity = total_size / page_size (number of pages)
   - Double buffering: capacity >= 2 * block_size enables overlap
   - `cb_reserve_back`/`cb_push_back` for producer, `cb_wait_front`/`cb_pop_front` for consumer
   - Page size typically set to tile size for efficient NoC transfers

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h`
   **Reason**: Understanding row_wise_mean implementation
   **Key Information**:
   - `row_wise_mean` uses `row_wise_accumulate_with_epilogue` internally
   - Epilogue applies 1/N scaling via `scale_dest`
   - Supports policies for CB sync behavior (FullBlock vs PartialBlock, WithPop vs WithoutPop)

2. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/generic/blocked_range.h`
   **Reason**: Understanding block iteration utilities
   **Key Information**:
   - `BlockedRange` class manages iteration over sequences in fixed-size blocks
   - Handles partial last blocks transparently
   - `full_block_size()` returns blk for CB alignment, `size()` returns actual count

3. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding scaler tile generation
   **Key Information**:
   - Fills tile with zeros via NoC read from MEM_ZEROS_BASE
   - Sets first 8 elements in each of 4 faces to scaler value
   - Pattern creates tile where first column of each face has scaler value

4. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp`
   **Reason**: Understanding broadcast scalar tile generation
   **Key Information**:
   - `generate_bcast_col_scalar`: Creates W-broadcast tile (column-wise scalar)
   - `generate_bcast_row_scalar`: Creates H-broadcast tile (row-wise scalar)
   - Used for epsilon in layernorm (column broadcast)

5. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h`
   **Reason**: Understanding partial tile scaler generation
   **Key Information**:
   - `generate_partial_reduce_scaler` creates scaler tile with zeros in unused columns
   - Iterates over tile faces and only sets values for columns < num_cols
   - Used when W is not divisible by TILE_WIDTH (32)
