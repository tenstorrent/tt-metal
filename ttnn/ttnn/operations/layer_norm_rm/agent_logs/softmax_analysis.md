# Softmax (tt-train) Implementation Analysis

## Overview

This analysis covers the softmax operation from the tt-train codebase. The operation computes numerically-stable softmax along the last dimension (dim=3) of a 4D tiled tensor:

```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
```

The implementation was recovered from git history (commit before `cde5a994cb`) because the program factory was subsequently stubbed out. The analysis focuses on the **compute kernel structure**, CB layout for intermediates, multi-pass data reuse, scalar/constant CB setup, reduce helper parameters, and binary op broadcast patterns -- all relevant as a reference for implementing a layer_norm_rm operation with similar row-wise reduction patterns.

**Program factory path**: `tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
**Compute kernel path**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp`
**Reader kernel path**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/reader_softmax_interleaved_start_id.cpp`
**Writer kernel path**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/writer_softmax_interleaved_start_id.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | row (of tiles) |
| **Unit size** | Wt tiles (one full row of the tiled inner dimension) |
| **Total units** | NC * Ht = batch_size * num_channels * height_in_tiles |
| **Loop structure** | Outer: rows assigned to this core. Inner: blocks of `block_size` tiles across the width dimension. Three passes per row (find max, compute exp sum, compute output). |

A single work unit is one tile-row (Wt tiles wide). Each row undergoes three sequential phases: find-max, exp-sum, and normalize.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW (4D required) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (enforced via TT_FATAL) |
| **Data type** | BFLOAT16 (Float16_b, enforced via TT_FATAL) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (enforced via TT_FATAL) |
| **Data type** | BFLOAT16 (Float16_b) |

### Layout Transformations

No tilize/untilize conversions. Input and output are both in TILE_LAYOUT. The operation is purely tile-to-tile.

## Data Flow Pattern

The softmax has two modes controlled by the `EVERYTHING_FITS_IN_L1` define:

### Mode 1: Everything Fits in L1

When all Wt input tiles plus intermediates fit in L1, the reader sends the entire row once, and the compute kernel reuses the data in-place across all three phases without re-reading from DRAM.

```
DRAM -> [Reader: read Wt tiles once] -> cb_input(Wt tiles, persist for row)
  Phase 1: Compute reads cb_input to find max -> cb_max_value_before_reduction -> reduce -> cb_max_value_after_reduction
  Phase 2: Compute reads cb_input + cb_max_value_after_reduction -> exp(x-max) -> cb_exp(Wt tiles, persist) -> accumulate sum -> cb_exp_sum_before_reduction -> matmul reduce -> recip -> cb_exp_sum_after_reduction
  Phase 3: Compute reads cb_exp + cb_exp_sum_after_reduction -> multiply -> cb_output
  [Writer: drain cb_output by blocks] -> DRAM
```

### Mode 2: Does Not Fit in L1 (Streaming, 3-Pass)

When L1 is insufficient, the reader sends the row three separate times (3x DRAM bandwidth), and the compute kernel processes each pass in blocks:

```
Pass 1 (Find Max):
  DRAM -> [Reader: stream row in blocks] -> cb_input(2*block_size)
  Compute: streaming max across blocks -> cb_max_value_before_reduction -> reduce -> cb_max_value_after_reduction

Pass 2 (Exp Sum):
  DRAM -> [Reader: stream row in blocks again] -> cb_input(2*block_size)
  Compute: sub max, exp, accumulate sum -> cb_exp_sum_before_reduction -> matmul reduce + recip -> cb_exp_sum_after_reduction

Pass 3 (Normalize):
  DRAM -> [Reader: stream row in blocks again] -> cb_input(2*block_size)
  Compute: sub max, exp, multiply by 1/sum -> cb_output
  [Writer: drain cb_output by blocks] -> DRAM
```

### Reader Kernel Summary (What It Provides)

The reader kernel:
1. Generates constant tiles at startup (before the row loop):
   - **cb_mask (c_1)**: Mask tile with 1.0 for valid columns, 0.0 for padding (only if `DO_MASK_W`)
   - **cb_max_mask (c_2)**: Mask tile with 0.0 for valid columns, -inf for padding (only if `DO_MASK_W`)
   - **cb_reduction_scaler (c_3)**: Tile filled with bfloat16 1.0 (for reduce_tile scaler)
   - **cb_matmul_reduce (c_4)**: Special matmul reduction tile (1.0 in first column of even faces, 0.0 elsewhere)
2. Reads input tiles from DRAM into cb_input, either once per row (L1 mode) or three times per row (streaming mode).

### Writer Kernel Summary (What It Consumes)

The writer kernel drains cb_output (c_10) in blocks of `block_size` tiles, writing them to DRAM via TensorAccessor.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Data Format | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|--------------------|-----------|-------------|----------|----------|----------|
| c_0 | cb_input | Input tiles | Wt (L1 mode) or 2*block_size (streaming) | block_size | Single (L1 mode) or Double (streaming) | Float16_b | Reader | Compute | Row (L1) or Block (streaming) |
| c_1 | cb_mask | Width mask (1/0) | 1 | 1 | Single | Float16_b | Reader | Compute | Program |
| c_2 | cb_max_mask | Width mask (0/-inf) | 1 | 1 | Single | Float16_b | Reader | Compute | Program |
| c_3 | cb_reduction_scaler | All-ones scaler for reduce | 1 | 1 | Single | Float16_b | Reader | Compute | Program |
| c_4 | cb_mat_mul_reduce | Matmul row-reduce tile | 1 | 1 | Single | Float16_b | Reader | Compute | Program |
| c_5 | cb_max_value_before_reduction | Pre-reduction row max | 2 | 1 | Double | Float16_b | Compute | Compute | Block (within phase 1) |
| c_6 | cb_max_value_after_reduction | Post-reduction row max | 2 | 1 | Double | Float16_b | Compute | Compute | Row (persists across phases 2-3) |
| c_7 | cb_exp | Exponentiated tiles | Wt (L1 mode) or 2*block_size (streaming) | block_size | Single (L1) or Double (streaming) | Float16_b | Compute | Compute | Row (L1 mode only; unused in streaming) |
| c_8 | cb_exp_sum_before_reduction | Pre-reduction exp sum | 2 | 1 | Double | Float32 | Compute | Compute | Block (within phase 2) |
| c_9 | cb_exp_sum_after_reduction | 1/sum(exp) after reduce+recip | 2 | 1 | Double | Float32 | Compute | Compute | Row (persists into phase 3) |
| c_10 | cb_output | Final softmax output | 2*block_size | block_size | Double | Float16_b | Compute | Writer | Block |

### Key Design Observations for CB Layout

1. **Scalar/constant CBs (c_1 through c_4)** are pushed once at program start and popped once at program end. They persist for the entire kernel lifetime.
2. **Reduction intermediate CBs (c_5, c_6, c_8, c_9)** use **Float32 for sum-related** (c_8, c_9) but **Float16_b for max-related** (c_5, c_6). This is a precision choice: sums accumulate error more than max.
3. **cb_exp (c_7)** is only meaningful in L1-fits mode, where it stores all Wt exponentiated tiles for reuse in phase 3. In streaming mode, the exp is recomputed in phase 3.
4. **Double-buffering (capacity=2)** on reduction intermediates allows overlap between compute phases even though only 1 tile is used per row.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Classification |
|----|----------|------------|-------|----------------|
| c_0 (input) | 2*block_size or Wt | block_size | 2x or Wt/bs | Double-buffered (streaming) or Multi-buffered (L1) |
| c_1 (mask) | 1 | 1 | 1x | Single-buffered (persistent) |
| c_2 (max_mask) | 1 | 1 | 1x | Single-buffered (persistent) |
| c_3 (scaler) | 1 | 1 | 1x | Single-buffered (persistent) |
| c_4 (matmul) | 1 | 1 | 1x | Single-buffered (persistent) |
| c_5 (max pre-red) | 2 | 1 | 2x | Double-buffered |
| c_6 (max post-red) | 2 | 1 | 2x | Double-buffered |
| c_7 (exp) | Wt or 2*bs | block_size | varies | Multi or Double |
| c_8 (sum pre-red) | 2 | 1 | 2x | Double-buffered |
| c_9 (sum post-red) | 2 | 1 | 2x | Double-buffered |
| c_10 (output) | 2*block_size | block_size | 2x | Double-buffered |

## Multi-Pass Data Reuse Patterns

This is the most critical aspect for the layer_norm_rm reference.

### Which CBs Persist Across Phases and Why

| CB | Persists Across Phases? | Reason |
|----|-------------------------|--------|
| **c_0 (input)** | YES in L1 mode, NO in streaming | In L1 mode, all Wt tiles remain available for all 3 phases. In streaming mode, tiles are consumed per-block and re-read from DRAM. |
| **c_6 (max_after_reduction)** | YES: phases 2 and 3 | The reduced max value (single tile, column vector) is needed to subtract from input in both phase 2 (exp-sum) and phase 3 (normalize). Popped only at end of row. |
| **c_7 (exp)** | YES in L1 mode only: phases 2 to 3 | In L1 mode, exp tiles are computed in phase 2 and reused in phase 3 for multiplication. In streaming mode, exp is recomputed. |
| **c_9 (exp_sum_after_reduction)** | YES: phase 2 to 3 | The reciprocal of exp-sum (single tile, column vector) is computed in phase 2 and used as a multiplier in phase 3. Popped at end of row. |
| **c_1, c_2, c_3, c_4** | YES: entire program | Constant tiles generated once, used across all rows and all phases. |

### Why 3 Passes in Streaming Mode

Softmax requires three dependent reductions:
1. **max(x)** across the row -- needs to see all elements before any can be processed
2. **sum(exp(x - max))** across the row -- depends on result of step 1
3. **exp(x - max) / sum** -- depends on results of steps 1 and 2

When the full row does not fit in L1, each pass must re-read the input from DRAM. This is a fundamental limitation of the softmax algorithm's data dependencies.

### Implication for Layer Norm

Layer norm has the same pattern:
1. **mean(x)** = sum(x) / N -- needs full row
2. **var(x)** = sum((x - mean)^2) / N -- depends on mean
3. **normalize**: (x - mean) * rsqrt(var + eps) -- depends on mean and variance

This maps to the same 3-pass streaming pattern or single-pass L1 pattern.

## Compute Kernel Structure -- Detailed Analysis

### Initialization Sequence

```cpp
// At kernel_main() entry:
if constexpr (do_mask_w) {
    cb_wait_front(cb_mask, onetile);       // Wait for reader to push mask tiles
    cb_wait_front(cb_max_mask, onetile);
}
cb_wait_front(cb_reduction_scaler, onetile);  // Wait for reduction scaler

init_sfpu(cb_input, cb_output);                    // Initialize SFPU hardware
binary_op_init_common(cb_input, cb_input, cb_output);  // Initialize binary op hardware
```

`init_sfpu` and `binary_op_init_common` must be called exactly once at kernel startup. They configure the SFPU, FPU math engine, unpacker, and packer hardware via MMIO writes. All subsequent operation-specific `*_init()` calls are lightweight reconfigurations.

### Phase 1: find_max_value_in_row()

**Purpose**: Find the maximum value across all Wt tiles in the row. Result is one tile in cb_max_value_before_reduction.

**Key API calls with exact signatures**:

```cpp
// Tile register management
tile_regs_acquire();     // Acquire DST registers for math+unpack

// Data movement from CB to DST register
copy_tile_init(cb_input);                    // Lightweight init for copy
copy_tile(cb_input, col, working_register);  // Copy tile at index `col` from cb_input to DST[working_register]

// Masking (only for last tile if DO_MASK_W)
copy_tile_init(cb_mask);
copy_tile(cb_mask, 0, mask_register);     // Load mask into adjacent register
mask_tile_init();
mask_tile(working_register, mask_register);  // Zero out padding elements

copy_tile_init(cb_max_mask);
copy_tile(cb_max_mask, 0, mask_register);    // Load -inf mask
add_binary_tile_init();
add_binary_tile(working_register, mask_register, working_register);  // Add -inf to padding positions

// Element-wise max accumulation
binary_max_tile_init();
binary_max_tile(max_value_register, tile_register, max_value_register);  // DST[0] = max(DST[0], DST[1])

// Commit results to packer
tile_regs_commit();

// Pack result
tile_regs_wait();
pack_reconfig_data_format(cb_max_value_before_reduction);
pack_tile(max_value_register, cb_max_value_before_reduction);  // Pack DST[0] to CB
tile_regs_release();
cb_push_back(cb_max_value_before_reduction, onetile);
```

**Register usage**: Two registers used -- DST[0] for accumulating max, DST[1] for current tile. Mask uses register adjacent to data register (hardware constraint: `mask_register = working_register + 1`).

**L1 vs Streaming difference**: In L1 mode, `cb_wait_front(cb_input, col + block_size)` -- waits for cumulative tiles (no pop). In streaming mode, `cb_wait_front(cb_input, block_size)` then `cb_pop_front(cb_input, block_size)` -- processes and releases blocks.

**Masking pattern for partial tiles**: The masking is two-step:
1. `mask_tile` zeroes out padding elements in the data
2. `add_binary_tile` with the max_mask (-inf values) ensures padding positions have -inf, so they never win the max comparison

### reduce_max_value()

**Purpose**: Reduce the 32x32 tile in cb_max_value_before_reduction to a column vector (max within each row of the tile).

**Key API calls**:

```cpp
cb_wait_front(cb_max_value_before_reduction, onetile);
cb_reserve_back(cb_max_value_after_reduction, onetile);

tile_regs_acquire();
reconfig_data_format(cb_max_value_before_reduction, cb_reduction_scaler);

reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_max_value_before_reduction, cb_reduction_scaler, cb_max_value_after_reduction);
reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_max_value_before_reduction, cb_reduction_scaler, 0, 0, reduction_register);
reduce_uninit();

tile_regs_commit();
tile_regs_wait();
pack_reconfig_data_format(cb_max_value_after_reduction);
pack_tile(reduction_register, cb_max_value_after_reduction);
tile_regs_release();

cb_push_back(cb_max_value_after_reduction, onetile);
cb_pop_front(cb_max_value_before_reduction, onetile);
```

**reduce API pattern (3-phase)**:
1. `reduce_init<PoolType, ReduceDim>(input_cb, scaler_cb, output_cb)` -- configures unpack, math, and pack units
2. `reduce_tile<PoolType, ReduceDim>(input_cb, scaler_cb, tile_idx, scaler_idx, dst_register)` -- performs the reduction into DST
3. `reduce_uninit()` -- resets packer edge masks (required before any non-reduce operation)

**REDUCE_ROW** reduces across columns (width), producing a column vector in the output tile. The scaler CB (cb_reduction_scaler) contains all-ones for MAX/SUM operations.

**Critical note**: `reconfig_data_format` is called before `reduce_init` to configure the unpacker for the correct input formats. `pack_reconfig_data_format` is called before `pack_tile` to configure the packer for the output format.

### Phase 2: calculate_sum_exp_x()

**Purpose**: Compute sum(exp(x - max(x))) across the row.

#### L1 Mode

```cpp
cb_wait_front(cb_max_value_after_reduction, onetile);
cb_reserve_back(cb_exp, Wt);  // Reserve ALL Wt tiles in cb_exp for later reuse

// Block loop: subtract max via broadcast, exp, optional mask
reconfig_data_format(cb_input, cb_max_value_after_reduction);
sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
sub_tiles_bcast<BroadcastType::COL>(
    cb_input, cb_max_value_after_reduction, col, 0, block_idx);

exp_tile_init<false>();           // false = not approximate
exp_tile<false>(block_idx);       // exp in-place on DST[block_idx]

// If last tile: mask padding
mask_tile(block_idx, mask_register);

// Pack exp tiles to cb_exp
pack_reconfig_data_format(cb_exp);
pack_tile(block_idx, cb_exp);
cb_push_back(cb_exp, block_size);

// After all blocks: accumulate sum
cb_wait_front(cb_exp, Wt);
cb_reserve_back(cb_exp_sum_before_reduction, onetile);

// Register accumulation of exp tiles
copy_tile(cb_exp, col, tile_register);
add_binary_tile(working_register, tile_register, working_register);

// Pack accumulated sum
pack_tile(working_register, cb_exp_sum_before_reduction);
cb_push_back(cb_exp_sum_before_reduction, onetile);
```

**Key insight**: In L1 mode, exp tiles are packed to cb_exp AND the sum is accumulated separately. The exp tiles persist in cb_exp for reuse in phase 3, avoiding recomputation.

#### Streaming Mode

```cpp
cb_wait_front(cb_max_value_after_reduction, onetile);
cb_reserve_back(cb_exp_sum_before_reduction, onetile);

// Load max into a register ONCE using unary_bcast
unary_bcast_init<BroadcastType::COL>(cb_max_value_after_reduction, cb_max_value_after_reduction);
unary_bcast<BroadcastType::COL>(cb_max_value_after_reduction, 0, max_value_register);

// Stream blocks: for each block
cb_wait_front(cb_input, block_size);
copy_tile(cb_input, block_idx, working_register);
sub_binary_tile_init();
sub_binary_tile(working_register, max_value_register, working_register);
exp_tile<false>(working_register);
// mask if needed
add_binary_tile(accum_register, working_register, accum_register);
cb_pop_front(cb_input, block_size);

// Pack sum
pack_tile(accum_register, cb_exp_sum_before_reduction);
```

**Binary vs unary broadcast difference**:
- `sub_tiles_bcast<BroadcastType::COL>` (binary): Reads from two CBs. Used in L1 mode -- subtracts cb_max_value_after_reduction (column vector) from cb_input tile.
- `unary_bcast<BroadcastType::COL>`: Reads from one CB, broadcasts column vector to fill a full tile in a DST register. Used in streaming mode to load max value once, then `sub_binary_tile` operates register-to-register.

**Why the difference**: In streaming mode, the max value must be kept in a DST register across the entire block loop (the CB cannot be re-read per block). The unary_bcast loads it once into DST[max_value_register], and then `sub_binary_tile` does register-register subtraction.

### reduce_sum_exp_x()

**Purpose**: Reduce the accumulated exp-sum tile to a column vector, then take reciprocal.

```cpp
cb_wait_front(cb_exp_sum_before_reduction, onetile);
cb_reserve_back(cb_exp_sum_after_reduction, onetile);
cb_wait_front(cb_mat_mul_reduce, onetile);

tile_regs_acquire();

// IMPORTANT: Uses matmul instead of reduce_tile for precision
mm_init(cb_exp_sum_before_reduction, cb_mat_mul_reduce, cb_exp_sum_after_reduction, 0);
matmul_tiles(cb_exp_sum_before_reduction, cb_mat_mul_reduce, 0, 0, reduction_register);

recip_tile_init();
recip_tile(reduction_register);  // DST[0] = 1/sum(exp(x))

tile_regs_commit();
tile_regs_wait();
pack_reconfig_data_format(cb_exp_sum_after_reduction);
pack_tile(reduction_register, cb_exp_sum_after_reduction);
tile_regs_release();

cb_push_back(cb_exp_sum_after_reduction, onetile);
cb_pop_front(cb_exp_sum_before_reduction, onetile);
```

**Why matmul instead of reduce_tile**: The code has a commented-out `reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>` version. The comment explains: "We used matmul_tiles instead of reduce_tile, because reduce_tile causes a loss of precision." This is a known issue also observed in moreh's ops. The matmul path uses the full-precision matrix multiplication hardware.

**matmul_tiles row reduction pattern**:
1. `mm_init(input_cb, reduce_cb, output_cb, transpose_flag)` -- initialize matmul hardware
2. `matmul_tiles(input_cb, reduce_cb, input_tile_idx, reduce_tile_idx, dst_register)` -- multiply input tile by the special reduction tile
3. The reduction tile (cb_mat_mul_reduce) has 1.0 in the first column of even faces and 0.0 elsewhere, so the matmul effectively sums each row of the input tile into a single column.

**recip_tile**: Computes 1/x element-wise on the DST register. After this, DST[0] contains 1/sum(exp(x)) as a column vector.

**Note**: cb_exp_sum_before_reduction and cb_exp_sum_after_reduction use **Float32** data format for precision during the sum reduction.

### Phase 3: Normalize (in kernel_main loop)

```cpp
cb_wait_front(cb_exp_sum_after_reduction, onetile);  // 1/sum persists for full row

for (uint32_t col = 0; col < Wt; col += block_size) {
    // Load 1/sum into a register via unary_bcast
    unary_bcast_init<BroadcastType::COL>(cb_exp_sum_after_reduction, cb_exp_sum_after_reduction);
    unary_bcast<BroadcastType::COL>(cb_exp_sum_after_reduction, 0, sum_exp_register);

    // L1 mode: read from cb_exp (pre-computed)
    copy_tile(cb_exp, col + block_idx, block_idx);

    // Streaming mode: recompute exp(x - max)
    sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max_value_after_reduction, block_idx, 0, block_idx);
    exp_tile<false>(block_idx);

    // Multiply by 1/sum
    mul_binary_tile_init();
    mul_binary_tile(block_idx, sum_exp_register, block_idx);

    // Pack to output
    pack_tile(block_idx, cb_output);
    cb_push_back(cb_output, block_size);
}

// Cleanup: pop row-scoped intermediates
cb_pop_front(cb_max_value_after_reduction, onetile);
cb_pop_front(cb_exp_sum_after_reduction, onetile);
// L1 mode only:
cb_pop_front(cb_input, Wt);
cb_pop_front(cb_exp, Wt);
```

**Key detail**: `unary_bcast<BroadcastType::COL>` is called inside the block loop to reload the 1/sum column vector into a DST register. This is because tile_regs_acquire/commit/release cycles per block clear the registers.

## Index Calculations

Tile indices are linear within the padded tensor volume:
- `idx = (start_row + row_within_core) * Wt + col`
- `start_row` is the pre-computed cumulative row offset for this core
- `Wt = padded_shape[-1] / TILE_WIDTH` (tiles in inner dimension)
- Total rows = `NC * Ht = (N * C) * (padded_shape[-2] / TILE_HEIGHT)`

The TensorAccessor handles the mapping from linear tile index to physical DRAM bank address.

## Memory Access Patterns

### Read Pattern
- **Sequential** tile reads within a row (contiguous tiles)
- Block-granularity: reads `block_size` tiles at a time
- In streaming mode: 3x read amplification (same row read 3 times)
- In L1 mode: 1x read (entire row read once, reused from L1)

### Write Pattern
- **Sequential** tile writes within a row
- Block-granularity: writes `block_size` tiles at a time
- Only one pass for writes (output produced in phase 3 only)

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration) |
| **Grid dimensions** | compute_with_storage_grid_size (device-dependent) |
| **Total cores** | min(total_rows, grid_x * grid_y) |
| **Work per core** | num_rows_per_core_group_1 or num_rows_per_core_group_2 rows |
| **Load balancing** | Two-group: group_1 gets ceil(rows/cores), group_2 gets floor(rows/cores) |

Core indexing: `core = {i / num_cores_y, i % num_cores_y}` -- column-major traversal.

Two compute kernel handles are created (compute_group_1 and compute_group_2), differing only in the `num_rows_per_core` compile-time argument.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Number of tile-rows this core processes (differs between group 1 and group 2) |
| 1 | block_size | uint32_t | Tiles processed per inner loop iteration (1-3, chosen by get_block_size) |
| 2 | Wt | uint32_t | Number of tiles in inner (width) dimension |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Block size for CB reserve/push |
| 1 | Wt | uint32_t | Tiles in inner dimension |
| 2 | mask_w | uint32_t | Width index of first padding element (0 = no masking) |
| 3+ | TensorAccessorArgs | uint32_t[] | Input buffer accessor args |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Block size for CB wait/pop |
| 1 | Wt | uint32_t | Tiles in inner dimension |
| 2+ | TensorAccessorArgs | uint32_t[] | Output buffer accessor args |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_address | uint32_t | Input buffer DRAM address |
| 1 | num_rows_to_process | uint32_t | Rows assigned to this core |
| 2 | start_row | uint32_t | Cumulative row offset (pre-computed) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_address | uint32_t | Output buffer DRAM address |
| 1 | num_rows_to_process | uint32_t | Rows assigned to this core |
| 2 | start_row | uint32_t | Cumulative row offset |

### Preprocessor Defines (Conditional Compilation)

| Define | Condition | Effect |
|--------|-----------|--------|
| `DO_MASK_W` | mask_w != 0 | Enables width masking for partial last tile |
| `EVERYTHING_FITS_IN_L1` | required_L1 <= available_L1 | Enables single-read L1 reuse mode |
| `REDUCE_OP` | Always "PoolType::SUM" | Required by LLK reduce (default template param) |
| `REDUCE_DIM` | Always "ReduceDim::REDUCE_ROW" | Required by LLK reduce (default template param) |

## Kernel Implementations

### Compute Kernel

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| softmax_kernel.cpp | Compute (RISCV T0-T2) | N/A | c_0 (input), c_1 (mask), c_2 (max_mask), c_3 (scaler), c_4 (matmul) | c_5 (max pre), c_6 (max post), c_7 (exp), c_8 (sum pre), c_9 (sum post), c_10 (output) | copy_tile, mask_tile, binary_max_tile, reduce_tile (MAX), sub_tiles_bcast (COL), exp_tile, add_binary_tile, matmul_tiles, recip_tile, mul_binary_tile, unary_bcast (COL), pack_tile |

**File**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp`

**Key Logic**:
- Three-phase row processing: find_max, exp_sum, normalize
- Two code paths via `#ifdef EVERYTHING_FITS_IN_L1`
- Uses `tile_regs_acquire/commit/wait/release` for DST register synchronization (not the deprecated `acquire_dst/release_dst`)
- Block processing with `block_size` tiles per inner iteration
- `reconfig_data_format` and `pack_reconfig_data_format` called when switching between Float16_b and Float32 CBs
- `fp32_dest_acc_en=true` in ComputeConfig for precision

### Reader Kernel

| Kernel | Core | NOC | Input | Output CBs | Operations |
|--------|------|-----|-------|------------|------------|
| reader_softmax_interleaved_start_id.cpp | RISCV_0 | NOC0 | DRAM | c_0, c_1, c_2, c_3, c_4 | generate_mask_tile, generate_tile_with_bfloat16_value, generate_matmul_row_reduce_tile, read_full_row_tiles |

### Writer Kernel

| Kernel | Core | NOC | Input CBs | Output | Operations |
|--------|------|-----|-----------|--------|------------|
| writer_softmax_interleaved_start_id.cpp | RISCV_1 | NOC1 | c_10 | DRAM | write_full_row_tiles |

## Scalar/Constant CB Setup Details

The reader kernel generates these constant tiles at startup (before the row loop):

### cb_reduction_scaler (c_3): All-Ones Tile
```cpp
constexpr uint16_t one = 0x00003F80;  // bfloat16 1.0
generate_tile_with_bfloat16_value(cb_reduction_scaler_idx, one);
```
Used as the scaler argument in `reduce_tile`. For PoolType::MAX and PoolType::SUM, a scaler of 1.0 means no scaling.

### cb_mat_mul_reduce (c_4): Matmul Row-Reduce Tile
```cpp
generate_matmul_row_reduce_tile(cb_matmul_reduce);
```
This generates a tile with 1.0 in the first column of even faces (left faces) and 0.0 elsewhere. When multiplied with an input tile via matmul, this effectively sums each row into the first column. The data format is auto-detected from the CB.

### cb_mask (c_1): Width Mask (1/0)
```cpp
generate_mask_tile(cb_mask_idx, /*fill=*/one, /*mask_fill=*/zero, mask_w);
```
Tile with 1.0 for columns 0..mask_w-1 and 0.0 for columns mask_w..31. Used with `mask_tile` to zero out padding.

### cb_max_mask (c_2): Width Mask (0/-inf)
```cpp
generate_mask_tile(cb_max_mask_idx, /*fill=*/zero, /*mask_fill=*/minus_inf, mask_w);
```
Tile with 0.0 for valid columns and -inf (0xFF80 in bfloat16) for padding columns. Added to data after mask_tile zeroing so padding positions become -inf (losing max comparison).

## Block Size Calculation

```cpp
uint32_t block_size = get_block_size(Wt, 3U);  // max_block_size = 3
```

The `get_block_size` function finds the largest divisor of Wt that is <= max_block_size. The max is 3 (not 4) because the compute kernel needs "one extra register during calculation" -- specifically, the mask register must be adjacent to the data register, consuming register slots.

## L1 Memory Budget Calculation

The program factory computes whether everything fits in L1:

```cpp
required_L1 = 2 * (Wt * bf16_tile_size)          // input + exp (both Wt tiles)
            + 2 * mask_tile_size                    // mask + max_mask
            + 2 * scaler_tile_size                  // scaler + matmul
            + 2 * block_size * bf16_tile_size       // output (double buffered)
            + (2+2) * bf16_tile_size                // max before/after reduction
            + (2+2) * f32_tile_size                 // exp sum before/after reduction
```

If `required_L1 <= available_L1`, the `EVERYTHING_FITS_IN_L1` define is set.

## Implementation Notes

### Numerical Stability
- The three-phase algorithm (max, exp-sum, normalize) is the standard numerically-stable softmax
- Masking with -inf ensures padding never affects the max computation
- Float32 accumulators for exp-sum (c_8, c_9) and fp32_dest_acc_en prevent precision loss during summation
- matmul_tiles used instead of reduce_tile for sum reduction due to known precision bug in reduce_tile

### ComputeConfig
```cpp
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = true,
    .math_approx_mode = false,
    .compile_args = ...,
    .defines = ...
}
```
HiFi4 + fp32_dest_acc + no approximation = maximum precision configuration.

### Register Allocation Strategy
The compute kernel uses a small number of DST registers:
- Phase 1: registers 0 (max accumulator), 1 (current tile), 2 (mask, when needed)
- Phase 2 L1: registers 0..block_size-1 (exp tiles), block_size (sum accumulator)
- Phase 2 streaming: registers 0 (sum accumulator), 1 (current tile), 2 (max value via unary_bcast)
- Phase 3: registers 0..block_size-1 (output tiles), block_size (1/sum via unary_bcast)

### Key Difference: L1 Mode vs Streaming Mode Data Flow

| Aspect | L1 Mode | Streaming Mode |
|--------|---------|----------------|
| DRAM reads per row | 1x | 3x |
| cb_input capacity | Wt tiles | 2*block_size tiles |
| cb_exp used? | Yes (stores all Wt exp tiles) | No (exp recomputed in phase 3) |
| Phase 2 subtract | Binary bcast from CB | Unary bcast to register + register-register sub |
| Phase 3 exp source | cb_exp (pre-computed) | Recompute from cb_input |
| cb_input pop | End of row | End of each block |

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do reduce_init, reduce_tile, and reduce_uninit work in compute kernels?"
   **Reason**: Understanding the 3-phase reduce API pattern used in reduce_max_value()
   **Key Findings**: reduce_init configures unpack/math/pack units. reduce_tile performs hardware reduction into DST. reduce_uninit resets packer edge masks (required before non-reduce ops). Scaler CB is essential -- 1.0 for SUM/MAX, 1/N for AVG.

2. **Query**: "How do sub_bcast_cols_init_short, sub_tiles_bcast with BroadcastType::COL, and unary_bcast work?"
   **Reason**: Understanding the two different broadcast subtract patterns in L1 vs streaming mode
   **Key Findings**: Binary bcast (sub_tiles_bcast) operates on two CBs. Unary bcast loads a single tile into DST with broadcast. COL broadcast replicates a column vector across all columns. sub_bcast_cols_init_short is a lightweight reconfiguration.

3. **Query**: "How do mask_tile, binary_max_tile work? What is the adjacency constraint?"
   **Reason**: Understanding the masking pattern and why max_block_size is 3 (needs spare register)
   **Key Findings**: mask_tile zeroes elements based on mask. The mask register MUST be adjacent (register_idx + 1) to the data register -- hardware limitation. binary_max_tile computes element-wise max of two DST registers.

4. **Query**: "How do tile_regs_acquire/commit/wait/release work?"
   **Reason**: Understanding the DST register synchronization model
   **Key Findings**: 4-phase sync: acquire (math gets DST), commit (math done, transfers to packer), wait (packer ready), release (DST available for next iteration). This is the modern replacement for deprecated acquire_dst/release_dst.

5. **Query**: "How does matmul_tiles work as a replacement for reduce_tile?"
   **Reason**: Understanding why the sum reduction uses matmul instead of reduce
   **Key Findings**: reduce_tile has a known precision bug. matmul_tiles with a special reduction tile (1.0 in first column of even faces) achieves the same row-sum but through the full-precision matmul hardware path. Pattern: mm_init + matmul_tiles.

6. **Query**: "How does split_work_to_cores work?"
   **Reason**: Understanding the two-group core distribution strategy
   **Key Findings**: Returns (num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2). Group 1 gets ceil(work/cores), group 2 gets floor. Handles remainder elegantly.

7. **Query**: "What is pack_reconfig_data_format and reconfig_data_format?"
   **Reason**: Understanding format switching between Float16_b and Float32 CBs
   **Key Findings**: reconfig_data_format reconfigures the unpacker for different input formats. pack_reconfig_data_format reconfigures the packer. Must be called when switching between CBs of different data formats.

8. **Query**: "What are init_sfpu and binary_op_init_common?"
   **Reason**: Understanding kernel startup initialization requirements
   **Key Findings**: init_sfpu initializes SFPU hardware for unary ops. binary_op_init_common initializes FPU for binary ops. Both must be called exactly once at kernel startup via MMIO writes.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding tile_regs synchronization and CB producer/consumer model
   **Key Information**: Confirmed the 4-phase tile_regs pattern (acquire->commit->wait->release) and that CBs are the synchronization mechanism between reader/compute/writer kernels.

2. **Source**: `tt-train/sources/ttml/metal/common/program_utils.hpp`
   **Reason**: Understanding host-side helper functions for CB creation, kernel creation, block size calculation
   **Key Information**: `get_block_size(Wt, max)` finds largest divisor <= max. `create_compute_kernel` wraps CreateKernel with ComputeConfig. `create_circular_buffer` wraps CircularBufferConfig.

3. **Source**: `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
   **Reason**: Understanding reader/writer utility functions and constant tile generation
   **Key Information**: `generate_mask_tile` fills a tile with per-column masking. `generate_matmul_row_reduce_tile` creates the special matmul reduction pattern. `generate_tile_with_bfloat16_value` fills all 1024 elements. `read_full_row_tiles` / `write_full_row_tiles` handle block-wise row I/O.
