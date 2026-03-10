# Softmax Implementation Analysis (Compute Core Focus)

## Overview

This analysis covers the softmax operation implemented in `tt-train/sources/ttml/metal/ops/softmax/device/`. The operation computes row-wise softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))` on 4D tensors with TILE layout, Float16_b data type, and DRAM-interleaved memory. It is analyzed here as a **compute-core reference** for implementing a new `layer_norm_rm` operation, which shares the same structural pattern of multi-pass row-wise reduction followed by element-wise transformation.

**Program Factory**: `tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`

**Structural Parallel to Layer Norm**:
| Softmax Phase | Layer Norm Phase |
|---|---|
| Row-wise max | Row-wise mean |
| Subtract max | Subtract mean |
| Exp | Square |
| Row-wise sum | Row-wise variance mean |
| Reciprocal | Add epsilon, rsqrt |
| Multiply (scale by 1/sum) | Multiply (scale by rsqrt(var+eps)) |
| --- | Optionally multiply gamma, add beta |

## Work Unit Definition

| Attribute | Value |
|---|---|
| **Granularity** | Row of tiles |
| **Unit size** | `Wt` tiles (one full row in the tile-width dimension) |
| **Total units** | `NC * Ht` = (batch * channels) * height-in-tiles |
| **Loop structure** | Outer loop: rows assigned to this core. Inner: passes over the row |

One "work unit" is processing a single row of `Wt` tiles through the full softmax pipeline (find-max, reduce-max, compute-sum-exp, reduce-sum-exp, final-scale). The row is the reduction dimension (last dim, dim=3).

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|---|---|---|
| **Logical shape** | [N, C, H, W] (4D required) | [N, C, H, W] |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (enforced by TT_FATAL) | DRAM (enforced by TT_FATAL) |
| **Data type** | Float16_b (enforced by TT_FATAL) | Float16_b |

### Key Derived Dimensions
- `Wt = padded_shape[-1] / TILE_WIDTH` -- tiles in the reduction (inner) dimension
- `Ht = padded_shape[-2] / TILE_HEIGHT` -- tiles in height dimension
- `NC = padded_shape[0] * padded_shape[1]` -- batch * channels
- `total_rows_to_process = NC * Ht` -- total tile-rows across all batch/channel/height
- `mask_w = logical_shape[-1] % TILE_WIDTH` -- nonzero if last tile has padding

### Layout Transformations
No tilize/untilize or reshard is performed. Input and output share identical layout.

## Data Flow Pattern

The operation has **two modes** controlled by `EVERYTHING_FITS_IN_L1`:

### Mode 1: Everything Fits in L1 (`EVERYTHING_FITS_IN_L1` defined)
The entire row of `Wt` input tiles is read once and persists in L1 across all phases.

| Stage | Description | Input CB | Output CB |
|---|---|---|---|
| 1 | Reader reads entire row (Wt tiles) into `cb_input` | DRAM | cb_input (c_0) |
| 2 | **Phase 1 - Find Max**: Scan all Wt tiles in cb_input, tile-wise max | cb_input | cb_max_before (c_5) |
| 3 | **Phase 2 - Reduce Max**: Row-reduce the max tile to column vector | cb_max_before (c_5) | cb_max_after (c_6) |
| 4 | **Phase 3 - Exp(x-max)**: Subtract max (broadcast COL), exp, mask, accumulate sum. Produces exp tiles into cb_exp AND accumulates into cb_exp_sum_before | cb_input (c_0), cb_max_after (c_6) | cb_exp (c_7), cb_exp_sum_before (c_8) |
| 5 | **Phase 4 - Reduce Sum & Reciprocal**: Row-reduce sum tile via matmul, take reciprocal | cb_exp_sum_before (c_8), cb_mat_mul (c_4) | cb_exp_sum_after (c_9) |
| 6 | **Phase 5 - Final Scale**: Read exp tiles from cb_exp, multiply by 1/sum (broadcast COL) | cb_exp (c_7), cb_exp_sum_after (c_9) | cb_output (c_10) |
| 7 | Writer drains cb_output to DRAM | cb_output (c_10) | DRAM |

**Key**: Input data is read from DRAM **once**. The exp values are stored in `cb_exp` and reused in Phase 5.

### Mode 2: Streaming (`EVERYTHING_FITS_IN_L1` NOT defined)
The row is too large for L1. Reader streams the row **3 times** from DRAM.

| Pass | Purpose | Reads from DRAM? |
|---|---|---|
| Pass 1 | Find max value in row (Phase 1) | Yes, full row |
| Pass 2 | Compute sum(exp(x - max)) (Phase 3) | Yes, full row again |
| Pass 3 | Compute final output: exp(x-max) * (1/sum) (Phase 5) | Yes, full row again |

In streaming mode, `cb_input` capacity is `2 * block_size` tiles (double-buffered), and tiles are consumed (popped) after each block is processed. There is no `cb_exp` persistence.

### Reader Kernel Summary (Not Deep-Dived)
- Generates constant tiles on first call: mask tile (`cb_mask`, c_1), max-mask tile (`cb_max_mask`, c_2), reduction scaler tile (`cb_reduction_scaler`, c_3), matmul reduce tile (`cb_mat_mul_reduce`, c_4)
- Reads input row tiles in blocks of `block_size` using `read_full_row_tiles`
- In streaming mode, reads the row 3 times per row; in L1 mode, reads once

### Writer Kernel Summary (Not Deep-Dived)
- Drains `cb_output` (c_10) in blocks of `block_size` using `write_full_row_tiles`
- One pass per row

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Data Format | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_input | Input staging from DRAM | Wt (L1 mode) or 2*block_size (stream) | Float16_b | Single (L1) / Double (stream) | Reader | Compute | Row (L1) / Block (stream) |
| c_1 | cb_mask | Width mask (1.0/0.0) | 1 | Float16_b | Single | Reader (generated once) | Compute | Program |
| c_2 | cb_max_mask | Max-padding mask (0.0/-inf) | 1 | Float16_b | Single | Reader (generated once) | Compute | Program |
| c_3 | cb_reduction_scaler | All-1.0 tile for reduce | 1 | Float16_b | Single | Reader (generated once) | Compute | Program |
| c_4 | cb_mat_mul_reduce | Row-reduce via matmul tile | 1 | Float16_b | Single | Reader (generated once) | Compute | Program |
| c_5 | cb_max_before_reduction | Pre-reduce max tile | 2 | Float16_b | Double | Compute (Phase 1) | Compute (Phase 2) | Row |
| c_6 | cb_max_after_reduction | Post-reduce max (col vector) | 2 | Float16_b | Double | Compute (Phase 2) | Compute (Phase 3, 5) | Row |
| c_7 | cb_exp | Exp(x-max) tiles | Wt (L1 mode) or 2*block_size (stream) | Float16_b | Matches cb_input | Compute (Phase 3) | Compute (Phase 5) | Row (L1 only) |
| c_8 | cb_exp_sum_before | Pre-reduce sum(exp) tile | 2 | Float32 | Double | Compute (Phase 3) | Compute (Phase 4) | Row |
| c_9 | cb_exp_sum_after | Post-reduce 1/sum(exp) (col vector) | 2 | Float32 | Double | Compute (Phase 4) | Compute (Phase 5) | Row |
| c_10 | cb_output | Output staging to DRAM | 2*block_size | Float16_b | Double | Compute (Phase 5) | Writer | Block |

### Critical CB Details for Layer Norm Adaptation

1. **Float32 CBs for Precision**: `cb_exp_sum_before` (c_8) and `cb_exp_sum_after` (c_9) use `Float32` data format for accumulation precision. Layer norm should similarly use Float32 for variance accumulation.

2. **Scalar/Constant CBs (c_1 through c_4)** persist for the entire program lifetime. They are generated once by the reader kernel and consumed repeatedly by compute. They are never popped until the very end of `kernel_main()`.

3. **The matmul-reduce tile (c_4)** has a specific pattern: 1.0 in the first column of left faces, 0.0 elsewhere. This implements row-sum via matrix multiplication, which is more precise than the built-in `reduce_tile<PoolType::SUM>`.

4. **Intermediate reduction CBs (c_5, c_6, c_8, c_9)** have capacity 2 tiles. This is for double-buffering between the produce phase and consume phase within the same compute kernel (not between different kernels). They are pushed/popped within a single row's processing.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Pattern |
|---|---|---|---|
| c_0 (input) | Wt or 2*BS | Wt or BS | Single (L1) / Double (stream) |
| c_1 (mask) | 1 | 1 | Single, persistent |
| c_2 (max_mask) | 1 | 1 | Single, persistent |
| c_3 (scaler) | 1 | 1 | Single, persistent |
| c_4 (matmul) | 1 | 1 | Single, persistent |
| c_5 (max pre) | 2 | 1 | Double |
| c_6 (max post) | 2 | 1 | Double |
| c_7 (exp) | Wt or 2*BS | BS | Single (L1) / Double (stream) |
| c_8 (sum pre) | 2 | 1 | Double |
| c_9 (sum post) | 2 | 1 | Double |
| c_10 (output) | 2*BS | BS | Double |

**BS** = `block_size` = result of `get_block_size(Wt, 3)`, which finds the largest divisor of Wt in {3, 2, 1}. The maximum is 3 (not 4) because the compute kernel needs spare DST registers for intermediate values.

## Multi-Pass Data Reuse Patterns (Key for Layer Norm)

### Which CBs Persist Across Phases and Why

**Program-lifetime CBs (persist across ALL rows)**:
- `cb_mask` (c_1): Width mask, same for every tile at the end of every row
- `cb_max_mask` (c_2): -inf padding mask, same for every row
- `cb_reduction_scaler` (c_3): All-1.0 tile, used in reduce_max
- `cb_mat_mul_reduce` (c_4): Row-reduce pattern tile, used in reduce_sum_exp

These are pushed once by the reader, waited on once by compute at kernel start, and only popped at the very end of the compute kernel. This avoids regenerating them per row.

**Row-lifetime CBs (persist across phases within one row, popped at row end)**:
- `cb_max_after_reduction` (c_6): The reduced max column-vector. Produced in Phase 2, consumed in Phase 3 (L1 mode: broadcast subtract) and Phase 5 (streaming mode: recompute exp). Popped at end of row.
- `cb_exp_sum_after_reduction` (c_9): The 1/sum(exp) column-vector. Produced in Phase 4, consumed in Phase 5 (multiply by reciprocal). Popped at end of row.
- In L1 mode: `cb_input` (c_0) and `cb_exp` (c_7) hold the full row of Wt tiles across all phases, popped at row end.

**Block-lifetime CBs (produced and consumed within one block iteration)**:
- `cb_max_before_reduction` (c_5): Produced in Phase 1, consumed in Phase 2, then freed.
- `cb_exp_sum_before_reduction` (c_8): Produced in Phase 3, consumed in Phase 4, then freed.
- `cb_output` (c_10): Produced in Phase 5 per block, consumed by writer.
- In streaming mode: `cb_input` (c_0) is per-block (popped after each block_size chunk).

### L1 Budget Calculation (from program factory)

```
required_L1 = 2 * (Wt * bf16_tile_size)        // input + exp (both Wt tiles)
            + 2 * mask_tiles * bf16_tile_size    // mask + max_mask
            + 2 * scaler_tiles * bf16_tile_size  // scaler + matmul
            + (2 + 2) * bf16_tile_size           // max_before + max_after
            + (2 + 2) * fp32_tile_size           // sum_before + sum_after
            + 2 * block_size * bf16_tile_size    // output
```

If `required_L1 <= available_L1`, the `EVERYTHING_FITS_IN_L1` define is set.

**For layer norm**: The same pattern applies. Layer norm needs the input to persist for: (1) compute mean, (2) subtract mean. If gamma/beta are applied, you need the intermediate result to persist for the multiply/add step. The L1 budget calculation should account for all intermediates similarly.

## Compute Kernel Structure and Helper Calls

### Initialization (Lines 373-381)

```cpp
void kernel_main() {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask, onetile);       // Wait for reader to generate mask
        cb_wait_front(cb_max_mask, onetile);   // Wait for reader to generate max mask
    }
    cb_wait_front(cb_reduction_scaler, onetile); // Wait for reader to generate scaler

    init_sfpu(cb_input, cb_output);              // Initialize SFPU for unary ops
    binary_op_init_common(cb_input, cb_input, cb_output); // Initialize binary op pipeline
```

**`init_sfpu(cb_in, cb_out)`**: Configures SFPU hardware (exp, recip, etc.) with input/output data formats.

**`binary_op_init_common(cb_a, cb_b, cb_out)`**: Configures unpack, math, and pack pipelines for binary operations. Sets up hardware for all three compute threads (unpack, math, pack).

### Phase 1: Find Max Value in Row (`find_max_value_in_row()`)

This function scans all `Wt` tiles in the row and computes element-wise maximum across tiles, producing one tile that contains the per-element maximum.

**Key compute calls (L1 mode)**:
```cpp
tile_regs_acquire();                              // Acquire DST registers (zeroed)
reconfig_data_format(cb_input, cb_input);         // Configure unpack for input format

// For each tile in the row:
copy_tile_init(cb_input);                         // Init unpack from cb_input
copy_tile(cb_input, col, working_register);       // Unpack tile[col] -> DST[reg]

// If last tile and masking needed:
copy_tile_init(cb_mask);
copy_tile(cb_mask, 0, mask_register);             // Load mask -> DST[reg+1]
mask_tile_init();
mask_tile(working_register, mask_register);       // Zero padding in data tile
copy_tile_init(cb_max_mask);
copy_tile(cb_max_mask, 0, mask_register);         // Load -inf mask -> DST[reg+1]
add_binary_tile_init();
add_binary_tile(working_register, mask_register, working_register); // Add -inf to padding

// Accumulate max:
binary_max_tile_init();
binary_max_tile(max_value_register, tile_register, max_value_register);

tile_regs_commit();                               // Hand DST to packer
tile_regs_wait();                                 // Packer waits for DST
pack_reconfig_data_format(cb_max_value_before_reduction);
pack_tile(max_value_register, cb_max_value_before_reduction); // Pack result
tile_regs_release();                              // Release DST
cb_push_back(cb_max_value_before_reduction, onetile);
```

**L1 vs Streaming difference**: In L1 mode, `cb_wait_front(cb_input, col + block_size)` -- waits for cumulative tiles, never pops. In streaming mode, `cb_wait_front(cb_input, block_size)` then `cb_pop_front(cb_input, block_size)` per block.

**DST register usage**: Register 0 = accumulating max, Register 1 = current tile. With masking, register `working_register + 1` = mask (must be adjacent to data register due to hardware constraint).

### Phase 2: Reduce Max Value (`reduce_max_value()`)

Row-reduces the max tile to produce a column vector (one value per row of the tile).

```cpp
cb_wait_front(cb_max_value_before_reduction, onetile);
cb_reserve_back(cb_max_value_after_reduction, onetile);

tile_regs_acquire();
reconfig_data_format(cb_max_value_before_reduction, cb_reduction_scaler);

reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_max_value_before_reduction,  // input CB
    cb_reduction_scaler,            // scaler CB (all 1.0)
    cb_max_value_after_reduction);  // output CB

reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_max_value_before_reduction,  // input CB
    cb_reduction_scaler,            // scaler CB
    0,                              // input tile index
    0,                              // scaler tile index
    reduction_register);            // DST register index

reduce_uninit();                    // Reset packer edge mask
tile_regs_commit();

tile_regs_wait();
pack_reconfig_data_format(cb_max_value_after_reduction);
pack_tile(reduction_register, cb_max_value_after_reduction);
tile_regs_release();

cb_push_back(cb_max_value_after_reduction, onetile);
cb_pop_front(cb_max_value_before_reduction, onetile);
```

**Note**: `ReduceDim::REDUCE_ROW` reduces across columns (the row dimension collapses), producing a column vector. The scaler tile (all 1.0) is required by the reduce API even for MAX -- it scales the result.

**For layer norm**: Use `PoolType::SUM` with a scaler of `1.0/Wt_elements` (or `1.0/W` where W is the number of elements in the reduction dimension) to compute the mean. Or use matmul-based reduction (see Phase 4) for better precision and manually divide.

### Phase 3: Calculate sum(exp(x - max(x))) (`calculate_sum_exp_x()`)

**L1 mode** -- Two sub-phases:

Sub-phase A: Compute exp(x - max) for all tiles, store in cb_exp:
```cpp
cb_wait_front(cb_max_value_after_reduction, onetile);
cb_reserve_back(cb_exp, Wt);

reconfig_data_format(cb_input, cb_max_value_after_reduction);
// For each block of tiles:
tile_regs_acquire();
sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
sub_tiles_bcast<BroadcastType::COL>(
    cb_input,                       // operand A (data tile from CB)
    cb_max_value_after_reduction,   // operand B (max col-vector, broadcast across cols)
    col,                            // A tile index
    0,                              // B tile index
    block_idx);                     // DST register index

exp_tile_init<false>();             // Init for exact exp (not approximate)
exp_tile<false>(block_idx);         // DST[block_idx] = exp(DST[block_idx])

// If last tile and masking:
mask_tile(block_idx, mask_register);  // Zero out padding positions

tile_regs_commit();
tile_regs_wait();
pack_reconfig_data_format(cb_exp);
pack_tile(block_idx, cb_exp);       // Pack each exp tile to cb_exp
tile_regs_release();
cb_push_back(cb_exp, block_size);
```

Sub-phase B: Accumulate sum of exp tiles:
```cpp
cb_wait_front(cb_exp, Wt);
cb_reserve_back(cb_exp_sum_before_reduction, onetile);
tile_regs_acquire();
// For each tile col in [0..Wt):
copy_tile_init(cb_exp);
copy_tile(cb_exp, col, tile_register);
if (col > 0) {
    add_binary_tile_init();
    add_binary_tile(working_register, tile_register, working_register);
}
tile_regs_commit();
tile_regs_wait();
pack_reconfig_data_format(cb_exp_sum_before_reduction);
pack_tile(working_register, cb_exp_sum_before_reduction);
tile_regs_release();
cb_push_back(cb_exp_sum_before_reduction, onetile);
```

**Streaming mode** -- Combines both sub-phases in a single pass. No cb_exp storage. Uses `unary_bcast` to load max value into a register once, then iterates through blocks:
```cpp
unary_bcast_init<BroadcastType::COL>(cb_max_value_after_reduction, cb_max_value_after_reduction);
unary_bcast<BroadcastType::COL>(
    cb_max_value_after_reduction,   // source CB
    0,                              // tile index in CB
    max_value_register);            // DST register to broadcast into

// Then for each block:
copy_tile(cb_input, block_idx, working_register);
sub_binary_tile_init();
sub_binary_tile(working_register, max_value_register, working_register); // x - max
exp_tile<false>(working_register);
// mask if needed
// accumulate: add_binary_tile(accum, working, accum)
cb_pop_front(cb_input, block_size);
```

**Key difference from L1 mode**: In streaming mode, `unary_bcast` + `sub_binary_tile` is used instead of `sub_tiles_bcast<COL>`, because the max value is held in a DST register rather than being read from CB each time.

**For layer norm**: This phase maps to: (1) subtract mean, (2) square, (3) accumulate sum of squares for variance. The mean replaces max. `square` (via `mul_binary_tile(x, x, x)`) replaces `exp`. Masking is still needed for padding.

### Phase 4: Reduce Sum and Reciprocal (`reduce_sum_exp_x()`)

This phase uses **matmul-based row reduction** instead of `reduce_tile<SUM>` for precision.

```cpp
cb_wait_front(cb_exp_sum_before_reduction, onetile);
cb_reserve_back(cb_exp_sum_after_reduction, onetile);
cb_wait_front(cb_mat_mul_reduce, onetile);      // The special reduction tile

tile_regs_acquire();
mm_init(cb_exp_sum_before_reduction, cb_mat_mul_reduce, cb_exp_sum_after_reduction, 0);
matmul_tiles(
    cb_exp_sum_before_reduction,    // A: the sum tile
    cb_mat_mul_reduce,              // B: reduction pattern tile (1.0 in col 0)
    0,                              // A tile index
    0,                              // B tile index
    reduction_register);            // DST register (accumulates)

recip_tile_init();
recip_tile(reduction_register);     // DST[0] = 1/sum(exp(x))

tile_regs_commit();
tile_regs_wait();
pack_reconfig_data_format(cb_exp_sum_after_reduction);
pack_tile(reduction_register, cb_exp_sum_after_reduction);
tile_regs_release();
cb_push_back(cb_exp_sum_after_reduction, onetile);
cb_pop_front(cb_exp_sum_before_reduction, onetile);
```

**Why matmul instead of reduce_tile**: The comments explicitly state that `reduce_tile` causes precision loss for SUM operations. The matmul approach multiplies the data tile by a special tile that has 1.0 in the first column of each left face and 0.0 elsewhere. The matrix multiply accumulates the row values into the first column, effectively performing row reduction.

**For layer norm**: Use this same matmul-based reduction pattern for both mean and variance reductions. After getting the sum, multiply by `1/N` (where N = number of elements) to get the mean/variance. For variance, use `recip_tile` + `sqrt_tile` (rsqrt) instead of just `recip_tile`.

### Phase 5: Final Scaling (in `kernel_main()` body)

```cpp
cb_wait_front(cb_exp_sum_after_reduction, onetile); // 1/sum(exp(x))

for (uint32_t col = 0; col < Wt; col += block_size) {
    // In streaming mode: wait for reader to provide next block
    // cb_wait_front(cb_input, block_size);
    cb_reserve_back(cb_output, block_size);

    tile_regs_acquire();
    // Broadcast 1/sum into a register:
    reconfig_data_format(cb_exp_sum_after_reduction, cb_exp_sum_after_reduction);
    unary_bcast_init<BroadcastType::COL>(cb_exp_sum_after_reduction, cb_exp_sum_after_reduction);
    unary_bcast<BroadcastType::COL>(
        cb_exp_sum_after_reduction, 0, sum_exp_register); // DST[block_size] = broadcast(1/sum)

    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
        // Read from stored exp tiles:
        reconfig_data_format(cb_exp, cb_exp);
        copy_tile_init(cb_exp);
        copy_tile(cb_exp, col + block_idx, block_idx);
#else
        // Recompute exp(x - max) from raw input:
        reconfig_data_format(cb_input, cb_max_value_after_reduction);
        sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
        sub_tiles_bcast<BroadcastType::COL>(
            cb_input, cb_max_value_after_reduction,
            block_idx, 0, block_idx);       // DST[i] = input[i] - max (broadcast)
        exp_tile_init<false>();
        exp_tile<false>(block_idx);          // DST[i] = exp(input[i] - max)
#endif
        mul_binary_tile_init();
        mul_binary_tile(block_idx, sum_exp_register, block_idx); // DST[i] *= 1/sum
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output);
    }
    tile_regs_release();
    cb_push_back(cb_output, block_size);
    // In streaming mode: cb_pop_front(cb_input, block_size);
}

// Cleanup at end of row:
cb_pop_front(cb_max_value_after_reduction, onetile);
cb_pop_front(cb_exp_sum_after_reduction, onetile);
// In L1 mode:
// cb_pop_front(cb_input, Wt);
// cb_pop_front(cb_exp, Wt);
```

**DST register layout in Phase 5**: Registers 0..block_size-1 hold the current block of output tiles. Register `block_size` (e.g., register 3 if block_size=3) holds the broadcast 1/sum value. This is why `get_block_size(Wt, 3)` caps at 3 -- we need at least one extra DST register for the scalar. With `fp32_dest_acc_en=true` and double-buffering (default), only 4 DST registers are available.

**For layer norm**: This phase maps to: multiply by rsqrt(var+eps) (broadcast COL), then optionally multiply by gamma and add beta. Gamma and beta would need additional CBs and additional mul/add steps.

## Compute Helper Call Signatures Reference

### Register Lifecycle
| Function | Signature | Purpose |
|---|---|---|
| `tile_regs_acquire()` | `void tile_regs_acquire()` | Acquire DST registers (zeroed), block until available |
| `tile_regs_commit()` | `void tile_regs_commit()` | Transfer DST ownership to packer |
| `tile_regs_wait()` | `void tile_regs_wait()` | Packer waits for DST to be ready |
| `tile_regs_release()` | `void tile_regs_release()` | Release DST for next acquire cycle |

### Data Format Configuration
| Function | Signature | Purpose |
|---|---|---|
| `reconfig_data_format` | `void reconfig_data_format(uint32_t srca_new, uint32_t srcb_new)` | Reconfigure unpacker data formats for sources A and B (pass CB indices) |
| `pack_reconfig_data_format` | `void pack_reconfig_data_format(uint32_t dst_cb)` | Reconfigure packer output format |

### Tile Copy (Unpack to DST)
| Function | Signature | Purpose |
|---|---|---|
| `copy_tile_init` | `void copy_tile_init(uint32_t cb_id)` | Configure unpacker for copy from given CB |
| `copy_tile` | `void copy_tile(uint32_t cb_id, uint32_t tile_idx, uint32_t dst_idx)` | Unpack tile from CB into DST register |

### Pack (DST to CB)
| Function | Signature | Purpose |
|---|---|---|
| `pack_tile` | `void pack_tile(uint32_t dst_idx, uint32_t cb_id)` | Pack DST register tile into CB |

### Binary Operations (DST register to DST register)
| Function | Signature | Purpose |
|---|---|---|
| `add_binary_tile_init` | `void add_binary_tile_init()` | Init for DST-to-DST addition |
| `add_binary_tile` | `void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | DST[odst] = DST[idst0] + DST[idst1] |
| `sub_binary_tile_init` | `void sub_binary_tile_init()` | Init for DST-to-DST subtraction |
| `sub_binary_tile` | `void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | DST[odst] = DST[idst0] - DST[idst1] |
| `mul_binary_tile_init` | `void mul_binary_tile_init()` | Init for DST-to-DST multiplication |
| `mul_binary_tile` | `void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | DST[odst] = DST[idst0] * DST[idst1] |
| `binary_max_tile_init` | `void binary_max_tile_init()` | Init for DST-to-DST element-wise max |
| `binary_max_tile` | `void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | DST[odst] = max(DST[idst0], DST[idst1]) |

### Broadcast Operations (CB to DST with broadcast)
| Function | Signature | Purpose |
|---|---|---|
| `sub_bcast_cols_init_short` | `void sub_bcast_cols_init_short(uint32_t cb_a, uint32_t cb_b)` | Init subtract with COL broadcast |
| `sub_tiles_bcast<BroadcastType::COL>` | `void sub_tiles_bcast<BroadcastType::COL>(uint32_t cb_a, uint32_t cb_b, uint32_t a_idx, uint32_t b_idx, uint32_t dst_idx)` | DST[dst] = CB_A[a_idx] - broadcast_col(CB_B[b_idx]) |
| `unary_bcast_init<BroadcastType::COL>` | `void unary_bcast_init<BroadcastType::COL>(uint32_t cb, uint32_t cb)` | Init unary broadcast (datacopy with broadcast) |
| `unary_bcast<BroadcastType::COL>` | `void unary_bcast<BroadcastType::COL>(uint32_t cb, uint32_t tile_idx, uint32_t dst_idx)` | DST[dst] = broadcast_col(CB[tile_idx]) -- first column broadcast across all columns |

**BroadcastType::COL semantics**: The first column of the source tile is replicated across all 32 columns. This creates a "column vector broadcast" where each row gets its own scalar value. Used for per-row operations like "subtract row-max" or "multiply by row-reciprocal".

### Unary SFPU Operations
| Function | Signature | Purpose |
|---|---|---|
| `exp_tile_init<false>` | `template<bool approx> void exp_tile_init()` | Init for exp. `false` = exact, `true` = approximate |
| `exp_tile<false>` | `template<bool approx, ...> void exp_tile(uint32_t idst)` | DST[idst] = exp(DST[idst]) |
| `recip_tile_init` | `void recip_tile_init()` | Init for reciprocal |
| `recip_tile` | `void recip_tile(uint32_t idst)` | DST[idst] = 1.0 / DST[idst] |

### Masking
| Function | Signature | Purpose |
|---|---|---|
| `mask_tile_init` | `void mask_tile_init()` | Init SFPU for masking |
| `mask_tile` | `void mask_tile(uint32_t idst_data, uint32_t idst_mask)` | Apply mask: zero elements where mask is 0. **Constraint: idst_mask MUST equal idst_data + 1** |

### Reduction
| Function | Signature | Purpose |
|---|---|---|
| `reduce_init<PoolType, ReduceDim>` | `void reduce_init(uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_out)` | Init reduce hardware |
| `reduce_tile<PoolType, ReduceDim>` | `void reduce_tile(uint32_t cb_in, uint32_t cb_scaler, uint32_t in_idx, uint32_t scaler_idx, uint32_t dst_idx)` | Reduce tile, result in DST |
| `reduce_uninit` | `void reduce_uninit()` | Reset packer edge mask after reduce |

### Matrix Multiply (used for precise row reduction)
| Function | Signature | Purpose |
|---|---|---|
| `mm_init` | `void mm_init(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t transpose)` | Init matmul engine |
| `matmul_tiles` | `void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t a_idx, uint32_t b_idx, uint32_t dst_idx)` | DST[dst] += CB_A[a_idx] * CB_B[b_idx]. **Accumulates** into DST |

### CB Operations
| Function | Signature | Purpose |
|---|---|---|
| `cb_wait_front` | `void cb_wait_front(uint32_t cb_id, uint32_t num_tiles)` | Wait until CB has num_tiles available to read |
| `cb_pop_front` | `void cb_pop_front(uint32_t cb_id, uint32_t num_tiles)` | Free num_tiles from CB read pointer |
| `cb_reserve_back` | `void cb_reserve_back(uint32_t cb_id, uint32_t num_tiles)` | Reserve space for num_tiles in CB |
| `cb_push_back` | `void cb_push_back(uint32_t cb_id, uint32_t num_tiles)` | Commit num_tiles to CB write pointer |

## Scalar/Constant CB Setup (Reader-Generated)

The reader kernel generates four constant tiles before entering the main loop. These tiles persist for the entire kernel lifetime.

### 1. Width Mask Tile (`cb_mask`, c_1)
```cpp
generate_mask_tile(cb_mask_idx, /*fill=*/one, /*mask_fill=*/zero, mask_w);
```
- Pattern: 1.0 for columns < mask_w, 0.0 for columns >= mask_w
- Used in compute to zero out padding positions via `mask_tile(data_reg, mask_reg)`
- Only generated if `mask_w != 0` (i.e., logical width is not a multiple of TILE_WIDTH=32)

### 2. Max Mask Tile (`cb_max_mask`, c_2)
```cpp
generate_mask_tile(cb_max_mask_idx, /*fill=*/zero, /*mask_fill=*/minus_inf, mask_w);
```
- Pattern: 0.0 for columns < mask_w, -inf for columns >= mask_w
- Added to the data tile after masking to replace padding zeros with -inf
- This ensures max operation ignores padding (max(-inf, x) = x for any finite x)
- Only generated if `mask_w != 0`

### 3. Reduction Scaler Tile (`cb_reduction_scaler`, c_3)
```cpp
generate_tile_with_bfloat16_value(cb_reduction_scaler_idx, one); // all 1.0
```
- Pattern: Every element is 1.0 (bfloat16)
- Used by `reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>` as the scaler operand
- The reduce API requires a scaler tile; 1.0 means no scaling

### 4. Matmul Row-Reduce Tile (`cb_mat_mul_reduce`, c_4)
```cpp
generate_matmul_row_reduce_tile(cb_matmul_reduce);
```
- Pattern: 1.0 in column 0 of left faces (faces 0, 2), 0.0 elsewhere
- When multiplied as `A * B` where A is the data and B is this tile, the result accumulates all columns into column 0
- This implements `sum_across_row(A)` as a matrix multiply, avoiding precision issues in `reduce_tile<SUM>`

**For layer norm**: You will need similar constant tiles:
- Width mask (same as softmax)
- Reduction scaler (for mean: could be 1/N packed as bfloat16, or just 1.0 and divide after)
- Matmul row-reduce tile (same pattern)
- Epsilon tile (new: a tile with epsilon value for variance stabilization)
- Gamma/beta tiles (new: if applying affine transform, need to read these from DRAM)

## Core Distribution Strategy

| Attribute | Value |
|---|---|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | compute_with_storage_grid_size (device-dependent) |
| **Total cores** | num_cores (from split_work_to_cores) |
| **Work per core** | num_rows_per_core_group_1 or num_rows_per_core_group_2 |
| **Load balancing** | Two groups: group 1 gets ceil(rows/cores), group 2 gets floor(rows/cores) |

The `split_work_to_cores` utility divides `total_rows_to_process` among available cores. It creates two core groups:
- **Group 1**: Cores with `num_rows_per_core_group_1` rows (the larger allocation)
- **Group 2**: Cores with `num_rows_per_core_group_2` rows (one fewer, handling remainder)

Core indexing is column-major: `core = {i / num_cores_y, i % num_cores_y}`.

Each core group gets its own compute kernel handle with different compile-time `num_rows_per_core` values, but the same kernel source.

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_rows_per_core | uint32_t | Number of tile-rows this core processes (differs by group) |
| 1 | block_size | uint32_t | Tiles processed per inner loop iteration (1-3) |
| 2 | Wt | uint32_t | Total tiles in the reduction (width) dimension |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_size | uint32_t | Block size for reading |
| 1 | Wt | uint32_t | Tiles in width dimension |
| 2 | mask_w | uint32_t | Width of valid data in last tile (0 = no masking) |
| 3+ | TensorAccessorArgs | uint32_t[] | Input buffer accessor parameters |

### Compile-Time Arguments (Writer Kernel)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_size | uint32_t | Block size for writing |
| 1 | Wt | uint32_t | Tiles in width dimension |
| 2+ | TensorAccessorArgs | uint32_t[] | Output buffer accessor parameters |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | input_address | uint32_t | DRAM address of input buffer |
| 1 | num_rows_per_core | uint32_t | Rows to process on this core |
| 2 | start_row | uint32_t | Row offset (cumulative from prior cores) |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_address | uint32_t | DRAM address of output buffer |
| 1 | num_rows_per_core | uint32_t | Rows to process on this core |
| 2 | start_row | uint32_t | Row offset (cumulative from prior cores) |

### Preprocessor Defines (Shared Across All Kernels)

| Define | Condition | Effect |
|---|---|---|
| `DO_MASK_W` | `mask_w != 0` | Enables width masking for padding |
| `EVERYTHING_FITS_IN_L1` | Required L1 <= available L1 | Enables single-read L1 persistence mode |
| `REDUCE_OP` | Always set to `PoolType::SUM` | Required by LLK reduce API for compilation |
| `REDUCE_DIM` | Always set to `ReduceDim::REDUCE_ROW` | Required by LLK reduce API for compilation |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|---|---|---|---|---|---|
| reader | RISCV_0 | NOC0 | DRAM input buffer | cb_input(c_0), cb_mask(c_1), cb_max_mask(c_2), cb_scaler(c_3), cb_matmul(c_4) | Generate constants, read input tiles |
| compute | RISCV_2 | N/A | cb_input(c_0), cb_mask(c_1), cb_max_mask(c_2), cb_scaler(c_3), cb_matmul(c_4), cb_max_after(c_6), cb_exp(c_7), cb_sum_after(c_9) | cb_max_before(c_5), cb_max_after(c_6), cb_exp(c_7), cb_sum_before(c_8), cb_sum_after(c_9), cb_output(c_10) | Max, reduce, exp, matmul-reduce, recip, multiply |
| writer | RISCV_1 | NOC1 | cb_output(c_10) | DRAM output buffer | Write output tiles |

### Compute Kernel Details
- **File**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp`
- **ComputeConfig**: `fp32_dest_acc_en=true`, `MathFidelity::HiFi4`, `math_approx_mode=false`
- **DST Register Count**: 4 (fp32 enabled + double-buffering default) -- this limits block_size to 3 (need 1 spare register)

## Implementation Notes

### 1. Precision-Critical Design Choices
- **Float32 accumulation** (`fp32_dest_acc_en=true`): Ensures intermediate results in DST registers maintain precision. Critical for sum accumulation.
- **Float32 CBs for sum** (c_8, c_9): The exp-sum tiles use Float32 data format to avoid precision loss during accumulation.
- **Matmul-based row reduction** instead of `reduce_tile<SUM>`: Explicitly noted that `reduce_tile` causes precision loss. The matmul approach uses the FPU matrix engine which has better accumulation properties.
- **Exact exp** (`exp_tile<false>`): Uses exact exponential, not approximate.
- **HiFi4 math fidelity**: Highest fidelity mode for all operations.

### 2. Block Size Constraint
`get_block_size(Wt, 3)` returns the largest value in {3, 2, 1} that divides Wt. The maximum is 3 (not 4, the typical max) because Phase 5 needs `block_size` registers for data tiles plus 1 register for the broadcast scalar, totaling `block_size + 1`. With fp32 DST, only 4 registers are available.

### 3. Masking Strategy (Two-Step for Max)
For the max-finding phase, masking is two-step:
1. `mask_tile` with the 1.0/0.0 mask: zeros out padding positions
2. `add_binary_tile` with the 0.0/-inf mask: adds -inf to padding positions

This is necessary because `NaN + (-inf) = NaN` and `(-inf) * 0 = NaN`. If padding contains NaN, just adding -inf would not work. The mask_tile first forces padding to 0, then the add replaces it with -inf.

For the exp/sum phase, only the 1.0/0.0 mask is used (masking to 0 after exp is correct since exp(anything)*0 = 0).

### 4. `cb_reserve_back` Placement
In L1 mode's `find_max_value_in_row`, the `cb_reserve_back` for `cb_max_value_before_reduction` is commented out because the tile is packed directly without a prior reserve -- this works because the capacity is 2 and only 1 tile is ever produced. The reserve is implicit from initialization. In streaming mode, it is explicitly called.

### 5. `reconfig_data_format` Calls
These are frequent throughout the kernel because different operations read from CBs with different data formats (Float16_b for most, Float32 for sum accumulators). Each time the compute kernel switches between reading from a bf16 CB and a fp32 CB, it must reconfigure the unpacker.

### 6. `unary_bcast` vs `sub_tiles_bcast` Pattern
Two different patterns for broadcast subtract appear:
- **`sub_tiles_bcast<COL>(cb_A, cb_B, ...)`**: Reads both operands from CBs. Used in L1 mode where both input and max are in CBs.
- **`unary_bcast<COL>` + `sub_binary_tile`**: Loads the broadcast value into a DST register once, then subtracts from each tile in DST. Used in streaming mode where the max value should be held in a register to avoid re-reading from CB for each block.

### 7. For Layer Norm: Additional Operations Needed
Layer norm requires operations not used in softmax:
- `sqrt_tile` / `rsqrt_tile`: For computing 1/sqrt(variance + epsilon). Check `api/compute/eltwise_unary/sqrt.h`.
- Additional broadcast-multiply and broadcast-add for gamma and beta application.
- A way to add epsilon to the variance tile before rsqrt. This could use `add_binary_tile` with a pre-generated epsilon constant tile, or use the `binop_with_scalar` API.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release work?"
   **Reason**: Understanding the DST register lifecycle is fundamental to understanding how the compute kernel orchestrates its phases.
   **Key Findings**: acquire zeroes DST and blocks until available; commit transfers ownership to packer; wait blocks packer until DST ready; release makes DST available for next acquire. This is a mandatory four-step cycle.

2. **Query**: "How does reduce_tile work with PoolType::MAX, PoolType::SUM, ReduceDim::REDUCE_ROW?"
   **Reason**: The softmax kernel uses reduce for max-value reduction.
   **Key Findings**: reduce_init takes (cb_in, cb_scaler, cb_out) as template+function params. reduce_tile accumulates into DST. ReduceDim::REDUCE_ROW reduces across columns producing a column vector. A scaler tile is always required. reduce_uninit resets packer edge mask.

3. **Query**: "How do broadcast APIs work - sub_bcast_cols_init_short, sub_tiles_bcast COL, unary_bcast?"
   **Reason**: Column broadcast is used throughout for per-row operations (subtract max, multiply reciprocal).
   **Key Findings**: BroadcastType::COL broadcasts the first column of operand B across all columns. unary_bcast copies a tile into DST with broadcast. sub_tiles_bcast reads both operands from CBs and subtracts with broadcast.

4. **Query**: "What are binary_max_tile, add_binary_tile, sub_binary_tile, mul_binary_tile signatures?"
   **Reason**: These DST-to-DST binary operations are the core compute primitives.
   **Key Findings**: All take (idst0, idst1, odst) -- two source DST indices and one output DST index. The _init variants take no parameters. They operate element-wise on 32x32 tiles.

5. **Query**: "How does mask_tile work and what is the adjacency constraint?"
   **Reason**: Masking is critical for handling padding in partial-width tiles.
   **Key Findings**: mask_tile zeroes elements where mask is 0. The mask DST register MUST be idst_data + 1 due to SFPU hardware limitation. The idst_mask parameter is actually unused internally; (idst_data+1) is always used.

6. **Query**: "What are exp_tile, recip_tile signatures and behavior?"
   **Reason**: These SFPU operations are core to softmax and have template parameters affecting precision.
   **Key Findings**: exp_tile<approx>(idst) -- false for exact. recip_tile(idst) computes 1/x in place. Both operate on a single DST register.

7. **Query**: "Effect of fp32_dest_acc_en on DST register count?"
   **Reason**: The block_size limit of 3 depends on available DST registers.
   **Key Findings**: With fp32_dest_acc_en=true and default double-buffering: 4 DST registers. Without fp32: 8 registers. This directly constrains the maximum block_size.

8. **Query**: "mm_init and matmul_tiles exact signatures and accumulation behavior?"
   **Reason**: The matmul-based reduction is a key precision optimization.
   **Key Findings**: mm_init(cb_a, cb_b, cb_out, transpose_flag). matmul_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx) -- **accumulates** into DST (does not overwrite). This is important: DST must be zeroed (by tile_regs_acquire) before first matmul_tiles call.

9. **Query**: "How do init_sfpu and binary_op_init_common work?"
   **Reason**: These are called once at kernel start and set up the entire compute pipeline.
   **Key Findings**: init_sfpu(cb_in, cb_out) configures SFPU for unary ops. binary_op_init_common(cb_a, cb_b, cb_out) configures unpack/math/pack for binary ops. Both called once before the main loop.

### Documentation References

1. **Source**: `tt-train/sources/ttml/metal/common/program_utils.hpp`
   **Reason**: Understanding get_block_size and create_circular_buffer helpers.
   **Key Information**: get_block_size(Wt, max=3) finds largest divisor of Wt up to max. create_compute_kernel uses MathFidelity::HiFi4 and configurable fp32_dest_acc_en.

2. **Source**: `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
   **Reason**: Understanding constant tile generation patterns.
   **Key Information**: generate_mask_tile creates per-element masks with configurable fill/mask values. generate_matmul_row_reduce_tile creates the reduction-via-matmul pattern tile. generate_tile_with_bfloat16_value fills all elements with a single value.

3. **Source**: `tech_reports/data_formats/reconfig_data_format.md`
   **Reason**: Understanding the reconfig_data_format API signatures.
   **Key Information**: Two-argument form: reconfig_data_format(srca_new, srcb_new) sets new unpack formats. Four-argument form includes old formats for optimization.
