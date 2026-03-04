# Softmax Implementation Analysis (Compute-Core Focus)

## Overview

Softmax computes `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))` along a specified dimension (typically the last/W dimension). This analysis covers **three** softmax compute kernel variants:

1. **moreh_softmax_w.cpp** (General W-small) - cleanest row-wise softmax, tile-at-a-time processing
2. **softmax.cpp** (Attention-optimized) - block-based processing with fused scale/mask support
3. **softmax_large_tensor.cpp** (Large tensor) - multi-pass streaming when W dimension exceeds L1 capacity

All three implement the same mathematical operation but differ in how they handle data reuse and L1 memory pressure.

**Program factory files analyzed:**
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_attention_optimized.cpp`

**Compute kernel files analyzed:**
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp`
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_large_tensor.cpp`
- `/localdev/dstoiljkovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax_sharded.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row (one 32-high strip across the full W dimension) |
| **Unit size** | Wt tiles (all tiles in one row of the reduction dimension) |
| **Total units** | NC * Ht (batch_count * height_in_tiles) |
| **Loop structure** | Outer: tile rows (NCHt), Inner: tiles across W dimension (Wt) |

Each "tile row" is one row of tiles spanning the full W dimension. The softmax reduction (max, sum) is computed across all Wt tiles in that row, then the normalization is broadcast back across the same Wt tiles.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] (arbitrary rank, treated as [..., H, W]) | Same as input |
| **Dimension convention** | Last dim = reduction dim (W for softmax_w) | Same |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED (general), SHARDED (optimized sharded variant) | Same as input |
| **Buffer type** | DRAM (general), L1 (sharded) | Same as input |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
No explicit tilize/untilize occurs within the operation. Input and output are both in TILE_LAYOUT. When `fp32_dest_acc_en` is set, intermediate CBs use Float32 format while input/output remain in their original format, requiring data format reconfiguration between phases.

## Data Flow Pattern

The softmax compute kernel executes a **multi-phase pipeline** over each tile row. Each phase reads the same input tiles (or intermediates from the prior phase) and produces a scalar or tile-row result.

### Phase 1: Row-wise MAX Reduction (Numeric Stability)
```
cb_in0 (Wt tiles) --[reduce MAX across W]--> cb_max (1 tile)
```
- Uses `compute_kernel_lib::reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>` in moreh variant
- Input tiles are NOT popped (WaitUpfrontNoPop) so they persist for Phase 2

### Phase 2: Subtract Max + Exp
```
cb_in0 (Wt tiles) + cb_max (1 tile) --[sub_bcast_cols + exp]--> cb_exps (Wt tiles)
```
- `sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst)` subtracts the max from each tile
- `exp_tile(dst)` computes exp on the result in DST registers
- Output packed to cb_exps (or cb_x_m_max + separate exp step in moreh variant)

### Phase 3: Row-wise SUM Reduction + Reciprocal
```
cb_exps (Wt tiles) --[reduce SUM across W + recip]--> cb_recipsumexps (1 tile)
```
- Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` with post-reduce lambda:
  ```cpp
  [](uint32_t dst_idx) {
      recip_tile_init();
      recip_tile(dst_idx);  // 1/sum(exp(x)) computed in-place in DST
  }
  ```
- The post_reduce_op lambda executes AFTER the reduce accumulation completes but BEFORE pack_tile
- Input tiles (cb_exps) are NOT popped (persist for Phase 4)

### Phase 4: Multiply by Reciprocal Sum
```
cb_exps (Wt tiles) * cb_recipsumexps (1 tile) --[mul_bcast_cols]--> cb_out0 (Wt tiles)
```
- `mul_tiles_bcast<BroadcastType::COL>(cb_exps, cb_recipsumexps, w, 0, dst)` broadcasts the scalar across each tile
- Final result written to output CB

### Reader/Writer Summary (De-emphasized)
- **Reader**: Provides input tiles to cb_in0, scaler tile to cb_bcast_scaler, optional mask/scale tiles. For the attention-optimized variant, also provides fused scale (cb_fused_scale) and attention mask (cb_fused_attn).
- **Writer**: Consumes cb_out0 tiles and writes to DRAM/L1 output buffer.

## Circular Buffer Configuration

### Variant 1: moreh_softmax_w (General W-small)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | Wt tiles | 1 tile | Multi (Wt) | Reader | Compute | Row |
| c_1 | cb_mask | Padding mask | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Reduce scaler (all 1s) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Final output | Wt tiles | 1 tile | Multi (Wt) | Compute | Writer | Row |
| c_24 | cb_exps | exp(x-max) | Wt tiles | 1 tile | Multi (Wt) | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row max value | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max(x) | Wt tiles | 1 tile | Multi (Wt) | Compute | Compute | Row |
| c_28 | cb_tmp | Temp for masked max | 1 tile | 1 tile | Single | Compute | Compute | Row |

**Data format:** Input/output use input tensor's format; intermediates (c_24 through c_28) use `intermed_data_format` which is Float32 when `fp32_dest_acc_en` is set, otherwise same as input.

### Variant 2: softmax.cpp (Attention-optimized)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | Wt or 2*block tiles | block_size | Double | Reader | Compute | Row |
| c_2 | cb_bcast_scaler | Reduce scaler | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_fused_scale | 1/sqrt(d) scaler | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_4 | cb_fused_attn | Attention mask | Wt tiles | block_size | Multi | Reader | Compute | Row* |
| c_5 | cb_mask_padded | Padding mask (-inf) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_6 | cb_exps | exp(x) values | Wt tiles | block_size | Multi (Wt) | Compute | Compute | Row |
| c_7 | cb_recipsumexps | 1/sum(exp) | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_8 | cb_max | Row max (numeric_stable) | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_9 | cb_scale_mask | Scaled+masked intermediate | Wt+block tiles | block_size | Multi | Compute | Compute | Row |
| c_10 | cb_x | x after processing (numeric_stable) | Wt tiles | block_size | Multi (Wt) | Compute | Compute | Row |
| c_11 | cb_out0 | Final output | 2*block tiles | block_size | Double | Compute | Writer | Row |

*cb_fused_attn has special lifetime: for non-causal mask, it persists across Ht rows within a batch (one mask row broadcast across H), popped only at batch boundary.

**block_size**: `find_max_divisor(Wt, 4)` for FP32, `find_max_divisor(Wt, 8)` for FP16. This is the number of tiles processed per inner loop iteration (maps to `ndst` in the kernel, representing the number of DST registers used).

### Variant 3: softmax_large_tensor.cpp (Multi-pass)

Additional CBs beyond the attention-optimized set:

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_12 | cb_prev_reduce | Previous partial sum | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_15 | cb_prev_max | Previous partial max | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_16 | cb_recip | 1/sum result | 1 tile | 1 tile | Single | Compute | Compute | Row |

These extra CBs enable **accumulation across multi-pass streaming** when Wt exceeds cb_length_t (the number of tiles that fit in L1).

## Pipeline Pattern Summary

### moreh_softmax_w
- **cb_in0 (Wt tiles)**: Multi-buffered -- entire row loaded, persists through MAX reduction via WaitUpfrontNoPop, then consumed in subtract phase
- **cb_exps (Wt tiles)**: Multi-buffered -- entire row of exp results, persists through SUM reduction via WaitUpfrontNoPop, then consumed in multiply phase
- **cb_out0 (Wt tiles)**: Multi-buffered -- entire row accumulated before writer consumes
- **Scalar CBs (cb_max, cb_recipsumexps, cb_bcast_scaler, cb_mask)**: Single-buffered, 1-tile capacity

### softmax.cpp (Attention-optimized)
- **cb_in0 (2*block)**: Double-buffered -- reader can push next block while compute processes current
- **cb_exps (Wt tiles)**: Full-row buffering -- entire row persists for SUM reduce + final multiply
- **cb_out0 (2*block)**: Double-buffered -- compute can push while writer drains

### softmax_large_tensor.cpp
- **cb_in0, cb_exps, cb_scale_mask, cb_x (80 tiles each)**: Fixed-size streaming windows; data is re-read from DRAM across 3 passes per row
- **cb_prev_reduce, cb_prev_max**: Accumulator CBs that carry partial results between passes

## CB Persistence and Multi-Pass Data Reuse Patterns

This is the most critical architectural pattern in softmax. Understanding which CBs persist across phases and why is essential for designing LayerNorm.

### Pattern: "WaitUpfrontNoPop" Persistence

The key to avoiding redundant DRAM reads is the `WaitUpfrontNoPop` policy on the reduce helper:

```cpp
// From moreh_softmax_w.cpp - Phase 1 (MAX reduce)
compute_kernel_lib::reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max, ReduceInputBlockShape::row(Wt - 1));
// cb_in0 tiles are NOT popped -- they remain available for Phase 2 subtraction
```

```cpp
// From moreh_softmax_w.cpp - Phase 3 (SUM reduce)
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps, ReduceInputBlockShape::row(Wt));
// cb_exps tiles are NOT popped -- they remain available for Phase 4 multiplication
```

**Why this matters for LayerNorm:** LayerNorm similarly needs the input tiles for:
1. Computing mean (SUM reduce) -- tiles must persist
2. Computing (x - mean) -- reuses the same input tiles
3. Computing variance (SUM of squares of (x-mean)) -- needs the (x-mean) intermediate
4. Final normalization -- needs (x-mean) again

### Pattern: Intermediate CB as Persistent Storage

In moreh_softmax_w, `cb_x_m_max` (c_27, Wt tiles) stores `x - max(x)` and persists across two consumers:
1. It feeds into the exp computation (cb_exps)
2. In the LOG variant, it's reused in the final subtraction phase

Similarly, `cb_exps` (c_24, Wt tiles) persists to serve both:
1. The SUM reduce (Phase 3)
2. The final multiplication (Phase 4)

### Pattern: Scalar CB Broadcast Persistence

Single-tile CBs like `cb_max`, `cb_recipsumexps`, and `cb_bcast_scaler` persist across their broadcast usage:
- `cb_bcast_scaler` (all 1.0 tile): loaded once at program start, never popped, used by all reduce calls throughout the kernel lifetime
- `cb_max`: produced by MAX reduce, consumed by subtract-broadcast loop, popped after all Wt subtractions
- `cb_recipsumexps`: produced by SUM reduce + recip, consumed by multiply-broadcast loop, popped after all Wt multiplications

### Pattern: Multi-Pass Accumulation (Large Tensor)

When the full W dimension cannot fit in L1, softmax_large_tensor.cpp uses a streaming approach with explicit accumulation CBs:

```cpp
uint32_t num_cb_passes = 1 + ((Wt - 1) / cb_length_t);  // ceiling divide

// Pass 1..N for MAX:
for (uint32_t cur_pass = 0; cur_pass < num_cb_passes; cur_pass++) {
    reduce_cb<PoolType::MAX>(cb_processed, cb_scaler, cb_prev_max, cb_max, use_prev_reduce, cur_cb_length_t);
    use_prev_reduce = true;
    std::swap(cb_max, cb_prev_max);  // ping-pong between two CBs
}
```

The `reduce_cb` template function in the large tensor variant includes a PostReduceOp lambda that conditionally accumulates with the previous partial result:

```cpp
[cb_prev_out, use_prev_reduce](uint32_t) {
    if (use_prev_reduce) {
        cb_wait_front(cb_prev_out, 1);
        copy_tile(cb_prev_out, 0, 1);       // Load previous into DST[1]
        add_binary_tile(0, 1, 0);            // accumulate: DST[0] = DST[0] + DST[1]
        cb_pop_front(cb_prev_out, 1);
    }
}
```

**Three-pass structure for large tensors:**
1. **Pass set 1**: Find global MAX across all W-chunks
2. **Pass set 2**: Re-read input, compute exp(x - max), accumulate SUM across chunks
3. **Pass set 3**: Re-read input again, compute exp(x - max), multiply by 1/sum -- write output

Input tiles must be re-read from DRAM 2-3 times because they cannot all fit in L1 simultaneously. This is the cost of the large tensor path.

## Compute Kernel API Reference (Exact Signatures)

### Initialization Functions

```cpp
// Primary init for binary operations -- MUST be called at kernel start
binary_op_init_common(cb_in0, cb_scaler, cb_out);
// Sets up unpacker for SRCA=cb_in0, SRCB=cb_scaler, packer for cb_out

// SFPU init (needed for large tensor variant)
init_sfpu(cb_a, cb_b);
```

### Reduce Helper Library

Full signature:
```cpp
template <
    PoolType reduce_type,           // SUM, MAX, AVG
    ReduceDim reduce_dim,           // REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR
    ReduceInputPolicy input_policy, // WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop
    ReduceDataFormatReconfigMode reconfig_mode,  // NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT
    typename AccumulateT,           // NoAccumulation or Accumulate
    typename PostReduceOp>          // NoOp or lambda(uint32_t dst_idx)
void compute_kernel_lib::reduce(
    uint32_t input_cb,
    uint32_t scaler_cb,
    uint32_t output_cb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{});
```

**Key ReduceInputBlockShape factories:**
```cpp
ReduceInputBlockShape::row(Wt)         // For REDUCE_ROW: 1 row of Wt cols
ReduceInputBlockShape::col(Ht)         // For REDUCE_COL: Ht rows of 1 col
ReduceInputBlockShape::of(Ht, Wt, NC)  // Full specification
ReduceInputBlockShape::single()        // 1x1x1
```

**Key ReduceInputPolicy behaviors:**
- `WaitUpfrontNoPop`: Library calls `cb_wait_front(input_cb, total_tiles)` once, never pops. Caller must pop. Tiles persist for reuse.
- `NoWaitNoPop`: Library does NOT wait or pop. Caller must have called `cb_wait_front()` before calling reduce. Uses indexed tile access.
- `BulkWaitBulkPop`: Library waits for all tiles per reduction unit, processes, then pops all. Clean symmetric pattern.
- `WaitAndPopPerTile`: Library waits/pops one tile at a time. Safest, lowest CB requirement.

**Accumulation pattern:**
```cpp
// For multi-pass reductions (large tensor):
const auto cfg = AccumulationConfig::with_cb(cb_accum);
for (uint32_t i = 0; i < num_blocks; ++i) {
    reduce<SUM, REDUCE_ROW>(..., Accumulate(cfg, i));
    // iteration 0: skip reload (first block)
    // iteration > 0: reload from cb_accum, add to new reduce result
}
```

**Post-reduce lambda pattern (softmax reciprocal):**
```cpp
// Lambda receives dst_idx: the DST register holding the reduce result
[](uint32_t dst_idx) {
    recip_tile_init();
    recip_tile(dst_idx);  // 1/x computed in-place in DST before pack
}
```

### Broadcast Operations

```cpp
// Subtract scalar from each tile (broadcast column-wise: same value for all rows in tile)
sub_bcast_cols_init_short(cb_data, cb_scalar);
sub_tiles_bcast<BroadcastType::COL>(cb_data, cb_scalar, data_idx, scalar_idx, dst_idx);
// cb_data[data_idx] - bcast_col(cb_scalar[scalar_idx]) -> DST[dst_idx]

// Multiply each tile by scalar (broadcast column-wise)
mul_bcast_cols_init_short(cb_data, cb_scalar);
mul_tiles_bcast<BroadcastType::COL>(cb_data, cb_scalar, data_idx, scalar_idx, dst_idx);
// cb_data[data_idx] * bcast_col(cb_scalar[scalar_idx]) -> DST[dst_idx]

// Multiply by HW-broadcast scalar (single value applied to entire tile)
mul_tiles_bcast_scalar_init_short(cb_data, cb_scalar);
mul_tiles_bcast_scalar(cb_data, cb_scalar, data_idx, scalar_idx, dst_idx);

// Add with row broadcast (broadcast rows: same row replicated across height)
add_bcast_rows_init_short(cb_data, cb_mask);
add_tiles_bcast_rows(cb_data, cb_mask, data_idx, mask_idx, dst_idx);
```

**Broadcast type semantics:**
- `BroadcastType::COL` (bcast_cols): The scalar tile's column values are broadcast -- effectively each element in a column of the scalar tile is applied to the corresponding column of the data tile. Used when the scalar is a row-reduction result (1 value per row of 32).
- `BroadcastType::ROW` (bcast_rows): The scalar tile's row values are broadcast -- used when the mask has shape [1, W] and needs to apply across all H rows.
- `BroadcastType::SCALAR`: Single value broadcast to entire tile.

### Tile Move and SFPU Operations

```cpp
// Copy tile from CB to DST register
copy_tile_to_dst_init_short(cb_src);       // or copy_tile_init(cb_src)
copy_tile(cb_src, tile_idx, dst_idx);

// Exp operation on DST register
exp_tile_init<EXP_APPROX>();  // EXP_APPROX is 0 or 1 (compile-time define)
exp_tile<EXP_APPROX>(dst_idx);

// Reciprocal on DST register
recip_tile_init();
recip_tile(dst_idx);

// Mask tile (zero out elements based on mask)
mask_tile_init();
mask_tile(dst_data, dst_mask);  // Applies mask from DST[dst_mask] to DST[dst_data]
```

### Data Format Reconfiguration

```cpp
// Reconfigure unpacker for new input data formats
reconfig_data_format(cb_srca, cb_srcb);
reconfig_data_format_srca(cb_srca);
reconfig_data_format_srcb(cb_srcb);

// Reconfigure packer for new output data format
pack_reconfig_data_format(cb_dst);

// Combined reconfig wrappers (from moreh_common.hpp)
// These add FP32_DEST_ACC_EN-conditional reconfig around standard ops:
sub_bcast_cols_init_short_with_dt(cb_data, cb_scalar);
mul_bcast_cols_init_short_with_dt(cb_data, cb_scalar);
copy_tile_init_with_dt(cb_src);
pack_tile_with_dt(dst_idx, cb_dst);
```

### DST Register Management

```cpp
tile_regs_acquire();   // Acquire exclusive access to DST registers
// ... perform compute operations writing to DST ...
tile_regs_commit();    // Signal that DST contents are ready for packing
tile_regs_wait();      // Wait until packer is ready to read DST
// ... pack_tile operations reading from DST ...
tile_regs_release();   // Release DST registers for next acquire
```

**DST register capacity** (from `dest_helpers.hpp`):
- Half-sync + FP16: 8 tiles (most common)
- Half-sync + FP32: 4 tiles
- Full-sync + FP16: 16 tiles
- Full-sync + FP32: 8 tiles

The `block_size` / `ndst` parameter in softmax is chosen to not exceed this limit: `find_max_divisor(Wt, 4)` for FP32, `find_max_divisor(Wt, 8)` for FP16.

### Circular Buffer Operations

```cpp
cb_wait_front(cb_id, num_tiles);    // Wait until num_tiles available for reading
cb_pop_front(cb_id, num_tiles);     // Release num_tiles from read side
cb_reserve_back(cb_id, num_tiles);  // Reserve space for num_tiles on write side
cb_push_back(cb_id, num_tiles);     // Signal num_tiles written
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | grid_size.x * grid_size.y (device compute grid) |
| **Total cores** | num_cores (from split_work_to_cores) |
| **Work per core** | num_tile_rows_per_core (variable between core groups) |
| **Load balancing** | Two core groups: group_1 gets ceil(total/cores) rows, group_2 gets floor(total/cores) rows |

Work is split at the **tile row** granularity (one tile row = Wt tiles). The `split_work_to_cores` utility divides `NC * Ht` tile rows across available cores.

## Arguments

### Compute Kernel Runtime Arguments (Attention-optimized softmax.cpp)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | NCHt | uint32_t | Total number of tile rows to process on this core |
| 1 | Ht | uint32_t | Height in tiles (for mask cycling) |
| 2 | Wt | uint32_t | Width in tiles (reduction dimension length) |
| 3 | ndst / block_size | uint32_t | Tiles per inner loop iteration (DST register count) |
| 4 | start_ht | uint32_t | Starting H-tile index (for mask offset calculation) |
| 5 | mask_padded_data | uint32_t | Boolean: whether last tile needs padding mask |
| 6 | cb_length | uint32_t | (Large tensor only) Max tiles per pass |

### Compute Kernel Compile-Time Arguments (moreh_softmax_w.cpp)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile rows to process on this core |
| 1 | Wt | uint32_t | Width in tiles (reduction dimension) |

### Compute Kernel Compile-Time Arguments (softmax_sharded.cpp)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_h | uint32_t | Number of tile rows to process |
| 1 | block_w | uint32_t | Width in tiles |
| 2 | subblock_w | uint32_t | Tiles per inner loop iteration |
| 3 | num_subblocks_w | uint32_t | Number of inner loop iterations (block_w / subblock_w) |

### Compile-Time Defines (All Variants)

| Define | Type | Description |
|--------|------|-------------|
| SOFTMAX | flag | When set, computes softmax (not logsoftmax) |
| LOG | flag | When set, computes logsoftmax instead of softmax |
| FUSED_SCALE_MASK | flag | Enables fused scale + attention mask path |
| CAUSAL_MASK | flag | Enables causal (triangular) attention mask |
| NUMERIC_STABLE | flag | Enables max-subtraction for numerical stability |
| EXP_APPROX | "0"/"1" | Whether to use approximate or exact exp |
| FP32_DEST_ACC_EN / ENABLE_FP32_DEST_ACC | "0"/"1" | Float32 accumulation in DST registers |

## Kernel Implementations

### moreh_softmax_w.cpp (General W-small)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input) | cb_in0, cb_mask, cb_bcast_scaler | Read Wt tiles per row, scaler tile, mask tile |
| compute | RISCV_2 | N/A | cb_in0, cb_mask, cb_bcast_scaler | cb_out0 | MAX reduce, subtract, exp, mask, SUM reduce+recip, multiply |
| writer | RISCV_1 | NOC1 | cb_out0 | DRAM (output) | Write Wt tiles per row |

**Key Logic:**
- Cleanest implementation, processes one tile at a time within each phase
- Uses `mask_tile_to_cb()` helper from moreh_common.hpp to handle the last tile's padding
- Accumulate pattern for MAX: first reduces Wt-1 tiles with WaitUpfrontNoPop, then reduces the masked last tile with `Accumulate::at(cb_max, 1)` to combine
- The cb_x_m_max intermediate (x - max) is produced as a batch of Wt tiles (`cb_reserve_back(cb_x_m_max, Wt)` before the loop, `cb_push_back` after), enabling the exp phase to stream through it
- In the non-LOG path, cb_exps persist through SUM reduce (WaitUpfrontNoPop) and the final multiply phase

### softmax.cpp (Attention-optimized)

**Key Logic:**
- Block-based processing: inner loops step by `ndst` (block_size) tiles at a time
- When FUSED_SCALE_MASK is defined, adds two extra phases before exp:
  1. Scale: `mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, ...)` -- multiply by 1/sqrt(d_k)
  2. Mask: `add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, ...)` -- add attention mask
- cb_exps serves dual purpose: in non-NUMERIC_STABLE mode, `cb_x` aliases `cb_exps` so the exp output is written directly to the buffer that the SUM reduce will read
- The attention mask (cb_fused_attn) has a special reuse pattern: for non-causal masks, it's loaded once per batch and broadcast across all Ht rows using cumulative `cb_wait_front` without popping until the batch boundary
- The `cb_wait_front(cb_fused_attn, wt + ndst)` pattern is a "cumulative wait" that avoids re-waiting for already-available tiles

### softmax_large_tensor.cpp (Multi-pass)

**Key Logic:**
- Factored into reusable helper functions: `apply_fused_scale_mask()`, `apply_fused_attn_mask()`, `pad_input()`, `exp_cb()`, `reduce_cb<PoolType>()`, `apply_recip()`
- `num_cb_passes = ceil(Wt / cb_length_t)` determines how many streaming passes are needed
- Each helper function processes `cb_length_t` tiles per invocation using block-based loops
- Ping-pong pattern between cb_max/cb_prev_max (and cb_sumexps/cb_prev_reduce) for cross-pass accumulation:
  ```cpp
  std::swap(cb_max, cb_prev_max);  // After each pass, swap so next pass reads previous result
  ```
- The `reduce_cb` template uses `compute_kernel_lib::reduce<..., NoWaitNoPop>` combined with a manual post-reduce lambda that conditionally loads the previous partial result and combines it
- Separate reciprocal computation phase: after all SUM passes complete, the sum is copied to DST, recip_tile applied, and result stored in cb_recip (c_16) -- this is NOT fused into the reduce post_reduce_op because it only happens once after all passes

## Scalar and Constant CB Setup

### cb_bcast_scaler (c_2 / c_1)
- Contains a tile filled with 1.0f values
- Required by the reduce hardware: `reduce_tile` multiplies each input tile element by the scaler before accumulation
- Loaded once by the reader at kernel start, never popped by compute (`cb_wait_front` at start, no matching `cb_pop_front`)
- For the general variant, the reader fills this:
  ```
  scaler = 1.0f  // passed as reinterpret_cast<uint32_t*>(&scaler) in runtime args
  ```

### cb_fused_scale (c_3)
- Contains a tile filled with `1/sqrt(d_k)` or user-provided scale value
- Loaded once by reader, consumed via `mul_tiles_bcast_scalar` (HW broadcast: single value applied to entire tile)
- Never popped in the compute kernel

### cb_mask / cb_mask_padded (c_1 / c_5)
- Mask tile for handling padded elements in the last tile of a row
- Contains values that zero out (or set to -inf) the padded positions
- `mask_w = logical_shape[-1] % TILE_WIDTH` determines which columns are valid
- Loaded once, persists for the entire program

## Implementation Notes

### Numerical Stability Design
The `NUMERIC_STABLE` define gates the max-subtraction path. When enabled:
1. An extra MAX reduce pass is added before the exp computation
2. The exp input becomes `exp(x - max(x))` instead of `exp(x)`
3. This prevents overflow in exp for large input values
4. Cost: one additional Wt-tile pass through the data (or extra L1 intermediate CB)

The moreh_softmax_w variant ALWAYS uses the numeric stable path (unconditionally computes max and subtracts it).

### Block Size Selection Strategy
The `find_max_divisor(Wt, max_block)` function finds the largest divisor of Wt that is <= max_block. This ensures:
- Wt is evenly divisible by block_size (no remainder handling needed)
- block_size fits within DST register capacity (4 for FP32, 8 for FP16)
- All inner loops can use uniform iteration counts

### L1 Memory Pressure and Large Tensor Fallback
The attention-optimized factory estimates total CB memory usage:
```cpp
uint32_t cb_size_sum_bytes = (in0_t * in0_tile_size) + (im0_t * im_tile_size) + (out0_t * out0_tile_size) + ...
if ((device->l1_size_per_core() * 0.9) < cb_size_sum_bytes) {
    use_large_kernel = true;  // Switch to streaming multi-pass variant
}
```
When triggered, all variable-size CBs are capped at 80 tiles, and the kernel re-reads input from DRAM multiple times.

### Relevance to LayerNorm
LayerNorm follows a very similar pattern with these differences:
1. **Two reductions** instead of one: mean (SUM) and variance (SUM of squares)
2. **No exp** operation; instead: subtract mean, square, reduce variance, rsqrt
3. **Affine parameters**: gamma and beta are additional inputs broadcast-multiplied/added
4. **The (x - mean) intermediate** must persist across both the variance computation and the final normalization, similar to how cb_exps persists across SUM reduce and final multiply in softmax

The reduce helper library, broadcast operations, and CB persistence patterns from softmax transfer directly to LayerNorm implementation.

## External Knowledge Sources

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understand the reduce helper API, input policies, accumulation patterns, and post-reduce operations
   **Key Information**: Full template signature with 6 template parameters, 4 input policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), Accumulate type for multi-pass reductions, PostReduceOp lambda pattern

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`
   **Reason**: Understand the implementation of reduce helper to verify CB management behavior
   **Key Information**: REDUCE_ROW implementation iterates over Wt tiles calling reduce_tile with indexed access; WaitUpfrontNoPop does cb_wait_front once then uses indexed access without popping; output reserve/push behavior depends on pop policy

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understand DST register capacity limits that drive block_size selection
   **Key Information**: Half-sync FP16 = 8 tiles, Half-sync FP32 = 4 tiles, Full-sync FP16 = 16, Full-sync FP32 = 8. DEST_AUTO_LIMIT constexpr auto-detects from JIT headers.

4. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understand _with_dt helper wrappers and composite operations like mask_tile_to_cb
   **Key Information**: _with_dt variants add conditional reconfig_data_format calls when FP32_DEST_ACC_EN is defined; mask_tile_to_cb combines copy_tile + mask_tile into a single helper

5. **Source**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp`
   **Reason**: Understand kernel path constants and operation parameters
   **Key Information**: SOFTMAX_KERNEL_PATH_GENERAL points to moreh_softmax kernels, SOFTMAX_KERNEL_PATH_ATTENTION points to attention-specific kernels; SoftmaxParams includes numeric_stable, scale, is_causal_mask flags
