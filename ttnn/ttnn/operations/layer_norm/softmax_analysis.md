# Softmax (W-Dimension, General) Implementation Analysis

## Overview

This analysis covers the **general-purpose softmax** operation along the W (last) dimension, implemented in two variants: **W-small** (all Wt tiles fit in L1) and **W-large** (tiles streamed one at a time, input read 3 times from DRAM). The softmax computes `y = exp(x - max(x)) / sum(exp(x - max(x)))` per row of tiles.

This analysis focuses on the **compute kernel structure** to serve as a reference for implementing a layer_norm operation with the formula `y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta`.

**Program factory files analyzed:**
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp`

**Compute kernel files:**
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp` (W-small)
- `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp` (W-large)

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | row of tiles |
| **Unit size** | Wt tiles (one tile-row along W dimension) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop over N rows assigned to this core; inner loops over Wt tiles per row |

Each core processes `num_tiles_per_core` rows, where each "row" is Wt tiles wide. The work unit is one complete tile-row because the reduction (max, sum) must see all Wt tiles before producing output.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary rank) | Same as input |
| **Dimension convention** | Last dim = W (reduction dim) | Same |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
None within the kernel. The host-side `softmax()` function calls `tilize_with_val_padding` if the input is not already in TILE_LAYOUT, padding with `-infinity`.

## Data Flow Pattern

### W-Small Variant (all Wt tiles fit in L1)

The reader loads **all Wt tiles for one row at once** into `cb_in0`. The compute kernel processes them with **indexed access** (no pop until the end of each phase). This allows multi-pass data reuse without re-reading from DRAM.

| Stage | Kernel | Reads From | Writes To | Description |
|-------|--------|------------|-----------|-------------|
| 1 | Reader | DRAM | cb_in0 (Wt tiles), cb_mask (1), cb_scaler (1) | Bulk-load one tile-row; generate mask and scaler once |
| 2 | Compute | cb_in0 | cb_max (1) | Reduce MAX across Wt tiles (row reduce) |
| 3 | Compute | cb_in0, cb_max | cb_x_m_max (Wt) | Subtract max from each tile (broadcast COL) |
| 4 | Compute | cb_x_m_max | cb_exps (Wt) | Compute exp(x - max), mask last tile |
| 5 | Compute | cb_exps | cb_recipsumexps (1) | Reduce SUM, then recip_tile for 1/sum |
| 6 | Compute | cb_exps, cb_recipsumexps | cb_out0 (Wt) | Multiply exp(x-max) * (1/sum) per tile |
| 7 | Writer | cb_out0 | DRAM | Write Wt output tiles |

### W-Large Variant (Wt tiles do not fit in L1)

The reader sends **the same row of tiles 3 times** from DRAM (3 passes). The compute kernel processes one tile at a time with streaming. This trades bandwidth for L1 space.

| Stage | Kernel | Reads From | Writes To | Description |
|-------|--------|------------|-----------|-------------|
| 1 (Pass 1) | Reader | DRAM | cb_in0 (1 tile at a time) | Stream Wt tiles for max-finding |
| 2 | Compute | cb_in0 | cb_max (1) | Reduce MAX, pop each tile after processing |
| 3 (Pass 2) | Reader | DRAM | cb_in0 (1 tile at a time) | Re-stream Wt tiles for exp computation |
| 4 | Compute | cb_in0, cb_max | cb_exps (1), cb_add (1) | sub, exp, accumulate sum tile-by-tile |
| 5 | Compute | cb_add | cb_recipsumexps (1) | Reduce SUM of accumulated tile, then recip |
| 6 (Pass 3) | Reader | DRAM | cb_in0 (1 tile at a time) | Re-stream Wt tiles for final output |
| 7 | Compute | cb_in0, cb_max, cb_recipsumexps | cb_out0 (1) | sub, exp, mul per tile |
| 8 | Writer | cb_out0 | DRAM | Write 1 output tile at a time |

## Circular Buffer Configuration

### W-Small Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (one full row) | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_mask | W-dimension mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Scaler (1.0) for reduce | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediates | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row-wise max value | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max intermediates | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary (masked tile) | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Intermediate data format**: When `fp32_dest_acc_en` is true, CBs c_24 through c_28 use `Float32` format; otherwise they match the input data format. This ensures accumulation precision.

### W-Large Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (streaming) | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_mask | W-dimension mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Scaler (1.0) for reduce | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles (streaming) | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_24 | cb_exps | exp(x-max) temporary | 2 tiles | 1 tile | Double | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_add | Running sum accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_max | Row-wise max value | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary (sub result) | 1 tile | 1 tile | Single | Compute | Compute | Block |

### Key Observations for Layer Norm

1. **Scalar/constant CBs (c_1, c_2) are loaded once** by the reader at program start and persist for the entire kernel lifetime. The compute kernel issues `cb_wait_front` on them at the top of `kernel_main()` but never pops them.
2. **Intermediate CBs that hold reduction results (cb_max, cb_recipsumexps) persist across one row** and are popped only after all Wt tiles in that row have been processed.
3. **The intermediate format** can differ from input/output format (Float32 vs BFLOAT16) for precision. This is controlled by `intermed_data_format` in the program factory.

## Pipeline Pattern Summary

### W-Small
All CBs are single-buffered (capacity equals block size). No overlap between reader and compute for the main input -- the reader writes the entire row before compute begins. The `cb_mask` and `cb_bcast_scaler` are pre-loaded constants.

### W-Large
`cb_in0` and `cb_out0` are double-buffered (capacity=2, block=1), enabling overlap between reader/compute and compute/writer on a per-tile basis. All intermediate CBs are single-buffered.

## Compute Kernel Deep Dive

### Initialization

Both kernels begin with:
```cpp
binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);
```
This initializes the unpacker and packer for binary operations. It sets up SRCA for `cb_in0`, SRCB for `cb_bcast_scaler`, and the packer for `cb_out0`.

### Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows assigned to this core |
| 1 | Wt | uint32_t | Number of tiles along W dimension (reduction width) |

### Phase 1: Find Row-Wise Maximum

**W-Small** uses `WaitUpfrontNoPop` policy so tiles remain in `cb_in0` for reuse:
```cpp
// For Wt > 1:
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

// Mask the last tile and reduce it with accumulation:
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_max, 1));
```

Key pattern: The last tile is masked (to handle padding in the W dimension) and accumulated into the existing max result using `Accumulate::at(cb_max, iteration=1)`. When `iteration > 0`, the reduce helper reloads the previously-computed partial result from `cb_max` before continuing the reduction.

**W-Large** uses default `WaitAndPopPerTile` policy (tiles are consumed):
```cpp
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_in0, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_max, 1));
```

### Phase 2: Compute x - max(x) and exp(x - max(x))

**W-Small** computes both in sequence, keeping all Wt results:

```cpp
// x - max: broadcast COL subtract (max has one value per row, broadcast to all columns)
cb_reserve_back(cb_x_m_max, Wt);
cb_wait_front(cb_in0, Wt);
cb_wait_front(cb_max, 1);

for (uint32_t w = 0; w < Wt; ++w) {
    tile_regs_acquire();
    sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);
    sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_x_m_max);
    tile_regs_release();
}
cb_pop_front(cb_max, 1);   // max consumed
cb_pop_front(cb_in0, Wt);  // input consumed
cb_push_back(cb_x_m_max, Wt);

// exp(x - max): element-wise exp, mask last tile
cb_reserve_back(cb_exps, Wt);
cb_wait_front(cb_x_m_max, Wt);
for (uint32_t w = 0; w < Wt; ++w) {
    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x_m_max);
    copy_tile(cb_x_m_max, w, dst0);
    exp_tile_init();
    exp_tile(dst0);
    if (w == Wt - 1) {
        copy_tile_init_with_dt(cb_mask);
        copy_tile(cb_mask, 0, dst1);
        mask_tile_init();
        mask_tile(dst0, dst1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exps);
    tile_regs_release();
}
cb_push_back(cb_exps, Wt);
```

**W-Large** fuses subtraction and exp into one pass, accumulating the sum tile-by-tile:
```cpp
for (uint32_t w = 0; w < Wt; ++w) {
    // sub_tiles_bcast_cols_to_cb: cb_in0[0] - cb_max[0] -> cb_tmp, pop input
    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);

    if (w == Wt - 1) {
        exp_tile_and_mask_tile_to_cb(cb_tmp, cb_mask, cb_exps, 0, 0, 1, 0);
    } else {
        exp_tile_to_cb(cb_tmp, cb_exps);
    }

    // Accumulate: first tile copied, subsequent tiles added
    if (w == 0) {
        copy_tile_to_cb(cb_exps, cb_add);
    } else {
        add_tiles_to_cb(cb_add, cb_exps, cb_add);
    }
}
```

### Phase 3: Compute 1/sum(exp)

**W-Small** uses `WaitUpfrontNoPop` to keep `cb_exps` tiles available for Phase 4:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

The **post-reduce lambda** `recip_tile` is applied in-place in the DST register before packing to the output CB. This avoids a separate CB for the sum -- the reciprocal is computed directly after the reduction completes.

**W-Large** uses `BulkWaitBulkPop` since the sum is already accumulated in a single tile:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
    cb_add, cb_bcast_scaler, cb_recipsumexps,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

### Phase 4: Final Output (exp(x-max) * (1/sum))

**W-Small** reads from persistent `cb_exps` and broadcasts `cb_recipsumexps`:
```cpp
cb_reserve_back(cb_out0, Wt);
cb_wait_front(cb_x_m_max, Wt);      // still in CB from Phase 2
cb_wait_front(cb_recipsumexps, 1);
cb_wait_front(cb_exps, Wt);          // still in CB (WaitUpfrontNoPop)

for (uint32_t w = 0; w < Wt; w++) {
    tile_regs_acquire();
    mul_bcast_cols_init_short_with_dt(cb_exps, cb_recipsumexps);
    mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_out0);
    tile_regs_release();
}

cb_pop_front(cb_recipsumexps, 1);
cb_pop_front(cb_x_m_max, Wt);
cb_push_back(cb_out0, Wt);
cb_pop_front(cb_exps, Wt);
```

**W-Large** recomputes exp(x-max) from the third pass of input tiles:
```cpp
for (uint32_t w = 0; w < Wt; w++) {
    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);
    exp_tile_to_cb(cb_tmp, cb_exps);
    mul_tiles_bcast_cols_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
}
cb_pop_front(cb_recipsumexps, onetile);
cb_pop_front(cb_max, onetile);
```

## Broadcast Operation Patterns

The softmax kernel uses two key broadcast patterns, both critical for layer_norm:

### BroadcastType::COL (sub_tiles_bcast / mul_tiles_bcast_cols)
- **What it does**: Takes a scalar tile (1 value per row of the 32x32 tile) and broadcasts it across all columns
- **Use in softmax**: `x - max(x)` and `exp(x) * (1/sum)` -- the max and reciprocal-sum are per-row scalars
- **Init + execute pattern**:
  ```cpp
  sub_bcast_cols_init_short_with_dt(cb_data, cb_scalar);  // reconfigure unpacker
  sub_tiles_bcast<BroadcastType::COL>(cb_data, cb_scalar, data_tile_idx, scalar_tile_idx, dst_reg);
  ```
- **Relevance to layer_norm**: Mean and variance are per-row scalars; the same COL broadcast pattern applies for `(x - mean)` and `normalized * gamma`

### Reduce with REDUCE_ROW
- **What it does**: Reduces all tiles across the W dimension into a single tile per row
- **Use in softmax**: Computing max(x) and sum(exp(x))
- **Relevance to layer_norm**: Computing mean(x) = sum(x)/N and var(x) = sum((x-mean)^2)/N

## Multi-Pass Data Reuse Patterns

### W-Small: CB Persistence via WaitUpfrontNoPop

The key insight is that `WaitUpfrontNoPop` tells the reduce helper to `cb_wait_front` for all tiles but **never pop them**. This keeps tiles resident in the CB for subsequent phases:

1. **cb_in0 persists through Phase 1 and Phase 2**: MAX reduce uses `WaitUpfrontNoPop`, so tiles stay for the subtraction loop
2. **cb_exps persists through Phase 3 and Phase 4**: SUM reduce uses `WaitUpfrontNoPop`, so exp tiles stay for the final multiplication
3. **cb_x_m_max persists through Phase 2-4** (in LOG mode) or is available for exp computation

**For layer_norm**: This pattern directly applies. After computing mean via `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>`, the input tiles remain in the CB for the subsequent `x - mean` subtraction.

### W-Large: Triple-Read from DRAM

When tiles do not fit in L1, the reader kernel reads the same row from DRAM 3 times:
```cpp
// Pass 1: for max computation
curr_tile = curr_offset_i;
for (w...) { read tile, push 1; }

// Pass 2: for exp and sum
curr_tile = curr_offset_i;  // reset to same offset
for (w...) { read tile, push 1; }

// Pass 3: for final output
curr_tile = curr_offset_i;  // reset again
for (w...) { read tile, push 1; }
```

**For layer_norm**: You would need similarly 2-3 passes (mean, variance, normalization) if the row does not fit in L1.

## Scalar/Constant CB Setup

The reader kernel generates constant tiles **programmatically in L1** (no DRAM read):

```cpp
// Scaler tile: all elements set to 1.0 (used as reduce scaler)
generate_bcast_scaler<uint16_t>(cb_scaler, scaler);  // scaler = bit-cast of 1.0f

// Mask tile: ones for valid columns, zeros for padding columns
generate_mask_w<uint16_t>(cb_mask, mask_w);  // mask_w = logical_shape[-1] % TILE_WIDTH
```

These are pushed once and never popped by compute (persistent for entire kernel lifetime).

**For layer_norm**: You would need:
- A scaler tile with value `1.0/Wt` (or `1.0` and divide later) for mean computation
- An epsilon tile (or fold epsilon into the rsqrt computation)
- Potentially a mask tile if the last dimension is not tile-aligned

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (full device compute grid) |
| **Total cores** | Up to `grid_x * grid_y` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` rows |
| **Load balancing** | Two core groups: group_1 gets ceil(rows/cores), group_2 gets floor |

Work splitting via `split_work_to_cores_wt_core_range`:
- Returns `(num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2)`
- Core traversal: `core = {i / core_h, i % core_h}` (column-major)
- Tile offset accumulation: `tile_offset += num_tiles_per_core * Wt` (each row is Wt tiles)

## Arguments

### Compute Kernel Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_core | uint32_t | Number of tile-rows this core processes (N) |
| 1 | Wt | uint32_t | Tiles along W dimension (reduction width) |

### Compute Kernel Defines

| Define | Value | Description |
|--------|-------|-------------|
| `SOFTMAX` | `"1"` | Enables softmax path (vs softmin which uses negative_tile before exp) |
| `FP32_DEST_ACC_EN` | `"1"` (conditional) | Enables FP32 accumulation in DST registers |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer address |
| 1 | num_tiles_per_core | uint32_t | Number of tile-rows (N) |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles along W |
| 4 | scaler | uint32_t | Bit-cast float (1.0f) for reduce scaler |
| 5 | mask_w | uint32_t | Number of valid elements in last W tile |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_tiles_per_core | uint32_t | Number of tile-rows (N) |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles along W |

## Kernel Implementations

### Compute Kernel (W-Small)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| moreh_softmax_w.cpp | Compute (TRISC) | N/A | cb_in0, cb_mask, cb_scaler | cb_out0 | MAX reduce, sub_bcast_col, exp, SUM reduce, recip, mul_bcast_col |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- **Key Logic**: All Wt tiles loaded into L1 at once. Uses indexed tile access (`cb_in0` tile index `w`) and `WaitUpfrontNoPop` policy for tile reuse across phases.

### Compute Kernel (W-Large)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| moreh_softmax_w_large.cpp | Compute (TRISC) | N/A | cb_in0, cb_mask, cb_scaler | cb_out0 | MAX reduce, sub_bcast_col, exp, mask, tile-add accumulate, SUM reduce, recip, mul_bcast_col |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`
- **Key Logic**: Processes one tile at a time from streaming `cb_in0`. Uses `cb_add` as a running accumulator for the sum (manual `add_tiles_to_cb` loop instead of bulk reduce). Reader sends the same data 3 times.

### Reader/Writer (Summary Only)

- **Reader** (both variants): Generates `cb_mask` and `cb_scaler` programmatically in L1, then reads input tiles from DRAM via TensorAccessor. W-small loads Wt tiles per push; W-large loads 1 tile per push (3 passes).
- **Writer** (both variants): Drains `cb_out0` to DRAM. W-small writes Wt tiles per row; W-large writes 1 tile at a time.

## Helper Function Signatures Reference

### From moreh_common.hpp (compute helpers)

```cpp
// Subtraction with column broadcast: data[tile_idx] - scalar[0], broadcast scalar across columns
ALWI void sub_tiles_bcast_cols_to_cb(
    uint32_t icb0, uint32_t icb1, uint32_t ocb,
    uint32_t itile0 = 0, uint32_t itile1 = 0,
    uint32_t pop0 = 1, uint32_t pop1 = 1);

// Multiplication with column broadcast: data[tile_idx] * scalar[0]
ALWI void mul_tiles_bcast_cols_to_cb(
    uint32_t icb0, uint32_t icb1, uint32_t ocb,
    uint32_t itile0 = 0, uint32_t itile1 = 0,
    uint32_t pop0 = 1, uint32_t pop1 = 1);

// Exponentiate tile and write to CB
ALWI void exp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1);

// Copy tile between CBs
ALWI void copy_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1);

// Add two tiles (element-wise) and write to CB
ALWI void add_tiles_to_cb(uint32_t icb0, uint32_t icb1, uint32_t ocb,
    uint32_t itile0 = 0, uint32_t itile1 = 0, uint32_t pop0 = 1, uint32_t pop1 = 1);

// Mask a tile (zero out padding elements)
ALWI void mask_tile_to_cb(uint32_t icb, uint32_t maskcb, uint32_t ocb,
    uint32_t itile = 0, uint32_t mtile = 0, uint32_t pop = 1, uint32_t popm = 1);

// FP32-aware packing/unpacking
ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb);
ALWI void copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0);
ALWI void sub_bcast_cols_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1);
ALWI void mul_bcast_cols_init_short_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1);
```

### From reduce_helpers_compute.hpp

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t input_cb,
    uint32_t scaler_cb,
    uint32_t output_cb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{});
```

Key types:
- `ReduceInputBlockShape::row(Wt)` -- reduce a single row of Wt tiles
- `ReduceInputBlockShape::single()` -- reduce a single tile
- `ReduceInputPolicy::WaitUpfrontNoPop` -- wait for tiles, do not pop (reuse)
- `ReduceInputPolicy::BulkWaitBulkPop` -- wait for all, process, pop all
- `Accumulate::at(cb, iteration)` -- reload from cb when iteration > 0
- Post-reduce lambda: `[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }`

## Implementation Notes

### Design Choices Relevant to Layer Norm

1. **Small vs Large dispatch**: The host checks if total CB memory fits in L1 (< 512KB). This same pattern should be used for layer_norm -- a "small" variant that holds the entire row in L1 (enabling multi-pass reuse) and a "large" variant that streams tiles and re-reads from DRAM.

2. **The `_with_dt` suffix pattern**: All compute helpers have `_with_dt` variants that call `reconfig_data_format` when `FP32_DEST_ACC_EN` is defined. This is essential for correct data movement when intermediate CBs use Float32 but input/output use BFLOAT16.

3. **DST register protocol**: Every compute operation follows `tile_regs_acquire` -> compute -> `tile_regs_commit` -> `tile_regs_wait` -> pack -> `tile_regs_release`. The moreh_common.hpp helpers encapsulate this.

4. **Reduction scaler**: The reduce library expects the scaler tile to contain the scaling factor. For softmax this is 1.0 (identity). For layer_norm mean computation, this could be `1.0/W` to compute the mean directly during reduction, or 1.0 with a subsequent division.

5. **Mask handling**: The W-mask zeros out padding elements in the last tile. For layer_norm, if the logical width is not a multiple of TILE_WIDTH (32), the same masking approach is needed to avoid including padding in mean/variance calculations.

6. **binary_op_init_common**: Called at kernel start to initialize unpacker/packer hardware. Takes input CB, secondary CB, and output CB as arguments.

### Mapping Softmax Phases to Layer Norm Phases

| Softmax Phase | Layer Norm Equivalent | Notes |
|---------------|----------------------|-------|
| max(x) via REDUCE_ROW | mean(x) via REDUCE_ROW with scaler=1/W | Use SUM instead of MAX; scaler tile = 1/W |
| x - max (sub_bcast_col) | x - mean (sub_bcast_col) | Identical pattern |
| exp(x - max) | (x - mean)^2 | Use mul_tiles (self-multiply) instead of exp |
| sum(exp) via REDUCE_ROW | sum((x-mean)^2) / W = variance | SUM reduce with scaler=1/W or post-divide |
| 1/sum via recip_tile | 1/sqrt(var+eps) via rsqrt_tile | Use rsqrt as post-reduce op; add eps first |
| exp * (1/sum) via mul_bcast_col | (x-mean) * rsqrt(var+eps) via mul_bcast_col | Same broadcast multiply pattern |
| (none) | * gamma + beta | Additional mul_bcast + add_bcast per element |

## External Knowledge Sources

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the unified reduce function API, input policies, accumulation patterns, and post-reduce operations
   **Key Information**: Complete function signature with all template parameters, ReduceInputPolicy semantics (WaitUpfrontNoPop for tile reuse, BulkWaitBulkPop for performance), Accumulate pattern for multi-block reductions, post-reduce lambda for fused operations like recip_tile

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding the helper functions used by the compute kernel (sub_tiles_bcast_cols_to_cb, exp_tile_to_cb, etc.)
   **Key Information**: All helpers follow the acquire/compute/commit/wait/pack/release DST protocol. The `_with_dt` variants handle FP32 data format reconfiguration. The helpers manage CB wait/pop internally with configurable pop counts.

3. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp`
   **Reason**: Understanding host-side utilities for CB creation, compute kernel creation, and work splitting
   **Key Information**: `CreateCircularBuffer` takes a vector of `CircularBufferArg{index, num_tiles, data_format}`. `split_work_to_cores_wt_core_range` returns two core groups for load balancing. `CreateComputeKernel` takes `ComputeKernelArg{core_spec, num_tiles_per_core, compile_args}`.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp`
   **Reason**: Understanding small vs large variant selection and L1 memory budget calculation
   **Key Information**: The `is_softmax_general_w_small_available` function sums all CB memory requirements and checks against 512KB L1 threshold. This pattern should be replicated for layer_norm dispatch.
