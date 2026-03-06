# Softmax (W-Small) Implementation Analysis

## Overview

This analysis covers the **softmax W-small** variant, which computes numerically-stable softmax along the W (width/innermost) dimension of a tiled tensor. The "W-small" designation means the entire W-tile-row (`Wt` tiles) fits in L1 simultaneously, enabling multi-pass data reuse without re-reading from DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`

**Compute kernel path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Mathematical formula** (SOFTMAX variant, `#define SOFTMAX` is set):

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This analysis focuses on the **compute kernel structure** as a reference for implementing layer_norm_rm.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row (1 x Wt tiles) |
| **Unit size** | Wt tiles (all tiles along one row of one tile-row) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer: N kernel rows assigned to this core; Inner: 4-phase compute pipeline per row |

One "work unit" is a single tile-row: the Wt tiles that span the W dimension for one row of one batch element. Each core processes N such tile-rows, where N is determined by the work splitting.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] | [..., H, W] |
| **Dimension convention** | Last 2 dims are H, W | Same |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
None. Input and output share the same tile layout, memory layout, and data type. No tilize/untilize is performed.

---

## Data Flow Pattern (Compute-Centric)

The compute kernel processes one tile-row at a time in a 4-phase pipeline. The key insight is that **intermediate CBs persist across phases within one tile-row**, enabling multi-pass data reuse.

### Phase 1: Find max(x) -- reduce across W dimension

**Purpose**: Compute `max(x)` for the entire tile-row for numerical stability.

Two cases based on Wt:

**Case Wt == 1**:
1. `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, pop0=0, popm=0)` -- Masks the single input tile and writes to `cb_tmp`. Does NOT pop `cb_in0` or `cb_mask` (both kept for reuse).
2. `compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_bcast_scaler, cb_max, ReduceInputBlockShape::single())` -- Row-reduces the masked tile to get the max, output to `cb_max`.

**Case Wt > 1**:
1. `compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(cb_in0, cb_bcast_scaler, cb_max, ReduceInputBlockShape::row(Wt - 1))` -- Reduces the first Wt-1 tiles using `WaitUpfrontNoPop` policy (tiles remain in `cb_in0` for later reuse). The output `cb_max` holds the partial max.
2. `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt-1, 0, pop0=0, popm=0)` -- Masks the last tile. Does NOT pop inputs.
3. `compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_bcast_scaler, cb_max, ..., Accumulate::at(cb_max, 1))` -- Reduces the masked last tile, **accumulating** with the previous partial max from `cb_max`. The `iteration=1` causes reload from `cb_max` before reducing.

**Key CB state after Phase 1**: `cb_in0` still holds all Wt input tiles (not popped). `cb_max` holds 1 tile. `cb_mask` still holds mask tile. `cb_bcast_scaler` still holds scaler tile.

### Phase 2: Compute x - max(x) -- broadcast subtract

**Purpose**: Subtract the max from every input tile (column broadcast).

```cpp
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
cb_pop_front(cb_max, 1);
cb_pop_front(cb_in0, Wt);
cb_push_back(cb_x_m_max, Wt);
```

**Broadcast pattern**: `BroadcastType::COL` means the first **column** of the `cb_max` tile is broadcast across all columns. Since `cb_max` is a row-reduction result, column 0 of each row holds the max for that row; COL broadcast replicates this across all 32 columns. Each input tile at index `w` is subtracted with this broadcast max.

**Key CB state after Phase 2**: `cb_in0` is now popped (freed). `cb_max` is popped. `cb_x_m_max` holds Wt tiles of (x - max).

### Phase 3: Compute exp(x - max(x)) and sum, then 1/sum

**Purpose**: Exponentiate, handle masking on last tile, reduce sum, then compute reciprocal.

```cpp
cb_reserve_back(cb_exps, Wt);
cb_wait_front(cb_x_m_max, Wt);
for (uint32_t w = 0; w < Wt; ++w) {
    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x_m_max);
    copy_tile(cb_x_m_max, w, dst0);
    // (#ifdef SOFTMAX path -- no negation)
    exp_tile_init();
    exp_tile(dst0);
    if (w == Wt - 1) {  // mask last tile
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

**Masking**: On the last tile (`w == Wt-1`), a mask tile is loaded into `dst1` and `mask_tile(dst0, dst1)` zeros out padding columns. The `mask_tile` function uses `dst0+1` internally for the mask, so `dst1` must be exactly `dst0 + 1`.

Then compute `1/sum(exp(x - max))`:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps,
    ReduceInputBlockShape::row(Wt), ..., NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

**Important**: Uses `WaitUpfrontNoPop` so `cb_exps` tiles persist for Phase 4. The `post_reduce_op` lambda applies `recip_tile` to compute 1/sum in-place in the DST register before packing to `cb_recipsumexps`.

**Key CB state after Phase 3**: `cb_x_m_max` still holds Wt tiles (not popped by compute -- needed for LOG variant but popped in Phase 4). `cb_exps` holds Wt tiles (not popped). `cb_recipsumexps` holds 1 tile.

### Phase 4: Compute final output = exp(x - max) * (1/sum)

**Purpose**: Multiply each exp tile by the reciprocal of the sum (column broadcast).

```cpp
cb_reserve_back(cb_out0, Wt);
cb_wait_front(cb_x_m_max, Wt);
cb_wait_front(cb_recipsumexps, 1);
cb_wait_front(cb_exps, Wt);

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

**Broadcast pattern**: `mul_tiles_bcast_cols` uses `BroadcastType::COL` -- the reciprocal-sum tile's column values are broadcast across all columns. Each `cb_exps` tile at index `w` is multiplied elementwise with this broadcasted 1/sum tile.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (one tile-row) | Wt tiles | Wt tiles | Single | Reader | Compute (Phase 1, 2) | Row |
| c_1 | cb_mask | Width mask tile (1s/0s for padding) | 1 tile | 1 tile | Single | Reader (once) | Compute (Phase 1, 3) | Program |
| c_2 | cb_bcast_scaler | Reduce scaler tile (all 1.0) | 1 tile | 1 tile | Single | Reader (once) | Compute (all phases) | Program |
| c_16 | cb_out0 | Output tiles (one tile-row) | Wt tiles | Wt tiles | Single | Compute (Phase 4) | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute (Phase 3) | Compute (Phase 3 reduce, Phase 4) | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar result | 1 tile | 1 tile | Single | Compute (Phase 3) | Compute (Phase 4) | Row |
| c_26 | cb_max | max(x) scalar result | 1 tile | 1 tile | Single | Compute (Phase 1) | Compute (Phase 2) | Row |
| c_27 | cb_x_m_max | x - max(x) intermediate | Wt tiles | Wt tiles | Single | Compute (Phase 2) | Compute (Phase 3, 4) | Row |
| c_28 | cb_tmp | Masked tile scratch | 1 tile | 1 tile | Single | Compute (Phase 1) | Compute (Phase 1 reduce) | Row |

### Data Format Notes
- **c_0, c_1, c_2, c_16**: Use `data_format` (matches input tensor dtype -- BFLOAT16 or FLOAT32).
- **c_24, c_25, c_26, c_27, c_28**: Use `intermed_data_format` which is FLOAT32 if `fp32_dest_acc_en`, otherwise same as input. This ensures intermediate precision is maintained for numerically sensitive computations.

---

## Multi-Pass Data Reuse Patterns (Critical for layer_norm_rm)

This is the most important section for understanding how to design a layer_norm_rm compute kernel.

### CBs That Persist Across Phases

| CB | Persists From | Persists Until | Why |
|----|---------------|----------------|-----|
| **cb_in0** (c_0) | Reader push | End of Phase 2 | Reused in Phase 1 (reduce MAX) and Phase 2 (subtract). The `WaitUpfrontNoPop` policy in Phase 1 keeps tiles alive. Explicit `cb_pop_front(cb_in0, Wt)` at end of Phase 2. |
| **cb_mask** (c_1) | Reader push (once) | End of program | Never popped. Used in Phase 1 (mask for max reduce) and Phase 3 (mask for exp). `pop=0` / `popm=0` in all calls. |
| **cb_bcast_scaler** (c_2) | Reader push (once) | End of program | Never popped. Used by every `reduce` call across all phases and all iterations of N. |
| **cb_exps** (c_24) | Phase 3 push | End of Phase 4 | `WaitUpfrontNoPop` in sum-reduce keeps tiles alive. Phase 4 reads them, then explicit `cb_pop_front(cb_exps, Wt)`. |
| **cb_x_m_max** (c_27) | Phase 2 push | End of Phase 4 | Used in Phase 3 (exp computation reads via indexed access) and conceptually present in Phase 4. Explicit pop at end of Phase 4. |

### Reuse Pattern: "Write Once, Read Multiple Phases"

The key pattern is:
1. **Reader writes once** per tile-row to `cb_in0` (Wt tiles).
2. **Compute reads `cb_in0` twice**: First via `WaitUpfrontNoPop` reduce in Phase 1, then via explicit `cb_wait_front` + indexed `sub_tiles_bcast` in Phase 2.
3. **`cb_in0` is freed** only after Phase 2 completes, by a single `cb_pop_front(cb_in0, Wt)`.

This pattern is directly applicable to layer_norm_rm, where the input tile-row must be read for:
- Phase 1: reduce to compute mean
- Phase 2: subtract mean
- Phase 3: square (for variance)
- Phase 4: reduce to compute variance
- Phase 5: normalize (or re-read from intermediate)

### Constant CBs: "Write Once, Never Pop"

`cb_mask` and `cb_bcast_scaler` are written by the reader kernel once at the start and never popped across the entire program lifetime:
- **cb_bcast_scaler**: Generated by `generate_bcast_scaler<T>(cb_scaler, scaler)` in the reader. Fills a tile where the first column of each face (indices 0..15 of each 256-element face) contains the scaler value (here 1.0), rest zeros. This is the standard reduce scaler format.
- **cb_mask**: Generated by `generate_mask_w<T>(cb_mask, mask_w)` in the reader. Creates a tile where active width positions are 1.0 and padding positions are 0.0.

The compute kernel does `cb_wait_front(cb_mask, onetile)` and `cb_wait_front(cb_bcast_scaler, onetile)` at the top, before the main loop. These tiles are never consumed (no pop), so subsequent iterations reuse them automatically.

---

## Scalar/Constant CB Setup Details

### Broadcast Scaler (cb_bcast_scaler, c_2)

**Generated by**: `generate_bcast_scaler<T>(cb_scaler, scaler)` in reader kernel.

**Scaler value**: 1.0 (passed as `float scaler = 1.0f`, reinterpreted to uint32_t).

**Tile layout**: A 32x32 tile organized as 4 faces of 16x16 elements. For each face `k` (0-3), the first 16 elements (indices `k*256 + 0` through `k*256 + 15`) contain the scaler value; remaining elements are zero. This format matches what the `reduce_tile` hardware expects for the SRCB operand.

**Why 1.0**: For MAX and SUM reductions, the scaler multiplies each column's contribution. A scaler of 1.0 means "no scaling" -- the reduction sums or finds max of the actual values.

### Width Mask (cb_mask, c_1)

**Generated by**: `generate_mask_w<T>(cb_mask, mask_w)` in reader kernel.

**mask_w calculation**: `mask_w = logical_shape[-1] % TILE_WIDTH`, or `TILE_WIDTH` if evenly divisible.

**Tile layout**: Each row has `mask_w` positions set to 1.0 and remaining positions set to 0.0. Handles the subtile structure (4 faces: top-left, top-right, bottom-left, bottom-right).

**Usage**: Passed to `mask_tile(dst0, dst1)` which zeroes out elements where the mask is 0, handling partial tiles at the right edge of the tensor.

---

## Index Calculations

### Tile Indexing in Compute Kernel

Within the compute kernel, tiles are accessed via their position in the circular buffer:
- `cb_in0` holds Wt tiles at indices 0 through Wt-1
- `sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0)` reads tile at index `w` from `cb_in0` and tile at index 0 from `cb_max`
- `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)` reads tile at index `w` from `cb_exps` and tile at index 0 from `cb_recipsumexps`

### Reader Tile Addressing

The reader uses `TensorAccessor` for DRAM addressing:
- `tile_offset`: Starting tile index for this core (computed from work splitting)
- For each row of N: reads Wt tiles sequentially from `curr_tile` through `curr_tile + Wt - 1`
- `curr_tile` advances by Wt per row

---

## Memory Access Patterns

### Read Pattern
- **Ordering**: Sequential tile reads within each tile-row (tiles 0..Wt-1)
- **Granularity**: One tile per NoC read
- **Barrier**: `noc_async_read_barrier()` after all Wt tiles in a row
- **Reuse**: Input tiles read once from DRAM, then reused in L1 across compute phases

### Write Pattern
- **Ordering**: Sequential tile writes within each tile-row
- **Granularity**: One tile per NoC write
- **Barrier**: `noc_async_write_barrier()` after all Wt tiles in a row

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (device max) |
| **Total cores** | num_cores (up to grid_x * grid_y) |
| **Work per core** | N tile-rows (each N tiles covers Wt actual tiles) |
| **Load balancing** | Two-group: core_group_1 gets `ceil(num_kernel_rows/num_cores)`, core_group_2 gets `floor(num_kernel_rows/num_cores)` |

Work splitting uses `split_work_to_cores_wt_core_range`:
- `num_kernel_rows = (physical_volume / H / W) * Ht` -- total tile-rows across all batch elements
- Core linearization: `core = {(i / core_h) + x_offset, (i % core_h) + y_offset}` -- column-major ordering
- `tile_offset` advances by `num_tiles_per_core * Wt` for each core (tile_offset tracks tiles, not tile-rows)

---

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows this core processes |
| 1 | Wt | uint32_t | Number of tiles along W dimension per tile-row |

These are passed via `CreateComputeKernel` as `{num_tiles_per_core_group_X, Wt}`.

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input is FLOAT32, 0 otherwise |
| 1+ | TensorAccessor args | uint32_t | Appended by `TensorAccessorArgs(*input_tensor.buffer())` |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles per tile-row |
| 4 | scaler | uint32_t | Float 1.0 reinterpreted as uint32_t |
| 5 | mask_w | uint32_t | Active width within last tile (1..32) |

### Runtime Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles per tile-row |

### Compute Defines

| Define | Value | Description |
|--------|-------|-------------|
| `SOFTMAX` | `"1"` | Selects softmax path (vs logsoftmax). Controls whether `negative_tile` is called before `exp_tile` and whether output uses `mul * recip` vs `sub - log` |
| `FP32_DEST_ACC_EN` | `"1"` (conditional) | Enables FP32 destination accumulation. Set when `fp32_dest_acc_en` is true. Affects `_with_dt` helper behavior. |

---

## Kernel Implementations

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (unpack+math+pack threads) | N/A | cb_in0, cb_mask, cb_bcast_scaler | cb_out0 | 4-phase: MAX reduce, SUB bcast, EXP+mask+SUM reduce+RECIP, MUL bcast |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`
- **Includes**: `reduce_helpers_compute.hpp`, `moreh_common.hpp` (compute variant)
- **Initialization**: `binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0)` -- configures unpack/math/pack hardware for binary operations

### Key Compute API Calls (Exact Signatures)

**Phase 1 -- MAX reduce with accumulation**:
```cpp
// First Wt-1 tiles, keep in CB
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

// Mask last tile
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt-1, 0, /*pop0=*/0, /*popm=*/0);

// Reduce masked tile, accumulate with previous max
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_max, 1));
```

**Phase 2 -- SUB with column broadcast**:
```cpp
sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);
sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
pack_tile_with_dt(dst0, cb_x_m_max);
```

**Phase 3 -- EXP + mask + SUM reduce + RECIP**:
```cpp
// Per tile: copy, exp, conditionally mask
copy_tile_init_with_dt(cb_x_m_max);
copy_tile(cb_x_m_max, w, dst0);
exp_tile_init();
exp_tile(dst0);
// Last tile only:
mask_tile_init();
mask_tile(dst0, dst1);

// SUM reduce with post-reduce reciprocal
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

**Phase 4 -- MUL with column broadcast**:
```cpp
mul_bcast_cols_init_short_with_dt(cb_exps, cb_recipsumexps);
mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
pack_tile_with_dt(dst0, cb_out0);
```

### Reader Kernel (Summary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM input | cb_in0, cb_mask, cb_bcast_scaler | Read tiles, generate mask/scaler |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`
- **Key responsibility**: Generates `cb_bcast_scaler` and `cb_mask` once, then reads Wt input tiles per tile-row for N iterations.

### Writer Kernel (Summary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | cb_out0 | DRAM output | Write tiles |

- **File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`
- **Key responsibility**: Waits for Wt output tiles per tile-row, writes sequentially to DRAM.

---

## Pipeline Pattern Summary

All CBs are **single-buffered** (capacity == block size for row-scoped CBs). This means:
- Reader and compute cannot overlap within the same tile-row (reader must finish pushing Wt tiles before compute can start)
- Compute and writer cannot overlap within the same tile-row (compute must finish pushing Wt output tiles before writer starts)
- The pipeline operates in a **row-level batch** pattern: reader produces a full row, compute consumes it through all 4 phases, then writer drains the output

This is acceptable for the "W-small" variant where Wt is small enough to fit entirely in L1.

---

## Implementation Notes

### FP32 Dest Accumulation
When `fp32_dest_acc_en` is true:
- `FP32_DEST_ACC_EN` define is set for the compute kernel
- All `_with_dt` helpers (e.g., `pack_tile_with_dt`, `copy_tile_init_with_dt`) call `pack_reconfig_data_format` / `reconfig_data_format_srca` before their respective operations
- Intermediate CBs use FLOAT32 data format regardless of input type
- This is required when input dtype is FLOAT32

### Conditional Compilation: SOFTMAX vs LOG
The compute kernel supports both softmax and logsoftmax via `#ifdef SOFTMAX`:
- **SOFTMAX** (this variant): `exp(x-max) * (1/sum)` -- multiply by reciprocal
- **LOG** (not analyzed): `(x - max) - log(sum)` -- subtract log of sum. In LOG mode, `cb_exps` are popped after the sum reduce (not needed for final computation), and `cb_x_m_max` is used in the final subtraction instead.

### mask_tile Implementation Detail
The `mask_tile(dst0, dst1)` function has a documented quirk: internally it uses `dst0 + 1` for the mask regardless of what `dst1` is passed. This means the mask tile MUST be loaded into exactly `dst0 + 1`. The kernel correctly uses `dst0 = 0, dst1 = 1`.

### Reader Generates Constants, Not Compute
The scaler and mask tiles are generated entirely in the reader kernel (data movement thread), not in the compute kernel. This is a design pattern that offloads constant generation to the data movement cores, keeping the compute cores focused on math.

---

## Relevance to Layer Norm RM

The softmax W-small variant demonstrates several patterns directly applicable to a layer_norm_rm operation:

1. **Multi-phase compute with CB persistence**: Input tiles read once, used across multiple computation phases (mean, variance, normalize). Same `WaitUpfrontNoPop` pattern for reduce + subsequent reuse.

2. **Row-reduce pattern**: `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` with `ReduceInputBlockShape::row(Wt)` is exactly what layer_norm needs for computing mean and variance along W.

3. **Column-broadcast subtract/multiply**: `sub_tiles_bcast<BroadcastType::COL>` for subtracting a scalar result from each tile, `mul_tiles_bcast_cols` for multiplying -- needed for `(x - mean) * rsqrt(var + eps)` and `* gamma + beta`.

4. **Post-reduce operations**: The `recip_tile` lambda in reduce is analogous to what layer_norm would use for `rsqrt_tile` after computing variance.

5. **Constant CB setup**: `generate_bcast_scaler` pattern for the reduce scaler (1/W for mean, or 1.0 for sum).

6. **Intermediate data format**: Using FLOAT32 intermediates when `fp32_dest_acc_en` is set -- important for numerical accuracy in normalization.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do binary_op_init_common, sub_bcast_cols_init_short, sub_tiles_bcast<BroadcastType::COL>, mul_bcast_cols_init_short, and mul_tiles_bcast_cols work in compute kernels?"
   **Reason**: Needed to understand broadcast binary operations used in Phases 2 and 4.
   **Key Findings**: `BroadcastType::COL` broadcasts the first column of operand B across all columns; `BroadcastType::ROW` broadcasts a row; `BroadcastType::SCALAR` broadcasts element [0,0]. `binary_op_init_common` configures unpack/math/pack hardware for binary ops.

2. **Query**: "How does tile_regs_acquire/tile_regs_commit/tile_regs_wait/tile_regs_release work? What is the DST register file?"
   **Reason**: Needed to understand the synchronization model between unpack, math, and pack threads.
   **Key Findings**: DST is the shared register file. `acquire` gives math ownership, `commit` transfers to pack, `wait` blocks pack until data ready, `release` frees for next iteration. This is the standard 4-call pattern around every math+pack operation.

3. **Query**: "How does split_work_to_cores_wt_core_range work?"
   **Reason**: Needed to understand core distribution.
   **Key Findings**: Returns 6-tuple: (num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2). group_1 gets ceil(work/cores), group_2 gets floor(work/cores). Applies coordinate offset from core_range start.

4. **Query**: "What does mask_tile_init() and mask_tile(dst0, dst1) do?"
   **Reason**: Needed to understand the masking mechanism for partial tiles.
   **Key Findings**: `mask_tile` zeroes elements where mask is 0. Internal implementation uses `dst_data + 1` for mask, so mask must be in consecutive DST register. `generate_mask_w` creates the 1s/0s mask tile in the reader.

5. **Query**: "How does CreateComputeKernel moreh helper work?"
   **Reason**: Needed to understand how compile-time args are passed to compute kernels.
   **Key Findings**: Takes vector of `{core_range_set, num_tiles_per_core, compile_args_vector}` tuples. Each core group can have different compile-time args. The `compile_args` are baked into the kernel binary at JIT-compile time.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Primary source for understanding the `compute_kernel_lib::reduce` API.
   **Key Information**: Full template signature, input policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), accumulation support, post-reduce operation callback, and data format reconfiguration modes.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Contains all `_with_dt` helper wrappers and `mask_tile_to_cb`, `copy_tile_to_cb`, etc.
   **Key Information**: Every `_with_dt` wrapper conditionally calls `reconfig_data_format`/`pack_reconfig_data_format` when `FP32_DEST_ACC_EN` is defined. This is critical for correct data format handling when mixing BFLOAT16/FLOAT32.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Contains `generate_bcast_scaler`, `generate_mask_w`, and related tile generation utilities.
   **Key Information**: Scaler tile layout (first 16 elements per face filled, rest zero), mask tile layout (1s for active width, 0s for padding), FP32/BF16 branching via template parameter.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/common_types.hpp`
   **Reason**: Definitions of `NoAccumulation` and `NoOp` tag types.
   **Key Information**: These are zero-cost abstractions eliminated by `if constexpr` -- passing `NoAccumulation{}` causes all accumulation code to compile away.
