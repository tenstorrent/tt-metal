# Softmax (W-Small Variant) Implementation Analysis

## Overview

This analysis covers the **softmax W-small** program factory, which computes numerically-stable softmax along the W (last) dimension for tiled, interleaved tensors where **all Wt tiles of a row fit in L1 simultaneously**. The "small" designation means the entire width of one row (Wt tiles) is loaded into circular buffers at once, enabling multi-pass reuse of data across compute phases without re-reading from DRAM.

**Program factory**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`

**Compute kernel**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Reader kernel**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`

**Writer kernel**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`

**Relevance to layer_norm_rm**: This operation is an excellent reference for row-wise reduction-then-broadcast patterns. Layer normalization requires the same fundamental structure: reduce across W to get a statistic (mean, variance), then broadcast that scalar back across all W tiles. The softmax kernel demonstrates how to chain multiple such phases (max-reduce, subtract-broadcast, exp, sum-reduce, reciprocal, multiply-broadcast) using intermediate CBs that persist across phases.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles (Wt tiles along the W dimension) |
| **Unit size** | Wt tiles |
| **Total units** | `num_kernel_rows = N_outer * Ht` where `N_outer = volume / (H * W)` |
| **Loop structure** | Outer loop over `N` rows assigned to this core; inner loops over `Wt` tiles per phase |

Each "row" is a contiguous set of Wt tiles spanning the full W dimension of the tensor. The compute kernel processes one such row per iteration of its outer `for (n = 0; n < N; ++n)` loop, executing all softmax phases (max, subtract, exp, sum, multiply) before moving to the next row.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary rank, last 2 dims are H, W) | Same as input |
| **Dimension convention** | Last dim = W (softmax dim), second-to-last = H | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations

No layout transformations (tilize/untilize) are performed. Input and output are both in TILE_LAYOUT. Padding on the W dimension is handled via a **mask tile** that zeroes out invalid elements in the last tile of each row (when `logical_W % 32 != 0`).

---

## Data Flow Pattern

The softmax computation for each row proceeds through 5 distinct phases. The critical insight for reuse as a reference is that **data produced in one phase persists in CBs and is consumed in later phases**.

### Phase 1: Row Max Reduction

**Purpose**: Find `max(x)` across the W dimension for numerical stability.

**Data flow**:
1. Reader pushes Wt tiles into `cb_in0` (c_0).
2. Compute uses `compute_kernel_lib::reduce<MAX, REDUCE_ROW>` to reduce the Wt input tiles to a single tile containing row-wise max values.
3. For `Wt == 1`: The single tile is first masked via `mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, ...)` then reduced.
4. For `Wt > 1`: The first `Wt-1` tiles are reduced with `WaitUpfrontNoPop` policy (tiles stay in cb_in0), then the last tile is masked to cb_tmp, then a second reduce call accumulates onto the existing max result using `Accumulate::at(cb_max, 1)`.
5. Result: 1 tile in `cb_max` (c_26).

**Key pattern**: `WaitUpfrontNoPop` policy means the reduce helper waits for all tiles but does NOT pop them. The tiles remain in `cb_in0` for reuse in Phase 2.

### Phase 2: Subtract Max (Centralization)

**Purpose**: Compute `x - max(x)` for each tile.

**Data flow**:
1. `cb_in0` still holds Wt input tiles (not popped from Phase 1).
2. `cb_max` holds 1 tile with row-max values.
3. For each tile w in [0, Wt): `sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0)` -- subtracts the max (broadcast across columns) from each input tile.
4. Results packed to `cb_x_m_max` (c_27), Wt tiles total.
5. Both `cb_in0` (Wt tiles) and `cb_max` (1 tile) are popped after the loop.

**Key pattern for layer_norm_rm**: This is exactly the "subtract mean" step. The `sub_tiles_bcast<COL>` operation broadcasts a single-column-of-scalars tile across all columns of the data tile. For layer norm, the mean tile (output of REDUCE_ROW) would be broadcast-subtracted from each input tile identically.

### Phase 3: Exponential with Masking

**Purpose**: Compute `exp(x - max(x))` and mask the last tile.

**Data flow**:
1. `cb_x_m_max` holds Wt tiles.
2. For each tile w in [0, Wt):
   - Copy tile from `cb_x_m_max` to DST via `copy_tile`.
   - Apply `exp_tile(dst0)` in DST.
   - If `w == Wt - 1` (last tile): load mask, apply `mask_tile(dst0, dst1)` to zero out padding elements.
   - Pack result to `cb_exps` (c_24).
3. `cb_x_m_max` tiles are NOT popped here (they persist for Phase 5 in the LOG variant, or are popped later).

**Key pattern**: The conditional masking on the last tile is important for any row-wise reduction on padded tensors. The mask tile persists for the entire program (never popped from `cb_mask`).

### Phase 4: Sum Reduction with Post-Reduce Reciprocal

**Purpose**: Compute `1/sum(exp(x - max(x)))`.

**Data flow** (SOFTMAX path, not LOG):
1. `cb_exps` holds Wt tiles of exp values.
2. `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` reduces Wt tiles to 1 tile.
3. The reduce helper applies a **post-reduce lambda**: `[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }` which computes the reciprocal of the sum in-place in DST before packing.
4. Result: 1 tile in `cb_recipsumexps` (c_25).
5. `cb_exps` tiles are NOT popped (WaitUpfrontNoPop), because they are needed in Phase 5.

**Key pattern for layer_norm_rm**: This demonstrates two critical techniques:
- **Post-reduce operation**: The lambda passed to reduce() applies an additional SFPU operation (reciprocal) to the reduce output while it is still in DST, before packing to the output CB. For layer norm, you would use `rsqrt_tile` instead of `recip_tile` to compute `1/sqrt(variance + eps)`.
- **Persistent tiles**: Using `WaitUpfrontNoPop` so the exp tiles remain available for the subsequent multiply phase.

### Phase 5: Final Multiply (Normalization)

**Purpose**: Compute `exp(x - max) * (1/sum)` for each tile.

**Data flow**:
1. `cb_exps` still holds Wt tiles (not popped from Phase 4).
2. `cb_recipsumexps` holds 1 tile with reciprocal sum.
3. For each tile w in [0, Wt): `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)` -- multiplies each exp tile by the reciprocal sum (broadcast across columns).
4. Results packed to `cb_out0` (c_16), Wt tiles total.
5. Both `cb_exps` (Wt tiles) and `cb_recipsumexps` (1 tile) are popped.
6. `cb_x_m_max` (Wt tiles) is also popped.

**Key pattern for layer_norm_rm**: This is the "scale by inv_std" step. The `mul_tiles_bcast_cols` broadcasts a single-column scalar tile across all columns. For layer norm, the inv_std tile would be broadcast-multiplied with each centralized tile. A second pass of `mul_tiles_bcast_cols` with gamma and `add_bcast_cols` with beta would complete the affine transform.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (full row) | Wt tiles | Wt tiles | Single | Reader | Compute | Row (persists across Phase 1 and 2) |
| c_1 | cb_mask | Mask tile for W-padding | 1 tile | 1 tile | Single | Reader | Compute | Program (loaded once, never popped) |
| c_2 | cb_bcast_scaler | Reduce scaler (1.0) | 1 tile | 1 tile | Single | Reader | Compute | Program (loaded once, never popped) |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Row (produced then consumed per row) |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row (persists across Phase 3, 4, 5) |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar | 1 tile | 1 tile | Single | Compute | Compute | Phase (produced in Phase 4, consumed in Phase 5) |
| c_26 | cb_max | row max scalar | 1 tile | 1 tile | Single | Compute | Compute | Phase (produced in Phase 1, consumed in Phase 2) |
| c_27 | cb_x_m_max | x - max(x) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row (persists across Phase 2, 3, 5 cleanup) |
| c_28 | cb_tmp | Masked last tile scratchpad | 1 tile | 1 tile | Single | Compute | Compute | Phase (temporary in Phase 1) |

### Data Format Selection

- **Input/Output CBs** (c_0, c_1, c_2, c_16): Use `data_format` derived from `input_tensor.dtype()` (BFLOAT16 or FLOAT32).
- **Intermediate CBs** (c_24, c_25, c_26, c_27, c_28): Use `intermed_data_format` which is Float32 if `fp32_dest_acc_en` is true, otherwise matches input data format. This ensures intermediate precision is maintained when FP32 accumulation is enabled.

### Multi-Pass Data Reuse Patterns (Critical for layer_norm_rm)

The following table summarizes which CBs persist across phases and why:

| CB | Produced in | Consumed in | Why persistent |
|----|-------------|-------------|----------------|
| cb_in0 (c_0) | Reader (before Phase 1) | Phase 1 (reduce max), Phase 2 (subtract) | Need raw input for both max reduction and subtraction. Reduce uses `WaitUpfrontNoPop` to keep tiles. |
| cb_mask (c_1) | Reader (once) | Phase 1 (mask last tile), Phase 3 (mask exp) | Mask is constant for all rows; `pop=0` in every call prevents consumption. |
| cb_bcast_scaler (c_2) | Reader (once) | Phase 1 (reduce max), Phase 4 (reduce sum) | Scaler is constant; reduce helper uses `cb_wait_front` without pop. |
| cb_exps (c_24) | Phase 3 | Phase 4 (reduce sum), Phase 5 (multiply) | Need exp values for both sum reduction and final multiply. `WaitUpfrontNoPop` in Phase 4 keeps tiles. |
| cb_x_m_max (c_27) | Phase 2 | Phase 3 (read for exp), Phase 5 (cleanup pop only) | In softmax path, cb_x_m_max is popped in Phase 5 cleanup. In LOG path, it is reused for the final subtraction. |

**Key insight**: The `WaitUpfrontNoPop` reduce input policy is the mechanism that enables multi-phase data reuse. The reduce helper waits for all tiles but does NOT call `cb_pop_front`, so tiles remain available for subsequent phases. The caller is responsible for eventually popping.

---

## Pipeline Pattern Summary

All CBs are **single-buffered** (capacity equals block size). There is no double-buffering or overlap between reader and compute for different rows. The reader produces one full row (Wt tiles) at a time, the compute processes all 5 phases, and the writer drains the output. Overlap occurs only at row boundaries (writer can drain row N while reader fills row N+1).

---

## Index Calculations

### Tile Indexing Within CBs

When `cb_wait_front(cb, Wt)` is called and Wt tiles are available, tiles are indexed 0 through Wt-1 from the current read pointer. The compute kernel accesses specific tiles by index:

```cpp
// Access tile w from cb_in0 which has Wt tiles available
sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
```

Here `w` is the tile index within cb_in0 (0 to Wt-1), and `0` is the tile index within cb_max (always the first and only tile).

### Row-to-Tile Offset Mapping

The program factory computes tile offsets for each core:

```cpp
tile_offset += num_tiles_per_core * Wt;  // Each "tile" in work split is a row; each row has Wt tiles
```

The reader kernel uses this offset to start reading from the correct position in the input tensor:

```cpp
uint32_t curr_tile = tile_offset;  // Starting tile index in the tensor
for (uint32_t w = 0; w < Wt; w++) {
    noc_async_read_tile(curr_tile, src_in, l1_write_addr_in);
    curr_tile++;
}
```

---

## Memory Access Patterns

### Read Pattern

- **Sequential tile reads**: Reader reads Wt consecutive tiles per row from DRAM via `noc_async_read_tile`.
- **Barrier after each row**: `noc_async_read_barrier()` ensures all tiles arrive before pushing to CB.
- **One-time generation**: Scaler and mask tiles are generated in L1 by the reader at startup (not read from DRAM).

### Write Pattern

- **Sequential tile writes**: Writer writes Wt consecutive tiles per row to DRAM via `noc_async_write_tile`.
- **Barrier after each row**: `noc_async_write_barrier()` ensures all tiles are written before popping from CB.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_coord.x * grid_coord.y` cores |
| **Total cores** | min(num_kernel_rows, available_cores) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` rows |
| **Load balancing** | Two-group uneven split (group 1 gets +1 row if remainder exists) |

The work unit is one "row" (Wt tiles along W). The total number of rows is `N_outer * Ht`. The function `split_work_to_cores_wt_core_range` divides rows across cores:

- If rows divide evenly: all cores get the same count.
- If not: `remainder` cores get one extra row (core_group_1), the rest get one fewer (core_group_2).

Core linearization uses column-major ordering within the grid:
```cpp
CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
```

---

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of rows assigned to this core (num_tiles_per_core_group_1 or _2) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (width of one row in tiles) |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input dtype is FLOAT32, 0 otherwise (controls scaler/mask generation) |
| 1+ | TensorAccessorArgs | varies | Accessor parameters for input tensor |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | varies | Accessor parameters for output tensor |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address in DRAM |
| 1 | num_tiles_per_core | uint32_t | Number of rows this core processes (N) |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles per row along W |
| 4 | scaler | uint32_t | Broadcast scaler value (1.0f reinterpreted as uint32_t) |
| 5 | mask_w | uint32_t | Number of valid elements in last W tile (1-32) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_tiles_per_core | uint32_t | Number of rows this core processes |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles per row along W |

### Compile Defines

| Define | Value | Description |
|--------|-------|-------------|
| SOFTMAX | 1 | Selects softmax path (exp, not negative+exp). Controls Phase 3 behavior. |
| FP32_DEST_ACC_EN | 1 (conditional) | Enables FP32 dest accumulation. Set when `fp32_dest_acc_en` is true. |

---

## Kernel Implementations

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| moreh_softmax_w.cpp | RISCV_2 (Unpack+Math+Pack) | N/A | cb_in0, cb_mask, cb_bcast_scaler | cb_out0 | reduce(MAX), sub_bcast_cols, exp, mask, reduce(SUM)+recip, mul_bcast_cols |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Includes**:
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` -- Provides `compute_kernel_lib::reduce<>()` helper.
- `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` -- Provides `mask_tile_to_cb()`, `pack_tile_with_dt()`, `copy_tile_init_with_dt()`, `sub_bcast_cols_init_short_with_dt()`, `mul_bcast_cols_init_short_with_dt()` and other `_with_dt` wrappers.

**Initialization**:
```cpp
binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);
```
This initializes the unpack, math, and pack hardware pipelines for binary operations. It is called once before any tile operations.

**Scalar/Constant CB Setup** (done by reader, consumed by compute):
- `cb_mask` (c_1): Contains a mask tile with 1.0 for valid positions and 0.0 for padding positions in the last W tile. Generated by `generate_mask_w<T>(cb_mask, mask_w)` in the reader. Waited on once at kernel start: `cb_wait_front(cb_mask, onetile)`. **Never popped** -- the `pop=0` / `popm=0` arguments in `mask_tile_to_cb` calls, and the mask being loaded to DST via `copy_tile` without popping, ensure this tile persists for the entire program.
- `cb_bcast_scaler` (c_2): Contains a tile filled with 1.0 in the scaler positions. Generated by `generate_bcast_scaler<T>(cb_scaler, scaler)` where scaler=1.0f. Waited on once at kernel start: `cb_wait_front(cb_bcast_scaler, onetile)`. **Never popped** -- the reduce helper uses indexed access without popping.

**Detailed Compute Kernel Call Sequence** (per row, SOFTMAX=1 path):

#### Phase 1: Row Max

```cpp
// Case Wt == 1:
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single());

// Case Wt > 1:
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1 -> reload
```

**Reduce helper parameters explained**:
- `ReduceInputBlockShape::row(Wt - 1)`: A single row of (Wt-1) tiles. The reduce will process (Wt-1) tiles producing 1 output.
- `WaitUpfrontNoPop`: Wait for all tiles, process them with indexed access, do NOT pop. Tiles remain in cb_in0.
- `Accumulate::at(cb_max, 1)`: On the second reduce call (iteration=1), reload the partial max from cb_max into DST before reducing cb_tmp. This effectively computes `max(first_Wt-1_tiles, masked_last_tile)`.

#### Phase 2: Subtract Max

```cpp
cb_reserve_back(cb_x_m_max, Wt);
cb_wait_front(cb_in0, Wt);   // Already available (not popped)
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

**Exact function signatures used**:
- `sub_bcast_cols_init_short_with_dt(uint32_t icb0, uint32_t icb1)` -- From moreh_common.hpp. Reconfigures data format (if FP32_DEST_ACC_EN), then calls `sub_bcast_cols_init_short(icb0, icb1)`.
- `sub_tiles_bcast<BroadcastType::COL>(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t dst)` -- Subtracts icb1[itile1] from icb0[itile0] with COL broadcast, result in DST[dst].
- `pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)` -- From moreh_common.hpp. Reconfigures pack data format (if FP32_DEST_ACC_EN), then calls `pack_tile(ifrom_dst, icb)`.

**BroadcastType::COL semantics**: The scalar tile (cb_max) is treated as having one column of values. Each row of the scalar tile's column 0 is broadcast across all 32 columns of the data tile's corresponding row. This is the correct broadcast for row-wise statistics (mean, max, variance) where the statistic is per-row within the tile.

**Bulk reserve/push pattern**: Space for all Wt output tiles is reserved at once with `cb_reserve_back(cb_x_m_max, Wt)`. The inner loop packs tiles one-by-one (each `pack_tile_with_dt` advances the write pointer by one tile). After the loop, `cb_push_back(cb_x_m_max, Wt)` makes all Wt tiles visible to the consumer at once. This is more efficient than reserving/pushing per tile.

#### Phase 3: Exponential with Masking

```cpp
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

**Exact function signatures**:
- `copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0)` -- Reconfigures SRCA format, calls `copy_tile_to_dst_init_short(icb)`.
- `copy_tile(uint32_t icb, uint32_t itile, uint32_t dst)` -- Unpacks tile at index itile from CB into DST[dst].
- `exp_tile_init()` / `exp_tile(uint32_t dst)` -- SFPU exponential operation on DST[dst].
- `mask_tile_init()` / `mask_tile(uint32_t dst_data, uint32_t dst_mask)` -- Element-wise multiply of DST[dst_data] by DST[dst_mask] (mask has 1.0 for valid, 0.0 for invalid).

**Note**: `cb_x_m_max` is NOT popped after Phase 3. In the LOG variant, it is reused in Phase 5 for subtraction. In the SOFTMAX variant, it is popped in Phase 5 cleanup.

#### Phase 4: Sum Reduction with Reciprocal

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

**Reduce helper parameters**:
- `ReduceInputBlockShape::row(Wt)`: A single row of Wt tiles.
- `WaitUpfrontNoPop`: Tiles in cb_exps remain available for Phase 5.
- `NoAccumulation{}`: No cross-call accumulation needed (single row processed in one call).
- **Post-reduce lambda**: `[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }` -- Applied to the reduce output in DST before packing. Computes `1/sum` in-place. The `dst_idx` parameter is the DST register index where the reduced tile resides (typically 0 for REDUCE_ROW).

**For layer_norm_rm**: Replace `recip_tile` with `rsqrt_tile` for computing `1/sqrt(var+eps)`. The pattern is identical: reduce sum, apply unary SFPU op, pack to scalar CB.

#### Phase 5: Final Multiply

```cpp
cb_reserve_back(cb_out0, Wt);
cb_wait_front(cb_x_m_max, Wt);
cb_wait_front(cb_recipsumexps, 1);
cb_wait_front(cb_exps, Wt);

for (uint32_t w = 0; w < Wt; w += onetile) {
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

**Exact function signatures**:
- `mul_bcast_cols_init_short_with_dt(uint32_t icb0, uint32_t icb1)` -- Reconfigures data format, calls `mul_bcast_cols_init_short(icb0, icb1)`.
- `mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t dst)` -- Multiplies icb0[itile0] by icb1[itile1] with COL broadcast, result in DST[dst].

---

### Reader Kernel (Summary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_softmax_w.cpp | RISCV_0 | NOC0 | DRAM input | cb_in0, cb_mask, cb_scaler | Read tiles, generate scaler/mask |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`

**Key responsibilities**:
1. Generate scaler tile (all 1.0s in scaler positions) into `cb_bcast_scaler` using `generate_bcast_scaler<T>(cb_scaler, scaler)`.
2. Generate mask tile (1.0 for valid W positions, 0.0 for padding) into `cb_mask` using `generate_mask_w<T>(cb_mask, mask_w)`.
3. For each row: read Wt tiles sequentially from DRAM into `cb_in0` using TensorAccessor + `noc_async_read_tile`.

### Writer Kernel (Summary)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_moreh_softmax_w.cpp | RISCV_1 | NOC1 | cb_out0 | DRAM output | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`

**Key responsibilities**:
1. For each row: wait for Wt tiles in `cb_out0`, write them sequentially to DRAM using TensorAccessor + `noc_async_write_tile`, pop after barrier.

---

## Implementation Notes

### Numerical Stability

The softmax uses the standard numerically-stable formulation: subtract the row max before exponentiation. This prevents overflow in `exp()` for large input values. The max is computed with a masked reduce to correctly handle padding.

### Two-Step Max Reduction for Padded Tensors

When `Wt > 1`, the max reduction is split into two calls:
1. First call reduces tiles 0 through Wt-2 (no masking needed for these).
2. The last tile (index Wt-1) is masked first (via `mask_tile_to_cb`), then reduced with accumulation from the first call's result.

This avoids applying the mask to all tiles (which would be wasteful) while ensuring padding values do not affect the max.

### FP32 Destination Accumulation

When `fp32_dest_acc_en` is true:
- The `FP32_DEST_ACC_EN` compile define activates `_with_dt` wrappers that call `reconfig_data_format` / `pack_reconfig_data_format` before each operation. This ensures correct format conversion between BFLOAT16 CBs and FP32 DST registers.
- Intermediate CBs use Float32 data format to preserve precision.
- The program factory enforces that FP32 inputs require FP32 accumulation.

### Mapping to Layer Normalization

The softmax phases map to layer norm as follows:

| Softmax Phase | Layer Norm Equivalent | Key Difference |
|---|---|---|
| Phase 1: reduce MAX | Compute mean: reduce SUM, scale by 1/W | Use SUM instead of MAX; apply scaler=1/W instead of 1.0 |
| Phase 2: subtract max | Subtract mean | Identical pattern: `sub_tiles_bcast<COL>` |
| Phase 3: exp + mask | Square centralized values | Replace exp with `mul_tiles` (x*x) for variance computation |
| Phase 4: reduce SUM + recip | Compute inv_std: reduce SUM (variance), add eps, rsqrt | Replace `recip_tile` with eps addition + `rsqrt_tile` in post-reduce lambda |
| Phase 5: mul by 1/sum | Scale by inv_std, then apply gamma*x+beta | Same `mul_tiles_bcast_cols` pattern; add extra passes for gamma and beta |

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do circular buffers work in tt-metal compute kernels? Specifically, explain the semantics of cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back, and how tiles in CBs are indexed."
   **Reason**: Needed to understand the CB synchronization model that underpins all data flow between phases.
   **Key Findings**: CBs act as FIFO queues. `cb_wait_front` blocks until tiles are available; `cb_pop_front` advances the read pointer. Tile indexing is relative to the current read pointer -- tile index `w` accesses the w-th tile from the front. Writing uses `cb_reserve_back` (ensures space) and `cb_push_back` (makes data visible).

2. **Query**: "Explain the broadcast operations: sub_tiles_bcast with BroadcastType::COL, mul_tiles_bcast_cols. What does COL broadcast mean?"
   **Reason**: The compute kernel uses COL broadcast for both subtraction (Phase 2) and multiplication (Phase 5). Understanding the broadcast semantics is critical for layer norm.
   **Key Findings**: COL broadcast takes a single column of values from the scalar tile and replicates it across all columns of the data tile. For row-wise operations, this means each row of the data tile is operated with the corresponding row value from the scalar tile's column. This is the correct pattern for broadcasting per-row statistics (mean, variance, max).

3. **Query**: "How does the split_work_to_cores function work in tt-metal?"
   **Reason**: Needed to understand how rows are distributed across cores.
   **Key Findings**: Returns a two-group split: core_group_1 gets `floor(N/cores)+1` rows, core_group_2 gets `floor(N/cores)` rows. The remainder cores (N % cores) go into group 1.

4. **Query**: "Explain the tile register management pattern: tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release."
   **Reason**: The compute kernel uses this 4-phase pattern extensively. Understanding it is essential for writing correct compute kernels.
   **Key Findings**: These coordinate three RISC-V cores (unpack, math, pack). `acquire` gives math core exclusive DST access; `commit` transfers ownership to packer; `wait` blocks packer until math is done; `release` frees DST for the next iteration. All four must be called even if no packing occurs.

5. **Query**: "How does binary_op_init_common work? What does mask_tile do?"
   **Reason**: The compute kernel calls `binary_op_init_common` at startup and `mask_tile` in multiple phases.
   **Key Findings**: `binary_op_init_common` initializes unpack, math, and pack hardware for binary operations. `mask_tile` performs element-wise multiplication of data tile by mask tile in DST, effectively zeroing out positions where mask is 0.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (header + `.inl` implementation)
   **Reason**: The compute kernel uses `compute_kernel_lib::reduce<>()` as its primary reduction mechanism.
   **Key Information**: The reduce helper is a fully-parameterized template that handles CB wait/pop, DST management, accumulation, and post-reduce operations. Key policy: `WaitUpfrontNoPop` waits for all tiles but does not pop, enabling multi-phase data reuse. The post_reduce_op lambda receives `dst_idx` and can apply any SFPU operation (recip, rsqrt, log, etc.) before packing.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Provides all `_with_dt` wrapper functions and utility helpers used by the compute kernel.
   **Key Information**: `pack_tile_with_dt`, `copy_tile_init_with_dt`, `sub_bcast_cols_init_short_with_dt`, `mul_bcast_cols_init_short_with_dt` are wrappers that conditionally call `reconfig_data_format` / `pack_reconfig_data_format` when `FP32_DEST_ACC_EN` is defined. The `mask_tile_to_cb` helper encapsulates copy+mask+pack into a single function with configurable pop counts.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DST register capacity limits, which affect reduce chunking.
   **Key Information**: DST capacity depends on sync mode and accumulation mode. Half-sync + FP32 = 4 tiles; Half-sync + FP16 = 8 tiles. The reduce helper uses `DEST_AUTO_LIMIT` (from `get_dest_limit()`) to automatically determine chunk sizes for REDUCE_COL. For REDUCE_ROW (used in softmax), each row reduces to a single tile in DST, so capacity is not a bottleneck.

4. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding how the reader generates constant tiles.
   **Key Information**: `generate_bcast_scaler<T>(cb, scaler)` fills a tile with the scaler value in the first 16 positions of each 256-element sub-tile (4 sub-tiles per tile). `generate_mask_w<T>(cb, mask_w)` creates a tile with 1.0 for the first `mask_w` columns and 0.0 for the rest, properly handling the 4-subtile layout.
