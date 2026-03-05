# Softmax General (W-dimension) Implementation Analysis

## Overview

This analysis covers the **general-purpose softmax** operation's W-dimension variants (`w_small` and `w_large`) as a compute-core reference for a new `layer_norm_rm` operation. The softmax general factory is a dispatcher that selects among dimension-specific sub-factories (`w_small`, `w_large`, `h_small`, `h_large`, `c_large`) based on the reduction dimension and whether the tensor fits in L1.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general.cpp` (dispatcher only; actual implementations in `softmax_program_factory_general_w_small.cpp` and `softmax_program_factory_general_w_large.cpp`)

**Kernel source path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`

**Relevance to layer_norm_rm**: Both softmax and layer norm perform row-wise reductions (W-dimension), subtraction/broadcast of a reduced scalar, element-wise nonlinear operations, and final scaling. The compute kernel structure, CB layout, reduce helper usage, and multi-pass data reuse patterns transfer directly.

---

## Variant Selection: w_small vs w_large

The device operation (`softmax_device_operation.cpp` line 148-152) selects between variants based on whether all intermediate CBs fit in L1:

- **w_small**: All `Wt` tiles of a row can be held simultaneously in L1. Enables in-place indexed access to tiles (no re-reading). Intermediates like `cb_x_m_max` and `cb_exps` are sized to `Wt` tiles.
- **w_large**: Row is too wide to hold entirely. Tiles are streamed one at a time (capacity=2 for double-buffering). Input data is re-read from DRAM 3 times (once per phase).

The L1 budget check (`is_softmax_general_w_small_available`, line 40-70) sums all CB requirements:
```
cb_usage = Wt*tile + 1*tile + 1*tile + Wt*tile + Wt*intermed + 1*intermed + 1*intermed + Wt*intermed + 1*intermed
```

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row |
| **Unit size** | 1 tile row = Wt tiles across the W dimension |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop over `N` assigned rows; inner loop over `Wt` tiles per row |

Each core processes `N` tile rows. One "work unit" is a complete tile row consisting of `Wt` tiles. The compute kernel iterates `N` times, where each iteration performs the full softmax pipeline on one tile row.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (any rank, W = last dim) | Same as input |
| **Dimension convention** | Last dim = W (reduction dim) | Same |
| **Tensor layout** | TILE (32x32 tiles) | TILE |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32, BFLOAT16, or BFLOAT8_B | Same as input |

### Layout Transformations
- None within the kernel. If input is ROW_MAJOR, the host-side `softmax()` function (line 422-428) calls `tilize_with_val_padding` with `-inf` padding before dispatching.
- For the general variant, the input must already be in TILE layout (validated at line 171).

---

## Circular Buffer Configuration

### w_large Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_mask | Width mask (padding) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Scaler (1.0) for reduce | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_24 | cb_exps | exp(x-max) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_add | Running sum accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_max | Row maximum | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Scratch space | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile) |

**Key note on w_large intermediate format**: c_24 through c_28 use `intermed_data_format`, which is `Float32` when `fp32_dest_acc_en` is true, otherwise matches input data format.

### w_small Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles (full row) | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_mask | Width mask (padding) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Scaler (1.0) for reduce | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles (full row) | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x-max) for full row | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row maximum | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max(x) for full row | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Scratch space | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Critical difference in CB index assignments**: Note that in `w_small`, `c_26` is `cb_max` and `c_27` is `cb_x_m_max`, while in `w_large`, `c_26` is `cb_add` (running sum) and `c_27` is `cb_max`. The CB indices are different between variants even though the program factory assigns them at different positions.

---

## Multi-Pass Data Reuse Patterns

### w_large: Three-Pass Streaming with DRAM Re-reads

The w_large reader kernel (`reader_moreh_softmax_w_large.cpp`) reads the same row from DRAM **three times** per iteration (lines 44-74):

1. **Pass 1 (Find Max)**: Read Wt tiles into `cb_in0` (capacity=2, streaming). Compute finds `max(row)`.
2. **Pass 2 (Compute exp)**: Re-read same Wt tiles. Compute does `exp(x - max)` and accumulates sum.
3. **Pass 3 (Final output)**: Re-read same Wt tiles. Compute does `exp(x - max) * (1/sum)` and produces output.

**CBs that persist across phases**:
- `cb_max` (c_27): Populated in Pass 1, consumed in Passes 2 and 3. Popped only at the very end of all three passes (line 159 of `moreh_softmax_w_large.cpp`).
- `cb_recipsumexps` (c_25): Populated at end of Pass 2, consumed in Pass 3. Popped at end (line 158).
- `cb_mask` (c_1) and `cb_bcast_scaler` (c_2): Written once by reader, never popped (persist for entire program).

**Why this design**: With only capacity=2, there is no room to hold Wt tiles. The trade-off is 3x DRAM bandwidth for reduced L1 footprint.

### w_small: Single-Read with L1 Persistence

The w_small reader kernel (`reader_moreh_softmax_w.cpp`) reads the row **once** as a bulk of `Wt` tiles (lines 44-54):

```cpp
cb_reserve_back(cb_in, Wt);
for (uint32_t w = 0; w < Wt; w++) {
    noc_async_read_tile(curr_tile, src_in, l1_write_addr_in);
    l1_write_addr_in += src_in_tile_bytes;
    curr_tile++;
}
noc_async_read_barrier();
cb_push_back(cb_in, Wt);
```

**CBs that persist across phases**:
- `cb_in0` (c_0, Wt tiles): Read once, used for max reduction (with `WaitUpfrontNoPop` policy - no pop after reduce), then popped after `x - max` subtraction is complete (line 72).
- `cb_x_m_max` (c_27, Wt tiles): Populated during subtraction phase, persists through exp phase into final output phase. Popped at line 170.
- `cb_exps` (c_24, Wt tiles): Populated during exp phase, persists through sum reduction (with `WaitUpfrontNoPop`) into final multiplication phase. Popped at line 173.
- `cb_max` (c_26, 1 tile): Populated during max reduction, consumed during subtraction. Popped at line 71.
- `cb_recipsumexps` (c_25, 1 tile): Populated after sum reduction, consumed during final multiplication. Popped at line 169.

**Why this design**: Holding all tiles in L1 avoids DRAM re-reads entirely. Each tile is read from DRAM exactly once.

---

## Scalar and Constant CB Setup

Both variants use the reader kernel to populate two constant CBs before the main loop:

### cb_bcast_scaler (c_2) - Scaler Tile
```cpp
generate_bcast_scaler<uint16_t>(cb_scaler, scaler);  // scaler = 1.0f
```
- Creates a tile with `1.0f` in the first row of each face (positions `k*256 + j` for k=0..3, j=0..15), rest zeros.
- This is the required format for `reduce_tile` - it applies this scaler to the accumulation result.
- For softmax, scaler = 1.0f (pure sum, no scaling). For layer norm, you would also use 1.0f for sum reduction.
- **Lifetime**: Written once, never popped. Persists for entire program.

### cb_mask (c_1) - Width Mask Tile
```cpp
generate_mask_w<uint16_t>(cb_mask, mask_w);  // mask_w = logical_W % TILE_WIDTH
```
- Creates a binary mask tile: 1.0 for valid columns (0..mask_w-1), 0.0 for padding columns (mask_w..31).
- Applied to the last tile in each row to zero out padding before reduction.
- If `mask_w == 0`, it is set to `TILE_WIDTH` (meaning the entire tile is valid).
- **Lifetime**: Written once, never popped. Persists for entire program.

---

## Compute Kernel Structure: w_small (moreh_softmax_w.cpp)

### Initialization
```cpp
binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);
```
Initializes UNPACK, MATH, and PACK hardware units for binary operations. Parameters specify the default CB data formats for the three hardware stages.

### Compile-Time Arguments
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile rows this core processes |
| 1 | Wt | uint32_t | Number of tiles in W dimension per row |

### Phase 1: Find Row Maximum (reduce MAX over W)

**Single tile case (Wt==1)**:
```cpp
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max, ReduceInputBlockShape::single());
```

**Multi-tile case (Wt>1)**:
```cpp
// Phase 1a: Reduce first Wt-1 tiles (no mask needed for non-last tiles)
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max, ReduceInputBlockShape::row(Wt - 1));

// Phase 1b: Mask the last tile, then reduce with accumulation from Phase 1a
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    ReduceInputBlockShape::single(),
    ReduceInputMemoryLayout::contiguous(),
    Accumulate::at(cb_max, 1));  // iteration=1 triggers reload from cb_max
```

**Key pattern - Two-phase reduce with accumulation**:
- First call reduces `Wt-1` tiles with `WaitUpfrontNoPop` (tiles stay in CB for reuse).
- Second call processes the masked last tile with `Accumulate::at(cb_max, 1)`:
  - `iteration=1` means "reload the previous partial result from `cb_max` into DST before reducing"
  - The reduce library handles: `cb_wait_front(cb_max, 1)` -> `copy_tile` to DST -> `cb_pop_front(cb_max, 1)` -> re-init reduce -> reduce the new tile with accumulated value
  - Final max result is packed back to `cb_max`.

### Phase 2: Compute x - max(x)

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

**Binary op broadcast pattern - BroadcastType::COL**:
- `sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0)`:
  - Reads tile `w` from `cb_in0` and tile `0` from `cb_max`
  - `cb_max` contains the row max (a "column vector" - same value replicated across all columns)
  - BroadcastType::COL means: broadcast the column vector across all columns, then subtract
  - Result: `input[w] - max` for every element in the tile
- `sub_bcast_cols_init_short_with_dt` handles data format reconfiguration when `FP32_DEST_ACC_EN`

**Important**: Uses indexed tile access (`w` parameter) into `cb_in0` without popping - this is why `cb_in0` needs `Wt` tiles capacity and was filled with `WaitUpfrontNoPop` semantics.

After this loop:
- `cb_in0` is fully consumed (popped)
- `cb_max` is consumed (popped)
- `cb_x_m_max` holds all Wt tiles of `(x - max(x))`

### Phase 3: Compute exp(x - max(x)) and Apply Mask

```cpp
cb_reserve_back(cb_exps, Wt);
cb_wait_front(cb_x_m_max, Wt);
for (uint32_t w = 0; w < Wt; ++w) {
    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x_m_max);
    copy_tile(cb_x_m_max, w, dst0);

    exp_tile_init();
    exp_tile(dst0);

    if (w == Wt - 1) {  // Mask last tile
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

**Sequence per tile**:
1. `copy_tile(cb_x_m_max, w, dst0)` - Load tile `w` from `cb_x_m_max` into DST register 0 (indexed access, no pop)
2. `exp_tile(dst0)` - Compute exp() in-place in DST register 0 (SFPU operation)
3. For last tile only: `mask_tile(dst0, dst1)` - Zero out padding columns by element-wise multiply with mask
4. `pack_tile_with_dt(dst0, cb_exps)` - Pack result to `cb_exps`

**Note**: `cb_x_m_max` tiles are accessed by index (`w`) without being popped. They persist for Phase 5.

### Phase 4: Reduce sum and compute reciprocal (1/sum)

```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_exps, cb_bcast_scaler, cb_recipsumexps,
    ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

**Reduce helper parameters explained**:
- `PoolType::SUM, ReduceDim::REDUCE_ROW`: Sum all columns, producing 1 output tile per row
- `ReduceInputPolicy::WaitUpfrontNoPop`: Wait for all `Wt` tiles in `cb_exps` upfront, do NOT pop them after (they will be reused in Phase 5)
- `ReduceInputBlockShape::row(Wt)`: Input is a single row of `Wt` tiles
- `NoAccumulation{}`: No multi-block accumulation needed
- **Post-reduce lambda**: `recip_tile(dst_idx)` computes `1/sum` in-place in DST after the reduction completes. The reduce library calls this lambda after all `Wt` tiles have been accumulated, before packing.

Output: `cb_recipsumexps` contains one tile with `1/sum(exp(x-max))` as a column vector.

### Phase 5: Final Output - exp(x-max) * (1/sum)

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

**Binary op broadcast pattern - mul_tiles_bcast_cols**:
- `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)`:
  - Reads tile `w` from `cb_exps` (exp values) and tile `0` from `cb_recipsumexps` (1/sum)
  - BroadcastType::COL: broadcasts `1/sum` column vector across all columns
  - Result: `exp(x-max) / sum(exp(x-max))` per element

After this phase, all intermediate CBs are popped and `cb_out0` is pushed with `Wt` output tiles.

---

## Compute Kernel Structure: w_large (moreh_softmax_w_large.cpp)

The w_large variant follows the same algorithmic phases but processes tiles one at a time via streaming. The reader supplies 3x the tiles (re-reading from DRAM).

### Key Differences from w_small

1. **No bulk CB reservation**: All operations use `_to_cb` helper functions from `moreh_common.hpp` that handle `cb_reserve_back(1)`, `cb_wait_front`, computation, `cb_pop_front`, and `cb_push_back(1)` internally.

2. **Phase 2 uses running accumulation instead of bulk store**:
```cpp
for (uint32_t w = 0; w < Wt; ++w) {
    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);
    exp_tile_to_cb(cb_tmp, cb_exps);
    if (w == 0) {
        copy_tile_to_cb(cb_exps, cb_add);
    } else {
        add_tiles_to_cb(cb_add, cb_exps, cb_add);
    }
}
```
   - Each tile: subtract max, compute exp, add to running sum
   - `cb_add` accumulates the element-wise sum of exp tiles
   - `cb_max` uses `pop1=0` (not popped) - persists across the loop

3. **Sum reduction is over `cb_add` (1 tile) not `cb_exps` (Wt tiles)**:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
    cb_add, cb_bcast_scaler, cb_recipsumexps,
    ReduceInputBlockShape::single(), ...,
    NoAccumulation{},
    [](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); });
```
   - Since `cb_add` already accumulated the sum across all Wt tiles element-wise, only need to reduce within a single tile.

4. **Phase 3 re-reads and recomputes**:
```cpp
for (uint32_t w = 0; w < Wt; w += onetile) {
    sub_tiles_bcast_cols_to_cb(cb_in0, cb_max, cb_tmp, 0, 0, /*pop0=*/1, /*pop1=*/0);
    exp_tile_to_cb(cb_tmp, cb_exps);
    mul_tiles_bcast_cols_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
}
```
   - Each tile: re-subtract max, re-compute exp, multiply by 1/sum
   - `cb_max` and `cb_recipsumexps` both use `pop1=0` to persist

### Helper Function Signatures (from moreh_common.hpp)

All `_to_cb` helpers follow this pattern:
```cpp
void sub_tiles_bcast_cols_to_cb(
    uint32_t icb0,       // Input CB A
    uint32_t icb1,       // Input CB B (broadcast source)
    uint32_t ocb,        // Output CB
    uint32_t itile0 = 0, // Tile index in CB A
    uint32_t itile1 = 0, // Tile index in CB B
    uint32_t pop0 = 1,   // Pop count for CB A (0 = don't pop)
    uint32_t pop1 = 1    // Pop count for CB B (0 = don't pop)
);
```

Inside each helper:
1. `cb_reserve_back(ocb, 1)` - Reserve output space
2. `cb_wait_front(icb0, itile0 + 1)` and `cb_wait_front(icb1, itile1 + 1)` - Wait for inputs
3. `tile_regs_acquire()` - Acquire DST registers
4. Init + compute (e.g., `sub_bcast_cols_init_short()` then `sub_tiles_bcast<COL>()`)
5. `tile_regs_commit()` / `tile_regs_wait()` / `pack_tile_with_dt()` / `tile_regs_release()`
6. Conditional pop of inputs based on `pop0`/`pop1`
7. `cb_push_back(ocb, 1)` - Make output available

---

## Reduce Helper Parameters Reference

### Function Signature
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim,
          ReduceInputPolicy input_policy = WaitAndPopPerTile,
          ReduceDataFormatReconfigMode reconfig_mode = INPUT_AND_OUTPUT,
          typename AccumulateT = NoAccumulation,
          typename PostReduceOp = NoOp>
void reduce(uint32_t input_cb, uint32_t scaler_cb, uint32_t output_cb,
            ReduceInputBlockShape input_block_shape,
            ReduceInputMemoryLayout input_memory_layout = contiguous(),
            AccumulateT accumulate = {},
            PostReduceOp post_reduce_op = {});
```

### Input Policies Used in Softmax

| Policy | When Used | CB Behavior | Why |
|--------|-----------|-------------|-----|
| `WaitUpfrontNoPop` | w_small: max reduce, sum reduce | Waits for all tiles, does not pop | Tiles needed for subsequent phases |
| `WaitAndPopPerTile` | w_large: max reduce (first Wt-1 tiles via default) | Wait 1, pop 1, streaming | Large row, can't hold all tiles |
| `BulkWaitBulkPop` | w_large: sum reduce over single cb_add tile | Wait all, pop all | Single accumulated tile, fully consumed |

### Accumulation Pattern

Used in the two-phase max reduction (both variants):
```cpp
// Phase 1: Reduce first Wt-1 tiles -> cb_max
reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>(cb_in0, scaler, cb_max, row(Wt-1));
// Phase 2: Reduce last masked tile, accumulating with Phase 1 result
reduce<MAX, REDUCE_ROW>(cb_tmp, scaler, cb_max, single(), contiguous(),
                         Accumulate::at(cb_max, 1));
```

The `Accumulate::at(cb_max, 1)` means:
- `cb_accumulator = cb_max`: Read previous partial max from this CB
- `iteration = 1`: Not the first iteration, so DO reload the accumulator
- Internally: `copy_tile(cb_max, 0, dst_idx)` loads previous max into DST, then `reduce_tile` on new input accumulates with it

### Post-Reduce Operations

Used for softmax's reciprocal after sum:
```cpp
reduce<SUM, REDUCE_ROW, ...>(..., NoAccumulation{},
    [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    });
```

The lambda is called after all input tiles are reduced but before `pack_tile`. This enables fusing the `1/sum` computation directly into the reduce pipeline without an extra CB round-trip.

**For layer_norm_rm**: You would use `rsqrt_tile` instead of `recip_tile` for the variance normalization step.

---

## Binary Op Broadcast Patterns Summary

| Operation | Function | Broadcast | Use Case |
|-----------|----------|-----------|----------|
| Subtract max from row | `sub_tiles_bcast<BroadcastType::COL>` | Column vector broadcast | x - max(x), where max is 1 value per row |
| Multiply by 1/sum | `mul_tiles_bcast_cols` | Column vector broadcast | exp(x-max) * (1/sum) |

**BroadcastType::COL semantics**: The second operand is treated as a column vector (one value per row of the 32x32 tile). That single column is broadcast/replicated across all 32 columns before the binary operation. Since REDUCE_ROW produces a column vector (one result per row), the natural pattern is: reduce rows -> broadcast result back across columns.

**Init pattern for FP32_DEST_ACC_EN**:
```cpp
sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);  // Reconfigures data format if FP32
sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
```
The `_with_dt` variant adds `reconfig_data_format(icb0, icb1)` when `FP32_DEST_ACC_EN` is defined, ensuring the unpacker knows the correct data formats for mixed-precision operation.

---

## Data Flow Pattern

### w_small Data Flow (single DRAM read)

| Stage | Phase | Reader | Compute | Data Movement |
|-------|-------|--------|---------|---------------|
| 1 | Setup | Write scaler to c_2, mask to c_1 | Wait for c_1, c_2 | Once at startup |
| 2 | Read row | Read Wt tiles to c_0 (bulk) | - | DRAM -> L1 |
| 3 | Max | - | Reduce MAX over c_0 -> c_26 | L1 internal |
| 4 | Subtract | - | c_0 - c_26 -> c_27 (Wt tiles) | L1 internal |
| 5 | Exp+Mask | - | exp(c_27) -> c_24 (Wt tiles) | L1 internal |
| 6 | Sum+Recip | - | Reduce SUM over c_24 -> recip -> c_25 | L1 internal |
| 7 | Multiply | - | c_24 * c_25 -> c_16 (Wt tiles) | L1 internal |
| 8 | Write row | - | - | L1 -> DRAM (writer reads c_16) |

### w_large Data Flow (3x DRAM reads)

| Stage | Phase | Reader | Compute | Data Movement |
|-------|-------|--------|---------|---------------|
| 1 | Setup | Write scaler to c_2, mask to c_1 | - | Once at startup |
| 2 | Read row (Pass 1) | Stream Wt tiles to c_0 | Reduce MAX -> c_27 | DRAM -> L1, 1 tile at a time |
| 3 | Read row (Pass 2) | Re-stream Wt tiles to c_0 | sub, exp, accumulate sum -> c_26 | DRAM -> L1, 1 tile at a time |
| 4 | Sum+Recip | - | Reduce SUM c_26 -> recip -> c_25 | L1 internal |
| 5 | Read row (Pass 3) | Re-stream Wt tiles to c_0 | sub, exp, mul -> c_16 | DRAM -> L1, 1 tile at a time |
| 6 | Write | - | - | L1 -> DRAM (writer reads c_16) |

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (fills column-first) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (full compute grid) |
| **Total cores** | `num_cores` (up to grid_x * grid_y) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tile rows |
| **Load balancing** | Two-group: group 1 gets `ceil(total/cores)`, group 2 gets `floor(total/cores)` |
| **Work unit** | 1 tile row = Wt tiles |

The work splitting uses `split_work_to_cores_wt_core_range()` which distributes `num_kernel_rows` across the available cores. Cores in `core_group_1` get one more row than cores in `core_group_2`.

Core indexing: `core = {i / core_h, i % core_h}` - fills columns first (y increments faster than x).

---

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile rows assigned to this core |
| 1 | Wt | uint32_t | Width of tensor in tiles (tiles per row) |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input dtype is FLOAT32, 0 otherwise |
| 1+ | TensorAccessorArgs | varies | Buffer type, page size, etc. for TensorAccessor |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | N | uint32_t | Number of tile rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index in flattened tile array |
| 3 | Wt | uint32_t | Tiles per row |
| 4 | scaler | uint32_t | Scaler value as bit-cast uint32 (1.0f) |
| 5 | mask_w | uint32_t | Number of valid elements in last tile column |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | N | uint32_t | Number of tile rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles per row |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_softmax_w.cpp (small) | RISCV_0 | NOC0 | DRAM | c_0, c_1, c_2 | Read Wt tiles bulk, gen scaler/mask |
| reader_moreh_softmax_w_large.cpp | RISCV_0 | NOC0 | DRAM | c_0, c_1, c_2 | Read Wt tiles 3x (streaming), gen scaler/mask |
| moreh_softmax_w.cpp (small) | RISCV_2 | N/A | c_0,c_1,c_2 | c_16 | MAX reduce, sub, exp, SUM reduce+recip, mul |
| moreh_softmax_w_large.cpp | RISCV_2 | N/A | c_0,c_1,c_2 | c_16 | Same algorithm, streaming tile-by-tile |
| writer_moreh_softmax_w.cpp (small) | RISCV_1 | NOC1 | c_16 | DRAM | Write Wt tiles per row (streaming) |
| writer_moreh_softmax_w_large.cpp | RISCV_1 | NOC1 | c_16 | DRAM | Write Wt tiles per row (streaming) |

### Compute Kernel Key Logic

**Define-based variants**: The same kernel file supports multiple operations via `#define`:
- `SOFTMAX`: When defined, uses `exp(x - max)` (standard softmax). When undefined, uses `exp(-(x - max))` (softmin).
- `LOG`: When defined, computes `log(softmax(x)) = x - max - log(sum(exp(x-max)))` instead of `exp(x-max)/sum`.
- `FP32_DEST_ACC_EN`: When defined, enables `pack_reconfig_data_format` / `reconfig_data_format` calls for mixed precision.

The program factory sets `compute_defines["SOFTMAX"] = "1"` (line 93 of w_large, line 96 of w_small).

---

## Pipeline Pattern Summary

### w_small
- **cb_in0 (c_0)**: Single-buffered at Wt tiles. Bulk read by reader, bulk consumed by compute.
- **cb_out0 (c_16)**: Single-buffered at Wt tiles. Bulk produced by compute, streaming consumed by writer.
- **All intermediates**: Single-buffered. No overlap between reader and compute for a given row (reader delivers entire row before compute starts).

### w_large
- **cb_in0 (c_0)**: Double-buffered (capacity=2). Enables overlap between reader and compute within a pass.
- **cb_out0 (c_16)**: Double-buffered (capacity=2). Enables overlap between compute and writer.
- **All intermediates**: Single-buffered at 1 tile (except cb_exps at 2 tiles).

---

## Relevance to layer_norm_rm

### Direct Analogies

| Softmax Phase | Layer Norm Equivalent | Key Difference |
|---------------|----------------------|----------------|
| MAX reduce over row | MEAN reduce over row | Use SUM + multiply by 1/W instead of MAX |
| x - max | x - mean | Same broadcast pattern |
| exp(x - max) | (x - mean)^2 | Square instead of exp |
| SUM reduce + recip | SUM of squares + recip (variance) | Then rsqrt instead of recip |
| exp * (1/sum) | (x - mean) * rsqrt(var + eps) * gamma + beta | More operations, needs gamma/beta CBs |

### Patterns to Reuse

1. **reduce_helpers_compute.hpp library**: Use `reduce<SUM, REDUCE_ROW>` for mean computation (with scaler = 1/W or post-reduce multiply)
2. **BroadcastType::COL pattern**: For subtracting mean from each tile in the row
3. **Post-reduce lambda**: Use `rsqrt_tile` for variance normalization (analogous to `recip_tile`)
4. **mask_tile for padding**: Same approach for non-tile-aligned widths
5. **Two-variant strategy**: Consider w_small (all in L1) vs w_large (streaming) based on available L1

### Additional CBs Needed for Layer Norm

- **cb_gamma**: Gamma affine parameter (1 row of Wt tiles, broadcast across batch)
- **cb_beta**: Beta affine parameter (1 row of Wt tiles, broadcast across batch)
- **cb_eps**: Epsilon scalar (1 tile, constant)
- **cb_mean**: Mean intermediate (1 tile per row, replaces cb_max)
- **cb_var**: Variance intermediate (1 tile per row, replaces cb_recipsumexps)
- **cb_x_m_mean**: x - mean (Wt tiles for small variant, replaces cb_x_m_max)

---

## Implementation Notes

1. **`binary_op_init_common` must be called first**: This is the hardware initialization for all subsequent binary and unary tile operations. Called once at kernel start.

2. **`_with_dt` suffix functions**: Handle FP32 dest accumulation mode by calling `reconfig_data_format()` before the operation. Always use these when `FP32_DEST_ACC_EN` might be defined.

3. **`pack_tile_with_dt` vs `pack_tile`**: The `_with_dt` version adds `pack_reconfig_data_format(icb)` when `FP32_DEST_ACC_EN` is defined. Use `pack_tile_with_dt` for all manual pack operations; the reduce library handles this internally.

4. **DST register management**: Every tile operation sequence follows:
   ```
   tile_regs_acquire() -> [compute] -> tile_regs_commit() -> tile_regs_wait() -> [pack] -> tile_regs_release()
   ```
   The `_to_cb` helpers and the reduce library handle this automatically.

5. **`cb_wait_front` with index access**: When using `cb_wait_front(cb, N)` followed by `copy_tile(cb, idx, dst)`, you can access any tile `0..N-1` without popping. This is the foundation of the w_small variant's efficiency.

6. **Pop semantics in helpers**: The `pop0=0` and `pop1=0` parameters in `_to_cb` helpers prevent consumption of persistent CBs (like `cb_max`, `cb_recipsumexps`). Always use `pop1=0` for scalar CBs that are reused across the row's inner loop.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the broadcast pattern work in compute kernels, specifically BroadcastType::COL for operations like sub_tiles_bcast?"
   **Reason**: Needed to understand the exact semantics of column broadcast in subtraction and multiplication operations.
   **Key Findings**: BroadcastType::COL treats the second operand as a column vector and broadcasts it across all columns of the first operand. REDUCE_ROW produces a column vector (one value per row), making COL broadcast the natural complement.

2. **Query**: "What is binary_op_init_common in compute kernels and what does it initialize?"
   **Reason**: Needed to understand the mandatory initialization before any binary/unary tile operations.
   **Key Findings**: Initializes UNPACK, MATH, and PACK hardware units. Parameters specify CB IDs for data format configuration. Must be called before any tile operations.

3. **Query**: "What is the mask_tile operation in compute kernels?"
   **Reason**: Needed to understand how padding is handled in the last tile of a row.
   **Key Findings**: `mask_tile(dst_data, dst_mask)` performs element-wise multiplication, zeroing out elements where the mask is 0. `generate_mask_w` creates a tile with 1.0 for valid columns and 0.0 for padding.

4. **Query**: "What is the exact semantics of the scaler tile used in reduce operations?"
   **Reason**: Needed to understand the scaler CB format and whether 1.0f is the correct value for sum reduction.
   **Key Findings**: The scaler tile has values in the first row of each face (16 elements per face), rest zeros. For SUM, scaler=1.0 gives unscaled sum. For AVG, scaler=1/N. The reduce hardware multiplies by this scaler during accumulation.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Primary API for all reduce operations used by softmax compute kernel
   **Key Information**: Full reduce function signature, input policies (WaitUpfrontNoPop, BulkWaitBulkPop, etc.), accumulation pattern, post-reduce operation lambda, ReduceInputBlockShape configuration

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Contains all `_to_cb` helper functions used by w_large compute kernel
   **Key Information**: Function signatures for `sub_tiles_bcast_cols_to_cb`, `mul_tiles_bcast_cols_to_cb`, `exp_tile_to_cb`, `copy_tile_to_cb`, `add_tiles_to_cb`, `mask_tile_to_cb` with pop parameter semantics

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Contains `generate_bcast_scaler` and `generate_mask_w` used by reader kernels
   **Key Information**: Scaler tile layout (values in first row of each face), mask tile layout (1.0 for valid, 0.0 for padding), subtile structure of 32x32 tiles

4. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp`
   **Reason**: Host-side helpers for CB creation, kernel creation, and work splitting
   **Key Information**: `CreateCircularBuffer` accepts `CircularBufferArg` with optional data format override, `CreateComputeKernel` accepts core groups with per-group compile args, `split_work_to_cores_wt_core_range` returns two core groups for load balancing
