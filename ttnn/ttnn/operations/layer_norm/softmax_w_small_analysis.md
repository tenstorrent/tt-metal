# Softmax W-Small Implementation Analysis

## Overview

This document analyzes the "W-small" softmax program factory, which computes the standard softmax function along the **width (W) dimension** of a tiled tensor. "W-small" means the entire tile-row (all Wt tiles forming one reduction row) fits simultaneously in L1 circular buffers, enabling multi-pass data reuse without re-reading from DRAM.

**Algorithm**: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))` for each row of width W.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`

**Compute kernel path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Reader kernel path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`

**Writer kernel path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | Wt tiles (one complete row of tiles across the W dimension) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop: N tile-rows assigned to this core. Inner: 4 sequential phases per tile-row |

One "work unit" is a single tile-row: all Wt tiles that share the same batch/channel/height-tile coordinate. The softmax reduction operates independently on each tile-row. The outer loop variable `N` (compile-time arg 0) counts how many tile-rows this core processes.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary batch dims) | [..., H, W] (same shape) |
| **Dimension convention** | Last dim = W (reduction dim) | Same as input |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
No tilize/untilize or reshard operations. Input and output are both in TILE_LAYOUT with INTERLEAVED memory. The operation is element-wise in structure (same shape in/out) but requires row-wide reductions internally.

---

## Data Flow Pattern

Each tile-row goes through **four sequential compute phases**, with specific CBs persisting across phases:

| Stage | Phase | Reads From | Writes To | Description |
|-------|-------|------------|-----------|-------------|
| 0 | Setup | Reader generates | cb_mask (c_1), cb_bcast_scaler (c_2) | One-time: scaler=1.0 tile, mask_w tile. Persist for entire program. |
| 1 | Read row | DRAM (via NoC) | cb_in0 (c_0) | Reader loads Wt tiles into cb_in0 |
| 2 | Phase 1: max | cb_in0, cb_mask, cb_bcast_scaler | cb_max (c_26) | Row-reduce MAX across Wt tiles -> 1 tile. cb_in0 tiles persist (not popped by reduce). |
| 3 | Phase 2: x-max | cb_in0, cb_max | cb_x_m_max (c_27) | Subtract max from each tile via COL broadcast. cb_in0 popped. cb_max popped. |
| 4 | Phase 3: exp+sum+recip | cb_x_m_max, cb_mask, cb_bcast_scaler | cb_exps (c_24), cb_recipsumexps (c_25) | exp(x-max) with masking on last tile. Row-reduce SUM, then recip. cb_exps persists (not popped by reduce). |
| 5 | Phase 4: normalize | cb_exps, cb_recipsumexps, cb_x_m_max | cb_out0 (c_16) | exp(x-max) * recip(sum). All intermediates popped. |
| 6 | Write row | cb_out0 | DRAM (via NoC) | Writer drains Wt tiles from cb_out0 |

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | Wt tiles | Wt tiles | Single | Reader | Compute | Row (loaded once, read by Phase 1+2, popped end of Phase 2) |
| c_1 | cb_mask | Mask tile (last-tile W padding) | 1 tile | 1 tile | Single | Reader | Compute | Program (generated once, never popped) |
| c_2 | cb_bcast_scaler | Scaler tile (1.0 for reduce) | 1 tile | 1 tile | Single | Reader | Compute | Program (generated once, never popped) |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Row (pushed Phase 4, drained by writer) |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row (produced Phase 3, consumed Phase 4) |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar tile | 1 tile | 1 tile | Single | Compute | Compute | Row (produced Phase 3, consumed Phase 4) |
| c_26 | cb_max | max(x) scalar tile | 1 tile | 1 tile | Single | Compute | Compute | Row (produced Phase 1, consumed Phase 2) |
| c_27 | cb_x_m_max | x - max(x) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row (produced Phase 2, consumed Phase 3+4) |
| c_28 | cb_tmp | Temporary for masked tile | 1 tile | 1 tile | Single | Compute | Compute | Transient (used within Phase 1 only) |

### Intermediate Data Format
CBs c_24 through c_28 use `intermed_data_format` which is `Float32` when `fp32_dest_acc_en` is true, otherwise matches the input data format. This ensures intermediate accumulations maintain precision.

### Multi-Pass Data Reuse Patterns

**Critical insight for layer_norm adaptation**: Several CBs are designed for **multi-pass reuse** -- tiles are written once and read by multiple subsequent phases without re-reading from DRAM:

1. **cb_in0 (c_0)**: Loaded by reader, read by Phase 1 (MAX reduce with `WaitUpfrontNoPop` policy -- tiles NOT consumed), then read again by Phase 2 (subtract), and only then popped. This avoids a second DRAM read.

2. **cb_exps (c_24)**: Produced by Phase 3 (exp), kept alive via `WaitUpfrontNoPop` in the SUM reduce (tiles NOT consumed), then read again by Phase 4 (multiply by reciprocal), and only then popped.

3. **cb_x_m_max (c_27)**: Produced by Phase 2, consumed by Phase 3 (exp reads from it) and also by Phase 4 (in the LOG path only). In the SOFTMAX path, it is popped at end of Phase 4.

4. **cb_mask (c_1) and cb_bcast_scaler (c_2)**: Generated once by reader before the main loop. Waited on once at compute kernel start (`cb_wait_front`), never popped -- persist for the entire program lifetime across all tile-rows.

---

## Compute Kernel Deep Dive

### Initialization

```cpp
binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);
```

This is the **hardware startup** function for compute kernels. It configures:
- Unpacker hardware for input CBs (cb_in0, cb_bcast_scaler)
- Math pipeline synchronization
- Packer hardware for output CB (cb_out0)

It must be called once at the start of every compute kernel. The three arguments specify the primary input, secondary input, and output CBs for initial data format configuration.

### Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows this core processes (from `num_tiles_per_core_group_{1,2}`) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (width in tiles) |

### Preprocessor Defines

| Define | Value | Effect |
|--------|-------|--------|
| `SOFTMAX` | 1 | Always set for softmax. Disables `negative_tile()` before `exp_tile()` (which would give `exp(-x)` for softmin). Also selects `exp*recip(sum)` path instead of `x-max-log(sum)` path. |
| `FP32_DEST_ACC_EN` | 1 (optional) | When set, enables FP32 data format reconfiguration in `_with_dt` helpers. Affects `pack_tile_with_dt`, `copy_tile_init_with_dt`, and all `*_init_short_with_dt` functions. |

### Phase 1: Row-Maximum Reduction

**Purpose**: Compute `max(x)` across the Wt tiles of one tile-row, producing a single tile in cb_max.

**Two code paths based on Wt**:

#### Case Wt == 1 (single tile per row):
```cpp
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single());
```

- `mask_tile_to_cb(icb, maskcb, ocb, itile, mtile, pop, popm)`: Copies tile `itile` from `icb` and mask tile `mtile` from `maskcb` into DST registers, applies `mask_tile()` (zeros out padding columns), packs result into `ocb`. Here `pop0=0, popm=0` means neither input nor mask are consumed.
- The masked tile goes through a standard single-tile MAX REDUCE_ROW with default `WaitAndPopPerTile` policy.

#### Case Wt > 1 (multiple tiles per row):
```cpp
// Step A: Reduce first (Wt-1) tiles with WaitUpfrontNoPop
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_in0, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

// Step B: Mask the last tile
mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

// Step C: Accumulate the masked last tile into the existing max
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
    cb_tmp, cb_bcast_scaler, cb_max,
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_max, 1));
```

**Key details**:
- **Step A** uses `WaitUpfrontNoPop`: waits for all (Wt-1) tiles in cb_in0 at once, performs the MAX reduction across them using indexed tile access, does NOT pop tiles. Output goes to cb_max. The `ReduceInputBlockShape::row(Wt-1)` means 1 row of (Wt-1) columns.
- **Step B**: `mask_tile_to_cb` reads tile at index `Wt-1` from cb_in0 (the last tile), applies the width mask, writes to cb_tmp. `pop0=0` preserves cb_in0 tiles for Phase 2.
- **Step C** uses `Accumulate::at(cb_max, 1)`: The `iteration=1` (non-zero) triggers reload of the previous partial max from cb_max into DST before reducing the masked last tile. This implements a two-pass accumulation: first pass reduces tiles [0..Wt-2], second pass combines with tile [Wt-1].

**reduce helper signature** (for reference):
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

### Phase 2: Subtract Maximum (x - max)

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

**Broadcast pattern**: `BroadcastType::COL` means the single-tile operand (cb_max) is broadcast along the column dimension. Since max is a row-reduced value (each row of the 32x32 tile holds the max for that row), COL broadcast replicates each row's max value across all columns -- exactly what `x[row][col] - max[row]` requires.

**Key calls**:
- `sub_bcast_cols_init_short_with_dt(icb0, icb1)`: Reconfigures data format (if FP32_DEST_ACC_EN) then initializes the subtract-broadcast-columns operation. The `_with_dt` suffix means it handles data type reconfiguration for FP32 mode.
- `sub_tiles_bcast<BroadcastType::COL>(icb0, icb1, itile0, itile1, dst)`: Performs `DST[dst] = icb0[itile0] - broadcast_col(icb1[itile1])`.
- `pack_tile_with_dt(dst, ocb)`: Reconfigures pack format (if FP32_DEST_ACC_EN) then packs DST[dst] into output CB.

**Note**: cb_in0 tiles are indexed `w = 0..Wt-1` while cb_max is always index 0. All Wt output tiles are reserved upfront, then pushed in bulk.

**After this phase**: cb_in0 is popped (Wt tiles freed), cb_max is popped (1 tile freed), cb_x_m_max holds Wt tiles.

### Phase 3: Exponentiate and Sum

```cpp
cb_reserve_back(cb_exps, Wt);
cb_wait_front(cb_x_m_max, Wt);
for (uint32_t w = 0; w < Wt; ++w) {
    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x_m_max);
    copy_tile(cb_x_m_max, w, dst0);
    // SOFTMAX defined: skip negative_tile
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

**Exponentiation loop** (manual, tile-by-tile):
- `copy_tile_init_with_dt(cb)` / `copy_tile(cb, idx, dst)`: Loads tile at index `idx` from CB into DST register `dst`.
- `exp_tile_init()` / `exp_tile(dst)`: Computes element-wise `exp()` on DST[dst] using SFPU.
- On the **last tile** (w == Wt-1): mask is applied to zero out padding columns after exponentiation. This ensures padding elements contribute 0 to the sum. Uses two DST registers (dst0 for data, dst1 for mask).

**Note**: cb_x_m_max is NOT popped here. It persists because in the LOG path it is reused in Phase 4. In the SOFTMAX path it is popped later.

**Sum reduction with reciprocal**:
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

**Key details**:
- `WaitUpfrontNoPop`: Waits for all Wt tiles in cb_exps, does NOT pop them. This is critical because cb_exps tiles are needed again in Phase 4.
- `ReduceInputBlockShape::row(Wt)`: Single row of Wt columns.
- **Post-reduce lambda** `recip_tile(dst_idx)`: Applied after the SUM reduction completes, computing `1/sum(exp)` in-place in the DST register before packing. This fuses the reciprocal with the reduction, avoiding an extra CB write/read cycle.
- Output goes to cb_recipsumexps (1 tile).

### Phase 4: Final Normalization

```cpp
cb_reserve_back(cb_out0, Wt);
cb_wait_front(cb_x_m_max, Wt);
cb_wait_front(cb_recipsumexps, 1);
cb_wait_front(cb_exps, Wt);  // SOFTMAX path only

for (uint32_t w = 0; w < Wt; w += onetile) {
    // SOFTMAX path: exp(x - max) * (1/sum)
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

**Broadcast pattern**: `mul_tiles_bcast_cols` with `BroadcastType::COL` -- the 1/sum scalar tile (one value per row) is broadcast across all columns, so each element is multiplied by the row's reciprocal sum. This is the same COL broadcast mechanism as Phase 2.

**Key calls**:
- `mul_bcast_cols_init_short_with_dt(icb0, icb1)`: Reconfigures data format + initializes multiply-broadcast-columns.
- `mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst)`: `DST[dst] = icb0[itile0] * broadcast_col(icb1[itile1])`.

**After Phase 4**: All intermediate CBs are drained (cb_exps, cb_recipsumexps, cb_x_m_max all popped). cb_out0 has Wt result tiles.

### Alternate LOG Path (not active for softmax, documented for completeness)

When `LOG` is defined instead of `SOFTMAX`, Phase 3 uses `BulkWaitBulkPop` (pops cb_exps after reduce) with `log_tile` post-reduce op, and Phase 4 computes `(x - max) - log(sum)` using `sub_tiles_bcast<COL>` instead of multiply.

---

## Pipeline Pattern Summary

All CBs use **single buffering** (capacity == block size). The four compute phases execute strictly sequentially within each tile-row:
- Phase 1 produces cb_max, Phase 2 consumes it
- Phase 2 produces cb_x_m_max, Phase 3 reads it, Phase 4 consumes it
- Phase 3 produces cb_exps + cb_recipsumexps, Phase 4 consumes both

**Reader-Compute overlap**: cb_in0 has capacity = Wt = block size, so it is single-buffered. The reader can push the next row's data only after compute completes Phase 2 (which pops cb_in0). Similarly, **Compute-Writer overlap**: cb_out0 is single-buffered, so writer drains while compute works on the next row's Phases 1-3.

---

## Scalar/Constant CB Setup

The reader kernel generates two constant tiles that persist for the entire program:

### cb_bcast_scaler (c_2): Reduce Scaler
```cpp
generate_bcast_scaler<T>(cb_scaler, scaler);  // scaler = 1.0f
```
Creates a tile where the first 16 elements of each 256-element subtile quadrant are filled with the scaler value (1.0), rest are 0. This matches the layout expected by `reduce_tile` which reads from the first row/column of the scaler tile. Value 1.0 means the reduce sums without scaling.

### cb_mask (c_1): Width Mask
```cpp
generate_mask_w<T>(cb_mask, mask_w);  // mask_w = logical_W % 32 (or 32 if aligned)
```
Creates a tile where columns [0, mask_w) are 1.0 and columns [mask_w, 32) are 0.0. Used to zero out padding elements in the last tile of each row so they don't affect max/sum computations.

Both tiles are `cb_wait_front`-ed once at the beginning of the compute kernel and never `cb_pop_front`-ed, so they remain available across all iterations.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration over device compute grid) |
| **Grid dimensions** | grid_coord.x * grid_coord.y (full device compute grid) |
| **Total cores** | `num_cores` (up to grid_x * grid_y, may be fewer if work < cores) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tile-rows |
| **Load balancing** | Two-group split: group_1 gets ceil(total/cores) rows, group_2 gets floor(total/cores) rows |

The work unit count is `num_kernel_rows = (physical_volume / H / W) * Ht`, i.e., one unit per height-tile across all batch dimensions. `split_work_to_cores` divides these evenly, with remainder rows going to core_group_1.

Core iteration order in the runtime args loop: `core = {(i / core_h) + x_offset, (i % core_h) + y_offset}` -- this is **column-major** ordering (Y varies fastest).

---

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows this core processes |
| 1 | Wt | uint32_t | Width in tiles (number of tiles per reduction row) |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input dtype is FLOAT32, 0 for BFLOAT16 |
| 1+ | TensorAccessor args | ... | Appended by TensorAccessorArgs |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address in DRAM |
| 1 | num_tiles_per_core | uint32_t | Number of tile-rows (N) for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Width in tiles |
| 4 | scaler | uint32_t | Bit-cast float 1.0f for scaler tile generation |
| 5 | mask_w | uint32_t | Number of valid columns in last tile (1-32) |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_tiles_per_core | uint32_t | Number of tile-rows (N) for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Width in tiles |

---

## Kernel Implementations

### Reader Kernel (Brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_softmax_w | RISCV_0 | NOC0 | DRAM | cb_in0, cb_mask, cb_bcast_scaler | Read tiles, generate constants |

**What it provides to compute**: Before the main loop, generates cb_mask and cb_bcast_scaler (one-time constants). In the main loop, reads Wt tiles per iteration into cb_in0 via `noc_async_read_tile` with TensorAccessor-based addressing.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| moreh_softmax_w | Compute threads (unpack/math/pack) | N/A | cb_in0, cb_mask, cb_bcast_scaler | cb_out0 | MAX reduce, subtract bcast, exp, SUM reduce+recip, multiply bcast |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Key Logic** (all detailed above in the Phase descriptions):
- Uses `compute_kernel_lib::reduce` helper for both MAX and SUM reductions
- Uses `WaitUpfrontNoPop` input policy to enable multi-pass tile reuse
- Uses `Accumulate::at()` for two-step MAX with masked last tile
- Uses post-reduce lambda to fuse `recip_tile` into the SUM reduction
- Manual tile-by-tile loops for subtract, exp, and multiply phases
- All bcast operations use `BroadcastType::COL` (row-reduced scalar broadcast across columns)

### Writer Kernel (Brief)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_moreh_softmax_w | RISCV_1 | NOC1 | cb_out0 | DRAM | Write tiles |

**What it consumes from compute**: Waits for Wt tiles in cb_out0 each iteration, writes them to DRAM via `noc_async_write_tile` with TensorAccessor, then pops.

---

## Implementation Notes

### DST Register Usage
- Phases 1, 2, 4 use `dst0 = 0` for the primary tile being processed.
- Phase 3 (exp loop) uses `dst0 = 0` for data and `dst1 = 1` for the mask tile (only on last tile iteration).
- The reduce helper internally manages DST registers.

### The `_with_dt` Pattern
All moreh_common compute helpers use `_with_dt` suffix variants (e.g., `sub_bcast_cols_init_short_with_dt`, `copy_tile_init_with_dt`, `pack_tile_with_dt`). These wrap the standard APIs with conditional `reconfig_data_format()` / `pack_reconfig_data_format()` calls guarded by `#if defined FP32_DEST_ACC_EN`. This is necessary because FP32 accumulation mode requires explicit data format switching between operations that may use different CB data formats (input format vs intermediate Float32 format).

### tile_regs Protocol
Every tile operation follows the strict acquire-commit-wait-release protocol:
1. `tile_regs_acquire()` -- claim DST registers for math thread
2. Perform unpack + math operations
3. `tile_regs_commit()` -- release to pack thread
4. `tile_regs_wait()` -- wait for pack thread to be ready
5. `pack_tile()` -- pack from DST to CB
6. `tile_regs_release()` -- release DST registers

### Relevance to Layer Norm

For `y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta` over width W:

The softmax W-small pattern provides a direct template:
- **Mean (E[x])**: Use `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` with scaler = `1/W` (instead of 1.0). Tiles persist for reuse.
- **x - mean**: Use `sub_tiles_bcast<COL>` (exactly like Phase 2).
- **Variance**: Square the difference tiles, then `reduce<SUM, REDUCE_ROW>` with scaler = `1/W`.
- **1/sqrt(var + eps)**: Add eps (via `add_bcast_scalar` or as part of the reduce post-op), then use `rsqrt_tile` as a post-reduce op (analogous to `recip_tile` in softmax).
- **Scale by gamma**: `mul_tiles_bcast_cols` or `mul_tiles` (if gamma is per-column, use COL broadcast).
- **Add beta**: `add_tiles_bcast_cols` or `add_tiles`.

CB layout would be similar: input (Wt), mean (1), x-mean (Wt), variance (1), rstd (1), output (Wt), plus gamma/beta (Wt each if weight/bias exist).

---

## External Knowledge Sources

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding Tensix core architecture, tile-based compute model, and kernel threading
   **Key Information**: Each Tensix core has 5 RISC-V CPUs (2 data movement, 3 compute: unpack/math/pack). The tile_regs acquire/commit/wait/release protocol mediates between math and pack threads.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the unified reduce helper API used by the compute kernel
   **Key Information**: The `reduce` function supports multiple input policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), accumulation (Accumulate type for multi-pass reductions), and post-reduce operations (lambda applied in DST before packing). REDUCE_ROW reduces the W dimension. The scaler CB must be pre-populated.

3. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding the `_with_dt` helper pattern and `mask_tile_to_cb` composite
   **Key Information**: `_with_dt` helpers wrap standard compute APIs with FP32 data format reconfiguration. `mask_tile_to_cb` copies a data tile and mask tile into DST, applies `mask_tile`, and packs. Pop parameters control whether inputs are consumed.

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Understanding `binary_op_init_common` hardware startup
   **Key Information**: Configures unpacker, math pipeline, and packer hardware. Must be called once at compute kernel start. Parameters set the initial data format configuration.

5. **Source**: `tt_metal/hw/inc/api/compute/bcast.h`
   **Reason**: Understanding broadcast type semantics (COL vs ROW)
   **Key Information**: `BroadcastType::COL` broadcasts the operand along the column direction -- each row of the scalar tile is replicated to all columns. This is correct for row-reduced values (max, sum) applied element-wise across a tile-row.

6. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding `generate_bcast_scaler` and `generate_mask_w` tile generation
   **Key Information**: `generate_bcast_scaler` fills the first 16 elements of each subtile quadrant with the scaler value. `generate_mask_w` creates a binary mask tile with 1.0 for valid columns and 0.0 for padding.

7. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.cpp` (lines 58-93)
   **Reason**: Understanding `split_work_to_cores_wt_core_range` work distribution
   **Key Information**: Delegates to `tt_metal::split_work_to_cores` which divides units evenly across grid, producing two core groups (one with ceil, one with floor work units). Column-major core ordering with offset support.
