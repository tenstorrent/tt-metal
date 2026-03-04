# Softmax General Implementation Analysis

## Overview

The "softmax general" operation implements `softmax(x, dim)` for arbitrary reduction dimensions on Tenstorrent hardware. Unlike the attention-optimized softmax (which handles only the last dimension of 4D tensors with optional scale/mask), this general variant supports softmax along width (W), height (H), or any other dimension (C-style) of tensors with arbitrary rank.

The implementation is dispatched across **five** dimension-specific and size-specific program factories:

| Factory | Dimension | Size Constraint | When Selected |
|---------|-----------|-----------------|---------------|
| `SoftmaxProgramFactoryGeneralWSmall` | W (last dim) | All CBs fit in L1 < 512KB | `dim == rank-1` and small |
| `SoftmaxProgramFactoryGeneralWLarge` | W (last dim) | Does not fit | `dim == rank-1` and large |
| `SoftmaxProgramFactoryGeneralHSmall` | H (second-to-last) | All CBs fit in L1 < 512KB | `dim == rank-2` and small |
| `SoftmaxProgramFactoryGeneralHLarge` | H (second-to-last) | Does not fit | `dim == rank-2` and large |
| `SoftmaxProgramFactoryGeneralCLarge` | Any other dim | Always "large" style | `dim < rank-2` |

The "small" variants load all tiles along the reduction dimension into L1 simultaneously, enabling efficient batch processing. The "large" variants stream tiles one-at-a-time through double-buffered CBs, trading throughput for reduced L1 pressure. The "large" variants require **three passes** over the input data (for max, exp+sum, normalize), while "small" variants read once and reuse data from L1.

**Program factory paths**:
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_small.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_h_large.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_c_large.cpp`

**Kernel source path**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles along the reduction dimension |
| **Unit size** | Dt tiles (where Dt = tiles along reduction dim: Wt, Ht, or dim_size) |
| **Total units** | `physical_volume / (reduction_dim_size * inner_dims_size) * inner_tiles` |
| **Loop structure** | Outer: N work units per core; Inner: Dt tiles per reduction |

A single work unit represents one complete softmax reduction: all tiles along the reduction dimension for a fixed position in the non-reduced dimensions. For example, with W-dimension softmax on a `[B, C, H, W]` tensor, one work unit is one tile-row (Wt tiles) at a fixed (b, c, h_tile) position.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary rank (>= 2), e.g. [N, C, H, W] |
| **Dimension convention** | NHWC-style, last two dims are H, W |
| **Tensor layout** | TILE_LAYOUT (required) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | BFLOAT16, FLOAT32, or BFLOAT8_B |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (typical) or L1 |
| **Data type** | Same as input |

### Layout Transformations
- No tilize/untilize performed inside the kernels. Input must already be in TILE_LAYOUT.
- The host-side `softmax()` function may call `ttnn::tilize_with_val_padding()` to convert row-major inputs before dispatching (padding with `-inf`).
- When `fp32_dest_acc_en` is true, intermediate CBs use Float32 format regardless of input type, providing higher precision for reduction operations.

## Data Flow Pattern

The softmax algorithm for `SOFTMAX` define (as opposed to `LOG` softmax) follows a three-phase reduction:

### Phase 1: Find Max (Numerical Stability)
```
Reader -> cb_in0 -> [REDUCE_MAX along dim] -> cb_max (1 tile)
```

### Phase 2: Compute exp(x - max) and Sum
```
cb_in0 (reread for large, reuse for small) -> [x - max] -> cb_x_m_max/cb_tmp -> [exp] -> cb_exps -> [REDUCE_SUM] -> cb_recipsumexps (after recip)
```

### Phase 3: Normalize
```
cb_in0 (reread for large, reuse from x_m_max for small) -> [x - max] -> [exp] -> [* 1/sum] -> cb_out0 -> Writer -> DRAM
```

### Small vs Large Data Flow

**Small variants** (W_small, H_small): The reader loads ALL Dt tiles into `cb_in0` at once. The compute kernel processes them in-place, accessing tiles by index within the CB. The data is read from DRAM only **once** per work unit. Key distinguishing feature: `cb_reserve_back(cb_in, Dt)` in reader, and `cb_wait_front(cb_in0, Dt)` in compute.

**Large variants** (W_large, H_large, C_large): The reader sends the same data **three times** (three separate DRAM read passes per work unit). Each pass feeds one phase of the computation. CBs are double-buffered (capacity=2 tiles), so tiles stream through one-at-a-time. This is necessary because L1 cannot hold all tiles simultaneously.

## Circular Buffer Configuration

### W-Small Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | Wt tiles | Wt tiles | Single | Reader | Compute | Block |
| c_1 | cb_mask | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Broadcast scaler (1.0) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Block |
| c_24 | cb_exps | exp(x - max) | Wt tiles | Wt tiles | Single | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_26 | cb_max | Row-wise max | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_27 | cb_x_m_max | x - max(x) | Wt tiles | Wt tiles | Single | Compute | Compute | Block |
| c_28 | cb_tmp | Temp scratch | 1 tile | 1 tile | Single | Compute | Compute | Block |

### W-Large Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_mask | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Broadcast scaler | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_24 | cb_exps | exp(x-max) | 2 tiles | 1 tile | Double | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_26 | cb_add | Running sum accumulator | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_27 | cb_max | Max accumulator | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_28 | cb_tmp | Temp scratch | 1 tile | 1 tile | Single | Compute | Compute | Block |

### H-Small and H-Large

Same pattern as W variants but with `Ht` replacing `Wt` for CB capacities in the small variant, and `REDUCE_COL` instead of `REDUCE_ROW`.

### C-Large

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_16 | cb_out0 | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_24 | cb_exps | exp(x-max) | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_26 | cb_add | Running sum accumulator | 2 tiles | 1 tile | Double | Compute | Compute | Block |
| c_27 | cb_max | Max value | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_28 | cb_tmp | Temp scratch | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Note**: C-large has no mask or scaler CBs because tile-wise element masking is only needed for partial tiles at the edge of W or H dimensions. The C dimension operates on whole tiles.

## Pipeline Pattern Summary

| Variant | CB_in0 Buffering | CB_out0 Buffering | Reader-Compute Overlap | Compute-Writer Overlap |
|---------|------------------|-------------------|------------------------|------------------------|
| W-small | Single (Wt) | Single (Wt) | No (bulk load) | No (bulk write) |
| W-large | Double (2) | Double (2) | Yes (tile-level) | Yes (tile-level) |
| H-small | Single (Ht) | Single (Ht) | No (bulk load) | No (bulk write) |
| H-large | Double (2) | Double (2) | Yes (tile-level) | Yes (tile-level) |
| C-large | Double (2) | Double (2) | Yes (tile-level) | Yes (tile-level) |

For "small" variants, the entire reduction dimension is loaded/stored at once, so there is no tile-level overlap. For "large" variants, double-buffering enables reader-compute and compute-writer overlap at the tile granularity.

## Index Calculations

### W-dimension (Small and Large)
- **Work unit index**: Linear tile offset. Each work unit is a tile row of Wt tiles.
- **DRAM tile address**: Sequential within a row: `tile_offset + w` for `w in [0, Wt)`.
- **tile_offset** advances by `Wt` per work unit (for small: `tile_offset += num_tiles_per_core * Wt`).

### H-dimension (Small and Large)
- **Work unit**: One tile column (Ht tiles at a fixed width position).
- **curr_tile**: Encodes `(nc_idx, w_idx)` where `w_idx = curr_tile % Wt`, `nc_idx = curr_tile / Wt`.
- **DRAM tile address**: `nc_idx * Ht * Wt + w_idx + h * Wt` for `h in [0, Ht)`. Tiles are strided by Wt in memory (jumping across tile rows).

### C-dimension (Large only)
- **Work unit**: One tile-position across the arbitrary dimension.
- **curr_tile**: Encodes `(outer_idx, inner_idx)` where `outer_idx = curr_tile / inner_size`, `inner_idx = curr_tile % inner_size`.
- **outer_stride**: Product of all dimensions from dim to last, in tiles.
- **inner_size**: `outer_stride / dim_size` -- tiles below the reduction dimension.
- **dim_stride**: `inner_size` -- step between consecutive slices along the reduction dimension.
- **DRAM tile address**: `outer_idx * outer_stride + inner_idx + d * inner_size` for `d in [0, dim_size)`.

All variants use `TensorAccessor` for DRAM address resolution, which handles interleaved bank mapping.

## Memory Access Patterns

### Read Pattern

**W-small**: Sequential burst reads. Reader does `cb_reserve_back(cb_in, Wt)` then reads Wt contiguous tiles into CB, followed by barrier. Maximally coalesced since tiles along W are contiguous in memory.

**W-large**: Three-pass streaming. Reader reads the same Wt tiles three times (one tile at a time with individual barriers), feeding data for max, exp+sum, and normalize phases. Each read is a single tile with its own `noc_async_read_barrier()`.

**H-small**: Strided burst. Reader does `cb_reserve_back(cb_in, Ht)` then reads Ht tiles with stride Wt between them (jumping rows). Single barrier after all reads.

**H-large**: Three-pass strided streaming. Same strided pattern as H-small but repeated three times, one tile at a time.

**C-large**: Three-pass strided streaming. Tiles along the reduction dimension are separated by `inner_size` stride. One tile at a time with individual barriers.

### Write Pattern

**W-small/H-small**: Bulk write of all output tiles at once (Wt or Ht tiles), single barrier.

**W-large/H-large/C-large**: One tile at a time with individual `noc_async_write_barrier()` per tile.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses full compute grid) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (device-dependent, e.g. 8x8) |
| **Total cores** | min(total_work_units, grid_x * grid_y) |
| **Work per core** | `ceil(total_work_units / total_cores)` or `floor(...)` |
| **Load balancing** | Two-group split: core_group_1 gets N+1 units, core_group_2 gets N units |

The work splitting uses `split_work_to_cores_wt_core_range()` which wraps tt_metal's `split_work_to_cores()`. This distributes work units as evenly as possible across available cores, with two core groups handling the remainder. Cores are enumerated column-major: `core = {i / core_h, i % core_h}`.

**Total work units by variant**:
- W: `num * Ht` (where num = batch*channel product, one unit per tile row)
- H: `num * Wt` (one unit per tile column)
- C: `physical_volume / shape[dim] / H / W * Ht * Wt` (one unit per spatial tile position)

## Arguments

### Compile-Time Arguments

#### Reader Kernel (W/H variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input dtype is FLOAT32, 0 otherwise |
| 1+ | TensorAccessorArgs | uint32_t[] | Bank mapping parameters for input tensor |

#### Reader Kernel (C-large)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Bank mapping parameters for input tensor |

#### Writer Kernel (all variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Bank mapping parameters for output tensor |

#### Compute Kernel (all variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of work units for this core |
| 1 | Dt | uint32_t | Tiles along reduction dimension (Wt, Ht, or dim_size) |

#### Compute Defines

| Define | Value | Description |
|--------|-------|-------------|
| `SOFTMAX` | `"1"` | Always set; selects softmax (vs. softmin) code paths |
| `FP32_DEST_ACC_EN` | `"1"` | Set when fp32_dest_acc_en is true; uses Float32 intermediate format |

### Runtime Arguments

#### Reader (W variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor buffer address |
| 1 | N | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles along width dimension |
| 4 | scaler | uint32_t | Broadcast scaler value (bit-cast float 1.0) |
| 5 | mask_w | uint32_t | Number of valid elements in last width tile |

#### Reader (H variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor buffer address |
| 1 | N | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index (encodes nc_idx, w_idx) |
| 3 | Ht | uint32_t | Tiles along height dimension |
| 4 | Wt | uint32_t | Tiles along width (stride) |
| 5 | scaler | uint32_t | Broadcast scaler value (bit-cast float 1.0) |
| 6 | mask_h | uint32_t | Number of valid elements in last height tile |

#### Reader (C-large)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor buffer address |
| 1 | num_tiles | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | outer_stride | uint32_t | Stride for the outer dimension loop |
| 4 | inner_size | uint32_t | Number of tiles in dimensions below the reduction dim |
| 5 | dim_size | uint32_t | Size of the reduction dimension |

#### Writer (W variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor buffer address |
| 1 | N | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles along width dimension |

#### Writer (H variants)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor buffer address |
| 1 | N | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Ht | uint32_t | Tiles along height dimension |
| 4 | Wt | uint32_t | Tiles along width (stride) |

#### Writer (C-large)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor buffer address |
| 1 | num_tiles | uint32_t | Number of work units for this core |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | outer_stride | uint32_t | Stride for outer dimension loop |
| 4 | inner_size | uint32_t | Number of tiles in dimensions below reduction dim |
| 5 | dim_size | uint32_t | Size of the reduction dimension |

## Kernel Implementations

### W-Small Compute: `moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_softmax_w | RISCV_0 | NOC0 | DRAM | c_0, c_1, c_2 | Read Wt tiles/row, generate mask+scaler |
| moreh_softmax_w | Compute | N/A | c_0, c_1, c_2 | c_16 | Max-reduce, sub, exp, sum-reduce, recip, mul |
| writer_moreh_softmax_w | RISCV_1 | NOC1 | c_16 | DRAM | Write Wt tiles/row |

**Key Logic (moreh_softmax_w.cpp - small)**:
1. **Max reduction**: Uses `compute_kernel_lib::reduce<MAX, REDUCE_ROW>` with `WaitUpfrontNoPop` policy. All Wt tiles are waited on upfront, and the last tile is masked via `mask_tile_to_cb` before accumulation. Result in `cb_max`.
2. **Subtract max**: `sub_tiles_bcast<COL>(cb_in0, cb_max, w, 0, dst0)` -- broadcasts max across columns, subtracts from each tile. Result stored in `cb_x_m_max`.
3. **Exp + mask**: `exp_tile()` on each tile of `cb_x_m_max`. The last tile is additionally masked to zero out padding elements.
4. **Sum reduction**: `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` with `WaitUpfrontNoPop` policy, followed by `recip_tile()` post-processing. Result: `1/sum` in `cb_recipsumexps`.
5. **Normalize**: `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)` -- broadcasts `1/sum` and multiplies with exp results. Output to `cb_out0`.

### W-Large Compute: `moreh_softmax_w_large.cpp`

**Key Logic**:
Same algorithm as W-small but processes tiles one-at-a-time with streaming input. The reader provides the same data three times:
- **Pass 1 (Max)**: Tiles streamed and reduced via `reduce<MAX, REDUCE_ROW>` one tile at a time (with pop). Last tile masked.
- **Pass 2 (Exp + Sum)**: Each tile: `sub_tiles_bcast_cols_to_cb` -> `exp_tile_to_cb` -> accumulated via `add_tiles_to_cb` into `cb_add`. After loop, `reduce<SUM, REDUCE_ROW>` on accumulated sum, then `recip_tile`.
- **Pass 3 (Normalize)**: Each tile: `sub_tiles_bcast_cols_to_cb` -> `exp_tile_to_cb` -> `mul_tiles_bcast_cols_to_cb` with `1/sum`.

### H-Small/Large Compute: `moreh_softmax_h.cpp` / `moreh_softmax_h_large.cpp`

Structurally identical to W variants but uses:
- `REDUCE_COL` instead of `REDUCE_ROW`
- `sub_tiles_bcast<ROW>` / `sub_bcast_rows_init_short_with_dt` instead of COL variants
- `mul_tiles_bcast_rows` instead of `mul_tiles_bcast_cols`
- `mask_h` (height masking) instead of `mask_w`

### C-Large Compute: `moreh_softmax_c_large.cpp`

Operates on individual tiles (element-wise across the C dimension):
- **Max**: Uses `binary_max_tile()` to iteratively find max across dim_size tiles.
- **Exp+Sum**: `sub_tiles_to_cb` + `exp_tile_to_cb` + `add_tiles_to_cb` for accumulation.
- **Recip**: `recip_tile_to_cb` on accumulated sum.
- **Normalize**: `sub_tiles_to_cb` + `exp_tile_to_cb` + `mul_tiles_to_cb`.

No broadcast operations are needed since each tile along C represents a single reduction element (not a row or column within a tile).

### Reader Kernels

**W-small reader** (`reader_moreh_softmax_w.cpp`): Generates `mask_w` and `scaler` tiles at startup. Reads Wt contiguous tiles per work unit in a bulk `cb_reserve_back(cb_in, Wt)`.

**W-large reader** (`reader_moreh_softmax_w_large.cpp`): Same mask/scaler generation. Reads each tile individually with `cb_reserve_back(cb_in, 1)`. Repeats the entire tile row **three times** (three separate loops over the same `curr_offset_i` range).

**H-small reader** (`reader_moreh_softmax_h.cpp`): Generates `mask_h` and `scaler`. Reads Ht tiles in bulk with stride Wt between them.

**H-large reader** (`reader_moreh_softmax_h_large.cpp`): Same as H-small but one tile at a time, three passes.

**C-large reader** (`reader_moreh_softmax_c_large.cpp`): No mask/scaler. Reads dim_size tiles with stride `inner_size` between them, three passes.

### Writer Kernels

All writers use `TensorAccessor` for output address resolution. Small variants do bulk waits and writes; large variants write one tile at a time.

## Multi-Pass Reduction Strategy (Detailed)

The softmax computation `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` requires three dependent computations:

### Pass 1: Max Reduction
- **Goal**: Find `max(x)` along the reduction dimension for numerical stability.
- **W/H variants**: Use the hardware-accelerated `compute_kernel_lib::reduce<MAX, REDUCE_ROW/COL>` helper which performs tile-level max reduction. The last tile is masked first to exclude padding elements.
- **C-large**: Uses iterative `binary_max_tile()` since the reduction is across tiles (not within a tile).
- **Output**: Single tile in `cb_max` containing per-row (or per-column) maximum values.

### Pass 2: Exp and Sum
- **Goal**: Compute `exp(x - max(x))` and `sum(exp(x - max(x)))`.
- **Subtract**: `x - max` using broadcast subtraction (COL broadcast for W, ROW broadcast for H, element-wise for C).
- **Exp**: `exp_tile()` applied to each shifted tile. Last tile masked to zero padding.
- **Sum**: For W/H-small, the reduce helper sums all Dt exp tiles. For large variants, running accumulation via `add_tiles_to_cb` builds the sum tile-by-tile, followed by a final row/col reduce.
- **Reciprocal**: `recip_tile()` converts sum to `1/sum` for multiplication in pass 3.
- **Output**: `1/sum` in `cb_recipsumexps`.

### Pass 3: Normalize
- **Goal**: Compute `exp(x - max(x)) * (1/sum)`.
- **Small variants**: Reuse `cb_exps` (kept from pass 2) and `cb_x_m_max`. Multiply each exp tile by the broadcast `1/sum`.
- **Large variants**: Must recompute `exp(x - max)` since tiles were consumed in pass 2. Reader provides the third copy of input data.
- **Output**: Final softmax result tiles pushed to `cb_out0`.

### Why Three Passes for Large Variants?
The max value must be known before computing exp (for numerical stability), and the sum must be known before normalization. Since not all tiles fit in L1 simultaneously, the input data must be re-read for each dependent phase. The reader explicitly loops three times over the same tile range with `curr_tile = curr_offset_i` resets.

## Implementation Notes

1. **Shared kernels with moreh_softmax**: The general softmax reuses kernel code from `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`. The `SOFTMAX` define selects forward softmax paths (vs. softmin via `#ifndef SOFTMAX` which negates before exp).

2. **LOG softmax support**: The kernels also support log-softmax via the `LOG` define (not set by the general softmax factory). When `LOG` is defined, the sum phase uses `log_tile()` instead of `recip_tile()`, and the normalize phase computes `x - max - log(sum)` instead of `exp(x-max) * (1/sum)`.

3. **Numerical stability**: The `numeric_stable` parameter is passed through `SoftmaxParams` but the general factories always perform the max-subtraction stabilization. The max reduction ensures no overflow in the exp computation.

4. **Mask generation**: `generate_mask_w<T>()` and `generate_mask_h<T>()` create tiles with 1s for valid positions and 0s for padding, generated once at reader startup and held in `cb_mask` for the program lifetime.

5. **Scaler generation**: `generate_bcast_scaler<T>()` creates a tile filled with the scaler value (always 1.0 for general softmax, since there's no attention scaling). Used by the reduce helpers as a multiplication factor during reduction.

6. **FP32 intermediate accumulation**: When `fp32_dest_acc_en` is set, intermediate CBs (c_24 through c_28) use Float32 data format even if the input is BFLOAT16. This prevents precision loss during the multi-step reduction.

7. **Program caching**: The `SoftmaxProgramFactoryGeneral` (the thin wrapper) only updates buffer addresses in `override_runtime_arguments()`. The actual program factories handle full creation. The program hash includes input shape, dtype, memory config, and all operation attributes.

8. **L1 threshold for small/large selection**: The `is_softmax_general_w_small_available()` function calculates total CB memory (input + mask + scaler + output + exp + reduce + max + x_m_max + tmp) and checks if `base_allocator_addr + cb_usage <= 512KB`. This conservative threshold leaves headroom for other L1 usage.

## External Knowledge Sources

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the Tensix core architecture (5 RISC-V CPUs, circular buffer mechanism, NoC data movement)
   **Key Information**: Reader kernel runs on RISCV_0 with NOC0, writer on RISCV_1 with NOC1, compute unpacks/maths/packs on RISCV_2/3/4. CBs are the synchronization mechanism between kernels.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how `TensorAccessor` and `TensorAccessorArgs` map tile indices to DRAM bank addresses
   **Key Information**: TensorAccessor encapsulates interleaved bank mapping, allowing `noc_async_read_tile(tile_id, accessor, l1_addr)` to resolve physical addresses.

3. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tile layout and interleaved memory layout used by the softmax operation
   **Key Information**: In TILE_LAYOUT with INTERLEAVED memory, tiles are distributed across DRAM banks in round-robin fashion. Page size equals tile size.

4. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp` and `.cpp`
   **Reason**: Understanding `split_work_to_cores_wt_core_range()`, `CreateCircularBuffer()`, `CreateReadKernel()`, `CreateWriteKernel()`, `CreateComputeKernel()` helpers
   **Key Information**: These are convenience wrappers around tt_metal APIs. `split_work_to_cores_wt_core_range` wraps `tt_metal::split_work_to_cores()` with core range offset support, producing two core groups for remainder handling.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding `compute_kernel_lib::reduce<PoolType, ReduceDim, Policy>()` template used extensively in compute kernels
   **Key Information**: Provides hardware-accelerated tile reduction with configurable accumulation policies (WaitUpfrontNoPop for small variants, BulkWaitBulkPop for large), post-processing lambdas (recip, log), and block shape specification.
