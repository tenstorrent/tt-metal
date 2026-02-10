# Softmax General (W-dimension) Implementation Analysis

## Overview

The **Softmax General** program factory implements numerically-stable softmax along the W (innermost) dimension of a tensor. It is part of the `ttnn::prim` namespace and is selected when the softmax dimension is the last dimension and the operation type does not use attention-specific features (scale, mask, in-place, etc.).

There are **two variants** for W-dimension softmax:

| Variant | When Selected | CB Capacity Strategy |
|---------|---------------|---------------------|
| **WSmall** | All CBs fit in L1 (< 512KB) | Capacity = Wt tiles (all tiles of a row loaded at once) |
| **WLarge** | CBs exceed L1 budget | Capacity = 2 tiles (streaming, one tile at a time with double-buffering) |

Both variants share the same `override_runtime_arguments` (in `softmax_program_factory_general.cpp`) and inherit from `SoftmaxProgramFactoryGeneral`.

**Primary program factory files analyzed**:
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_large.cpp`

**Kernel files**:
- Reader: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp` (WSmall)
- Reader: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w_large.cpp` (WLarge)
- Compute: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp` (WSmall)
- Compute: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp` (WLarge)
- Writer: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp` (WSmall)
- Writer: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w_large.cpp` (WLarge)

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | row (one tile-row across W dimension) |
| **Unit size** | Wt tiles (one complete row of tiles) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop: N rows assigned to this core. Inner loops: Wt tiles per row. |

One "work unit" is a single **tile-row**: all Wt tiles that form one horizontal strip of tiles across the W dimension. The softmax reduction operates across these Wt tiles per row to compute max, sum(exp), and the final normalized output.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary rank (flattened to [..., H, W]) |
| **Dimension convention** | Last two dims are H, W |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16 or FLOAT32 (BFLOAT8_B also accepted at validation) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input |

### Layout Transformations
- No tilize/untilize within the operation itself. Input must already be in TILE_LAYOUT (enforced by validation).
- The higher-level `softmax()` wrapper may call `tilize_with_val_padding` with `-inf` padding if the input is in ROW_MAJOR layout.

## Data Flow Pattern

### WSmall Variant (All tiles fit in L1)

The WSmall variant loads all Wt tiles of a row into L1 simultaneously, allowing the compute kernel to access any tile in the row by index without re-reading from DRAM.

```
For each tile-row n in [0, N):
  Reader: Read Wt tiles from DRAM into cb_in0 (bulk)
  Compute:
    Phase 1: MAX reduction across Wt tiles (row-reduce) -> cb_max (1 tile, col-0 holds per-row maxima)
    Phase 2: x - max(x) for all Wt tiles -> cb_x_m_max (Wt tiles)
    Phase 3: exp(x - max(x)) for all Wt tiles -> cb_exps (Wt tiles), apply mask on last tile
    Phase 4: SUM reduction of exp tiles -> reciprocal -> cb_recipsumexps (1 tile, col-0 holds 1/sum)
    Phase 5: exp(x-max) * (1/sum) for all Wt tiles -> cb_out0 (Wt tiles)
  Writer: Write Wt tiles from cb_out0 to DRAM (bulk)
```

### WLarge Variant (Streaming, one tile at a time)

The WLarge variant cannot hold all Wt tiles simultaneously, so the reader sends the row data **three times** (once for max, once for exp+sum, once for final output), and the compute kernel processes tiles one at a time.

```
For each tile-row n in [0, N):
  Reader Pass 1: Stream Wt tiles one-at-a-time for max reduction
  Compute: MAX reduction, streaming tiles -> cb_max

  Reader Pass 2: Stream Wt tiles one-at-a-time for exp(x-max) + running sum
  Compute: For each tile: sub max, exp, accumulate sum in cb_add -> then recip(sum) -> cb_recipsumexps

  Reader Pass 3: Stream Wt tiles one-at-a-time for final normalization
  Compute: For each tile: sub max, exp, mul by 1/sum -> cb_out0
  Writer: Write each output tile to DRAM
```

## Circular Buffer Configuration

### WSmall Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_mask | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Broadcast scaler (1.0) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_max | Row-wise max result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | x - max(x) intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary scratchpad | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Data format**: CBs c_24 through c_28 use `intermed_data_format` (Float32 if `fp32_dest_acc_en`, otherwise same as input). All other CBs use the input data format.

### WLarge Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_mask | Width mask tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_bcast_scaler | Broadcast scaler (1.0) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_out0 | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_24 | cb_exps | exp(x-max) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Block |
| c_25 | cb_recipsumexps | 1/sum(exp) result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_add | Running sum accumulator | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_max | Row-wise max result | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_28 | cb_tmp | Temporary scratchpad | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Key difference**: WLarge uses capacity=2 for streaming CBs (c_0, c_16, c_24) enabling double-buffering, whereas WSmall uses capacity=Wt for bulk access.

## Pipeline Pattern Summary

### WSmall
- **cb_in0, cb_out0, cb_exps, cb_x_m_max**: Single-buffered at row granularity (capacity = Wt = block size). Reader fills entire row before compute starts. No overlap between reader and compute within a row.
- **cb_mask, cb_bcast_scaler**: Generated once by reader, consumed throughout the program. No pipeline overlap needed.
- **cb_max, cb_recipsumexps, cb_tmp**: Single-tile intermediates, produced and consumed within compute.

### WLarge
- **cb_in0, cb_out0, cb_exps**: Double-buffered (capacity=2, block=1). Allows reader/compute or compute/writer overlap within a pass. The reader can push the next tile while compute processes the current one.
- **Other CBs**: Same as WSmall.

## Index Calculations

### Tile Indexing
- **Linear tile offset**: `tile_offset` is computed per-core as the cumulative sum of tiles assigned to previous cores: `tile_offset += num_tiles_per_core * Wt`.
- **Row-major iteration**: Within each core, tiles are iterated as `curr_tile = tile_offset + (row * Wt) + w`, where `row` is the tile-row index and `w` is the column-tile index within that row.
- **TensorAccessor**: Both reader and writer use `TensorAccessor` constructed from `TensorAccessorArgs` to resolve tile indices to physical DRAM addresses via `noc_async_read_tile` / `noc_async_write_tile`.

### Mask Index Calculation
- `mask_w = logical_shape[-1] % TILE_WIDTH` (or `TILE_WIDTH` if evenly divisible).
- This determines how many valid elements exist in the last tile of each row. The mask tile has 1.0 for valid positions and 0.0 for padding positions. It is applied to the last tile in each row to zero out padded elements before reduction.

### Scaler Generation
- The scaler value is `1.0f`, passed as a bit-reinterpreted `uint32_t` via `*reinterpret_cast<uint32_t*>(&scaler)`.
- `generate_bcast_scaler<T>` fills a tile with the scaler value in the reduce-compatible layout: the first 16 elements of each of the 4 sub-tile faces are set to the scaler value (when bfloat16: upper 16 bits of the float; when float32: the full 32-bit value). All other positions are zero.

## Memory Access Patterns

### Read Pattern (WSmall)
- **Bulk sequential**: Reader reads Wt contiguous tiles per row in one burst. `cb_reserve_back(cb_in, Wt)` followed by Wt sequential `noc_async_read_tile` calls, then `noc_async_read_barrier` and `cb_push_back(cb_in, Wt)`.
- **DRAM access**: Sequential tile reads within each row (contiguous in memory for interleaved layout).

### Read Pattern (WLarge)
- **Streaming with re-reads**: Reader sends the same row of tiles **3 times** (3 passes). Each pass reads tiles one-at-a-time with per-tile `cb_reserve_back(1)` / `cb_push_back(1)`.
- Pass 1: For max reduction.
- Pass 2: For exp(x-max) and sum computation.
- Pass 3: For final normalization (sub, exp, mul).

### Write Pattern
- **WSmall**: Bulk write of Wt tiles per row. `cb_wait_front(cb_out, Wt)` then Wt sequential `noc_async_write_tile` calls.
- **WLarge**: Streaming write, one tile at a time. `cb_wait_front(cb_out, 1)` per tile.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration over grid) |
| **Grid dimensions** | `grid_coord.x` x `grid_coord.y` (device max compute grid) |
| **Total cores** | `num_cores` (up to total available compute cores) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tile-rows |
| **Load balancing** | Two groups: group 1 gets `ceil(num_kernel_rows / num_cores)` rows, group 2 gets `floor(...)` rows |

- `num_kernel_rows = (physical_volume / H / W) * Ht` -- total number of tile-rows across all batches.
- `split_work_to_cores_wt_core_range` delegates to `tt_metal::split_work_to_cores` which divides `num_kernel_rows` across available cores with at most 1 row difference between groups.
- Core mapping: `core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset}` -- column-major ordering.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t | 1 if input dtype is FLOAT32, 0 otherwise |
| 1+ | TensorAccessorArgs | varies | Buffer metadata for input tensor (page size, bank mapping, etc.) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | varies | Buffer metadata for output tensor |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of tile-rows assigned to this core |
| 1 | Wt | uint32_t | Number of tiles along W dimension |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles along W dimension |
| 4 | scaler | uint32_t | Bit-cast float 1.0 for reduce scaler |
| 5 | mask_w | uint32_t | Number of valid elements in last tile column (1-32) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | N | uint32_t | Number of tile-rows for this core |
| 2 | tile_offset | uint32_t | Starting tile index for this core |
| 3 | Wt | uint32_t | Tiles along W dimension |

## Kernel Implementations

### Reader Kernel (WSmall: `reader_moreh_softmax_w.cpp`)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | cb_in0, cb_mask, cb_scaler | Read input tiles, generate mask and scaler |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`

**Key Logic**:
1. Generates the broadcast scaler tile (1.0) into `cb_scaler` using `generate_bcast_scaler<T>`. The scaler fills the first 16 elements of each 256-element sub-tile face. This is the format expected by the reduce hardware.
2. Generates the width mask tile into `cb_mask` using `generate_mask_w<T>`. The mask has 1.0 for valid columns and 0.0 for padding columns, applied to the last tile in each row.
3. Reads Wt tiles per row in bulk: `cb_reserve_back(cb_in, Wt)`, then sequentially reads all Wt tiles with `noc_async_read_tile`, issues `noc_async_read_barrier`, and `cb_push_back(cb_in, Wt)`.

### Reader Kernel (WLarge: `reader_moreh_softmax_w_large.cpp`)

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w_large.cpp`

**Key Logic**:
- Same scaler and mask generation as WSmall.
- For each row, reads the tiles **three times** (three identical loops over Wt tiles), each time pushing one tile at a time to `cb_in0`. Between passes, `curr_tile` is reset to `curr_offset_i` to re-read the same row.
- This triple-read pattern corresponds to the three phases of the WLarge compute kernel: (1) max reduction, (2) exp+sum, (3) final normalization.

### Compute Kernel (WSmall: `moreh_softmax_w.cpp`)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | cb_in0, cb_mask, cb_bcast_scaler | cb_out0 | MAX reduce, SUB bcast, EXP, SUM reduce, RECIP, MUL bcast |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Key Logic -- Math Sequence (with `SOFTMAX` define active)**:

The compute kernel processes N tile-rows. For each row:

**Step 1: Find row-wise maximum** (`cb_in0` -> `cb_max`)
- Uses `compute_kernel_lib::reduce<MAX, REDUCE_ROW>` from `reduce_helpers_compute.hpp`.
- If `Wt == 1`: Applies mask to the single input tile, then reduces it.
- If `Wt > 1`: Reduces first `Wt-1` tiles with `WaitUpfrontNoPop` policy (tiles stay in CB for reuse), then masks the last tile and reduces it with accumulation (`Accumulate::at(cb_max, 1)` -- reloads previous max from cb_max before reducing the final tile).
- **Output**: `cb_max` contains 1 tile where column 0 of each row holds the maximum value for that row. Other columns are zeros.
- **Policy choice**: `WaitUpfrontNoPop` is critical -- it keeps input tiles in `cb_in0` so they can be reused in subsequent steps without re-reading from DRAM.

**Step 2: Compute x - max(x)** (`cb_in0`, `cb_max` -> `cb_x_m_max`)
- Reserves Wt tiles in `cb_x_m_max`.
- For each of Wt tiles: `sub_bcast_cols_init_short_with_dt` + `sub_tiles_bcast<BroadcastType::COL>`. This broadcasts the column-0 max values across all columns and subtracts from the input.
- **BroadcastType::COL semantics**: `result[h,w] = A[h,w] - B[h,0]`. The max tile's column-0 value for each row is subtracted from every element in that row.
- After the loop, pops `cb_max` and `cb_in0`.

**Step 3: Compute exp(x - max(x))** (`cb_x_m_max` -> `cb_exps`)
- For each of Wt tiles: copies tile from `cb_x_m_max` to DST, calls `exp_tile_init()` + `exp_tile(dst0)`.
- For the last tile (w == Wt-1): additionally applies the mask (`mask_tile`) to zero out padding positions in the exponential result.
- Note: With `#ifndef SOFTMAX`, there would be a `negative_tile` before `exp_tile` (for softmin), but with `SOFTMAX` defined, it is skipped.

**Step 4: Compute 1/sum(exp(x-max))** (`cb_exps` -> `cb_recipsumexps`)
- Uses `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` across all Wt tiles in `cb_exps`.
- Post-reduce lambda applies `recip_tile_init()` + `recip_tile(dst_idx)` to compute 1/sum.
- **Policy**: `WaitUpfrontNoPop` keeps `cb_exps` tiles available for the final step.

**Step 5: Final normalization** (`cb_exps`, `cb_recipsumexps` -> `cb_out0`)
- For each of Wt tiles: `mul_bcast_cols_init_short_with_dt` + `mul_tiles_bcast_cols`. This multiplies each exp tile by the column-broadcast reciprocal sum.
- **BroadcastType::COL semantics for multiply**: `result[h,w] = exp_tile[h,w] * (1/sum)[h,0]`.
- After the loop, pops `cb_recipsumexps`, `cb_x_m_max`, `cb_exps`.

### Compute Kernel (WLarge: `moreh_softmax_w_large.cpp`)

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp`

**Key Logic** -- Same algorithm but streaming. Key differences:

**Step 1: Find max** -- Same two-phase approach but uses default `WaitAndPopPerTile` policy (tiles are consumed immediately since they cannot all fit).

**Step 2: Compute exp(x-max) and accumulate sum** -- Combined into one pass:
- For each tile: `sub_tiles_bcast_cols_to_cb` (x - max -> cb_tmp), then `exp_tile_to_cb` (or with mask for last tile) -> `cb_exps`.
- Running sum: `copy_tile_to_cb(cb_exps, cb_add)` for w=0, then `add_tiles_to_cb(cb_add, cb_exps, cb_add)` for w>0. The `cb_add` accumulates the element-wise sum of all exp tiles.
- After the loop: `reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>` on `cb_add` (single tile now) with `recip_tile` post-op -> `cb_recipsumexps`.

**Step 3: Final normalization** -- Third pass through input:
- For each tile: `sub_tiles_bcast_cols_to_cb` (x - max -> cb_tmp), `exp_tile_to_cb` (cb_tmp -> cb_exps), `mul_tiles_bcast_cols_to_cb` (exp * 1/sum -> cb_out0).
- Each output tile is pushed to cb_out0 one at a time for the writer.
- After the loop: pops `cb_recipsumexps` and `cb_max`.

### Writer Kernel (WSmall: `writer_moreh_softmax_w.cpp`)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | cb_out0 | DRAM | Write output tiles |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`

**Key Logic**: For each row, waits for Wt tiles in `cb_out0`, writes them sequentially to DRAM via `noc_async_write_tile`, then `noc_async_write_barrier` and `cb_pop_front`.

### Writer Kernel (WLarge: `writer_moreh_softmax_w_large.cpp`)

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w_large.cpp`

**Key Logic**: Same as WSmall writer but writes one tile at a time (inner loop over Wt tiles with per-tile `cb_wait_front(1)` / `cb_pop_front(1)`).

## Implementation Notes

### Numerically Stable Softmax
The implementation uses the standard numerically stable softmax formula:
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```
Subtracting the max prevents overflow in the exponential.

### Conditional Compilation Defines
- `SOFTMAX`: When defined (which it always is for softmax), uses direct `exp()`. When undefined, uses `negative_tile()` + `exp()` (for softmin: `exp(-x)`).
- `LOG`: When defined, computes `log_softmax(x) = x - max - log(sum(exp(x-max)))` instead of regular softmax. Not used by the general softmax factory.
- `FP32_DEST_ACC_EN`: When defined, intermediates use Float32 data format and all `_with_dt` helpers reconfigure data formats for the packer/unpacker.

### WSmall vs WLarge Design Choice
The selection between WSmall and WLarge is based on L1 memory budget:
- `is_softmax_general_w_small_available()` computes total CB memory as: `Wt * tile_size * 2 (input+output) + 3 * tile_size (mask+scaler+output) + Wt * intermed_tile_size * 2 (exps + x_m_max) + 3 * intermed_tile_size (reduce + max + tmp)`.
- If `base_allocator_addr + cb_usage <= 512KB`, WSmall is chosen.
- WSmall is significantly more efficient because it avoids re-reading data from DRAM (the WLarge variant reads each row 3 times).

### Reduce Helper Library Usage
The compute kernels use `compute_kernel_lib::reduce<>` from `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, which provides a unified interface for:
- `PoolType::MAX` with `ReduceDim::REDUCE_ROW` -- finds row-wise maximum
- `PoolType::SUM` with `ReduceDim::REDUCE_ROW` -- computes row-wise sum
- Input policies: `WaitUpfrontNoPop` (keeps tiles for reuse), `BulkWaitBulkPop` (wait all, process, pop all), `WaitAndPopPerTile` (streaming)
- Post-reduce lambdas: `recip_tile` for 1/sum, `log_tile` for log-softmax
- Accumulation: `Accumulate::at(cb, iteration)` for multi-phase reduction (first Wt-1 tiles, then masked last tile)

### Broadcast Semantics Critical for Layer Norm
- `sub_tiles_bcast<BroadcastType::COL>`: `C[h,w] = A[h,w] - B[h,0]`. Broadcasts column-0 of B across all columns. Used to subtract per-row max/mean from all elements.
- `mul_tiles_bcast_cols`: `C[h,w] = A[h,w] * B[h,0]`. Broadcasts column-0 of B across all columns. Used to multiply by per-row 1/sum or 1/std.
- These are directly applicable to layer_norm_rm where you need to subtract per-row mean and multiply by per-row 1/std.

### Scaler Tile Format
The reduce scaler tile (`cb_bcast_scaler`) has value 1.0 packed into a specific sub-tile layout:
- For bfloat16: `ptr[k*256 + j] = uint16_t(float_bits >> 16)` for j in [0,16), k in [0,4) -- first 16 elements of each of 4 faces.
- For float32: `ptr[k*256 + j] = float_bits` for the same positions.
- All other positions are zero.
- This layout matches what the reduce hardware expects as its scaling factor.

### Mask Tile Format
The `generate_mask_w<T>` function creates a tile where:
- For the left half-tiles (subtiles 0 and 2): columns 0 through `min(mask_w, 16)-1` are 1.0, remaining columns are 0.0.
- For the right half-tiles (subtiles 1 and 3): columns 0 through `max(mask_w-16, 0)-1` are 1.0, remaining are 0.0.
- Applied via `mask_tile(data_dst, mask_dst)` in compute, which zeros out elements where the mask is 0.0.

### Relevance to Layer Norm RM
For implementing `layer_norm_rm`, the softmax general pattern provides a direct template for:

1. **Row-wise mean**: Replace `reduce<MAX, REDUCE_ROW>` with `reduce<SUM, REDUCE_ROW>` followed by multiplication with `1/W` (or use scaler = 1/W directly in the reduce).
2. **Row-wise variance**: After subtracting mean, square each element, then `reduce<SUM, REDUCE_ROW>` and multiply by `1/W`.
3. **Normalization**: `sub_tiles_bcast<COL>` for (x - mean), then `mul_tiles_bcast_cols` for multiplication by `1/sqrt(var + eps)`.
4. **Gamma/Beta application**: Additional CBs for gamma and beta weight tiles, with `mul_tiles` and `add_tiles` (or bcast variants if gamma/beta are per-row scalars).
5. **Epsilon handling**: Pack epsilon as a scalar tile (similar to scaler tile), add to variance before rsqrt.

The CB allocation pattern (intermediates in c_24-c_28, input in c_0, output in c_16) and the WSmall/WLarge bifurcation strategy are directly reusable.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the sub_tiles_bcast function work in compute kernels? Specifically, what does BroadcastType::COL mean?"
   **Reason**: Needed to confirm the broadcast semantics for the subtraction step (x - max) to understand how a single-column reduction result is applied across all columns.
   **Key Findings**: `BroadcastType::COL` broadcasts column-0 of the second operand across all columns. `C[h,w] = A[h,w] - B[h,0]`. The second input is expected to have non-zero values only in column 0.

2. **Query**: "How does the reduce operation work in TT-Metal? For REDUCE_ROW with MAX/SUM, what does the scaler do and what is the output format?"
   **Reason**: Needed to understand the output format of row reduction (where the reduced values live in the output tile) and the role of the scaler.
   **Key Findings**: REDUCE_ROW places reduced values in column 0 of the output tile. The scaler is applied as a multiplicative factor to the reduction result (1.0 for MAX/SUM). The scaler tile must be pre-loaded in its CB before calling reduce.

3. **Query**: "How does the mask_tile function work in compute kernels?"
   **Reason**: Needed to understand how padding elements are handled in the last tile of a row.
   **Key Findings**: `mask_tile` operates on DST registers. It zeros out elements where the mask value is 0.0 and preserves elements where the mask is 1.0. Used to prevent padding elements from affecting max/sum computations.

4. **Query**: "What is the output format of REDUCE_ROW? Does the result have reduced values in column 0?"
   **Reason**: Needed to confirm that column-broadcast operations (sub_bcast_cols, mul_bcast_cols) correctly pick up the reduction result from column 0.
   **Key Findings**: Confirmed: REDUCE_ROW places results in column 0 of each row. Other columns are zeroed out. This is exactly the format needed for subsequent `BroadcastType::COL` operations.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the unified reduce API used by the compute kernel.
   **Key Information**: Provides `reduce<PoolType, ReduceDim, InputPolicy>()` with support for accumulation, post-reduce ops (recip, log), and multiple input policies. Automatically manages DST register acquire/release, CB wait/pop, reduce init/uninit.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding helper functions like `mask_tile_to_cb`, `sub_tiles_bcast_cols_to_cb`, `exp_tile_to_cb`, `mul_tiles_bcast_cols_to_cb` used by the WLarge compute kernel.
   **Key Information**: These are convenience wrappers that handle `cb_reserve_back`, `cb_wait_front`, `tile_regs_acquire/commit/wait/release`, `pack_tile_with_dt`, and `cb_push_back/pop_front` in a single function call. They also handle FP32_DEST_ACC_EN reconfiguration.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding `generate_bcast_scaler`, `generate_mask_w`, and `TensorAccessor` usage in the reader kernel.
   **Key Information**: `generate_bcast_scaler<T>` fills a tile with the scaler in reduce-compatible format (first 16 elements of each face). `generate_mask_w<T>` creates a width mask with 1.0/0.0 based on how many valid columns exist. Both handle bfloat16 vs float32 via template parameter.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp`
   **Reason**: Understanding factory selection logic and L1 memory budget calculation.
   **Key Information**: WSmall is selected if `base_allocator_addr + cb_usage <= 512KB`. The L1 budget calculation accounts for all 9 CBs with their respective capacities and data formats.
