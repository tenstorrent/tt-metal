# Softmax General W-Small Implementation Analysis

## Overview

This document analyzes the **softmax general w_small** program factory, which implements the softmax function along the W (width / last) dimension of a tensor for cases where all W tiles of a single row fit into L1 simultaneously. The "w_small" designation means Wt (number of tiles along W) is small enough that the entire row can be loaded into circular buffers at once, enabling a multi-pass compute pattern without re-reading from DRAM.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_general_w_small.cpp`

**Kernel paths** (all under `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/`):
- Reader: `reader_moreh_softmax_w.cpp`
- Compute: `moreh_softmax_w.cpp`
- Writer: `writer_moreh_softmax_w.cpp`

**Mathematical operation** (with `SOFTMAX` define active):
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
```

This is the numerically-stable softmax, computed in four phases per row:
1. Find `max(x)` across the row (row-wise MAX reduction)
2. Compute `x - max(x)` for all tiles in the row
3. Compute `exp(x - max(x))`, then sum via row-wise SUM reduction to get `1/sum`
4. Multiply `exp(x - max(x)) * (1/sum)` to produce the final output

**Key relevance to layer normalization**: Layer norm follows an identical multi-pass pattern over the same row data: (1) compute mean via SUM reduction, (2) subtract mean, (3) square differences, (4) compute variance via SUM reduction, (5) compute inverse sqrt, (6) multiply. This analysis documents the exact mechanisms for row-wise reduction, scalar broadcasting, CB-based data reuse, and multi-pass compute.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles (1 x Wt tiles) |
| **Unit size** | Wt tiles (one full tile-row along W) |
| **Total units** | `num_kernel_rows = (physical_volume / H / W) * Ht` |
| **Loop structure** | Outer loop over N rows assigned to this core; inner loops over Wt tiles within each row |

Each "work unit" is one tile-row: all Wt tiles that share the same (batch, height-tile) index. The reader loads Wt tiles per iteration, compute processes all Wt tiles through four phases, and the writer drains Wt tiles per iteration.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary rank, flattened to num x H x W) | Same as input |
| **Dimension convention** | Last two dims are H, W | Same as input |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (accessed via TensorAccessor) | DRAM (accessed via TensorAccessor) |
| **Data type** | BFLOAT16 or FLOAT32 | Same as input |

### Layout Transformations
No tilize/untilize or reshard operations. Input and output are both in TILE_LAYOUT with INTERLEAVED memory layout. The operation is purely in-tile-space.

## Data Flow Pattern

### High-Level Pipeline (per row iteration)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 0 (setup) | Reader | N/A | cb_scaler (c_2), cb_mask (c_1) | generate_bcast_scaler, generate_mask_w (one-time) |
| 1 | Reader | DRAM | cb_in (c_0) | reserve_back(Wt), noc_async_read x Wt, push_back(Wt) |
| 2a | Compute | cb_in (c_0) | cb_max (c_26) | reduce MAX across row (via cb_tmp c_28 for mask) |
| 2b | Compute | cb_in (c_0), cb_max (c_26) | cb_x_m_max (c_27) | sub_tiles_bcast COL for each of Wt tiles |
| 2c | Compute | cb_x_m_max (c_27) | cb_exps (c_24) | exp_tile per tile, mask last tile |
| 2d | Compute | cb_exps (c_24) | cb_recipsumexps (c_25) | reduce SUM across row, then recip_tile |
| 2e | Compute | cb_exps (c_24), cb_recipsumexps (c_25) | cb_out (c_16) | mul_tiles_bcast_cols for each of Wt tiles |
| 3 | Writer | cb_out (c_16) | DRAM | wait_front(Wt), noc_async_write x Wt, pop_front(Wt) |

### Detailed Data Flow

**Reader (one-time setup)**:
1. Generates `cb_scaler` (c_2): A tile where specific positions contain the scaler value (1.0f for softmax). This tile has the scaler in the first 16 elements of each of the 4 faces (subtiles), with zeros elsewhere. This is the format required by `reduce_tile`.
2. Generates `cb_mask` (c_1): A width mask tile with 1.0 in valid positions and 0.0 in padding positions. Used to zero out padding in the last tile of each row when `logical_shape[-1] % 32 != 0`.

**Reader (per row)**:
3. Reserves Wt slots in cb_in, reads Wt tiles sequentially from DRAM using TensorAccessor, pushes all Wt tiles at once.

**Compute (per row, Phase 1 - Find max)**:
4. If `Wt == 1`: Applies mask to the single input tile (via `mask_tile_to_cb` into cb_tmp), then reduces that single masked tile with `reduce<MAX, REDUCE_ROW>` into cb_max.
5. If `Wt > 1`: First reduces the first `Wt-1` tiles with `reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>` (tiles stay in cb_in for reuse). Then masks the last tile into cb_tmp, and performs a second reduce with `Accumulate::at(cb_max, 1)` to combine with the partial max from the first Wt-1 tiles.

**Compute (per row, Phase 2 - Subtract max)**:
6. Waits for cb_in (Wt tiles) and cb_max (1 tile). For each of Wt tiles: `sub_tiles_bcast<COL>(cb_in[w], cb_max[0], dst0)` -- broadcasts the max column-vector across the row and subtracts. Result packed to cb_x_m_max. After all tiles: pops cb_max(1) and cb_in(Wt), pushes cb_x_m_max(Wt).

**Compute (per row, Phase 3 - Exponential)**:
7. For each tile in cb_x_m_max: copies to DST, applies `exp_tile`. For the last tile (w == Wt-1), also applies mask_tile to zero out padding. Results packed to cb_exps.

**Compute (per row, Phase 4 - Sum and reciprocal)**:
8. Uses `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` on cb_exps (all Wt tiles) into cb_recipsumexps. The `WaitUpfrontNoPop` policy keeps tiles in cb_exps for reuse in Phase 5. A post-reduce lambda applies `recip_tile` to produce 1/sum.

**Compute (per row, Phase 5 - Final multiply)**:
9. For each tile: `mul_tiles_bcast_cols(cb_exps[w], cb_recipsumexps[0], dst0)` -- multiplies each exp tile by the reciprocal sum (broadcast as column). Results packed to cb_out. After loop: pops cb_recipsumexps(1), cb_x_m_max(Wt), cb_exps(Wt), pushes cb_out(Wt).

**Writer (per row)**:
10. Waits for Wt tiles in cb_out, writes them sequentially to DRAM via TensorAccessor, pops Wt tiles.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-----------|----------|----------|----------|
| c_0 | cb_in | Input tile staging | Wt | Wt | Single | Reader | Compute | Row (persists across phases 1-2) |
| c_1 | cb_mask | Width mask tile | 1 | 1 | Single | Reader (once) | Compute | Program (generated once, never popped) |
| c_2 | cb_scaler | Broadcast scaler (1.0) | 1 | 1 | Single | Reader (once) | Compute | Program (generated once, never popped) |
| c_16 | cb_out | Output tile staging | Wt | Wt | Single | Compute | Writer | Row |
| c_24 | cb_exps | exp(x - max) intermediates | Wt | Wt | Single | Compute | Compute | Row (persists across phases 4-5) |
| c_25 | cb_recipsumexps | 1/sum(exp) scalar | 1 | 1 | Single | Compute | Compute | Row |
| c_26 | cb_max | Row-wise max scalar | 1 | 1 | Single | Compute | Compute | Row |
| c_27 | cb_x_m_max | (x - max) intermediates | Wt | Wt | Single | Compute | Compute | Row (persists across phases 3 and 5) |
| c_28 | cb_tmp | Masked last tile scratch | 1 | 1 | Single | Compute | Compute | Block (temporary within phase 1) |

**Data format notes**:
- c_0, c_1, c_2, c_16 use the input tensor's data format (BFLOAT16 or FLOAT32)
- c_24, c_25, c_26, c_27, c_28 use `intermed_data_format`: FLOAT32 if `fp32_dest_acc_en`, otherwise same as input

**Critical observation for layer norm**: The intermediate CBs (c_24-c_28) have capacity Wt, allowing the entire row of intermediate results to reside in L1 simultaneously. This is what makes the "w_small" variant work -- the row must fit in L1.

## Pipeline Pattern Summary

All CBs are single-buffered (capacity == block size). There is no reader/compute overlap within a single row iteration because:
- The reader produces the entire row (Wt tiles) into c_0 at once
- Compute consumes all Wt tiles through multiple phases before the writer can start
- The writer drains Wt tiles from c_16 after compute finishes

However, there IS potential overlap BETWEEN rows:
- While the writer drains row N from c_16, the reader could start loading row N+1 into c_0 (they use different CBs)
- The bottleneck is compute, which must fully process each row before output is available

## Index Calculations

### Tile Addressing
The program factory computes a linear `tile_offset` for each core, representing the starting tile index in the flattened tensor. Within each row of Wt tiles, tiles are addressed sequentially:

```
curr_tile = tile_offset + (row_index * Wt) + w   // where w in [0, Wt)
```

The TensorAccessor handles mapping from linear tile index to physical DRAM bank/address. This is configured via `TensorAccessorArgs` passed as compile-time arguments.

### Mask Width Calculation
```cpp
uint32_t mask_w = input_tensor.logical_shape()[-1] % TILE_WIDTH;  // 0..31
if (mask_w == 0) mask_w = TILE_WIDTH;  // Full tile -> mask all 32 cols
```
This handles the case where the logical tensor width is not a multiple of 32. The mask zeros out padding columns in the last tile of each row.

## Memory Access Patterns

### Read Pattern
- **Ordering**: Sequential tile reads within each row, rows processed sequentially
- **Granularity**: Full tiles (one tile per `noc_async_read_tile` call)
- **Burst**: All Wt tiles of a row are read in a burst (reserve Wt, read Wt, barrier, push Wt)
- **Source**: DRAM via TensorAccessor (handles bank interleaving automatically)

### Write Pattern
- **Ordering**: Sequential tile writes within each row, rows processed sequentially
- **Granularity**: Full tiles (one tile per `noc_async_write_tile` call)
- **Burst**: All Wt tiles of a row are written in a burst (wait Wt, write Wt, barrier, pop Wt)
- **Destination**: DRAM via TensorAccessor

### Compute Internal Access
- cb_in is accessed with indexed reads (tile index w within the Wt-tile window) -- this is how `WaitUpfrontNoPop` policy works
- cb_exps similarly accessed with indexed reads during the final multiply phase
- cb_max and cb_recipsumexps are always accessed at index 0 (single scalar tiles)

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `grid_coord.x` x `grid_coord.y` (device compute grid) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` rows |
| **Load balancing** | Two groups: group_1 gets ceil(rows/cores), group_2 gets floor(rows/cores) |

The work is split at the **row** level (not individual tiles). Each core processes a contiguous block of rows. The `split_work_to_cores_wt_core_range` function divides `num_kernel_rows` across the available cores:

- `core_group_1`: Gets `num_tiles_per_core_group_1` rows (the larger share)
- `core_group_2`: Gets `num_tiles_per_core_group_2` rows (the smaller share, or 0 if evenly divisible)

Core linearization: `core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset}` -- column-major ordering.

The `tile_offset` for each core accounts for all tiles in previous cores' rows: `tile_offset += num_tiles_per_core * Wt` per core.

## Arguments

### Compile-Time Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_fp32 | uint32_t (bool) | 1 if input dtype is FLOAT32, 0 otherwise. Controls generate_bcast_scaler/generate_mask_w template type. |
| 1+ | TensorAccessorArgs | multiple uint32_t | Input tensor accessor parameters (bank mapping, page size, etc.) |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | multiple uint32_t | Output tensor accessor parameters |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | N | uint32_t | Number of rows this core processes (`num_tiles_per_core_group_1` or `_2`) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (row width in tiles) |

**Compute defines**:
- `SOFTMAX=1`: Always defined. Selects `exp(x)` path (no negation before exp). Without this define, the kernel computes `exp(-x)` for softmin.
- `FP32_DEST_ACC_EN`: Defined when fp32 destination accumulation is enabled.

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor DRAM buffer address |
| 1 | N | uint32_t | Number of rows this core processes |
| 2 | tile_offset | uint32_t | Starting tile index (linear, across all rows assigned to this core) |
| 3 | Wt | uint32_t | Tiles per row (W dimension) |
| 4 | scaler | uint32_t | Bit-cast float scaler value (1.0f for softmax; reinterpreted as uint32_t) |
| 5 | mask_w | uint32_t | Number of valid columns in last tile (1-32) |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor DRAM buffer address |
| 1 | N | uint32_t | Number of rows this core processes |
| 2 | tile_offset | uint32_t | Starting tile index |
| 3 | Wt | uint32_t | Tiles per row |

## Kernel Implementations

### Reader: `reader_moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM (input tensor) | cb_in (c_0), cb_scaler (c_2), cb_mask (c_1) | Read tiles, generate scaler/mask |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp`

**Key Logic**:
1. **One-time setup**: Calls `generate_bcast_scaler<T>(cb_scaler, scaler)` and `generate_mask_w<T>(cb_mask, mask_w)`. Template parameter T is `uint32_t` for FP32, `uint16_t` for BFLOAT16. The scaler tile is filled with the value in the first 16 elements of each of 4 faces (4 x 16 = 64 positions filled, 960 zeros). The mask tile has 1.0 for valid columns and 0.0 for padding.
2. **Per-row loop**: For each of N rows: reserves Wt slots in cb_in, reads Wt tiles sequentially from DRAM via `noc_async_read_tile(curr_tile, src_in, l1_write_addr)`, issues read barrier, pushes Wt tiles.
3. Uses TensorAccessor for DRAM address generation (handles bank interleaving).

### Compute: `moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2,3,4) | N/A | cb_in (c_0), cb_mask (c_1), cb_scaler (c_2) | cb_out (c_16) | MAX reduce, subtract, exp, SUM reduce, recip, multiply |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_w.cpp`

**Key Logic**:

**Initialization**:
- `binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0)` -- sets up unpack/math/pack hardware
- Waits once for cb_mask and cb_bcast_scaler (they persist for the entire program)

**Phase 1 -- Find max(x) along row**:
- **Single tile case (Wt==1)**: Uses `mask_tile_to_cb` helper to copy the input tile and apply the mask (zeroing padding), writing to cb_tmp. Then `compute_kernel_lib::reduce<MAX, REDUCE_ROW>` on that single masked tile to produce cb_max.
- **Multi-tile case (Wt>1)**: First call uses `WaitUpfrontNoPop` policy to reduce the first Wt-1 tiles (keeping them in cb_in for later). Then masks the last tile (index Wt-1) into cb_tmp. Second call uses `Accumulate::at(cb_max, 1)` to combine the masked last tile's max with the running max from the first pass.

**Phase 2 -- Compute (x - max)**:
- Reserves Wt slots in cb_x_m_max. For each tile w in [0, Wt): acquires DST, calls `sub_bcast_cols_init_short_with_dt` then `sub_tiles_bcast<COL>(cb_in, cb_max, w, 0, dst0)`, packs to cb_x_m_max.
- After loop: pops cb_max(1), cb_in(Wt), pushes cb_x_m_max(Wt).
- **COL broadcast**: The max tile (1 tile) is broadcast as a column vector -- each row in the max tile contains the max for that tile-row, and it is subtracted from every column position.

**Phase 3 -- Compute exp(x - max)**:
- Reserves Wt slots in cb_exps. For each tile w: copies from cb_x_m_max[w] to DST, applies `exp_tile`. On the last tile (w == Wt-1), also copies mask tile and applies `mask_tile` to zero padding in the exponentiated output.
- After loop: pushes cb_exps(Wt).
- Note: cb_x_m_max is NOT popped here; it remains available for Phase 5 (LOG variant only).

**Phase 4 -- Sum and reciprocal** (SOFTMAX path, `#else` branch):
- `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` on cb_exps with post-reduce lambda `recip_tile`. This:
  - Waits for all Wt tiles in cb_exps upfront
  - Reduces them to a single tile (sum across row)
  - Applies `recip_tile` to produce 1/sum
  - Does NOT pop cb_exps (tiles persist for Phase 5)
  - Output to cb_recipsumexps (1 tile)

**Phase 5 -- Final output**:
- Reserves Wt slots in cb_out. For each tile w: `mul_bcast_cols_init_short_with_dt`, `mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0)`, packs to cb_out.
- After loop: pops cb_recipsumexps(1), cb_x_m_max(Wt), cb_exps(Wt), pushes cb_out(Wt).

### Writer: `writer_moreh_softmax_w.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | NCRISC (RISCV_1) | NOC1 | cb_out (c_16) | DRAM (output tensor) | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`

**Key Logic**:
1. Per-row loop: Waits for Wt tiles in cb_out, reads L1 addresses, writes Wt tiles sequentially to DRAM via `noc_async_write_tile`, issues write barrier, pops Wt tiles.
2. Uses TensorAccessor for DRAM address generation.

## Implementation Notes

### Multi-Pass Data Reuse (Critical for Layer Norm Reference)
The key architectural insight is how data is reused across multiple compute phases WITHOUT re-reading from DRAM:
1. **cb_in tiles persist** through Phase 1 (max reduction) and Phase 2 (subtract max) via `WaitUpfrontNoPop` in the reduce call and delayed `cb_pop_front` until after Phase 2.
2. **cb_exps tiles persist** through Phase 4 (sum reduction) and Phase 5 (multiply by 1/sum) via `WaitUpfrontNoPop` in the sum reduce and delayed pop until after Phase 5.
3. **cb_x_m_max tiles persist** through Phase 3 (exp) and into Phase 5 (for the LOG variant) -- they are popped at the end of Phase 5.

This pattern is DIRECTLY applicable to layer norm:
- Read input row once into cb_in
- Phase 1: Reduce SUM for mean computation (with scaler = 1/W), keep tiles via WaitUpfrontNoPop
- Phase 2: Subtract mean from each tile (sub_tiles_bcast COL)
- Phase 3: Square the differences
- Phase 4: Reduce SUM for variance, apply rsqrt via post-reduce lambda
- Phase 5: Multiply (x - mean) * inv_sqrt_var

### Scaler Construction
The `generate_bcast_scaler` function creates a tile with the scalar value placed in specific positions (first 16 elements of each face). This format is required by the `reduce_tile` hardware instruction. For softmax, scaler = 1.0f (just sum, no averaging). For layer norm, this would be set to 1/W to compute the mean directly during reduction.

### Mask Handling for Partial Tiles
When `logical_shape[-1] % 32 != 0`, the last tile in each row contains padding. The mask tile (generated once) has 1.0 for valid columns and 0.0 for padding. It is applied at two points:
1. Before MAX reduction (Phase 1): The last tile is masked before reduction so padding does not affect the max.
2. After EXP computation (Phase 3): The last exp tile is masked so padding elements are zero (do not contribute to sum).

### FP32 Accumulation Support
When `fp32_dest_acc_en` is true:
- Intermediate CBs use Float32 data format (wider tiles, more L1 usage)
- `_with_dt` helper functions call `reconfig_data_format` or `pack_reconfig_data_format` before operations to handle format changes between input/intermediate CBs
- This is critical for numerical precision in large reductions

### Conditional Compilation Variants
The compute kernel uses preprocessor defines to support multiple operation modes:
- `SOFTMAX` (defined here): Uses `exp(x)` directly, computes `1/sum` via recip, multiplies
- `LOG` (not defined here): Would compute `log(sum)` instead of `1/sum`, subtract from `(x - max)` for log-softmax
- Neither (softmin): Negates before exp via `negative_tile`

### compute_kernel_lib::reduce Helper
The `reduce_helpers_compute.hpp` library abstracts away the complexity of:
- DST register management (acquire/commit/wait/release)
- reduce_init/reduce_uninit lifecycle
- CB synchronization (wait_front, pop_front, reserve_back, push_back)
- Accumulation across multiple reduce calls (via `Accumulate` type)
- Post-reduce operations via lambda (e.g., `recip_tile`, `log_tile`)

The key policies used in this kernel:
- `WaitUpfrontNoPop`: Waits for all tiles, processes with indexed access, does NOT pop -- tiles remain for subsequent operations
- `BulkWaitBulkPop` (not used here but referenced in LOG variant): Waits and pops in bulk

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the reduce_tile operation work in TT-Metal compute kernels? Specifically, how does REDUCE_ROW work?"
   **Reason**: Needed to understand the fundamental reduction mechanism used throughout this kernel.
   **Key Findings**: REDUCE_ROW reduces along the W dimension producing a column vector. The scaler parameter is applied during reduction (1.0 for SUM/MAX, 1/N for AVG). The operation involves unpacking via `llk_unpack_AB_reduce` and math via `llk_math_reduce`.

2. **Query**: "How does sub_tiles_bcast with BroadcastType::COL work in TT-Metal compute kernels?"
   **Reason**: This is the core mechanism for subtracting the row-wise max from all tiles in the row.
   **Key Findings**: COL broadcast treats the second operand as a column vector, broadcasting its single column value across all columns of the first operand. Used in softmax to subtract max and in layernorm to subtract mean.

3. **Query**: "How does mask_tile work in TT-Metal compute kernels?"
   **Reason**: Needed to understand how partial tile padding is handled.
   **Key Findings**: mask_tile zeros out elements based on a mask tile pattern. The mask tile must be in the DST register adjacent to the data tile (dst0, dst1). Used to handle partial tiles where logical dimensions are not multiples of 32.

4. **Query**: "What does generate_bcast_scaler do in moreh_common.hpp?"
   **Reason**: Needed to understand how the scaler tile is constructed for reduce operations.
   **Key Findings**: Fills 1024-element tile with zeros, then sets first 16 elements of each of 4 faces to the scaler value. For bfloat16, uses upper 16 bits of the float-to-uint32 bit-cast. This specific layout is required by the reduce hardware.

5. **Query**: "What does binary_op_init_common do in TT-Metal compute kernels?"
   **Reason**: This is the first call in the compute kernel; needed to understand what it initializes.
   **Key Findings**: Sets up unpack, math, and pack hardware pipelines for binary operations. Configures `llk_unpack_hw_configure`, `llk_math_pack_sync_init`, `llk_pack_hw_configure`, etc. Does NOT replace operation-specific init calls (e.g., `add_tiles_init`).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the reduce helper library API and its policies
   **Key Information**: Documents ReduceInputPolicy enum (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), ReduceInputBlockShape, Accumulate type, and the unified reduce() function signature. WaitUpfrontNoPop is explicitly documented as "ideal for softmax pattern" where tiles are reused.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding helper functions used in the compute kernel (mask_tile_to_cb, pack_tile_with_dt, copy_tile_init_with_dt, etc.)
   **Key Information**: The `_with_dt` suffix functions handle FP32_DEST_ACC_EN data format reconfiguration. `mask_tile_to_cb` is a composite helper that copies a tile, copies a mask, applies mask_tile, and packs the result to a CB.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding reader helper functions (generate_bcast_scaler, generate_mask_w)
   **Key Information**: `generate_bcast_scaler` creates the tile format required by reduce hardware. `generate_mask_w` creates a mask with 1s for valid columns and 0s for padding, structured across the 4 subtiles of a tile.

4. **Source**: `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp` and `.cpp`
   **Reason**: Understanding work distribution and CB creation helpers
   **Key Information**: `split_work_to_cores_wt_core_range` divides work into two groups (ceil/floor), `CircularBufferArg` takes (buffer_index, num_tiles, optional data_format), `CreateComputeKernel` accepts per-core-group compile args.
