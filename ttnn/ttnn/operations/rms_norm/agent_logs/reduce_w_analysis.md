# Reduce W Implementation Analysis

## Overview

The reduce_w operation reduces a 4D tiled tensor along the W (width) dimension, producing one output tile per tile-row. For input shape [N, C, Ht, Wt] (in tiles), the output shape is [N, C, Ht, 1]. This is the canonical W-dimension reduction used for sum, average, and max operations.

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Critical dual-path architecture**: The reduce_w compute kernel has TWO entirely different code paths selected by the `REDUCE_ROW_SUM_VIA_MM` define:
- **Path A (MM path)**: For SUM and AVG operations on W dimension -- uses `matmul_tiles()` with a column-vector scaler
- **Path B (Reduce helper path)**: For MAX operations (and the general case) -- uses `reduce_init/reduce_tile/reduce_uninit` LLK primitives via the `compute_kernel_lib::reduce` helper

The `get_defines()` function in `reduce_op.cpp` sets `REDUCE_ROW_SUM_VIA_MM=1` when `reduce_dim == W && (reduce_op == SUM || reduce_op == AVG)`.

There is also a **negate variant** (`reduce_w_neg.cpp`) used for `reduce_min` which implements `-reduce_max(-x)`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row (one row of Wt tiles) |
| **Unit size** | Wt input tiles produce 1 output tile |
| **Total units** | N * C * Ht tile-rows |
| **Loop structure** | outer: NC batches, inner: Ht rows, innermost: Wt tiles per row |

Each core is assigned `num_rows_per_core` contiguous tile-rows. The core processes each tile-row by accumulating Wt input tiles into a single output tile through reduction.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, 1] (padded to tile) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | Any (bf16, fp32, etc.) | Matches output_dtype param |

### Layout Transformations

No explicit tilize/untilize within this operation. The host-side `reduce()` wrapper in `reduce_op.cpp` performs `tilize_with_val_padding` before invoking the device operation, so the program factory always receives tiled input.

## Data Flow Pattern

### Path A: REDUCE_ROW_SUM_VIA_MM (SUM/AVG on W)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (input tensor) | CB c_0 | reserve_back(1), push_back(1) per tile |
| 1a | Reader | L1 (generated) | CB c_2 | generate_mm_scaler fills scaler tile once |
| 2 | Compute | CB c_0, CB c_2 | CB c_3 | wait_front/pop_front on c_0, matmul_tiles accumulates into DST, pack_tile to c_3 |
| 3 | Writer | CB c_3 | DRAM (output tensor) | wait_front(1), pop_front(1) per output tile |

### Path B: Reduce Helper (MAX, or general case)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (input tensor) | CB c_0 | reserve_back(1), push_back(1) per tile |
| 1a | Reader | L1 (generated) | CB c_2 | prepare_reduce_scaler fills scaler tile once |
| 2 | Compute | CB c_0, CB c_2 | CB c_3 | reduce helper manages all CB ops internally |
| 3 | Writer | CB c_3 | DRAM (output tensor) | wait_front(1), pop_front(1) per output tile |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Reduction scaler constant | 2 tiles | 1 tile | Double (over-provisioned; only 1 tile used) | Reader | Compute | Program (generated once, persists) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per output tile) |
| c_4 | cb_acc | Accumulator intermediate (negate variant only) | 1 tile | 1 tile | Single | Compute | Compute | Row (across Wt iterations) |
| c_5 | cb_ineg | Negated input intermediate (negate variant only) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile) |

### CB Layout Notes for RMS Norm Reference

1. **CB c_2 (scaler) is a constant CB**: The scaler tile is generated once by the reader kernel at program start and persists for the entire program. The compute kernel does `cb_wait_front(scaler_cb, 1)` once before the main loop and never pops it. This is the canonical pattern for constant tiles.

2. **CB c_0 (input) uses double-buffering**: Capacity = 2 tiles allows the reader to fill the next tile while compute processes the current one.

3. **CB c_3 (output) uses double-buffering**: Capacity = 2 tiles allows compute to pack the next output while the writer drains the current one.

4. **CB c_4 (accumulator) is compute-only, single-buffered**: In the negate variant, this CB is used as a scratchpad to persist partial reduction results across the Wt inner loop. It is both produced and consumed by the compute kernel itself (pack_tile to c_4, then wait_front/copy_tile from c_4 on the next iteration). This is a multi-pass data reuse pattern.

5. **CB c_5 (negated intermediate) is compute-only, single-buffered**: Used to hold the negated copy of each input tile before feeding it into the reduce operation. This is a typical intermediate CB pattern where one compute phase writes and the next reads.

### Multi-Pass Data Reuse Patterns

**Scaler CB (c_2) persistence**: The scaler tile is pushed once and never popped. Both the MM path and the reduce helper path rely on this -- the scaler tile stays at the front of CB c_2 for the entire program execution. The `cb_wait_front(scaler_cb, 1)` call at the beginning of the compute kernel ensures the tile is ready, and subsequent reduce_tile calls reference it by index 0.

**Accumulator CB (c_4) in negate variant**: This CB implements a "spill-and-reload" pattern within the compute kernel:
- Iteration 0 (wt=0): Reduce first tile, pack result to c_4
- Iterations 1..Wt-1: Wait on c_4, copy accumulated value back to DST, pop c_4, reduce next tile into DST, pack updated result back to c_4
- After Wt loop: Wait on c_4, copy final accumulated value, negate, pack to output c_3

This is necessary because DST registers cannot hold values across the tile_regs_acquire/release cycle in half-sync mode. The accumulator CB provides persistent storage in L1.

## Pipeline Pattern Summary

- **CB c_0**: Double-buffered (2 tiles capacity, 1 tile block) -- enables reader/compute overlap
- **CB c_2**: Allocated as double-buffered but functionally acts as persistent single (push once, never pop)
- **CB c_3**: Double-buffered (2 tiles capacity, 1 tile block) -- enables compute/writer overlap
- **CB c_4**: Single-buffered (1 tile) -- compute-to-compute intermediate, no overlap needed
- **CB c_5**: Single-buffered (1 tile) -- compute-to-compute intermediate, no overlap needed

## Compute Kernel Structure (Primary Focus)

### Path A: REDUCE_ROW_SUM_VIA_MM (reduce_w.cpp, when SUM/AVG)

This path implements W-reduction as a matmul operation. Conceptually, reducing a row of tiles along W is equivalent to multiplying each tile by a column vector of ones (or a scaler value for AVG).

```
File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp
Guarded by: #ifdef REDUCE_ROW_SUM_VIA_MM
```

**Initialization**:
```cpp
mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
```
- `mm_init(icb0, icb1, ocb)` -- initializes the matmul hardware pipeline
- Parameters: input CB, scaler/weight CB, output CB

**Compile-time arguments**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (adjusted per core group) |
| 1 | Wt | uint32_t | Number of tiles in W dimension |
| 2 | NC | uint32_t | Always 1 (batch dimension is flattened into Ht across cores) |

**Main loop structure**:
```
cb2.wait_front(1)              // Wait for scaler tile (once, before loop)
for nc in 0..NC:
    for ht in 0..Ht:
        acquire_dst()           // Lock DST registers (old API, equivalent to tile_regs_acquire)
        for wt in 0..Wt:
            cb0.wait_front(1)   // Wait for input tile
            matmul_tiles(c_0, c_2, 0, 0, 0)  // Multiply-accumulate into DST[0]
            cb0.pop_front(1)    // Release input tile
        cb3.reserve_back(1)     // Reserve output space
        pack_tile(0, c_3)       // Pack DST[0] to output CB
        cb3.push_back(1)        // Signal output ready
        release_dst()           // Unlock DST registers
```

**Key function signatures**:
- `mm_init(uint32_t icb0, uint32_t icb1, uint32_t ocb)` -- initializes matmul hardware
- `matmul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` -- performs matmul on tile from icb0[itile0] with tile from icb1[itile1], accumulating into DST[idst]
- `acquire_dst()` / `release_dst()` -- deprecated DST management (combines math acquire + pack wait)
- `pack_tile(uint32_t idst, uint32_t ocb)` -- packs DST[idst] to output CB

**Why matmul works for row reduction**: A 32x32 tile multiplied by a 32x32 scaler tile (with only row 0 populated) effectively computes the sum of each row in the input tile. When multiple tiles in a row are processed, `matmul_tiles` accumulates into the same DST[0], achieving the sum across the W dimension. The scaler tile has the scaler value in row 0 of each face, which when multiplied performs the equivalent of `sum * scaler`.

### Path B: Reduce Helper (reduce_w.cpp, when NOT REDUCE_ROW_SUM_VIA_MM)

This path uses the `compute_kernel_lib::reduce` helper function which encapsulates all DST management, CB operations, and reduce LLK calls.

```
File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp
Guarded by: #ifndef REDUCE_ROW_SUM_VIA_MM
```

**Initialization**:
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
```
- `compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb)` -- performs all hardware configuration (unpack, math, pack) for the three CBs. Must be called exactly once at kernel start before any other compute API.

**Reduce helper call**:
```cpp
compute_kernel_lib::reduce<
    REDUCE_OP,                                              // Template: PoolType (SUM, AVG, MAX)
    REDUCE_DIM,                                             // Template: ReduceDim (REDUCE_ROW for W reduction)
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,  // Template: streaming mode
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(   // Template: no reconfig (first op)
    tt::CBIndex::c_0,                                       // input_cb
    tt::CBIndex::c_2,                                       // scaler_cb
    tt::CBIndex::c_3,                                       // output_cb
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));  // block shape
```

**What the reduce helper does internally** (from `reduce_helpers_compute.inl`, REDUCE_ROW path with WaitAndPopPerTile):

```
reduce_init<reduce_type, reduce_dim>(input_cb, scaler_cb, output_cb)
cb_wait_front(scaler_cb, 1)           // Wait for scaler tile (once)

for nc in 0..NC:
    for ht in 0..Ht:
        tile_regs_acquire()            // Lock DST for MATH thread
        for wt in 0..Wt:
            cb_wait_front(input_cb, 1) // Wait for input tile
            reduce_tile<reduce_type, reduce_dim>(input_cb, scaler_cb, 0, 0, 0)
            cb_pop_front(input_cb, 1)  // Release input tile
        cb_reserve_back(output_cb, 1)  // Reserve output space
        tile_regs_commit()             // Release DST from MATH, signal PACK
        tile_regs_wait()               // PACK acquires DST
        pack_tile(0, output_cb)        // Pack DST[0] to output CB
        tile_regs_release()            // Release DST from PACK
        cb_push_back(output_cb, 1)     // Signal output ready

reduce_uninit()                        // Reset packer edge mask
```

**Key LLK function signatures**:

- `reduce_init<PoolType, ReduceDim, bool enforce_fp32_acc>(uint32_t icb, uint32_t icb_scaler, uint32_t ocb)`:
  - Configures unpacker for reduce (sets up SRCA/SRCB unpack)
  - Configures math unit for reduce operation
  - Sets packer edge masks (reduce_mask_config)
  - For fp32 accumulation, writes to hardware debug register to enable feature

- `reduce_tile<PoolType, ReduceDim, bool enforce_fp32_acc>(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst)`:
  - MATH thread: `llk_math_reduce` performs the actual reduction math on DST[idst]
  - UNPACK thread: `llk_unpack_AB_reduce` unpacks tile from icb[itile] and scaler from icb_scaler[itile_scaler]
  - The reduce operation accumulates into DST[idst] -- multiple calls with the same idst sum/max/avg the results

- `reduce_uninit<bool enforce_fp32_acc>(uint32_t icb)`:
  - Resets math configuration for reduce
  - Clears packer reduce mask (`llk_pack_reduce_mask_clear`)
  - **Must be called before any non-reduce operation** to avoid corrupt packing

### DST Register Management (New vs Old API)

The code uses two different DST management patterns:

**Old API (used in MM path)**:
```cpp
acquire_dst()    // MATH: wait_for_dest_available + PACK: wait_for_math_done (combined)
// ... compute ...
pack_tile(...)   // Pack from DST
release_dst()    // MATH: dest_section_done + PACK: dest_section_done (combined)
```

**New API (used in reduce helper and negate variant)**:
```cpp
tile_regs_acquire()  // MATH thread only: wait for DST availability
// ... compute (MATH fills DST) ...
tile_regs_commit()   // MATH releases DST, signals it's ready for PACK
tile_regs_wait()     // PACK thread: wait for MATH to commit
pack_tile(...)       // PACK reads from DST
tile_regs_release()  // PACK releases DST for next MATH cycle
```

The new API provides finer-grained control: MATH and PACK threads are synchronized separately, enabling better pipelining.

### Negate Variant (reduce_w_neg.cpp)

This is used for `reduce_min = -reduce_max(-x)`. It does NOT use the reduce helper and instead manually manages all CB and DST operations.

```
File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp
Additional CBs used: c_4 (accumulator), c_5 (negated intermediate)
```

**Phase structure per tile-row**:

```
compute_kernel_hw_startup(c_0, c_2, c_3)
cb_scaler.wait_front(1)                    // Scaler persists

for nc in 0..NC:
    for ht in 0..Ht:
        for wt in 0..Wt:
            // --- Phase 1: Negate input tile ---
            cb_input.wait_front(1)
            tile_regs_acquire()
            copy_tile_init(cb_input)         // Init copy from input CB
            copy_tile(cb_input, 0, dst_idx)  // Copy input[0] -> DST[0]
            negative_tile_init()             // Init negate SFPU op
            negative_tile(dst_idx)           // Negate DST[0] in-place
            tile_regs_wait()
            cb_input.pop_front(1)
            cb_ineg.reserve_back(1)
            tile_regs_commit()
            pack_tile(dst_idx, cb_ineg)      // Pack negated tile to c_5
            tile_regs_release()
            cb_ineg.push_back(1)

            // --- Phase 2: Accumulate reduction ---
            tile_regs_acquire()
            if (wt > 0):
                cb_acc.wait_front(1)
                copy_tile_init(cb_acc)       // Reload accumulated partial
                copy_tile(cb_acc, 0, dst_idx)
            cb_ineg.wait_front(1)
            reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc)
            reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx)
            reduce_uninit()
            tile_regs_wait()
            cb_ineg.pop_front(1)
            if (wt > 0):
                cb_acc.pop_front(1)
            cb_acc.reserve_back(1)
            tile_regs_commit()
            pack_tile(dst_idx, cb_acc)       // Spill accumulated value to c_4
            tile_regs_release()
            cb_acc.push_back(1)

        // --- Phase 3: Final negate of accumulated result ---
        cb_acc.wait_front(1)
        tile_regs_acquire()
        copy_tile_init(cb_acc)
        copy_tile(cb_acc, 0, dst_idx)
        negative_tile_init()
        negative_tile(dst_idx)               // Negate the max to get min
        tile_regs_wait()
        cb_acc.pop_front(1)
        cb_output.reserve_back(1)
        tile_regs_commit()
        pack_tile(dst_idx, cb_output)
        tile_regs_release()
        cb_output.push_back(1)
```

**Key additional function signatures used in negate variant**:
- `copy_tile_init(uint32_t icb)` -- initializes tile copy from specified CB (configures unpacker)
- `copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)` -- copies tile icb[itile] to DST[idst]
- `negative_tile_init()` -- initializes the SFPU negation operation
- `negative_tile(uint32_t idst)` -- negates DST[idst] in-place using SFPU

## Scalar/Constant CB Setup

### How the Scaler Tile is Created

The scaler tile is created by the **reader kernel** before the main read loop. There are two paths:

**Path A (REDUCE_ROW_SUM_VIA_MM)**: Uses `generate_mm_scaler(cb_id_in2, packed_bf16)`:
- Scaler is passed as compile-time arg (float bits via `std::bit_cast<uint32_t>(operation_attributes.scaler)`)
- Reader converts float32 bits to packed bf16 (upper 16 bits duplicated)
- `generate_mm_scaler` zeroes the entire tile via NOC reads from MEM_ZEROS_BASE, then fills specific positions
- The scaler is placed at every 8th position in faces 0 and 2 (column positions for the matrix multiply)
- This creates a column-vector-like tile where `matmul_tiles` effectively multiplies each row by the scaler

**Path B (Reduce helper)**: Uses `dataflow_kernel_lib::prepare_reduce_scaler<cb_id>(scaler_f)`:
- Also receives scaler as compile-time arg, but uses the float directly via `__builtin_bit_cast`
- `prepare_reduce_scaler`:
  1. Detects data format and tile shape from the CB
  2. Converts float to packed format (bf16: two packed values per u32; fp32: one value per u32)
  3. `cb_reserve_back(cb_id, 1)` -- reserves space in the scaler CB
  4. `zero_faces` -- zeroes all faces using NOC reads from MEM_ZEROS_BASE
  5. `fill_row0` -- fills row 0 of each face with the scaler value
  6. `cb_push_back(cb_id, 1)` -- signals the tile is ready

**Scaler value semantics**:
- For SUM: scaler = 1.0 (or user-provided)
- For AVG: scaler = 1/N where N is the number of elements being reduced
- For MAX: scaler = 1.0
- The scaler is the `operation_attributes.scaler` field, passed from host

### For RMS Norm: Scaler Pattern

RMS norm needs reductions with specific scalers:
- For `mean(x^2)`: scaler = 1/W (where W is the width in elements, not tiles)
- The `prepare_reduce_scaler` function is the recommended approach for custom operations since it handles both bf16 and fp32 formats automatically

## Reduce Helper Parameters Reference

### Template Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `reduce_type` | `PoolType` | `SUM`, `AVG`, `MAX` | Mathematical reduction operation |
| `reduce_dim` | `ReduceDim` | `REDUCE_ROW`, `REDUCE_COL`, `REDUCE_SCALAR` | Which dimension(s) to reduce |
| `input_policy` | `ReduceInputPolicy` | `WaitAndPopPerTile`, `BulkWaitBulkPop`, `WaitUpfrontNoPop`, `NoWaitNoPop` | CB synchronization strategy |
| `reconfig_mode` | `ReduceDataFormatReconfigMode` | `NONE`, `INPUT`, `OUTPUT`, `INPUT_AND_OUTPUT` | Data format reconfiguration |

### Function Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_cb` | `uint32_t` | Input CB with tiles to reduce |
| `scaler_cb` | `uint32_t` | CB containing the scaler tile (must be pre-filled) |
| `output_cb` | `uint32_t` | Output CB for reduced tiles |
| `input_block_shape` | `ReduceInputBlockShape` | Dimensions: rows (Ht) x cols (Wt) x batches (NC) |
| `input_memory_layout` | `ReduceInputMemoryLayout` | Row stride for non-contiguous layouts (default: contiguous) |
| `accumulate` | `AccumulateT` | Accumulation config for block-wise reduction (default: NoAccumulation) |
| `post_reduce_op` | `PostReduceOp` | Lambda called after each row's reduction (e.g., for rsqrt) |

### ReduceInputPolicy Details (for RMS Norm)

- **WaitAndPopPerTile** (used in reduce_w): Safest, minimal CB requirement (needs only 1 tile in input CB). Waits for each tile, processes it, pops it. Good for streaming.
- **WaitUpfrontNoPop**: Waits for ALL input tiles at start, processes with indexed access, never pops. Tiles remain in CB for reuse by subsequent operations. **Ideal for softmax/RMS norm patterns where the same input is needed multiple times.**
- **NoWaitNoPop**: Caller must manage all wait/pop. For pre-loaded or custom CB management scenarios.

### Post-Reduce Operation Pattern (Critical for RMS Norm)

The `post_reduce_op` lambda is called with `dst_idx` after all tiles in a row have been reduced, but **before** the result is packed to the output CB. The result is still in DST registers. This is the ideal place to apply operations like `rsqrt_tile`:

```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    cb_in, cb_scaler, cb_out,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
    {},
    NoAccumulation{},
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```

## Binary Op Broadcast Patterns

The reduce_w operation itself does not use binary operations with broadcast. However, for the RMS norm use case, after obtaining the normalization factor (1/sqrt(mean(x^2) + eps)), you need to multiply it element-wise with the input tensor. This requires a **broadcast pattern** where the 1-wide reduction result is broadcast across the W dimension.

The relevant pattern from the reduce_w analysis:
- The output of reduce_w is [N, C, Ht, 1] -- one tile per tile-row
- To multiply this with the [N, C, Ht, Wt] input, you need column broadcast (REDUCE_ROW result broadcast across columns)
- This is typically done with `mul_tiles_bcast` or `mul_bcast_cols_init_short` + `mul_tiles_bcast<BroadcastType::COL>` in a separate compute phase

## Index Calculations

Tile indexing is linear and sequential within each core's assigned work:

```
For core i with start_tile_id and num_rows_per_core:
    input tiles:  start_tile_id to start_tile_id + (num_rows_per_core * Wt) - 1
    output tiles: start_tile_id / Wt to start_tile_id / Wt + num_rows_per_core - 1
```

The reader processes tiles in sequential order (tile 0, 1, 2, ...). Within the compute kernel, tiles arrive in row-major order: all Wt tiles for row 0, then all Wt tiles for row 1, etc.

## Memory Access Patterns

### Read Pattern
- **Sequential tile reads**: Reader reads input tiles one at a time in linear order using TensorAccessor
- **Single tile per NoC transaction**: `noc.async_read` with `tile_bytes` per call
- **Read barrier per tile**: `noc.async_read_barrier()` after each tile (ensures data is in L1 before push_back)

### Write Pattern
- **Sequential tile writes**: Writer writes output tiles one at a time in linear order
- **Single tile per NoC transaction**: `noc_async_write_page` with page_bytes per call
- **Write flush per tile, barrier at end**: `noc_async_writes_flushed()` per tile, `noc_async_write_barrier()` at loop end

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to compute_with_storage_grid_size.x * y |
| **Total cores** | min(num_rows, available_cores) |
| **Work per core** | num_rows_per_core tile-rows (each = Wt input tiles -> 1 output tile) |
| **Load balancing** | Two core groups: group_1 gets ceil(num_rows/num_cores), group_2 gets floor |

Work splitting uses `tt::tt_metal::split_work_to_cores`:
- `core_group_1`: gets `num_rows_per_core_group_1` rows (larger share)
- `core_group_2`: gets `num_rows_per_core_group_2` rows (smaller share, may be empty)
- The two groups get different compute kernels with different compile-time Ht values

## Arguments

### Compile-Time Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | scaler_bits | uint32_t | Float scaler value bit-cast to uint32_t |
| 1+ | TensorAccessorArgs | varies | Source buffer tensor accessor parameters |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_3) |
| 1+ | TensorAccessorArgs | varies | Destination buffer tensor accessor parameters |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows this core processes |
| 1 | Wt | uint32_t | Number of tiles in W dimension |
| 2 | NC | uint32_t | Always 1 (batch folded into core assignment) |

**Compute defines** (via `reduce_defines`):

| Define | Value | Description |
|--------|-------|-------------|
| `REDUCE_OP` | `PoolType::SUM` / `PoolType::AVG` / `PoolType::MAX` | Reduction math operation |
| `REDUCE_DIM` | `ReduceDim::REDUCE_ROW` | Always REDUCE_ROW for W reduction |
| `REDUCE_ROW_SUM_VIA_MM` | `1` (only for SUM/AVG) | Selects matmul path vs reduce helper path |

### Runtime Arguments

**Reader kernel** (per core):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles for this core (= num_rows * Wt) |
| 2 | start_id | uint32_t | Global tile index to start reading from |

**Writer kernel** (per core):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_output_tiles | uint32_t | Output tiles for this core (= num_rows * 1) |
| 2 | start_id | uint32_t | Global output tile index to start writing from |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input tensor) | CB c_0, CB c_2 | Read input tiles sequentially; generate scaler tile once |
| compute | RISCV_2 (TRISC0/1/2) | N/A | CB c_0, CB c_2 | CB c_3 | Reduce W dimension via matmul or reduce_tile |
| writer | RISCV_1 | NOC1 | CB c_3 | DRAM (output tensor) | Write reduced tiles sequentially |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- **Key Logic**: Generates scaler tile before main loop. Uses TensorAccessor for input reads. The scaler generation path depends on REDUCE_ROW_SUM_VIA_MM define (generate_mm_scaler vs prepare_reduce_scaler).

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- **Key Logic**: Two code paths (MM vs reduce helper) selected by REDUCE_ROW_SUM_VIA_MM. The reduce helper path is a single function call that handles all iteration and DST management internally. The MM path uses explicit acquire_dst/release_dst with matmul_tiles accumulation.

### Compute Kernel (Negate Variant)
- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`
- **Key Logic**: Three-phase operation per tile (negate -> accumulate -> final negate). Uses two additional intermediate CBs (c_4, c_5). Does NOT use the reduce helper -- all DST and CB management is manual.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic writer shared across many operations. Waits on output CB, writes to DRAM via TensorAccessor, pops tiles.

## Implementation Notes

### REDUCE_ROW_SUM_VIA_MM Optimization
For SUM and AVG reductions along W, using `matmul_tiles` is more efficient than `reduce_tile` because:
1. The FPU (matrix unit) is better utilized for multiply-accumulate operations
2. The matmul naturally accumulates across multiple tiles in DST
3. The scaler tile format is different: mm_scaler uses a column-vector pattern, reduce scaler uses row-0 pattern

### NC=1 and Batch Flattening
The compute kernel always receives NC=1. The actual N*C batch dimension is flattened into the tile-row count distributed across cores. Each core processes `num_rows_per_core` contiguous tile-rows which may span batch boundaries. This simplification avoids needing batch-aware logic in the compute kernel.

### Scaler Data Format
The scaler CB always uses `Float16_b` (bfloat16) format regardless of the input data type. This is hardcoded in the program factory:
```cpp
tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
```
The compute hardware handles the format mismatch between input CB (which may be fp32) and scaler CB (bf16) through unpacker data format configuration.

### compute_kernel_hw_startup Requirements
This function performs MMIO writes that require idle execution units. It MUST be called:
- Exactly once at kernel start
- Before any other compute API calls
- With CB IDs matching the first operation's CBs

If subsequent operations need different CB formats, use `reconfig_data_format()` or `pack_reconfig_data_format()` instead of calling hw_startup again.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How do reduce_init, reduce_tile, and reduce_uninit work in the compute kernel API?"
   **Reason**: Needed to understand the exact semantics of the low-level reduce operations
   **Key Findings**: DeepWiki was unavailable; information obtained directly from `tt_metal/hw/inc/api/compute/reduce.h` header documentation

2. **Query**: "What is the circular buffer programming model? How do tile_regs_acquire/commit/wait/release work?"
   **Reason**: Needed to understand DST register lifecycle for compute kernel analysis
   **Key Findings**: DeepWiki was unavailable; information obtained from `tt_metal/hw/inc/api/compute/reg_api.h`

### Documentation References
1. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Understanding reduce_init, reduce_tile, reduce_uninit LLK API
   **Key Information**: reduce_init configures unpacker, math, and packer for reduce; reduce_tile performs the actual math+unpack; reduce_uninit MUST be called before next non-reduce operation to clear packer edge masks

2. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding hw_startup initialization requirements
   **Key Information**: Performs MMIO writes, must be called exactly once at kernel start, configures unpack/math/pack hardware for the provided CB IDs

3. **Source**: `tt_metal/hw/inc/api/compute/reg_api.h`
   **Reason**: Understanding DST register management (tile_regs_acquire/commit/wait/release)
   **Key Information**: New API separates MATH and PACK thread synchronization; tile_regs_acquire/commit for MATH, tile_regs_wait/release for PACK

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the reduce helper library that wraps low-level reduce operations
   **Key Information**: Single unified function handling all reduce patterns; supports multiple input policies, post-reduce operations, accumulation; auto-detects DEST register capacity

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Understanding scaler tile generation
   **Key Information**: `prepare_reduce_scaler` fills row 0 of each face with the scaler value; `calculate_and_prepare_reduce_scaler` auto-computes the value based on pool type and dimension

6. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: Understanding get_defines() which sets REDUCE_OP, REDUCE_DIM, REDUCE_ROW_SUM_VIA_MM
   **Key Information**: REDUCE_ROW_SUM_VIA_MM is set only for (SUM|AVG + W dimension); ReduceOpDim::W maps to ReduceDim::REDUCE_ROW

7. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity detection
   **Key Information**: DEST_AUTO_LIMIT = 4-16 tiles depending on sync mode and fp32 accumulation; used by reduce helper for REDUCE_COL chunking

8. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`
   **Reason**: Understanding the matmul-path scaler tile format
   **Key Information**: Fills specific positions (every 8th in faces 0 and 2) with single_packed_scalar (lower 16 bits only), different layout than prepare_reduce_scaler
