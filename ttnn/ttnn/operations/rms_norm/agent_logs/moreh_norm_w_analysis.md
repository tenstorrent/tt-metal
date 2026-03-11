# moreh_norm_w Implementation Analysis (Compute-Core Focus)

## Overview

The `moreh_norm_w` operation computes a norm along the **W (last) dimension** of a tensor. It supports multiple norm types via the parameter `p`, including L0 norm (count non-zero, `p=0`), L-infinity norm (max, general `p`), and negative-infinity norm (`p=-inf`). The general case applies `f(x) = abs(x)` per element, then reduces across W using `MAX`. The L0 case applies `f(x) = (x != 0)` and reduces with `SUM`.

**Program factory path**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp`

**Compute kernel path**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`

**Key relevance for RMSNorm**: This operation demonstrates the complete pattern for reducing along the W (final) dimension with a row-reduction strategy: iterating over Wt tiles per row, applying an element-wise transform, accumulating across tiles using an intermediate CB, then calling the `reduce` helper to produce one scalar-per-row output tile. The compute kernel structure (tile loop, CB accumulation, reduce helper call, mask handling) directly maps to what RMSNorm needs for `mean(x^2, dim=-1)`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | 1 tile-row = Wt input tiles producing 1 output tile |
| **Total units** | `num_units = (physical_volume / (H * W)) * Ht` -- i.e., number of tile-rows across all batches and H tiles |
| **Loop structure** | Outer: `num_rows_per_core` rows. Inner: `Wt` columns per row. Then one `reduce` call per row. |

Each "work unit" is a single tile-row: the Wt tiles that span the full W dimension at a given (batch, h-tile) position. After processing all Wt tiles for a row, the kernel produces exactly one output tile.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (arbitrary rank) | [..., H, 1] (W dimension reduced) |
| **Dimension convention** | Last two dims are H, W | Last two dims are H, 1 |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 (runtime check) | DRAM or L1 (runtime check) |
| **Data type** | bfloat16 or float32 | bfloat16 or float32 |

### Layout Transformations

None. Both input and output are in TILE_LAYOUT. The operation reads Wt tiles per row and writes 1 tile per row. The output tile contains the reduced values in column 0 of each row within the tile (standard output of REDUCE_ROW).

## Data Flow Pattern

### High-Level Flow (Per Row)

```
Reader: input tiles (Wt tiles) --> cb_x (c_in0)
Reader: scalar "1.0" tile    --> cb_one (c_in1)    [once, program lifetime]
Reader: mask_w tile           --> cb_mask_w (c_in2) [once, program lifetime, if needed]

Compute Phase 1 - Element-wise transform (per tile):
  cb_x --> f(x) --> cb_val (c_intermed0)

Compute Phase 2 - Accumulate across W dimension:
  col_idx==0: cb_val --> copy --> cb_cal (c_intermed1)
  col_idx>0:  cb_val + cb_cal --> max/add --> cb_cal (overwrite)

Compute Phase 3 - Reduce row:
  cb_cal --> reduce<REDUCE_OP, REDUCE_ROW>(cb_cal, cb_one, cb_reduce) --> cb_reduce (c_intermed2)

Compute Phase 4 - Copy to output:
  cb_reduce --> copy --> cb_y (c_out0)

Writer: cb_y (c_out0) --> output tensor in DRAM
```

### Detailed Compute Kernel Walkthrough

**Initialization** (line 34):
```cpp
binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);
```
This initializes the hardware for binary operations, setting up unpacker and packer data formats. The first two args specify source CBs, the third specifies the output CB. This is the standard way to bootstrap compute hardware configuration.

**Constant CB setup** (lines 36-44):
```cpp
cb_wait_front(cb_one, onetile);   // Wait for "1.0" scalar tile from reader
if (do_mask_w) {
    cb_wait_front(cb_mask_w, onetile);  // Wait for width mask tile from reader
}
```
These tiles persist for the entire program. They are waited on once at the start and popped once at the very end (lines 157-159). This is the **Program-lifetime persistent CB** pattern -- the tiles stay in the CB across all row iterations.

**Per-row loop** (lines 46-155):
```
for (row_idx = 0..num_rows_per_core):
    for (col_idx = 0..Wt):
        Phase 1: Apply f(x) to input tile --> cb_val
        Phase 2: Accumulate cb_val into cb_cal (running max or sum across W)
    Phase 3: Reduce cb_cal --> cb_reduce (single tile, row-reduced)
    Phase 4: Copy cb_reduce --> cb_y (output)
```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_in0 (0) | cb_x | Input tile staging | 1 tile | 1 tile | Single | Reader | Compute | Block (per tile within row) |
| c_in1 (1) | cb_one | Scalar constant "1.0" for reduce scaler | 1 tile | 1 tile | Single | Reader | Compute | Program (entire kernel) |
| c_in2 (2) | cb_mask_w | Width mask for partial last tile | 1 tile | 1 tile | Single | Reader | Compute | Program (entire kernel, conditional) |
| c_out0 (16) | cb_y | Output tile staging | 1 tile | 1 tile | Single | Compute | Writer | Block (per row) |
| c_intermed0 (24) | cb_val / cb_tmp0 | f(x) result per tile | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile within row) |
| c_intermed1 (25) | cb_cal / cb_tmp1 | Running accumulation across W dimension | 1 tile | 1 tile | Single | Compute | Compute | Row (persists across col_idx iterations within a row) |
| c_intermed2 (26) | cb_reduce / cb_tmp2 | Reduced result (output of reduce helper) | 1 tile | 1 tile | Single | Compute | Compute | Block (per row, consumed immediately) |

### Data Format Details

- **cb_x, cb_one, cb_mask_w, cb_y**: Use `cb_data_format` (matches input tensor dtype, e.g., bfloat16)
- **cb_val, cb_cal, cb_reduce** (intermediates): Use `intermed_data_format`, which is `Float32` when `fp32_dest_acc_en` is true, otherwise matches `cb_data_format`. This provides higher precision during accumulation.

### Multi-Pass Data Reuse Patterns

**cb_one (c_in1)** -- Program-lifetime persistent:
- Reader fills once with `fill_cb_with_value(cb_id_one, 1.0f)` which calls `cb_reserve_back` + `cb_push_back`
- Compute calls `cb_wait_front(cb_one, onetile)` once at program start
- The tile remains in the CB throughout all row iterations -- used repeatedly by the `reduce` helper (which internally does `cb_wait_front(scaler_cb, 1)` again, but the tile is already there)
- Compute calls `cb_pop_front(cb_one, onetile)` once at program end

**cb_mask_w (c_in2)** -- Program-lifetime persistent (conditional):
- Same pattern as cb_one. Only allocated and used when `origin_w % 32 != 0`
- Used during the f(x) phase on the last tile of each row (`col_idx == Wt - 1`)

**cb_cal (c_intermed1)** -- Row-lifetime accumulator:
- At `col_idx == 0`: initialized by copying cb_val into it (push_back)
- At `col_idx > 0`: popped, combined with cb_val (max or add), result pushed back
- After all Wt tiles: consumed by the `reduce` helper, then the CB is empty for the next row
- This is a **read-modify-write** pattern where the CB is both consumed and produced by the compute kernel within the same phase

## Pipeline Pattern Summary

All CBs have capacity = 1 tile and block size = 1 tile, making them **single-buffered**. This means no overlap between reader and compute for data tiles (cb_x). The reader must complete writing a tile before compute can start processing it, and compute must consume it before the reader can write the next one.

The constant CBs (cb_one, cb_mask_w) avoid this limitation because they are filled once and persist -- no reader/compute alternation needed during steady-state.

## Index Calculations

### Input Tile Indexing (Reader)
```
tile_idx = tile_offset + row_idx * Wt + col_idx
```
Where `tile_offset` is the starting tile index assigned to this core. Tiles are accessed in standard row-major order through the TensorAccessor.

### Output Tile Indexing (Writer)
```
tile_idx = tile_offset / Wt + row_idx
```
Since the output has W reduced to 1 tile, the output tile index is simply the row index offset by the starting row for this core.

### Masking Logic
```cpp
const bool do_mask_w = (origin_w % TILE_W) != 0;
const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;
```
When the original W dimension is not a multiple of 32, the last tile in each row has padding. The mask_w tile contains 1.0 in valid positions and 0.0 in padding positions. It is applied via `mask_tile(dst0, dst1)` which zeros out the padded elements. For the `-inf` norm variant, `mask_posinf_tile` is used instead (sets invalid positions to positive infinity, which is then negated).

## Memory Access Patterns

### Read Pattern
- **Sequential tile-by-tile**: Reader reads input tiles in row-major order. For each row, it reads Wt tiles sequentially. Each tile is read individually with `noc_async_read_tile` followed by `noc_async_read_barrier`.
- **Constant tiles**: cb_one and cb_mask_w are computed/filled once in L1 by the reader, not read from DRAM.

### Write Pattern
- **Sequential, one tile per row**: Writer outputs one tile per processed row, sequentially.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized over 2D grid) |
| **Grid dimensions** | Up to device `compute_with_storage_grid_size` |
| **Total cores** | `num_cores_to_be_used` (from `split_work_to_cores`) |
| **Work per core** | `num_units_per_core_group_1` or `num_units_per_core_group_2` tile-rows |
| **Load balancing** | Two core groups: group_1 gets `ceil(num_units / num_cores)` rows, group_2 gets `floor(num_units / num_cores)` rows |
| **Core enumeration** | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |

The `split_work_to_cores` utility divides `num_units` work items across available cores, creating two groups to handle remainders. Core group 1 processes one more unit than core group 2 when the division is uneven.

## Arguments

### Compile-Time Arguments

**Reader kernel**: TensorAccessorArgs for input buffer (variable-length, appended to `reader_ct_args`)

**Writer kernel**: TensorAccessorArgs for output buffer (variable-length, appended to `writer_ct_args`)

**Compute kernel**: No compile-time arguments. Behavior is controlled via preprocessor defines:

| Define | Type | Description |
|--------|------|-------------|
| `REDUCE_DIM` | string | Always `"ReduceDim::REDUCE_ROW"` -- reduction along W |
| `REDUCE_OP` | string | `"PoolType::SUM"` when p=0, else `"PoolType::MAX"` |
| `IS_ZERO` | string | Defined as `"1"` when p=0 (L0 norm mode) |
| `MINUS_INF` | string | Defined as `"1"` when p=-inf (negative infinity norm mode) |

### Runtime Arguments

**Compute kernel runtime args**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Number of tile-rows this core processes (a.k.a. num_units_per_core) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (tiles per row) |
| 2 | origin_w | uint32_t | Original (unpadded) W dimension size, for mask calculation |

**Reader kernel runtime args**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer DRAM address |
| 1 | input_is_dram | uint32_t | 1 if input is in DRAM, 0 if L1 |
| 2 | num_rows_per_core | uint32_t | Number of tile-rows |
| 3 | Wt | uint32_t | Tiles along W |
| 4 | tile_offset | uint32_t | Starting tile index for this core |
| 5 | origin_w | uint32_t | Original W size for mask generation |

**Writer kernel runtime args**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer DRAM address |
| 1 | output_is_dram | uint32_t | 1 if output is in DRAM |
| 2 | num_rows_per_core | uint32_t | Number of tile-rows |
| 3 | Wt | uint32_t | Tiles along W (used for output tile index calculation) |
| 4 | tile_offset | uint32_t | Starting tile offset |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_norm_w | RISCV_0 | NOC0 | DRAM (input tensor) | c_in0 (cb_x), c_in1 (cb_one), c_in2 (cb_mask_w) | Read input tiles, fill scalar, generate mask |
| moreh_norm_w_kernel | RISCV_2 | N/A | c_in0, c_in1, c_in2 | c_out0 (cb_y) via intermediates c_intermed0-2 | Element-wise transform, accumulate across W, reduce, output |
| writer_moreh_norm_w | RISCV_1 | NOC1 | c_out0 (cb_y) | DRAM (output tensor) | Write reduced output tiles |

### Compute Kernel Detailed Structure

**File**: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`

**Includes**:
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` -- the `compute_kernel_lib::reduce<>()` template
- `ttnn/kernel/compute/moreh_common.hpp` -- helper wrappers like `copy_tile_init_with_dt`, `pack_tile_with_dt`, `add_tiles_init_with_dt`, `mask_tile_init`, `mask_tile`, `abs_tile_init`, `abs_tile`, `negative_tile_init`, `negative_tile`, `binary_max_tile_init`, `binary_max_tile`, `unary_ne_tile_init`, `unary_ne_tile`

**Phase 1: Element-wise f(x) transform** (lines 49-85)

For each input tile, the compute kernel:
1. `tile_regs_acquire()` -- acquire DST registers
2. `cb_wait_front(cb_x, 1)` -- wait for reader to deliver input tile
3. `cb_reserve_back(cb_val, 1)` -- reserve space in intermediate CB
4. `copy_tile_init_with_dt(cb_x)` + `copy_tile(cb_x, 0, dst0)` -- unpack input tile to DST[0]
5. If last tile in row and masking needed: `copy_tile(cb_mask_w, 0, dst1)` + `mask_tile(dst0, dst1)` -- zero out padding
6. Apply element-wise function:
   - **IS_ZERO mode** (p=0): `unary_ne_tile_init()` + `unary_ne_tile(dst0, 0)` -- result is 1 where x!=0, 0 otherwise
   - **Default mode**: `abs_tile_init()` + `abs_tile(dst0)` -- absolute value
   - **MINUS_INF mode**: additionally `negative_tile_init()` + `negative_tile(dst0)` -- negate (so max of negated abs = min of abs)
7. `tile_regs_commit()` -- hand DST to packer
8. `tile_regs_wait()` + `pack_tile_with_dt(dst0, cb_val)` -- pack result to cb_val
9. `tile_regs_release()` -- release DST
10. `cb_pop_front(cb_x, 1)` + `cb_push_back(cb_val, 1)` -- consume input, produce intermediate

**Key function signatures**:
- `copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0)` -- reconfigures SRCA data format (for FP32 mode) and inits copy
- `copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)` -- copies tile at index `itile` from CB `icb` to DST register `idst`
- `pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)` -- reconfigures pack format (for FP32 mode) and packs DST[ifrom_dst] to CB `icb`
- `mask_tile_init()` -- initializes mask hardware
- `mask_tile(uint32_t dst0, uint32_t dst_mask)` -- applies mask: DST[dst0] elements where DST[dst_mask]==0 are set to 0
- `abs_tile_init()` + `abs_tile(uint32_t dst)` -- in-place absolute value on DST[dst]
- `negative_tile_init()` + `negative_tile(uint32_t dst)` -- in-place negation on DST[dst]
- `unary_ne_tile_init()` + `unary_ne_tile(uint32_t dst, uint32_t param)` -- compares DST[dst] != param, result is 1.0/0.0
- `binary_max_tile_init()` + `binary_max_tile(uint32_t dst0, uint32_t dst1, uint32_t dst_out)` -- element-wise max

**Phase 2: Accumulate across W** (lines 88-130)

This implements a running accumulation of f(x) values across the Wt tiles of a row:

- **First tile** (`col_idx == 0`): Simply copy cb_val to cb_cal
  ```cpp
  copy_tile_init_with_dt(cb_val);
  copy_tile(cb_val, 0, dst0);
  pack_tile_with_dt(dst0, cb_cal);
  ```

- **Subsequent tiles** (`col_idx > 0`): Combine cb_val with cb_cal
  - **IS_ZERO mode**: `add_tiles_init_with_dt(cb_val, cb_cal)` + `add_tiles(cb_val, cb_cal, 0, 0, dst0)` -- element-wise addition
  - **Default mode**: Load both to DST[0] and DST[1], then `binary_max_tile(dst0, dst1, dst0)` -- element-wise max

  The key **read-modify-write CB pattern**: `cb_wait_front(cb_cal, 1)` waits for the previous accumulated value, then `cb_pop_front(cb_cal, 1)` frees it, and `cb_push_back(cb_cal, 1)` writes the updated value back. This works because cb_cal has capacity=1 and the CB is used as a single-element scratchpad.

**Key function signatures for Phase 2**:
- `add_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)` -- reconfigures data format and inits add
- `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` -- adds tiles from two CBs into DST[idst]

**Phase 3: Row reduction** (lines 133-134)

```cpp
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_cal, cb_one, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::single());
```

This is the `reduce` helper from `reduce_helpers_compute.hpp`. Key parameters:
- **Template params**: `REDUCE_OP` (PoolType::SUM or PoolType::MAX), `REDUCE_DIM` (ReduceDim::REDUCE_ROW)
- **Default template params**: `input_policy = WaitAndPopPerTile`, `reconfig_mode = INPUT_AND_OUTPUT`
- **`input_block_shape = single()`**: This means `{rows=1, cols=1, batches=1}` -- a single tile. This is correct because Phase 2 already accumulated all Wt tiles into a single tile in cb_cal. The reduce helper just needs to perform the row-reduction (summing/maxing across the 32 columns within the single tile).
- **`cb_cal`**: input (1 tile containing the accumulated f(x) values)
- **`cb_one`**: scaler tile (contains 1.0, used as the reduce scaler)
- **`cb_reduce`**: output (1 tile containing the reduced result)

Since `input_block_shape::single()` is `{1,1,1}` and the policy is `WaitAndPopPerTile`, the reduce helper internally:
1. Calls `reduce_init<REDUCE_OP, REDUCE_ROW>()` to configure hardware
2. Waits for the scaler tile in cb_one (already present)
3. For the single tile: `cb_wait_front(cb_cal, 1)` + `reduce_tile(cb_cal, cb_one, 0, 0, 0)` + `cb_pop_front(cb_cal, 1)`
4. Packs the result into `cb_reduce`
5. Calls `reduce_uninit()`

The `reduce_tile` hardware operation for `REDUCE_ROW` sums/maxes all 32 columns in each row of the tile, producing a column vector (values replicated in column 0 of the output tile).

**Phase 4: Copy to output** (lines 136-154)

```cpp
tile_regs_acquire();
cb_wait_front(cb_reduce, onetile);
cb_reserve_back(cb_y, onetile);
copy_tile_init_with_dt(cb_reduce);
copy_tile(cb_reduce, 0, dst0);
// Optional: negative_tile for MINUS_INF mode
tile_regs_commit();
tile_regs_wait();
pack_tile_with_dt(dst0, cb_y);
tile_regs_release();
cb_pop_front(cb_reduce, onetile);
cb_push_back(cb_y, onetile);
```

This moves the reduced result from cb_reduce to cb_y (output CB). For the `-inf` norm, an additional `negative_tile` is applied to undo the earlier negation.

**Cleanup** (lines 157-160):
```cpp
cb_pop_front(cb_one, onetile);
if (do_mask_w) {
    cb_pop_front(cb_mask_w, onetile);
}
```
Pop the persistent constant tiles at program end.

## Implementation Notes

### Scalar/Constant CB Setup Pattern
The reader kernel generates two constant tiles that persist for the program's duration:
1. **cb_one**: Filled with `1.0f` via `fill_cb_with_value(cb_id_one, one.u)`. This writes the float value to all 1024 elements in the tile. Used as the scaler for the `reduce` helper.
2. **cb_mask_w**: Generated via `generate_mask_w(cb_id_mask_w, mask_w)`. Creates a tile with 1.0 in valid column positions and 0.0 in padding positions. The mask respects the tile's 4-subtile layout (16x16 each): subtile 0 (top-left), subtile 1 (top-right), subtile 2 (bottom-left), subtile 3 (bottom-right).

### Why the Two-Phase Accumulation Pattern (Phase 2 + Phase 3)
Instead of calling `reduce<SUM, REDUCE_ROW>(cb_x, cb_one, cb_reduce, ReduceInputBlockShape::row(Wt))` directly on the input, this kernel uses a manual accumulation loop (Phase 2) followed by a single-tile reduce (Phase 3). This is because:
- Phase 2 applies different operations (MAX vs SUM) depending on the norm type, which cannot be expressed as a standard reduce
- Phase 2 includes masking on the last tile, which happens before accumulation
- The `reduce` helper in Phase 3 handles the final within-tile row reduction (collapsing 32 columns to 1)

**For RMSNorm**, this two-phase approach is not needed since RMSNorm uses a simple SUM reduction. RMSNorm can directly use `reduce<SUM, REDUCE_ROW>` with `ReduceInputBlockShape::row(Wt)` to reduce the squared values across all Wt tiles in a single call, avoiding the manual accumulation loop entirely.

### Compute Defines as Branch Selection
The kernel uses preprocessor defines (`IS_ZERO`, `MINUS_INF`) rather than runtime conditionals for the norm-type branching. These are set in the program factory and compiled into the kernel at dispatch time, enabling dead-code elimination.

### Data Format Handling (FP32 Accumulation)
When `fp32_dest_acc_en` is true:
- Intermediate CBs use `Float32` format for higher precision during accumulation
- All `_with_dt` helper functions (`copy_tile_init_with_dt`, `pack_tile_with_dt`, etc.) include `#if defined FP32_DEST_ACC_EN` blocks that call `reconfig_data_format_srca()` or `pack_reconfig_data_format()` before the actual operation. This is necessary because the unpacker/packer need to be reconfigured when switching between different data formats (e.g., bfloat16 input to float32 intermediate).

### Binary Op Broadcast Pattern in Reduce
The `reduce_tile<SUM/MAX, REDUCE_ROW>()` hardware instruction internally multiplies the input tile by the scaler tile (cb_one) during reduction. Since the scaler is all 1.0, this is effectively a no-op multiplication, but the hardware architecture requires a scaler operand for all reduce operations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do tile_regs_acquire, tile_regs_commit, tile_regs_wait, and tile_regs_release work together for DST register management in compute kernels?"
   **Reason**: The compute kernel uses these four functions extensively and understanding their synchronization semantics is critical for understanding the pipeline.
   **Key Findings**: These implement a producer-consumer handoff between the math engine and packer. `acquire` gives the math core exclusive access to DST. `commit` transfers ownership to the packer. `wait` blocks the packer until data is ready. `release` makes DST available for the next iteration. This is a two-phase lock that enables overlapped execution between unpack/math and pack stages.

2. **Query**: "How does the reduce operation work in tt-metal compute kernels? Specifically, how does reduce_tile work for REDUCE_ROW with PoolType::SUM and PoolType::MAX?"
   **Reason**: Understanding the `reduce_tile` hardware instruction is essential for understanding Phase 3 of the compute kernel and how it produces the final reduced output.
   **Key Findings**: `reduce_tile` takes an input tile, a scaler tile, and accumulates results in a DST register. For REDUCE_ROW, it reduces across the W (column) dimension, producing a column vector. The scaler tile multiplies the input during reduction (typically 1.0 for SUM/MAX). `reduce_init` must be called before and `reduce_uninit` after. The function handles both the unpack and math phases internally.

3. **Query**: "What is the split_work_to_cores function in tt-metal and how does it distribute work across core groups?"
   **Reason**: Understanding how work units are distributed across cores is essential for the core distribution analysis.
   **Key Findings**: DeepWiki query failed, but the function signature and usage in the program factory is clear: it takes a grid size and number of work units, returns two core groups where group_1 has `ceil(units/cores)` work items and group_2 has `floor(units/cores)`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the full reduce helper API including all template parameters, input policies, and the REDUCE_ROW implementation.
   **Key Information**: The `reduce` template function handles REDUCE_ROW by looping over `Wt` tiles, calling `reduce_tile` for each, then packing the result. The `WaitAndPopPerTile` policy (default) waits and pops one tile at a time. `ReduceInputBlockShape::single()` means `{1,1,1}` -- suitable when the caller has pre-accumulated tiles.

2. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding all the `_with_dt` helper wrappers used in the compute kernel.
   **Key Information**: These wrappers handle FP32_DEST_ACC_EN data format reconfiguration. Key functions: `copy_tile_init_with_dt`, `pack_tile_with_dt`, `add_tiles_init_with_dt`, `mul_tiles_init_with_dt`, and broadcast variants. Also provides composite helpers like `mul_tiles_to_cb`, `recip_tile_to_cb`, etc.

3. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding how constant tiles (scalar, mask) are generated in the reader.
   **Key Information**: `fill_cb_with_value(cb_id, value)` writes a float/uint value to all 1024 elements in a tile. `generate_mask_w(cb_mask, mask_w)` creates a tile with 1.0/0.0 values respecting the 4-subtile layout, where columns < mask_w get 1.0 and columns >= mask_w get 0.0.
