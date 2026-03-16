# Softmax (tt-train) Implementation Analysis

## Overview

This analysis covers the tt-train softmax implementation, which computes `softmax(x) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along the width (innermost) dimension of a 4D tensor. The implementation uses max subtraction for numerical stability, preventing overflow in the exponential. It operates on DRAM-interleaved, tiled, bfloat16 tensors.

**Program factory path**: `tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`

**Focus**: This analysis emphasizes the compute kernel structure, CB layout for intermediates, multi-pass data reuse patterns, scalar/constant CB setup, reduce helper parameters, binary op broadcast patterns, and the three-phase softmax algorithm (max, exp-sum, normalize).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles |
| **Unit size** | Wt tiles (one tile-row across the width dimension) |
| **Total units** | `NC * Ht` rows, where NC = N*C (batch dims), Ht = padded_H / TILE_HEIGHT |
| **Loop structure** | Outer: rows assigned to this core; Inner: blocks of `block_size` tiles within each row |

One "work unit" is a single tile-row of width Wt tiles. Each core processes `num_rows_per_core` such rows. The softmax algorithm requires three full passes over each row (in the non-L1-fit path), or a single read with persistent storage (in the L1-fit path).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (enforced via assertion) |
| **Data type** | Float16_b (bfloat16, enforced via assertion) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (enforced via assertion) |
| **Data type** | Float16_b |

### Layout Transformations
No tilize/untilize or reshard conversions. Input and output share the same layout. All computation occurs in tile format.

## Data Flow Pattern

The softmax algorithm proceeds in four phases per row, with two distinct execution paths depending on whether the entire row fits in L1.

### Path A: EVERYTHING_FITS_IN_L1 (when `2 * Wt * tile_size + overhead <= available_L1`)

| Stage | Phase | Kernel | Reads From | Writes To | Description |
|-------|-------|--------|------------|-----------|-------------|
| 1 | Setup | Reader | DRAM | CB_mask(c_1), CB_max_mask(c_2), CB_scaler(c_3), CB_matmul(c_4) | Generate constant tiles once |
| 2 | Read Row | Reader | DRAM | CB_input(c_0) | Read entire Wt tiles for the row (single read, persists) |
| 3 | Find Max | Compute | CB_input(c_0) | CB_max_before(c_5) | Scan all Wt tiles, find element-wise max across row |
| 4 | Reduce Max | Compute | CB_max_before(c_5) | CB_max_after(c_6) | Row-reduce max tile to column vector |
| 5 | Exp+Sum | Compute | CB_input(c_0), CB_max_after(c_6) | CB_exp(c_7), CB_sum_before(c_8) | Sub max, exp, store exp tiles, accumulate sum |
| 6 | Reduce Sum | Compute | CB_sum_before(c_8), CB_matmul(c_4) | CB_sum_after(c_9) | Row-reduce sum, compute reciprocal |
| 7 | Normalize | Compute | CB_exp(c_7), CB_sum_after(c_9) | CB_output(c_10) | Multiply exp tiles by 1/sum |
| 8 | Write | Writer | CB_output(c_10) | DRAM | Write output tiles |

Key insight: CB_input (c_0) persists across phases 3 and 5 (not popped until end of row). CB_exp (c_7) persists across phases 5 and 7. This avoids re-reading from DRAM but requires `2 * Wt` tiles of L1 for input + exp.

### Path B: Non-L1-fit (default, when row is too large for L1)

| Stage | Phase | Kernel | Reads From | Writes To | Description |
|-------|-------|--------|------------|-----------|-------------|
| 1 | Setup | Reader | DRAM | CB_mask(c_1), CB_max_mask(c_2), CB_scaler(c_3), CB_matmul(c_4) | Generate constant tiles once |
| 2 | Read Row (pass 1) | Reader | DRAM | CB_input(c_0) | Stream Wt tiles in blocks of block_size |
| 3 | Find Max | Compute | CB_input(c_0) | CB_max_before(c_5) | Process blocks, pop each block after use |
| 4 | Reduce Max | Compute | CB_max_before(c_5) | CB_max_after(c_6) | Row-reduce max tile to column vector |
| 5 | Read Row (pass 2) | Reader | DRAM | CB_input(c_0) | Re-read entire row from DRAM |
| 6 | Exp+Sum | Compute | CB_input(c_0), CB_max_after(c_6) | CB_sum_before(c_8) | Sub max, exp, accumulate sum in registers (no exp storage) |
| 7 | Reduce Sum | Compute | CB_sum_before(c_8), CB_matmul(c_4) | CB_sum_after(c_9) | Row-reduce sum, compute reciprocal |
| 8 | Read Row (pass 3) | Reader | DRAM | CB_input(c_0) | Re-read entire row from DRAM |
| 9 | Normalize | Compute | CB_input(c_0), CB_max_after(c_6), CB_sum_after(c_9) | CB_output(c_10) | Sub max, exp, multiply by 1/sum |
| 10 | Write | Writer | CB_output(c_10) | DRAM | Write output tiles |

Key insight: Reader reads the row 3 times from DRAM. Compute pops input blocks after each use. CB_max_after (c_6) persists across phases 6 and 9 (used in both exp+sum and normalize). The non-L1 normalize path recomputes `exp(x - max)` rather than storing it.

### Reader/Writer Summary (de-emphasized per scope)

- **Reader** provides: constant tiles (mask, max_mask, scaler, matmul_reduce), and input tiles (1x or 3x per row depending on L1 path). Uses `TensorAccessor` for address generation.
- **Writer** consumes: CB_output (c_10) tiles, writes to DRAM via `TensorAccessor`.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Data Format | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|---------------------|-------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | Wt (L1-fit) or 2*block_size (streaming) | block_size | Float16_b | Single (L1-fit) or Double (streaming) | Reader | Compute | Row (L1-fit) or Block (streaming) |
| c_1 | cb_mask | Width mask (1/0 pattern) | 1 | 1 | Float16_b | Single | Reader | Compute | Program |
| c_2 | cb_max_mask | Width mask (0/-inf pattern) | 1 | 1 | Float16_b | Single | Reader | Compute | Program |
| c_3 | cb_reduction_scaler | All-ones tile for reduce_tile | 1 | 1 | Float16_b | Single | Reader | Compute | Program |
| c_4 | cb_mat_mul_reduce | Column-vector tile (1s in col 0) for matmul row-reduce | 1 | 1 | Float16_b | Single | Reader | Compute | Program |
| c_5 | cb_max_value_before_reduction | Unreduced row-max tile | 2 | 1 | Float16_b | Double | Compute | Compute | Per-phase |
| c_6 | cb_max_value_after_reduction | Reduced row-max (column vector) | 2 | 1 | Float16_b | Double | Compute | Compute | Row |
| c_7 | cb_exp | exp(x-max) tiles (L1-fit only) | Wt (L1-fit) or 2*block_size (streaming) | block_size | Float16_b | Same as c_0 | Compute | Compute | Row (L1-fit only) |
| c_8 | cb_exp_sum_before_reduction | Unreduced exp-sum tile | 2 | 1 | Float32 | Double | Compute | Compute | Per-phase |
| c_9 | cb_exp_sum_after_reduction | Reduced 1/sum(exp) (column vector) | 2 | 1 | Float32 | Double | Compute | Compute | Per-phase |
| c_10 | cb_output | Output tile staging | 2*block_size | block_size | Float16_b | Double | Compute | Writer | Block |

### CB Persistence Across Phases

| CB | Phase: Find Max | Phase: Exp+Sum | Phase: Reduce Sum | Phase: Normalize | Notes |
|----|----------------|----------------|-------------------|-----------------|-------|
| c_0 (input) | Read, **kept** (L1) / popped (stream) | Read, **kept** (L1) / popped (stream) | N/A | **Kept** (L1) / re-read (stream) | L1-fit: persists entire row; stream: popped per block |
| c_6 (max_after) | N/A | Read, **kept** | N/A | Read, **popped at end of row** | Persists from reduce_max through normalize |
| c_7 (exp) | N/A | Written, **kept** (L1 only) | N/A | Read, **popped at end of row** (L1 only) | Only used in L1-fit path |
| c_9 (sum_after) | N/A | N/A | Written | Read, **popped at end of row** | Contains 1/sum(exp) |

### Scalar/Constant CB Setup (Reader Kernel)

The reader kernel generates four constant tiles at program start (before the row loop):

1. **cb_mask (c_1)**: Width mask tile via `generate_mask_tile(cb_mask_idx, one=0x3F80, zero=0x0, mask_w)`. For each 16x16 face, columns `< mask_w` get bfloat16 1.0, columns `>= mask_w` get 0.0. Used to zero out padding in exp tiles.

2. **cb_max_mask (c_2)**: Max-mask tile via `generate_mask_tile(cb_max_mask_idx, zero=0x0, minus_inf=0xFF80, mask_w)`. Columns `< mask_w` get 0.0, columns `>= mask_w` get -inf. Added to the max-finding tile so padding positions become -inf and cannot win the max.

3. **cb_reduction_scaler (c_3)**: All-ones tile via `generate_tile_with_bfloat16_value(cb_reduction_scaler_idx, 0x3F80)`. Every element is bfloat16 1.0. Used as the scaler argument to `reduce_tile<PoolType::MAX, REDUCE_ROW>`.

4. **cb_mat_mul_reduce (c_4)**: Column-vector tile via `generate_matmul_row_reduce_tile(cb_matmul_reduce)`. Pattern: 1.0 in the first column of even (left) faces, 0.0 everywhere else. Used with `matmul_tiles` for precision-preserving row reduction of the exp-sum.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Buffering Type |
|----|----------|------------|-------|----------------|
| c_0 (input, L1-fit) | Wt | block_size | Wt/block_size | Multi (all tiles resident) |
| c_0 (input, streaming) | 2*block_size | block_size | 2 | Double |
| c_5 (max_before) | 2 | 1 | 2 | Double |
| c_6 (max_after) | 2 | 1 | 2 | Double |
| c_7 (exp, L1-fit) | Wt | block_size | Wt/block_size | Multi (all tiles resident) |
| c_8 (sum_before) | 2 | 1 | 2 | Double |
| c_9 (sum_after) | 2 | 1 | 2 | Double |
| c_10 (output) | 2*block_size | block_size | 2 | Double |
| c_1, c_2, c_3, c_4 | 1 | 1 | 1 | Single (constant) |

## Index Calculations

### Tile Indexing Within a Row

The tensor is viewed as `NC * Ht` rows, each `Wt` tiles wide. For a given row `r` (0-indexed across all batches/heights):
- Tile index in linearized tensor: `r * Wt + col`, where `col` ranges from 0 to Wt-1
- The reader computes: `idx = (start_row + i) * Wt` as the base tile index for row `i`

### Block Indexing

Within a row, tiles are processed in blocks of `block_size`:
- Block start: `col` (incremented by `block_size` each iteration)
- Tile within block: `block_idx` (0 to `block_size - 1`)
- For the L1-fit path, tiles in CB are indexed absolutely: `cb_input[col]`, `cb_exp[col + block_idx]`
- For the streaming path, tiles are indexed relative to the CB front: `cb_input[block_idx]`

### Mask Width Calculation

`mask_w = num_inner % TILE_WIDTH` (width of valid data in the last tile of a row). When `mask_w == 0`, no masking is needed (all tiles are fully populated). The `DO_MASK_W` define is only set when `mask_w != 0`.

### Block Size Selection

`block_size = get_block_size(Wt, 3)`: finds the largest divisor of Wt in {3, 2, 1}. The max is 3 (not 4) because the compute kernel needs one extra DST register during calculation (e.g., for the mask register adjacent to the data register, or for the max/sum accumulator).

## Memory Access Patterns

### Read Pattern
- **DRAM reads**: Sequential tile reads within each row, organized in blocks of `block_size`
- **Streaming path**: Each row is read 3 times from DRAM (once per phase: max, exp-sum, normalize)
- **L1-fit path**: Each row is read once from DRAM; all subsequent accesses are from L1 CBs
- `read_full_row_tiles` issues `noc_async_read_page` calls for each tile, with a barrier after each block

### Write Pattern
- **DRAM writes**: Sequential tile writes within each row, organized in blocks of `block_size`
- `write_full_row_tiles` issues `noc_async_write_page` calls for each tile, with a barrier after each block

### L1 Access
- Compute kernel accesses CBs for all intermediate tile operations
- Constants (mask, scaler, matmul-reduce) are resident in L1 for the entire program lifetime
- In the L1-fit path, both input and exp CBs hold the entire row simultaneously

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores` (min of available cores and total rows) |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` rows |
| **Load balancing** | Two-group: group_1 gets `ceil(total_rows/num_cores)` rows, group_2 gets `floor(total_rows/num_cores)` rows |

Cores are addressed using column-major ordering: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides `total_rows_to_process = NC * Ht` across available cores, producing two core groups where group_1 cores handle one more row than group_2 cores.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Number of tile-rows this core processes (differs between group_1 and group_2) |
| 1 | block_size | uint32_t | Tiles processed per inner-loop iteration (1, 2, or 3) |
| 2 | Wt | uint32_t | Number of tiles in the width dimension of the padded tensor |

#### Compile-Time Defines (shared across all kernels)

| Define | Condition | Effect |
|--------|-----------|--------|
| `DO_MASK_W` | `mask_w != 0` | Enables width masking in last tile of each row |
| `EVERYTHING_FITS_IN_L1` | `required_L1 <= available_L1` | Selects single-read path with persistent CB storage |
| `REDUCE_OP` | Always | Set to `PoolType::SUM` (required by LLK reduce API default template param) |
| `REDUCE_DIM` | Always | Set to `ReduceDim::REDUCE_ROW` (required by LLK reduce API default template param) |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Tiles per block for read operations |
| 1 | Wt | uint32_t | Tiles in width dimension |
| 2 | mask_w | uint32_t | Number of valid elements in the last tile's width |
| 3+ | TensorAccessor args | uint32_t[] | Address generation parameters for input buffer |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Tiles per block for write operations |
| 1 | Wt | uint32_t | Tiles in width dimension |
| 2+ | TensorAccessor args | uint32_t[] | Address generation parameters for output buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_address | uint32_t | DRAM address of input buffer |
| 1 | num_rows_to_process | uint32_t | Number of rows assigned to this core |
| 2 | start_row | uint32_t | First row index for this core (cumulative offset) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | DRAM address of output buffer |
| 1 | num_rows_to_process | uint32_t | Number of rows assigned to this core |
| 2 | start_row | uint32_t | First row index for this core (cumulative offset) |

#### Compute Kernel
No runtime arguments. All parameters are compile-time.

## Kernel Implementations

### Kernel Overview

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | c_0, c_1, c_2, c_3, c_4 | Read input tiles, generate constants |
| compute | RISCV_2 | N/A | c_0..c_4 | c_5..c_10 | Max, exp, sum, normalize |
| writer | RISCV_1 | NOC1 | c_10 | DRAM | Write output tiles |

### Compute Kernel Deep Dive

**File**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp`

**Configuration**: `MathFidelity::HiFi4`, `fp32_dest_acc_en = true`, `math_approx_mode = false`

#### Initialization (kernel_main, lines 373-381)

```
if (do_mask_w): cb_wait_front(cb_mask, 1); cb_wait_front(cb_max_mask, 1)
cb_wait_front(cb_reduction_scaler, 1)
init_sfpu(cb_input, cb_output)
binary_op_init_common(cb_input, cb_input, cb_output)
```

The kernel waits for constant tiles to be available, then initializes the SFPU and binary operation hardware. `init_sfpu` configures the vector engine. `binary_op_init_common` configures the unpack/math pipeline for binary operations.

#### Phase 1: find_max_value_in_row()

**Exact function signatures used**:
- `copy_tile_init(cb_id)` / `copy_tile(cb_id, tile_idx, register_idx)` -- unpack tile from CB to DST register
- `mask_tile_init()` / `mask_tile(idst_data, idst_mask)` -- zero out padding elements; mask register must be `data_register + 1`
- `add_binary_tile_init()` / `add_binary_tile(reg_a, reg_b, reg_dst)` -- element-wise add in DST registers
- `binary_max_tile_init()` / `binary_max_tile(reg_a, reg_b, reg_dst)` -- element-wise max in DST registers
- `reconfig_data_format(cb_a, cb_b)` -- reconfigure unpacker for new data formats
- `pack_reconfig_data_format(cb_out)` -- reconfigure packer for output format
- `pack_tile(register_idx, cb_id)` -- pack DST register to CB

**Algorithm (L1-fit path)**:
1. `tile_regs_acquire()` -- acquire DST registers
2. For each tile `col` in [0, Wt), processed in blocks of `block_size`:
   - `cb_wait_front(cb_input, col + block_size)` -- wait for tiles to be available (accumulating, not popping)
   - `copy_tile(cb_input, col, working_register)` -- unpack tile `col` from CB to DST
   - If `col == 0`: use register 0 (max accumulator)
   - If `col > 0`: use register 1 (temp), then `binary_max_tile(reg_0, reg_1, reg_0)` to accumulate max
   - If last tile AND masking enabled:
     - Copy mask to `working_register + 1` (adjacency constraint)
     - `mask_tile(working_register, mask_register)` -- zero out padding
     - Copy max_mask to `mask_register`
     - `add_binary_tile(working_register, mask_register, working_register)` -- add -inf to padding positions
3. `tile_regs_commit()` / `tile_regs_wait()` -- hand off to packer
4. `pack_tile(max_value_register, cb_max_value_before_reduction)` -- store unreduced max
5. `cb_push_back(cb_max_value_before_reduction, 1)`

**Algorithm (streaming path)**: Same logic but:
- `cb_wait_front(cb_input, block_size)` per block (not accumulating)
- `copy_tile` uses `block_idx` not `col` (relative to CB front)
- `cb_pop_front(cb_input, block_size)` after each block

**Masking for numerical stability**: The mask+max_mask pattern is critical. For the last tile:
1. `mask_tile` zeroes padding positions (prevents NaN from padding containing NaN)
2. `add_binary_tile` with max_mask adds -inf to zeroed padding positions
3. This ensures padding cannot produce a max value; the max is determined solely by valid data

#### Phase 2: reduce_max_value()

**Exact function signatures used**:
- `reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out)` -- initialize reduce hardware
- `reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, tile_idx_in, tile_idx_scaler, dst_reg)` -- perform reduction
- `reduce_uninit()` -- reset packer edge mask

**Algorithm**:
1. Wait for unreduced max in c_5, reserve space in c_6
2. `tile_regs_acquire()`
3. `reconfig_data_format(cb_max_value_before_reduction, cb_reduction_scaler)`
4. `reduce_init<MAX, REDUCE_ROW>(c_5, c_3, c_6)` -- configure for max-reduce along row
5. `reduce_tile<MAX, REDUCE_ROW>(c_5, c_3, 0, 0, reg_0)` -- reduce the 32x32 max tile: each row's max is condensed into the first column, producing a column vector
6. `reduce_uninit()` -- clean up packer state
7. `tile_regs_commit()` / `tile_regs_wait()`
8. Pack result to c_6, pop from c_5

**The scaler (c_3) is all 1.0**: For MAX reduction, the scaler multiplies each element during reduction. With scaler=1.0, the values are unchanged.

**Result**: c_6 contains a tile where each row holds the maximum value for that row of the original input (broadcast-ready as a column vector).

#### Phase 3: calculate_sum_exp_x()

**Exact function signatures used (L1-fit path)**:
- `sub_bcast_cols_init_short(cb_a, cb_b)` -- init subtraction with column broadcast from cb_b
- `sub_tiles_bcast<BroadcastType::COL>(cb_a, cb_b, tile_a, tile_b, dst_reg)` -- subtract column vector from tile
- `exp_tile_init<false>()` / `exp_tile<false>(dst_reg)` -- compute exp (exact, not approximate)
- `copy_tile_init/copy_tile` -- tile copy for accumulation
- `add_binary_tile_init/add_binary_tile` -- element-wise add for sum accumulation

**Algorithm (L1-fit path)**:
1. Wait for max in c_6; reserve Wt tiles in c_7 (exp storage)
2. For each block of `block_size` tiles:
   - `tile_regs_acquire()`
   - `sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction)`
   - For each tile in block:
     - `sub_tiles_bcast<COL>(cb_input, cb_max_after, col, 0, block_idx)` -- compute `x - max(x)` with column broadcast
     - `exp_tile<false>(block_idx)` -- compute `exp(x - max(x))` in place in DST register
     - If last tile + masking: apply mask to zero out padding in exp
   - `tile_regs_commit()` / `tile_regs_wait()`
   - Pack block to c_7, push
3. Wait for all Wt exp tiles in c_7
4. Accumulate sum: load exp tiles one by one from c_7, add them together in DST register
5. Pack accumulated sum to c_8 (exp_sum_before_reduction)

**Algorithm (streaming path)**:
- Uses `unary_bcast_init/unary_bcast<COL>` to preload max value into a register (reg_2), since it needs to be used tile-by-tile as input is streamed
- `sub_binary_tile(working_register, max_value_register, working_register)` instead of `sub_tiles_bcast<COL>` for subtraction (operates on register-to-register, not CB-to-register)
- Does NOT store exp tiles; instead accumulates the element-wise sum on the fly
- `cb_pop_front(cb_input, block_size)` after each block
- Result goes directly to c_8

**Key difference**: L1-fit stores all exp tiles (for reuse in normalization), while streaming recomputes exp in phase 4.

**Broadcast pattern**: `sub_tiles_bcast<BroadcastType::COL>` broadcasts the first column of the max tile across all columns during subtraction. Since the max was row-reduced, each row of the max tile has the same value in column 0, and COL broadcast replicates that across the full 32-wide tile.

#### Phase 4: reduce_sum_exp_x()

**Exact function signatures used**:
- `mm_init(cb_a, cb_b, cb_out, transpose)` -- initialize matmul hardware
- `matmul_tiles(cb_a, cb_b, tile_a, tile_b, dst_reg)` -- matrix multiply for reduction
- `recip_tile_init()` / `recip_tile(dst_reg)` -- compute 1/x in place in DST

**Algorithm**:
1. Wait for sum tile in c_8, reserve space in c_9
2. Wait for matmul_reduce tile in c_4
3. `tile_regs_acquire()`
4. `mm_init(c_8, c_4, c_9, 0)` -- configure matmul: input x column_vector
5. `matmul_tiles(c_8, c_4, 0, 0, reg_0)` -- multiply sum tile (32x32) by column vector (1s in col 0, 0s elsewhere). This computes the dot product of each row with the ones-vector, effectively summing each row into column 0.
6. `recip_tile(reg_0)` -- compute `1 / sum(exp(x))` in place, producing the reciprocal for normalization
7. Pack to c_9, pop from c_8

**Why matmul instead of reduce_tile**: The commented-out code (lines 344-351) shows the original `reduce_tile<SUM, REDUCE_ROW>` approach. The comment on line 354 explains: "We used matmul_tiles instead of reduce_tile, because reduce_tile causes a loss of precision. The same issue has been observed in moreh's ops." The matmul approach uses the FPU's matrix unit, which can accumulate in FP32 when `fp32_dest_acc_en = true`, providing better numerical accuracy for the summation.

**Note on precision**: c_8 and c_9 are Float32 format, while most other CBs are Float16_b. This preserves precision during the critical sum and reciprocal computation.

#### Phase 5: Normalization (in kernel_main loop, lines 390-454)

**Exact function signatures used**:
- `unary_bcast_init<BroadcastType::COL>(cb, cb)` / `unary_bcast<BroadcastType::COL>(cb, tile_idx, dst_reg)` -- broadcast column vector to full tile in DST
- `copy_tile_init/copy_tile` -- copy tile from CB to DST (L1-fit path)
- `sub_bcast_cols_init_short/sub_tiles_bcast<COL>` -- subtract with column broadcast (streaming path)
- `exp_tile_init<false>()/exp_tile<false>()` -- exp computation (streaming path only)
- `mul_binary_tile_init()` / `mul_binary_tile(reg_a, reg_b, reg_dst)` -- element-wise multiply in DST

**Algorithm**:
1. Wait for 1/sum in c_9
2. `unary_bcast<COL>(cb_exp_sum_after_reduction, 0, sum_exp_register)` -- broadcast the reciprocal column vector to a full tile in DST register `block_size` (so register 0..block_size-1 are for data, register `block_size` holds 1/sum)
3. For each block of tiles:
   - **L1-fit**: `copy_tile(cb_exp, col + block_idx, block_idx)` -- read precomputed exp from c_7
   - **Streaming**: `sub_tiles_bcast<COL>(cb_input, cb_max_after, block_idx, 0, block_idx)` then `exp_tile<false>(block_idx)` -- recompute exp(x - max)
   - `mul_binary_tile(block_idx, sum_exp_register, block_idx)` -- multiply by 1/sum
4. Pack to c_10 (output), push
5. At end of row: pop c_6 (max), c_9 (sum); in L1-fit also pop c_0 (Wt tiles) and c_7 (Wt tiles)

**Register allocation in normalization**:
- Registers 0 through `block_size - 1`: working tiles
- Register `block_size`: holds the broadcast 1/sum value (persists across the block loop)
- This is why `block_size` max is 3 (needs register `block_size` = 3 for 1/sum, using 4 registers total, which fits in the available DST register space)

### Reader Kernel (summary)

**File**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/reader_softmax_interleaved_start_id.cpp`

Generates constant tiles (mask, max_mask, scaler, matmul_reduce) once at startup. Then for each row, calls `read_full_row_tiles` which issues block-sized DRAM reads. In the streaming path, it reads the same row 3 times. Uses `TensorAccessor` for address generation from interleaved buffer.

### Writer Kernel (summary)

**File**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/writer_softmax_interleaved_start_id.cpp`

For each row, calls `write_full_row_tiles` which waits for compute to fill cb_output, then issues block-sized DRAM writes. Uses `TensorAccessor` for address generation.

## Implementation Notes

### Numerical Stability Strategy
The implementation uses the standard numerically stable softmax: subtract the row maximum before exponentiating. This prevents exp overflow (since max(x-max) = 0, so exp values are in [0, 1]). The max subtraction also uses masking to set padding positions to -inf, ensuring they become exactly 0 after exp.

### Two-Phase Masking for Max Finding
A subtle but important detail: the max-finding phase uses TWO mask tiles:
1. `cb_mask (c_1)`: 1.0 for valid, 0.0 for padding -- applied via `mask_tile` to zero padding
2. `cb_max_mask (c_2)`: 0.0 for valid, -inf for padding -- added via `add_binary_tile`

The reason for this two-step approach (zero then add -inf, rather than directly setting to -inf): `mask_tile` is a hardware SFPU operation that zeroes elements based on the mask pattern. The -inf values need to be added separately because `mask_tile` only supports zeroing, not arbitrary value replacement. The comment in the code (lines 79-81) explains that NaN in padding can cause issues: `NaN + (-inf) = NaN` and `-inf * 0 = NaN`, so the padding must first be cleanly zeroed before -inf is added.

### L1 Memory Budget Decision
The program factory computes `required_L1_in_bytes = 2 * input_memory + masks + scalers + max_value + exp_sum + output`. The factor of 2 on `input_memory` accounts for both `cb_input` (Wt tiles) and `cb_exp` (Wt tiles) needing to be resident simultaneously. If this fits, the single-read path is used; otherwise, the three-read streaming path is used.

### FP32 Accumulation for Sum
The exp-sum CBs (c_8, c_9) use Float32 format even though input/output are Float16_b. Combined with `fp32_dest_acc_en = true` in the compute config, this ensures the sum of exponentials is accumulated at full precision, preventing the numerical instability that would arise from accumulating many small bfloat16 values.

### matmul_tiles for Row Reduction (Precision Workaround)
The `reduce_tile<SUM, REDUCE_ROW>` API was found to lose precision for the exp-sum reduction. The workaround uses `matmul_tiles` with a specially crafted column vector tile (1.0 in first column of left faces, 0.0 elsewhere). Matrix-multiplying a 32x32 tile by this column vector effectively sums each row into a single column, using the FPU's matrix multiply accumulator which operates at higher precision. The `reduce_tile<MAX, REDUCE_ROW>` API is still used for the max reduction, where precision loss is less critical.

### Block Size Constraint
`get_block_size(Wt, 3)` limits block size to 3 maximum. The comment in program_factory.cpp (line 151) says "we need one extra register during calculation." This extra register is needed for:
- The mask register (must be adjacent to data register, so data + mask = 2 registers per tile in worst case)
- The 1/sum broadcast register during normalization (stored at DST[block_size])
- The max accumulator + temp register during max-finding (reg 0 = accum, reg 1 = temp)

### Compute Configuration
- `MathFidelity::HiFi4`: Highest math fidelity, no approximation tradeoffs
- `fp32_dest_acc_en = true`: DST registers use 32-bit accumulation (halves available register count but provides full precision)
- `math_approx_mode = false`: Exact computation, no polynomial approximations
- `exp_tile<false>`: The `false` template parameter means exact exp, not approximate

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do the compute kernel APIs work for reduce operations? Specifically: reduce_init, reduce_tile, reduce_uninit with PoolType::MAX and PoolType::SUM, ReduceDim::REDUCE_ROW. What are the required defines REDUCE_OP and REDUCE_DIM? Also, what does sub_bcast_cols_init_short and sub_tiles_bcast with BroadcastType::COL do?"
   **Reason**: Needed to understand the reduce API's three-phase init/compute/uninit pattern, and how the REDUCE_OP/REDUCE_DIM defines serve as default template parameters for the LLK.
   **Key Findings**: reduce_init configures unpacker and math units; reduce_tile combines unpack+math; reduce_uninit resets packer edge mask. The defines REDUCE_OP and REDUCE_DIM are used as default template parameters when not explicitly specified. sub_bcast_cols_init_short configures subtraction with column broadcast where a single column from SrcB is replicated across all columns.

2. **Query**: "What are unary_bcast_init and unary_bcast with BroadcastType::COL in compute kernels? How does it broadcast a single tile's column values across a tile register? Also explain binary_max_tile_init and binary_max_tile."
   **Reason**: Needed to understand how the max value and 1/sum are broadcast for subtraction and multiplication across full tiles.
   **Key Findings**: unary_bcast<COL> takes a tile from a CB and broadcasts its first column across all columns into a DST register. binary_max_tile computes element-wise maximum between two DST registers. The COL broadcast is used to replicate a scalar column vector across the full tile width.

3. **Query**: "How does matmul_tiles work for row reduction in compute kernels? Specifically when using mm_init and matmul_tiles with a special column vector tile to sum across rows. How does this compare to using reduce_tile for precision?"
   **Reason**: Critical to understand why the implementation uses matmul instead of reduce_tile for sum reduction, and the precision implications.
   **Key Findings**: matmul_tiles multiplies input tile by a column vector (ones in col 0), effectively summing rows. This uses the FPU's matrix multiply accumulator at FP32 precision. reduce_tile can lose precision because it uses a different accumulation path. The REDUCE_ROW_SUM_VIA_MM pattern is a known workaround in the codebase.

4. **Query**: "What does mask_tile do in compute kernels? How does the mask_tile function work - what is the constraint that the mask register must be adjacent to the data register?"
   **Reason**: Needed to understand the masking mechanism for handling padding in the last tile and the hardware constraint on register adjacency.
   **Key Findings**: mask_tile zeros out elements based on a mask pattern. The mask register must be at index `data_register + 1` due to a hardware constraint in the SFPU. The function calls llk_math_eltwise_unary_sfpu_mask internally.

5. **Query**: "How does tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release work in tt-metal compute kernels?"
   **Reason**: Needed to understand the synchronization model between math and pack phases used throughout this kernel.
   **Key Findings**: acquire gives math core ownership of DST registers; commit transfers ownership to packer; wait makes packer wait for math to finish; release makes registers available for next acquire. This is the standard 4-phase handoff between the math core (unpack+math) and the packer.

6. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Needed to understand the core distribution and load balancing strategy.
   **Key Findings**: Returns (num_cores, all_cores, core_group_1, core_group_2, units_per_group_1, units_per_group_2). When work doesn't divide evenly, group_1 gets ceil(total/cores) and group_2 gets floor(total/cores). The remainder determines how many cores are in group_1.

### Documentation References

1. **Source**: `tt-train/sources/ttml/metal/common/program_utils.hpp`
   **Reason**: Understand get_block_size logic and create_circular_buffer/create_compute_kernel helper signatures.
   **Key Information**: get_block_size finds largest divisor of Wt up to max_block_size. create_compute_kernel sets MathFidelity::HiFi4 and passes fp32_dest_acc_en and math_approx_mode=false.

2. **Source**: `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
   **Reason**: Understand constant tile generation and dataflow utility functions.
   **Key Information**: generate_mask_tile fills a 4-face tile with fill_value for valid columns and mask_fill_value for padding columns. generate_matmul_row_reduce_tile creates a tile with 1.0 in first column of even faces. read_full_row_tiles and write_full_row_tiles handle block-based row I/O.

3. **Source**: `tt-train/sources/ttml/metal/ops/softmax/device/softmax_device_operation_types.hpp`
   **Reason**: Understand operation attributes and tensor args.
   **Key Information**: operation_attributes_t has only dim (default 3, last dimension). tensor_args_t contains input tensor and optional preallocated output.
