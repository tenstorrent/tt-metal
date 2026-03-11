# tt-train Softmax Implementation Analysis

## Overview

This document analyzes the tt-train softmax operation, which computes `softmax(x) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along the last dimension (dim=-1) of a 4D tensor. The implementation uses numeric stability (subtracting the row-wise max before exponentiation) and has two code paths: one where all tiles fit in L1 (single-pass over DRAM) and one where they do not (triple-pass over DRAM).

**Source files** (retrieved from git history before nuke commit `cde5a994cb^`):

- **Program factory**: `tt-train/sources/ttml/metal/ops/softmax/device/softmax_program_factory.cpp`
- **Compute kernel**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/compute/softmax_kernel.cpp`
- **Reader kernel**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/reader_softmax_interleaved_start_id.cpp`
- **Writer kernel**: `tt-train/sources/ttml/metal/ops/softmax/device/kernels/dataflow/writer_softmax_interleaved_start_id.cpp`
- **Dataflow utilities**: `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
- **Program utilities**: `tt-train/sources/ttml/metal/common/program_utils.hpp`

**Note**: The current codebase has these files stubbed with `TT_THROW`. All analysis is based on the pre-nuke version.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles |
| **Unit size** | Wt tiles (number of tiles along the innermost dimension) |
| **Total units** | `NC * Ht` where NC = N*C (batch dims), Ht = height in tiles |
| **Loop structure** | Outer loop over rows assigned to this core, inner loops over tile columns in blocks of `block_size` |

One "row" is `Wt` tiles spanning the full innermost dimension. Each row is processed through the complete softmax pipeline (max, subtract-exp, sum, reciprocal, multiply) before moving to the next row.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] |
| **Dimension convention** | NCHW (4D required) | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 (Float16_b) | BFLOAT16 (Float16_b) |

### Layout Transformations
No tilize/untilize or format conversions occur. Input and output share identical specs. The operation only supports dim=-1 (last dimension), which means softmax is computed along the tile-width direction within each row of tiles.

## Softmax Phase Orchestration (Compute Kernel)

The compute kernel processes each row through 4 sequential phases, then a final output phase. There are two compile-time variants controlled by the `EVERYTHING_FITS_IN_L1` define:

### Phase 1: find_max_value_in_row()
**Goal**: Compute the elementwise maximum across all Wt tiles in the row, producing a single tile where each row-position holds the max value across all column-positions.

**API calls used**:
- `copy_tile_init(cb)` / `copy_tile(cb, tile_idx, register_idx)` -- Load tile from CB into DST register
- `mask_tile_init()` / `mask_tile(data_reg, mask_reg)` -- Zero out padding elements (where `col >= num_inner % 32`); mask register MUST be adjacent to data register (data_reg + 1)
- `add_binary_tile_init()` / `add_binary_tile(idst0, idst1, odst)` -- Elementwise add in DST registers (SFPU operation)
- `binary_max_tile_init()` / `binary_max_tile(idst0, idst1, odst)` -- Elementwise max in DST registers (SFPU operation)
- `pack_reconfig_data_format(cb)` / `pack_tile(reg, cb)` -- Pack DST register back to CB

**Register usage**: Two DST registers -- `max_value_register=0` (accumulates running max) and `tile_register=1` (holds current tile). For the last tile when masking is needed, a third register (`working_register + 1`) is temporarily used for the mask.

**Key pattern -- NaN-safe masking**: Before max comparison, the last tile's padding is explicitly handled:
1. The padding is zeroed via `mask_tile` with a 0/1 mask (`cb_mask`)
2. A `-inf` additive mask (`cb_max_mask`) is added to the padding positions, so they become `-inf` and never win the max comparison. This prevents NaN propagation from garbage padding data.

**FITS_IN_L1 variant**: All Wt tiles are in the CB simultaneously. The code uses `cb_wait_front(cb_input, col + block_size)` to progressively wait for tiles, referencing them by absolute index. No `cb_pop_front` during this phase -- data stays in the CB for reuse.

**NOT FITS variant**: Tiles arrive in blocks of `block_size`. Each block is consumed with `cb_pop_front(cb_input, block_size)` after processing. Data must be re-read from DRAM for subsequent phases.

### Phase 2: reduce_max_value()
**Goal**: Reduce the per-element max tile (32x32) down to a column vector (one value per row of the tile) using hardware row reduction.

**API calls used**:
- `reconfig_data_format(icb, icb_scaler)` -- Reconfigure unpacker for new CB pair
- `reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(icb, icb_scaler, ocb)` -- Initialize hardware for MAX row reduction
- `reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(icb, icb_scaler, tile_idx, scaler_idx, dst_reg)` -- Execute reduction
- `reduce_uninit()` -- Reset packer edge mask to default state

**How it works**: `REDUCE_ROW` collapses all columns within each row into a single value. The `cb_reduction_scaler` contains all-1.0 values, so the max is unscaled. After reduction, `cb_max_value_after_reduction` holds a tile where only column 0 of each face row contains the row-max (the packer edge mask zeroes the rest).

### Phase 3: calculate_sum_exp_x()
**Goal**: Compute `sum(exp(x - max(x)))` for the entire row, producing a single tile.

**FITS_IN_L1 variant** (two sub-steps):

Sub-step A -- Compute exp(x - max):
- `sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction)` -- Init broadcast-subtract
- `sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max, tile_idx, 0, reg)` -- Subtract max from each tile. `BroadcastType::COL` means the B operand (max tile) has a filled column-0 that broadcasts across all columns of A. This effectively subtracts the row-max from every element.
- `exp_tile_init<false>()` / `exp_tile<false>(reg)` -- Exact exponentiation (not approximate) in-place in DST register
- Masking of the last tile's padding to zero (after exp, so `exp(garbage)` becomes 0 and doesn't affect sum)
- Result is packed to `cb_exp` (all Wt tiles)

Sub-step B -- Accumulate sum:
- Loads all exp tiles from `cb_exp` back into DST registers and accumulates via `add_binary_tile`
- Packs accumulated sum tile to `cb_exp_sum_before_reduction`

**NOT FITS variant** (single pass, streaming):
- Uses `unary_bcast<BroadcastType::COL>` to broadcast the max value into a DST register once, then reuses it
- `sub_binary_tile(working_reg, max_value_reg, working_reg)` -- Subtract max using SFPU binary op (both operands already in DST)
- `exp_tile<false>(working_reg)` -- Exponentiate in place
- Accumulates the running sum in `accum_register=0` via `add_binary_tile`
- Pops input blocks as they are consumed (data must be re-read for Phase 5)

**Critical difference**: In the FITS_IN_L1 path, intermediate exp results are stored in `cb_exp` for reuse in Phase 5. In the streaming path, exp results are not saved -- Phase 5 must recompute `exp(x - max(x))`.

### Phase 4: reduce_sum_exp_x()
**Goal**: Reduce the sum-of-exp tile to a column vector, then compute its reciprocal.

**Key design decision -- matmul_tiles instead of reduce_tile**: The code explicitly comments that `reduce_tile<PoolType::SUM>` causes precision loss (same issue observed in moreh's ops). Instead, it uses:

- `mm_init(cb_exp_sum_before, cb_mat_mul_reduce, cb_exp_sum_after, 0)` -- Init matmul
- `matmul_tiles(cb_exp_sum_before, cb_mat_mul_reduce, 0, 0, dst_reg)` -- Matrix multiply the sum tile by the row-reduce tile

The `cb_mat_mul_reduce` tile has 1.0 in the first column of even faces and 0 elsewhere (generated by `generate_matmul_row_reduce_tile`). Multiplying `[32x32] * [32x32]` where B has this pattern effectively sums each row into column 0, achieving row reduction with higher precision than the hardware reduce path.

- `recip_tile_init()` / `recip_tile(dst_reg)` -- Compute `1.0 / sum` in-place. After this, the DST register holds the reciprocal of the sum for each row.
- Result packed to `cb_exp_sum_after_reduction`

### Phase 5: Final Output (in kernel_main loop)
**Goal**: Compute `exp(x - max) * (1 / sum)` for each tile and write to output.

- Loads the reciprocal-sum tile via `unary_bcast<BroadcastType::COL>` into DST register `sum_exp_register = block_size`
- Processes tiles in blocks:

**FITS_IN_L1 variant**:
- Reads pre-computed exp tiles from `cb_exp` (no recomputation needed)
- `copy_tile(cb_exp, col + block_idx, block_idx)` -- Load cached exp tile

**NOT FITS variant**:
- Must re-read input tiles from DRAM (3rd read of the same data)
- `sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max_value_after_reduction, ...)` -- Recompute subtract
- `exp_tile<false>(block_idx)` -- Recompute exp

Both variants then:
- `mul_binary_tile(block_idx, sum_exp_register, block_idx)` -- Multiply exp by reciprocal-sum (SFPU binary op, both operands in DST)
- Pack results to `cb_output`

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Data Format | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|-------------------|-------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tiles | Wt (fits) or 2*block_size (streaming) | Float16_b | See note | Reader | Compute | Row (fits) or Block (streaming) |
| c_1 | cb_mask | Zero/one mask for padding | 1 | Float16_b | Single | Reader (generated) | Compute | Program |
| c_2 | cb_max_mask | -inf/zero mask for padding | 1 | Float16_b | Single | Reader (generated) | Compute | Program |
| c_3 | cb_reduction_scaler | All-1.0 tile for reduce_tile scaler | 1 | Float16_b | Single | Reader (generated) | Compute | Program |
| c_4 | cb_mat_mul_reduce | Row-reduce tile for matmul-based sum reduction | 1 | Float16_b | Single | Reader (generated) | Compute | Program |
| c_5 | cb_max_value_before_reduction | Per-element max across row | 2 | Float16_b | Double | Compute | Compute | Block (within row) |
| c_6 | cb_max_value_after_reduction | Row-reduced max (column vector) | 2 | Float16_b | Double | Compute | Compute | Row (persists across Phases 2-5) |
| c_7 | cb_exp | Cached exp(x-max) tiles | Wt (fits) or 2*block_size (streaming) | Float16_b | See note | Compute | Compute | Row (fits only) |
| c_8 | cb_exp_sum_before_reduction | Sum of exp tiles (pre-reduction) | 2 | Float32 | Double | Compute | Compute | Block (within row) |
| c_9 | cb_exp_sum_after_reduction | Reciprocal of reduced sum | 2 | Float32 | Double | Compute | Compute | Block (within row, Phase 4-5) |
| c_10 | cb_output | Output tiles | 2*block_size | Float16_b | Double | Compute | Writer | Block |

### CB Capacity Notes

**c_0 (cb_input)**:
- FITS_IN_L1: capacity = Wt tiles (entire row). The reader loads all tiles once; compute reads them up to 3 times without re-reading from DRAM.
- Streaming: capacity = 2 * block_size tiles. Double-buffered to allow reader to prefetch next block while compute processes current.

**c_7 (cb_exp)**:
- FITS_IN_L1: capacity = Wt tiles. Stores all exp results from Phase 3 for reuse in Phase 5.
- Streaming: capacity = 2 * block_size, but is NOT used as an intermediate cache (exp results are recomputed in Phase 5).

**c_8, c_9 (exp_sum CBs)**: Use Float32 data format for higher precision in the sum accumulation path.

### Multi-Pass Data Reuse Patterns

**FITS_IN_L1 path** (single DRAM read per row):
- `cb_input` (c_0): Persists for the entire row. Read in Phase 1 (find_max) and Phase 3 (calculate_sum_exp). Popped after Phase 5.
- `cb_exp` (c_7): Written in Phase 3, read in Phase 5. Popped after Phase 5.
- `cb_max_value_after_reduction` (c_6): Written in Phase 2, read in Phases 3 and 5 (FITS variant only reads it in Phase 3; streaming reads it also in Phase 5). Popped after Phase 5.

**Streaming path** (triple DRAM read per row):
- The reader sends the same row of tiles 3 times: once for Phase 1 (find_max), once for Phase 3 (sum_exp), once for Phase 5 (final output).
- No intermediate data persists between phases except scalar-like results (max, sum).

### Scalar/Constant CB Setup

All constant tiles are generated by the **reader kernel** at startup (before the row loop):

1. **cb_mask (c_1)**: Generated by `generate_mask_tile(cb_mask_idx, one, zero, mask_w)`. Creates a tile where columns `[0, mask_w)` have value 1.0 (bfloat16 `0x3F80`) and columns `[mask_w, 32)` have value 0.0. Only generated when `DO_MASK_W` is defined (i.e., `mask_w != 0`).

2. **cb_max_mask (c_2)**: Generated by `generate_mask_tile(cb_max_mask_idx, zero, minus_inf, mask_w)`. Creates a tile where columns `[0, mask_w)` have value 0.0 and columns `[mask_w, 32)` have value `-inf` (bfloat16 `0xFF80`). Used to add `-inf` to padding positions before max comparison.

3. **cb_reduction_scaler (c_3)**: Generated by `generate_tile_with_bfloat16_value(cb_reduction_scaler_idx, one)`. All elements are 1.0. Used as the scaler argument to `reduce_tile<PoolType::MAX>`.

4. **cb_mat_mul_reduce (c_4)**: Generated by `generate_matmul_row_reduce_tile(cb_matmul_reduce)`. Has 1.0 in the first column of even faces (faces 0 and 2), 0 elsewhere. When used as the right operand of `matmul_tiles`, it sums each row into column 0. This is the precision-preserving alternative to `reduce_tile<PoolType::SUM>`.

All four constant tiles have **Program lifetime** -- they are pushed once and popped only at the very end of `kernel_main()`.

## Data Flow Pattern

| Stage | Phase | Kernel | Reads From | Writes To | CB Operations |
|-------|-------|--------|------------|-----------|---------------|
| 0 | Setup | Reader | N/A | c_1, c_2, c_3, c_4 | Generate constant tiles: reserve_back, push_back |
| 1 | Per-row input | Reader | DRAM | c_0 | read_full_row_tiles: reserve_back, noc_async_read, push_back |
| 2 | find_max | Compute | c_0, c_1, c_2 | c_5 | wait_front on c_0; pack_tile to c_5; pop_front(c_0) if streaming |
| 3 | reduce_max | Compute | c_5, c_3 | c_6 | wait_front(c_5), reserve_back(c_6), reduce_tile, pop_front(c_5), push_back(c_6) |
| 4a | sum_exp (input re-read) | Reader | DRAM | c_0 | Only if streaming: 2nd read_full_row_tiles |
| 4b | sum_exp | Compute | c_0, c_6 | c_7 (fits) or c_8 | sub_bcast, exp, mask; pack to c_7 or accumulate to c_8 |
| 5 | reduce_sum | Compute | c_8, c_4 | c_9 | matmul_tiles, recip_tile, pack to c_9 |
| 6a | output (input re-read) | Reader | DRAM | c_0 | Only if streaming: 3rd read_full_row_tiles |
| 6b | output | Compute | c_0 or c_7, c_6, c_9 | c_10 | sub_bcast (streaming) or copy from c_7 (fits), mul_binary_tile, pack to c_10 |
| 7 | write output | Writer | c_10 | DRAM | wait_front, noc_async_write, pop_front |

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| c_0 (input) | Wt or 2*BS | BS | Double (streaming) or Full-row (fits) | Reader/Compute overlap in streaming mode |
| c_1 (mask) | 1 | 1 | Single | N/A (constant) |
| c_2 (max_mask) | 1 | 1 | Single | N/A (constant) |
| c_3 (scaler) | 1 | 1 | Single | N/A (constant) |
| c_4 (matmul) | 1 | 1 | Single | N/A (constant) |
| c_5 (max_before) | 2 | 1 | Double | Compute self-overlap |
| c_6 (max_after) | 2 | 1 | Double | Compute self-overlap |
| c_7 (exp) | Wt or 2*BS | BS | Full-row or Double | Compute self-overlap |
| c_8 (sum_before) | 2 | 1 | Double | Compute self-overlap |
| c_9 (sum_after) | 2 | 1 | Double | Compute self-overlap |
| c_10 (output) | 2*BS | BS | Double | Compute/Writer overlap |

## Index Calculations

### Row-to-Tile Index Mapping

The tensor is viewed as a flattened sequence of tiles in row-major order:
```
tile_index = row * Wt + col
```
Where:
- `row = (start_row + i)` for the i-th row assigned to this core
- `col` ranges from 0 to Wt-1
- `start_row` is the cumulative number of rows assigned to previous cores

`Wt = padded_shape[-1] / TILE_WIDTH` -- tiles along the innermost dimension.
`total_rows = NC * Ht` where `NC = shape[0] * shape[1]`, `Ht = padded_shape[-2] / TILE_HEIGHT`.

### Block Size Calculation

`block_size = get_block_size(Wt, 3)` -- Finds the largest value in [3, 2, 1] that evenly divides Wt. The maximum is 3 (not the typical 4) because the compute kernel needs "one extra register during calculation" for operations like masking and max comparison.

### Mask Width

`mask_w = num_inner % TILE_WIDTH` -- If the logical width is not a multiple of 32, `mask_w` indicates where valid data ends within the last tile. When `mask_w == 0`, no masking is needed (the `DO_MASK_W` define is omitted).

## Memory Access Patterns

### Read Pattern
- **DRAM reads**: Sequential tile reads along rows via TensorAccessor. Each row is read as `Wt / block_size` blocks.
- **FITS_IN_L1**: One DRAM read per row per program execution.
- **Streaming**: Three DRAM reads per row (once for each of the three phases that need input data).
- **Access order**: Row-major within each row, rows processed in core-assigned order.

### Write Pattern
- **DRAM writes**: Sequential tile writes along rows via TensorAccessor. Output tiles are written in blocks of `block_size`.
- **Access order**: Same as read order -- row-major within each row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (column-major traversal of 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores()` |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` rows |
| **Load balancing** | Two groups: group_1 gets `ceil(total_rows/num_cores)` rows, group_2 gets `floor(total_rows/num_cores)` rows |

Core indexing: `core = {i / num_cores_y, i % num_cores_y}` -- column-major traversal (x = i / num_cores_y, y = i % num_cores_y).

Two compute kernel instances are created (group_1 and group_2) with different `num_rows_per_core` compile-time args, since `split_work_to_cores` may assign one extra row to some cores to handle remainders.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Number of tiles processed per inner-loop iteration |
| 1 | Wt | uint32_t | Number of tiles along the innermost dimension |
| 2 | mask_w | uint32_t | Width of valid data in the last tile (0 means no masking) |
| 3+ | TensorAccessor args | uint32_t[] | Input buffer tensor accessor parameters |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Number of tiles per block |
| 1 | Wt | uint32_t | Number of tiles along the innermost dimension |
| 2+ | TensorAccessor args | uint32_t[] | Output buffer tensor accessor parameters |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Number of rows this core processes (differs between group_1 and group_2) |
| 1 | block_size | uint32_t | Number of tiles per block |
| 2 | Wt | uint32_t | Number of tiles along the innermost dimension |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_address | uint32_t | DRAM address of the input buffer |
| 1 | num_rows_to_process | uint32_t | Number of rows for this core |
| 2 | start_row | uint32_t | Row offset (cumulative rows from prior cores) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_address | uint32_t | DRAM address of the output buffer |
| 1 | num_rows_to_process | uint32_t | Number of rows for this core |
| 2 | start_row | uint32_t | Row offset (cumulative rows from prior cores) |

### Preprocessor Defines

| Define | Condition | Effect |
|--------|-----------|--------|
| `DO_MASK_W` | `mask_w != 0` | Enables padding masking logic in compute and reader kernels |
| `EVERYTHING_FITS_IN_L1` | `required_L1 <= available_L1` | Selects single-pass (L1 cached) vs triple-pass (streaming) variant |
| `REDUCE_OP` | Always `PoolType::SUM` | Required by LLK reduce API as default template parameter |
| `REDUCE_DIM` | Always `ReduceDim::REDUCE_ROW` | Required by LLK reduce API as default template parameter |

### Compute Config

| Setting | Value |
|---------|-------|
| `math_fidelity` | `MathFidelity::HiFi4` |
| `fp32_dest_acc_en` | `true` |
| `math_approx_mode` | `false` |

`fp32_dest_acc_en=true` enables FP32 accumulation in DST registers, critical for precision in the sum-of-exp computation and the matmul-based reduction.

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input) | c_0, c_1, c_2, c_3, c_4 | Generate constant tiles; read input tiles (1x or 3x per row) |
| compute | RISCV_2 | N/A | c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9 | c_5, c_6, c_7, c_8, c_9, c_10 | Full softmax pipeline: max, reduce, exp, sum, recip, mul |
| writer | RISCV_1 | NOC1 | c_10 | DRAM (output) | Write output tiles by blocks |

### Compute Kernel -- Key Logic

**Initialization** (`kernel_main` entry):
```cpp
init_sfpu(cb_input, cb_output);
binary_op_init_common(cb_input, cb_input, cb_output);
```
These calls configure the SFPU and binary op hardware for the data formats of input/output CBs.

**Register pressure management**: The `block_size` is limited to max 3 (not 4) to leave room for auxiliary registers needed by masking, broadcast, and reduction operations. In Phase 5, `sum_exp_register = block_size` is used to hold the broadcast reciprocal-sum while registers 0..block_size-1 hold the current block's tiles.

**tile_regs_acquire/commit/wait/release pattern**: The standard 4-phase DST register lifecycle is used throughout:
1. `tile_regs_acquire()` -- Lock DST registers for math thread
2. (Perform computations -- copy, math ops, SFPU ops)
3. `tile_regs_commit()` -- Signal math is done
4. `tile_regs_wait()` -- Wait for pack thread to be ready
5. `pack_tile(...)` -- Pack results to CB
6. `tile_regs_release()` -- Release DST registers

**reconfig_data_format usage**: Called frequently before operations that change the CB pair being unpacked. This reconfigures the unpacker hardware for the data formats of the new source CBs.

### Reader Kernel -- Summary

The reader generates all constant tiles at startup, then enters a per-row loop. In FITS_IN_L1 mode, it reads the row once via `read_full_row_tiles`. In streaming mode, it reads the same row 3 times (the reader has no knowledge of phases; it just pushes 3 * Wt tiles per row into `cb_input` and the compute kernel consumes them in order).

### Writer Kernel -- Summary

The writer enters a per-row loop calling `write_full_row_tiles` which waits for blocks of output tiles in `cb_output`, writes them to DRAM via TensorAccessor, and pops them.

## Implementation Notes

### Precision Considerations
1. **FP32 accumulation**: `fp32_dest_acc_en=true` ensures DST registers use 32-bit precision, preventing accumulation errors in the exp-sum.
2. **Float32 CBs for sum**: `cb_exp_sum_before_reduction` (c_8) and `cb_exp_sum_after_reduction` (c_9) use `DataFormat::Float32` and `float32_single_tile_size_bytes`, maintaining full precision through the sum-reduce-reciprocal pipeline.
3. **matmul_tiles instead of reduce_tile for SUM**: The code explicitly avoids `reduce_tile<PoolType::SUM>` due to observed precision loss. Instead, it uses a matmul with a specially constructed tile to achieve row reduction. The `reduce_tile<PoolType::MAX>` for max reduction is fine because MAX is less precision-sensitive.
4. **Exact exp (not approximate)**: `exp_tile<false>` uses the exact implementation, not the faster approximate version.

### NaN-Safe Padding Handling
The two-mask approach (zero-mask + minus-inf mask) handles multiple NaN-producing edge cases in padding:
- `NaN + (-inf)` would produce NaN if not masked first
- `-inf * 0` would produce NaN if not masked first
- The code first zeros the padding (via `mask_tile` with the 0/1 mask), then adds `-inf` to the padding positions (via `add_binary_tile` with the 0/-inf mask). This ensures padding positions have clean `-inf` values.

### L1 Memory Budget Calculation
The program factory calculates whether everything fits in L1:
```
required = 2 * (Wt * bf16_tile_size)     // input + exp CBs
         + 2 * mask_tile_size              // mask + max_mask
         + 2 * scaler_tile_size            // scaler + matmul
         + 2*block_size * bf16_tile_size   // output
         + 3 * bf16_tile_size              // max_before(2) + max_after(1) -- approximation
         + 4 * fp32_tile_size              // sum_before(2) + sum_after(2)
```
If `required <= available_L1`, the FITS_IN_L1 define is set, enabling single-pass execution.

### Limitations
- Only supports `dim=-1` (last dimension softmax)
- Only supports `Float16_b` (BFLOAT16) data type
- Only supports INTERLEAVED memory layout on DRAM
- Must be 4D tensor
- Does not support optional attention mask or scale factor

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How do reduce_init, reduce_tile, and reduce_uninit work in the compute kernel API?"
   **Reason**: Needed to understand the reduction pipeline used in Phase 2 (reduce_max_value).
   **Key Findings**: `reduce_init` configures unpack/math/pack for reduction. `reduce_tile` performs the actual reduction. `reduce_uninit` resets packer edge masks. The `icb_scaler` should be all-1.0 for MAX/SUM ops. Templates `PoolType` (SUM/AVG/MAX) and `ReduceDim` (REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR) control behavior.

### Documentation References
1. **Source**: `tt_metal/hw/inc/api/compute/bcast.h`
   **Reason**: Understanding broadcast subtraction and column broadcast semantics.
   **Key Information**: `BroadcastType::COL` means B has a filled column-0 that broadcasts across all columns. `sub_tiles_bcast<COL>` computes `A[h,w] - B[w]`. `unary_bcast<COL>` performs a datacopy with column broadcast from CB to DST register. `sub_bcast_cols_init_short` is the lightweight init for switching to broadcast-subtract.

2. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Understanding reduce_tile API and its limitations (precision).
   **Key Information**: `reduce_tile` requires init/uninit lifecycle. The scaler CB must have scaling factors in the first row of each face. For SUM, scaler should be 1.0. The API documentation notes the packer edge mask must be cleared via `reduce_uninit` before other operations.

3. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: Understanding the `binary_max_tile` SFPU operation used for finding max.
   **Key Information**: `binary_max_tile(idst0, idst1, odst)` performs elementwise max on two DST registers. Operates on float/bfloat16 data. All three arguments are DST register indices (data must already be in registers).

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Understanding `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile` used throughout.
   **Key Information**: These are SFPU operations on DST registers: `op(idst0, idst1, odst)`. They require both operands already loaded into DST registers (unlike FPU ops that work directly on CBs). Each requires a corresponding `_init()` call before first use.

5. **Source**: `tt_metal/hw/inc/api/compute/mask.h`
   **Reason**: Understanding the `mask_tile` function and its constraint on register adjacency.
   **Key Information**: `mask_tile(idst_data, idst2_mask)` performs elementwise masking. **Critical constraint**: Despite taking two separate register arguments, the implementation currently uses `(idst_data + 1)` as the mask register (the `idst2_mask` parameter is effectively ignored). This is why the compute kernel always places the mask in `working_register + 1`.

6. **Source**: `METALIUM_GUIDE.md` (Register control and Data Flow section)
   **Reason**: Understanding DST register lifecycle (acquire/commit/wait/release).
   **Key Information**: DST registers are shared between math and pack units. The 4-phase lifecycle (acquire -> compute -> commit -> wait -> pack -> release) ensures safe concurrent access. SFPU operations require explicit `copy_tile` to load data into DST, while FPU operations like `add_tiles` can work directly on CBs.

7. **Source**: `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
   **Reason**: Understanding constant tile generation and the matmul row-reduce tile pattern.
   **Key Information**: `generate_matmul_row_reduce_tile` creates a tile with 1.0 in the first column of even faces (left faces), 0 elsewhere. This tile, when right-multiplied via `matmul_tiles`, sums each row of the left operand into a single column. This is the precision-preserving alternative to `reduce_tile<PoolType::SUM>`.

8. **Source**: `tt-train/sources/ttml/metal/common/program_utils.hpp`
   **Reason**: Understanding `get_block_size`, `create_circular_buffer`, and kernel creation helpers.
   **Key Information**: `get_block_size(Wt, max=3)` finds the largest divisor of Wt in [1..max]. `create_circular_buffer` is a convenience wrapper around `CircularBufferConfig`. Compute kernels are created with `MathFidelity::HiFi4` and configurable `fp32_dest_acc_en`.
