# moreh_sum_h Implementation Analysis

## Overview

**moreh_sum_h** performs a summation reduction along the H (height) dimension of a tiled tensor. Given an input tensor of shape `[..., H, W]` in tile layout, it produces an output tensor where the H dimension is collapsed to a single tile row (32 elements), yielding shape `[..., 1, W]` in tile units. Each output tile is the sum of all tiles in its corresponding column across the H dimension.

**Program factory path**: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_program_factory.cpp`

**Focus of this analysis**: Compute kernel structure, CB layout for intermediates, multi-pass data reuse patterns, scalar/constant CB setup, reduce helper parameters, and binary op broadcast patterns -- all oriented toward informing a softmax operation design.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Column of tiles (Ht tiles tall, 1 tile wide) |
| **Unit size** | Ht tiles input, 1 tile output |
| **Total units** | `num_cols = other_dims_product * Wt` |
| **Loop structure** | Outer: NC (batch, always 1 as host folds batches into num_cols), Inner: Wt (tile columns per core) |

One "work unit" is a single column of `Ht` tiles in the H dimension. Each column reduces to exactly 1 output tile. The total work across all cores is `other_dims_product * Wt` such columns.

---

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [..., H, W] (rank >= 2) | [..., 1, W] (H reduced to 1 tile row) |
| **Dimension convention** | Last two dims are H, W | Same, with H = TILE_HEIGHT |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | Configurable (bfloat16, float32, etc.) | Configurable (may differ from input) |

### Layout Transformations
- No explicit tilize/untilize within this operation
- The operation assumes tile layout on input and produces tile layout output
- The H dimension is reduced within the compute kernel using hardware reduce instructions
- When `origin_H` is not a multiple of 32, a **mask tile** is applied to zero out padding rows in the last H tile before reduction

---

## Data Flow Pattern

### High-Level Flow

```
DRAM Input --> [Reader] --> CB_input (c_0) --> [Compute] --> CB_output (c_16) --> [Writer] --> DRAM Output
                  |                                 |
                  +--> CB_scaler (c_2)    CB_accum_dst (c_24) [intermediate accumulator]
                  +--> CB_mask_h (c_3)    CB_masked_input (c_25) [masked last tile]
```

### Step-by-Step

1. **Reader** generates a scaler tile (value 1.0 for SUM) into `CB_scaler` (c_2), and optionally a mask tile into `CB_mask_h` (c_3)
2. **Reader** reads tiles from DRAM in **column-major order within each batch**: for each assigned column, it reads all Ht tiles in the H dimension sequentially, pushing each to `CB_input` (c_0)
3. **Compute** waits on scaler (once, persists for entire kernel) and mask_h (once, persists)
4. For each column:
   - **Phase 1**: If Ht > 1, reduces first (Ht-1) tiles via `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>()` into intermediate accumulator `CB_accum_dst` (c_24)
   - **Phase 2 (masked path)**: If last H tile needs masking, copy last input tile to DST, mask it, pack to `CB_masked_input` (c_25), then reduce that single masked tile into `CB_output` (c_16) with accumulation from c_24
   - **Phase 2 (unmasked path)**: Reduce last input tile directly into `CB_output` (c_16) with accumulation from c_24
   - **Single-tile fast path**: If Ht == 1, skip Phase 1 entirely and perform Phase 2 without accumulation reload (iteration=0)
5. **Writer** reads tiles from `CB_output` (c_16) one at a time and writes to DRAM

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging from DRAM | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Reduce scaler constant (1.0 for SUM) | 1 tile | 1 tile | Single | Reader (once) | Compute (persistent read) | Program (entire kernel) |
| c_3 | cb_mask_h | H-dimension mask tile (1s/0s) | 1 tile | 1 tile | Single | Reader (once) | Compute (persistent read) | Program (entire kernel) |
| c_16 | cb_out | Final output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per column) |
| c_24 | cb_accum_dst | Intermediate accumulator for partial reduce | 1 tile | 1 tile | Single | Compute (Phase 1 output) | Compute (Phase 2 reload) | Block (per column, reset each column) |
| c_25 | cb_masked_input | Masked version of last H tile | 1 tile | 1 tile | Single | Compute (mask step) | Compute (Phase 2 reduce input) | Block (per column, only if do_mask_h) |

### Data Format Details
- **c_0 (input)**: matches input tensor dtype (e.g., bfloat16)
- **c_2 (scaler)**: `Float16_b` always (NOTE: tile size computed using src0 format, which may be a minor inconsistency in the code; functionally the scaler generation in the reader uses `Float16_b` semantics via `calculate_and_prepare_reduce_scaler`)
- **c_3 (mask_h)**: `Float16_b` (binary mask: 1.0 or 0.0 per element)
- **c_24 (accum_dst)**: `Float32` if `fp32_dest_acc_en`, else `Float16_b` -- matches accumulation precision
- **c_25 (masked_input)**: `Float16_b` always (the packed masked tile before final reduce)
- **c_16 (output)**: matches output tensor dtype

### Multi-Pass Data Reuse Patterns

**CBs that persist across phases (and why):**

1. **cb_scaler (c_2)**: Pushed once by reader via `calculate_and_prepare_reduce_scaler`, then `cb_wait_front(cb_scaler, 1)` is called once at the start of compute. The scaler tile is never popped until the very end of the kernel (`cb_pop_front(cb_scaler, onetile)` at line 99). The reduce helper internally also calls `cb_wait_front(scaler_cb, 1)` but since the tile is already present, this is a no-op wait. This means **one scaler tile serves all columns across the entire kernel execution**.

2. **cb_mask_h (c_3)**: Similarly generated once by reader via `generate_mask_h()`, waited on once at compute start, and popped only at kernel end (line 97). The mask tile is reused across all columns -- the compute kernel reads it via `copy_tile(cb_mask_h, 0, mask_dst_idx)` without popping, using tile index 0 repeatedly.

3. **cb_accum_dst (c_24)**: This is the per-column intermediate accumulator. It is produced by Phase 1 (`reduce` outputs to it) and consumed by Phase 2 (`Accumulate::at(cb_accum_dst, ...)` reloads it). It does **not** persist across columns -- each column's Phase 1 overwrites it, and Phase 2 consumes it. This is a **block-scoped** intermediate.

4. **cb_masked_input (c_25)**: Produced and consumed within a single column's Phase 2 masked path. Does not persist across columns.

---

## Compute Kernel Deep Dive

### File
`ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp`

### Includes and Dependencies
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"  // Main reduce library
#include "ttnn/kernel/compute/moreh_common.hpp"                  // Utility helpers (copy_tile, mask_tile, etc.)
```

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tiles in H dimension (height tile count) |
| 1 | Wt | uint32_t | Number of tile-columns assigned to this core |
| 2 | NC | uint32_t | Number of batch iterations (always 1; batches folded into Wt) |
| 3 | origin_H | uint32_t | Original (unpadded) H dimension in elements -- used to compute mask |

### Preprocessor Defines (Set by Program Factory)

| Define | Value (for moreh_sum_h) | Purpose |
|--------|------------------------|---------|
| `REDUCE_OP` | `PoolType::SUM` | Selects summation as the reduce operation |
| `REDUCE_DIM` | `ReduceDim::REDUCE_COL` | H-reduction maps to column reduction at LLK level |
| `FP32_DEST_ACC_EN` | `"1"` (conditional) | Enables FP32 accumulation in destination registers |

**Key mapping**: `ReduceOpDim::H` at the TTNN API level maps to `ReduceDim::REDUCE_COL` at the LLK/hardware level. This is because at the tile level, reducing along H means collapsing rows within each column, which the hardware calls "column reduction." The result lands in the first row of the destination tile.

### Initialization Call

```cpp
binary_op_init_common(cb_input, cb_input, cb_out);
```

This initializes the unpacker, math, and packer hardware pipelines. Parameters are `(SRCA_CB, SRCB_CB, output_CB)`. Using `cb_input` for both SRCA and SRCB is a common pattern when the actual operation (reduce) will re-initialize its own unpack configuration -- this just sets up the packer for `cb_out`.

### Compute Kernel Structure

The compute kernel processes `NC * Wt` columns. For each column:

#### Phase 1: Bulk Reduce (Ht-1 tiles)

```cpp
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_input,      // input tiles (Ht-1 will be consumed)
    cb_scaler,     // scaler tile (persistent, index 0)
    cb_accum_dst,  // output: intermediate accumulator
    compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));
```

**Signature breakdown**:
- Template params: `<PoolType::SUM, ReduceDim::REDUCE_COL>` (from defines)
- Default template params used: `input_policy = WaitAndPopPerTile`, `reconfig_mode = INPUT_AND_OUTPUT`
- `ReduceInputBlockShape::col(Ht - 1)` = `{rows: Ht-1, cols: 1, batches: 1}`
- No accumulation parameter (default `NoAccumulation{}`)
- No post-reduce op (default `NoOp{}`)

**What this does internally** (from `reduce_helpers_compute.inl`, REDUCE_COL path):
- Since `input_policy = WaitAndPopPerTile` and `Wt = 1` (single column in the block shape):
  - `chunk_size = DEST_AUTO_LIMIT` (4-16 depending on sync/accum mode)
  - But since Wt=1, only 1 DST register is used per chunk
  - For each of the (Ht-1) height tiles: `cb_wait_front(cb_input, 1)` -> `reduce_tile(...)` -> `cb_pop_front(cb_input, 1)`
  - All (Ht-1) tiles accumulate into DST[0]
- Result is packed to `cb_accum_dst`

#### Phase 2a: Masked Last Tile Path (when `do_mask_h == true`)

This path handles the case where `origin_H % 32 != 0`, meaning the last tile in H has padding rows that must be zeroed before reduction.

```cpp
// Step 1: Copy input tile to DST, apply mask
tile_regs_acquire();
cb_wait_front(cb_input, onetile);
reconfig_data_format_srca(cb_input);              // Only if FP32_DEST_ACC_EN
copy_tile_to_dst_init_short(cb_input);
copy_tile(cb_input, 0, reduce_dst_idx);           // Input -> DST[0]
copy_tile(cb_mask_h, 0, mask_dst_idx);            // Mask -> DST[1]
mask_tile_init();
mask_tile(reduce_dst_idx, mask_dst_idx);           // DST[0] *= DST[1] (element-wise)
tile_regs_commit();

// Step 2: Pack masked tile to intermediate CB
cb_reserve_back(cb_masked_input, onetile);
tile_regs_wait();
pack_reconfig_data_format(cb_masked_input);        // Only if FP32_DEST_ACC_EN
pack_tile(reduce_dst_idx, cb_masked_input);        // DST[0] -> cb_masked_input
tile_regs_release();
cb_push_back(cb_masked_input, onetile);
cb_pop_front(cb_input, onetile);

// Step 3: Reduce masked tile with accumulation
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_masked_input,  // input: the masked tile
    cb_scaler,        // scaler
    cb_out,           // output: final result
    compute_kernel_lib::ReduceInputBlockShape::single(),               // {1, 1, 1}
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),         // default
    compute_kernel_lib::Accumulate::at(cb_accum_dst, is_h_single_tile ? 0 : 1));
```

**Accumulate parameter details**:
- `Accumulate::at(cb_accum_dst, iteration)` where:
  - `cb_accum_dst = c_24` -- the CB holding the partial sum from Phase 1
  - `iteration = is_h_single_tile ? 0 : 1`
    - If `Ht == 1`: iteration=0, meaning this is the first (and only) reduce call, so **no accumulator reload** -- DST starts from zero
    - If `Ht > 1`: iteration=1, meaning **reload the accumulator** from c_24 into DST before reducing, so the final result includes both the Phase 1 partial sum and this last tile

**Inside reduce() with Accumulate**:
1. `tile_regs_acquire()`
2. `reload_accumulator_if_needed()` -- if iteration > 0: `cb_wait_front(c_24, 1)`, `copy_tile_to_dst_init_short_with_dt(input_cb, c_24)`, `copy_tile(c_24, 0, dst_idx)`, `cb_pop_front(c_24, 1)`, then `reduce_init_short_with_dt(c_24, input_cb, scaler_cb)` to re-init reduce after copy corrupted SRCA config
3. `reduce_tile(cb_masked_input, cb_scaler, 0, 0, dst_idx)` -- reduces the single masked tile, accumulating onto the reloaded value
4. Pack result to `cb_out`

#### Phase 2b: Unmasked Last Tile Path (when `do_mask_h == false`)

```cpp
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_input,     // input: last tile directly from reader
    cb_scaler,
    cb_out,       // output: final result
    compute_kernel_lib::ReduceInputBlockShape::single(),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::Accumulate::at(cb_accum_dst, is_h_single_tile ? 0 : 1));
```

Identical logic to Phase 2a but reading directly from `cb_input` instead of `cb_masked_input`.

### DST Register Usage

| DST Index | Purpose |
|-----------|---------|
| 0 (`reduce_dst_idx`) | Main reduction accumulation target; also used for copy+mask of last tile |
| 1 (`mask_dst_idx`) | Holds the mask tile during the mask operation |

The compute kernel uses at most 2 DST registers simultaneously (during the mask step). The reduce library itself uses DST[0] for accumulation.

### Scalar/Constant CB Setup

The **scaler tile** is generated in the reader kernel using:
```cpp
dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
    cb_id_in2,                        // CB c_2
    ckernel::PoolType::SUM,           // For SUM: scaler = 1.0
    ckernel::ReduceDim::REDUCE_COL    // H-reduction
>();
```

For `PoolType::SUM`, the scaler is always 1.0f. The helper:
1. Calls `cb_reserve_back(cb_id, 1)`
2. Zeros the entire tile via `zero_faces()`
3. Fills row 0 of each face with the bfloat16-packed scaler value (1.0)
4. Calls `cb_push_back(cb_id, 1)`

The scaler tile layout is: only row 0 of each face contains the scaler value; all other rows are zero. This is the format expected by the `reduce_tile` LLK instruction.

### FP32 Accumulation Handling

When `fp32_dest_acc_en` is true:
- `cb_accum_dst` (c_24) uses `Float32` data format (8 bytes per element vs 2 bytes for bfloat16)
- `UnpackToDestMode::UnpackToDestFp32` is set for c_24 specifically
- In the compute kernel, `#if defined FP32_DEST_ACC_EN` guards trigger `reconfig_data_format_srca()` and `pack_reconfig_data_format()` calls around format transitions
- The reduce library auto-detects FP32 mode via `get_fp32_dest_acc_enabled()` and passes it as a template parameter to `reduce_init`, `reduce_tile`, and `reduce_uninit`

---

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Buffering Type |
|----|----------|------------|-------|----------------|
| c_0 (input) | 2 tiles | 1 tile | 2:1 | Double-buffered |
| c_2 (scaler) | 1 tile | 1 tile | 1:1 | Single (persistent) |
| c_3 (mask_h) | 1 tile | 1 tile | 1:1 | Single (persistent) |
| c_16 (output) | 2 tiles | 1 tile | 2:1 | Double-buffered |
| c_24 (accum_dst) | 1 tile | 1 tile | 1:1 | Single (intermediate) |
| c_25 (masked_input) | 1 tile | 1 tile | 1:1 | Single (intermediate) |

- **c_0 and c_16** are double-buffered, allowing overlap between reader/compute and compute/writer respectively
- **c_2 and c_3** are single-buffered but effectively persistent -- written once, read many times
- **c_24 and c_25** are single-buffered intermediates with block lifetime

---

## Index Calculations

### Reader: Column-Major Tile Traversal

The reader walks tiles in **NWH order** (columns first, then rows within each column):

```
For each assigned column i (0..num_cols-1):
    curr_id = col_start_tile_id   // Starting tile ID for this column
    For each row j (0..Ht-1):
        Read tile at curr_id
        curr_id += Wt             // Stride by Wt to get next H-tile in same column

    Advance col_start_tile_id:
        If at end of row (w == Wt): jump to start of next batch row
        Else: move to next column (col_start_tile_id++)
```

The initial `col_start_tile_id` is computed as: `(num_cols_read / Wt * HtWt) + (num_cols_read % Wt)`, which maps the linear column index to the tile ID accounting for batch boundaries.

### Compute: Sequential Column Processing

The compute kernel processes columns sequentially. For each column, it processes Ht tiles in the order they arrive from the reader (bottom of H stack first, top last when masked).

### Writer: Linear Output Tile Write

The writer simply writes `num_cols_per_core` tiles starting from `start_id = num_cols_read`, using TensorAccessor for address translation.

---

## Memory Access Patterns

### Read Pattern
- **Sequential within column**: Each column's Ht tiles are read consecutively with stride `Wt * tile_size` in the logical tile space (adjacent in H, same W position)
- **Column-to-column**: Adjacent columns in the same batch are accessed with stride 1 tile in the W dimension
- **DRAM access**: Via TensorAccessor with `noc_async_read_tile`, one tile at a time with barrier after each read
- **Pattern**: Strided (stride = Wt tiles between consecutive H reads), sequential across columns

### Write Pattern
- **Sequential**: Output tiles are written in linear order `[start_id, start_id + num_tiles)` via `noc_async_write_tile` with barrier after each
- **Pattern**: Sequential, one tile at a time

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores` (determined by `split_work_to_cores_wt_core_range`) |
| **Work per core** | `num_cols_per_core_group_1` or `num_cols_per_core_group_2` columns |
| **Load balancing** | Two-group split: group_1 gets `ceil(num_cols/num_cores)` columns, group_2 gets `floor(num_cols/num_cores)` |

The `split_work_to_cores_wt_core_range` function divides `num_cols = other_dims_product * Wt` columns across available cores. Core indexing is column-major within the grid: `core = {i / num_cores_y, i % num_cores_y}`.

Two separate compute kernel instances are created:
- **core_group_1**: Gets `num_cols_per_core_group_1` columns (the larger share)
- **core_group_2**: Gets `num_cols_per_core_group_2` columns (the smaller share, may be empty)

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Tiles in H dimension |
| 1 | Wt | uint32_t | Tiles in W dimension (global, not per-core) |
| 2 | HtWt | uint32_t | Total tiles in one HxW batch (Ht * Wt) |
| 3+ | TensorAccessorArgs | varied | Tensor accessor metadata for input buffer |

**Defines**: `REDUCE_SCALER=1` (always), `DO_MASK_H=1` (conditional on padding)

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Tiles in H dimension |
| 1 | Wt | uint32_t | Tile-columns assigned to this core (num_cols_per_core_group_N) |
| 2 | NC | uint32_t | Batch count (always 1; batches folded into Wt) |
| 3 | origin_H | uint32_t | Unpadded H dimension in elements (constexpr, used for mask check) |

**Defines**: `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_COL`, optionally `FP32_DEST_ACC_EN=1`

**ComputeConfig extras**: `unpack_to_dest_mode[c_24] = UnpackToDestFp32` when fp32 accumulation is enabled

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_16) |
| 1+ | TensorAccessorArgs | varied | Tensor accessor metadata for output buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address in DRAM |
| 1 | col_start_tile_id | uint32_t | Starting tile ID for this core's first column |
| 2 | curr_col_in_batch | uint32_t | Starting W-position within current batch |
| 3 | num_cols_per_core | uint32_t | Number of columns this core processes |
| 4 | mask_h | uint32_t | Number of valid rows in last H tile (1-32) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_cols_per_core | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Output tile start index (num_cols_read at assignment time) |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_moreh_sum_h | RISCV_0 | NOC0 | DRAM (input tensor) | CB c_0, c_2, c_3 | Read input tiles, generate scaler, generate mask |
| moreh_sum_h (compute) | RISCV_2 (unpack/math/pack) | N/A | CB c_0, c_2, c_3 | CB c_16, c_24, c_25 | reduce_tile, copy_tile, mask_tile, pack_tile |
| writer_moreh_sum_h | RISCV_1 | NOC1 | CB c_16 | DRAM (output tensor) | Write output tiles |

### Reader: `reader_moreh_sum_h.cpp`
- **Provides**: Input tiles to c_0 (Ht tiles per column, num_cols columns), scaler tile to c_2, mask tile to c_3
- **Consumes**: DRAM input tensor data
- **Key detail**: Scaler and mask are generated once at start, then tile streaming begins

### Compute: `moreh_sum_h.cpp`
- **Key Logic**: Two-phase reduction with optional masking, detailed extensively in the "Compute Kernel Deep Dive" section above
- **Critical observation for softmax adaptation**: The reduce library's `Accumulate` mechanism enables multi-pass reduction where Phase 1 produces a partial result to an intermediate CB, and Phase 2 reloads it before the final reduction step. This same pattern could be used for softmax's multi-phase operations (max-reduce, exp, sum-reduce).

### Writer: `writer_moreh_sum_h.cpp`
- **Provides**: Nothing (terminal)
- **Consumes**: Output tiles from c_16, writes them to DRAM
- **Key detail**: Simple sequential write using TensorAccessor

---

## Implementation Notes

### Design Patterns Relevant to Softmax

1. **Persistent scalar/constant CBs**: The scaler (c_2) and mask (c_3) tiles are written once by the reader and read many times by compute without popping. This pattern is directly applicable to softmax's scaler tile (for `1/sum(exp(x))` broadcast) and any mask tiles.

2. **Two-phase reduce with intermediate CB**: The `cb_accum_dst` (c_24) pattern -- where Phase 1 reduces (Ht-1) tiles into an intermediate, and Phase 2 reloads + reduces the last tile -- is the canonical way to handle partial reductions when the last tile needs special treatment (masking). For softmax, this same pattern could handle the max-reduction phase.

3. **Accumulate::at() iteration control**: The iteration parameter (0 = first call/no reload, 1+ = reload from accumulator) provides clean control over whether the reduce starts fresh or continues from a partial result. Softmax could use this for block-wise sum(exp) accumulation.

4. **mask_tile for padding**: The `copy_tile` + `mask_tile` sequence in DST registers is the standard way to zero out padding elements. For softmax, this prevents padding values from affecting max or sum computations.

5. **Data format reconfiguration**: The `FP32_DEST_ACC_EN` guards around `reconfig_data_format_srca` and `pack_reconfig_data_format` are essential when CBs have different data formats (e.g., c_24 is Float32 while c_0 is bfloat16). Softmax will need similar reconfig when switching between operations.

6. **binary_op_init_common as generic initializer**: Even though this is a reduce operation, `binary_op_init_common(cb_input, cb_input, cb_out)` is used for initial pipeline setup. The reduce library then re-initializes its own specific configuration via `reduce_init`.

### Reduce Helper Library Signatures (Complete Reference)

```cpp
// Main reduce function
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

// Block shape factories
ReduceInputBlockShape::of(rows, cols, batches)  // General
ReduceInputBlockShape::single()                  // {1, 1, 1}
ReduceInputBlockShape::row(cols, batches)        // {1, cols, batches}
ReduceInputBlockShape::col(rows, batches)        // {rows, 1, batches}

// Accumulation
Accumulate::at(cb, iteration, dst=0)  // iteration 0 = no reload

// Post-reduce op example (for softmax reciprocal):
[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }

// Input policies:
// WaitAndPopPerTile  - streaming, 1 tile at a time (default)
// BulkWaitBulkPop    - wait for bulk, process, pop bulk
// WaitUpfrontNoPop   - wait all upfront, tiles persist for reuse
// NoWaitNoPop        - caller manages CB wait/pop externally
```

### Softmax-Specific Observations

For a softmax `exp(x_i - max) / sum(exp(x_j - max))` along H:

- **Max reduction**: Use `reduce<MAX, REDUCE_COL>` with the same column-major tile ordering
- **Subtract max + exp**: Per-tile elementwise ops using `sub_tiles_bcast` (broadcast the max tile, which is a row vector, across columns) + `exp_tile`
- **Sum reduction**: Use `reduce<SUM, REDUCE_COL>` on the exp results
- **Reciprocal of sum**: Use `post_reduce_op` with `recip_tile` during the sum reduce, or as a separate step
- **Multiply by reciprocal**: Per-tile `mul_tiles_bcast` to broadcast the reciprocal sum across the column

The `WaitUpfrontNoPop` input policy is explicitly designed for softmax patterns where input tiles need to be reused (e.g., reading tiles for max, then reusing them for exp without re-reading from DRAM).

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does reduce_tile compute API work in tt-metal, specifically for REDUCE_COL?"
   **Reason**: Needed to understand the hardware-level behavior of column reduction and how the scaler CB interacts with the reduce instruction.
   **Key Findings**: `reduce_tile` with `REDUCE_COL` reduces the H dimension within a tile, placing output values into the first row of the destination tile. The scaler is applied during both unpack and math phases. For SUM, the scaler should be 1.0.

2. **Query**: "How does binary_op_init_common work in tt-metal compute kernels?"
   **Reason**: The compute kernel calls `binary_op_init_common` as its first operation; needed to understand what it initializes.
   **Key Findings**: It configures the unpacker (SRCA+SRCB), math, and packer hardware pipelines. Parameters are (input_cb_A, input_cb_B, output_cb). It provides generic setup that specific operations (like reduce) then specialize.

3. **Query**: "How does mask_tile work in tt-metal compute kernels?"
   **Reason**: The masking step in Phase 2 uses `mask_tile`; needed to confirm it performs element-wise multiplication.
   **Key Findings**: `mask_tile(data_dst, mask_dst)` multiplies the data tile element-wise by the mask tile in DST registers. Mask elements of 0 zero out the corresponding data elements; elements of 1 preserve them.

4. **Query**: "Relationship between ReduceOpDim::H and ReduceDim::REDUCE_COL"
   **Reason**: The program factory uses `ReduceOpDim::H` but the compute kernel uses `REDUCE_COL`; needed to confirm the mapping.
   **Key Findings**: `ReduceOpDim::H` at the TTNN level maps to `ReduceDim::REDUCE_COL` at the LLK level via `get_defines()`. This is because reducing H means collapsing rows within columns, which the hardware calls "column reduction."

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Primary reference for the reduce library's API, template parameters, and implementation
   **Key Information**: Complete reduce function signature with all policy options, accumulation mechanism, post-reduce ops, DEST auto-limit chunking for REDUCE_COL

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Understanding scaler tile preparation
   **Key Information**: `calculate_and_prepare_reduce_scaler` computes appropriate scaler (1.0 for SUM), generates tile with row-0-only pattern, pushes to CB

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register limits used by the reduce library
   **Key Information**: `DEST_AUTO_LIMIT` is 4-16 tiles depending on sync mode and FP32 accumulation. Half-sync + FP32 = 4 tiles, Full-sync + FP16 = 16 tiles.

4. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding compute helper utilities available for tile operations
   **Key Information**: Provides `_with_dt` variants of common ops (copy, add, sub, mul, bcast), `exp_tile_to_cb`, `recip_tile_to_cb`, `mask_tile_to_cb`, and broadcast variants (`mul_tiles_bcast_rows_to_cb`, `sub_tiles_bcast_cols_to_cb`, etc.) that handle full acquire/commit/wait/release/reserve/push cycles.

5. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: Understanding how `get_defines()` maps high-level reduce ops to LLK defines
   **Key Information**: `ReduceOpDim::H` -> `"ReduceDim::REDUCE_COL"`, `ReduceOpMath::SUM` -> `"PoolType::SUM"`, plus `REDUCE_ROW_SUM_VIA_MM` special case for W+SUM.

6. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`
   **Reason**: Understanding `generate_mask_h` implementation
   **Key Information**: Generates a tile of 1.0s and 0.0s in bfloat16 format. Rows 0..mask_h-1 get 1.0, rows mask_h..31 get 0.0, respecting the 4-face tile layout (top-left, top-right, bottom-left, bottom-right subtiles of 16x16 each).
