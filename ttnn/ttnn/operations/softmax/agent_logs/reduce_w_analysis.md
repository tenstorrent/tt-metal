# Reduce W (Width-Dimension Reduction) Implementation Analysis

## Overview

The **reduce W** operation reduces the width (innermost) dimension of a tiled tensor, collapsing each row of tiles into a single output tile. For a tensor of shape `[N, C, H, W]` with tile dimensions `Ht = H/32` and `Wt = W/32`, the operation produces an output where the W dimension becomes 1 tile wide. Concretely, for each "tile-row" (a row of `Wt` tiles at a given batch/channel/height position), all `Wt` tiles are reduced to a single tile.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

This analysis serves as a **reduce_w_pattern reference** for the softmax operation, which needs to reduce along the width dimension (e.g., sum of exp values).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | Wt tiles (input) -> 1 tile (output) |
| **Total units** | NC * Ht (i.e., batch*channel * height_in_tiles) |
| **Loop structure** | For each assigned tile-row: read Wt input tiles, reduce all Wt to 1 output tile, write 1 output tile |

A "work unit" is one tile-row: a contiguous sequence of `Wt` tiles sharing the same (batch, channel, height) index. Each core is assigned some number of complete tile-rows to process.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW (W innermost) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | bfloat16, float32 (determined at runtime) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [N, C, H, 1] (in tiles: NC * Ht tiles total) |
| **Dimension convention** | NCHW with W=1 tile |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Matches output_dtype from ReduceParams |

### Layout Transformations

No explicit tilize/untilize occurs within this program factory. The input is expected to already be in TILE_LAYOUT. The upstream `reduce()` function in `reduce_op.cpp` calls `tilize_with_val_padding` if the input is in ROW_MAJOR layout before invoking this factory.

## Data Flow Pattern

The data flows through three kernels in a pipeline:

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (input tensor) | CB c_0 (input tiles), CB c_2 (scaler tile) | reserve_back, push_back |
| 2 | Compute | CB c_0, CB c_2 | CB c_3 (output tile) | wait_front, pop_front (c_0), wait_front (c_2), reserve_back, push_back (c_3) |
| 3 | Writer | CB c_3 | DRAM (output tensor) | wait_front, pop_front |

**Detailed flow for one tile-row (Wt input tiles -> 1 output tile):**

1. **Reader** generates the scaler tile into CB c_2 once at startup (before the main loop). Then for each tile-row's worth of tiles, it sequentially reads `Wt` tiles from DRAM into CB c_0, one tile at a time.

2. **Compute** (primary path, SUM/AVG with `REDUCE_ROW_SUM_VIA_MM` defined):
   - Waits for scaler tile in CB c_2 (once).
   - For each tile-row: acquires DST registers, then for each of the `Wt` tiles: waits for one tile in CB c_0, performs `matmul_tiles(c_0, c_2, 0, 0, 0)` which accumulates the row-sum into DST[0], then pops the input tile.
   - After all `Wt` tiles are processed for the row, packs DST[0] to CB c_3 and releases DST.

3. **Writer** waits for each output tile in CB c_3, writes it to the output DRAM buffer, then pops the tile.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (0) | cb_src0 | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler tile for reduction | 2 tiles | 1 tile | Double | Reader | Compute | Program (written once, consumed once) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per output tile) |
| c_4* | cb_acc | Accumulator (negate path only) | 1 tile | 1 tile | Single | Compute | Compute | Row (across W tiles) |
| c_5* | cb_inv | Negated intermediate (negate path only) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile) |

*CB c_4 and c_5 are only created when `operation_attributes.negate == true`.

**Key observations for softmax reuse:**
- CB c_0 is double-buffered (2 tiles), allowing the reader to prefetch the next tile while compute processes the current one.
- CB c_2 holds the scaler tile and is sized at 2 tiles but only 1 tile is ever written. The scaler persists for the entire program lifetime.
- CB c_3 is double-buffered (2 tiles), allowing compute to produce the next output while writer sends the previous one.

## Pipeline Pattern Summary

- **CB c_0**: Double-buffered (capacity=2, block=1) -- Reader and Compute can overlap.
- **CB c_2**: Double-buffered allocation (capacity=2) but functionally single-use -- scaler written once, read persistently.
- **CB c_3**: Double-buffered (capacity=2, block=1) -- Compute and Writer can overlap.
- **CB c_4, c_5 (negate only)**: Single-buffered (capacity=1, block=1) -- Compute self-produces and self-consumes.

## Two Compute Paths: `REDUCE_ROW_SUM_VIA_MM` vs `reduce_tile`

The reduce W operation has **two distinct compute implementations** selected at compile time via the `REDUCE_ROW_SUM_VIA_MM` define.

### Path Selection Logic (in `reduce_op.cpp::get_defines`)

```
If reduce_dim == W AND (reduce_op == SUM OR reduce_op == AVG):
    defines["REDUCE_ROW_SUM_VIA_MM"] = "1"  --> matmul path
Else (reduce_op == MAX):
    no REDUCE_ROW_SUM_VIA_MM define          --> reduce_tile path
```

For **softmax** (which needs SUM reduction along W), the **matmul path** will be used.

### Path A: Matmul-Based Row Sum (`REDUCE_ROW_SUM_VIA_MM` defined) -- PRIMARY FOR SOFTMAX

**Initialization**: `mm_init(c_0, c_2, c_3)`

**Scaler tile format**: The reader calls `generate_mm_scaler(cb_id_in2, packed_bf16)` which creates a tile with the scaler value in the first column of each even face (faces 0 and 2). Specifically:
- Fills entire tile with zeros
- Sets `ptr[i]` for `i = 0, 8, 16, ..., 120` (16 values in face 0, stride 8)
- Sets `ptr[i]` for `i = 256, 264, ..., 376` (16 values in face 2, stride 8)
- Each value is `single_packed_scalar = scaler & 0xFFFF` (lower 16 bits only)

This creates a column vector tile where only the first column has the scaler value. When you multiply an input tile by this column vector tile, the result is the sum of each row of the input tile (scaled by the scaler), producing a single-column output. This is effectively a dot product of each row with the scaler column.

**Core loop** (from `reduce_w.cpp`):
```cpp
cb2.wait_front(1);  // scaler tile persists
for (uint32_t nc = 0; nc < NC; nc++) {
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        acquire_dst();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb0.wait_front(onetile);
            matmul_tiles(c_0, c_2, 0, 0, 0);  // accumulates into DST[0]
            cb0.pop_front(onetile);
        }
        cb3.reserve_back(onetile);
        pack_tile(0, c_3);
        cb3.push_back(onetile);
        release_dst();
    }
}
```

**Why matmul instead of reduce_tile**: The matmul approach provides better numerical accuracy for SUM/AVG operations. There was a known precision issue with `reduce_tile` for sum reductions, so `matmul_tiles` with a column-vector scaler is used as a more accurate workaround.

### Path B: reduce_tile-Based Path (no `REDUCE_ROW_SUM_VIA_MM`) -- USED FOR MAX

**Initialization**: `compute_kernel_hw_startup(c_0, c_2, c_3)` followed by the reduce helper library.

**Scaler tile format**: The reader calls `prepare_reduce_scaler<cb_id_in2>(scaler_f)` which:
- Zeros all faces of the tile
- Fills row 0 of each face with the scaler value (all 16 columns per half-face)
- Format is bfloat16 (packed two per uint32)

**Core loop** (from `reduce_w.cpp`, non-MM path):
```cpp
compute_kernel_hw_startup(c_0, c_2, c_3);
compute_kernel_lib::reduce<
    REDUCE_OP,
    REDUCE_DIM,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    c_0, c_2, c_3,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
```

This calls the unified reduce helper which internally handles the DST register management, tile iteration, and pack_tile operations.

### Path C: Negate Variant (`reduce_w_neg.cpp`)

Used when `operation_attributes.negate == true` (e.g., for `reduce_min` which is implemented as `-reduce_max(-x)`). This path:
1. Copies each input tile to DST, negates it, packs to CB c_5 (negated intermediate)
2. Reduces from CB c_5 using `reduce_tile`, accumulating in CB c_4
3. After all Wt tiles in a row, copies accumulator from CB c_4, negates again, packs to CB c_3 (output)

This path does NOT use the `REDUCE_ROW_SUM_VIA_MM` define and always uses `reduce_tile`.

## Scaler Setup Details (Critical for Softmax)

The scaler value is passed from the host as `operation_attributes.scaler` (a float). The program factory passes it to the reader kernel as the first compile-time argument:

```cpp
std::vector<uint32_t> reader_compile_time_args = {std::bit_cast<uint32_t>(operation_attributes.scaler)};
```

In the reader kernel, this is recovered as:
```cpp
constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
```

**For the matmul path** (`REDUCE_ROW_SUM_VIA_MM`): The float bits are converted to packed bfloat16:
```cpp
constexpr uint16_t bf16_val = static_cast<uint16_t>(scaler_bits >> 16);
constexpr uint32_t packed_bf16 = static_cast<uint32_t>(bf16_val) | (static_cast<uint32_t>(bf16_val) << 16);
generate_mm_scaler(cb_id_in2, packed_bf16);
```

**For the reduce_tile path** (non-MM): The float bits are reinterpreted and passed to `prepare_reduce_scaler`:
```cpp
float scaler_f = __builtin_bit_cast(float, scaler_bits);
dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f);
```

**For softmax**: The scaler would be 1.0 (for plain sum) or 1/N (for average). For a sum-of-exps, the scaler is 1.0.

## Index Calculations

### Input Tile Indexing

Input tiles are accessed via `TensorAccessor` with a linear page ID. The tiles are stored in row-major order within the tensor (NCHW with W-contiguous tiles). For a tensor with dimensions NC (combined batch*channel), Ht (height in tiles), Wt (width in tiles):

- Total tiles = NC * Ht * Wt
- Tile at position (nc, ht, wt) has linear index: `nc * Ht * Wt + ht * Wt + wt`

The reader kernel simply iterates from `start_id` to `start_id + num_tiles` linearly:
```cpp
for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
    // read tile with page_id = i
}
```

This works because work is assigned in contiguous tile-row groups. Each core's `start_id` is the first tile of its first assigned tile-row, and `num_tiles = num_rows_per_core * Wt`.

### Output Tile Indexing

Output tiles are also linearly indexed. Each tile-row produces exactly 1 output tile. The writer for each core writes starting at output tile index `num_tiles_read / Wt` (which equals the tile-row start index for that core), writing `num_tensor_tiles_per_core / Wt` tiles total.

## Memory Access Patterns

### Read Pattern

- **Sequential**: The reader reads tiles in strict linear order from `start_id` to `start_id + num_tiles - 1`. Within a tile-row (Wt tiles), tiles are contiguous in memory (W-contiguous layout). Across tile-rows (incrementing H or NC), tiles are also contiguous since the full width is read before moving to the next row.
- **Granularity**: One tile per NoC read transaction.
- **Source**: DRAM (interleaved), accessed via TensorAccessor which handles bank mapping.

### Write Pattern

- **Sequential**: The writer writes output tiles in strict linear order from its start output index.
- **Granularity**: One tile per NoC write transaction.
- **Destination**: DRAM (interleaved), accessed via TensorAccessor.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D work assignment) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` (device-dependent) |
| **Total cores** | min(num_rows, total_available_cores) |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` tile-rows |
| **Load balancing** | Two-group split: group_1 gets ceil(num_rows/num_cores) rows, group_2 gets floor(num_rows/num_cores) rows |

### Work Splitting Details

The total work is `num_rows = NC * Ht` tile-rows. The `split_work_to_cores` function divides this into two groups:

- **core_group_1**: Gets `num_rows_per_core_group_1` tile-rows each (the larger share, or equal if evenly divisible)
- **core_group_2**: Gets `num_rows_per_core_group_2` tile-rows each (the smaller share, or empty if evenly divisible)

Each core processes `num_rows_per_core * Wt` input tiles and produces `num_rows_per_core` output tiles.

The compute kernel is instantiated separately for each core group with different compile-time args for `Ht` (which is set to the per-core row count):
- Group 1 compile args: `{num_rows_per_core_group_1, Wt, 1}`
- Group 2 compile args: `{num_rows_per_core_group_2, Wt, 1}`

Note: NC is set to 1 in the compile-time args because the batch dimension is already folded into the per-core row count. The compute kernel sees its work as `NC=1` batch of `Ht=num_rows_per_core` rows, each `Wt` tiles wide.

### Core Enumeration

Cores are enumerated via `grid_to_cores()` (row-major order through the compute grid) or by iterating over `sub_core_grids` if specified. Runtime args are set per-core with a running `num_tiles_read` counter that tracks the global tile offset for each core's starting position.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_reduce_universal_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | scaler_bits | uint32_t | Bit-cast of float scaler value (e.g., 1.0 for SUM) |
| 1+ | tensor_args | TensorAccessorArgs | Source buffer accessor metadata (bank mapping, page size, etc.) |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index to read output from (c_3 = 3) |
| 1+ | tensor_args | TensorAccessorArgs | Destination buffer accessor metadata |

**Compute Kernel** (`reduce_w.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows this core processes (per-core row count) |
| 1 | Wt | uint32_t | Width in tiles (number of tiles to reduce per row) |
| 2 | NC | uint32_t | Always 1 (batch dimension folded into Ht) |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles for this core (num_rows * Wt) |
| 2 | start_id | uint32_t | Global tile index of first tile for this core |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address in DRAM |
| 1 | num_output_tiles | uint32_t | Number of output tiles for this core (num_rows) |
| 2 | start_id | uint32_t | Global output tile index for this core's first output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM | CB c_0, CB c_2 | Read input tiles, generate scaler tile |
| reduce_w (compute) | RISCV_2 | N/A | CB c_0, CB c_2 | CB c_3 | matmul_tiles (MM path) or reduce_tile (reduce path) |
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_3 | DRAM | Write output tiles |

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`

**Key Logic**:
1. **Scaler generation** (one-time at startup):
   - MM path: Calls `generate_mm_scaler(cb_id_in2, packed_bf16)` to create a column-vector scaler tile in CB c_2.
   - Reduce path: Calls `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)` to create a row-0-filled scaler tile in CB c_2.
2. **Input tile streaming**: Loops from `start_id` to `start_id + num_tiles`, reading one tile per iteration via TensorAccessor NoC reads into CB c_0. Each tile gets `reserve_back(1)`, async_read, barrier, `push_back(1)`.

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

**Key Logic (MM path -- relevant for softmax)**:
1. Calls `mm_init(c_0, c_2, c_3)` to initialize matmul hardware.
2. Waits for scaler tile in CB c_2 (persists for entire kernel).
3. Triple-nested loop: NC (always 1) x Ht (per-core rows) x Wt (tiles per row).
4. For each tile-row:
   - `acquire_dst()` -- claims DST registers
   - For each of Wt tiles: `cb0.wait_front(1)`, `matmul_tiles(c_0, c_2, 0, 0, 0)`, `cb0.pop_front(1)`
   - The `matmul_tiles` call accumulates: DST[0] += input_tile * scaler_column_tile. After Wt iterations, DST[0] contains the row-reduced result.
   - `cb3.reserve_back(1)`, `pack_tile(0, c_3)`, `cb3.push_back(1)`, `release_dst()`

**Key Logic (reduce_tile path)**:
1. Calls `compute_kernel_hw_startup(c_0, c_2, c_3)`.
2. Delegates to `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(c_0, c_2, c_3, ReduceInputBlockShape::of(Ht, Wt, NC))`.
3. The reduce helper internally does the same pattern: for each tile-row, acquires DST, iterates Wt tiles calling `reduce_tile`, then packs result.

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Generic tile writer that loops from `start_id` to `start_id + num_pages`.
- For each output tile: `cb_wait_front(cb_id_out, 1)`, reads L1 address, `noc_async_write_page`, flush, `cb_pop_front(1)`.
- Ends with `noc_async_write_barrier()`.

## How Width Reduction Works (Tile-Level Mechanics)

For softmax context, here is exactly what happens when reducing along W:

**Given**: Input tensor of shape [1, 1, 32, 128] = 1 batch, 1 channel, 1 tile height, 4 tiles width (Wt=4).

Each tile is 32x32 elements. A "tile-row" is 4 tiles side-by-side, representing 32 rows of 128 elements each.

**The matmul approach**: The scaler tile is a 32x32 tile with only the first column populated (value = scaler, e.g., 1.0). When you multiply a 32x32 input tile by this column-vector tile using `matmul_tiles`:
- Result[i][j] = sum over k of (input[i][k] * scaler_tile[k][j])
- Since scaler_tile[k][j] is 0 for j>0, only column 0 of the result is non-zero
- Result[i][0] = sum over k of (input[i][k] * scaler_value) = scaler_value * sum(input[i][:])

So `matmul_tiles` effectively computes the row-wise sum (scaled) of the input tile. When called Wt times with accumulation (DST not cleared between calls), the final result in DST[0] is the row-wise sum across ALL Wt tiles.

**Output**: 1 tile where column 0 of each row contains the reduced value (sum of all 128 elements in that row of the original tensor).

## Implementation Notes

### Compile-Time Define Propagation

The `reduce_defines` map is passed to all three kernels (reader, compute, writer) via their respective config objects. Key defines:
- `REDUCE_OP`: `"PoolType::SUM"`, `"PoolType::AVG"`, or `"PoolType::MAX"`
- `REDUCE_DIM`: `"ReduceDim::REDUCE_ROW"` (always, for W reduction)
- `REDUCE_ROW_SUM_VIA_MM`: `"1"` (only for SUM/AVG, not MAX)

### Scaler CB Data Format Hardcoded

The scaler CB (c_2) is hardcoded to `Float16_b` (bfloat16) format regardless of input data type. This is because the tile generation code in the reader (both `generate_mm_scaler` and `prepare_reduce_scaler`) assumes bfloat16 format. The comment in the program factory explicitly notes this: "Scaler datatype is hardcoded bfloat16 due to tile creation in reader".

### Program Caching

The program factory supports program caching via `override_runtime_arguments`. When the program is reused, only the buffer addresses (source and destination) are updated. All other parameters (tile counts, start IDs) remain the same. This works because the tensor shape and core assignment don't change between cached invocations -- only the buffer pointers may change.

### NC Folding into Ht

A subtle but important design choice: the compute kernel receives `NC=1` and `Ht=num_rows_per_core`. This means the batch and channel dimensions are folded into the height dimension from the compute kernel's perspective. Since W reduction treats each tile-row independently, there is no semantic difference between "height row 5 of batch 0" and "height row 0 of batch 1" -- they are all just independent rows to reduce. This simplification avoids the compute kernel needing to know about batch boundaries.

### Softmax Relevance

For a softmax operation that needs `sum(exp(x))` along the W dimension:
1. The exp values would be computed first and stored in a CB.
2. A reduce-W pattern would then sum those exp tiles along the W dimension.
3. The scaler would be 1.0 (for plain sum).
4. The matmul path (`REDUCE_ROW_SUM_VIA_MM`) provides the best accuracy for this sum.
5. The key pattern to replicate is: `mm_init`, then for each row: `acquire_dst`, loop Wt times calling `matmul_tiles` with accumulation, `pack_tile`, `release_dst`.

Alternatively, the `compute_kernel_lib::reduce` helper library can be used directly, which encapsulates all the DST management:
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
    cb_exp, cb_scaler, cb_sum_exp,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
```

For softmax specifically, the `WaitUpfrontNoPop` or `NoWaitNoPop` policies may be more appropriate if the exp tiles need to be reused for the subsequent division step.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does reduce_tile work in the compute kernel for width (REDUCE_ROW) reductions? What is the relationship between reduce_tile and matmul_tiles for implementing row reduction?"
   **Reason**: Needed to understand the two compute paths and why matmul is used as an alternative to reduce_tile.
   **Key Findings**: `reduce_tile` calls `llk_math_reduce` and `llk_unpack_AB_reduce` internally. `matmul_tiles` is used as a workaround for precision issues in `reduce_tile` for SUM operations. The scaler tile format differs between the two paths.

2. **Query**: "How does split_work_to_cores work in tt-metal? What are core_group_1 and core_group_2?"
   **Reason**: Needed to understand the work distribution strategy used by the program factory.
   **Key Findings**: `split_work_to_cores` divides work units across cores, creating two groups when work doesn't divide evenly. Group 1 gets the larger share (ceil), group 2 gets the smaller share (floor). Returns num_cores, all_cores range, and per-group work counts.

3. **Query**: "What is the REDUCE_ROW_SUM_VIA_MM define and how does it change the reduce W implementation?"
   **Reason**: Needed to understand why there are two compute paths and when each is used.
   **Key Findings**: `REDUCE_ROW_SUM_VIA_MM` is set for W-dimension SUM/AVG reductions to use matmul instead of reduce_tile for better numerical accuracy. The scaler tile is crafted as a column vector for the matmul path.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Comprehensive documentation of the unified reduce helper function.
   **Key Information**: Detailed API documentation for `compute_kernel_lib::reduce<>()` including all input policies, accumulation support, and post-reduce operations. The `WaitUpfrontNoPop` policy is explicitly mentioned as ideal for softmax patterns.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   **Reason**: Understanding scaler tile generation for reduce operations.
   **Key Information**: `prepare_reduce_scaler` creates a tile with scaler values in row 0 of each face. `calculate_and_prepare_reduce_scaler` can automatically compute the correct scaler based on pool type.

3. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp` (lines 20-42)
   **Reason**: Understanding how `get_defines` maps ReduceOpMath/ReduceOpDim to compile-time defines.
   **Key Information**: W maps to REDUCE_ROW, H maps to REDUCE_COL, HW maps to REDUCE_SCALAR. SUM/AVG with W triggers REDUCE_ROW_SUM_VIA_MM.

4. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding the hardware initialization function called in the reduce_tile path.
   **Key Information**: Must be called exactly once at kernel start. Configures UNPACK, MATH, and PACK hardware. Takes input CB IDs and output CB ID. Cannot be called mid-kernel.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DST register capacity limits referenced in the reduce helper.
   **Key Information**: DEST capacity depends on sync mode (Half/Full) and accumulation mode (16/32-bit). Half-sync + fp32 = 4 tiles, Half-sync + fp16 = 8 tiles. This affects chunking strategy for REDUCE_COL but not REDUCE_ROW (which only uses DST[0]).
