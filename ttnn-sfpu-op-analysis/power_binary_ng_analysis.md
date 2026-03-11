# POWER (binary_ng) Implementation Analysis

## Overview

The POWER operation computes element-wise exponentiation: `c = a ** b`, where `a` is the base tensor and `b` is either a tensor or a scalar exponent. It is implemented as an SFPU-only binary operation within the `binary_ng` framework -- it cannot run on the FPU path. The operation supports full NumPy-style broadcasting across up to 6+ dimensions.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The `binary_ng` framework is a generalized binary operation engine that handles many operations (ADD, MUL, DIV, POWER, etc.) through a single program factory with operation-specific behavior injected via compile-time defines. POWER is distinguished by:
1. It is always routed to the SFPU path (`is_sfpu = true`)
2. It uses a custom `UnpackToDestMode` policy (conditional on dtype, not always FP32)
3. The SFPU kernel implements `base**pow` via `log2(base)` followed by `2**(pow * log2(base))` using polynomial approximation

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` = total output tiles |
| **Loop structure** | Flat loop over tiles assigned to each core; nested ND iteration in reader/writer for broadcast stride calculation |

## Tensor Format and Layout

### Input Tensor A (base)

| Property | Value |
|----------|-------|
| **Logical shape** | Up to 6+ dimensions, e.g. [nD, D, N, C, H, W] |
| **Dimension convention** | Last 5 dims mapped as [D, N, C, Ht, Wt]; dims beyond 5 collapsed into `nD` |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT, WIDTH, BLOCK) |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32 (primary); INT32 not supported for POWER |

### Input Tensor B (exponent) -- tensor path

| Property | Value |
|----------|-------|
| **Logical shape** | Broadcastable to output shape |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as A (typically BFLOAT16 or FLOAT32) |

### Input B -- scalar path

When B is a scalar, no B tensor exists. The scalar value is packed into a runtime argument for the writer kernel, which fills a single tile in CB c_1 with the scalar value. The compute kernel then reuses this single tile for every output tile.

### Output Tensor C

| Property | Value |
|----------|-------|
| **Logical shape** | Broadcast result of A and B shapes |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input dtype or explicitly specified via `output_dtype` |

### Layout Transformations

No tilize/untilize occurs within the operation. All inputs and outputs must already be in TILE_LAYOUT. The program factory handles broadcasting via stride computations (stride = 0 when a dimension is 1 in the input but > 1 in the output).

## Data Flow Pattern

### Tensor-Tensor Path (B is a tensor, SubtileBroadcastType::NONE)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (RISCV_0) | DRAM/L1 (A buffer) | CB c_0 | `noc_async_read_page` -> `cb_reserve_back` / `cb_push_back` |
| 1 | Reader (RISCV_0) | DRAM/L1 (B buffer) | CB c_1 | `noc_async_read_page` -> `cb_reserve_back` / `cb_push_back` |
| 2 | Compute (RISCV_2) | CB c_0, CB c_1 | CB c_2 | `cb_wait_front` -> `copy_tile` to DST -> `power_binary_tile` -> `pack_tile` -> `cb_push_back` / `cb_pop_front` |
| 3 | Writer (RISCV_1) | CB c_2 | DRAM/L1 (C buffer) | `cb_wait_front` -> `noc_async_write_page` -> `cb_pop_front` |

### Scalar Path (B is a scalar)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Writer (RISCV_1) | Scalar arg | CB c_1 | Fill tile once with scalar, `cb_push_back` (1 tile, never popped) |
| 1b | Reader (RISCV_0) | DRAM/L1 (A buffer) | CB c_0 | `noc_async_read_page` -> `cb_reserve_back` / `cb_push_back` |
| 2 | Compute (RISCV_2) | CB c_0, CB c_1 | CB c_2 | `cb_wait_front(c_1)` once before loop; per tile: `cb_wait_front(c_0)` -> SFPU op -> `cb_push_back(c_2)` / `cb_pop_front(c_0)` |
| 3 | Writer (RISCV_1) | CB c_2 | DRAM/L1 (C buffer) | `cb_wait_front` -> `noc_async_write_page` -> `cb_pop_front` |

**Key difference**: In the scalar path, the writer kernel is responsible for both filling the scalar tile into CB c_1 AND writing output tiles from CB c_2. The reader only reads tensor A into CB c_0.

## Circular Buffer Configuration

### Interleaved (non-sharded) case

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src_b | Input B tiles (tensor) / scalar tile | 2 tiles (tensor) or 1 tile (scalar) | 1 tile | Double (tensor) / Single (scalar) | Reader (tensor) or Writer (scalar) | Compute | Block (tensor) / Program (scalar) |
| c_2 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_lhs_interim | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_rhs_interim | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Notes**:
- CB c_3 and c_4 are only created if `lhs_activations` / `rhs_activations` are non-empty (POWER itself does not set `process_lhs` or `process_rhs`, but user-supplied activations may be present).
- CB c_5 and c_6 are only created for ROW_A/ROW_B or ROW_A_COL_B/ROW_B_COL_A broadcast types.
- When sharded, CB capacities are set to the shard volume (number of tiles in the shard) instead of 2, and the CB is backed directly by the tensor's L1 buffer.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| c_0 | 2 tiles | 1 tile | Double-buffered | Reader can fill next tile while compute processes current |
| c_1 (tensor) | 2 tiles | 1 tile | Double-buffered | Same as c_0 |
| c_1 (scalar) | 1 tile | 1 tile | Single-buffered | Written once, consumed repeatedly (never popped) |
| c_2 | 2 tiles | 1 tile | Double-buffered | Compute can produce next tile while writer drains current |

## Index Calculations

The reader and writer kernels use a 6-level nested loop structure to map a flat starting tile ID to multi-dimensional coordinates `(nD, D, N, C, Ht, Wt)`.

**Starting position decomposition** (from `start_tile_id`):
```
tiles_per_n  = C * Ht * Wt
tiles_per_d  = N * tiles_per_n
tiles_per_nd = D * tiles_per_d
start_nd = start_tile_id / tiles_per_nd
start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
start_n  = ... / tiles_per_n
start_c  = ... / (Ht * Wt)
start_th = ... / Wt
start_tw = ... % Wt
```

**Broadcasting stride logic**: The program factory computes strides for each dimension of both A and B:
- `nD_stride = aHt * aWt * aC * aN * aD * (aND > 1)` -- zero when dimension is 1 (broadcast)
- `d_stride = aHt * aWt * aC * aN * (aD > 1)` -- similarly for each level
- This means when a dimension in A (or B) is 1, the stride is 0, causing the reader to re-read the same tile index for that dimension, implementing broadcast.

**TensorAccessor**: Used for mapping tile indices to physical DRAM/L1 addresses. Compile-time args encode the accessor configuration; common runtime args encode the buffer address and shape info.

## Memory Access Patterns

### Read Pattern
- **Tile-by-tile sequential** within each (Ht, Wt) slice
- For each output tile, the reader computes an input tile index using stride-based offsets
- When no broadcast, the pattern is simple linear iteration
- When broadcasting, the same input tile is re-read for multiple output positions (stride = 0)
- Each tile read uses `noc_async_read_page` followed by `noc_async_read_barrier` (synchronous per tile)
- In the no-broadcast tensor-tensor case, A and B tiles are read in lockstep (same loop body)

### Write Pattern
- **Tile-by-tile sequential** through the output tensor
- Each tile uses `noc_async_write_page` followed by `noc_async_write_barrier` (synchronous per tile)
- For sharded output, the writer loop adjusts `dst_tile_offset` by `(Wt - dst_shard_width)` at the end of each tile row to skip inter-shard gaps

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (worker grid from device) |
| **Grid dimensions** | Device-dependent (e.g., 8x8 for Wormhole) |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` for zero-start grids |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(total_tiles / num_cores)`, group 2 gets `floor(total_tiles / num_cores)` |

**Zero-start grid optimization**: When the grid is a single rectangular range starting at (0,0) and any sharding also starts at (0,0), a fast path uses `split_work_to_cores(compute_with_storage_grid, c_num_tiles, row_major)` which avoids iterating over CoreRangeSets.

**Sharded case**: Each core processes its own shard. The core count and tile distribution are determined by the shard spec grid. Shard shapes may vary on edge cores (handled by `ShardShapeGenerator`).

**Inactive cores**: Cores outside the work range receive all-zero runtime arguments and effectively no-op.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per compute iteration |

**Compile-time defines** (not positional args, but critical):
- `BINARY_SFPU_INIT` = `power_binary_tile_init();`
- `BINARY_SFPU_OP` = `power_binary_tile`
- `BCAST_INPUT` = `""` (no broadcast for NONE type) or `"0"`/`"1"` for broadcast types
- `PROCESS_LHS_ACTIVATIONS(i)` = `""` (empty unless user adds activations)
- `PROCESS_RHS_ACTIVATIONS(i)` = `""` (empty unless user adds activations)
- `PROCESS_POST_ACTIVATIONS(i)` = `""` (empty unless user adds post-activations)

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(A) | uint32_t[] | Compile-time portion of tensor accessor for A |
| N+1..M | TensorAccessorArgs(B) | uint32_t[] | Compile-time portion of tensor accessor for B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Compile-time defines**:
- `SRC_SHARDED` = `"0"` or `"1"` -- whether A is sharded in L1
- `SRC_SHARDED_B` = `"0"` or `"1"` -- whether B is sharded in L1

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(C) | uint32_t[] | Compile-time portion of tensor accessor for C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Compile-time defines**:
- `SRC_SHARDED` = `"0"` or `"1"` -- whether B is sharded (only for scalar writer)
- `DST_SHARDED` = `"0"` or `"1"` -- whether C is sharded

### Runtime Arguments

#### Reader Kernel (21 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of tensor A buffer |
| 1 | start_tile_id | uint32_t | Output tile ID where this core begins (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (0 if not sharded) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (0 if not sharded) |
| 5 | nD_stride | uint32_t | A's stride for collapsed dims > 5 (0 if broadcast) |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed dimensions > 5 |
| 15 | src_addr_b | uint32_t | Base address of tensor B buffer (0 if scalar) |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed dims > 5 |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (0 if not sharded) |

#### Writer Kernel -- Tensor Path (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer C |
| 1 | start_tile_id | uint32_t | Output start tile ID for this core |
| 2 | dst_num_tiles | uint32_t | Number of output tiles to write |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (0 if not sharded) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed dimensions > 5 |
| 10 | (unused) | uint32_t | Padding to 11 args (set to 0) |

#### Writer Kernel -- Scalar Path (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar B value packed as uint32 (bfloat16 pair or float bits) |
| 1 | dst_addr | uint32_t | Base address of output buffer C |
| 2 | start_tile_id | uint32_t | Output start tile ID |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Collapsed dimensions > 5 |

#### Compute Kernel (4 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of tiles this core must process |
| 1 | freq | uint32_t | Broadcast frequency: 1 (no bcast), Wt (col bcast), Ht*Wt (scalar bcast) |
| 2 | counter | uint32_t | Starting broadcast counter offset |
| 3 | compute_scalar_value | uint32_t | Always 0 for POWER (used by quant/where ops) |

## Kernel Implementations

### Reader Kernel -- Tensor-Tensor, No Broadcast

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles in lockstep |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads one A tile and one B tile per inner loop iteration. Uses TensorAccessor for address translation. The 6-level nested loop iterates over `(nD, D, N, C, Ht, Wt)`. Both A and B have independent stride calculations to support broadcasting where one input has size-1 dimensions.

### Reader Kernel -- Scalar Path

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 (A only) | CB c_0 | Read A tiles only |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Same 6-level nested loop but only reads tensor A into CB c_0. The scalar B is handled by the writer kernel.

### Compute Kernel -- Tensor-Tensor, No Broadcast (SFPU)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0, CB c_1 | CB c_2 | SFPU power_binary_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: Per tile:
  1. Wait for A tile in `cb_post_lhs` and B tile in `cb_post_rhs`
  2. Acquire DST registers
  3. `copy_tile` A to DST slot `i*2`, B to DST slot `i*2+1`
  4. Call `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which resolves to `power_binary_tile(i*2, i*2+1, i*2)` -- computes `a**b` and stores result in DST slot `i*2`
  5. Apply post-activations if any
  6. Commit, wait, pack result from DST to CB c_2
  7. Pop consumed tiles from input CBs

### Compute Kernel -- Scalar Path (SFPU)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0, CB c_1 | CB c_2 | SFPU power_binary_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: Identical to tensor-tensor except:
  - The RHS (scalar) tile in CB c_1 is waited on ONCE before the loop and popped ONCE after all tiles are processed
  - The LHS tile changes each iteration, the RHS tile remains constant

### Writer Kernel -- Tensor Path

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: 6-level nested loop mirrors the reader. Waits for computed tiles in CB c_2, writes them out via `noc_async_write_page`, then pops.

### Writer Kernel -- Scalar Path

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | Scalar arg -> CB c_1; CB c_2 -> DRAM/L1 (C) | CB c_1, DRAM/L1 (C) | Fill scalar tile + write output |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills CB c_1 with the packed scalar value (one tile). Then enters the standard 6-level write loop for output tiles from CB c_2.

## SFPU Power Implementation Details

The SFPU kernel `power_binary_tile` dispatches to `calculate_sfpu_binary_pow` which iterates 8 times over SFPI vector slices (covering the full 32x32 tile). The core algorithm:

**When `is_fp32_dest_acc_en = false`** (BFLOAT16 mode): Uses `_sfpu_binary_power_21f_` -- a 3rd-order polynomial approximation:
1. Compute `log2(base)` by normalizing base to [1,2], evaluating a 3rd-degree Remez polynomial, then adding back the exponent
2. Compute `2**(pow * log2(base))` using the `exp_21f` algorithm from Moroz et al. 2022, which uses bit manipulation (BMT) for the final exponentiation
3. Post-processing handles special cases: `0**negative = NaN`, negative base with integer power gets correct sign, negative base with non-integer power = NaN
4. Explicit `float_to_fp16b` round-to-nearest-even conversion before storing to avoid bfloat16 truncation artifacts

**When `is_fp32_dest_acc_en = true`** (FLOAT32 mode): Uses `_sfpu_binary_power_61f_` -- a 5th-degree Chebyshev polynomial for higher accuracy:
1. Same two-step approach (log2 then exp2) but with degree-5 log2 polynomial and degree-6 exp polynomial
2. All polynomial coefficients are floating-point (no BMT tricks)
3. No explicit bfloat16 conversion (result stays in float32)

**Programmable constants** (set during init):
- `vConstFloatPrgm0 = 1.442695f` (1/ln(2))
- `vConstFloatPrgm1 = -127.0f` (clamping threshold)
- `vConstFloatPrgm2 = NaN` (for special case results)

## POWER-Specific UnpackToDestMode

Unlike most SFPU binary operations that unconditionally use `UnpackToDestFp32` for all source CBs, POWER uses a **conditional** unpack mode (program factory lines 741-755):

- For BFLOAT16 inputs: `UnpackToDestMode::Default` (no FP32 upconversion during unpack)
- For FLOAT32 inputs: `UnpackToDestMode::UnpackToDestFp32`

This is because the POWER SFPU kernel has its own internal precision management (the 21f vs 61f algorithm selection) and does not benefit from forcing FP32 unpack when the data is already BFLOAT16.

## Implementation Notes

1. **No FPU path**: POWER always requires SFPU. Attempting to use it with an FPU config will throw: `"Unsupported binary op for FPU"`.

2. **Broadcast handling**: The `binary_ng` framework supports 9 subtile broadcast types. For POWER, any of these may apply depending on the input tensor shapes. The broadcast type determines which reader/compute kernel variants are selected.

3. **Program caching**: The operation uses `override_runtime_arguments` for efficient re-execution with different tensors of the same shape/config, avoiding kernel recompilation.

4. **Sharding fallback**: If the output tensor has uneven shards (shard shape doesn't evenly divide tensor shape), the operation falls back to treating tensors as interleaved even if they're sharded, to avoid kernel deadlocks from asymmetric core workloads.

5. **Accuracy tradeoff**: The 21f (BFLOAT16) variant uses a 3rd-order polynomial for log2 approximation, while the 61f (FLOAT32) variant uses 5th-order. This matches the precision requirements of each data format.

6. **Scalar packing**: For BFLOAT16 scalars, two copies of the BF16 value are packed into a single uint32. For FLOAT32, the scalar's bits are reinterpreted as uint32. The writer kernel's fill function uses the appropriate unpack (`float*` reinterpret for FP32, direct value for int types).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What is its program factory structure, what kernels does it use, and how does it handle different subtypes like scalar, broadcast, and element-wise binary operations?"
   **Reason**: Initial reconnaissance to understand the binary_ng framework architecture before diving into source code.
   **Key Findings**: Confirmed that binary_ng uses a unified program factory with dynamic kernel selection based on SubtileBroadcastType. POWER is an SFPU-only operation. Nine broadcast types are supported. The framework handles tensor-tensor and tensor-scalar paths differently.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` and `.cpp`
   **Reason**: Understanding kernel name resolution, OpConfig for POWER, and how defines are generated.
   **Key Information**: POWER maps to `SfpuBinaryOp::POWER`, generates defines `BINARY_SFPU_INIT = power_binary_tile_init();` and `BINARY_SFPU_OP = power_binary_tile`. No pre/post processing activations are set by the operation itself.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
   **Reason**: Understanding the actual SFPU math implementation for power.
   **Key Information**: Two algorithm variants (21f for BF16, 61f for FP32) based on Moroz et al. 2022 polynomial approximation of log2 + exp2. Special case handling for negative bases and zero bases. 8 iterations per tile to cover all 256 vector slices.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Confirming the API-level function signatures for power_binary_tile and power_binary_tile_init.
   **Key Information**: `power_binary_tile(idst0, idst1, odst)` dispatches to `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>`. The `DST_ACCUM_MODE` template parameter controls whether the 21f or 61f algorithm is used.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`
   **Reason**: Understanding the operation_attributes_t structure and SubtileBroadcastType enum.
   **Key Information**: The operation supports `lhs_activations`, `rhs_activations`, and `post_activations` as user-configurable pre/post-processing steps. The `is_sfpu` flag is set in the operation attributes before reaching the program factory.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_pow.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** calls `power_binary_tile(idst0, idst1, odst)` (defined in `eltwise_binary_sfpu.h`), which is wrapped in the `MATH(...)` macro to ensure it runs only on the math RISC-V.
2. **API layer** dispatches to `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>(dst_index0, dst_index1, odst)`, where `DST_ACCUM_MODE` is a global compile-time constant that selects between BF16 and FP32 precision paths.
3. **LLK dispatch** calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu_func, dst_index0, dst_index1, odst, VectorMode::RC)`, passing `calculate_sfpu_binary_pow<APPROX, 8, is_fp32_dest_acc_en>` as the callable.
4. **Params function** sets the DST write address, stalls until SFPU is ready via `TTI_STALLWAIT`, then loops over 4 faces (in RC mode), calling `sfpu_func(dst_index_in0, dst_index_in1, dst_index_out)` for each face, with `TTI_SETRWC` advancing the DEST pointer by 16 rows between faces.
5. **Core SFPU function** `calculate_sfpu_binary_pow` iterates 8 times (2 per face, covering 4 rows of 32 elements each iteration = 8 rows per call = 32 rows per face over 4 calls), loading base and exponent from DEST, computing the result via `_sfpu_binary_power_<is_fp32_dest_acc_en>`, and writing back to DEST.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;   // 1/ln(2), used by both log2 and exp2 steps
    sfpi::vConstFloatPrgm1 = -127.0f;     // clamping threshold to prevent overflow in exp2
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN(); // used for special-case NaN results
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32; // 64/SFP_DESTREG_STRIDE = 32 sfpi-addressable rows per tile
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD: load base from DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD: load exponent from DEST

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1); // dispatch to 21f or f32

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE: write result to DEST
        sfpi::dst_reg++; // TTI_INCRWC: advance DEST pointer by SFP_DESTREG_STRIDE (2)
    }
}

// Template specialization dispatch:
// is_fp32_dest_acc_en=false -> _sfpu_binary_power_21f_<false>(base, pow)
// is_fp32_dest_acc_en=true  -> _sfpu_binary_power_f32_(base, pow)

template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Implementation notes, see the original file for more details

    // Step 1: Compute log2(base)
    sfpi::vFloat absbase = setsgn(base, 0);       // SFPSETSGN: clear sign bit
    sfpi::vFloat x = sfpi::setexp(absbase, 127);  // SFPSETEXP: normalize to [1, 2) range

    // 3rd order Remez polynomial approx of ln(x) for x in [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Extract and convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base); // SFPEXEXP: extract biased exponent as signed int
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); } // SFPSETSGN: negate via ones-complement + 1 with sign bit set
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0); // SFPIADD: convert int32 to float

    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0; // 1/ln(2)
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // Step 2: Compute 2^(pow * log2(base)) using Moroz exp_21f
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1; // -127.0f
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; } // clamp to prevent overflow
    v_endif;

    // Implementation notes, see the original file for more details

    z_f32 = addexp(z_f32, 23);  // SFPDIVP2 with ADD mode: multiply by 2^23 in single cycle
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias); // helper: SFPEXEXP + SFPEXMAN + SFPSHFT

    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // SFPEXEXP: extract exponent bits
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXMAN with PAD9: extract mantissa

    // Horner-form polynomial evaluation (Moroz et al. Section 5)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0); // SFPIADD + int32_to_float
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent to reconstruct result
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii)); // SFPSETEXP

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Post-processing: special case handling
    sfpi::vInt pow_int =
        sfpi::float_to_int16(pow, 0);  // SFPIADD: convert float to int16 (truncation)
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0); // SFPIADD: convert back for comparison

    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN for 0^(negative)
    }
    v_endif;

    v_if(base < 0.0f) {
        y = setsgn(y, pow_int << 31); // SFPSETSGN + SFPSHFT: set sign from LSB of integer power
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN for negative base with non-integer power
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // SFPSTOCHRND: round-to-nearest-even FP32->BF16 to avoid truncation artifacts
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base) using improved log with Newton-Raphson reciprocal
    sfpi::vFloat abs_base = sfpi::abs(base); // SFPABS: absolute value
    sfpi::vFloat m = sfpi::setexp(abs_base, 127); // SFPSETEXP: normalize mantissa to [1,2)
    sfpi::vInt exp = sfpi::exexp(abs_base); // SFPEXEXP: extract exponent

    // Range reduction: ensure m in [sqrt(2)/2, sqrt(2)]
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = m * 0.5f;
        exp = exp + 1;
    }
    v_endif;

    // Transform to z = (m - 1) / (m + 1) via Newton-Raphson reciprocal
    sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
    sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
    sfpi::vFloat recip = sfpi::vConst1 - 0.2426406871192851f * m_plus_1; // linear initial guess
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st Newton-Raphson iteration
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd Newton-Raphson iteration
    sfpi::vFloat z = m_minus_1 * recip;

    // Degree-5 odd-powers polynomial: ln(m) = 2*z*P(z^2)
    sfpi::vFloat z2 = z * z;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, sfpi::vConst1, 0.3333333333333333f, 0.2f, 0.14285714285714285f, 0.1111111111111111f, 0.09090909090909091f);
    sfpi::vFloat ln_m = 2.0f * (z * p);

    // Convert exponent to float with proper sign handling
    sfpi::vInt sign_bit = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(exp) >> 31);
    sfpi::vInt exp_sign = sfpi::vInt(0) - sign_bit;
    sfpi::vInt exp_abs = (exp ^ exp_sign) - exp_sign; // two's complement abs
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(sfpi::setsgn(exp_abs, exp_sign), 0); // SFPIADD + SFPSETSGN

    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat log2_result = exp_f32 + ln_m * vConst1Ln2;

    // Step 2: 2^(pow*log2(base)) via Cody-Waite + 7th-order Taylor exp
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    constexpr float LN2 = 0.693147180559945309f;
    sfpi::vFloat y = _sfpu_exp_f32_accurate_(z_f32 * LN2); // Cody-Waite reduction + Taylor series

    // Special case handling (same as 21f variant)
    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;
    }
    v_endif;

    v_if(base < 0.0f) {
        sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
        sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);
        y = sfpi::setsgn(y, pow_int << 31);
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    return y;
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | SFPU Opcode | Description |
|--------------------------|-------------|-------------|
| `sfpi::dst_reg[N]` (read) | SFPLOAD | Loads a 32-element vector from DEST register file into an LReg |
| `sfpi::dst_reg[N] = val` (write) | SFPSTORE | Stores an LReg vector back to DEST register file |
| `sfpi::dst_reg++` | SFPINCRWC (TTI_INCRWC) | Advances the DEST read/write pointer by `SFP_DESTREG_STRIDE` (2 rows) |
| `setsgn(v, imm)` | SFPSETSGN | Sets or clears the sign bit of each lane; used for absolute value, negation, and sign transfer |
| `setexp(v, imm)` | SFPSETEXP | Sets the exponent field of each lane to an immediate value; used for range normalization |
| `exexp(v)` | SFPEXEXP | Extracts the biased exponent as a signed integer (debiased by 127); used for log2 decomposition |
| `exman9(v)` | SFPEXMAN (PAD9 mode) | Extracts the 23-bit mantissa with 9-bit zero padding; used in Moroz exp_21f |
| `addexp(v, imm)` | SFPDIVP2 (ADD mode) | Adds `imm` to the exponent field, equivalent to multiplying by `2^imm`; single-cycle power-of-2 scaling |
| `abs(v)` | SFPABS (FLOAT mode) | Computes absolute value by clearing sign bit; used only in FP32 path |
| `float_to_int16(v, 0)` | SFPIADD | Converts float to int16 by truncation; used to check if exponent is integer |
| `int32_to_float(v, 0)` | SFPIADD | Converts int32 to float; used for exponent-to-float conversion and integer-power roundtrip |
| `float_to_fp16b(v, 0)` | SFPSTOCHRND (FP32_TO_FP16B mode) | Rounds FP32 mantissa to 7 bits using round-to-nearest-even; BF16 path only |
| `v_if / v_endif` | SFPSETCC / SFPPUSHC / SFPPOPC / SFPENCC | Predicated execution via condition code stack; comparisons set CC, branches push/pop |
| `reinterpret<T>(v)` | (no instruction) | Compile-time type cast between vFloat/vInt/vUInt with no runtime cost |
| `vConstFloatPrgm0/1/2` | SFPLOADI (CREG access) | Reads from programmable constant registers loaded during init via SFPLOADI |
| `vConst1` | SFPLOADI (CREG access) | Reads hardcoded constant 1.0f from the constant register file |
| `operator*`, `operator+`, `operator-` | SFPMAD / SFPMUL | Multiply-accumulate or multiply; the compiler fuses `a*b+c` into a single SFPMAD when possible |
| `operator<<` (vInt) | SFPSHFT | Left-shift integer vector; used to move LSB of pow_int to sign position |
| `operator>>` (vUInt) | SFPSHFT | Right-shift unsigned integer vector; used for sign extraction in FP32 path |
| `operator~`, `operator^` | SFPNOT / SFPXOR | Bitwise NOT and XOR; used for two's complement negation of exponent |

Additionally, the FP32 path calls `_sfpu_exp_f32_accurate_()` which internally uses:

| Instruction / Intrinsic | SFPU Opcode | Description |
|--------------------------|-------------|-------------|
| `exexp_nodebias(v)` | SFPEXEXP (no debias mode) | Extracts raw exponent without subtracting bias; used for ldexp-style scaling |
| `_sfpu_round_nearest_int32_(z, k_int)` | SFPIADD variants | Round-to-nearest-even float-to-int conversion for Cody-Waite reduction |
| `PolynomialEvaluator::eval(...)` | Multiple SFPMAD | Horner-form polynomial evaluation; compiles to a chain of SFPMAD instructions |

### SFPU Register Usage

**LRegs (L0-L7)**: The SFPU has 8 local registers for intermediate computation. The power kernel is register-intensive:
- The BF16 path (`_sfpu_binary_power_21f_`) uses at minimum 6-7 simultaneous live values during the Moroz exp_21f section (z_f32, bias, zii, zif, d1, d2, d3), pushing close to the 8-LReg limit. The compiler must carefully schedule to avoid spills.
- The FP32 path (`_sfpu_binary_power_f32_`) is even more register-heavy due to the Newton-Raphson reciprocal (m, m_plus_1, m_minus_1, recip, z, z2, p, ln_m) and the subsequent Cody-Waite + Taylor exp inside `_sfpu_exp_f32_accurate_`. This path likely causes register spills.
- `in0` (base) and `in1` (pow) are loaded into LRegs at the start of each iteration and must remain live through much of the computation.

**DEST register file**: The kernel accesses DEST via indexed addressing:
- `dst_index_in0 * 32`: Base tile location in DEST (e.g., DST slot 0 = row 0)
- `dst_index_in1 * 32`: Exponent tile location in DEST (e.g., DST slot 1 = row 32)
- `dst_index_out * 32`: Output tile location in DEST (typically same as in0, so slot 0)
- `dst_reg++` increments the base pointer by `SFP_DESTREG_STRIDE` (2 rows) each iteration, so 8 iterations * 2 rows = 16 rows per face, and the params function advances by another 16 rows (2x `TTI_SETRWC` of 8) between faces.

**Programmable constant registers** (CREGs):
- `vConstFloatPrgm0` = 1.442695f (1/ln(2)) -- used in both log2 and exp2 computations
- `vConstFloatPrgm1` = -127.0f -- overflow clamping threshold
- `vConstFloatPrgm2` = NaN -- special case output value

### Address Mode Configuration

The SFPU binary power operation uses address mode configuration set during `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`:

**ADDR_MOD_7** is configured with all increments set to zero:
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

This means the SFPU does not auto-increment DEST addresses between instructions -- all DEST pointer advancement is handled explicitly by `dst_reg++` (which compiles to `TTI_INCRWC`) and `TTI_SETRWC` calls in the params function.

The ADDR_MOD_6 variant (with `.dest = {.incr = 2}`) is NOT used for POWER -- it is only configured for specific operations like `mul_int32`, `max`, `min`, etc. Since `SfpuType::unused` is passed to the init template, the `if constexpr` branch for ADDR_MOD_6 is not taken.

**Wormhole vs Blackhole differences**: The `_llk_math_eltwise_binary_sfpu_start_` function differs slightly:
- **Wormhole B0**: Calls `math::set_addr_mod_base()` at start and `math::clear_addr_mod_base()` at done, plus `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` at done.
- **Blackhole**: Omits the `set_addr_mod_base()` / `clear_addr_mod_base()` calls and the `STALL_CFG` wait at done. The start still issues `TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)`.

The address mode configuration itself (ADDR_MOD_7 with zero increments) is identical across both architectures.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary_ng POWER operation work? What SFPU kernel does it use? Trace from the compute kernel through to the ckernel SFPU implementation for power/pow."
   **Reason**: Establish the full call chain from the compute kernel API down to the core SFPU implementation.
   **Key Findings**: Confirmed the call path: `power_binary_tile` -> `llk_math_eltwise_binary_sfpu_binary_pow` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary_pow` -> `_sfpu_binary_power_<is_fp32_dest_acc_en>`. Two algorithm variants: `_sfpu_binary_power_21f_` for BF16 and `_sfpu_binary_power_f32_` for FP32.

2. **Query**: "How is the SFPU power (pow) function implemented in the LLK layer? What is the call chain from llk_math through to ckernel_sfpu_power? What SFPU instructions does it use?"
   **Reason**: Understand the LLK-level dispatch mechanism and identify the SFPU instructions.
   **Key Findings**: The params function handles face iteration with `TTI_SETRWC` for DEST pointer advancement. The core SFPU functions use `exexp`, `setexp`, `setsgn`, `addexp`, `int32_to_float`, `float_to_int16`, `float_to_fp16b`, and standard arithmetic (SFPMAD) instructions.

3. **Query**: "What are the SFPU instructions: SFPSETSGN, SFPSETEXP, SFPEXEXP, SFPEXMAN, SFPDIVP2 (addexp), SFPIADD, SFPSHFT, SFPABS, SFPSTOCHRND?"
   **Reason**: Get precise ISA-level documentation for each SFPU instruction used in the power kernel.
   **Key Findings**: Confirmed opcodes and operand formats for all instructions. SFPSETSGN (0x89) sets sign bits, SFPSETEXP (0x82) sets exponent fields, SFPEXEXP (0x77) extracts exponents, SFPEXMAN (0x78) extracts mantissa, SFPDIVP2 (0x76) adds to exponent, SFPSHFT (0x7A) performs bit shifts, SFPABS (0x7D) computes absolute value, SFPSTOCHRND (0x85) performs rounding mode conversions.

4. **Query**: "How do dst_reg, v_if/v_endif, setsgn, setexp, exexp, exman9, addexp, float_to_int16, int32_to_float, float_to_fp16b, reinterpret, abs, shft, vConstFloatPrgm0/1/2, and vConst1 map to SFPU instructions?"
   **Reason**: Understand the SFPI-to-hardware mapping for every intrinsic used in the power kernel.
   **Key Findings**: Each SFPI function maps to exactly one SFPU instruction via GCC builtins. `dst_reg` accesses map to SFPLOAD/SFPSTORE. `v_if/v_endif` use condition code stack (SFPSETCC/SFPPUSHC/SFPPOPC). LRegs are 8 local vector registers; register pressure is a real concern for the power kernel.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instruction details were sufficiently covered by DeepWiki queries to `tenstorrent/tt-isa-documentation` and `tenstorrent/sfpi`.

### Glean References
No Glean references were needed for this analysis.
