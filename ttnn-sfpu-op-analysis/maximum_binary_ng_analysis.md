# MAXIMUM (binary_ng) Implementation Analysis

## Overview

The MAXIMUM operation computes the element-wise maximum of two input tensors: `c[i] = max(a[i], b[i])`. It is implemented as an SFPU-only binary operation within the `binary_ng` (next-generation binary) framework. The `binary_ng` framework is a unified program factory that handles all binary element-wise operations (ADD, SUB, MUL, MAXIMUM, MINIMUM, etc.) through compile-time define substitution, selecting the appropriate LLK function per operation type.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

MAXIMUM is exclusively an SFPU operation (`is_sfpu = true`). It cannot run on the FPU path -- attempting to do so raises a fatal error. It supports three data type variants:
- **float** (BFLOAT16/FLOAT32): calls `binary_max_tile` (generic floating-point max)
- **INT32**: calls `binary_max_int32_tile`
- **UINT32**: calls `binary_max_uint32_tile`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Single tile per read-compute-write cycle (`num_tiles_per_cycle = 1`) |

Each work unit is one output tile. The compute kernel processes exactly 1 tile per iteration: it copies one LHS tile and one RHS tile into DST registers, executes `BINARY_SFPU_OP` (which resolves to `binary_max_tile`), and packs the result.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|----------------|
| **Logical shape** | Up to rank 6+ (collapsed for >5D) | Up to rank 6+ (collapsed for >5D) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims used) | [..., D, N, C, H, W] (last 5 dims used) |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT/WIDTH/BLOCK) | INTERLEAVED or SHARDED (HEIGHT/WIDTH/BLOCK) |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as A (matched) |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-expanded shape of A and B |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input dtype |

### Layout Transformations

No tilize/untilize or format conversions are performed within the operation itself. Both inputs must already be in TILE_LAYOUT. When a typecast is needed (input dtype differs from output dtype), a post-activation `TYPECAST` is appended automatically by the program factory, but this is not specific to MAXIMUM.

### Scalar Variant

When one operand is a scalar (not a tensor), the writer kernel (`writer_interleaved_scalar.cpp`) fills a single tile in CB c_1 with the scalar value. The compute kernel then reuses this single tile for every output tile. The scalar is packed via `pack_scalar_runtime_arg()` which handles BFLOAT16, FLOAT32, INT32, and UINT32 packing.

## Data Flow Pattern

### Two-Tensor Path (both A and B are tensors)

The reader kernel reads BOTH inputs (A into CB c_0, B into CB c_1). The writer kernel ONLY writes the output from CB c_2.

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (tensor A) | CB c_0 | reserve_back(c_0, 1), noc_async_read_page, push_back(c_0, 1) |
| 1 | Reader | DRAM/L1 (tensor B) | CB c_1 | reserve_back(c_1, 1), noc_async_read_page, push_back(c_1, 1) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front(c_0, 1), wait_front(c_1, 1), reserve_back(c_2, 1), copy_tile to DST, binary_max_tile, pack_tile, push_back(c_2, 1), pop_front(c_0, 1), pop_front(c_1, 1) |
| 3 | Writer | CB c_2 | DRAM/L1 (output) | wait_front(c_2, 1), noc_async_write_page, pop_front(c_2, 1) |

### Scalar Path (B is a scalar)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Writer | (scalar arg) | CB c_1 | reserve_back(c_1, 1), fill_with_val, push_back(c_1, 1) -- once |
| 1b | Reader | DRAM/L1 (tensor A) | CB c_0 | reserve_back(c_0, 1), noc_async_read_page, push_back(c_0, 1) -- per tile |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | Same as above but c_1 is never popped (scalar persists) |
| 3 | Writer | CB c_2 | DRAM/L1 (output) | wait_front(c_2, 1), noc_async_write_page, pop_front(c_2, 1) |

**Key naming caveat**: In the two-tensor path, the "reader" kernel (`ReaderNoBcastNg` / `reader_interleaved_no_bcast.cpp` in `kernels_ng/`) reads both A and B. The "writer" kernel (`WriterNoBcastNg` / `writer_interleaved_no_bcast.cpp` in `kernels_ng/`) only writes C. In the scalar path, the "writer" kernel (`WriterScalar` / `writer_interleaved_scalar.cpp` in `kernels/`) fills CB c_1 with the scalar AND writes C.

## Circular Buffer Configuration

### Base Configuration (no broadcast, no activations, interleaved)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (tensor) or 1 tile (scalar) | 1 tile | Double/Single | Reader (tensor) or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

### Optional CBs (activated by pre/post processing)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_3 | cb_lhs_intermediate | LHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |
| c_4 | cb_rhs_intermediate | RHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |
| c_5 | cb_row_bcast_a | Row broadcast buffer for A | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_6 | cb_row_bcast_b | Row broadcast buffer for B | 2 tiles | 1 tile | Double | Reader | Compute | Program |

For MAXIMUM without pre/post activations, only c_0, c_1, and c_2 are used. CBs c_3 and c_4 are only created if `PROCESS_LHS_ACTIVATIONS` or `PROCESS_RHS_ACTIVATIONS` defines are non-empty (which they are not for plain MAXIMUM).

### Sharded Configuration

When tensors are sharded, CBs c_0, c_1, and c_2 are backed by the tensor's L1 buffer directly. Their capacity equals the shard volume in tiles rather than 2. The buffer pointer is set via `UpdateDynamicCircularBufferAddress`.

## Pipeline Pattern Summary

- **c_0 (Input A)**: Double-buffered (capacity=2, block=1) -- allows overlap of reader read with compute consumption
- **c_1 (Input B)**: Double-buffered when B is a tensor (capacity=2, block=1); Single-buffered when B is a scalar (capacity=1, block=1, filled once)
- **c_2 (Output)**: Double-buffered (capacity=2, block=1) -- allows overlap of compute production with writer write

Double-buffering on all three primary CBs enables pipelining: while the reader fills one tile slot, compute can process from the other slot, and similarly for the compute-to-writer handoff.

## Index Calculations

### Tile Index Decomposition

The reader and writer kernels decompose a linear `start_tile_id` into multi-dimensional indices for the 5D+ tensor shape `[ND, D, N, C, Ht, Wt]`:

```
tiles_per_n   = C * Ht * Wt
tiles_per_d   = N * tiles_per_n
tiles_per_nd  = D * tiles_per_d

start_nd = start_tile_id / tiles_per_nd
start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
start_n  = (start_tile_id % tiles_per_d) / tiles_per_n
start_c  = (start_tile_id % tiles_per_n) / (Ht * Wt)
start_th = (start_tile_id % (Ht * Wt)) / Wt
start_tw = start_tile_id % Wt
```

### Broadcasting Stride Mechanism

For each input tensor, "strides" are computed in the host program factory to handle broadcasting:

```cpp
nD_stride = aHt * aWt * aC * aN * aD * (aND > 1)
d_stride  = aHt * aWt * aC * aN * (aD > 1)
n_stride  = aHt * aWt * aC * (aN > 1)
c_stride  = aHt * aWt * (aC > 1)
```

The `(dim > 1)` factor is the key broadcasting trick: when a dimension is 1 (broadcast), the stride becomes 0, causing the reader to re-read the same data for every iteration along that dimension. When the dimension is >1, the stride advances normally.

The reader accumulates a `tile_offset` and applies shift corrections at each dimension boundary:
```
next_c_shift  = c_stride - Ht*Wt
next_n_shift  = n_stride - c_stride*C
next_d_shift  = d_stride - n_stride*N
next_nd_shift = nD_stride - d_stride*D
```

### TensorAccessor Usage

Both reader and writer use `TensorAccessor` for address computation. Compile-time args encode the tensor's memory layout (bank mapping), and the accessor's `noc_async_read_page` / `noc_async_write_page` functions translate a logical tile index to a physical NoC address accounting for interleaved bank distribution.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Tiles are read one-at-a-time via `noc_async_read_page`. The iteration order is `ND -> D -> N -> C -> Ht -> Wt` (innermost loop on Wt). This is row-major within each tile-row, stepping sequentially through width tiles before advancing to the next height tile. Each read is followed by `noc_async_read_barrier()` (blocking).
- **Sharded**: For sharded inputs, no NoC reads occur at all. The reader simply does `cb_reserve_back` / `cb_push_back` to expose the already-present L1 data to the compute kernel.
- Both A and B are read in lockstep (same iteration, same tile position in the output space), but each applies its own stride-based offset (supporting broadcasting).

### Write Pattern

- **Interleaved**: Tiles are written one-at-a-time via `noc_async_write_page`, same iteration order as reads. Each write is followed by `noc_async_write_barrier()` (blocking).
- **Sharded**: No NoC writes occur; output is already in L1 at the correct shard location.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (device compute grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (remainder group) |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |

### Work Splitting Details

For **interleaved** mode, `tt::tt_metal::split_work_to_cores()` divides the total output tiles across available cores, creating two core groups:
- **core_group_1**: Gets `num_tiles_per_core_group_1` tiles (the larger share)
- **core_group_2**: Gets `num_tiles_per_core_group_2` tiles (the smaller share, may be 0)
- Cores not in either group receive zero-args and return immediately (no-op cores)

For **sharded** mode, the core grid is determined by the shard spec. Each core processes exactly its shard's tiles. The shard shape generator handles edge cores that may have fewer tiles due to uneven division.

### Zero-Start Grid Optimization

When the worker grid is a single rectangular region starting at (0,0) and any sharded tensors also start at (0,0), a fast path is used (`zero_start_grid = true`) that avoids the overhead of generic CoreRangeSet operations.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

**Compile-time defines** (set via `compute_kernel_defines`):
- `BINARY_SFPU_INIT` = `"binary_max_tile_init();"` (or `binary_max_int32_tile_init()` / `binary_max_uint32_tile_init()`)
- `BINARY_SFPU_OP` = `"binary_max_tile"` (or `binary_max_int32_tile` / `binary_max_uint32_tile`)
- `BCAST_INPUT` = `""` (no broadcast) or `"0"/"1"` (which input to broadcast)
- `PROCESS_LHS_ACTIVATIONS(i)` = `""` (empty for MAXIMUM)
- `PROCESS_RHS_ACTIVATIONS(i)` = `""` (empty for MAXIMUM)
- `PROCESS_POST_ACTIVATIONS(i)` = `""` (empty for MAXIMUM, unless typecast needed)

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Memory layout info for tensor A |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Memory layout info for tensor B |
| M+1 | has_sharding | uint32_t | Whether native L1 sharding is active (0 or 1) |

**Compile-time defines**: `SRC_SHARDED`, `SRC_SHARDED_B` (whether inputs are sharded in L1)

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Memory layout info for output tensor |
| N+1 | has_sharding | uint32_t | Whether native L1 sharding is active (0 or 1) |

**Compile-time defines**: `DST_SHARDED`, `SRC_SHARDED` (for scalar variant, whether B/C are sharded)

### Runtime Arguments

#### Reader Kernel (21 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Buffer address of tensor A |
| 1 | start_tile_id | uint32_t | Output tile ID this core starts at (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (sharded only, else 0) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded only, else 0) |
| 5 | nD_stride | uint32_t | A stride for collapsed ND dimensions (0 if broadcast) |
| 6 | d_stride | uint32_t | A stride for D dimension |
| 7 | n_stride | uint32_t | A stride for N dimension |
| 8 | c_stride | uint32_t | A stride for C dimension |
| 9 | D | uint32_t | Output D dimension (tiles) |
| 10 | N | uint32_t | Output N dimension (tiles) |
| 11 | C | uint32_t | Output C dimension (tiles) |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed ND dimension count |
| 15 | src_addr_b | uint32_t | Buffer address of tensor B |
| 16 | nD_stride_b | uint32_t | B stride for collapsed ND dimensions |
| 17 | d_stride_b | uint32_t | B stride for D dimension |
| 18 | n_stride_b | uint32_t | B stride for N dimension |
| 19 | c_stride_b | uint32_t | B stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (sharded only, else 0) |

#### Writer Kernel -- Two-Tensor Path (11 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Buffer address of output tensor C |
| 1 | start_tile_id | uint32_t | Output tile ID this core starts at |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (sharded only) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed ND dimension |
| 10 | (reserved) | uint32_t | Always 0 |

#### Writer Kernel -- Scalar Path (11 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar value packed for the data type |
| 1 | dst_addr | uint32_t | Buffer address of output tensor C |
| 2 | start_tile_id | uint32_t | Output tile ID this core starts at |
| 3 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Collapsed ND dimension |

#### Compute Kernel (4 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of tiles this core processes |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE, Wt for COL, Ht*Wt for SCALAR) |
| 2 | counter | uint32_t | Starting offset within broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Always 0 for MAXIMUM (used by quant ops only) |

## Kernel Implementations

### Compute Kernel: No-Broadcast SFPU

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu_no_bcast.cpp | RISCV_2 (Math) | N/A | CB c_0 (LHS), CB c_1 (RHS) | CB c_2 (output) | copy_tile to DST, binary_max_tile, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: For each tile: waits for LHS and RHS, acquires DST registers, copies LHS to even DST slot (i*2) and RHS to odd DST slot (i*2+1), calls `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which resolves to `binary_max_tile(0, 1, 0)` writing the result to DST slot 0, then packs to output CB. Since `num_tiles_per_cycle=1`, there is no inner batching.
- **Used when**: `SubtileBroadcastType::NONE` (both tensors have matching tile dimensions)

### Compute Kernel: Broadcast SFPU

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu.cpp | RISCV_2 (Math) | N/A | CB c_0 (LHS), CB c_1 (RHS) | CB c_2 (output) | copy_tile to DST, binary_max_tile, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp`
- **Key Logic**: Handles scalar and column broadcast cases. The broadcast input is loaded once and held while iterating over the other input `freq` times. Uses `tile_freq` and `tile_start` runtime args to manage the broadcast pattern.
- **Used when**: `SubtileBroadcastType::SCALAR_A/B` or `COL_A/B` or `ROW_A_COL_B`/`ROW_B_COL_A`

### Compute Kernel: Scalar SFPU

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu_scalar.cpp | RISCV_2 (Math) | N/A | CB c_0 (LHS), CB c_1 (scalar) | CB c_2 (output) | copy_tile to DST, binary_max_tile, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: The RHS scalar tile is loaded once (waited for before the loop) and never popped until all tiles are done. Each iteration only waits/pops the LHS tile.
- **Used when**: B is passed as a scalar value (not a tensor)

### Reader Kernel: No-Broadcast NG

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_interleaved_no_bcast.cpp (kernels_ng/) | RISCV_0 (Brisc) | NOC0 | DRAM (A and B) | CB c_0, CB c_1 | noc_async_read_page per tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads both A and B in a single 6-deep nested loop (ND, D, N, C, Ht, Wt). For each output tile position, reads one tile from A and one from B using their respective stride-based offsets. Handles broadcasting via the stride mechanism (stride=0 when dimension=1).
- **Used when**: Two-tensor path with `SubtileBroadcastType::NONE`

### Reader Kernel: Legacy (scalar path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_interleaved_no_bcast.cpp (kernels/) | RISCV_0 (Brisc) | NOC0 | DRAM (A only) | CB c_0 | noc_async_read_page per tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Only reads tensor A. Same nested loop structure but only one tensor.
- **Used when**: Scalar path (B is a scalar, not a tensor)

### Writer Kernel: No-Broadcast NG

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_interleaved_no_bcast.cpp (kernels_ng/) | RISCV_1 (Ncrisc) | NOC1 | CB c_2 | DRAM (C) | noc_async_write_page per tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles using the same nested loop decomposition. Only writes (no reading of inputs).
- **Used when**: Two-tensor path

### Writer Kernel: Scalar

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_interleaved_scalar.cpp | RISCV_1 (Ncrisc) | NOC1 | CB c_2 | DRAM (C) + fills CB c_1 | fill_with_val + noc_async_write_page |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills one tile in CB c_1 with the packed scalar value, then writes output tiles from CB c_2. The scalar fill uses `fill_with_val` (for INT32/UINT32) or `fill_with_val<float>` (for FLOAT32) or `fill_with_val_bfloat16` (for BFLOAT16).
- **Used when**: B is a scalar value

## Implementation Notes

### SFPU-Only Constraint
MAXIMUM is strictly an SFPU operation. The `OpConfig` constructor maps `BinaryOpType::MAXIMUM` to `SfpuBinaryOp::MAXIMUM` only when `is_sfpu_op()` returns true. The FPU path throws `TT_THROW("Unsupported binary op for FPU")`.

### UnpackToDestMode
For SFPU operations (including MAXIMUM, except POWER), all source CBs (c_0, c_1, c_3, c_4) are configured with `UnpackToDestMode::UnpackToDestFp32`. This ensures the unpack stage converts data to FP32 in the DST register before the SFPU operates on it, providing maximum precision.

### fp32_dest_acc_en
This flag is enabled when output or both inputs are FLOAT32, INT32, or UINT32. It configures the DST accumulator to use 32-bit precision.

### No Pre/Post Activations for Plain MAXIMUM
MAXIMUM has no `process_lhs`, `process_rhs`, or `postprocess` unary operations. The `OpConfig` for MAXIMUM simply sets `binary_op = SfpuBinaryOp::MAXIMUM` with no additional activation functions. This means CBs c_3 and c_4 are never created, and the `PREPROCESS` macros compile to no-ops.

### High-Rank Tensor Support
Tensors with rank > 5 have their higher dimensions collapsed into a single `ND` dimension via `extract_nD_dims()`. The 5 innermost dimensions (D, N, C, Ht, Wt) are extracted from the padded shape and used for the nested iteration loops.

### Sharding Constraints
Native L1 sharding (where data stays in L1 without NoC transfers) is only used when:
1. The output shard spec is even (no remainder tiles)
2. Both inputs have the same shape and same memory config
3. No DRAM buffers are involved
4. All shard grids are identical
If any constraint fails, the operation falls back to the interleaved (tensor accessor) path.

### Program Caching
The `override_runtime_arguments` method enables program caching: once the program is compiled, subsequent calls with different tensors (same shapes/layouts) only update the buffer addresses and runtime args without recompilation.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng operation work in TTNN? What is its program factory structure, what kernels does it use, and how does it handle different subtypes like MAXIMUM?"
   **Reason**: Initial architectural understanding of the binary_ng framework
   **Key Findings**: Confirmed MAXIMUM maps to `SfpuBinaryOp::MAXIMUM`, the operation uses reader/compute/writer kernel triplet, different kernel variants exist for broadcast types (no-bcast, scalar, row, col, mixed), and the `BinaryNgKernelConfig` constructor selects kernels based on `SubtileBroadcastType`.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` and `.cpp`
   **Reason**: Understanding the OpConfig mapping and kernel file path resolution
   **Key Information**: MAXIMUM maps to `binary_max_tile` / `binary_max_int32_tile` / `binary_max_uint32_tile` depending on dtype. Kernel paths are resolved via `get_kernel_file_path()` using the `KernelName` enum.

2. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: Understanding the LLK-level SFPU function signatures
   **Key Information**: `binary_max_tile(idst0, idst1, odst)` performs element-wise max on DST register tiles. Three variants exist for float, int32, and uint32. The function delegates to `llk_math_eltwise_binary_sfpu_binary_max`.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`
   **Reason**: Understanding `SubtileBroadcastType` enum values
   **Key Information**: Nine broadcast types covering all combinations of scalar, row, column, and mixed broadcasting for both inputs.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. The MAXIMUM operation has three variants (float, int32, uint32) which share a common ckernel implementation parameterized by template arguments.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/binary_max_min.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `binary_max_tile(idst0, idst1, odst)` (defined in the API header `binary_max_min.h`), which is gated by the `MATH(...)` macro so it only runs on the math RISC-V processor.
2. Inside the `MATH` gate, it calls `llk_math_eltwise_binary_sfpu_binary_max<APPROX>(idst0, idst1, odst, vector_mode)` in the LLK dispatch layer (`llk_math_eltwise_binary_sfpu_max_min.h`).
3. That function calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_binary_max_min<true>, ...)` in the parameters dispatch layer (`llk_math_eltwise_binary_sfpu_params.h`), which handles stall/sync, iterates over tile faces (4 faces for `VectorMode::RC`), and invokes the SFPU function once per face.
4. Each face invocation calls `calculate_binary_max_min<IS_MAX_OP=true, ITERATIONS=8>(dst_index_in0, dst_index_in1, dst_index_out)` in the core SFPU implementation (`ckernel_sfpu_binary_max_min.h`), which issues the actual SFPU load/swap/store instructions.

For the init path: `binary_max_tile_init()` calls `llk_math_eltwise_binary_sfpu_binary_max_init<APPROX>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::max, APPROXIMATE>(sfpu::binary_max_min_init<true>)`. This initializes the SFPU config register, configures address modes (ADDR_MOD_7 and ADDR_MOD_6), resets counters, and -- when SFPLOADMACRO is available -- programs the instruction templates and macro descriptors for the pipelined execution.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h
// NOTE: Wormhole and Blackhole implementations are identical except for ADDR_MOD indices
//       (WH uses ADDR_MOD_3/ADDR_MOD_2; BH uses ADDR_MOD_7/ADDR_MOD_6).
//       The Blackhole version is shown here (ADDR_MOD_7/ADDR_MOD_6).

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true selects maximum; ITERATIONS=8 processes 8 rows per face (one 16x16 face)
    uint offset0 = (dst_index_in0 * 32) << 1; // byte offset into DST for input tile 0
    uint offset1 = (dst_index_in1 * 32) << 1; // byte offset into DST for input tile 1
    uint offset2 = (dst_index_out * 32) << 1; // byte offset into DST for output tile

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // mod1=1: VD=min, VC=max
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2); // ADDR_MOD_6 has dest.incr=2
    }
#else
    // Implementation notes, see the original file for more details
    // SFPLOADMACRO achieves 3 cycles per input row throughput by pipelining
    // load, swap_minmax, round (L16 conversion), and store across SFPU stages.

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2)); // Macro 0: load + scheduled swap
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1); // regular load of second operand
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2)); // Macro 1: scheduled store
    }

    TTI_SFPNOP; // pipeline drain: 3 NOPs to flush the SFPLOADMACRO pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true, IS_UNSIGNED=false for INT32; IS_UNSIGNED=true for UINT32
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0 (or reversed if unsigned)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // mod1=9 for unsigned: inverted min/max polarity; mod1=1 for signed

        // Conditionally swap again to fix the cases where SFPSWAP got the result backwards
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0); // set CC if LREG0 < 0 (signed) or >= 0 (unsigned)
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0); // set CC if LREG1 < 0 (signed) or >= 0 (unsigned)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP); // mod1=0: conditional swap based on CC
        TTI_SFPENCC(0, 0, 0, 0); // disable condition codes (clear enable+result)

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
    }
#else
    // Implementation notes, see the original file for more details
    // SFPLOADMACRO achieves 5 cycles per input row for int32 (more complex due to
    // conditional swap correction for cross-sign-boundary comparisons).

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    load_replay_buf(0, 10, [offset0, offset1, offset2] {
        // first iteration, with a0, b0, c
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b0 >> 2));
        TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));

        // second iteration, with a1, b1, c
        TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));
    });

#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10); // replay 10-instruction sequence for 2 rows per iteration
    }

    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);  // handle odd iteration count
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;          // pipeline drain
        TTI_SFPNOP;
        lltt::replay(2, 2);  // drain remaining SETCC/ENCC
    }

    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
    // Only active when SFPLOADMACRO is available (not DISABLE_SFPLOADMACRO)
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP with min/max mode
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // mod1=9 for max: VD=max,VC=min; mod1=1 for min: VD=min,VC=max

    // InstructionTemplate[1]: SFPSHFT2 with immediate shift (used for L16 round/convert)
    TTI_SFPSHFT2(0, 0, 13, 6); // SFPSHFT2_MOD1_SHFT_IMM=6

    // Macro 0: Load input A, execute swap template, round result
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;   // enable=1, template_idx=1, wait_cycles=4
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;    // enable=1, is_fp16=1, template_idx=3, wait_cycles=5
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0); // program Macro 0
    }

    // Macro 1: Store result from previous iteration
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;    // is_fp16=1, template_idx=2, wait_cycles=3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0); // program Macro 1
    }

    // Misc config: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1}, UnitDelayKind={1,1}
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
    // IS_MAX_OP=true, IS_UNSIGNED=false for signed int32 max
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPSWAP(
        0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // XOR to invert polarity for unsigned

    // InstructionTemplate[1]
    TTI_SFPSWAP(
        0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[2]: SFPSETCC for sign-correction
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]: SFPSHFT2 for L16 round/convert
    TTI_SFPSHFT2(0, 0, 15, 6); // SFPSHFT2_MOD1_SHFT_IMM=6

    // Macros 0-3: configure 4 macro descriptors for the 5-cycle pipelined int32 path
    // (Macro 0: swap + round for pair a0/b0)
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // (Macro 1: swap + round for pair a1/b1)
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // (Macro 2: setcc for sign correction)
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // (Macro 3: store result)
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc config: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1,1,1}, UnitDelayKind={1,1,1,1}
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` (via `TT_SFPLOAD`) | Loads a row of data from DST register file into an SFPU local register (LREG). Used to bring input tile rows from DST into LREG0/LREG1 for comparison. The `InstrModLoadStore` parameter selects DEFAULT (float) or INT32 mode. |
| `SFPSWAP` (via `TTI_SFPSWAP`) | Core min/max instruction. With `mod1=SFPSWAP_MOD1_VEC_MIN_MAX` (1): performs element-wise comparison and places min in VD, max in VC. With `mod1=9` (1\|8, inverted polarity): places max in VD, min in VC. With `mod1=SFPSWAP_MOD1_SWAP` (0): conditional swap based on the current condition code (CC). Latency: 2 cycles. |
| `SFPSTORE` (via `TT_SFPSTORE`) | Stores a row of data from an SFPU local register back to the DST register file. Selects LREG1 for max result or LREG0 for min result based on `IS_MAX_OP`. |
| `SFPSETCC` (via `TTI_SFPSETCC`) | Sets the per-lane condition code based on an LREG value. With `SFPSETCC_MOD1_LREG_LT0` (0): CC is set where LREG < 0 (used for signed int32). With `SFPSETCC_MOD1_LREG_GTE0` (4): CC is set where LREG >= 0 (used for unsigned uint32). Used in the int32 path to correct SFPSWAP results for cross-sign comparisons. |
| `SFPENCC` (via `TTI_SFPENCC`) | Disables (clears) the condition code enable and result flags. Called after the conditional swap to return to unconditional execution. With all args 0: clears both enable and result bits. |
| `SFPLOADMACRO` (via `TT_SFPLOADMACRO`) | Macro-scheduled load that combines a load with pre-programmed operations (swap, round, store) across multiple SFPU pipeline stages. Achieves 3 cycles/row (float) or 5 cycles/row (int32) throughput by overlapping operations across the Load, Simple, MAD, Round, and Store stages. |
| `SFPNOP` (via `TTI_SFPNOP`) | No-operation used to drain the SFPU pipeline after the last SFPLOADMACRO iteration. 3 NOPs are needed for float, varying count for int32. |
| `SFPLOADI` (via `TTI_SFPLOADI`) | Loads an immediate value into LREG0. Used during init to program SFPLOADMACRO descriptor bit fields (simple_bits, mad_bits, round_bits, store_bits). |
| `SFPCONFIG` (via `TTI_SFPCONFIG`) | Programs SFPLOADMACRO configuration: instruction templates (slots 0-3), macro descriptors (slots 4-7), and misc settings (slot 8). |
| `SFPSHFT2` (via `TTI_SFPSHFT2`) | Shift instruction used as an instruction template for the L16 round/convert step in the SFPLOADMACRO pipeline. `SFPSHFT2_MOD1_SHFT_IMM` (6) selects immediate-value shift mode. |
| `SETRWC` (via `TTI_SETRWC`) | Sets the read/write counters. Used in the params dispatch layer to advance the DST face pointer by 16 rows (2 increments of 8) between faces. Not an SFPU instruction per se, but critical for face iteration. |

### SFPU Register Usage

**Local Registers (LREGs) -- Float Path:**
- `LREG0`: Input operand A row (alternates with LREG1 in SFPLOADMACRO path)
- `LREG1`: Input operand B row (alternates with LREG0 in SFPLOADMACRO path). After SFPSWAP with VEC_MIN_MAX, LREG1 holds the max values (VC register).
- `LREG2` (`b`): Used in SFPLOADMACRO path for the second operand load
- `LREG3` (`c`): Used in SFPLOADMACRO path for the store target

**Local Registers (LREGs) -- Int32 Path:**
- `LREG0` (`a0`): First input A row
- `LREG1` (`b0`): First input B row
- `LREG2` (`a1`): Second input A row (double-buffered with a0)
- `LREG3` (`b1`): Second input B row (double-buffered with b0)
- `LREG7` (`c`): Store target register

**DST Register File:**
- `DST[idst0*32 .. idst0*32+31]`: Input tile A (loaded by copy_tile before SFPU runs)
- `DST[idst1*32 .. idst1*32+31]`: Input tile B (loaded by copy_tile before SFPU runs)
- `DST[odst*32 .. odst*32+31]`: Output tile (written back by SFPSTORE). In practice `odst == idst0`, so the result overwrites input A in DST.

**Condition Code Register (CC):**
- Used only in the int32 path. Per-lane CC bits track whether a correction swap is needed for cross-sign-boundary comparisons (e.g., comparing a positive and negative int32 where SFPSWAP's float-based comparison gives wrong results).

### Address Mode Configuration

Two address modes are configured for the binary max/min operation:

**ADDR_MOD_7** (Wormhole: ADDR_MOD_3):
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 0`
- Purpose: Used for SFPLOAD instructions. No auto-increment -- the offset is managed explicitly by the loop and the `(dst_index * 32) << 1` calculation.

**ADDR_MOD_6** (Wormhole: ADDR_MOD_2):
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 2`
- Purpose: Used for SFPSTORE instructions. The `dest.incr = 2` auto-increments the DST write address by 2 rows after each store, allowing consecutive rows to be written without explicit offset updates. This is because each SFPU instruction processes a row of data, and in the binary case, the two input tiles occupy interleaved DST slots.

Both Wormhole B0 and Blackhole use the same address mode field values. The only difference is the ADDR_MOD slot number (Wormhole uses ADDR_MOD_3/ADDR_MOD_2; Blackhole uses ADDR_MOD_7/ADDR_MOD_6). This avoids conflicts with other operations that may use ADDR_MOD_0 and ADDR_MOD_2 (noted in the code comment about A2D coexistence).

The address modes are set in `eltwise_binary_sfpu_configure_addrmod<SfpuType::max>()`, called from `_llk_math_eltwise_binary_sfpu_init_<SfpuType::max>()`. The same modes are used for `SfpuType::min`, `max_int32`, `min_int32`, `max_uint32`, `min_uint32`, and `mul_int32`/`mul_uint16`.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng SFPU compute kernel work for operations like maximum?"
   **Reason**: Understanding the full call chain from compute kernel through LLK to SFPU implementation
   **Key Findings**: Confirmed the dispatch chain: `BINARY_SFPU_OP` macro -> LLK `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary`. Confirmed SFPSWAP with `SFPSWAP_MOD1_VEC_MIN_MAX` is the core instruction, and SFPLOADMACRO optimization achieves 3 cycles/row for float.

2. **Query**: "How is the SFPU maximum operation implemented in tt-llk?"
   **Reason**: Locating the exact ckernel files and understanding the LLK-level function signatures
   **Key Findings**: Confirmed `ckernel_sfpu_binary_max_min.h` as the core implementation file. Identified the `calculate_binary_max_min` and `calculate_binary_max_min_int32` functions. Confirmed address mode configuration with `ADDR_MOD_6` having `dest.incr=2`.

3. **Query**: "What does the SFPSWAP instruction do, specifically SFPSWAP_MOD1_VEC_MIN_MAX and mod1=9?"
   **Reason**: Understanding the hardware semantics of the core min/max instruction
   **Key Findings**: SFPSWAP performs element-wise min/max comparison between two LREGs. Default VEC_MIN_MAX (mod1=1) puts min in VD, max in VC. The mod1=9 value (1|8) inverts the polarity so VD=max, VC=min. Latency is 2 cycles. Confirmed from `sfpi_constants.h` that `SFPSWAP_MOD1_VEC_MIN_MAX=1`, `SFPSWAP_MOD1_SWAP=0`.

### Confluence References
- Attempted to retrieve the Tensix SFPU Instruction Set Architecture page (page ID `1170505767`) for detailed SFPSWAP documentation. The page was not available due to a transport endpoint migration notice. SFPSWAP semantics were instead confirmed from `runtime/sfpi/include/sfpi_constants.h` constant definitions and code comments in the ckernel source.

### Glean References
- Not used. Sufficient detail was obtained from source code and DeepWiki.
