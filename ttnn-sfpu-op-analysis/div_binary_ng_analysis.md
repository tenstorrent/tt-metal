# DIV (binary_ng) Implementation Analysis

## Overview

The DIV operation performs element-wise division `c = a / b` using the `binary_ng` (next-generation binary) framework. It supports two execution paths depending on data types and configuration:

1. **SFPU path** (default for most dtypes): Uses `SfpuBinaryOp::DIV` which calls `div_binary_tile` (float) or `div_int32_tile` (INT32) on the SFPU vector unit.
2. **FPU path** (non-SFPU mode): Decomposes into `RECIP(b)` followed by `MUL(a, reciprocal_b)` on the FPU matrix unit.

The operation supports tensor-tensor division, tensor-scalar division, and broadcasting across all dimensions (row, column, scalar, mixed).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` -- total output tiles |
| **Loop structure** | Per-core loop over assigned tiles; nested 6-deep loops (nD, D, N, C, Ht, Wt) for index mapping with broadcasting strides |

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input Tensor A | Input Tensor B (optional) |
|----------|---------------|--------------------------|
| **Logical shape** | Arbitrary rank (up to 6+) | Arbitrary rank (broadcastable to A) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims) | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT, WIDTH, BLOCK) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32 | BFLOAT16, FLOAT32, INT32 (or scalar float) |

When `b` is absent (scalar mode), the scalar value is packed into a runtime argument and written into a single tile by the writer kernel.

### Output Tensor(s)

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-compatible output of A and B shapes |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (BFLOAT16/FLOAT32); INT32 inputs produce FLOAT32 output for DIV |

### Layout Transformations

- No explicit tilize/untilize in the kernels; all data is expected in TILE_LAYOUT.
- For INT32 DIV, the output dtype is automatically converted to FLOAT32 (handled at the device operation level, not in the program factory).
- When input and output dtypes differ and the operation is not integer division, a TYPECAST post-activation is appended.

## Data Flow Pattern

### Path 1: Tensor-Tensor DIV (b is a tensor)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (ng) | DRAM/L1 (A and B) | CB c_0 (A), CB c_1 (B) | reserve_back, noc_async_read_page, push_back (per tile, for both A and B) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front, copy_tile to DST regs, SFPU div_binary_tile or FPU RECIP+MUL, pack_tile, push_back, pop_front |
| 3 | Writer (ng) | CB c_2 | DRAM/L1 (C) | wait_front, noc_async_write_page, pop_front |

The reader kernel reads both A and B tiles (using the `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` reader which handles both inputs). The writer uses `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`.

### Path 2: Tensor-Scalar DIV (b is a scalar)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Writer (scalar) | Scalar arg | CB c_1 | Fills one tile with scalar value (once), push_back |
| 1b | Reader (old) | DRAM/L1 (A only) | CB c_0 | reserve_back, noc_async_read_page, push_back (per tile) |
| 2 | Compute (scalar) | CB c_0, CB c_1 | CB c_2 | wait_front on RHS once (scalar stays), per-tile: wait_front LHS, compute, pack, push_back, pop_front LHS |
| 3 | Writer (scalar) | CB c_2 | DRAM/L1 (C) | wait_front, noc_async_write_page, pop_front |

In scalar mode, the writer kernel (`writer_interleaved_scalar.cpp`) both fills the scalar tile into CB c_1 and writes output tiles from CB c_2. The reader only reads input A.

### Sharded Path

When native L1 sharding is active, the reader does `cb_reserve_back` + `cb_push_back` for the entire shard (no NoC reads needed since data is already in L1). The writer similarly skips NoC writes for sharded output. CB addresses are updated dynamically via `UpdateDynamicCircularBufferAddress`.

## Circular Buffer Configuration

### Non-sharded (Interleaved) Mode

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src_b | Input B tiles (or scalar tile) | 2 tiles (tensor) / 1 tile (scalar) | 1 tile | Double / Single | Reader or Writer(scalar) | Compute | Block / Program (scalar stays) |
| c_2 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_lhs_interim | LHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |
| c_4 | cb_rhs_interim | RHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |

Notes:
- CB c_3 is only created when LHS pre-activations are non-empty (not the case for standard DIV).
- CB c_4 is only created when RHS pre-activations are non-empty. For FPU DIV, `process_rhs = RECIP` so c_4 IS created with capacity 1.
- CBs c_5 and c_6 are created for ROW_A/ROW_B broadcast patterns (capacity 2 tiles each) but are not relevant for the standard no-broadcast DIV path.

### Sharded Mode

In sharded mode, CB capacities are set to the shard volume (number of tiles per shard) instead of 2. The CB is backed directly by the sharded buffer in L1.

## Pipeline Pattern Summary

- **Non-sharded**: CB c_0 and c_1 have capacity=2, block_size=1 -- **Double-buffered**. This allows the reader to write the next tile while compute processes the current one. CB c_2 is also double-buffered for the same overlap between compute and writer.
- **Scalar mode**: CB c_1 has capacity=1 -- **Single-buffered** (scalar tile written once and consumed repeatedly without being popped until the end).
- **Sharded mode**: CBs are sized to the full shard -- effectively **Single-buffered** at the shard level (entire shard is available at once).

## Index Calculations

The reader kernel maps a linear `start_tile_id` into a 6D coordinate system `(nD, D, N, C, Ht, Wt)` using modular arithmetic:

```
tiles_per_nd = D * N * C * Ht * Wt
start_nd = start_tile_id / tiles_per_nd
offset_nd = start_tile_id % tiles_per_nd
...
start_tw = offset_c % Wt
```

For broadcasting, each input tensor has its own stride pattern computed in the program factory:
- `nD_stride = Ht * Wt * C * N * D * (nD > 1)` -- zero if dimension is 1 (broadcast)
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

The `(dim > 1)` multiplier causes the stride to be zero when a dimension has size 1, effectively broadcasting along that dimension. The reader uses stride-based offsets to walk through input tiles while the output iterates linearly through the full output shape.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads within each tile-row (Wt dimension), then advancing through Ht, C, N, D, nD. Each tile is read individually via `noc_async_read_page` with a barrier after each tile (or after each A+B pair in the ng reader).
- **Sharded**: No explicit reads -- data is already in L1. The CB is made to point to the shard buffer.
- For broadcasting, the input tile offsets use stride-based indexing that may revisit the same tiles (stride=0 for broadcast dimensions).

### Write Pattern
- **Interleaved**: Sequential tile writes using `noc_async_write_page` with per-tile barriers. Output tile offset grows linearly from `start_tile_id`.
- **Sharded**: No explicit writes -- output is already in L1 shard buffer.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (rectangular grid) |
| **Grid dimensions** | Device-dependent (from `operation_attributes.worker_grid`) |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start grid) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (remainder group) |
| **Load balancing** | Two-group split: `split_work_to_cores` divides total output tiles across cores. Group 1 gets `ceil(total/cores)` tiles, Group 2 gets `floor(total/cores)` tiles. Cores outside both groups get zero-filled args (no-op). |

For sharded mode, work distribution follows the shard grid directly -- each core processes its own shard.

## Arguments

### Compile-Time Arguments

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

**Reader Kernel (compile-time via TensorAccessorArgs):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | tensor_accessor_a | varies | TensorAccessor compile-time args for buffer A |
| N+1..M | tensor_accessor_b | varies | TensorAccessor compile-time args for buffer B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer Kernel (compile-time via TensorAccessorArgs):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | tensor_accessor_c | varies | TensorAccessor compile-time args for buffer C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Reader Kernel (21 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core (c_start_id) |
| 2 | src_num_tiles (a) | uint32_t | Number of A tiles in shard (sharded mode) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded mode, 0 otherwise) |
| 5 | nD_stride | uint32_t | A's stride for collapsed nD dimension |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed nD dimension of output |
| 15 | src_addr_b | uint32_t | Input B buffer address (0 if scalar) |
| 16 | nD_stride_b | uint32_t | B's stride for nD dimension |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard |

**Writer Kernel (11 args, tensor-tensor mode):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed nD dimension |
| 10 | (unused) | uint32_t | Set to 0 |

**Writer Kernel (11 args, scalar mode):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar value packed as bfloat16x2 or float bits |
| 1 | dst_addr | uint32_t | Output C buffer address |
| 2 | start_tile_id | uint32_t | Starting output tile ID |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Shard width in tiles |
| 5-10 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |

**Compute Kernel (4 args):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for no-bcast, Wt for col-bcast, Ht*Wt for scalar-bcast) |
| 2 | counter | uint32_t | Starting broadcast counter position |
| 3 | compute_scalar_value | uint32_t | Reserved for quantization zero-point (0 for DIV) |

## Kernel Implementations

### SFPU Compute Kernel (Tensor-Tensor, No Broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (compute) | N/A | CB c_0 (A), CB c_1 (B) | CB c_2 (C) | copy_tile A to DST[0], copy_tile B to DST[1], div_binary_tile(0,1,0), pack_tile(0, c_2) |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: For each output tile: wait for both LHS and RHS tiles, acquire tile registers, copy LHS to DST register slot 0, copy RHS to DST register slot 1, execute `BINARY_SFPU_OP(0, 1, 0)` which expands to `div_binary_tile(0, 1, 0)`, optionally run post-activations, commit registers, pack result to output CB. The `copy_tile_to_dst_init_short_with_dt` calls handle data format reconfiguration between the two inputs.

### FPU Compute Kernel (Tensor-Tensor, No Broadcast)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (compute) | N/A | CB c_0 (A), CB c_4 (RECIP(B)) | CB c_2 (C) | RECIP preprocess on B, then MUL tiles |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: In FPU mode, DIV is decomposed: `process_rhs = RECIP` causes CB c_4 to be allocated and a PREPROCESS step runs RECIP on each B tile before the main FPU MUL operation. The `BINARY_OP` macro expands to `mul_tiles(cb_post_lhs, cb_post_rhs, 0, 0, 0)`.

### SFPU Compute Kernel (Scalar)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (compute) | N/A | CB c_0 (A), CB c_1 (scalar B) | CB c_2 (C) | RHS waited once, per-tile: copy A to DST[0], copy scalar to DST[1], div_binary_tile, pack |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: The scalar tile in CB c_1 is waited on once before the loop and popped once after the loop ends. Each iteration only waits/pops the LHS tile.

### Reader Kernel (Tensor-Tensor, ng)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 (data movement) | NOC0 | DRAM/L1 (A and B) | CB c_0, CB c_1 | Read A and B tiles with broadcasting strides |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: This reader handles BOTH input tensors A and B. It iterates through the 6D output tile space using nested loops, reading corresponding A and B tiles using their respective stride-based offsets. For each tile position, it reads one A tile and one B tile, waits for the NoC barrier, then pushes both. When sharded, it simply does `cb_reserve_back` + `cb_push_back` for the full shard count.

### Writer Kernel (Tensor-Tensor, ng)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 (data movement) | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Iterates through the same 6D loop structure as the reader but writes output tiles. For each tile: wait for compute to produce it in CB c_2, get the read pointer, write via `noc_async_write_page`, barrier, pop. When sharded, the entire section is skipped (output is already in L1).

### Writer Kernel (Scalar mode)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 (data movement) | NOC1 | Scalar arg, CB c_2 | CB c_1 (scalar fill), DRAM/L1 (C) | Fill scalar tile once, then write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills a single tile in CB c_1 with the packed scalar value (using `fill_with_val` or `fill_with_val_bfloat16`), then enters the same 6D write loop as the tensor-tensor writer.

### Reader Kernel (Scalar mode, old)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 (data movement) | NOC0 | DRAM/L1 (A only) | CB c_0 | Read A tiles only |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Only reads tensor A (one input). Uses the same 6D loop and stride-based offset pattern but without any B-tensor handling.

## Implementation Notes

### DIV-Specific Behavior
- **SFPU path**: `OpConfig` sets `binary_op = SfpuBinaryOp::DIV`, producing defines `BINARY_SFPU_INIT = "div_binary_tile_init();"` and `BINARY_SFPU_OP = "div_binary_tile"` for float types, or `div_int32_tile_init()/div_int32_tile` for INT32.
- **FPU path**: `OpConfig` sets `process_rhs = UnaryOpType::RECIP` and `binary_op = FpuBinaryOp::MUL`. This means the RHS pre-activation macro is non-empty, causing CB c_4 (intermediate) to be created. Each RHS tile goes through RECIP before the MUL.
- **Integer division**: When `a_dtype == INT32 && b_dtype == INT32`, the `is_integer_division` flag prevents the TYPECAST post-activation from being added (line 604 of program factory).
- **UnpackToDestMode**: For SFPU DIV (non-POWER), all input CBs use `UnpackToDestFp32` mode, ensuring FP32 precision in the destination registers during SFPU computation.
- **FP32 dest accumulation**: Enabled when output is UInt32/Int32/Float32, or when both inputs are Float32 or Int32.

### Kernel Selection Logic
The program factory uses a two-tier kernel selection:
1. `BinaryNgKernelConfig` determines the default kernel set based on `SubtileBroadcastType`.
2. When `b` is a tensor (has_value), the reader is overridden to a `kernels_ng` reader (via `get_reader_kernel_name_and_defines`) and the writer is overridden to `WriterNoBcastNg`. The old reader only handles A; the ng reader handles both A and B.
3. When `b` is absent (scalar), the original `WriterScalar` and `ComputeScalar` kernels are used.

### Broadcasting Support
The stride-based approach enables broadcasting without special kernels. When a dimension has size 1, its stride is set to 0 (via the `(dim > 1)` multiplication), causing the reader to re-read the same tiles for that dimension. The `SubtileBroadcastType` enum handles sub-tile broadcasting (within a single tile dimension) via specialized reader/compute kernels with fill operations.

### Sharding Support
Native L1 sharding is only used when:
- Both inputs have the same shape and memory config (no broadcast needed)
- None of the buffers are in DRAM
- The output is evenly sharded
- Shard grids match between inputs and output

Otherwise, the operation falls back to the interleaved (tensor accessor) path even for sharded tensors.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng operation work in TTNN? What are the kernel files, the SubtileBroadcastType enum, the OpConfig class, and how does DIV specifically get implemented?"
   **Reason**: Initial architectural understanding of the binary_ng framework and DIV-specific implementation paths.
   **Key Findings**: Confirmed dual FPU/SFPU paths, identified kernel file locations in both `kernels/` and `kernels_ng/` directories, understood that DIV uses SFPU native div or FPU RECIP+MUL decomposition.

### Documentation References
1. **Source**: `binary_ng_utils.cpp` (OpConfig constructor, lines 139-146)
   **Reason**: Understanding how DIV is mapped to either SFPU or FPU operations.
   **Key Information**: SFPU path uses `SfpuBinaryOp::DIV` directly; FPU path sets `process_rhs = RECIP` and uses `FpuBinaryOp::MUL`.

2. **Source**: `binary_ng_utils.cpp` (get_sfpu_init_fn, lines 394-399)
   **Reason**: Identifying the exact SFPU function calls for DIV.
   **Key Information**: Float DIV uses `div_binary_tile_init()` / `div_binary_tile`; INT32 DIV uses `div_int32_tile_init()` / `div_int32_tile`.

3. **Source**: `binary_ng_program_factory.cpp` (lines 600-602)
   **Reason**: Understanding integer division special handling.
   **Key Information**: `is_integer_division` flag prevents TYPECAST post-activation from being added when both inputs are INT32.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (metal-level wrapper) and `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` (tt_llk-level core logic with `_calculate_sfpu_binary_` and `_sfpu_binary_init_`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `div_binary_tile(0, 1, 0)` (via the `BINARY_SFPU_OP` macro), defined in `eltwise_binary_sfpu.h`. This wraps the call in a `MATH(...)` guard so it only executes on the math RISC-V.
2. `div_binary_tile` calls `llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>(idst0, idst1, odst)` in `llk_math_eltwise_binary_sfpu_binop.h`.
3. `llk_math_eltwise_binary_sfpu_binop_div` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>` (in `llk_math_eltwise_binary_sfpu_params.h`), passing `calculate_sfpu_binary_div<APPROX, BinaryOp::DIV, 8, is_fp32_dest_acc_en>` as the callable. The params function handles face iteration (4 faces in RC mode, each advancing the DEST pointer by 16 rows via two `TTI_SETRWC` of 8), calling the SFPU function once per face.
4. `calculate_sfpu_binary_div` (in `ckernel_sfpu_binary.h` at metal level) is the core SFPU function. It loops 8 iterations (one per row-group within a face), loads two inputs from DEST registers, computes `in0 * _sfpu_reciprocal_<2>(in1)`, handles special cases (0/0=NaN, x/0=signed-inf, x/x=1.0), optionally applies bf16 RNE rounding, and writes the result back to DEST.

The init path is analogous: `div_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>()`, which calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures addrmod + resets counters) followed by `sfpu_binary_init<APPROX, BinaryOp::DIV>()`, which calls `_sfpu_binary_init_<APPROX, BinaryOp::DIV>()`. For DIV, this calls `_init_sfpu_reciprocal_<false>()` to load the quadratic approximation coefficients (Wormhole) or set `vConstFloatPrgm0 = 2.0f` (Blackhole).

### Annotated SFPU Kernel Source

**Wormhole B0 and Blackhole metal-level wrapper** (identical on both architectures):

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1; // extract bit 16 for tie-breaking
    bits = bits + 0x7fffU + lsb; // RNE: add 0x7fff + lsb to round correctly at midpoint
    bits = bits & 0xFFFF0000U; // truncate lower 16 bits
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=APPROX (compile-time), BINOP=BinaryOp::DIV, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 rows / SFP_DESTREG_STRIDE(2) = 32 SFPI-addressable rows per tile
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // load numerator from DEST tile slot 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // load divisor from DEST tile slot 1
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1); // a/b = a * (1/b), 2 Newton-Raphson iterations

        v_if(in1 == 0) { // division by zero handling
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); } // 0/0 = NaN
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0); // sign(inf) = sign(numerator) via SFPSETSGN
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; } // x/x = 1.0 exactly
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // truncate to bf16 precision when dest is not fp32
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // write result to DEST tile slot 0
        sfpi::dst_reg++; // advance DEST row pointer (via SFPU auto-increment)
    }
}
```

**Wormhole B0 reciprocal** (`_sfpu_reciprocal_<2>` -- the core of DIV):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: scale input to [-2,-1), preserving mantissa

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2 (Sollya-optimized coefficients)
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // k1=1.4545, k0=0.3232

    // Scale factor: scale.Exp = 255-in.Exp via bitwise NOT (SFPNOT), handles 0->inf and inf->0 naturally
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    y = sfpi::vConstFloatPrgm2 + y * negative_x; // k2=2.1212

    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN: clear mantissa of scale

    // Newton-Raphson iteration 1: t = 1 - x*y; y = y + y*t
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    scale *= 0.5f; // adjust scale exponent: 255-E -> 254-E (float32 bias correction)

    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Newton-Raphson iteration 2
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    y = y * scale; // apply scaling factor to de-normalize reciprocal
    y = sfpi::setsgn(y, in); // SFPSETSGN: set sign of result to match input sign

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Sollya-optimized quadratic coefficients for 1/x over [1,2)
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;  // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;        // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;      // k2
}
```

**Blackhole reciprocal** (`_sfpu_reciprocal_<2>` -- different strategy using hardware `approx_recip`):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x)
{
    sfpi::vFloat y = sfpi::approx_recip(x); // SFPARECIP hardware instruction: ~7-bit initial approximation

    if constexpr (max_iter > 0)
    {
        // Negated Newton-Raphson: t = x*y - 2.0 (negated so NaN from 0*inf has positive sign for easy detection)
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0; // vConstFloatPrgm0 = 2.0f

        if constexpr (max_iter > 1)
        {
            sfpi::vFloat y1 = y * -t - sfpi::vConst0; // y1 = y*(2-x*y), vConst0=0 used as addend
            v_if (t < 0) // t<0 means normal case (NaN would be >=0); CC set by preceding SFPMAD
            {
                t = x * y1 - sfpi::vConstFloatPrgm0; // second NR iteration
                y = y1 * -t - sfpi::vConst0;
            }
            v_endif;
        }
        else
        {
            v_if (t < 0)
            {
                y = y * -t - sfpi::vConst0;
            }
            v_endif;
        }
    }

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 2.0f; // constant used in Newton-Raphson: t = x*y - 2.0
    }
}
```

**tt_llk-level binary init** (same on both architectures):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>(); // always non-approximate for DIV
    }
    // ...
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `sfpi::dst_reg[offset]` (load) | **SFPLOAD**: Loads a 32-bit value from a DEST register row into an SFPU local register (vFloat). Used to read numerator and divisor tiles. |
| `sfpi::dst_reg[offset] = val` (store) | **SFPSTORE**: Writes a 32-bit value from an SFPU local register back to a DEST register row. Used to write the division result. |
| `sfpi::dst_reg++` | **SFPINCRWC** (implicit): Increments the DEST register row counter by the stride amount (SFP_DESTREG_STRIDE=2), advancing to the next row-pair. |
| `*` (vFloat multiply) | **SFPMAD** (multiply-add): Performs `a * b + c`. Used for `in0 * reciprocal(in1)`, Newton-Raphson steps (`y + y*t`), and scaling. When used as pure multiply, the addend is 0. |
| `+` / `-` (vFloat add/sub) | **SFPMAD** / **SFPIADD**: Addition and subtraction are implemented via MAD with one multiplicand set to 1.0 or via integer add. Used in Newton-Raphson: `1.0 + negative_x * y`. |
| `sfpi::setman(dst, src)` | **SFPSETMAN**: Replaces the mantissa field of `dst` with the mantissa from `src` (or an immediate). Used to normalize input to [1,2) range and to clear scale mantissa. |
| `sfpi::setsgn(val, sign_src)` | **SFPSETSGN**: Sets the sign bit of `val` to match `sign_src`. Used to transfer the input sign to the reciprocal result and to set infinity's sign to match the numerator. |
| `sfpi::setexp(val, exp)` | **SFPSETEXP**: Sets the exponent field of a float. Not directly used in DIV, but mentioned for context. |
| `~` (vUInt bitwise NOT) | **SFPNOT**: Bitwise NOT of a register. Used to compute `255 - exponent` efficiently for the scale factor in Wormhole reciprocal. |
| `sfpi::reinterpret<T>(val)` | **SFPMOV** (or no-op cast): Reinterprets the bit pattern between vFloat/vInt/vUInt without changing bits. |
| `v_if` / `v_elseif` / `v_endif` | **SFPCOMPC** / **SFPENCC** / **SFPPUSHC** / **SFPPOPC**: Predicated execution using the SFPU condition code stack. Compares set CC lanes, and `v_if`/`v_endif` push/pop the CC to enable nested conditionals. Used for zero-division and x==y special cases. |
| `==` / `<` (vFloat comparison) | **SFPCOMPC**: Compares two vFloat values lane-wise, setting condition codes. `in1 == 0` checks for division-by-zero; `in0 == in1` checks for identity division. |
| `sfpi::vConst1` | Built-in constant register holding 1.0f. Used for the `x/x = 1.0` identity case. |
| `sfpi::vConstNeg1` | Built-in constant register holding -1.0f. Used as the sign+exponent donor in Wormhole `setman`. |
| `sfpi::vConstFloatPrgm0/1/2` | **SFPLOADI** (to program): Programmable constant registers loaded during init. Wormhole: polynomial coefficients (0.3232, 1.4545, 2.1212). Blackhole: 2.0f for Newton-Raphson. |
| `sfpi::approx_recip(x)` | **SFPARECIP** (Blackhole only): Hardware approximate reciprocal instruction providing ~7-bit initial estimate. Not available on Wormhole. |
| `>> 16`, `& mask`, `+ imm` | **SFPSHFT** / **SFPAND** / **SFPIADD**: Integer bitwise operations used in `float32_to_bf16_rne` for software Round-to-Nearest-Even when dest is not fp32. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[idst0 * 32 + row]** | Source for numerator (`in0`). Tile slot 0 (idst0=0) loaded by `copy_tile` from CB c_0. |
| **DEST[idst1 * 32 + row]** | Source for divisor (`in1`). Tile slot 1 (idst1=1) loaded by `copy_tile` from CB c_1. |
| **DEST[odst * 32 + row]** | Output destination for the division result. Same as slot 0 (odst=0), so the numerator is overwritten in-place. |
| **LREG (vFloat locals)** | `in0`, `in1`, `result`, `negative_x`, `y`, `t`, `scale`, `scale_bits` -- SFPU local registers (LREGs 0-7). The reciprocal function is heavily register-intensive, using ~6 live vFloat/vInt values simultaneously. |
| **vConstFloatPrgm0** | Wormhole: quadratic coefficient k0 = 0.3232. Blackhole: Newton-Raphson constant 2.0. |
| **vConstFloatPrgm1** | Wormhole: quadratic coefficient k1 = 1.4545. Blackhole: unused by DIV. |
| **vConstFloatPrgm2** | Wormhole: quadratic coefficient k2 = 2.1212. Blackhole: unused by DIV. |
| **vConst1** | Hardwired 1.0f. Used in Newton-Raphson (`1.0 + neg_x * y`) and as the result for x/x. |
| **vConstNeg1** | Hardwired -1.0f. Used as sign+exponent donor in Wormhole's `setman` normalization. |
| **vConst0** | Hardwired 0.0f. Used in Blackhole as the zero addend in `y * -t - 0`. |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`, which calls `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

For DIV (and all binary SFPU ops that are not mul_int32/max/min variants), only **ADDR_MOD_7** is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment of SRC_A register address |
| `srcb.incr` | 0 | No auto-increment of SRC_B register address |
| `dest.incr` | 0 | No auto-increment of DEST register address |

This configuration is identical on both **Wormhole B0** and **Blackhole**. The SFPU binary operations do not use hardware auto-increment for DEST addressing because the SFPU kernel manages DEST row advancement explicitly via `sfpi::dst_reg++` (which emits SFPINCRWC). The `ADDR_MOD_7` slot is chosen to avoid conflicts with A2D (unpack-to-dest) operations that use `ADDR_MOD_0` and `ADDR_MOD_2`.

Between faces, the params function advances the DEST pointer by 16 rows (two `TTI_SETRWC` calls each incrementing by 8), ensuring the SFPU processes all four 16x16 faces of the 32x32 tile.

Note: The `ADDR_MOD_6` slot (with `dest.incr = 2`) is only configured for `SfpuType::mul_int32`, `max`, `min`, and their int32/uint32 variants -- not for DIV.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "Where is div_binary_tile implemented? Trace from the compute_kernel_api header through LLK dispatch to the ckernel SFPU implementation."
   **Reason**: Needed to identify the complete file chain for the DIV SFPU kernel across all abstraction layers.
   **Key Findings**: Confirmed the 4-layer call chain: `eltwise_binary_sfpu.h` -> `llk_math_eltwise_binary_sfpu_binop.h` -> `llk_math_eltwise_binary_sfpu_params.h` -> `ckernel_sfpu_binary.h`. The dedicated `calculate_sfpu_binary_div` function exists at the metal level for both Wormhole and Blackhole.

2. **Query**: "How is div_binary_tile implemented in the LLK layer? What SFPU ckernel function does it call?"
   **Reason**: Needed to understand the LLK-level dispatch and the params function that manages face iteration.
   **Key Findings**: The `_llk_math_eltwise_binary_sfpu_params_` function accepts the SFPU callable and handles RC/R/C vector modes with face iteration. For RC mode (default), it calls the SFPU function 4 times (once per face) with `TTI_SETRWC` advancing between faces. The ADDR_MOD_7 configuration avoids conflicts with A2D.

3. **Query**: "What is approx_recip in SFPI? How does it work on Blackhole? What SFPU instruction does it map to?"
   **Reason**: The Blackhole reciprocal implementation uses `sfpi::approx_recip` which is a hardware instruction not available on Wormhole.
   **Key Findings**: `approx_recip` maps to the `SFPARECIP` hardware instruction on Blackhole, providing a ~7-bit initial approximation of 1/x. This eliminates the need for the quadratic polynomial initial estimate used on Wormhole, reducing the instruction count for the reciprocal seed.

4. **Query**: "What do setman, setsgn, setexp do in SFPI? What SFPU instructions do they map to?"
   **Reason**: These intrinsics are used extensively in the Wormhole reciprocal implementation and needed documentation.
   **Key Findings**: `setman` maps to SFPSETMAN (replaces mantissa field), `setsgn` maps to SFPSETSGN (replaces sign bit), `setexp` maps to SFPSETEXP (replaces exponent field). `vConstFloatPrgm0/1/2` are programmable constant registers that can be loaded with arbitrary float values during kernel init.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instructions used (SFPMAD, SFPSETMAN, SFPSETSGN, SFPNOT, SFPARECIP, SFPCOMPC, etc.) were sufficiently documented through DeepWiki and source code inspection.

### Glean References
No Glean references were needed for this analysis.
