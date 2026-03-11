# LOGADDEXP Implementation Analysis

## Overview

LOGADDEXP computes `log(exp(a) + exp(b))` element-wise for two input tensors. It is implemented through the **binary_ng** (next-generation binary) program factory, which provides a unified framework for all binary element-wise operations in TTNN.

Rather than implementing the full formula as a single fused SFPU kernel, LOGADDEXP is **decomposed** into three stages using the binary_ng activation pipeline:
1. **LHS pre-activation**: `exp(a)` (unary EXP applied to input A)
2. **RHS pre-activation**: `exp(b)` (unary EXP applied to input B)
3. **Core binary operation**: `exp(a) + exp(b)` (FPU or SFPU ADD)
4. **Post-activation**: `log(exp(a) + exp(b))` (unary LOG applied to the sum)

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

**SFPU vs FPU determination**: LOGADDEXP is classified as an SFPU operation (and thus uses SFPU ADD) **only** when both inputs are FLOAT32. For BFLOAT16 inputs, it uses FPU ADD. This is controlled by `is_binary_sfpu_op()` in `binary_ng_device_operation.cpp` (line 35).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Flat loop over assigned tiles per core, with nested dimension traversal (ND, D, N, C, Ht, Wt) for memory addressing |

## Tensor Format and Layout

### Input Tensors

| Property | Input A | Input B |
|----------|---------|---------|
| **Logical shape** | Arbitrary (up to 6+ dims) | Arbitrary (broadcastable to A) |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT/WIDTH/BLOCK) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32 (SFPU path) or BFLOAT16 (FPU path) | Same as A |

### Output Tensor

| Property | Output C |
|----------|----------|
| **Logical shape** | Broadcast-compatible shape of A and B |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input dtype (or explicit override via `output_dtype`) |

### Layout Transformations

No explicit tilize/untilize within the operation. Both inputs must already be in TILE_LAYOUT. The pre-activation and post-activation stages (EXP, LOG) are applied in-place within the compute kernel's register file, with intermediate results stored through circular buffers c_3 and c_4.

For LOGADDEXP specifically, because `op_has_exp` is true (line 632 of program factory), the intermediate CBs for LHS/RHS activations use **Float16_b** format on the FPU path (lines 647, 665), which is the natural format for the FPU's internal representation. On the SFPU path, the intermediates use the same data format as the inputs.

## Data Flow Pattern

### FPU Path (BFLOAT16 inputs) -- uses `eltwise_binary_no_bcast.cpp`

| Stage | Kernel | Reads From | Writes To | CB Operations | Description |
|-------|--------|------------|-----------|---------------|-------------|
| 1 | Reader | DRAM/L1 | CB c_0 (A), CB c_1 (B) | reserve_back, push_back | Read one tile of A and one tile of B per iteration |
| 2 | Compute (LHS preprocess) | CB c_0 | CB c_3 | wait_front(c_0), reserve_back(c_3), push_back(c_3), pop_front(c_0) | Copy tile to DST regs, apply EXP, pack to c_3 |
| 3 | Compute (RHS preprocess) | CB c_1 | CB c_4 | wait_front(c_1), reserve_back(c_4), push_back(c_4), pop_front(c_1) | Copy tile to DST regs, apply EXP, pack to c_4 |
| 4 | Compute (binary + post) | CB c_3, CB c_4 | CB c_2 | wait_front(c_3,c_4), reserve_back(c_2), push_back(c_2), pop_front(c_3,c_4) | FPU ADD_tiles on c_3 and c_4, apply LOG, pack to c_2 |
| 5 | Writer | CB c_2 | DRAM/L1 | wait_front, pop_front | Write output tile to memory |

### SFPU Path (FLOAT32 inputs) -- uses `eltwise_binary_sfpu_no_bcast.cpp`

Same overall flow, but stage 4 uses SFPU operations:
- `copy_tile` LHS from c_3 to DST[0]
- `copy_tile` RHS from c_4 to DST[1]
- `add_binary_tile(0, 1, 0)` -- SFPU binary add
- `PROCESS_POST_ACTIVATIONS(0)` -- unary LOG on DST[0]
- `pack_tile(0, cb_out)`

## Circular Buffer Configuration

### FPU Path (BFLOAT16, no sharding, tensor-tensor)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A tiles | 2 tiles | 1 tile | Double | Reader | Compute (LHS preprocess) | Program |
| c_1 | cb_src_b | Input B tiles | 2 tiles | 1 tile | Double | Reader | Compute (RHS preprocess) | Program |
| c_2 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Program |
| c_3 | cb_lhs_interim | LHS after EXP | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (binary) | Block |
| c_4 | cb_rhs_interim | RHS after EXP | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (binary) | Block |

### SFPU Path (FLOAT32, no sharding, tensor-tensor)

Same structure as FPU path. The key difference is that c_3 and c_4 use the input data format (FLOAT32) rather than Float16_b.

**Note on intermediate format**: The program factory at lines 645-651 and 663-669 selects the intermediate format. For LOGADDEXP, `op_has_exp` is true, so:
- FPU path: `a_intermediate_format = Float16_b` (since `is_sfpu_op` is false)
- SFPU path: `a_intermediate_format = a_data_format` (e.g., Float32)

### Sharded Path

When sharded, c_0, c_1, and c_2 capacities are set to the shard volume (tiles per shard) instead of 2, and the buffers are backed by the tensor's L1 memory directly.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| c_0 (input A) | 2 tiles | 1 tile | Double-buffered | Reader can prefetch while compute processes |
| c_1 (input B) | 2 tiles | 1 tile | Double-buffered | Reader can prefetch while compute processes |
| c_2 (output) | 2 tiles | 1 tile | Double-buffered | Compute can produce while writer drains |
| c_3 (LHS interim) | 1 tile | 1 tile | Single-buffered | No overlap -- sequential within compute |
| c_4 (RHS interim) | 1 tile | 1 tile | Single-buffered | No overlap -- sequential within compute |

The double-buffering on c_0, c_1, c_2 enables reader-compute and compute-writer overlap. The intermediate CBs (c_3, c_4) are single-buffered because they are produced and consumed entirely within the compute kernel.

## Index Calculations

The reader kernel uses a 6-level nested loop structure to traverse the output tensor's logical dimensions and map them to input tile indices:

```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt + tw
```

**Stride-based broadcasting**: Input strides are computed in the host as:
- `nD_stride = aHt * aWt * aC * aN * aD * (aND > 1)` -- zero if dimension is broadcast
- `d_stride = aHt * aWt * aC * aN * (aD > 1)` -- zero if D=1
- `n_stride = aHt * aWt * aC * (aN > 1)` -- zero if N=1
- `c_stride = aHt * aWt * (aC > 1)` -- zero if C=1

When a dimension size is 1, the stride is 0, which means the offset does not advance along that dimension -- effectively broadcasting that dimension. The same pattern applies to tensor B with its own set of strides.

**TensorAccessor** is used for physical address resolution: the kernel creates a `TensorAccessor` object from compile-time and common runtime args, then calls `noc_async_read_page(tile_offset, src, l1_addr)` to read tiles. The TensorAccessor handles bank-interleaved address mapping internally.

## Memory Access Patterns

### Read Pattern
- **Pattern**: Sequential tile reads within the innermost W-dimension, with strided jumps at dimension boundaries
- **Granularity**: One tile at a time (no block reads in the no-bcast case)
- **Both A and B read in lockstep**: For each output tile, the reader reads exactly one tile from A and one from B (with broadcast-aware offsets)
- **Barrier**: `noc_async_read_barrier()` after each pair of reads ensures data arrives before pushing to CBs
- **Access type**: DRAM bank-interleaved via TensorAccessor (or direct L1 if sharded)

### Write Pattern
- **Pattern**: Sequential tile writes matching the output tensor's dimension traversal order
- **Granularity**: One tile at a time
- **Barrier**: `noc_async_write_barrier()` after each tile write
- **Access type**: DRAM bank-interleaved via TensorAccessor (or direct L1 if sharded)

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (or 1D depending on available cores) |
| **Grid dimensions** | `compute_with_storage_grid.x` x `compute_with_storage_grid.y` (device-dependent) |
| **Total cores** | Up to all available compute-with-storage cores |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |

Work splitting is performed by `tt::tt_metal::split_work_to_cores()`, which divides total output tiles across available cores. Cores not assigned any work receive zero-filled runtime arguments and effectively become no-ops.

For sharded inputs, the core grid is determined by the shard specification rather than work splitting, and each core processes its local shard.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per read-compute-write cycle |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessorArgs(A) | uint32_t[] | Compile-time tensor accessor args for input A |
| M+1..P | TensorAccessorArgs(B) | uint32_t[] | Compile-time tensor accessor args for input B |
| P+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessorArgs(C) | uint32_t[] | Compile-time tensor accessor args for output C |
| M+1 | has_sharding | uint32_t | 1 if any tensor is natively sharded, 0 otherwise |

### Runtime Arguments

#### Reader Kernel (21 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input A buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core (= c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (0 if not sharded) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles (0 if not sharded) |
| 5 | nD_stride | uint32_t | A's collapsed ND-dimension stride (0 if broadcast) |
| 6 | d_stride | uint32_t | A's D-dimension stride (0 if broadcast) |
| 7 | n_stride | uint32_t | A's N-dimension stride (0 if broadcast) |
| 8 | c_stride | uint32_t | A's C-dimension stride (0 if broadcast) |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed dimensions beyond rank 5 |
| 15 | src_addr_b | uint32_t | Input B buffer address |
| 16 | nD_stride_b | uint32_t | B's collapsed ND-dimension stride |
| 17 | d_stride_b | uint32_t | B's D-dimension stride |
| 18 | n_stride_b | uint32_t | B's N-dimension stride |
| 19 | c_stride_b | uint32_t | B's C-dimension stride |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (0 if not sharded) |

#### Writer Kernel (11 args per core, tensor-tensor case)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output C buffer address |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles this core writes |
| 3 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed dimensions beyond rank 5 |
| 10 | (unused) | uint32_t | Set to 0 |

#### Compute Kernel (4 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles this core must process |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE broadcast type) |
| 2 | counter | uint32_t | Broadcast counter start (0 for NONE broadcast type) |
| 3 | compute_scalar_value | uint32_t | Unused for LOGADDEXP (set to 0) |

## Kernel Implementations

### Reader Kernel: `reader_interleaved_no_bcast.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (A buffer, B buffer) | CB c_0, CB c_1 | Read tiles of A and B using TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Uses stride-based dimension traversal with 6-level nested loops (ND, D, N, C, Ht, Wt). Strides of 0 implement broadcasting. Both tensors A and B are read in the same kernel, with separate TensorAccessor instances and separate stride sets. Each iteration reads one tile from A into c_0 and one tile from B into c_1, then issues a read barrier.

### Compute Kernel (FPU path): `eltwise_binary_no_bcast.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 -> c_3 (LHS), CB c_1 -> c_4 (RHS), then c_3+c_4 -> c_2 | CB c_2 | EXP, ADD_tiles, LOG |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**:
  1. `PREPROCESS(LHS, c_0, c_3, c_2, 1)`: Wait for c_0, copy tile to DST, apply EXP via `PROCESS_LHS_ACTIVATIONS(i)`, pack to c_3
  2. `PREPROCESS(RHS, c_1, c_4, c_2, 1)`: Wait for c_1, copy tile to DST, apply EXP via `PROCESS_RHS_ACTIVATIONS(i)`, pack to c_4
  3. Wait for c_3 and c_4, reserve c_2
  4. `BINARY_OP(c_3, c_4, 0, 0, 0)` expands to `ADD_tiles(c_3, c_4, 0, 0, 0)` on FPU
  5. `PROCESS_POST_ACTIVATIONS(0)` applies LOG
  6. Pack result to c_2, pop c_3 and c_4

### Compute Kernel (SFPU path): `eltwise_binary_sfpu_no_bcast.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 | N/A | CB c_0 -> c_3 (LHS), CB c_1 -> c_4 (RHS), then c_3+c_4 -> c_2 | CB c_2 | EXP, SFPU add_binary_tile, LOG |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: Same preprocessing as FPU path. The binary operation uses `copy_tile` to load both operands into DST registers (at offsets 2i and 2i+1), then calls `add_binary_tile(0, 1, 0)` as the SFPU binary operation. Uses `UnpackToDestFp32` mode for FLOAT32 precision throughout.

### Writer Kernel: `writer_interleaved_no_bcast.cpp`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM (C buffer) | Write output tiles |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Same 6-level nested loop structure as reader. For each tile, waits on c_2, reads L1 address, issues `noc_async_write_page`, barriers, then pops c_2.

## Implementation Notes

1. **Decomposed vs Fused**: LOGADDEXP is NOT a fused SFPU kernel. It is decomposed into `EXP + ADD + LOG` using the binary_ng activation pipeline. This means each tile goes through 3 separate compute passes within the compute kernel loop, with intermediate results packed to CBs c_3/c_4 between the preprocess and binary stages.

2. **Numerical precision**: Because EXP can produce very large values, using FLOAT32 inputs (SFPU path) is recommended for numerical stability. The `UnpackToDestFp32` mode ensures full 32-bit precision in the DST register file during SFPU operations.

3. **Intermediate format for FPU path**: When using BFLOAT16 inputs on the FPU path, the intermediate CBs (c_3, c_4) use Float16_b format specifically because `op_has_exp` is true (line 632). This matches the FPU's native internal format and avoids unnecessary format conversions.

4. **Broadcast support**: The binary_ng framework transparently handles broadcasting through the `SubtileBroadcastType` enum and stride-based dimension traversal. For LOGADDEXP, all broadcast variants (scalar, row, col, none) are supported. The appropriate reader and compute kernel variants are selected at program creation time.

5. **Program caching**: The binary_ng framework supports program caching through `override_runtime_arguments()`. When the same operation is called with different tensor data but the same shapes/dtypes/memory configs, only runtime arguments are updated without rebuilding the program.

6. **Kernel define expansion for LOGADDEXP**:
   - `PROCESS_LHS_ACTIVATIONS(i)` expands to the EXP init + EXP function call
   - `PROCESS_RHS_ACTIVATIONS(i)` expands to the EXP init + EXP function call
   - `PROCESS_POST_ACTIVATIONS(i)` expands to the LOG init + LOG function call
   - `BINARY_OP` expands to `ADD_tiles` (FPU) or `BINARY_SFPU_OP` expands to `add_binary_tile` (SFPU)
   - `BINARY_SFPU_INIT` expands to `add_binary_tile_init();`

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng (next generation binary) operation framework work in TTNN? What is its architecture, how does it handle different subtypes like LOGADDEXP, and what kernels does it use?"
   **Reason**: Needed to understand the overall architecture of the binary_ng framework before diving into source code.
   **Key Findings**: The framework uses a single unified ProgramFactory with operation type (FPU vs SFPU) determined by `is_binary_sfpu_op`. Operations like LOGADDEXP are decomposed into pre-activations + core binary op + post-activations. Program caching is supported. Kernel selection depends on SubtileBroadcastType.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 226-231)
   **Reason**: Needed to understand how LOGADDEXP is decomposed into primitive operations.
   **Key Information**: LOGADDEXP sets `process_lhs = EXP`, `process_rhs = EXP`, `binary_op = ADD`, `postprocess = LOG`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` (lines 31-35)
   **Reason**: Needed to determine when LOGADDEXP uses SFPU vs FPU path.
   **Key Information**: LOGADDEXP is classified as SFPU only when both inputs are FLOAT32.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp` and `eltwise_utils_sfpu.hpp`
   **Reason**: Needed to understand the PREPROCESS macro that implements pre-activation processing.
   **Key Information**: PREPROCESS copies tile to DST, applies activation function, packs result to intermediate CB. FPU version also reconfigures data format source A; SFPU version only reconfigures pack format.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` (line 632)
   **Reason**: Needed to understand special handling for LOGADDEXP intermediate formats.
   **Key Information**: `op_has_exp` flag is true for LOGADDEXP, LDEXP, and LOGADDEXP2, causing intermediate CBs to use Float16_b on the FPU path.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. LOGADDEXP is a composite operation decomposed into three SFPU stages: **EXP** (pre-activation on both inputs), **ADD** (binary SFPU), and **LOG** (post-activation). Each stage has its own SFPU kernel implementation documented below.

### SFPU Abstraction Layers

Because LOGADDEXP is decomposed into three separate SFPU operations, there are three sets of abstraction layers:

**EXP (pre-activation on LHS and RHS)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (via `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macro) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` (shared) and `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` (arch-specific wrapper) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**ADD (binary SFPU)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h` (shared) and `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (arch-specific wrapper) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**LOG (post-activation)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

LOGADDEXP's SFPU path is invoked from the compute kernel `eltwise_binary_sfpu_no_bcast.cpp`. The call chain proceeds through three separate SFPU dispatches per tile:

**EXP pre-activation (for both LHS and RHS):**
1. The `PREPROCESS(LHS, ...)` macro expands to `PROCESS_LHS_ACTIVATIONS(i)`, which calls `exp_tile_init()` then `exp_tile(i)`.
2. `exp_tile_init<APPROX, fast_and_approx, scale, clamp>()` calls `SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, ...)`, which invokes `llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROX>(sfpu::exp_init<...>)` -- this sets up ADDR_MOD_7, resets counters, and calls the init function to program SFPU constants/macros.
3. `exp_tile<APPROX, fast_and_approx, ...>(idst, vector_mode, scale)` calls `SFPU_TEMPLATE_PARAMS_KERNEL_FN(calculate_exponential, ...)`, which invokes `_llk_math_eltwise_unary_sfpu_params_<APPROX>(sfpu::calculate_exponential<...>, idst, vector_mode, scale)`.
4. The params function stalls SFPU, sets the DST write address, then loops over 4 faces (for VectorMode::RC), calling `calculate_exponential<...>(scale)` once per face (8 iterations each, covering 8 rows of 32 elements).
5. `calculate_exponential` dispatches to either `_calculate_exponential_` (approximate path) or a loop calling `_sfpu_exp_improved_` (non-approximate path) depending on `APPROXIMATION_MODE`.

**Binary ADD:**
1. Inside the tile loop, after both operands are in DST, `BINARY_SFPU_INIT` expands to `add_binary_tile_init()` which calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()` -- this invokes the binary sfpu init (ADDR_MOD_7 setup, counter reset) and `sfpu_binary_init<APPROX, ADD>()` (a no-op for ADD since it needs no reciprocal setup).
2. `BINARY_SFPU_OP(i*2, i*2+1, i*2)` expands to `add_binary_tile(i*2, i*2+1, i*2)`, which calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(dst0, dst1, odst)`.
3. This invokes `_llk_math_eltwise_binary_sfpu_params_<APPROX>(sfpu::calculate_sfpu_binary<APPROX, ADD, 8, false>, dst0, dst1, odst, VectorMode::RC)`.
4. The params function stalls SFPU, then loops over 4 faces, calling `calculate_sfpu_binary` (which delegates to `_calculate_sfpu_binary_<APPROX, ADD, 8>`) with TTI_SETRWC to advance the DST read/write pointer between faces.
5. `_calculate_sfpu_binary_` iterates 8 times, loading `dst_reg[in0 * 32]` and `dst_reg[in1 * 32]`, computing `in0 + in1`, and storing to `dst_reg[out * 32]`.

**LOG post-activation:**
1. `PROCESS_POST_ACTIVATIONS(i*2)` expands to `log_tile_init()` then `log_tile(i*2)`.
2. `log_tile_init<fast_and_approx>()` calls `llk_math_eltwise_unary_sfpu_log_init<APPROX, fast_and_approx, DST_ACCUM_MODE>()` which sets up the unary SFPU init and calls `sfpu::log_init<APPROX, fast_and_approx, is_fp32_dest_acc_en>()` to program SFPU constants (ln(2), polynomial coefficients, or reciprocal init for FP32 path).
3. `log_tile<fast_and_approx>(idst)` calls `llk_math_eltwise_unary_sfpu_log<APPROX, fast_and_approx, DST_ACCUM_MODE>(idst)`.
4. This invokes `_llk_math_eltwise_unary_sfpu_params_<APPROX>(sfpu::calculate_log<APPROX, fast_and_approx, false, is_fp32_dest_acc_en>, idst, VectorMode::RC, 0)`.
5. `calculate_log` iterates 8 times per face, reading `dst_reg[0]`, computing either `calculate_log_body` (BF16 path) or `calculate_log_f32_body` (FP32 path), and writing back to `dst_reg[0]`.

### Annotated SFPU Kernel Source

#### EXP Kernel -- Shared Core (`_calculate_exponential_` and helpers)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126); // clamp exponent to -1 (bias 126)
    }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    v_if (exp >= 0)
    {
        val = val * val; // repeated squaring for integer exponent part
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0); // narrow predication on each loop
            val = val * val;
        }
    }
    v_endif;

    return val;
}

inline sfpi::vFloat _calculate_exponential_approx_(sfpi::vFloat in)
{
    sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0; // 1/ln(2)
    sfpi::vFloat c23_73         = sfpi::vConstFloatPrgm1;
    sfpi::vInt adj_exp          = sfpi::vConstIntPrgm2;
    in                          = in * vConstLn2Recip + c23_73;

    // Remove Exponent of 7 and bias the Mantissa to 127.
    sfpi::vInt in_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(in);

    // SHL to move integer bits to exponent
    in_short <<= 10 - p_exp::FRAC_BITS;
    return sfpi::reinterpret<sfpi::vFloat>(in_short);
}

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat result = 0.0f;
    if constexpr (SCALE_EN)
    {
        in = in * sfpi::s2vFloat16b(exp_base_scale_factor);
    }
    if constexpr (APPROXIMATION_MODE)
    {
        if constexpr (!SKIP_POSITIVE_CHECK)
        {
            v_if (in >= 89)
            {
                sfpi::vFloat in_inf = std::numeric_limits<float>::infinity();
                result              = in_inf;
            }
            v_elseif (in < -42)
            {
                result = 0.0f;
            }
            v_else
            {
                result = _calculate_exponential_approx_(in);
            }
            v_endif;
        }
        else
        {
            v_if (in < -42)
            {
                result = 0.0f;
            }
            v_else
            {
                result = _calculate_exponential_approx_(in);
            }
            v_endif;
        }
    }
    else
    {
        result = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if (in < 0)
        {
            result = _sfpu_reciprocal_<2>(result);
        }
        v_endif;
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor) // APPROXIMATION_MODE, FAST_APPROX, SCALE_EN, SKIP_POSITIVE_CHECK, CLAMP_NEGATIVE resolved at compile time
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // Implementation notes, see the original file for more details
        // Uses SFPLOADMACRO with SWAP to sanitize inputs to [-88.5, +inf] range,
        // then MAD + ROUND + SHIFT pipeline to compute Schraudolph approximation
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 2);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 4);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 6);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 8);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 10);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 12);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 14);

        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 10);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 12);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14);
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8)
    {
        // Implementation notes, see the original file for more details
        // 8-element version using replay buffer, ~2.5 cycles/element
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        lltt::replay(0, 16);

        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else
    {
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vFloat in     = sfpi::dst_reg[0];
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(in, exp_base_scale_factor);
            sfpi::dst_reg[0]    = result;
            sfpi::dst_reg++;
        }
    }
}
```

#### EXP Kernel -- Arch-specific (Blackhole, non-approximate with FP32 DST)

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val); // Moroz et al. exp_21f, bf16 truncation
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val); // Cody-Waite range reduction, 7th order Taylor, <1 ULP for fp32
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_exponential_<
            APPROXIMATION_MODE, SCALE_EN, ITERATIONS,
            FAST_APPROX, SKIP_POSITIVE_CHECK, CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}
```

#### Binary ADD Kernel

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
// BINOP=ADD for LOGADDEXP
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 64/SFP_DESTREG_STRIDE = 32 rows per tile in SFPI addressing
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result                        = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPMAD: result = in0 * 1.0 + in1, or SFPADD
        }
        // ... other BINOP cases omitted (SUB, MUL, DIV, RSUB, POW, XLOGY)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // For ADD: no special initialization needed
}
```

#### LOG Kernel

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {
    // BF16 path: minimax polynomial approximation of log(x) over [1,2]
    sfpi::vFloat x = sfpi::setexp(in, 127); // normalize to [1, 2)

    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,  // programmed constant
        sfpi::vConstFloatPrgm2,  // programmed constant
        -2.800232410430908,
        1.3681391477584839,
        -0.3706687390804291,
        0.04224011301994324);

    sfpi::vInt exp = sfpi::exexp(in);

    // Convert negative exponents: two's complement -> sign-magnitude
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0; // ln(2)
    sfpi::vFloat result = expf * vConstLn2 + series_result; // ln(x) = exp_part * ln(2) + series_result

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    v_if(in == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    // FP32-accurate path: uses (m-1)/(m+1) transformation with reciprocal
    sfpi::vFloat result;

    sfpi::vInt exp = sfpi::exexp(val);

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < 0.f) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == 0.f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        sfpi::vFloat m = sfpi::setexp(val, 127); // normalize mantissa to [1, 2)

        // Range reduction: if m >= sqrt(2), halve m and increment exponent
        v_if(m >= sfpi::vConstFloatPrgm1) { // vConstFloatPrgm1 = sqrt(2)
            m = m * 0.5f;
            exp = exp + 1;
        }
        v_endif;

        // z = (m - 1) / (m + 1) maps [0.707, 1.414] to [-0.172, 0.172]
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

        sfpi::vFloat z2 = z * z;

        // ln(m) = 2z(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11) via Horner's method
        sfpi::vFloat p = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,
            0.3333333333333333f,
            0.2f,
            0.14285714285714285f,
            0.1111111111111111f,
            .09090909090909091f);

        sfpi::vFloat ln_m = 2.0f * (z * p);

        v_if(exp < 0) {
            sfpi::vInt exp_abs = ~exp + 1;
            exp = sfpi::setsgn(exp_abs, 1);
        }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        result = expf * sfpi::vConstFloatPrgm2 + ln_m; // vConstFloatPrgm2 = ln(2)

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }
    }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) { // HAS_BASE_SCALING=false, is_fp32_dest_acc_en=true for LOGADDEXP SFPU path
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.693147182464599609375;  // ln(2)
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;      // polynomial coefficient
        sfpi::vConstFloatPrgm2 = 3.767500400543213;        // polynomial coefficient
    } else {
        _init_sfpu_reciprocal_<false>(); // sets vConstFloatPrgm0 = 2.0f (for Newton-Raphson reciprocal)
        sfpi::vConstFloatPrgm1 = 1.4142135381698608f;   // sqrt(2) -- range reduction threshold
        sfpi::vConstFloatPrgm2 = 0.69314718246459961f;   // ln(2) -- exponent-to-log conversion
    }
}
```

### SFPU Instructions Used

LOGADDEXP uses three distinct SFPU kernel stages. The instructions below are organized by stage.

**EXP stage (approximate path -- `FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE`):**
- `SFPLOADI` -- Load immediate 16-bit values into LREG[0] (lower/upper halves) for constructing 32-bit constants
- `SFPCONFIG` -- Store LREG[0] value into a target LREG (LREG[12..14]) or program a macro sequence register
- `SFPMAD` -- Multiply-accumulate: computes `A * y + (B-C)` where A=256/ln(2), y=input, (B-C)=bias adjustment. Also used as backdoor load for macro instruction register 5
- `SFP_STOCH_RND` -- Stochastic rounding: converts FP32 MAD result to INT16. Also backdoor loads macro instruction register 6
- `SFPSHFT` -- Shift left by 15 bits to place INT16 result into FP32 exponent field. Also backdoor loads macro instruction register 7
- `SFPLOADMACRO` -- Executes a programmed macro sequence (sanitize or compute) at a specific DEST offset using a specified LREG
- `SFPNOP` -- Pipeline delay required between SFPLOADMACRO operations (SWAP takes 2 cycles)
- `SFPSWAP` -- (via macro instruction 0) Compares loaded value against LREG[14] (-88.5), keeps the larger value for input clamping

**EXP stage (approximate path -- `FAST_APPROX && APPROXIMATION_MODE && !CLAMP_NEGATIVE`):**
- `SFPLOADI` / `SFPCONFIG` -- Same constant setup as above
- `SFPMAD` / `SFP_STOCH_RND` / `SFPSHFT` -- Backdoor macro instruction loading
- `SFPSETSGN` -- Backdoor loads macro instruction register 7; when executed via LOADMACRO, restores sign from STOCHRND result to the shifted value
- `SFPLOADMACRO` -- Executes the compute macro sequence (MAD, ROUND, SHIFT, SETSGN, STORE)
- `SFPSHFT2` -- Shift operation using VC register as shift amount (LREG[14]=15); used in the discrete drain phase after replay
- `lltt::record` / `lltt::replay` -- Programs the replay buffer with 32 instructions (16 LM+SHFT2 pairs) and replays them

**EXP stage (non-approximate path, `is_fp32_dest_acc_en=true`):**
- `SFPLOAD` (via `dst_reg[0]`) -- Load a 32-element vector from DEST register file into an LREG
- `SFPSTORE` (via `dst_reg[0] = ...`) -- Store a 32-element vector from LREG back to DEST register file
- `SFPMUL` (via `val * INV_LN2`) -- Vector multiply (used for x/ln(2) and Cody-Waite range reduction)
- `SFPMAD` (via `k * LN2_HI + val`) -- Fused multiply-add for range reduction
- `SFPSETSGN` (via `setsgn(in, 0)`) -- Force sign bit to 0 (make positive)
- `SFPEXEXP` (via `exexp(z)`) -- Extract biased exponent from float
- `SFPEXMAN` (via `exman8`, `exman9`) -- Extract mantissa with implicit bit
- `SFPSETEXP` (via `setexp(p, new_exp)`) -- Set the exponent field of a float
- `SFPIADD` (via `p_exp + k_int`) -- Integer addition on exponent fields
- `SFPLUT` (via `_sfpu_reciprocal_<2>`) -- LUT-based reciprocal approximation (for negative input handling)
- Condition code manipulation via `v_if`/`v_elseif`/`v_else`/`v_endif` -- SFPSETCC, SFPCOMPC, SFPPUSHC, SFPPOPC, SFPLZ instructions

**Binary ADD stage:**
- `SFPLOAD` (via `dst_reg[idx * 32]`) -- Load vector from DEST at tile offset
- `SFPSTORE` (via `dst_reg[idx * 32] = result`) -- Store result to DEST at tile offset
- `SFPADD` or `SFPMAD` (via `in0 + in1`) -- Vector addition of two operands
- `TTI_SETRWC` -- Set read/write counters to advance DEST face addressing between the 4 faces

**LOG stage (FP32 accurate path, `is_fp32_dest_acc_en=true`):**
- `SFPLOAD` / `SFPSTORE` -- Load/store from DEST
- `SFPSETEXP` (via `setexp(val, 127)`) -- Normalize mantissa to [1, 2)
- `SFPEXEXP` (via `exexp(val)`) -- Extract debiased exponent
- `SFPMUL` (via `m * 0.5f`, `z * z`, `z * p`, `2.0f * (z * p)`) -- Vector multiplications
- `SFPMAD` (via `expf * ln2 + ln_m`) -- Fused multiply-add for final combination
- `SFPADD` / `SFPIADD` (via `m - 1`, `m + 1`, `exp + 1`, `~exp + 1`) -- Vector add/subtract, integer add
- `SFPLUT` (via `_sfpu_reciprocal_<2>(m_plus_1)`) -- Newton-Raphson reciprocal for division
- `SFPSETSGN` (via `setsgn(exp_abs, 1)`) -- Set sign bit for negative exponent conversion
- `SFPLZ` / conditional instructions -- For special case handling (0, inf, NaN, negative)
- PolynomialEvaluator::eval -- Expands to a sequence of SFPMAD instructions for Horner-form polynomial evaluation

### SFPU Register Usage

**DEST Register File:**
- DST[0] (face-relative offset via `dst_reg[0]`): Used by both EXP and LOG as the in-place read/write location. Each face processes 8 iterations (8 rows of 32 elements = 256 elements per face, 4 faces = 1024 elements per tile).
- DST[i*2] and DST[i*2+1] (tile offsets): Used by the binary ADD stage. LHS is loaded at offset `i*2` (via `copy_tile(cb_post_lhs, i, i*2)`), RHS at offset `i*2+1`. Result is written back to offset `i*2`.

**LREG (Local Registers) -- EXP approximate path:**
- LREG[0..3]: Working registers used by SFPLOADMACRO for loading values from DEST and intermediate computation
- LREG[4]: Used as SHFT2 output target (stores shifted result before SETSGN applies sign)
- LREG[12]: Constant A = 256/ln(2) = 369.33 (Schraudolph scale factor)
- LREG[13]: Constant (B-C) = 32500.82 (Schraudolph bias minus adjustment)
- LREG[14]: Either -88.5 (clamping threshold, CLAMP_NEGATIVE path) or 15 (shift amount, non-clamping path)
- LREG[16]: Staging register (used by SETSGN macro to avoid write port conflicts)

**Programmable Constants (vConstFloatPrgm0..2):**
- EXP approximate (non-fast): PrgmC0 = 1/ln(2), PrgmC1 = C23_73 (FxP conversion constant), PrgmC2 = ADJ_EXP (bias adjustment)
- EXP non-approximate: PrgmC0 = 2.0 (reciprocal init)
- LOG BF16: PrgmC0 = ln(2), PrgmC1 = -2.007 (poly coeff), PrgmC2 = 3.768 (poly coeff)
- LOG FP32: PrgmC0 = 2.0 (reciprocal init), PrgmC1 = sqrt(2) (range reduction threshold), PrgmC2 = ln(2)

**Note on register conflicts:** Because EXP and LOG use the same programmable constant registers (vConstFloatPrgm0..2), each activation stage must call its init function before execution. The compute kernel handles this via the `PROCESS_ACTIVATIONS` macro which calls init before the compute function. The binary SFPU init (`add_binary_tile_init`) is strategically placed: when post-activations exist (as in LOGADDEXP), `BINARY_SFPU_INIT` is called inside the tile loop just before `BINARY_SFPU_OP`, ensuring it runs after EXP's constants are no longer needed and before LOG re-programs them.

### Address Mode Configuration

**EXP (unary SFPU) and LOG (unary SFPU):**

Both use the standard unary SFPU address mode configuration set by `_llk_math_eltwise_unary_sfpu_init_()`:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

ADDR_MOD_7 is configured with all increments set to 0. DEST address advancement between faces is handled explicitly by `TTI_SETRWC` instructions in the params dispatch layer (`_llk_math_eltwise_unary_sfpu_params_`), which increments the DEST address by 8 twice (16 rows) between each face call, while within each face the 8 iterations advance via `dst_reg++` (SFPI auto-increment).

For the EXP fast approximate path (`FAST_APPROX && !CLAMP_NEGATIVE`), ADDR_MOD_7 is reconfigured inside `_calculate_exponential_` with `dest.incr = 2` to auto-increment the DEST offset by 2 for each SFPLOADMACRO instruction in the replay sequence.

**Binary ADD (binary SFPU):**

Set by `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

Same as unary -- ADDR_MOD_7 with all increments at 0. DEST face advancement is handled by `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice between each face (incrementing by 16 rows total) in `_llk_math_eltwise_binary_sfpu_params_`.

The address mode configuration is identical across Wormhole B0 and Blackhole for these operations. The `addr_mod_t` struct fields (`srca`, `srcb`, `dest`) each have an `incr` field that controls automatic increment of the corresponding register bank pointer after each SFPU instruction. For LOGADDEXP, all three are set to 0 because address advancement is managed explicitly.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary_ng compute kernel dispatch SFPU operations for logaddexp? What is the call chain from the compute kernel through LLK to the SFPU implementation for binary operations like logaddexp?"
   **Reason**: Needed to understand the complete dispatch chain from the compute kernel through the LLK abstraction layers to the core SFPU implementations for all three component operations (EXP, ADD, LOG).
   **Key Findings**: LOGADDEXP is decomposed into EXP + ADD + LOG via OpConfig. The dispatch uses compile-time macro defines (PROCESS_LHS_ACTIVATIONS, BINARY_SFPU_OP, PROCESS_POST_ACTIVATIONS) that expand to the appropriate init + compute calls. The `binary_op_has_exp` flag triggers special intermediate format handling.

### Confluence References
No Confluence page consultation was necessary for this analysis. The SFPU instruction details were sufficiently documented in the source code comments and DeepWiki.

### Glean References
No Glean consultation was necessary for this analysis.
