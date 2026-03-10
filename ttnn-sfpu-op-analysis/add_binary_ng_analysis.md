# ADD (binary_ng) Implementation Analysis

## Overview

The ADD operation in the `binary_ng` framework computes element-wise addition of two tensors: `c = a + b`. When dispatched as an SFPU operation (`is_sfpu = true`), it uses the SFPU vector unit on each Tensix core to perform the addition, rather than the FPU matrix unit. The binary_ng program factory is a unified, flexible factory that handles all binary operations (ADD, SUB, MUL, DIV, etc.) and all broadcast subtypes (no-broadcast, scalar, row, column, mixed) through a single code path with compile-time defines.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The ADD operation is mapped to `SfpuBinaryOp::ADD`, which produces the defines:
- `BINARY_SFPU_INIT` = `add_binary_tile_init();`
- `BINARY_SFPU_OP` = `add_binary_tile`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` = total output tiles |
| **Loop structure** | Flat loop over `num_tiles` assigned to each core; each iteration processes 1 tile (`num_tiles_per_cycle = 1`) |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (up to rank 8+, dims > 5 collapsed) | Same as A (or broadcastable) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims explicit) | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same as A (or compatible) |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcasted output shape of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input or specified output dtype |

### Layout Transformations
- No tilize/untilize within the operation; tensors must already be in TILE_LAYOUT.
- When `is_sfpu = true` and the operation is not POWER, `UnpackToDestFp32` mode is enabled for all source CBs (c_0, c_1, c_3, c_4), which causes `copy_tile` to unpack data directly to DEST registers in FP32 precision.
- When input and output data types differ (and it is not a quant or integer division op), a TYPECAST post-activation is appended.

## Data Flow Pattern

For the **tensor + tensor** case (both inputs are tensors, `SubtileBroadcastType::NONE`):

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (ReaderNoBcastNg) | DRAM/L1 (A buffer, B buffer) | CB c_0 (A tiles), CB c_1 (B tiles) | reserve_back, noc_async_read_page, push_back (one tile at a time per input) |
| 2 | Compute (eltwise_binary_sfpu_no_bcast) | CB c_0, CB c_1 | CB c_2 | wait_front, copy_tile (to DEST), add_binary_tile (SFPU), pack_tile, push_back, pop_front |
| 3 | Writer (WriterNoBcastNg) | CB c_2 | DRAM/L1 (C buffer) | wait_front, noc_async_write_page, pop_front |

For the **tensor + scalar** case (B is a scalar value):

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1a | Writer (WriterScalar) | Scalar arg | CB c_1 | reserve_back, fill_with_val, push_back (once, single tile filled with scalar) |
| 1b | Reader (ReaderNoBcast) | DRAM/L1 (A buffer) | CB c_0 | reserve_back, noc_async_read_page, push_back |
| 2 | Compute (eltwise_binary_sfpu_scalar) | CB c_0, CB c_1 | CB c_2 | Same as above, but CB c_1 contains the replicated scalar tile |
| 3 | Writer (WriterScalar) | CB c_2 | DRAM/L1 (C buffer) | wait_front, noc_async_write_page, pop_front |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_pre_lhs / cb_src_a | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_pre_rhs / cb_src_b | Input B staging | 2 tiles (interleaved, tensor) or 1 tile (scalar) or shard volume (sharded) | 1 tile | Double/Single | Reader/Writer | Compute | Program |
| c_2 | cb_out / cb_dst | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_post_lhs | LHS post-activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_post_rhs | RHS post-activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_5 | Row bcast A temp | Row broadcast temp for A (only ROW_A/ROW_A_COL_B) | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_6 | Row bcast B temp | Row broadcast temp for B (only ROW_B/ROW_B_COL_A) | 2 tiles | 1 tile | Double | Reader | Compute | Program |

Notes:
- CB c_3 is only created when `PROCESS_LHS_ACTIVATIONS(i)` is non-empty (e.g., for operations like LOGICAL_AND that pre-process LHS).
- CB c_4 is only created when `PROCESS_RHS_ACTIVATIONS(i)` is non-empty.
- For plain ADD with no pre/post activations, only c_0, c_1, c_2 are used.

## Pipeline Pattern Summary

- **Interleaved mode**: c_0 and c_1 are double-buffered (capacity = 2 tiles, block = 1 tile), enabling overlap between reader and compute. c_2 is double-buffered, enabling overlap between compute and writer.
- **Sharded mode**: CBs are sized to the full shard volume and backed by the tensor's L1 buffer directly. No double-buffering; all tiles are available at once.

## Index Calculations

The reader kernel computes a multi-dimensional tile offset for broadcasting. For each output tile position, the reader maps output coordinates (nd, d, n, c, th, tw) to input tile offsets using per-dimension strides:

```
tile_offset_a = nd * nD_stride_a + d * d_stride_a + n * n_stride_a + c * c_stride_a + th * Wt + tw
tile_offset_b = nd * nD_stride_b + d * d_stride_b + n * n_stride_b + c * c_stride_b + th * Wt + tw
```

The strides encode broadcasting: if a dimension has size 1 in the input, its stride is set to 0 (via the `(dim > 1)` multiplier in the program factory), causing the same tile to be re-read for all positions along that dimension.

For sharded tensors with non-full-width shards, `end_tw = start_tw + dst_shard_width` limits the width iteration, and the writer adjusts `dst_tile_offset` by `(Wt - dst_shard_width)` at each row boundary.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: One tile at a time, read via `noc_async_read_page` with immediate barrier. Sequential within each row of tiles (tw loop), then across rows (th), channels (c), batches (n, d, nd). Both A and B are read in lockstep in the ng reader.
- **Sharded**: All tiles are pre-resident in L1. The reader simply calls `cb_reserve_back` / `cb_push_back` on the full shard to make them available to compute.

### Write Pattern
- **Interleaved**: One tile at a time, written via `noc_async_write_page` with immediate barrier. Sequential tile ordering matching the read pattern.
- **Sharded**: Output CB is backed by the output buffer in L1 directly; no explicit writes needed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses full worker grid) |
| **Grid dimensions** | Device-dependent (e.g., 8x8 for Wormhole, 8x10 for Blackhole) |
| **Total cores** | Up to `worker_grid.num_cores()` |
| **Work per core** | `ceil(total_output_tiles / num_cores)` for group 1; remainder for group 2; 0 for unused cores |
| **Load balancing** | Two-group split via `split_work_to_cores`: core_group_1 gets `num_tiles_per_core_group_1`, core_group_2 gets `num_tiles_per_core_group_1 - 1` |

When sharding is active, the core grid is determined by the shard spec's grid, and each core processes exactly its shard's tiles. Cores outside the shard grid receive zero-tile arguments and return immediately.

An optimization path exists for grids that start at (0,0) (`zero_start_grid = true`), using faster `grid_to_cores` functions instead of the generic `corerange_to_cores`.

## Arguments

### Compile-Time Arguments

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Number of tiles processed per compute cycle (always 1) |

Additional compile-time defines (not indexed args):
- `BINARY_SFPU_INIT`: Expands to `add_binary_tile_init();` for ADD
- `BINARY_SFPU_OP`: Expands to `add_binary_tile` for ADD
- `BCAST_INPUT`: `""` for NONE broadcast, `"0"` or `"1"` for bcast variants
- `PROCESS_LHS_ACTIVATIONS(i)`, `PROCESS_RHS_ACTIVATIONS(i)`, `PROCESS_POST_ACTIVATIONS(i)`: Empty for plain ADD

**Reader Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (A) | mixed | Compile-time tensor accessor params for buffer A |
| N+1..M | TensorAccessorArgs (B) | mixed | Compile-time tensor accessor params for buffer B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs (C) | mixed | Compile-time tensor accessor params for output buffer C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Compute Kernel** (per-core):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of output tiles this core must process |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE; Ht*Wt for SCALAR; Wt for COL) |
| 2 | counter | uint32_t | Starting tile counter within the broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Quantization zero point (0 for non-quant ops like ADD) |

**Reader Kernel** (per-core, tensor+tensor case with ng reader):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Buffer address of tensor A |
| 1 | start_tile_id | uint32_t | Starting output tile ID (c_start_id) |
| 2 | a_num_tiles | uint32_t | Number of A tiles in shard (0 if interleaved) |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 5 | nD_stride | uint32_t | A stride for collapsed dims > 5 |
| 6 | d_stride | uint32_t | A stride for D dimension |
| 7 | n_stride | uint32_t | A stride for N dimension |
| 8 | c_stride | uint32_t | A stride for C dimension |
| 9 | D | uint32_t | Output D dimension size |
| 10 | N | uint32_t | Output N dimension size |
| 11 | C | uint32_t | Output C dimension size |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed dims > 5 |
| 15 | src_addr_b | uint32_t | Buffer address of tensor B |
| 16 | nD_stride_b | uint32_t | B stride for collapsed dims > 5 |
| 17 | d_stride_b | uint32_t | B stride for D dimension |
| 18 | n_stride_b | uint32_t | B stride for N dimension |
| 19 | c_stride_b | uint32_t | B stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of B tiles in shard (0 if interleaved) |

**Writer Kernel** (per-core, tensor+tensor case):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Buffer address of output tensor C |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output Ht |
| 8 | Wt | uint32_t | Output Wt |
| 9 | cND | uint32_t | Collapsed dims > 5 |
| 10 | (unused) | uint32_t | Set to 0 |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader (ReaderNoBcastNg) | RISCV_0 | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles in lockstep |
| Compute (eltwise_binary_sfpu_no_bcast) | RISCV_2 (math) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile (unpack to DEST), add_binary_tile (SFPU add), pack_tile |
| Writer (WriterNoBcastNg) | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (C) | Write output tiles |

### Reader Kernel (tensor+tensor, no broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads both A and B tiles in the same nested loop, one tile at a time in lockstep. Supports broadcasting via stride-based indexing (stride=0 for broadcast dimensions). For sharded inputs, simply marks all shard tiles as available via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel (tensor+tensor, no broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles one at a time via `noc_async_write_page`. Handles shard-width adjustments for block-sharded tensors where only a subset of columns belong to this core. For sharded output, the CB is backed directly by the output buffer, so no explicit writes occur.

### Compute Kernel

This section covers the compute kernel for the no-broadcast SFPU case, which is the default when both A and B have the same tile dimensions.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"        // Provides add_binary_tile, add_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"                  // Macro infrastructure: HAS_ACTIVATIONS, PREPROCESS, etc.
#include "eltwise_utils_sfpu.hpp"                    // SFPU-specific PREPROCESS macro (handles data format reconfig)

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);   // Runtime arg 0: total tiles to process on this core

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // Always 1 for this operation

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;    // CB for input A (pre-activation)
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;    // CB for input B (pre-activation)
    constexpr auto cb_out = tt::CBIndex::c_2;         // CB for output C

    // If LHS/RHS activations are defined, use intermediate CBs c_3/c_4; otherwise alias to pre CBs
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize unpacker and packer for the given CB pair
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // If RELU post-processing is fused into the packer, configure hardware RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // When no pre- or post-activations, init SFPU once before the loop for efficiency
    // For ADD: expands to add_binary_tile_init(); which calls llk_math_eltwise_binary_sfpu_binop_init<APPROX, ADD>()
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS macros: for plain ADD, these expand to nothing (no pre-activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);  // Wait for 1 LHS tile from reader

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);  // Wait for 1 RHS tile from reader

        cb_reserve_back(cb_out, num_tiles_per_cycle);     // Reserve space for 1 output tile

        // Re-init SFPU if pre-activations were applied (not the case for plain ADD)
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();  // Acquire DEST registers (blocks until available from packer)

        // Configure unpacker for LHS data format, then copy LHS tile to DEST[0]
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);   // Unpack LHS tile 0 from CB c_0 into DEST[0]
        }

        // Reconfigure unpacker for RHS data format, then copy RHS tile to DEST[1]
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // Unpack RHS tile 0 from CB c_1 into DEST[1]

            // Re-init SFPU per-tile if post-activations exist (not for plain ADD)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute SFPU binary ADD: DEST[0] = DEST[0] + DEST[1]
            // For ADD: expands to add_binary_tile(0, 1, 0)
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Apply post-activations if any (empty for plain ADD)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();   // Hand DEST registers to packer

        tile_regs_wait();     // Wait for packer to be ready

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // Pack DEST[0] result into output CB c_2
        }
        tile_regs_release();  // Release DEST registers for next iteration

        cb_push_back(cb_out, num_tiles_per_cycle);      // Signal writer: 1 output tile ready
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);  // Free LHS tile in CB
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);  // Free RHS tile in CB
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File

The SFPU kernel function resides in the LLK layer:
- **API header**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
- **LLK dispatch (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
- **Core SFPU implementation**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
- **Params dispatcher**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`

#### Annotated SFPU Kernel Source

**API layer** (`eltwise_binary_sfpu.h`):
```cpp
// Element-wise binary ADD on the SFPU
// idst0: index of first operand tile in DEST register buffer
// idst1: index of second operand tile in DEST register buffer
// odst: index of output tile in DEST register buffer (can overlap with idst0 or idst1)
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    // MATH() macro ensures this runs only on the math RISC-V (TRISC_MATH)
    // Dispatches to LLK with BinaryOp::ADD template parameter
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1, odst)));
}

ALWI void add_binary_tile_init() {
    // Initialize SFPU for ADD operation (no special init needed for ADD, unlike DIV/POW)
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::ADD>()));
}
```

**LLK dispatch layer** (`llk_math_eltwise_binary_sfpu_binop.h`):
```cpp
template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    // Calls the SFPU binary init function; for ADD, this is a no-op
    // (only DIV and POW need reciprocal/log initialization)
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>);
}

template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    // Dispatches to the params handler which manages face iteration
    // calculate_sfpu_binary<APPROX, ADD, 8, false> is the actual SFPU math function
    // 8 = ITERATIONS (processes 8 sub-blocks per face invocation)
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}
```

**Params dispatcher** (`llk_math_eltwise_binary_sfpu_params.h`):
```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    // Validate DEST indices are within bounds
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<...>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<...>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<...>()), "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0);  // Begin SFPU math phase

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    // Default mode is RC (Row-Column): process all 4 faces of the tile
    if (mode == VectorMode::RC)
    {
        // Process all 4 faces (top-left, top-right, bottom-left, bottom-right of 32x32 tile)
        // Each face is 16x16 elements
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            // Call the SFPU function which internally loops 8 times (8 rows of 64 elements)
            // For a 16x16 face: 8 iterations * 1 row/iteration * 64 elements covers the face
            // (actually processes 8 sub-rows, with dst_reg++ advancing through face data)
            sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
            // Advance DEST register pointer by 16 rows (2 * 8 via SETRWC)
            // to move to the next face
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }

    _llk_math_eltwise_binary_sfpu_done_();  // End SFPU math phase
}
```

**Core SFPU binary implementation** (`ckernel_sfpu_binary.h`):
```cpp
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    // SFPU microcode: iterate through 8 rows of the current face
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile occupies 32 rows in DEST when accessed via SFPI
        // (64 / SFP_DESTREG_STRIDE = 32)
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

        // Load one row (64 elements) from input tile 0 at current DEST position
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        // Load one row (64 elements) from input tile 1 at current DEST position
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = 0.0f;

        // Compile-time branch: for ADD, only the first branch is compiled
        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1;  // SFPU vector addition: adds 64 elements in parallel
                                 // Maps to sfpadd hardware instruction
        }
        // ... (other BINOP cases like SUB, MUL, DIV, RSUB, POW, XLOGY omitted)

        // Store result back to DEST at the output tile position
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance to next row within the face
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD: no initialization needed (no LUTs, no reciprocal setup)
    // Only DIV, POW, and XLOGY need special initialization
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // ADD, SUB, MUL, RSUB: no init needed
}
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[idx]` (read) | Loads a 64-element vector from DEST register at the given row index |
| `sfpi::dst_reg[idx]` (write) | Stores a 64-element vector result to DEST register at the given row index |
| `sfpi::dst_reg++` | Advances the DEST register row pointer by `SFP_DESTREG_STRIDE` (moves to next row) |
| `vFloat + vFloat` | Element-wise vector floating-point addition of 64 elements; maps to `sfpadd` hardware instruction |
| `TTI_SETRWC` | Sets the Read/Write Counter for DEST register access; used to advance between tile faces |

#### SFPU Register Usage

- **DEST registers**: Two input tiles are loaded into DEST at indices `dst_index_in0 * 32` and `dst_index_in1 * 32`. For the no-broadcast case, these are DEST[0] (LHS, tile index 0) and DEST[1] (RHS, tile index 1, at offset 32). The output overwrites DEST[0] (at `dst_index_out * 32`).
- **vFloat registers**: `in0` and `in1` are 64-element SFPU vector registers loaded from DEST. `result` holds the computed sum before writing back to DEST.
- **LREG (Local Registers)**: The SFPU has internal local registers (LREG[0-11]) used implicitly by the SFPI compiler for intermediate values. For simple addition, minimal LREG usage is needed.

#### SFPU Execution Flow

1. **Tile acquire**: `tile_regs_acquire()` obtains exclusive access to the DEST register buffer.
2. **Unpack LHS**: `copy_tile(cb_post_lhs, 0, 0)` unpacks LHS tile from CB c_0 into DEST[0]. With `UnpackToDestFp32` mode, data goes directly to DEST in FP32 precision.
3. **Unpack RHS**: `copy_tile(cb_post_rhs, 0, 1)` unpacks RHS tile from CB c_1 into DEST[1].
4. **SFPU compute**: `add_binary_tile(0, 1, 0)` is called:
   - The params dispatcher iterates over 4 faces (16x16 sub-tiles within the 32x32 tile).
   - For each face, `_calculate_sfpu_binary_<APPROX, ADD, 8>` runs 8 iterations.
   - Each iteration loads one row of 64 elements from DEST[0] and DEST[1], adds them, and writes the result back to DEST[0].
   - `dst_reg++` advances to the next row within the face.
   - `TTI_SETRWC` advances between faces.
   - Total: 4 faces x 8 iterations = 32 rows x 64 elements = 2048 elements... which covers a 32x32 tile (1024 elements) stored in DEST with stride 2 (face interleaving).
5. **Tile commit**: `tile_regs_commit()` signals the packer that DEST data is ready.
6. **Pack**: `pack_tile(0, cb_out)` reads from DEST[0] and packs the result into the output CB c_2.
7. **Tile release**: `tile_regs_release()` frees DEST for the next iteration.

#### SFPU Configuration

- **APPROX mode**: Compile-time template parameter; for ADD, approximation mode has no effect (addition is exact).
- **FP32 DEST accumulation**: Enabled when output is FLOAT32/INT32/UINT32, or when both inputs are FLOAT32/INT32/UINT32. Controlled by `fp32_dest_acc_en` in `ComputeConfig`.
- **UnpackToDestFp32**: For SFPU ADD (not POWER), all source CBs (c_0, c_1, c_3, c_4) are configured with `UnpackToDestMode::UnpackToDestFp32`, meaning `copy_tile` unpacks directly to DEST in FP32 precision without going through SrcA/SrcB registers.
- **No LUT or special init**: ADD requires no reciprocal tables, log tables, or other pre-computation. The init function `_sfpu_binary_init_<APPROX, ADD>()` is a no-op.

#### Hardware Compatibility Notes

- The SFPU binary add implementation is identical between Wormhole B0 and Blackhole architectures. Both use the same `ckernel_sfpu_binary.h` source from the tt_llk submodule.
- The params dispatcher (`llk_math_eltwise_binary_sfpu_params.h`) is also architecture-independent, using the same face iteration pattern.
- The `dst_tile_size_sfpi = 32` constant (64 / SFP_DESTREG_STRIDE = 32) is consistent across both architectures.

## Implementation Notes

1. **Unified factory**: The `binary_ng` program factory handles all binary operations through a single `ProgramFactory::create()` method. The specific operation (ADD) is selected via compile-time defines (`BINARY_SFPU_INIT`, `BINARY_SFPU_OP`) generated by `OpConfig::as_defines()`.

2. **Reader consolidation**: When both inputs are tensors, the `binary_ng` framework uses a single reader kernel (`ReaderNoBcastNg`) that reads both A and B tiles in lockstep, rather than having separate reader kernels. This reduces kernel overhead.

3. **Scalar path**: When one operand is a scalar, the writer kernel (`WriterScalar`) fills a single tile with the scalar value and pushes it to CB c_1 once. The compute kernel then reads this tile repeatedly for each output tile.

4. **Broadcast handling**: Broadcasting is handled through two mechanisms:
   - **Stride-based**: For dimension broadcasting (a dim of size 1 gets stride 0), the reader naturally re-reads the same tiles.
   - **Subtile broadcast**: For row/column broadcasting within a tile, specialized reader and compute kernels handle the sub-tile replication.

5. **Sharding optimization**: When tensors are sharded in L1, the reader/writer are essentially no-ops (just marking tiles as available). The CBs are backed directly by the tensor's L1 buffer, eliminating all DRAM/NoC traffic.

6. **No-op for idle cores**: Cores not assigned any tiles receive `num_tiles = 0` in their runtime args and either return immediately (compute) or do nothing (reader/writer with zero-element arrays).

7. **Integer type support**: For INT32/UINT32/UINT16 dtypes, the ADD dispatches to `add_int_tile` instead of `add_binary_tile`, which uses integer-specific SFPU operations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtypes (scalar, bcast, etc.) and how does it distribute work across cores?"
   **Reason**: Understanding the overall architecture of the binary_ng framework before diving into source code.
   **Key Findings**: Binary_ng uses a single unified ProgramFactory (unlike the older binary op which had multiple factories). SubtileBroadcastType enum controls kernel selection. Work distribution uses `split_work_to_cores`.

2. **Query**: "How does the SFPI programming interface work? Explain dst_reg access, vFloat, vInt types, the iteration model over tile faces, and what dst_reg++ does."
   **Reason**: Understanding the low-level SFPU execution model for the annotated kernel source.
   **Key Findings**: SFPU processes tiles in 8 iterations with `dst_reg++` advancing through rows. `vFloat` is a 64-element vector type. DEST register file has 512 entries. Tiles are divided into faces processed sequentially.

3. **Query**: "How do copy_tile and copy_tile_to_dst_init_short_with_dt work? What do tile_regs_acquire/commit/wait/release do? How does UnpackToDestFp32 work?"
   **Reason**: Understanding the compute kernel's register management and data movement patterns.
   **Key Findings**: `copy_tile` unpacks from CB to DEST. `tile_regs_*` functions manage exclusive access between unpacker/math/packer. `UnpackToDestFp32` bypasses SrcA/SrcB for direct FP32 precision in DEST.

### Confluence References

Not consulted for this analysis. The ADD operation uses a straightforward `sfpadd` instruction whose behavior is well-documented in DeepWiki and source code.

### Glean References

Not consulted for this analysis. No confidential hardware specifications were needed beyond what the open-source code provides.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Understanding how `BinaryOpType::ADD` maps to SFPU defines.
   **Key Information**: ADD maps to `SfpuBinaryOp::ADD`, which produces `add_binary_tile_init()` and `add_binary_tile` defines. No pre/post processing needed for plain ADD.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Understanding the actual SFPU microcode for binary addition.
   **Key Information**: ADD is implemented as `result = in0 + in1` using SFPI `vFloat` vector addition. No special initialization or approximation needed. 8 iterations per face, 4 faces per tile.
