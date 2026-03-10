# GCD (binary_ng) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the element-wise greatest common divisor of two integer tensors. It is implemented as an SFPU-only binary operation within the `binary_ng` framework. The operation requires INT32 input tensors and produces an INT32 output tensor. The underlying algorithm is the Binary GCD algorithm, implemented entirely in SFPU instructions with a replay-based loop for efficiency.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The compute kernel processes one output tile per read-compute-write cycle (`num_tiles_per_cycle = 1`). Each tile requires loading two input tiles (LHS and RHS) into DEST registers, running the SFPU GCD kernel across all 8 faces (ITERATIONS=8, processing 4 rows of 32 elements each per face), and packing the result back to the output circular buffer.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Tensor A (LHS) | Tensor B (RHS) |
|---|---|---|
| Dimension Convention | [nD..., D, N, C, H, W] (up to rank 6+, higher dims collapsed) | [nD..., D, N, C, H, W] |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (Height, Width, or Block) | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | INT32 | INT32 |

### Output Tensor(s)

| Property | Tensor C (Output) |
|---|---|
| Dimension Convention | [nD..., D, N, C, H, W] (broadcasted shape of A and B) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | INT32 |

### Layout Transformations

No tilize/untilize or format conversion is performed within the program factory. Both inputs and output must already be in tiled INT32 format. Broadcasting is handled by the reader kernel through stride manipulation -- dimensions of size 1 in an input are broadcast to match the output shape.

## Data Flow Pattern

1. **Reader kernel** reads one tile from tensor A into CB c_0, and one tile from tensor B into CB c_1. For interleaved memory, it uses `TensorAccessor` with NoC async reads; for sharded memory, it simply reserves/pushes the pre-loaded shard tiles.
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1, copies them into DEST registers (DEST[0] for LHS, DEST[1] for RHS), invokes `gcd_tile(0, 1, 0)` which runs the SFPU Binary GCD algorithm, then packs the result from DEST[0] into CB c_2.
3. **Writer kernel** reads the output tile from CB c_2 and writes it to the output tensor via NoC async writes (interleaved) or simply leaves it in L1 (sharded).

## Circular Buffer Configuration

| CB ID | Purpose | Capacity (tiles) | Data Format | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|
| c_0 | Input A (LHS) | 2 (interleaved) or shard volume (sharded) | Int32 | Double-buffered (interleaved) / Single (sharded) | Reader | Compute |
| c_1 | Input B (RHS) | 2 (interleaved) or shard volume (sharded) | Int32 | Double-buffered (interleaved) / Single (sharded) | Reader | Compute |
| c_2 | Output C | 2 (interleaved) or shard volume (sharded) | Int32 | Double-buffered (interleaved) / Single (sharded) | Compute | Writer |

**Notes**:
- CBs c_3 and c_4 (intermediate LHS/RHS activation buffers) are NOT created for GCD because GCD has no pre-activations or post-activations defined via `OpConfig`.
- CBs c_5 and c_6 (row broadcast buffers) are only created for ROW_A/ROW_B broadcast types, not relevant to the standard no-broadcast GCD path.

## Pipeline Pattern Summary

For interleaved tensors, CB capacity is 2 tiles with a block size of 1 tile, enabling **double-buffering** -- the reader can fill the next tile while compute processes the current one. For sharded tensors, the entire shard is loaded at once, so buffering is effectively single-shot.

## Index Calculations

The reader kernel decomposes the linear `start_tile_id` into multi-dimensional coordinates (nD, D, N, C, Ht, Wt) using the output tensor dimensions. Each input tensor has independent stride values (`nD_stride`, `d_stride`, `n_stride`, `c_stride`) that encode broadcasting: if a dimension has size 1 (broadcasted), its stride is 0. The reader iterates through the output tile grid and computes the corresponding input tile offsets using these strides.

For tensor A: `tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt`

Stride values are computed in the host as: `stride = product_of_inner_dims * (dim > 1)`, which yields 0 when the dimension is 1 (broadcast).

## Memory Access Patterns

### Read Pattern

Sequential tile reads within the innermost dimension (Wt), with stride-based offset jumps at each higher dimension boundary. For interleaved memory, each tile is read individually via `noc_async_read_page`. For sharded memory, all tiles are pre-loaded in L1 and the reader simply signals availability via `cb_reserve_back`/`cb_push_back`.

### Write Pattern

Sequential tile writes matching the output tensor's linear tile order. For interleaved memory, each tile is written individually via `noc_async_write_page` with a write barrier. For sharded memory, the output is already in L1 and no explicit writes occur.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | Output tiles evenly divided across cores via `split_work_to_cores` |
| Load Balancing | Two core groups: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |
| Remainder Handling | Cores outside both groups receive zero-filled runtime args and exit immediately |
| Sharded Mode | Core grid comes from the shard spec; each core processes its own shard |

## Arguments

### Compile-Time Arguments

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `num_tiles_per_cycle` | uint32_t | Number of tiles processed per compute cycle (always 1) |

**Reader Kernel**: TensorAccessor compile-time args for tensor A and tensor B, plus a `has_sharding` flag.

**Writer Kernel**: TensorAccessor compile-time args for tensor C, plus a `has_sharding` flag.

### Runtime Arguments

**Compute Kernel** (per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `num_tiles` | uint32_t | Total tiles assigned to this core |
| 1 | `freq` | uint32_t | Broadcast frequency (1 for NONE subtype) |
| 2 | `counter` | uint32_t | Broadcast start counter (0 for NONE subtype) |
| 3 | `compute_scalar_value` | uint32_t | Unused for GCD (set to 0) |

**Reader Kernel** (per core, 21 args):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `src_addr` | uint32_t | Tensor A DRAM/L1 base address |
| 1 | `start_tile_id` | uint32_t | Starting output tile index for this core |
| 2 | `src_num_tiles` | uint32_t | Number of A shard tiles (sharded only) |
| 3 | `dst_num_tiles` | uint32_t | Number of output tiles for this core |
| 4 | `dst_shard_width` | uint32_t | Shard width in tiles (sharded only) |
| 5-8 | `nD_stride`, `d_stride`, `n_stride`, `c_stride` | uint32_t | A strides (0 if broadcast) |
| 9-14 | `D`, `N`, `C`, `Ht`, `Wt`, `cND` | uint32_t | Output tensor dimensions |
| 15 | `src_addr_b` | uint32_t | Tensor B DRAM/L1 base address |
| 16-19 | `nD_stride_b`, `d_stride_b`, `n_stride_b`, `c_stride_b` | uint32_t | B strides |
| 20 | `src_num_tiles_b` | uint32_t | Number of B shard tiles (sharded only) |

**Writer Kernel** (per core, 11 args for tensor-tensor mode):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Output tensor DRAM/L1 base address |
| 1 | `start_tile_id` | uint32_t | Starting output tile index |
| 2 | `dst_num_tiles` | uint32_t | Number of output tiles for this core |
| 3 | `dst_shard_width` | uint32_t | Shard width in tiles |
| 4-8 | `D`, `N`, `C`, `Ht`, `Wt` | uint32_t | Output dimensions |
| 9 | `cND` | uint32_t | Collapsed high dimensions |
| 10 | (reserved) | uint32_t | Set to 0 |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads tiles from both tensor A (into CB c_0) and tensor B (into CB c_1) in a single reader kernel. Uses nested loops over the output tensor dimensions (nD, D, N, C, Ht, Wt) with stride-based indexing to handle broadcasting. Supports both interleaved (NoC reads) and sharded (pre-loaded L1) memory configurations via `SRC_SHARDED` / `SRC_SHARDED_B` compile-time defines.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles from CB c_2 to DRAM/L1. Uses the same dimensional iteration pattern as the reader. For sharded outputs (`DST_SHARDED`), the kernel body is effectively empty since data is already in the correct L1 location.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

This is the kernel selected for `SubtileBroadcastType::NONE` with `is_sfpu = true` (i.e., `KernelName::ComputeNoBcast` mapped via `get_kernel_file_path`).

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
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
#include "api/compute/gcd.h"              // Provides gcd_tile_init() and gcd_tile()
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"        // Macro infrastructure: HAS_ACTIVATIONS, PROCESS_ACTIVATIONS, etc.
#include "eltwise_utils_sfpu.hpp"          // PREPROCESS macro for activation preprocessing

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // runtime arg 0: total tiles for this core

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;   // CB for input A (raw from reader)
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;   // CB for input B (raw from reader)
    constexpr auto cb_out = tt::CBIndex::c_2;        // CB for output

    // If LHS/RHS activations are defined, use intermediate CBs; otherwise alias to raw CBs.
    // For GCD, no activations are defined, so cb_post_lhs == cb_pre_lhs, cb_post_rhs == cb_pre_rhs.
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);  // Initialize unpack/pack pipeline for the given CB pair
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // Not used for GCD
#endif

    // For GCD: no LHS/RHS/POST activations, so BINARY_SFPU_INIT runs here once before the loop.
    // This expands to: gcd_tile_init() which calls llk_math_eltwise_binary_sfpu_gcd_init<APPROX>()
    // That in turn calls calculate_sfpu_gcd_init() which programs the REPLAY buffer with 4 iterations
    // of the inner GCD loop (7 instructions each = 28 instructions recorded).
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT   // Expands to: gcd_tile_init();
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS is a no-op for GCD (no LHS activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);  // Wait for reader to produce 1 LHS tile

        // PREPROCESS is a no-op for GCD (no RHS activations)
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);  // Wait for reader to produce 1 RHS tile

        cb_reserve_back(cb_out, num_tiles_per_cycle);  // Reserve space in output CB for 1 tile

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT   // Not taken for GCD
#endif
        tile_regs_acquire();  // Acquire exclusive access to DEST register file

        // Configure unpack pipeline for LHS: set data format context so copy_tile reads from cb_post_lhs
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // Unpack LHS tile 0 from CB c_0 into DEST[0]
        }

        // Reconfigure unpack pipeline for RHS
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // Unpack RHS tile 0 from CB c_1 into DEST[1]

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT   // Not taken for GCD
#endif
            // Execute the SFPU GCD operation: gcd_tile(idst0=0, idst1=1, odst=0)
            // This dispatches the Binary GCD algorithm on DEST[0] and DEST[1], storing result in DEST[0].
            // Internally calls llk_math_eltwise_binary_sfpu_gcd which invokes calculate_sfpu_gcd
            // across all 4 faces of the tile (8 iterations total, each processing 4 rows of 32 elements).
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);  // Expands to: gcd_tile(0, 1, 0);
            PROCESS_POST_ACTIVATIONS(i * 2);  // No-op for GCD (no post-activations)
        }
        tile_regs_commit();  // Signal that DEST writes are complete, math pipeline can proceed

        tile_regs_wait();  // Wait for math pipeline to finish before packing

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // Pack DEST[0] into output CB c_2
        }
        tile_regs_release();  // Release DEST register file

        cb_push_back(cb_out, num_tiles_per_cycle);       // Signal writer that 1 output tile is ready
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);  // Free LHS tile slot in CB c_0
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);  // Free RHS tile slot in CB c_1
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
(Blackhole version is identical: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// The main body of the Binary GCD algorithm, executed on the SFPU.
// This function implements the setup phase (ensuring b is odd, computing common
// factor of 2) and then the iterative reduction phase using REPLAY for efficiency.
//
// Template parameter max_input_bits controls the number of iterations (default 31
// for full INT32 range). The algorithm needs at most max_input_bits iterations,
// but optimizations reduce this to 30 effective iterations.
template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    // --- Phase 1: Compute trailing zeros of (a | b) to extract common factor of 2 ---

    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);    // LREG2 (c) = LREG0 (a); copy a into temporary
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);      // LREG2 (c) |= LREG1 (b); c = a | b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);     // LREG3 (d) = c; save a|b
    // Negate d: d = -d (two's complement negation via adding 0 with 2SCOMP flag on DST)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);     // d &= c; isolates lowest set bit of (a|b)
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);      // d = clz(d); count leading zeros to get bit position

    // --- Phase 2: Ensure b is odd by conditionally swapping a and b ---

    // Shift b left by d positions; if result is 0, b has no bits in the positions where a|b had trailing zeros,
    // meaning b is even relative to the common factor. In that case, swap a and b.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);                // Set condition codes: true if c == 0 (b was even)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);    // Conditionally swap a and b (only in lanes where CC is set)
    TTI_SFPENCC(0, 0, 0, 0);                              // Disable conditional execution (re-enable all lanes)

    // Take absolute values of both operands (GCD is defined on non-negative integers)
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);     // a = |a|
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);     // b = |b|

    // Negate a and d for use in the iterative loop
    // -a is stored in LREG0; the loop uses {-a, a} pair for efficient LSB isolation
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a
    // -d is the right-shift amount used to remove trailing zeros in each iteration
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d

    // --- Phase 3: Iterative Binary GCD reduction using REPLAY ---

    // The REPLAY buffer was pre-loaded during calculate_sfpu_gcd_init() with 7 instructions
    // that form one iteration of the Binary GCD step. Each TTI_REPLAY(0, count, 0, 0) replays
    // `count` instructions from the buffer.

    int iterations = max_input_bits - 1;  // 30 iterations for 31-bit inputs

    // Replay in chunks of 4 iterations (28 instructions each)
    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);  // Replay 4 iterations = 28 instructions from the REPLAY buffer
        iterations -= 4;
    }

    // Replay the remaining iterations. The worst case for 31-bit inputs is 31 iterations,
    // but we skip the final iteration (it only affects a, not the GCD result in b).
    // We also skip the last instruction of the penultimate iteration (same reason).
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);

    TTI_SFPENCC(0, 0, 0, 0);  // Ensure all lanes are re-enabled after any conditional execution
}

// Top-level SFPU GCD function called by the LLK dispatch layer.
// Processes all 8 faces of a tile pair (ITERATIONS=8 by default).
// Each face consists of 4 rows of 32 elements processed by 32 SIMD lanes.
template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile in DEST occupies 64 rows (32x32 = 1024 elements stored as 64 rows of 16 elements
        // in the DEST register file, but addressed with dst_tile_size=64 offset)
        constexpr uint dst_tile_size = 64;

        // Load one face of input tile A from DEST[dst_index_in0] into LREG0
        // Mode 4 = int32 raw load, addressing mode 3 = use dst_tile_size offset
        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a
        // Load one face of input tile B from DEST[dst_index_in1] into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b

        // Execute the Binary GCD algorithm body for 31-bit integers
        calculate_sfpu_gcd_body<31>();

        // Store the GCD result (in LREG1 = b, which holds gcd(|a|,|b|) after convergence)
        // back to DEST[dst_index_out]
        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size);
        dst_reg++;  // Advance to the next face (next 4 rows of 32 elements)
    }
}

// Initialization function called once before the main tile loop.
// Programs the REPLAY buffer with 4 iterations of the inner Binary GCD step.
// The REPLAY buffer can hold up to 32 instructions; 4 iterations x 7 instructions = 28 entries.
inline void calculate_sfpu_gcd_init() {
    // TTI_REPLAY with mode 1 starts RECORDING into the replay buffer.
    // The next 28 (7*4) instructions will be captured.
    TTI_REPLAY(0, 7 * 4, 0, 1);

    // Record 4 iterations of the Binary GCD inner loop into the REPLAY buffer.
    // Each iteration performs: isolate LSB of a -> count leading zeros -> shift to make a odd ->
    // ensure b <= a -> compute a = b - a (making a even for next iteration)
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // LREG0 holds -a from previous iteration. Take abs to get +a in LREG2.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);     // LREG2 = |LREG0| = +a

        // AND -a with +a to isolate the lowest set bit of a (LSB trick: -a & a = lowest set bit)
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);      // LREG0 = (-a) & (+a) = lowest set bit

        // Count leading zeros of the isolated bit to get its position.
        // SFPLZ_MOD1_CC_NE0 disables lanes where a == 0 (GCD is already found in those lanes).
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0);

        // Add d (the common trailing-zero shift) to the leading-zero count.
        // This gives the total right-shift needed to make a odd.
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE);  // LREG0 += d

        // Right-shift +a by the computed amount to strip all trailing zeros, making a odd.
        // SFPSHFT2 with SHFT_LREG mode: LREG0 = LREG2 >> (-LREG0) since LREG0 is negative.
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG);

        // Swap so that b <= a (min in LREG1, max in LREG0) using hardware min/max swap.
        // After this: LREG1 = min(a, b), LREG0 = max(a, b)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);

        // Compute a = b - a. Since b <= a (after swap), result is <= 0.
        // ARG_2SCOMP_LREG_DST negates LREG0 before adding LREG1: LREG0 = LREG1 + (-LREG0) = b - a
        // The result is non-positive, which means -a is stored in LREG0 (convenient for next iteration).
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    }
    // After this function returns, the REPLAY buffer contains 28 instructions (4 iterations of 7).
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `SFPMOV` | Lanewise register move: copies one LREG to another |
| `SFPOR` | Bitwise OR between two LREGs |
| `SFPAND` | Bitwise AND between two LREGs |
| `SFPIADD` | Integer add/subtract with optional two's complement negation of destination; can set condition codes |
| `SFPLZ` | Count leading zeros of an LREG value; optionally sets per-lane condition codes for zero detection |
| `SFPSHFT2` | Barrel shift: shifts one LREG by an amount specified in another LREG (signed: positive=left, negative=right) |
| `SFPSETCC` | Sets per-lane condition flags based on comparisons (e.g., == 0, != 0, < 0, >= 0) |
| `SFPSWAP` | Swaps two LREGs; in `VEC_MIN_MAX` mode, assigns min to one and max to the other |
| `SFPENCC` | Enable/disable conditional lane execution based on per-lane flags |
| `SFPABS` | Computes absolute value (two's complement for integers) |
| `SFPLOAD` | Loads data from DEST register file into an LREG (mode 4 = int32 raw) |
| `SFPSTORE` | Stores data from an LREG back to DEST register file |
| `REPLAY` | Records or replays a sequence of instructions from the hardware replay buffer (up to 32 entries) |

#### SFPU Register Usage

| Register | Role in GCD |
|---|---|
| `LREG0` | Holds `-a` (negated first operand) during the iterative loop; used for LSB isolation |
| `LREG1` | Holds `b` (second operand); after convergence, contains the GCD result |
| `LREG2` | Temporary: holds `+a` (absolute value of a), intermediate computation results |
| `LREG3` | Holds `d` = negated count of common trailing zeros in `a|b` (the common power-of-2 factor); used as shift amount |
| `LCONST_0` | Constant 0, used for negation via `SFPIADD` with `ARG_2SCOMP_LREG_DST` |

#### SFPU Execution Flow

1. **Initialization** (`calculate_sfpu_gcd_init`, called once):
   - Begins recording into the REPLAY buffer (`TTI_REPLAY(..., 1)` in record mode).
   - Emits 4 unrolled iterations of the inner GCD loop (7 instructions each = 28 total), which are captured into the replay buffer.
   - Each recorded iteration: `abs(a)` -> `isolate LSB` -> `clz` -> `add shift offset` -> `shift right to make odd` -> `min/max swap` -> `subtract`.

2. **Per-face execution** (`calculate_sfpu_gcd`, called per tile, 8 iterations):
   - **Load**: `SFPLOAD` reads one face (4 rows x 32 lanes) of tile A into LREG0 and tile B into LREG1 from DEST registers.
   - **Body** (`calculate_sfpu_gcd_body<31>`):
     - Computes `c = a | b`, isolates the lowest set bit, counts leading zeros to find the common trailing-zero factor `d`.
     - Conditionally swaps a and b to ensure b is odd.
     - Takes absolute values of both operands.
     - Negates a and d for the iterative loop.
     - Replays the recorded loop: 7 chunks of 4 iterations (28 instructions each) for 28 iterations, then a final partial replay for the remaining 2 iterations minus 1 instruction, totaling 30 effective iterations.
   - **Store**: `SFPSTORE` writes the GCD result from LREG1 back to DEST at the output tile location.
   - `dst_reg++` advances to the next face.

3. **Result**: After all 8 faces, DEST[odst] contains the complete GCD tile.

#### SFPU Configuration

- **Unpack-to-DEST mode**: `UnpackToDestFp32` is set for CBs c_0, c_1, c_3, c_4 (since GCD is not `BinaryOpType::POWER`, the non-POWER SFPU path is taken). This ensures INT32 data is unpacked directly into DEST as 32-bit values.
- **FP32 DEST accumulation**: `fp32_dest_acc_en = true` because both input and output data formats are Int32.
- **APPROX template parameter**: Passed through from the LLK layer; for GCD, the approximation mode does not affect the algorithm (it is exact integer computation).
- **Compile-time defines**: `BINARY_SFPU_INIT` -> `gcd_tile_init();`, `BINARY_SFPU_OP` -> `gcd_tile`.
- **No pre/post activations**: GCD does not define `process_lhs`, `process_rhs`, or `postprocess` in `OpConfig`, so all activation macros expand to no-ops.

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of the GCD SFPU kernel (`ckernel_sfpu_gcd.h`) are **identical**. Both architectures support all the required instructions (`SFPSWAP` with `VEC_MIN_MAX`, `SFPLZ`, `SFPSHFT2` with `SHFT_LREG`, `REPLAY`). The only known architectural difference relevant to this kernel is that Blackhole fixed a hardware bug in `SFPSHFT2_MOD1_SUBVEC_SHFLSHR1` mode, but that mode is not used here (only `SHFT_LREG` mode is used).

## Implementation Notes

1. **Binary GCD Algorithm Choice**: The implementation uses the binary GCD algorithm rather than the Euclidean algorithm because the binary GCD avoids division operations, which are expensive on the SFPU. Instead, it uses only shifts, comparisons, and subtractions -- all of which map efficiently to SFPU instructions.

2. **REPLAY Buffer Optimization**: The 7-instruction inner loop is recorded into the hardware REPLAY buffer during `gcd_tile_init()`. This avoids re-fetching the same instructions from the RISC-V instruction stream on every iteration, significantly reducing instruction bandwidth pressure. The REPLAY buffer holds 32 entries; 28 are used (4 iterations x 7 instructions).

3. **30 Iterations for 31-bit Inputs**: The algorithm is proven to converge in at most `max_input_bits` iterations. The implementation executes 30 iterations instead of 31, exploiting the fact that the 31st iteration would only modify `a` (which is discarded), and the final instruction of iteration 30 also only affects `a`.

4. **Per-lane Convergence via Conditional Execution**: The `SFPLZ` instruction with `SFPLZ_MOD1_CC_NE0` mode disables lanes where `a == 0` (meaning convergence has been reached for that element). The `SFPENCC` instruction at the end re-enables all lanes. This allows different SIMD lanes to converge at different rates without affecting correctness.

5. **The LSB Isolation Trick**: The pattern `(-a) & a` isolates the lowest set bit of `a`. Combined with `clz`, this efficiently computes the number of trailing zeros, which tells us exactly how many times to right-shift `a` to make it odd.

6. **MIN/MAX Swap**: `SFPSWAP` with `VEC_MIN_MAX` mode is a hardware-supported simultaneous min/max operation that ensures `b <= a` after each iteration, which is required by the binary GCD algorithm.

7. **INT32-only Operation**: GCD is only meaningful for integer types. The program factory forces `is_sfpu = true` for GCD (it has no FPU path), and the `UnpackToDestFp32` mode ensures proper 32-bit integer handling through the unpack-SFPU-pack pipeline.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtypes (scalar, bcast, no-bcast) and how does it select kernels?"
   **Reason**: Needed to understand the overall binary_ng framework architecture and kernel selection logic.
   **Key Findings**: binary_ng supports NONE, SCALAR, ROW, COL, and mixed broadcast types. Kernel selection is based on `SubtileBroadcastType` enum. GCD uses `ComputeNoBcast` for no-broadcast and `ComputeBcast` for broadcast cases.

2. **Query**: "What is the GCD SFPU operation in binary_ng?"
   **Reason**: Needed to understand how GCD maps to SFPU kernel functions.
   **Key Findings**: GCD maps to `gcd_tile`/`gcd_tile_init` which call `llk_math_eltwise_binary_sfpu_gcd`. The core algorithm is in `ckernel_sfpu_gcd.h` using the Binary GCD algorithm.

3. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do?" (tt-llk repo)
   **Reason**: Needed to understand the LLK dispatch layer between the compute API and the SFPU kernel.
   **Key Findings**: It dispatches the SFPU function across tile faces (4 faces for RC mode), handles DEST address setup, and calls the SFPU function with ITERATIONS=8 (8 faces per tile).

4. **Query**: "SFPU instruction descriptions: SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS" (tt-isa-documentation repo)
   **Reason**: Needed precise instruction semantics for annotating the SFPU kernel.
   **Key Findings**: Detailed per-instruction behavior including SFPSWAP's VEC_MIN_MAX mode, SFPLZ's condition code modes, SFPIADD's two's complement negation flag.

5. **Query**: "What is the REPLAY instruction in the SFPU?"
   **Reason**: The GCD kernel heavily relies on REPLAY for loop optimization.
   **Key Findings**: REPLAY is part of the Tensix Coprocessor frontend Replay Expander, not the SFPU itself. It records sequences of up to 32 instructions and replays them without re-fetching from RISC-V.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Maps `BinaryOpType::GCD` to `SfpuBinaryOp::GCD` and defines the init/op function names.
   **Key Information**: GCD init = `gcd_tile_init();`, GCD op = `gcd_tile`.

2. **Source**: `tt_metal/hw/inc/api/compute/gcd.h`
   **Reason**: Public compute API header defining `gcd_tile` and `gcd_tile_init`.
   **Key Information**: `gcd_tile(idst0, idst1, odst)` dispatches via `llk_math_eltwise_binary_sfpu_gcd<APPROX>`.
