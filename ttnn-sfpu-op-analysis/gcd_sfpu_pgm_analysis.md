# GCD (element_wise_multi_core_sfpu) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the element-wise greatest common divisor of two integer tensors: `output[i] = gcd(a[i], b[i])`. It is a binary SFPU operation that operates on Int32 data. The implementation uses the Binary GCD algorithm (also known as Stein's algorithm), which avoids division by using only subtraction, absolute value, leading-zero count, and bitwise operations -- all of which map efficiently to SFPU instructions.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `block_size` tiles (dynamically computed; 1 tile for interleaved, up to `find_max_block_size(num_tiles_per_shard)` for sharded) |
| **Total units** | `per_core_block_cnt` blocks per core, where `per_core_block_cnt * per_core_block_size = num_tiles_per_core` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles within each block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (must match B) | Arbitrary (must match A) |
| **Dimension convention** | NHWC | NHWC |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 | INT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as inputs |
| **Dimension convention** | NHWC |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 |

### Layout Transformations

No layout transformations (tilize/untilize) are performed within the program factory. Input and output tensors are expected to already be in TILE_LAYOUT. The `UnpackToDestMode::UnpackToDestFp32` mode is enabled for all input CBs (since GCD is not `BinaryOpType::POWER`), which ensures 32-bit precision is maintained in the DEST registers during SFPU computation.

## Data Flow Pattern

1. **Reader kernel** reads tiles from input tensor A into `cb_in0` (c_0) and from input tensor B into `cb_in1` (c_1), one tile at a time via NoC async reads. If inputs are sharded, the reader simply does `cb_reserve_back`/`cb_push_back` to make the pre-existing L1 shard data visible.
2. **Compute kernel** waits for both `cb_in0` and `cb_in1` to have `per_core_block_size` tiles available. For each tile `i` in the block:
   - Copies tile from `cb_in0` position `i` into DEST register `i*2` (even slot)
   - Copies tile from `cb_in1` position `i` into DEST register `i*2+1` (odd slot)
   - Calls `gcd_tile_init()` (via `BINOP_INIT` define) to set up the SFPU replay buffer
   - Calls `gcd_tile(i*2, i*2+1, i*2)` (via `BINARY_SFPU_OP` define) to compute GCD and store the result back in DEST register `i*2`
   - Packs the result from DEST `i*2` into `cb_out0` (c_2)
3. **Writer kernel** waits for tiles in `cb_out0` and writes them to the output buffer in DRAM/L1 via NoC async writes. If output is sharded, the writer simply waits for data in the CB (which is backed by the output buffer directly).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (reader produces 1 at a time) | Double (interleaved) / Full-shard (sharded) | Reader | Compute | Block |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (reader produces 1 at a time) | Double (interleaved) / Full-shard (sharded) | Reader | Compute | Block |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (writer consumes 1 at a time) | Double (interleaved) / Full-shard (sharded) | Compute | Writer | Block |

**Note**: For GCD, the interim circular buffers c_3 and c_4 are NOT created because GCD does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` (no input pre-scaling is needed).

## Pipeline Pattern Summary

- **Interleaved mode**: All CBs have capacity `2 * max_block_size` with block size `max_block_size`, enabling **double-buffering**. The reader can fill one buffer while the compute processes the other.
- **Sharded mode**: CBs are backed by the tensor's L1 buffer directly (globally allocated), so the entire shard is available at once. This is effectively **single-buffered** since the data is already in place.

## Index Calculations

- **Interleaved**: The reader uses a simple linear tile ID starting from `start_id` (assigned per core) and increments sequentially. The writer does the same.
- **Block/Width sharded**: The reader traverses tiles in a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns. The start tile ID is computed as: `start_id = (core_row * block_height * block_width * num_shards_per_width) + (core_col * block_width)`.
- **TensorAccessor**: Used for DRAM address computation in both reader and writer kernels. Compile-time args encode the accessor configuration.
- **DEST register mapping**: Input A tile `i` goes to DEST slot `i*2`, input B tile `i` goes to DEST slot `i*2+1`. Output is taken from DEST slot `i*2` (overwrites input A's slot).

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads from `start_id` to `start_id + num_tiles - 1`. Each tile read is a NoC async read followed by a barrier before pushing to CB.
- **Sharded**: No reads required; data is already in L1. The reader just makes it visible via `cb_reserve_back`/`cb_push_back`.

### Write Pattern
- **Interleaved**: Sequential tile writes from `start_id`. Each tile write waits for CB data, reads from CB, issues NoC async write, then pops.
- **Sharded**: No writes required; output CB is backed by output buffer directly. Writer just waits for all tiles.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (interleaved) or 2D (sharded, matching shard grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | Up to all available compute cores |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles. Uses `split_work_to_cores` for interleaved. |

The runtime args function supports both a fast path (zero-start rectangular grid) and a general path (arbitrary CoreRangeSet). The fast path is used when the worker grid is a single rectangle starting at core (0,0) and any shard grid also starts at (0,0).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | uint32_t[] | Accessor config for input A (only if not IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | uint32_t[] | Accessor config for input B (only if not IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | uint32_t[] | Accessor config for output buffer |

#### Compute Kernel

No explicit compile-time arguments. Configuration is done via preprocessor defines:
- `BINOP_INIT` = `gcd_tile_init();`
- `BINARY_SFPU_OP` = `gcd_tile(i*2, i*2+1, i*2);`
- `fp32_dest_acc_en` = true (since output is Int32)
- `UnpackToDestMode::UnpackToDestFp32` for all input CBs

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Input tensor A buffer address |
| 1 | src1_addr | uint32_t | Input tensor B buffer address |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width (0 if interleaved) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

#### Writer Kernel (interleaved, non-block-sharded output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_binary_interleaved_start_id | RISCV_0 | NOC0 | DRAM/L1 | CB c_0, c_1 | Read input tiles A and B |
| eltwise_binary_sfpu_kernel | RISCV_2 (math) | N/A | CB c_0, c_1 | CB c_2 | copy_tile, gcd_tile, pack_tile |
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Supports four modes via `#ifdef` directives: both sharded, src0-only sharded, src1-only sharded, or both interleaved. For interleaved, it iterates through tile IDs and uses `noc_async_read_tile` via TensorAccessor. For sharded, it just does `cb_reserve_back`/`cb_push_back` since data is already in L1. The `block_or_width_sharded` compile-time flag enables a 2D traversal pattern (row x column) vs. a simple linear traversal.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard unary writer pattern. Iterates over tiles linearly, waits for each tile in CB, writes via `noc_async_write_page` using TensorAccessor. For sharded output, just does `cb_wait_front` for all tiles (data is already in the output buffer).

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
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
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"          // provides gcd_tile() and gcd_tile_init()
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if either input requires pre-processing (e.g. exp before add for logaddexp).
// For GCD, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // runtime arg: number of blocks
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // runtime arg: tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input B circular buffer

    // For GCD, SFPU_OP_INIT_PRE_IN0_0 is not defined, so cb_inp0 == cb_in0
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;
#endif

    // For GCD, SFPU_OP_INIT_PRE_IN1_0 is not defined, so cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // output circular buffer

    unary_op_init_common(cb_in0, cb_out0);  // initializes unpack/pack pipeline for the given CB pair

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE_SCALE sections are skipped for GCD (no input pre-processing) ---

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

        // (SFPU_OP_INIT_PRE_IN0_0 and SFPU_OP_INIT_PRE_IN1_0 blocks not active for GCD)

        // Wait for both inputs to have per_core_block_size tiles ready
        cb_wait_front(cb_inp0, per_core_block_size);  // blocks until reader has produced tiles in cb_in0
        cb_wait_front(cb_inp1, per_core_block_size);  // blocks until reader has produced tiles in cb_in1
        cb_reserve_back(cb_out0, per_core_block_size); // reserve space in output CB for the block

        tile_regs_acquire();  // acquire exclusive access to DEST registers
        tile_regs_wait();     // wait for DEST registers to be ready (pack done)

        // Initialize copy_tile for input B's data type, reading from cb_inp1
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy tile i from cb_inp0 (input A) to DEST register slot i*2 (even)
            copy_tile(cb_inp0, i, i * 2);
        }

        // Switch copy_tile config to input A's data type for reading from cb_inp1
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy tile i from cb_inp1 (input B) to DEST register slot i*2+1 (odd)
            copy_tile(cb_inp1, i, i * 2 + 1);

            // For GCD, the BINOP_INIT define expands to: gcd_tile_init();
            // This records 4 iterations of the inner GCD loop body into the SFPU replay buffer.
#ifdef BINOP_INIT
            BINOP_INIT   // gcd_tile_init();
#endif
            // (ADD_INT_INIT, SUB_INT_INIT, etc. are not defined for GCD)

            // For GCD, BINARY_SFPU_OP expands to: gcd_tile(i*2, i*2+1, i*2);
            // This executes the Binary GCD algorithm on DEST[i*2] and DEST[i*2+1],
            // storing the result back in DEST[i*2].
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP   // gcd_tile(i*2, i*2+1, i*2);
#endif
            // (SFPU_OP_INIT_0 and SFPU_OP_CHAIN_0 are not defined for GCD)

            // Pack the result from DEST register i*2 into the output CB
            pack_tile(i * 2, cb_out0);
        }

        tile_regs_commit();   // signal that DEST register writes are complete
        tile_regs_release();  // release DEST registers for next acquisition

        cb_pop_front(cb_inp0, per_core_block_size);  // free input A tiles
        cb_pop_front(cb_inp1, per_core_block_size);  // free input B tiles
        cb_push_back(cb_out0, per_core_block_size);  // publish output tiles for writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`

Both files are identical in implementation.

#### LLK Wrapper File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h`

The LLK wrapper calls `_llk_math_eltwise_binary_sfpu_params_` which iterates over 4 tile faces (each 16x16) in `VectorMode::RC` mode, calling `calculate_sfpu_gcd` for each face. Between faces, it advances the DEST register read/write pointer via `TTI_SETRWC`.

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

// calculate_sfpu_gcd_body: Core of the Binary GCD algorithm for 31-bit signed integers.
// Operates on LREG0 (a) and LREG1 (b) across all 32 SIMD lanes simultaneously.
// On entry: LREG0 = a, LREG1 = b (both signed int32, loaded from DEST).
// On exit: LREG1 = gcd(|a|, |b|).
//
// The algorithm:
//   1. Compute trailing zeros of (a | b) to find the common factor of 2.
//   2. Ensure b is odd (swap with a if needed).
//   3. Take absolute values, negate a, negate the trailing-zero count.
//   4. Loop 30 times (sufficient for 31-bit inputs):
//      a. Strip factors of 2 from a (using abs, AND, clz, shift).
//      b. Ensure b <= a via min/max swap.
//      c. Compute a = b - a (making a even again for next iteration).
//   5. Result is in LREG1 (b), which is the GCD divided by the common power of 2.
//      The caller would need to shift back, but the algorithm as used here works on the
//      full integer representation.

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    // Step 1: Find the largest power of 2 dividing both a and b.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);   // LREG2 (c) = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);     // c = a | b (set all bits present in either)

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);    // LREG3 (d) = c
    // d = -d (two's complement negation: LREG3 = 0 - LREG3)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    // d = d & c = (-c) & c, which isolates the lowest set bit of c (the common trailing-zero factor)
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    // d = clz(d): count leading zeros of the isolated LSB gives (31 - trailing_zeros) of (a|b)
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);

    // Step 2: Ensure b is odd by conditionally swapping a and b.
    // c = b << d (left-shift b by the leading-zero count of the isolated LSB)
    // If c == 0, then b is even and a is odd, so swap them.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG);
    // Set condition code: lanes where c == 0 (b was even)
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);
    // Conditional swap: in lanes where b was even, swap a and b so b becomes odd
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    // Disable conditional execution (clear lane flags)
    TTI_SFPENCC(0, 0, 0, 0);

    // Step 3: Take absolute values and prepare for iterative loop.
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);   // a = |a| (integer absolute value)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);   // b = |b|

    // Negate a (for the subtraction step in the loop: a = b - a is done as a = b + (-a))
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    // Negate d (the shift count), so we can use it as a right-shift amount
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);

    // Step 4: Main iteration loop.
    // Each iteration is 7 instructions (recorded in the replay buffer by calculate_sfpu_gcd_init).
    // We need 30 iterations total for 31-bit inputs.
    // The #pragma unroll produces 7 iterations of 4 replays each = 28 iterations.
    int iterations = max_input_bits - 1;  // 30

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        // Replay 4 iterations (7 instructions each = 28 instructions) from the replay buffer
        TTI_REPLAY(0, 7 * 4, 0, 0);  // start_idx=0, len=28, exec_while_loading=0, load_mode=0 (playback)
        iterations -= 4;
    }

    // Replay the remaining 2 iterations (30 - 28 = 2), minus 1 instruction.
    // The last instruction of the final iteration only affects 'a', which we don't need.
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);  // 7*2-1 = 13 instructions replayed

    // Clear any remaining conditional execution flags
    TTI_SFPENCC(0, 0, 0, 0);
}

// calculate_sfpu_gcd: Top-level SFPU function called per tile face.
// Iterates over ITERATIONS (default 8) rows of 32 elements within one 16x16 face.
// Each iteration processes one row pair from the DEST register.
template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile in DEST occupies 64 rows (4 faces of 16 rows each)
        constexpr uint dst_tile_size = 64;

        // Load row 'd' of input A from DEST into LREG0.
        // Mod0=4 selects int32 format, Mod1=3 selects LREG addressing mode.
        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a
        // Load row 'd' of input B from DEST into LREG1.
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b

        // Execute the Binary GCD algorithm body (30 iterations for 31-bit integers)
        calculate_sfpu_gcd_body<31>();

        // Store the result (in LREG1 = gcd) back to DEST at the output tile position.
        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size);
        // Advance the DEST row pointer for the next iteration within this face
        dst_reg++;
    }
}

// calculate_sfpu_gcd_init: Records 4 iterations of the inner GCD loop into the replay buffer.
// This is called once before the main GCD computation.
// The replay buffer stores 7*4 = 28 instructions, which represent 4 iterations of the loop body.
// Each iteration:
//   1. abs(a) -> LREG2 (positive a)
//   2. LREG0 &= LREG2 (isolate LSB of a, since LREG0 = -a)
//   3. clz(LREG0), skip if a==0 (handles the case where a is already 0, meaning GCD found)
//   4. LREG0 += d (add the common trailing-zero offset)
//   5. a = a >> -LREG0 (right-shift to strip all trailing zeros, making a odd)
//   6. swap(a, b) with min/max (ensure b <= a)
//   7. a = b - a (difference is even, ready for next iteration)
inline void calculate_sfpu_gcd_init() {
    // TTI_REPLAY with load_mode=1: record the next 28 instructions into the replay buffer
    TTI_REPLAY(0, 7 * 4, 0, 1);
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // LREG0 holds -a from previous iteration. LREG2 = |a| = abs(LREG0).
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // Isolate LSB of a: LREG0 = (-a) & (|a|). Since -a and |a| share only the LSB pattern,
        // this gives the lowest set bit of |a|.
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        // Count leading zeros of the isolated LSB. The CC_NE0 flag disables lanes where a == 0
        // (GCD has converged for those lanes; b already holds the result).
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0);
        // Add the common trailing-zero count d to get the total right-shift amount.
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE);
        // Right-shift |a| by the computed amount, stripping all trailing zeros.
        // LREG0 = LREG2 >> (-LREG0). After this, a is guaranteed to be odd.
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG);
        // Min/max swap: LREG1 = min(a, b), LREG0 = max(a, b).
        // This ensures b <= a for the subtraction step.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);
        // a = b - a (two's complement subtraction via negation of LREG_DST then add).
        // Since both a and b are odd and a >= b, the result (b - a) is even and non-positive.
        // The negation makes LREG0 = -(LREG0) + LREG1 = b - a.
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    }
    // After this function returns, the replay buffer contains the 28-instruction sequence.
    // Subsequent TTI_REPLAY(0, N, 0, 0) calls will play back portions of this sequence.
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPMOV` | Move/copy a value from one LREG to another across all 32 SIMD lanes |
| `TTI_SFPOR` | Bitwise OR: `VD = VB \| VC` |
| `TTI_SFPAND` | Bitwise AND: `VD = VB & VC` |
| `TTI_SFPIADD` | Two's complement integer add/subtract. With `ARG_2SCOMP_LREG_DST` modifier, negates the destination before adding: `VD = VC - VD` or `VD = 0 - VD` |
| `TTI_SFPLZ` | Count leading zeros. With `CC_NE0` modifier, also sets per-lane condition codes for lanes where input != 0 |
| `TTI_SFPSHFT2` | Bitwise shift by register amount. With `SHFT_LREG` modifier: `VD = VB << VC` (left shift; negative VC means right shift) |
| `TTI_SFPSETCC` | Set per-lane condition codes based on a comparison (here: `VC == 0`) |
| `TTI_SFPSWAP` | Swap two registers. With `VEC_MIN_MAX` modifier: `VD = min(VC, VD)`, `VC = max(VC, VD)` |
| `TTI_SFPENCC` | Enable/disable conditional execution. With all-zero args: clears conditional execution (all lanes active) |
| `TTI_SFPABS` | Integer absolute value: `VD = \|VC\|` |
| `TTI_REPLAY` | Instruction replay buffer control. With `load_mode=1`: record next N instructions. With `load_mode=0`: play back N recorded instructions |
| `TT_SFPLOAD` | Load data from DEST register file into an LREG. Mod0=4 selects int32 format |
| `TT_SFPSTORE` | Store data from an LREG back to DEST register file. Mod0=4 selects int32 format |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Holds `a` (first operand). During the loop, holds `-a` (negated for efficient subtraction) |
| `LREG1` | Holds `b` (second operand). At convergence, holds the GCD result |
| `LREG2` | Temporary register `c`. Used for intermediate computations (OR, abs, shift results) |
| `LREG3` | Holds `d`, the negated count of common trailing zeros (used as shift amount for stripping factors of 2) |
| `DEST[idst0]` | Source tile for input A (loaded via `TT_SFPLOAD` into LREG0) |
| `DEST[idst1]` | Source tile for input B (loaded via `TT_SFPLOAD` into LREG1) |
| `DEST[odst]` | Destination tile for GCD result (stored from LREG1 via `TT_SFPSTORE`) |
| Replay buffer | Stores 28 instructions (4 loop iterations of 7 instructions each) |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` / `tile_regs_wait()` to get exclusive access to DEST registers. Input tiles are copied from CBs to DEST using `copy_tile`.

2. **Initialization** (`gcd_tile_init()` -> `calculate_sfpu_gcd_init()`):
   - Issues `TTI_REPLAY(0, 28, 0, 1)` to begin recording into the replay buffer.
   - Executes 4 unrolled iterations of the inner loop body (7 instructions each). These are NOT executed on data -- they are captured into the replay buffer for later playback.
   - The recorded sequence: abs -> isolate LSB -> clz -> add shift offset -> right-shift -> min/max swap -> subtract.

3. **Per-face execution** (`gcd_tile()` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_gcd()`):
   - The LLK params wrapper iterates over 4 faces (VectorMode::RC). For each face:
     - `calculate_sfpu_gcd` loops over 8 row-pairs within the face.
     - For each row: loads `a` and `b` from DEST into LREG0 and LREG1.
     - Executes `calculate_sfpu_gcd_body<31>()`:
       - **Preamble** (12 instructions): Computes common trailing zeros, ensures b is odd, takes absolute values, negates a and d.
       - **Main loop** (30 iterations via replay): Plays back the 7-instruction sequence from the replay buffer. Uses `TTI_REPLAY(0, 28, 0, 0)` for groups of 4 iterations, then `TTI_REPLAY(0, 13, 0, 0)` for the final 2 iterations (minus the last instruction which only affects `a`).
       - **Cleanup**: `TTI_SFPENCC` clears conditional execution.
     - Stores result from LREG1 to DEST.
     - Advances `dst_reg++` to the next row.

4. **Pack**: After all tiles in the block are processed, `pack_tile(i*2, cb_out0)` moves results from DEST to the output CB.

5. **Release**: `tile_regs_commit()` / `tile_regs_release()` free the DEST registers.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Set to `true` because the output data type is Int32 (which requires 32-bit DEST accumulation).
- **`UnpackToDestMode::UnpackToDestFp32`**: Enabled for all input CBs (c_0, c_1, c_3, c_4). Since GCD is not `POWER` type, all CBs get FP32 unpack-to-dest mode, ensuring full 32-bit integer values are preserved in DEST.
- **`APPROX` template parameter**: Passed through from the LLK layer but not meaningfully used by the GCD algorithm (the algorithm is exact for integers).
- **Defines passed to compute kernel**:
  - `BINOP_INIT` = `"gcd_tile_init();"`
  - `BINARY_SFPU_OP` = `"gcd_tile(i*2, i*2+1, i*2);"`

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `ckernel_sfpu_gcd.h` are **identical**. The GCD algorithm uses only fundamental SFPU instructions (integer arithmetic, bitwise ops, shifts, abs, clz, swap) that are available and behave identically on both architectures. The `TTI_REPLAY` instruction is supported on both architectures with the same semantics.

## Implementation Notes

1. **Binary GCD algorithm choice**: The implementation avoids division entirely, using only subtraction, absolute value, shifts, and bitwise operations. This is well-suited to the SFPU, which has efficient integer shift and bitwise instructions but no integer division.

2. **Replay buffer optimization**: The inner loop body (7 instructions) is recorded once into the replay buffer and replayed 30 times. This significantly reduces instruction fetch overhead. The replay buffer can store up to 28 instructions (4 iterations), so the 30 iterations are executed as 7 groups of 4 (via replay) plus a final partial replay of 2 iterations.

3. **Convergence handling**: The `SFPLZ_MOD1_CC_NE0` flag on the leading-zero count instruction disables lanes where `a == 0` (GCD has converged). Once a lane converges, `b` holds the GCD for that lane, and subsequent operations on that lane are effectively no-ops due to conditional execution.

4. **Worst case iterations**: For 31-bit signed integers, the worst case requires at most 31 iterations of the inner loop. The implementation runs 30 full iterations plus handles the edge case by noting the 31st iteration only affects `a` (not the output `b`), so it can be skipped.

5. **DEST register interleaving**: Input A tiles are placed in even DEST slots (`i*2`) and input B in odd slots (`i*2+1`). The GCD result overwrites the even slot, which is then packed to the output CB.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the element_wise_multi_core_sfpu program factory work for binary eltwise operations?"
   **Reason**: Needed to understand the overall program factory structure, kernel selection, and CB configuration patterns.
   **Key Findings**: Confirmed the three-kernel architecture (reader/compute/writer), CB configuration patterns for sharded vs interleaved, and the SPMD work distribution model.

2. **Query**: "How is the GCD SFPU operation implemented in tt-metal?"
   **Reason**: Needed to locate the specific kernel files and understand the call chain from TTNN to SFPU.
   **Key Findings**: Identified the call chain: `ttnn.gcd` -> `BinaryOpType::GCD` -> `gcd_tile()` (in `api/compute/gcd.h`) -> `llk_math_eltwise_binary_sfpu_gcd` -> `calculate_sfpu_gcd` (in `ckernel_sfpu_gcd.h`).

3. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do?" (tt-llk repo)
   **Reason**: Needed to understand how the LLK wrapper iterates over tile faces when calling the SFPU function.
   **Key Findings**: The function iterates over 4 tile faces in VectorMode::RC mode, advancing DEST pointers via `TTI_SETRWC` between faces. Each face call processes 8 row-pairs.

4. **Query**: "Explain SFPU instructions used in GCD" (tt-isa-documentation repo)
   **Reason**: Needed detailed semantics of each SFPU instruction and modifier flag.
   **Key Findings**: Detailed descriptions of SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS, SFPLOAD, SFPSTORE with modifier flag semantics.

5. **Query**: "What is TTI_REPLAY?" (tt-llk repo)
   **Reason**: The GCD kernel uses TTI_REPLAY extensively but it was not found in the local codebase (defined in the tt_llk submodule).
   **Key Findings**: TTI_REPLAY is an instruction replay buffer mechanism. Parameters: `(start_idx, len, exec_while_loading, load_mode)`. `load_mode=1` records instructions, `load_mode=0` plays them back.

### Confluence References

- **Page**: Tensix SFPU Instruction Set Architecture (page ID 1170505767)
- **Section consulted**: LOADMACRO/Replay section. Confirmed replay buffer exists but the Confluence page focused on SFPLOADMACRO replay sequences rather than the TTI_REPLAY instruction. The DeepWiki query on tt-llk was more informative for TTI_REPLAY specifics.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 358-361)
   **Reason**: Needed to confirm what preprocessor defines are generated for GCD.
   **Key Information**: GCD generates `BINOP_INIT` = `"gcd_tile_init();"` and `BINARY_SFPU_OP` = `"gcd_tile(i*2, i*2+1, i*2);"` via the `get_defines_fp32` function.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand how runtime args are set for each core.
   **Key Information**: Contains the `set_eltwise_binary_runtime_args` template function that handles work splitting, shard index computation, and per-core runtime arg assignment for reader/compute/writer kernels.
