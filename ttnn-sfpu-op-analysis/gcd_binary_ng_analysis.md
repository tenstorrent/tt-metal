# GCD (Binary NG) -- SFPU Operation Analysis

## Overview

**Operation**: GCD (Greatest Common Divisor)
**Category**: Eltwise Binary (binary_ng framework)
**SFPU**: Yes -- this is an SFPU-only operation; it cannot run on the FPU
**Data Type**: INT32 (both inputs must be int32)
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`
**Namespace**: `ttnn::operations::binary_ng`

The GCD operation computes the greatest common divisor of two integer tensors element-wise. It uses the binary GCD (Stein's algorithm) implemented entirely in SFPU instructions, running on the vector unit of each Tensix core. The algorithm operates on 31-bit signed integers (taking absolute values first), requiring up to 30 iterations of the core loop per element.

---

## Operation Configuration

### BinaryOpType Mapping

In `binary_ng_utils.cpp`, the `OpConfig` constructor maps `BinaryOpType::GCD` to `SfpuBinaryOp::GCD`:

```cpp
case BinaryOpType::GCD:
    if (is_sfpu_op()) {
        binary_op = SfpuBinaryOp::GCD;
    } else {
        TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
    }
    break;
```

GCD has no pre-processing (`process_lhs`, `process_rhs`) and no post-processing (`postprocess`). It is a pure SFPU binary operation.

### Compile-Time Defines

The `get_sfpu_init_fn` function returns:
- **BINARY_SFPU_INIT**: `gcd_tile_init();`
- **BINARY_SFPU_OP**: `gcd_tile`

These are injected as preprocessor defines into the compute kernel, so the generic SFPU binary compute kernel calls `gcd_tile(idst0, idst1, odst)` for each tile.

### FP32 Dest Accumulation

Since GCD operates on INT32 data, `fp32_dest_acc_en` is set to `true` in the program factory (the condition `a_data_format == DataType::Int32 && b_data_format == DataType::Int32` is satisfied). This ensures the DEST register file uses 32-bit precision.

### Unpack-to-Dest Mode

For SFPU binary ops other than POWER, all source CBs use `UnpackToDestMode::UnpackToDestFp32`. This means data is unpacked directly into DEST registers in 32-bit format, bypassing the source registers. This is critical for GCD because the algorithm operates on raw 32-bit integer bit patterns.

---

## Program Structure

### Three-Kernel Architecture

The binary_ng program factory creates three kernels per Tensix core:

| Kernel | Role | File (no-bcast, SFPU) |
|--------|------|-----------------------|
| Reader | Reads tiles A and B from DRAM/L1 into CBs c_0 and c_1 | `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| Compute | Unpacks tiles to DEST, runs SFPU GCD, packs result | `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| Writer | Writes output tiles from CB c_2 to DRAM/L1 | `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |

For broadcast variants (scalar, row, col), different kernel files are selected via `BinaryNgKernelConfig` and `get_kernel_file_path`.

### Circular Buffers

| CB Index | Name | Purpose | Double-Buffered |
|----------|------|---------|-----------------|
| c_0 | cb_src_a | Input tensor A tiles | Yes (2 tiles, or shard volume if sharded) |
| c_1 | cb_src_b | Input tensor B tiles | Yes (2 tiles, or shard volume if sharded) |
| c_2 | cb_out | Output tensor C tiles | Yes (2 tiles, or shard volume if sharded) |
| c_3 | cb_intermediate_a | LHS activation intermediate (only if LHS activations present) | No (1 tile) |
| c_4 | cb_intermediate_b | RHS activation intermediate (only if RHS activations present) | No (1 tile) |

For GCD without pre/post activations, only c_0, c_1, and c_2 are used.

### Work Distribution

The program factory distributes output tiles across available cores using `split_work_to_cores`. Each core processes a contiguous range of output tiles. Runtime arguments tell each core:
- `num_tiles`: how many tiles this core processes
- `start_tile_id`: the global tile offset for this core's first tile
- Shape dimensions (D, N, C, Ht, Wt) for multi-dimensional indexing

### Broadcast Support

GCD supports all `SubtileBroadcastType` variants through the binary_ng framework:
- **NONE**: Element-wise, both tensors same shape
- **SCALAR_A/B**: One input is a scalar tile
- **ROW_A/B**: One input has a single tile row
- **COL_A/B**: One input has a single tile column
- **ROW_A_COL_B / ROW_B_COL_A**: Mixed row/column broadcast

The reader kernel handles broadcasting by adjusting stride calculations for the broadcast dimension.

---

## Kernel Implementations

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File (no-bcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU split includes provide the SFPU math functions based on which RISC-V core is running
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Binary SFPU operation headers -- each provides tile-level init/op functions
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
#include "api/compute/gcd.h"           // Provides gcd_tile_init() and gcd_tile()
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

// Utility macros for preprocessing activations and common patterns
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime arg 0: number of output tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: tiles produced per read-compute-write cycle (always 1 for binary_ng)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB indices for the two inputs and the output
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;  // Input A
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;  // Input B
    constexpr auto cb_out = tt::CBIndex::c_2;       // Output C

    // If LHS/RHS activations are defined, use intermediate CBs; otherwise alias to input CBs.
    // For GCD, no activations are used, so cb_post_lhs == cb_pre_lhs, cb_post_rhs == cb_pre_rhs.
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize unpack/pack hardware for the given CB data formats
    unary_op_init_common(cb_post_lhs, cb_out);

#ifdef PACK_RELU
    // Not used for GCD, but the framework supports optional ReLU on pack
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // When no pre/post activations exist, initialize the SFPU op once before the loop.
    // For GCD, this expands to: gcd_tile_init();
    // This sets up the REPLAY buffer with the 7-instruction inner loop body.
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // Main tile processing loop: one tile per iteration
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS expands to nothing when no activations are configured
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for one LHS tile to be available in the CB (written by reader kernel)
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        // Wait for one RHS tile to be available
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in the output CB for one tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Conditional re-init when activations change the SFPU state (not used for GCD)
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif

        // Acquire exclusive access to DEST tile registers (blocking if pack is still using them)
        tile_regs_acquire();

        // Configure unpack hardware to match RHS->LHS format transition, then copy LHS tile to DEST[0]
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Unpack tile from cb_post_lhs position i into DEST register at slot i*2 (even slots for LHS)
            copy_tile(cb_post_lhs, i, i * 2);
        }

        // Reconfigure unpack for LHS->RHS format, then copy RHS tile to DEST[1]
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Unpack tile from cb_post_rhs position i into DEST register at slot i*2+1 (odd slots for RHS)
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // Per-tile post-activation re-init (not used for GCD)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU operation: gcd_tile(i*2, i*2+1, i*2)
            // This reads DEST[0] (LHS) and DEST[1] (RHS), computes GCD, writes result to DEST[0]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Post-activation processing (not used for GCD)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }

        // Signal that DEST registers are ready for packing
        tile_regs_commit();

        // Wait for DEST registers to be committed (ensures math is complete)
        tile_regs_wait();

        // Pack the result tile from DEST[0] into the output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }

        // Release DEST registers so the next iteration's math can proceed
        tile_regs_release();

        // Push the completed output tile and free the consumed input tiles
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### Compute Kernel File (bcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp`

The bcast variant adds a `process_tile` function that handles the broadcast dimension by iterating over the non-broadcast tiles while holding the broadcast tile constant. The inner computation logic (copy_tile, BINARY_SFPU_OP, pack_tile) is identical.

#### Compute Kernel File (scalar variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`

The scalar variant reads the RHS scalar tile once before the loop, then processes all LHS tiles against it. The core SFPU dispatch is the same.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
(Identical implementation in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// The core GCD loop body, parameterized by max_input_bits (default 31 for int32).
// This function implements the "binary GCD" or "Stein's algorithm" using SFPU instructions.
// It operates on LREG0 (a) and LREG1 (b), producing the result in LREG1.
// LREG2 and LREG3 are used as temporaries.
template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    // --- Phase 1: Compute the shared factor of 2 ---
    // Determine the number of trailing zeros in (a | b), which gives the
    // power-of-2 factor shared by both a and b.

    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);    // c = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);      // c = a | b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);     // d = c = (a | b)
    // Negate d: d = -d (two's complement negation via SFPIADD with zero and 2SCOMP flag)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3,
                SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
    // Isolate the lowest set bit of (a|b): d = d & c = (-c) & c
    // This gives a single bit: the position of the least significant 1-bit.
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    // Count leading zeros of the isolated bit to get the shift amount.
    // d = clz(d). Since d has exactly one bit set, clz gives (31 - bit_position).
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);

    // --- Phase 2: Ensure b is odd ---
    // If b is even, swap a and b so that b becomes the odd operand.
    // Test: shift b left by d (the clz result). If the result is zero, b was even.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2,
                 SFPSHFT2_MOD1_SHFT_LREG);                // c = b << d
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);                 // Set CC if c == 0 (b was even)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);     // Conditionally swap a and b
    TTI_SFPENCC(0, 0, 0, 0);                               // Disable conditional execution

    // Take absolute values to handle negative inputs
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);      // a = |a|
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);      // b = |b|

    // Negate a (for use in the subtraction step of the loop)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0,
                SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);  // a = -|a|
    // Negate d (the clz value becomes a right-shift amount when negated)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3,
                SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);  // d = -d

    // --- Phase 3: Main loop via REPLAY ---
    // The inner loop body (7 instructions, recorded in calculate_sfpu_gcd_init) is replayed
    // to perform the iterative GCD computation. Each replay executes 4 loop iterations
    // (7 instructions * 4 = 28 instructions per REPLAY call).

    int iterations = max_input_bits - 1;  // = 30 for 31-bit inputs

    // Unroll by 4 iterations at a time using REPLAY
    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);  // Replay 28 instructions (4 iterations of 7 instructions)
        iterations -= 4;
    }

    // Handle remaining iterations (2 for the default case: 30 mod 4 = 2).
    // Skip the last instruction of the final iteration since it only updates a, not b.
    // The result (GCD) is in LREG1 (b).
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);

    // Re-enable all lanes (disable conditional execution)
    TTI_SFPENCC(0, 0, 0, 0);
}

// Top-level GCD function called by the LLK layer for each tile.
// ITERATIONS=8 means 8 "faces" of 4 rows each = 32 rows in one tile pass.
// (The SFPU processes 32 elements per row in SIMD fashion, so 8 iterations * 4 rows = 32 rows.)
template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile in DEST occupies 64 rows (32x32 tile = 1024 elements, laid out as 64 rows of 16)
        constexpr uint dst_tile_size = 64;

        // Load 32-element vectors from DEST into SFPU local registers
        // Mode 4 = int32 load, Mode 3 = address mode with dst_reg offset
        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a = DEST[in0]
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b = DEST[in1]

        // Run the full binary GCD algorithm for 31-bit inputs (30 iterations)
        calculate_sfpu_gcd_body<31>();

        // Store the result (in LREG1 = b = gcd) back to DEST
        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size);

        // Advance to next face (group of rows within the tile)
        dst_reg++;
    }
}

// Initialization function: records the 7-instruction inner loop body into the REPLAY buffer.
// This is called once per tile batch (or once before the tile loop) via gcd_tile_init().
inline void calculate_sfpu_gcd_init() {
    // TTI_REPLAY with mode=1 starts RECORDING the next 28 instructions into the replay buffer.
    TTI_REPLAY(0, 7 * 4, 0, 1);

    // Record 4 iterations of the 7-instruction loop body:
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // Step 1: Get absolute value of a (which is stored as -a in LREG0)
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // LREG2 = |a| (positive value)

        // Step 2: Isolate the LSB of a by ANDing the positive and negative representations
        // Since LREG0 = -|a| and LREG2 = |a|, (-|a|) & |a| isolates the lowest set bit
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        // LREG0 = (-a) & a = lowest set bit of a

        // Step 3: Count leading zeros of the isolated bit, with lane masking
        // Lanes where a == 0 are disabled (those lanes have already converged)
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0);
        // LREG0 = clz(lowest_bit_of_a), lanes where a==0 are masked off

        // Step 4: Add d (the shared trailing-zero count) to get the total shift amount
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE);
        // LREG0 = clz(lowest_bit) + d = right-shift amount (negative = right shift)

        // Step 5: Arithmetic right shift a by the computed amount, making a odd
        // LREG2 still holds |a|; shift it right to remove all factors of 2
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG);
        // LREG0 = |a| >> shift_amount (now a is definitely odd)

        // Step 6: Swap so that b <= a (ensure the smaller value is in LREG1)
        // VEC_MIN_MAX mode: LREG0 gets max(a, b), LREG1 gets min(a, b)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);
        // Now b <= a (both odd and positive)

        // Step 7: Compute a = b - a (the difference is even, will be shifted next iteration)
        // Uses 2SCOMP flag: LREG0 = LREG1 + (-LREG0) = b - a
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0,
                     SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
        // LREG0 = b - a (even, and a > b so this is negative; will be abs'd next iteration)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### LLK Wrapper File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h`

```cpp
#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_gcd.h"

namespace ckernel {

// Initialization: calls the generic binary SFPU init with GCD-specific setup function
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gcd_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROXIMATE>(
        sfpu::calculate_sfpu_gcd_init);
}

// Per-tile execution: dispatches to the generic binary SFPU params handler
// which manages dst_reg iteration and calls calculate_sfpu_gcd
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gcd(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_sfpu_gcd, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
```

#### API Header
`tt_metal/hw/inc/api/compute/gcd.h`

```cpp
#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_gcd.h"
#endif

namespace ckernel {

// Public API: computes gcd(DEST[idst0], DEST[idst1]) -> DEST[odst]
ALWI void gcd_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst)));
}

// Public API: one-time initialization (records replay buffer)
ALWI void gcd_tile_init() { MATH((llk_math_eltwise_binary_sfpu_gcd_init<APPROX>())); }

}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | TTI Macro | Description |
|------------|-----------|-------------|
| SFPMOV | `TTI_SFPMOV` | Register-to-register move (VD = VC) |
| SFPOR | `TTI_SFPOR` | Bitwise OR (VD = VB \| VC) |
| SFPAND | `TTI_SFPAND` | Bitwise AND (VD = VB & VC) |
| SFPIADD | `TTI_SFPIADD` | Integer add/subtract with optional 2's complement (VD = VC +/- VD or imm) |
| SFPLZ | `TTI_SFPLZ` | Count leading zeros (VD = clz(VC)), with optional lane masking |
| SFPSHFT2 | `TTI_SFPSHFT2` | Barrel shifter (VD = VB << VC or VB >> -VC) using register shift amount |
| SFPSETCC | `TTI_SFPSETCC` | Set condition code flags based on register value (e.g., == 0) |
| SFPSWAP | `TTI_SFPSWAP` | Swap two registers, or VEC_MIN_MAX mode for element-wise min/max |
| SFPENCC | `TTI_SFPENCC` | Enable/disable conditional execution (lane masking via flags) |
| SFPABS | `TTI_SFPABS` | Absolute value (VD = \|VC\|) |
| SFPLOAD | `TT_SFPLOAD` | Load 32-element vector from DEST register file into LREG |
| SFPSTORE | `TT_SFPSTORE` | Store 32-element vector from LREG back to DEST register file |
| REPLAY | `TTI_REPLAY` | Record or replay a sequence of instructions from the replay buffer |

#### SFPU Register Usage

| Register | Role in GCD |
|----------|-------------|
| LREG0 | Operand `a` (stored as -\|a\| during the main loop for efficient subtraction) |
| LREG1 | Operand `b` (always positive and odd after initial setup; holds final GCD result) |
| LREG2 | Temporary: holds \|a\|, intermediate results for bit isolation |
| LREG3 | Holds `d`: the negated count of shared trailing zeros (used as shift amount) |
| LCONST_0 | Constant zero, used for negation via SFPIADD |
| dst_reg | Auto-incrementing pointer into the DEST register file, advances per face |

#### SFPU Execution Flow

1. **Initialization** (`calculate_sfpu_gcd_init`):
   - Issues `TTI_REPLAY(mode=1)` to begin recording 28 instructions into the replay buffer.
   - Records 4 unrolled iterations of the 7-instruction inner loop body.
   - The replay buffer persists across all tile processing until re-initialized.

2. **Per-tile execution** (`calculate_sfpu_gcd`, called 8 times per tile for 8 faces):
   - **Load**: `TT_SFPLOAD` loads a 32-element int32 vector from DEST[in0] into LREG0 (a) and from DEST[in1] into LREG1 (b).
   - **Setup** (`calculate_sfpu_gcd_body`):
     - Computes `d = clz((-c) & c)` where `c = a | b`. This finds the number of shared trailing zeros.
     - Ensures `b` is odd by conditionally swapping `a` and `b`.
     - Takes absolute values of both operands.
     - Negates `a` and `d` for the main loop.
   - **Main loop** (30 iterations via REPLAY):
     - Each iteration: isolate LSB of `a`, compute shift amount, right-shift `a` to make it odd, swap so `b <= a`, compute `a = b - a`.
     - The loop converges when `a == 0`, at which point `b` holds the GCD (without the shared power-of-2 factor).
     - Lanes where `a == 0` are masked via `SFPLZ_MOD1_CC_NE0`, avoiding unnecessary computation.
   - **Store**: `TT_SFPSTORE` writes LREG1 (the GCD result) back to DEST[out].
   - `dst_reg++` advances to the next face.

3. **Why 30 iterations**: For 31-bit signed integers, the binary GCD algorithm requires at most 31 iterations. The implementation uses 30 because: the final (31st) iteration would only affect `a`, not `b` (which holds the result), and the last instruction of the 30th iteration is also skipped for the same reason.

#### SFPU Configuration

- **Math fidelity**: Not applicable (integer operation, no floating-point approximation).
- **APPROX template parameter**: Passed through the LLK layer but has no effect on the GCD algorithm since it operates purely on integers.
- **Unpack-to-Dest mode**: `UnpackToDestFp32` -- data is unpacked directly into DEST in 32-bit format, which is essential for preserving int32 bit patterns.
- **FP32 dest accumulation**: Enabled (`fp32_dest_acc_en = true`) since both inputs are INT32.
- **Replay buffer**: The 7-instruction loop body is recorded once during init. Each `TTI_REPLAY` call replays 4 iterations (28 instructions) efficiently from the hardware replay buffer, avoiding RISC-V instruction fetch overhead.

#### Hardware Compatibility Notes

The Blackhole and Wormhole implementations of `ckernel_sfpu_gcd.h` are **identical**. Both architectures support all the SFPU instructions used by the GCD kernel. Key considerations:

- **SFPSHFT2**: Blackhole has bug fixes for this instruction compared to Wormhole, but the usage pattern in GCD (register-based shift) works correctly on both.
- **SFPSWAP with VEC_MIN_MAX**: Available on both architectures. Performs element-wise signed comparison and routes min/max to the appropriate registers.
- **REPLAY**: Both architectures support the replay expander with the same recording/playback semantics.
- **Lane masking (SFPLZ with CC_NE0)**: Works identically on both architectures, disabling computation on converged lanes.

---

### Reader Kernel

#### Reader Kernel File (no-bcast, ng variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

The reader kernel reads tiles for both input tensors A and B from DRAM (or directly from L1 if sharded) into circular buffers c_0 and c_1. It uses TensorAccessor for DRAM reads and supports multi-dimensional tensor indexing with stride calculations for broadcasting dimensions. For sharded tensors, it simply does `cb_reserve_back` / `cb_push_back` to make the pre-existing L1 data visible to the compute kernel.

### Writer Kernel

#### Writer Kernel File (no-bcast, ng variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`

The writer kernel reads completed output tiles from circular buffer c_2 and writes them to DRAM (or does nothing for sharded output since the data is already in L1). It uses the same multi-dimensional tile offset logic as the reader.

---

## External Knowledge Sources

### DeepWiki References
- **tenstorrent/tt-metal**: Binary_ng program factory structure, kernel file paths, OpConfig mapping for GCD, SFPU binary operation dispatch pattern
- **tenstorrent/tt-isa-documentation**: SFPU instruction semantics (SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS, SFPLOAD, SFPSTORE, REPLAY)
- **tenstorrent/sfpi**: SFPU local registers (LREG0-LREG7), dst_reg mechanism, SFPSWAP VEC_MIN_MAX mode, DEST register file layout

### Confluence References
Not consulted for this analysis. The DeepWiki sources provided sufficient detail on all SFPU instructions used by the GCD kernel.

### Glean References
Not consulted for this analysis. No confidential hardware specifications were needed beyond what is documented in the open-source repositories.

---

## Algorithm Summary

The GCD kernel implements the **binary GCD (Stein's algorithm)**, which avoids division and relies only on:
1. Subtraction
2. Comparison (via min/max swap)
3. Counting/removing factors of 2 (via clz and shift)

This is well-suited to the SFPU because these operations map directly to available SFPU instructions (SFPIADD for subtraction, SFPSWAP for comparison, SFPLZ for leading-zero count, SFPSHFT2 for shifting). A naive Euclidean GCD using division would be far more expensive on the SFPU since integer division is not a single instruction.

The algorithm processes all 32 SIMD lanes independently. Lanes that converge early (where `a` becomes 0) are masked out via condition codes, but the instruction stream still executes for all lanes. The worst case is 30 iterations for 31-bit inputs.

---

## Key Design Decisions

1. **Why binary GCD instead of Euclidean**: The SFPU lacks an efficient integer modulo or division instruction. Binary GCD uses only bit manipulation and subtraction, which are native SFPU operations.

2. **Why REPLAY**: The inner loop is 7 instructions repeated 30 times. Recording these 7 instructions into the replay buffer and replaying them avoids 30 * 7 = 210 instruction fetches from the RISC-V, replacing them with a handful of REPLAY commands. This significantly reduces instruction bandwidth pressure.

3. **Why 8 ITERATIONS in calculate_sfpu_gcd**: A 32x32 tile is laid out as 64 rows in DEST (with 16 elements per row for the addressing scheme). The SFPU processes 32 elements per SIMD operation, so 8 face iterations (each advancing dst_reg) cover the full tile.

4. **Why negate a before the loop**: Storing `a` as `-|a|` allows the subtraction `b - a` to be computed as an addition (`b + (-a)`), which is a single SFPIADD instruction. The absolute value at the start of each iteration recovers the positive value for bit manipulation.

5. **Why identical Blackhole/Wormhole implementations**: The GCD kernel uses only basic SFPU instructions that are supported identically on both architectures. There are no architecture-specific optimizations needed.
