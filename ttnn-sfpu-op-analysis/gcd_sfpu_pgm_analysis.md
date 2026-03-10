# GCD (Greatest Common Divisor) -- SFPU Binary Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | GCD (Greatest Common Divisor) |
| **Operation Type** | Binary element-wise SFPU |
| **BinaryOpType Enum** | `BinaryOpType::GCD` |
| **Program Factory** | `BinaryDeviceOperation::ElementWiseMultiCoreSfpu` |
| **Program Factory File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` |
| **Supported Input Dtypes** | `INT32 x INT32`, `UINT32 x UINT32` |
| **Output Dtype** | Same as input (INT32 or UINT32) |
| **Compute Engine** | SFPU (vector unit) -- not FPU (matrix unit) |
| **Algorithm** | Binary GCD (Stein's algorithm) implemented entirely in SFPU instructions |

## Program Factory Selection

The GCD operation is routed to the `ElementWiseMultiCoreSfpu` program factory through the following decision chain:

1. `BinaryDeviceOperation::select_program_factory()` in `binary_device_operation.cpp` checks if input tensors have matching height and width dimensions.
2. If dimensions match, it calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`.
3. For `BinaryOpType::GCD`, the function returns `true` when both inputs are `INT32` or both are `UINT32`.
4. The factory returns `ElementWiseMultiCoreSfpu{}`.

Relevant code from `binary_device_operation.cpp`:
```cpp
case BinaryOpType::GCD:
case BinaryOpType::LCM:
case BinaryOpType::LEFT_SHIFT:
case BinaryOpType::RIGHT_SHIFT:
case BinaryOpType::LOGICAL_RIGHT_SHIFT:
    return ((a == DataType::INT32 && b == DataType::INT32) || (a == DataType::UINT32 && b == DataType::UINT32));
```

## Circular Buffer Configuration

| CB Index | Name | Purpose | Size (non-sharded) | Data Format |
|---|---|---|---|---|
| `c_0` | `cb_src0` | Input tensor A | `2 * max_block_size * tile_size` | Matches input A dtype |
| `c_1` | `cb_src1` | Input tensor B | `2 * max_block_size * tile_size` | Matches input B dtype |
| `c_2` | `cb_out0` | Output tensor | `2 * max_block_size * tile_size` | Matches output dtype |
| `c_3` | (unused for GCD) | Pre-scaling input 0 | Not allocated | N/A |
| `c_4` | (unused for GCD) | Pre-scaling input 1 | Not allocated | N/A |

For GCD, circular buffers `c_3` and `c_4` are NOT allocated because GCD does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`. The GCD operation has no pre-scaling steps -- it operates directly on the raw integer inputs.

When sharded, the input CBs are backed by globally-allocated addresses from the tensor buffers, and their sizes are `num_tiles_per_shard * tile_size`.

## Compile-Time Defines

The `get_defines_fp32()` function in `binary_op_utils.cpp` generates the following defines for GCD:

```cpp
case BinaryOpType::GCD:
    new_defines.insert({"BINOP_INIT", fmt::format("gcd_tile_init();")});
    op_name = "gcd_tile";
    break;
```

This produces the following define map entries:

| Define Key | Value | Purpose |
|---|---|---|
| `BINOP_INIT` | `gcd_tile_init();` | Initializes the SFPU for GCD computation (records replay buffer) |
| `BINARY_SFPU_OP` | `gcd_tile(i*2, i*2+1, i*2);` | Executes GCD on tile pair, result overwrites first input slot |

The `BINARY_SFPU_OP` define is constructed at the end of `get_defines_fp32()`:
```cpp
new_defines.insert({"BINARY_SFPU_OP", fmt::format("{}({}, {}, {});", op_name, idst1, idst2, idst1)});
```
where `idst1 = "i*2"` (input A / output position), `idst2 = "i*2+1"` (input B position).

## Compute Configuration

```cpp
bool fp32_dest_acc_en = true;  // Always true for INT32/UINT32
UnpackToDestMode = UnpackToDestFp32;  // For all CBs (since op_type != POWER)
```

GCD always operates with `fp32_dest_acc_en = true` because its input types are INT32 or UINT32 (which map to `DataFormat::Int32` or `DataFormat::UInt32`). The `UnpackToDestMode::UnpackToDestFp32` is set for all circular buffers, which means data is unpacked directly into the 32-bit destination registers without format conversion.

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

The reader kernel reads tiles from two input tensors (A and B) from DRAM into circular buffers `c_0` and `c_1`. It supports both interleaved and sharded memory layouts. For sharded inputs, it simply reserves and pushes back the pre-allocated pages. For interleaved inputs, it reads one tile at a time using `noc_async_read_tile()` and blocks on `noc_async_read_barrier()`.

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

The writer kernel writes output tiles from circular buffer `c_2` to DRAM. For sharded output, it simply waits on the front of the CB. For interleaved output, it writes one tile at a time using `noc_async_write_page()`.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

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
#include "api/compute/gcd.h"          // Provides gcd_tile() and gcd_tile_init() for this operation
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if any pre-scaling defines are set. For GCD, neither is set, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime args: how many blocks of tiles to process, and how many tiles per block
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // Input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // Input B circular buffer

    // For GCD, SFPU_OP_INIT_PRE_IN0_0 is NOT defined, so cb_inp0 == cb_in0 (no pre-scaling intermediate buffer)
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;            // GCD takes this path: reads directly from c_0
#endif

    // For GCD, SFPU_OP_INIT_PRE_IN1_0 is NOT defined, so cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;            // GCD takes this path: reads directly from c_1
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // Output circular buffer

    // Initialize the unary op common infrastructure (configures tile copy, unpack, pack)
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Main processing loop: iterate over all tile blocks assigned to this core
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // PRE_SCALE is false for GCD, so the entire pre-scaling sections (lines 59-105) are skipped.
        // The #ifdef SFPU_OP_INIT_PRE_IN0_0 and #ifdef SFPU_OP_INIT_PRE_IN1_0 blocks are not compiled.

        // Wait for input tiles from both circular buffers to be available (written by reader kernel)
        cb_wait_front(cb_inp0, per_core_block_size);    // Wait for input A tiles
        cb_wait_front(cb_inp1, per_core_block_size);    // Wait for input B tiles
        cb_reserve_back(cb_out0, per_core_block_size);  // Reserve space in output CB

        tile_regs_acquire();    // Acquire exclusive access to the DST register file
        tile_regs_wait();       // Wait for DST registers to be available

        // Configure tile copy for input B -> DST, using input A's data format as reference
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy input A tile i into DST at position i*2 (even slots hold input A)
            copy_tile(cb_inp0, i, i * 2);
        }

        // Switch data format configuration for copying input B tiles
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy input B tile i into DST at position i*2+1 (odd slots hold input B)
            copy_tile(cb_inp1, i, i * 2 + 1);

            // For GCD, BINOP_INIT is defined as "gcd_tile_init();" -- this records the replay buffer
            // containing the core GCD iteration loop (7 instructions x 4 iterations).
            // This must be called before each gcd_tile() invocation to ensure the replay buffer is fresh.
#ifdef BINOP_INIT
            BINOP_INIT              // Expands to: gcd_tile_init();
#endif

            // For GCD, BINARY_SFPU_OP is defined as "gcd_tile(i*2, i*2+1, i*2);"
            // This computes gcd(DST[i*2], DST[i*2+1]) and stores result in DST[i*2].
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP          // Expands to: gcd_tile(i*2, i*2+1, i*2);
#endif

            // No SFPU_OP_INIT_0 or SFPU_OP_CHAIN_0 defines for GCD (no fused activations or typecast)

            // Pack the result from DST[i*2] into the output circular buffer
            pack_tile(i * 2, cb_out0);
        }

        tile_regs_commit();     // Signal that DST register writes are complete
        tile_regs_release();    // Release DST register file for other kernels

        // Free consumed input tiles and publish output tiles
        cb_pop_front(cb_inp0, per_core_block_size);     // Release input A tiles
        cb_pop_front(cb_inp1, per_core_block_size);     // Release input B tiles
        cb_push_back(cb_out0, per_core_block_size);     // Publish output tiles to writer kernel
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
(Identical implementation exists at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`)

#### LLK Bridge File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h`

#### API Header
`tt_metal/hw/inc/api/compute/gcd.h`

#### Call Chain

```
gcd_tile(idst0, idst1, odst)                                     [api/compute/gcd.h]
  -> MATH(llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst))
    -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(             [llk_math_eltwise_binary_sfpu_gcd.h]
         sfpu::calculate_sfpu_gcd, dst_index0, dst_index1, odst, VectorMode::RC)

gcd_tile_init()                                                    [api/compute/gcd.h]
  -> MATH(llk_math_eltwise_binary_sfpu_gcd_init<APPROX>())
    -> llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROX>(  [llk_math_eltwise_binary_sfpu_gcd.h]
         sfpu::calculate_sfpu_gcd_init)
```

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

// calculate_sfpu_gcd_body: The inner loop of the Binary GCD (Stein's) algorithm.
// This function implements the core GCD reduction step that is replayed multiple times.
//
// PRECONDITIONS when entering:
//   LREG0 = a (one operand, sign varies during computation)
//   LREG1 = b (other operand, always the "smaller" after swap)
//   LREG3 = d (negated shared trailing-zero count, used for final shift)
//
// The algorithm works on signed 32-bit integers in the SFPU SIMD lanes.
// It uses the Binary GCD property: if both a and b are odd, then |a - b| is even
// and gcd(a, b) = gcd(|a - b| / 2^k, min(a, b)).
//
// Template parameter max_input_bits controls iteration count (default 31 for int32).
template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    // --- PREAMBLE: Determine shared factor of 2 and ensure b is odd ---

    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);  // c = a (copy a to LREG2 as temporary)
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);    // c = a | b (bitwise OR to find combined trailing zeros)

    // Isolate the lowest set bit of (a | b) to find the shared power of 2.
    // The shared factor of 2 in gcd(a, b) = 2^k where k = count_trailing_zeros(a | b).
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);   // d = c = (a | b)
    // Negate d: d = -(a | b). This is used to isolate the lowest set bit via d & c.
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3,
        SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);   // d = d & c = (-c) & c, isolates lowest set bit
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);    // d = clz(lowest_set_bit) = 31 - ctz(a | b)

    // Check if b is odd. If b is even, swap a and b so that the "b" register holds the odd value.
    // We left-shift b by d (= clz of lowest set bit). If the result is 0, b had no bits above the
    // shared trailing zeros, meaning b was even relative to the shared factor.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2,
        SFPSHFT2_MOD1_SHFT_LREG);                      // c = b << d (test if b is effectively even)
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);              // Set lane flags if c == 0 (b is even)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // Conditional swap: if flag set, swap(a, b)
    TTI_SFPENCC(0, 0, 0, 0);                            // Disable conditional execution (all lanes active again)

    // Take absolute values of both operands to work with positive integers.
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);   // a = abs(a)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);   // b = abs(b)

    // Negate a (for subtraction in the main loop) and negate d (for right-shifting later).
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0,
        SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a (for subtraction a = b - a)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3,
        SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d (negate for right-shift usage)

    // --- MAIN LOOP: Replay the recorded GCD iteration ---
    // The main iteration is recorded in the replay buffer by calculate_sfpu_gcd_init().
    // Each iteration (7 instructions) performs:
    //   1. abs(a) -> isolate LSB -> clz -> add d -> right-shift to make a odd
    //   2. swap(a, b) to ensure b <= a
    //   3. a = b - a (result is even since both are odd, loop continues)

    int iterations = max_input_bits - 1;    // 30 iterations for 31-bit inputs

    // Replay in batches of 4 iterations (7 instructions each = 28 instructions per replay)
    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0);        // Replay 4 iterations (28 instructions) from buffer
        iterations -= 4;
    }

    // Replay remaining iterations. For 31-bit inputs: 30 - 28 = 2 remaining iterations.
    // The worst case for 31-bit inputs needs 31 iterations, but we skip the final iteration
    // because it only affects a (which is discarded). We also skip the last instruction of
    // iteration 30 since it only modifies a.
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0);   // Replay remaining iterations minus last instruction

    TTI_SFPENCC(0, 0, 0, 0);               // Ensure all lanes are active (disable conditional execution)
}

// calculate_sfpu_gcd: Top-level function called by the LLK layer for each tile.
// ITERATIONS=8 processes all 8 faces of a 32x32 tile (4 rows per face, 32 elements per row).
template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm applied to each face of the tile.
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile occupies 64 rows in the destination register file.
        // (32x32 tile = 1024 elements, stored as 64 rows of 16 elements in Dest,
        //  but indexed as 64 "rows" with 32-bit SIMD width per row.)
        constexpr uint dst_tile_size = 64;

        // Load input A face from DST register at the tile's base offset
        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // LREG0 = a (input A face)
        // Load input B face from DST register
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // LREG1 = b (input B face)

        // Execute the binary GCD algorithm for 31-bit signed integers
        calculate_sfpu_gcd_body<31>();

        // Store the result (in LREG1 = b, which holds gcd at convergence) back to DST
        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size);

        // Advance to the next face within the tile
        dst_reg++;
    }
}

// calculate_sfpu_gcd_init: Records the core GCD iteration into the SFPU replay buffer.
// This function is called once before processing tiles (via gcd_tile_init()).
// It records 7 instructions x 4 iterations = 28 instructions into the replay buffer.
// These instructions implement one iteration of the binary GCD reduction:
//   Given: a is even (or zero), b is odd
//   1. SFPABS: compute |a|
//   2. SFPAND: isolate lowest set bit of a (i.e., a & |a|)
//   3. SFPLZ: count leading zeros of lowest set bit, with conditional disable for zero lanes
//   4. SFPIADD: add d (trailing zero offset) to the CLZ result
//   5. SFPSHFT2: right-shift |a| by the computed amount, making a odd
//   6. SFPSWAP: ensure b <= a using min/max swap
//   7. SFPIADD: a = b - a (result is even since both operands are now odd)
inline void calculate_sfpu_gcd_init() {
    // Start recording into the replay buffer. 7*4=28 instructions will be recorded.
    // The "1" in the last argument means "start recording" mode.
    TTI_REPLAY(0, 7 * 4, 0, 1);

    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // LREG0 holds -a (negated). LREG2 will hold +a via abs.
        // The pair {-a, +a} in {LREG0, LREG2} is used to isolate the lowest set bit.
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            // LREG2 = |LREG0| = |a| (absolute value of a)
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            // LREG0 = LREG0 & LREG2 = (-a) & |a| = lowest set bit of a
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0);
            // LREG0 = clz(lowest_set_bit), also sets lane flags for non-zero lanes
            // Lanes where a == 0 are disabled by the CC_NE0 flag
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE);
            // LREG0 = LREG0 + LREG3 = clz + d (d is negated shared trailing zero count)
            // This computes the total right-shift needed: clz(lsb) + (-shared_tz)
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG);
            // LREG0 = |a| >> (-LREG0) = |a| right-shifted to remove all trailing zeros
            // Now a is guaranteed to be odd (or zero if the lane was disabled)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX);
            // Simultaneous min/max: LREG1 = min(a, b), LREG0 = max(a, b)
            // This ensures b <= a for the subtraction step
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0,
            SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
            // LREG0 = LREG1 - LREG0 = b - a (since a >= b, result is non-positive)
            // The ARG_2SCOMP_LREG_DST flag means "negate LREG0 (dst) before adding LREG1"
            // Result is even (since both a and b are odd), ready for next iteration
    }
    // Recording ends automatically after 28 instructions have been recorded.
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Mnemonic | Description |
|---|---|---|
| `TTI_SFPMOV` | SFPMOV | Register-to-register move: `VD = VC` |
| `TTI_SFPOR` | SFPOR | Bitwise OR: `VD = VB \| VC` |
| `TTI_SFPAND` | SFPAND | Bitwise AND: `VD = VB & VC` |
| `TTI_SFPIADD` | SFPIADD | Integer add/subtract with modifier flags. Used for negation (`2SCOMP_LREG_DST`), addition, and subtraction |
| `TTI_SFPLZ` | SFPLZ | Count leading zeros: `VD = clz(VC)`. With `CC_NE0` flag, also sets lane flags for non-zero values |
| `TTI_SFPSHFT2` | SFPSHFT2 | Bitwise shift with register-specified shift amount: `VD = VB << VC` (or `>>` if VC is negative). `SHFT_LREG` mode |
| `TTI_SFPSETCC` | SFPSETCC | Set per-lane condition codes based on register value (e.g., `== 0`) |
| `TTI_SFPSWAP` | SFPSWAP | Simultaneous register swap. With `VEC_MIN_MAX` mode: `VD = min(VC, VD)`, `VC = max(VC, VD)` |
| `TTI_SFPENCC` | SFPENCC | Enable/disable conditional execution based on per-lane flags. Called with all zeros to disable (make all lanes active) |
| `TTI_SFPABS` | SFPABS | Absolute value (two's complement integer): `VD = \|VC\|` |
| `TT_SFPLOAD` | SFPLOAD | Load a vector from the destination register file into an SFPU local register |
| `TT_SFPSTORE` | SFPSTORE | Store a vector from an SFPU local register back to the destination register file |
| `TTI_REPLAY` | REPLAY | Record or replay a sequence of SFPU instructions. Mode 1 = record, mode 0 = replay |

#### SFPU Register Usage

| Register | Role in GCD Algorithm |
|---|---|
| `LREG0` | Holds operand `a`. During iterations, holds `-a` (negated for subtraction). After iteration: `a = b - a` (even result) |
| `LREG1` | Holds operand `b`. After the SFPSWAP min/max, holds the smaller value. At convergence, holds the GCD result |
| `LREG2` | Temporary register: used for `\|a\|`, `a \| b`, and intermediate shift test results |
| `LREG3` | Holds `d`: the negated count of shared trailing zeros (shared power-of-2 factor). Used throughout the iteration loop |
| DST registers | Input tiles are loaded at `dst_index * 64` offsets. `dst_reg++` advances the face pointer by `SFP_DESTREG_STRIDE` |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel waits for input tiles on `cb_in0` and `cb_in1` via `cb_wait_front()`, then reserves output space on `cb_out0` via `cb_reserve_back()`.

2. **DST register acquisition**: `tile_regs_acquire()` and `tile_regs_wait()` obtain exclusive access to the destination register file.

3. **Unpack to DST**: Input A tiles are copied to even DST slots (`i*2`) and input B tiles to odd DST slots (`i*2+1`) using `copy_tile()`. This uses the unpack engine to move data from circular buffers into the 32-bit destination registers. `copy_tile_to_dst_init_short_with_dt()` configures the data format for each copy direction.

4. **SFPU initialization** (`gcd_tile_init()` -> `calculate_sfpu_gcd_init()`):
   - Records 28 SFPU instructions (4 unrolled iterations of the 7-instruction GCD reduction step) into the hardware replay buffer.
   - Each iteration: `abs -> and -> clz -> add -> shift -> min/max swap -> subtract`.

5. **SFPU execution** (`gcd_tile(i*2, i*2+1, i*2)` -> `calculate_sfpu_gcd()`):
   - For each of the 8 faces in a tile:
     a. `SFPLOAD` loads one face of input A into `LREG0` and one face of input B into `LREG1`.
     b. `calculate_sfpu_gcd_body<31>()` runs the preamble (determine shared power of 2, ensure b is odd, take absolute values, negate a and d).
     c. The main loop replays the recorded 7-instruction iteration 30 times total (7 batches of 4 via `TTI_REPLAY`, plus 2 trailing iterations minus 1 instruction).
     d. At convergence, `LREG1` holds the GCD (the odd part), which still needs to be multiplied by the shared power of 2. However, looking at the algorithm more carefully: the shared trailing zero factor `d` is incorporated into each iteration's shift, so the result in `LREG1` is already the full GCD.
     e. `SFPSTORE` writes `LREG1` back to the output DST slot.
     f. `dst_reg++` advances to the next face.

6. **Pack to output**: `pack_tile(i*2, cb_out0)` moves the result from DST back to the output circular buffer.

7. **Release and publish**: `tile_regs_commit()` and `tile_regs_release()` free the DST registers. `cb_pop_front()` releases consumed input tiles. `cb_push_back()` makes output tiles available to the writer kernel.

#### SFPU Configuration

- **`fp32_dest_acc_en = true`**: Required for INT32/UINT32 operations, ensures full 32-bit precision in destination registers.
- **`UnpackToDestMode::UnpackToDestFp32`**: All input CBs use this mode, bypassing float format conversion and preserving the raw 32-bit integer bit patterns.
- **`APPROX` template parameter**: Passed through from the compute config but not meaningfully used by the GCD kernel (the GCD algorithm is exact, not approximate).
- **`SfpuType::gcd`**: Enum value used during `llk_math_eltwise_binary_sfpu_init` to configure SFPU address modifiers and state.
- **`VectorMode::RC`**: Default vector mode, processes all rows and columns (all 8 faces of the tile).
- **Replay buffer**: The `TTI_REPLAY` mechanism avoids instruction cache pressure by recording the 7-instruction iteration body once and replaying it 30 times. This is critical for the GCD algorithm which needs many iterations.

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `ckernel_sfpu_gcd.h` are **identical**. This is because:

1. The GCD kernel uses only integer SFPU instructions (bitwise ops, integer add, shift, abs, clz, swap) which have the same behavior on both architectures.
2. No floating-point instructions or architecture-specific intrinsics are used.
3. The `TT_SFPLOAD` and `TT_SFPSTORE` macros resolve to architecture-specific builtins (`__builtin_rvtt_wh_sfpload` vs `__builtin_rvtt_bh_sfpload`) at compile time, but the kernel source code is the same.

The LLK bridge files (`llk_math_eltwise_binary_sfpu_gcd.h`) are also identical between architectures.

## Runtime Arguments

### Reader Kernel Runtime Args
| Arg Index | Name | Description |
|---|---|---|
| 0 | `src0_addr` | DRAM address of input tensor A |
| 1 | `src1_addr` | DRAM address of input tensor B |
| 2 | `num_tiles` | Total tiles to read for this core |
| 3 | `start_id` | Starting tile ID for this core |
| 4 | `block_height` | Shard block height (tiles) |
| 5 | `block_width` | Shard block width (tiles) |
| 6 | `num_cores_y` | Number of cores in Y dimension for stride calculation |

### Compute Kernel Runtime Args
| Arg Index | Name | Description |
|---|---|---|
| 0 | `per_core_block_cnt` | Number of tile blocks to process |
| 1 | `per_core_block_size` | Number of tiles per block |

### Writer Kernel Runtime Args
| Arg Index | Name | Description |
|---|---|---|
| 0 | `dst_addr` | DRAM address of output tensor |
| 1 | `num_pages` | Number of tiles to write |
| 2 | `start_id` | Starting tile ID for output |

## Work Distribution

The program factory uses `split_work_to_cores()` for interleaved tensors and shard-spec-based distribution for sharded tensors. Key logic:

- For non-sharded: tiles are evenly distributed across available cores with remainder tiles going to the first N cores (two-group split).
- For sharded: each core processes exactly its shard's tiles, with `num_tiles_per_shard = shard_height * shard_width / TILE_HW`.
- `max_block_size` is computed via `find_max_block_size()` to optimize CB utilization -- this finds the largest power-of-2 that evenly divides the per-core tile count.

## Sharding Support

| Sharding Config | Supported | Notes |
|---|---|---|
| Height sharded | Yes | Single column of shards |
| Width sharded | Yes | Single row of shards |
| Block sharded | Yes | 2D grid of shards |
| Input A sharded | Yes | CB backed by tensor buffer |
| Input B sharded | Yes | CB backed by tensor buffer |
| Output sharded | Yes | CB backed by tensor buffer |
| Mixed (some sharded, some interleaved) | Yes | Each tensor independently configured |

## Program Caching

The `override_runtime_arguments()` method enables program caching. When the program factory is reused for tensors with the same shapes and dtypes but different addresses:
- Only runtime arguments (buffer addresses, tile counts, start IDs) are updated.
- Kernel binaries, compile-time defines, and CB configurations are reused.
- This is implemented via `set_eltwise_binary_runtime_args<false>()` which updates existing args in-place.

## Algorithm Analysis: Binary GCD (Stein's Algorithm)

The SFPU implementation uses the Binary GCD algorithm, which avoids division and modulus operations -- neither of which are available as single SFPU instructions. Instead, it relies on:

1. **Shared trailing zeros**: The GCD shares a power-of-2 factor equal to `2^ctz(a | b)`. This factor is extracted once in the preamble and tracked in `LREG3` (as `-d`).
2. **Odd reduction**: After removing the shared power of 2, the algorithm ensures one operand (b) is odd.
3. **Iterative reduction**: Each iteration:
   - Makes `a` odd by right-shifting to remove trailing zeros.
   - Ensures `b <= a` via min/max swap.
   - Computes `a = b - a` (even result, since both are odd).
4. **Convergence**: After at most 31 iterations (for 31-bit inputs), `a` becomes 0 and `b` holds the odd part of the GCD. The shared power-of-2 factor is already accounted for in the shift operations.

The total iteration count is 30 (not 31) because:
- The worst case needs 31 iterations, but the final iteration only modifies `a` (which is discarded).
- The last instruction of iteration 30 is also skipped since it only affects `a`.

**Instruction count per tile face**: approximately 12 (preamble) + 28 (7 batches x 4) + 13 (2 remaining iterations - 1) = ~53 SFPU instructions per face, times 8 faces = ~424 SFPU instructions per tile.

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Binary SFPU program factory architecture, GCD kernel file locations, define map generation
- `tenstorrent/tt-llk`: LLK binary SFPU dispatch mechanism (`_llk_math_eltwise_binary_sfpu_params_`), ckernel namespace organization
- `tenstorrent/tt-isa-documentation`: SFPU instruction semantics for SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS, modifier flags
- `tenstorrent/sfpi`: SFPU local registers (LREG0-LREG7), dst_reg, TT_SFPLOAD/TT_SFPSTORE, TTI_REPLAY recording/replay mechanism, face processing model

### Confluence References
Not consulted -- DeepWiki provided sufficient SFPU instruction detail for the integer-only instructions used by GCD.

### Glean References
Not consulted -- no confidential hardware specifications were needed beyond what DeepWiki provided.

## File Inventory

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` | Program factory: creates program, CBs, kernels |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp` | Runtime args helper (`set_eltwise_binary_runtime_args`) |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` | Factory selection logic (`is_binary_sfpu_op`) |
| `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` | Compile-time define generation (`get_defines_fp32`) |
| `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp` | `BinaryOpType` enum |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel |
| `tt_metal/hw/inc/api/compute/gcd.h` | API header: `gcd_tile()`, `gcd_tile_init()` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` | LLK bridge (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` | LLK bridge (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` | SFPU kernel implementation (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` | SFPU kernel implementation (Blackhole) |
