# MUL (binary_ng) SFPU Operation Analysis

## Overview

**Operation**: Element-wise multiplication (MUL)
**Variant**: `binary_ng` (next-generation binary operation framework)
**Namespace**: `ttnn::operations::binary_ng`
**Operation Type**: `BinaryOpType::MUL`
**SFPU Operation**: Yes -- MUL uses the SFPU path for floating-point types AND has a dedicated SFPU integer path for INT32/UINT32/UINT16 types
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The binary_ng MUL operation is a generalized element-wise multiply that supports both FPU and SFPU execution paths, multiple data types, broadcasting patterns, sharding configurations, and pre/post activation fusions. When `is_sfpu=true`, it dispatches to `mul_binary_tile` (floating-point) or `mul_int_tile` (integer) on the SFPU vector unit rather than the FPU matrix unit's `ELWMUL` instruction. The SFPU path is required for certain data type combinations and enables additional flexibility (e.g., fused post-activations).

---

## Program Factory Structure

### File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

### Entry Point
`BinaryNgDeviceOperation::ProgramFactory::create()`

### Operation Attributes
The factory receives `operation_attributes_t` which includes:
- `binary_op_type`: `BinaryOpType::MUL`
- `is_sfpu`: Whether to use the SFPU path (determined by `is_binary_sfpu_op()` based on data types)
- `is_quant_op`: Whether this is a quantization operation (always false for plain MUL)
- `subtile_broadcast_type`: Broadcast pattern (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, combinations)
- `lhs_activations`, `rhs_activations`, `post_activations`: Fused unary operations
- `scalar`: Optional scalar value for tensor-scalar operations
- `worker_grid`: Core grid for execution
- `input_dtype`: Data type for activation processing

### OpConfig for MUL

When `BinaryOpType::MUL` is encountered in `OpConfig::OpConfig()`:
```cpp
case BinaryOpType::MUL: binary_op = EnumT::MUL; break;
```

For the SFPU path (`EnumT = SfpuBinaryOp`), this sets `binary_op = SfpuBinaryOp::MUL`.

The `get_sfpu_init_fn()` function then resolves the SFPU defines:

**Floating-point types** (BFLOAT16, FLOAT32):
- `BINARY_SFPU_INIT` = `mul_binary_tile_init();`
- `BINARY_SFPU_OP` = `mul_binary_tile`

**Integer types** (INT32, UINT32, UINT16):
- `BINARY_SFPU_INIT` = `mul_int_tile_init<DataFormat::Int32>();` (or UInt32/UInt16)
- `BINARY_SFPU_OP` = `mul_int_tile<DataFormat::Int32>` (or UInt32/UInt16)

For the FPU path (`EnumT = FpuBinaryOp`), it sets `binary_op = FpuBinaryOp::MUL`, which maps to `BINARY_OP = mul_tiles` and `BINARY_OP_TYPE = EltwiseBinaryType::ELWMUL`.

---

## Circular Buffer Configuration

| CB Index | Name | Purpose | Double-buffered | Notes |
|----------|------|---------|-----------------|-------|
| c_0 | cb_src_a (cb_pre_lhs) | Input tensor A | Yes (2 tiles) or shard volume | Sharded if A is sharded |
| c_1 | cb_src_b (cb_pre_rhs) | Input tensor B / scalar | Yes (2 tiles), 1 tile for scalar, or shard volume | Sharded if B is sharded; 1 tile for scalar mode |
| c_2 | cb_out | Output tensor C | Yes (2 tiles) or shard volume | Sharded if C is sharded |
| c_3 | cb_post_lhs | LHS after pre-activation | 1 tile | Only allocated if LHS activations are defined |
| c_4 | cb_post_rhs | RHS after pre-activation | 1 tile | Only allocated if RHS activations are defined |
| c_5 | (row bcast A) | Row broadcast buffer for A | 2 tiles | Only for ROW_A or ROW_A_COL_B broadcast |
| c_6 | (row bcast B) | Row broadcast buffer for B | 2 tiles | Only for ROW_B or ROW_B_COL_A broadcast |

### SFPU-Specific CB Notes
- When `is_sfpu_op` is true and `op_type != POWER`, all source CBs (c_0, c_1, c_3, c_4) use `UnpackToDestMode::UnpackToDestFp32`. This is because SFPU operations work in FP32 internally -- data must be unpacked directly to DEST registers in FP32 format.
- `fp32_dest_acc_en` is set to true when output or both inputs are Float32/Int32/UInt32.

---

## Kernel Registration

### Compute Kernel Selection

The compute kernel is selected based on `SubtileBroadcastType` and `is_sfpu`:

| Broadcast Type | SFPU Kernel File |
|---|---|
| NONE | `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| SCALAR_A, SCALAR_B, COL_A, COL_B, ROW_B_COL_A, ROW_A_COL_B | `kernels/compute/eltwise_binary_sfpu.cpp` |
| (scalar mode, no B tensor) | `kernels/compute/eltwise_binary_sfpu_scalar.cpp` |
| ROW_A, ROW_B | `kernels_ng/compute/eltwise_binary_sfpu_row_bcast.cpp` |
| ROW_A_COL_B, ROW_B_COL_A | `kernels_ng/compute/eltwise_binary_sfpu_row_col_bcast.cpp` |

### Compute Kernel Compile-Time Args
- `compile_args = {num_tiles_per_cycle}` where `num_tiles_per_cycle = 1` (always 1 output tile per cycle)

### Compute Kernel Runtime Args
- `{c_num_tiles, freq, counter, compute_scalar_value}` (4 args)
  - `c_num_tiles`: Total tiles to process on this core
  - `freq`: Broadcast frequency (tiles between broadcast reloads; 1 for NONE/ROW broadcast)
  - `counter`: Starting offset within broadcast cycle
  - `compute_scalar_value`: 0 for MUL (used by quantization/WHERE operations)

### Reader Kernel
Selected based on broadcast type and overridden to `ReaderNoBcastNg` (or row/col/scalar variants) when B tensor is present. Reader reads both A and B, while writer only writes output.

### Writer Kernel
- `WriterScalar` when B is a scalar (writer fills B tile and writes output)
- `WriterNoBcastNg` when B is a tensor (writer only writes output)

---

## Kernel Implementations

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis. The primary compute kernel for MUL (no-broadcast, two-tensor mode) is shown below.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU unary operation includes -- needed for pre/post activation fusions
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// SFPU binary operation API headers -- each provides tile-level init/op functions
#include "api/compute/eltwise_binary_sfpu.h"  // mul_binary_tile, add_binary_tile, etc.
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"         // mul_int_tile<DataFormat>
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

// Macro utilities for pre/post activation processing and broadcast logic
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime arg 0: number of output tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: tiles produced per read-compute-write cycle (always 1)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB assignments: c_0 = LHS input, c_1 = RHS input, c_2 = output
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // If LHS/RHS activations are defined, use intermediate CBs c_3/c_4; otherwise alias to input CBs
    // For plain MUL, HAS_ACTIVATIONS(LHS) and HAS_ACTIVATIONS(RHS) are both 0
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize unary op common state (sets up tile copy/pack infrastructure)
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // If RELU is the sole post-activation, it is fused into the packer hardware
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Initialize the SFPU binary operation once if no pre/post activations are present.
    // For MUL, this expands to: mul_binary_tile_init(); (or mul_int_tile_init<DataFormat::X>())
    // This configures SFPU address modifiers and any programmable constants.
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // Main processing loop: one tile per iteration
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS: If LHS activations exist, apply them (copy LHS to intermediate CB via unary SFPU ops)
        // For plain MUL, PREPROCESS(LHS, ...) is a no-op
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for LHS tile to be available in its CB (reader must have pushed it)
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        // Same for RHS
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in the output CB for the result tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Re-initialize SFPU if activations changed the SFPU state
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire DEST register file -- blocks until DEST is available
        tile_regs_acquire();

        // Unpack LHS tile from CB to DEST register at even indices (0, 2, 4, ...)
        // The init_short_with_dt call reconfigures the unpacker for the LHS data format
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // copy_tile(cb, cb_tile_index, dst_index): unpack tile from CB into DEST[dst_index]
            // LHS goes to DEST[0] (i*2 = 0 when i=0, num_tiles_per_cycle=1)
            copy_tile(cb_post_lhs, i, i * 2);
        }

        // Unpack RHS tile from CB to DEST register at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // RHS goes to DEST[1] (i*2+1 = 1 when i=0)
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // Re-initialize SFPU if post-activations require fresh state each tile
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU binary operation: DEST[0] = mul(DEST[0], DEST[1])
            // For float MUL: mul_binary_tile(0, 1, 0)
            // For int MUL: mul_int_tile<DataFormat::Int32>(0, 1, 0)
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Apply post-activations (e.g., RELU, GELU, etc.) to the result in DEST[0]
            // For plain MUL with no post-activations, this is empty
            PROCESS_POST_ACTIVATIONS(i * 2);
        }

        // Signal that DEST registers are ready for packing
        tile_regs_commit();

        // Wait for DEST to be committed, then pack result into output CB
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Pack tile from DEST[0] to output CB (format conversion happens here)
            pack_tile(i * 2, cb_out);
        }
        // Release DEST registers for next iteration
        tile_regs_release();

        // Push the output tile to the output CB (writer can now consume it)
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Free the input tiles from their CBs (reader can now write new tiles)
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### Additional Compute Kernel Variants

**Broadcast variant** (`eltwise_binary_sfpu.cpp`): Used for SCALAR_A/B, COL_A/B broadcast types. Has a `process_tile()` function with an inner loop that re-reads the "other" operand for each broadcast iteration while keeping the "bcast" operand loaded once. Uses `tile_freq` and `tile_start` runtime args to manage broadcast cycle.

**Scalar variant** (`eltwise_binary_sfpu_scalar.cpp`): Used when B is a scalar (not a tensor). The RHS CB contains a single tile (filled by the writer kernel) that is loaded once before the main loop. LHS tiles iterate normally. The scalar tile is popped only after all tiles are processed.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to.

#### Floating-Point Path: `mul_binary_tile`

##### SFPU Kernel File
`tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (API header)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (implementation, identical for Blackhole)

##### Annotated SFPU Kernel Source (Floating-Point)

```cpp
// API-level function called from the compute kernel
// Located in: tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    // MATH() macro ensures this runs only on the math RISC-V processor (TRISC_MATH)
    // Dispatches to the specialized mul function (not the generic binary function)
    // because MUL needs special handling for bf16 rounding and zero-multiplication semantics
    MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, ckernel::BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)));
}

ALWI void mul_binary_tile_init() {
    // Initializes SFPU configuration registers and address modifiers for the MUL binary op
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::MUL>()));
}
```

```cpp
// LLK-level dispatch
// Located in: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h

template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    // 1. Initialize the base SFPU binary infrastructure (address modifiers, config registers)
    // 2. Call the SFPU-specific init function for the binary op
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>);
}

template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_mul(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    // _llk_math_eltwise_binary_sfpu_params_ handles:
    //   1. Setting DEST write address based on odst
    //   2. Stalling until math unit is ready
    //   3. Calling the SFPU function (calculate_sfpu_binary_mul) for each vector section
    //   4. Clearing DEST address and waiting for SFPU completion
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_mul<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}
```

```cpp
// Core SFPU multiply implementation
// Located in: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

// Helper: Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This is needed because the SFPU computes in FP32 but the output may need BF16 precision
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    // Extract bit 16 (LSB of bf16 mantissa) for tie-breaking
    sfpi::vUInt lsb = (bits >> 16) & 1;
    // Add 0x7fff + lsb: implements banker's rounding
    // - Lower 16 bits > 0x8000: rounds up
    // - Lower 16 bits < 0x8000: rounds down
    // - Lower 16 bits = 0x8000 (exact midpoint): rounds to even
    bits = bits + 0x7fffU + lsb;
    // Zero out the lower 16 bits to get bf16 in upper 16 bits
    bits = bits & 0xFFFF0000U;
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // DEST tiles are 32 rows when addressed via SFPI (64 / SFP_DESTREG_STRIDE = 32)
    constexpr uint dst_tile_size_sfpi = 32;

    // Process 8 iterations, each handling 4 elements (32 lanes wide)
    // 8 iterations * 4 elements = 32 rows = one tile face
    // A full 32x32 tile has 32 rows, processed in one pass through the SFPU
    for (int d = 0; d < ITERATIONS; d++) {
        // Load 4 elements from DEST at the input tile positions
        // dst_reg[index * 32] reads 4 consecutive elements from the tile at 'index'
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Element-wise multiply using SFPI vFloat operator*
        // This compiles to the SFPMUL hardware instruction
        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // When DEST is in BF16 mode, apply software RNE rounding
            // This ensures the SFPU result matches FPU ELWMUL behavior for BF16
            result = float32_to_bf16_rne(result);

            // Special case: ensure 0 * x = 0 and x * 0 = 0
            // The FPU ELWMUL instruction guarantees this, but SFPU floating-point multiply
            // can produce denormals or NaN for 0 * inf. This conditional enforces FPU parity.
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        // Store result back to DEST at the output tile position
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        // Advance the SFPU row pointer by 1 row (4 elements)
        sfpi::dst_reg++;
    }
}
```

#### Integer Path: `mul_int_tile`

##### SFPU Kernel File (Wormhole)
`tt_metal/hw/inc/api/compute/mul_int_sfpu.h` (API header)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` (implementation)

##### SFPU Kernel File (Blackhole)
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` (implementation -- significantly different from Wormhole)

##### Annotated SFPU Kernel Source (Integer -- Wormhole)

```cpp
// API-level function
// Located in: tt_metal/hw/inc/api/compute/mul_int_sfpu.h
template <DataFormat data_format>
ALWI void mul_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    // For Int32/UInt32: dispatches to mul_int32 (32-bit integer multiply)
    // For UInt16: dispatches to _mul_int_ (16-bit integer multiply via 8-bit decomposition)
    MATH((llk_math_eltwise_binary_sfpu_mul_int<APPROX, data_format>(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void mul_int_tile_init() {
    // For UInt16: uses SfpuType::mul_uint16 with _init_mul_int_ init function
    // For Int32/UInt32: uses SfpuType::mul_int32 with mul_int32_init function
    MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, data_format>()));
}
```

```cpp
// Wormhole 32-bit integer multiply implementation
// Located in: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h
//
// Strategy: Split each 32-bit integer into three 11-bit chunks, convert to FP32,
// compute partial products using SFPMAD, then reassemble the full 32-bit result.
// This is necessary because the SFPU has no native 32-bit integer multiply instruction.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        // Decomposition: a = (a2 << 22) | (a1 << 11) | a0
        //                b = (b2 << 22) | (b1 << 11) | b0
        // Then: a * b = (top << 22) + (mid << 11) + low
        // Where:
        //   top = a0*b2 + a1*b1 + a2*b0  (max 23 bits)
        //   mid = a0*b1 + a1*b0           (max 23 bits)
        //   low = a0*b0                   (max 22 bits)

        // Load a0 from DEST as INT32
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);

        // a1 = a >> 11 (shift right by 11 using SFPSHFT2 with programmable constant LREG13 = -11)
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5);
        // a2 = a1 >> 11 = a >> 22
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5);

        // a1 = (a1 & 0x7ff) -- mask to 11 bits using programmable constant LREG12 = 0x7ff
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);
        // Convert a1 from integer to FP32 for multiplication
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);

        // a2 as FP32 (no masking needed -- upper bits are zero after double shift)
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);

        // a0 = (a0 & 0x7ff) as FP32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // Load b0 from DEST as INT32
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // b1 = b >> 11
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5);
        // b2 = b >> 22
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5);

        // b2 as FP32
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // top = a0*b2 + 2**23 (the 2**23 bias is in LREG14 for the mantissa extraction trick)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

        // b1 = (b1 & 0x7ff) as FP32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // top += a1*b1
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // b0 = (b0 & 0x7ff) as FP32
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        // top += a2*b0
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // mid = a0*b1 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);

        // low = a0*b0 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);

        // mid += a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

        // Extract integer values from FP32 mantissas
        // The trick: adding 2**23 to a 23-bit integer gives an FP32 whose mantissa bits
        // are exactly the integer value. SFPEXMAN extracts these bits.
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);

        // Shift partial products to their correct positions
        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);  // top <<= 22
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);  // mid <<= 11

        // Sum: result = low + mid + top
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

        // Store result back to DEST as INT32
        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    // Program SFPU constant registers:
    sfpi::vConstIntPrgm0 = 0x7ff;       // LREG12: 11-bit mask
    sfpi::vConstIntPrgm1 = -11;         // LREG13: shift amount for 11-bit decomposition
    sfpi::vConstFloatPrgm2 = 8388608.0f; // LREG14: 2**23 (mantissa extraction bias)
}
```

##### Annotated SFPU Kernel Source (Integer -- Blackhole)

The Blackhole implementation uses `SFPLOADMACRO` and `SFPMUL24` instructions that are unique to the Blackhole architecture. This achieves significantly higher throughput (8 cycles per row vs. many more on Wormhole).

```cpp
// Located in: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {

    constexpr uint dst_tile_size = 64;

    uint offset_in0 = dst_index_in0 * dst_tile_size;
    uint offset_in1 = dst_index_in1 * dst_tile_size;
    uint offset_out = dst_index_out * dst_tile_size;

    // Blackhole strategy: Use SFPLOADMACRO to schedule parallel operations.
    // The algorithm splits each 32-bit input into a low 23-bit part and a high part (>>23),
    // then uses SFPMUL24 for 23-bit integer multiplication:
    //
    // a1 = a >> 23
    // b1 = b >> 23
    // cross0 = mul24_lo(a1, b)
    // cross1 = mul24_lo(a, b1)
    // lo = mul24_lo(a, b)
    // hi = mul24_hi(a, b)
    // result = ((hi + cross0 + cross1) << 23) + lo
    //
    // SFPLOADMACRO schedules instructions across 4 sub-units (simple, MAD, round, store)
    // achieving up to 4 instructions per cycle throughput.

    constexpr uint a0 = p_sfpu::LREG0;
    constexpr uint b0 = p_sfpu::LREG0;
    constexpr uint a1 = p_sfpu::LREG1;
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint b2 = p_sfpu::LREG3;
    constexpr uint c = p_sfpu::LREG4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load b0 from DEST
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);
        // Macro 0: loads a, schedules shft2(a, -23) in round unit, mul24_lo(b, a1) in MAD unit
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2));
        // Macro 1: loads b, schedules shft2(b, -23) in round, mul24_lo(a, b1) in MAD, iadd in simple
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2));
        // Load a0 (overwriting b0 since they share LREG0)
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);
        // Macro 2: loads b again, schedules mul24_lo(a, b) in MAD, mul24_hi(a, b2) in explicit instr
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2));
        // c = mul24_hi(a0, b2): upper 23 bits of 23-bit * 23-bit product
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, sfpi::SFPMUL24_MOD1_UPPER);
        // b1 = b1 + a1 (accumulate cross products)
        TTI_SFPIADD(0, a1, b1, sfpi::SFPIADD_MOD1_CC_NONE);
        // Macro 3: schedules shft(b1, 23), iadd(b2+c), and store to DEST
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));
    }
    // Pipeline drain: NOPs to allow scheduled macro instructions to complete
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint c = p_sfpu::LREG4;

    // Load instruction templates into SFPLOADMACRO configuration.
    // Templates are stored via "backdoor load" (VD=12+i selects template i).
    // Template 0: SFPSHFT2 with immediate -23 (arithmetic right shift by 23)
    TTI_SFPSHFT2(-23 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    // Template 1: SFPMUL24 lower 23 bits
    TTI_SFPMUL24(0, 0, p_sfpu::LCONST_0, 13, sfpi::SFPMUL24_MOD1_LOWER);
    // Template 2: SFPSHFT with immediate 23 (left shift by 23), using VC for source
    TTI_SFPSHFT(23, b1, 14, 1 | 4);
    // Template 3: SFPIADD (integer add)
    TTI_SFPIADD(0, c, 15, sfpi::SFPIADD_MOD1_CC_NONE);

    // Configure macro 0-3 sequences via SFPCONFIG
    // Each macro specifies which templates to execute in which sub-units (simple, MAD, round, store)
    // Macro 0: round=shft2, mad=mul24_lo
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x80 | (1 << 3) | (4 + 1)) << 8);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x80 | (0 << 3) | (4 + 0)) << 0);
    TTI_SFPCONFIG(0, 4, 0);

    // Macro 1: simple=iadd(b1+c), mad=mul24_lo, round=shft2
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER,
        ((0x80 | (1 << 3) | (4 + 1)) << 8) | (0x80 | (4 << 3) | (4 + 3)));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x80 | (0 << 3) | (4 + 0)) << 0);
    TTI_SFPCONFIG(0, 4+1, 0);

    // Macro 2: simple=iadd(VD=16, delay=4), mad=mul24_lo
    TTI_SFPCONFIG(
        ((0x80 | (1 << 3) | (4 + 1)) << 8) | (0x80 | 0x40 | (4 << 3) | (4 + 3)),
        4+2, 1);

    // Macro 3: simple=shft(23), store to DEST via VD=16
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x80 | (0 << 3) | (4 + 2)));
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x40 | (2 << 3) | 3) << 8);
    TTI_SFPCONFIG(0, 4+3, 0);

    // Misc config: StoreMod0 format, UsesLoadMod0ForStore, UnitDelayKind
    TTI_SFPCONFIG(0xff0, 8, 1);
}
```

#### SFPU Instructions Used

**Floating-Point MUL (`calculate_sfpu_binary_mul`)**:

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[index]` (read) | SFPLOAD: Load 4 elements from DEST register file into an SFPU LREG |
| `in0 * in1` (vFloat operator*) | SFPMUL: Element-wise floating-point multiplication across 32 lanes |
| `sfpi::reinterpret<vUInt>(x)` | Reinterpret float bits as unsigned int (no instruction, register reinterpretation) |
| `bits >> 16`, `bits & 1` | SFPSHFT, SFPAND: Bit manipulation for RNE rounding |
| `bits + 0x7fffU + lsb` | SFPIADD: Integer addition for rounding bias |
| `bits & 0xFFFF0000U` | SFPAND: Mask lower 16 bits for bf16 truncation |
| `v_if(in0 == 0 \|\| in1 == 0)` | SFPSETCC + conditional execution: Check for zero inputs |
| `sfpi::dst_reg[index]` (write) | SFPSTORE: Write 4 elements from SFPU LREG back to DEST |
| `sfpi::dst_reg++` | Advance SFPU row pointer (DEST address increment) |

**Integer MUL -- Wormhole (`mul_int32`)**:

| Instruction | Description |
|---|---|
| `TT_SFPLOAD` | Load 32-bit integer from DEST into LREG |
| `TTI_SFPSHFT2` | Arithmetic right shift (used to extract 11-bit chunks) |
| `TTI_SFPAND` | Bitwise AND (mask to 11 bits using 0x7ff constant) |
| `TTI_SFPCAST` | Convert sign-magnitude integer to FP32 |
| `TTI_SFPMAD` | Fused multiply-add: compute partial products and accumulate with 2^23 bias |
| `TTI_SFPEXMAN` | Extract mantissa bits from FP32 (recovers integer from biased float) |
| `TTI_SFPSHFT` | Left shift (position partial products: top<<22, mid<<11) |
| `TTI_SFPIADD` | Integer addition (sum partial products) |
| `TT_SFPSTORE` | Store 32-bit integer result back to DEST |

**Integer MUL -- Blackhole (`mul_int32`)**:

| Instruction | Description |
|---|---|
| `TT_SFPLOAD` | Load 32-bit integer from DEST into LREG |
| `TT_SFPLOADMACRO` | Load from DEST + schedule up to 4 instructions across sub-units |
| `TTI_SFPMUL24` | 23-bit integer multiply (upper or lower 23 bits of product) -- Blackhole only |
| `TTI_SFPSHFT2` (via macro) | Arithmetic right shift by 23 |
| `TTI_SFPIADD` | Integer addition (accumulate cross products) |
| `TTI_SFPSHFT` (via macro) | Left shift by 23 |
| `TTI_SFPNOP` | Pipeline drain (wait for macro-scheduled instructions to complete) |
| `TTI_SFPCONFIG` | Configure SFPLOADMACRO instruction templates and sequences |
| `TTI_SFPLOADI` | Load immediate values for macro configuration |

#### SFPU Register Usage

**Floating-Point Path**:
- DEST registers: Two input tiles at even/odd indices (e.g., DEST[0] for LHS, DEST[1] for RHS)
- SFPU LREGs: Implicitly used by SFPI `vFloat` operations (compiler allocates from LREG0-LREG7)
- No programmable constants used for float mul

**Integer Path (Wormhole)**:
- DEST registers: Same two-tile layout as float
- LREG0: a0, then low partial product, then final result
- LREG1: b0
- LREG2: a1
- LREG3: b1
- LREG4: a2
- LREG5: b2, then top partial product
- LREG6: mid partial product
- LREG12 (vConstIntPrgm0): 0x7ff (11-bit mask)
- LREG13 (vConstIntPrgm1): -11 (shift amount)
- LREG14 (vConstFloatPrgm2): 2^23 = 8388608.0f (mantissa extraction bias)

**Integer Path (Blackhole)**:
- LREG0: shared for a0/b0 (reused)
- LREG1: a1
- LREG2: b1
- LREG3: b2
- LREG4: c (accumulator / output)
- Instruction templates stored in template slots 0-3 (accessed via VD=12-15)
- SFPLOADMACRO configuration registers for scheduling

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front()` on input CBs and `cb_reserve_back()` on the output CB. This blocks until the reader has pushed tiles and there is space for the output.

2. **Unpack to DEST**: `copy_tile(cb, index, dst_index)` unpacks a tile from the circular buffer into the DEST register file. For SFPU binary ops, LHS goes to an even DEST slot (e.g., 0) and RHS to the next odd slot (e.g., 1). When `UnpackToDestMode::UnpackToDestFp32` is set (which it is for SFPU MUL), the unpacker converts input data to FP32 in DEST.

3. **SFPU computation**: `BINARY_SFPU_OP(0, 1, 0)` invokes `mul_binary_tile(0, 1, 0)` which:
   - For float: Iterates 8 times over 32 rows, loading 4 elements per iteration from DEST[0] and DEST[1], multiplying them, optionally applying bf16 RNE rounding, and storing back to DEST[0].
   - For int32: Decomposes each 32-bit integer into 11-bit (WH) or 23-bit (BH) chunks, computes partial products via SFPMAD/SFPMUL24, reassembles the full product, and stores back.

4. **Pack to output CB**: `pack_tile(0, cb_out)` reads the result from DEST[0] and packs it into the output circular buffer, performing any necessary format conversion (e.g., FP32 to BF16).

5. **CB management**: `cb_push_back(cb_out, 1)` makes the output tile available to the writer kernel. `cb_pop_front()` on input CBs frees them for the reader to reuse.

6. **Looping**: Steps 1-5 repeat for each tile assigned to this core. For broadcast variants, the "bcast" operand is loaded once and reused across multiple iterations of the "other" operand.

#### SFPU Configuration

- **`UnpackToDestMode::UnpackToDestFp32`**: For all SFPU MUL ops except POWER, both input CBs are configured to unpack directly to FP32 in DEST. This ensures the SFPU operates on full-precision data.
- **`fp32_dest_acc_en`**: Enabled when output or both inputs are FP32/INT32/UINT32. This controls whether the bf16 RNE rounding and zero-multiplication guard are applied in the SFPU kernel.
- **`DST_ACCUM_MODE`**: Template parameter passed through to `calculate_sfpu_binary_mul`, corresponding to `fp32_dest_acc_en`.
- **`APPROX`**: Approximation mode flag (compile-time). For MUL, approximation mode has no effect since the operation is exact (no iterative approximation needed).
- **`BINARY_SFPU_INIT` placement**: The init call can be hoisted outside the loop (when no activations change SFPU state) or placed per-tile (when post-activations require re-initialization).

#### Hardware Compatibility Notes

**Wormhole B0**:
- Integer multiply uses an 11-bit decomposition strategy with SFPMAD and mantissa extraction tricks.
- Requires 6 LREGs for partial products and 3 programmable constant registers.
- Higher instruction count per row (many SFPSHFT2, SFPAND, SFPCAST, SFPMAD, SFPEXMAN, SFPSHFT, SFPIADD instructions).
- No SFPMUL24 or SFPLOADMACRO support.

**Blackhole**:
- Integer multiply uses a 23-bit decomposition with the native `SFPMUL24` instruction (new in Blackhole).
- Uses `SFPLOADMACRO` for instruction-level parallelism: schedules up to 4 instructions per cycle across simple, MAD, round, and store sub-units.
- Achieves approximately 8 cycles per row throughput (vs. significantly more on Wormhole).
- Requires careful macro configuration via `SFPCONFIG` during init.
- Known hardware bugs: automatic stalling may not work correctly for SFPAND, SFPIADD, SFPSHFT, SFPCONFIG -- the Blackhole implementation avoids these by using SFPLOADMACRO scheduling instead.

**Floating-point multiply**: Identical implementation on both Wormhole and Blackhole. Uses SFPI `vFloat` operator* which compiles to SFPMUL. The bf16 RNE rounding and zero-guard logic is the same.

---

## Data Flow Architecture

```
Reader Kernel                    Compute Kernel                    Writer Kernel
+-----------+                    +-------------+                   +-----------+
| Read A    |---> CB c_0 (LHS)  |             |                   |           |
| from DRAM |    (or sharded)   | Unpack LHS  |                   |           |
|           |                    | to DEST[0]  |                   |           |
| Read B    |---> CB c_1 (RHS)  |             |                   |           |
| from DRAM |    (or sharded)   | Unpack RHS  |                   |           |
|           |                    | to DEST[1]  |                   |           |
+-----------+                    |             |                   |           |
                                 | SFPU MUL:   |                   |           |
                                 | DEST[0] =   |                   |           |
                                 | DEST[0] *   |                   |           |
                                 | DEST[1]     |                   |           |
                                 |             |                   |           |
                                 | Pack DEST[0]|---> CB c_2 (OUT) | Write OUT |
                                 | to output CB|    (or sharded)  | to DRAM   |
                                 +-------------+                   +-----------+
```

For scalar mode, the writer kernel fills CB c_1 with the scalar tile before the compute loop begins. The reader only reads tensor A into CB c_0.

---

## Broadcasting Support

The binary_ng framework supports 9 broadcasting patterns via `SubtileBroadcastType`:

| Type | Description | Compute Kernel | Behavior |
|---|---|---|---|
| NONE | No broadcast, equal tile dims | `eltwise_binary_sfpu_no_bcast.cpp` | 1:1 tile processing |
| SCALAR_A | A is a scalar tile | `eltwise_binary_sfpu.cpp` | A loaded once, B iterates |
| SCALAR_B | B is a scalar tile | `eltwise_binary_sfpu.cpp` | B loaded once, A iterates |
| COL_A | A has single tile column | `eltwise_binary_sfpu.cpp` | A reloaded every Wt tiles |
| COL_B | B has single tile column | `eltwise_binary_sfpu.cpp` | B reloaded every Wt tiles |
| ROW_A | A has single tile row | `eltwise_binary_sfpu_row_bcast.cpp` | Row broadcast with LLK support |
| ROW_B | B has single tile row | `eltwise_binary_sfpu_row_bcast.cpp` | Row broadcast with LLK support |
| ROW_A_COL_B | A=row, B=column | `eltwise_binary_sfpu_row_col_bcast.cpp` | Mixed broadcast |
| ROW_B_COL_A | B=row, A=column | `eltwise_binary_sfpu_row_col_bcast.cpp` | Mixed broadcast |

The `freq` and `counter` runtime args control the broadcast reloading pattern. For SCALAR broadcast, `freq = Ht * Wt` (all tiles share one scalar). For COL broadcast, `freq = Wt` (reload every width-stride).

---

## Work Distribution

Tiles are distributed across cores using `split_work_to_cores()`:
- For non-sharded tensors: Output tiles are evenly split across the worker grid, with remainder tiles assigned to a second core group.
- For sharded tensors: Each core processes its own shard. The shard shape may vary on edge cores (last row/column of the core grid).

Runtime args per core:
- **Reader**: 21 args (buffer address, start tile, tile counts, dimension strides, shapes for A and B)
- **Writer**: 11 args (buffer address, start tile, tile counts, dimension shapes)
- **Compute**: 4 args (num_tiles, freq, counter, scalar_value)
- Cores outside the active set receive zeroed-out args and exit immediately.

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: binary_ng operation structure, program factory patterns, OpConfig, kernel dispatch
- `tenstorrent/tt-llk`: ckernel namespace, binary SFPU operations, LLK API hierarchy, BinaryOp enum
- `tenstorrent/sfpi`: SFPI vFloat operator* compiling to SFPMUL, SFPMAD instruction semantics
- `tenstorrent/tt-isa-documentation`: SFPMAD, SFPLOAD, SFPSTORE, SFPSHFT2, SFPAND, SFPCAST, SFPEXMAN, SFPSHFT, SFPIADD, SFPMUL24, SFPLOADMACRO, SFPCONFIG instruction descriptions

### Confluence References
Not consulted for this analysis -- DeepWiki provided sufficient SFPU instruction detail.

### Glean References
Not consulted for this analysis -- all necessary information was available from source code and DeepWiki.

---

## Source File Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` | Program factory: creates kernels, CBs, distributes work |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` | OpConfig, KernelName enum, utility declarations |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` | OpConfig implementation, get_sfpu_init_fn, kernel path resolution |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp` | Device operation struct, attributes, tensor args |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | Compute kernel: no-broadcast SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` | Compute kernel: broadcast SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` | Compute kernel: scalar SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp` | PREPROCESS macro for pre-activation processing |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp` | HAS_ACTIVATIONS, BCAST_OP macros |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp` | Reader kernel: reads A (and B) tiles from DRAM/L1 |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` | Writer kernel: fills scalar tile, writes output |
| `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` | API: mul_binary_tile, mul_binary_tile_init |
| `tt_metal/hw/inc/api/compute/mul_int_sfpu.h` | API: mul_int_tile, mul_int_tile_init |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` | LLK: dispatch to calculate_sfpu_binary_mul |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` | SFPU kernel: float mul (WH) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` | SFPU kernel: int32 mul (WH, 11-bit decomposition) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` | SFPU kernel: float mul (BH, identical to WH) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` | SFPU kernel: int32 mul (BH, SFPMUL24 + SFPLOADMACRO) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h` | LLK: dispatch to mul_int32 / _mul_int_ |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` | LLK: binary SFPU init infrastructure |
