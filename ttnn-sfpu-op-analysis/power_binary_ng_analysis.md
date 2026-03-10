# POWER (binary_ng) -- SFPU Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| Operation Name | POWER (binary) |
| Operation Type | Binary SFPU elementwise |
| Framework | binary_ng (next-generation binary operation framework) |
| Namespace | `ttnn::operations::binary_ng` |
| BinaryOpType | `BinaryOpType::POWER` |
| SfpuBinaryOp | `SfpuBinaryOp::POWER` |
| Compute Semantics | `output[i] = base[i] ** exponent[i]` (elementwise power) |
| SFPU-Only | Yes -- POWER is exclusively an SFPU operation; FPU path throws `TT_THROW` |
| FP32 Dest Accumulator | Conditional -- only when input dtype is FLOAT32 (unlike most other SFPU ops which unconditionally enable fp32 dest) |

### Why binary_ng?

The binary_ng framework is the next-generation binary operation infrastructure that replaces the legacy binary operation path. It provides a unified program factory for all binary operations (FPU and SFPU) with support for:
- Subtile broadcasting (scalar, row, column, mixed)
- Pre/post activation fusions (LHS, RHS, and POST processing)
- Sharded memory layouts with native L1 sharding
- Multi-dimensional tensor broadcasting (up to rank 6+)
- Scalar operand mode (one tensor, one scalar)

POWER uses binary_ng because it requires two input tiles (base and exponent), which is the natural fit for the binary operation framework rather than the unary framework.

## Program Factory

### File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

### Operation Attributes

The `BinaryNgDeviceOperation::operation_attributes_t` struct (defined in `binary_ng_device_operation.hpp`) carries:

| Field | Type | Role for POWER |
|---|---|---|
| `binary_op_type` | `BinaryOpType` | Set to `BinaryOpType::POWER` |
| `is_sfpu` | `bool` | `true` -- POWER is SFPU-only |
| `is_quant_op` | `bool` | `false` |
| `is_where_op` | `bool` | `false` |
| `subtile_broadcast_type` | `SubtileBroadcastType` | Determined by input shapes |
| `scalar` | `optional<ScalarVariant>` | Set when one operand is a scalar |
| `lhs_activations` | `SmallVector<EltwiseUnaryWithParam>` | Pre-processing for LHS (base) |
| `rhs_activations` | `SmallVector<EltwiseUnaryWithParam>` | Pre-processing for RHS (exponent) |
| `post_activations` | `SmallVector<EltwiseUnaryWithParam>` | Post-processing on result |
| `input_dtype` | `DataType` | Input data type |
| `worker_grid` | `CoreRangeSet` | Available compute cores |

### OpConfig for POWER

In `binary_ng_utils.cpp`, the `OpConfig` constructor maps `BinaryOpType::POWER` to `SfpuBinaryOp::POWER`. There is no pre/post processing -- POWER maps directly to the SFPU op with no decomposition:

```cpp
case BinaryOpType::POWER:
    if (is_sfpu_op()) {
        binary_op = SfpuBinaryOp::POWER;
    } else {
        TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
    }
    break;
```

The `as_defines()` method then calls `get_sfpu_init_fn(SfpuBinaryOp::POWER, dtype)` which returns:
- `BINARY_SFPU_INIT` = `"power_binary_tile_init();"`
- `BINARY_SFPU_OP` = `"power_binary_tile"`

### POWER-Specific UnpackToDestMode Handling

POWER has unique behavior compared to other binary SFPU operations. Most SFPU ops unconditionally set `UnpackToDestMode::UnpackToDestFp32` for all source CBs. POWER conditionally sets it based on the actual input data type:

```cpp
if (op_type != BinaryOpType::POWER) {
    // All other SFPU ops: unconditionally fp32 unpack-to-dest
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    // ...
} else {
    // POWER: only fp32 unpack-to-dest if input is actually float32
    unpack_to_dest_mode[src0_cb_index] =
        (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    unpack_to_dest_mode[src1_cb_index] =
        (b_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    // ...
}
```

This design decision exists because the SFPU power kernel has two algorithm paths (`_sfpu_binary_power_21f_` for bfloat16 and `_sfpu_binary_power_f32_` for float32), and forcing fp32 unpack-to-dest for bfloat16 inputs would waste DEST register capacity (halving the number of available tiles) without a meaningful accuracy benefit for the 21f approximation path.

### Kernel Selection Logic

The compute kernel selected depends on the `SubtileBroadcastType` and whether `b` is a tensor or scalar:

| Condition | Compute Kernel | Reader Kernel | Writer Kernel |
|---|---|---|---|
| `b` is scalar (no tensor) | `ComputeScalar` -> `eltwise_binary_sfpu_scalar.cpp` | `ReaderNoBcast` -> `reader_interleaved_no_bcast.cpp` | `WriterScalar` -> `writer_interleaved_scalar.cpp` |
| `b` is tensor, NONE broadcast | `ComputeNoBcast` -> `eltwise_binary_sfpu_no_bcast.cpp` | `ReaderNoBcastNg` -> `reader_interleaved_no_bcast.cpp` (ng) | `WriterNoBcastNg` -> `writer_interleaved_no_bcast.cpp` (ng) |
| `b` is tensor, SCALAR_A/B | `ComputeBcast` -> `eltwise_binary_sfpu.cpp` | `ReaderScalarBcastNg` | `WriterNoBcastNg` |
| `b` is tensor, COL_A/COL_B | `ComputeBcast` -> `eltwise_binary_sfpu.cpp` | `ReaderColBcastNg` | `WriterNoBcastNg` |
| `b` is tensor, ROW_A/ROW_B | `ComputeNoBcast` (or `ComputeRowBcastNg` if LLK bcast) | `ReaderRowBcastNg` | `WriterNoBcastNg` |

All SFPU compute kernels use `is_sfpu=true` in `get_kernel_file_path()`, which selects the `*_sfpu_*` variant of the compute kernel file.

## Circular Buffer Configuration

| CB Index | Constant | Purpose | Size (tiles) | Data Format |
|---|---|---|---|---|
| `c_0` | `cb_pre_lhs` / `cb_src_a` | Input A (base) | `a_num_tiles_per_shard` or 2 (double buffer) | `a_data_format` |
| `c_1` | `cb_pre_rhs` / `cb_src_b` | Input B (exponent) | `b_num_tiles_per_shard` or 2 (1 if scalar) | `b_data_format` |
| `c_2` | `cb_out` / `cb_src_c` | Output | `c_num_tiles_per_shard` or 2 (double buffer) | `c_data_format` |
| `c_3` | `cb_post_lhs` | LHS intermediate (only if LHS activations present) | 1 | Same as `a_data_format` (for SFPU ops) |
| `c_4` | `cb_post_rhs` | RHS intermediate (only if RHS activations present) | 1 | Same as `b_data_format` (for SFPU ops) |
| `c_5` | (row bcast A) | Only for ROW_A / ROW_A_COL_B broadcast | 2 | `a_data_format` |
| `c_6` | (row bcast B) | Only for ROW_B / ROW_B_COL_A broadcast | 2 | `b_data_format` |

For POWER, `c_3` and `c_4` are typically not used since POWER has no `process_lhs` or `process_rhs` activations by default (it maps directly to the SFPU op). However, the user can supply custom lhs/rhs/post activations via the API.

## Kernel Implementations

### Compute Kernel

The primary compute kernel for POWER (no broadcast, two tensor inputs) is:

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU split includes provide individual SFPU math function headers
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
// Eltwise unary compute API (used for unary_op_init_common, copy_tile, pack_tile, etc.)
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Binary SFPU API header -- provides power_binary_tile, power_binary_tile_init, etc.
#include "api/compute/eltwise_binary_sfpu.h"
// Additional binary SFPU includes for other operations that share this kernel file
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

// Common macro utilities for activation preprocessing and CB management
#include "eltwise_utils_common.hpp"
// SFPU-specific preprocessing macros (PREPROCESS, etc.)
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime argument 0: total number of tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time argument 0: number of tiles processed per read-compute-write cycle (always 1)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB indices for input A (base), input B (exponent), and output
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // If LHS/RHS activations are defined (via compile-time defines), use intermediate CBs c_3/c_4.
    // HAS_ACTIVATIONS is a macro that evaluates to 1 if PROCESS_LHS_ACTIVATIONS(i) is non-empty.
    // For POWER with no user-supplied activations, cb_post_lhs == cb_pre_lhs (c_0), cb_post_rhs == cb_pre_rhs (c_1).
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize common compute state: sets up unpack config for cb_post_lhs and pack config for cb_out.
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // If PACK_RELU is defined, configure hardware-level ReLU clamping on the packer output.
    // Not typically used with POWER, but supported if user adds relu post-activation.
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // SFPU init placement strategy:
    // If there are NO activations (LHS, RHS, or POST), initialize the SFPU once before the loop.
    // This calls power_binary_tile_init() which sets programmable constants:
    //   vConstFloatPrgm0 = 1.442695f  (1/ln(2))
    //   vConstFloatPrgm1 = -127.0f    (clamping threshold)
    //   vConstFloatPrgm2 = NaN        (for special-case returns)
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // Main tile processing loop -- one tile per iteration
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS: If LHS activations exist, apply them (copy to intermediate CB, run SFPU op, pack back).
        // For POWER with no activations, this is a no-op (PREPROCESS_0 macro).
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for LHS tile to be available in the post-activation CB
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        // Same for RHS (exponent)
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in output CB for the result tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // If activations changed the SFPU state, re-initialize before the binary op.
        // This handles the case where LHS/RHS preprocessing used different SFPU functions
        // that overwrote the programmable constants.
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire DEST register file -- blocks until DEST is available for writing
        tile_regs_acquire();

        // Copy LHS (base) tile into DEST at slot i*2 (even slots for input A)
        // First configure unpack for the LHS data format
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // copy_tile unpacks tile from CB into DEST register at specified index
            // Slot i*2 = 0 for first tile of cycle (base goes to DEST[0])
            copy_tile(cb_post_lhs, i, i * 2);
        }

        // Copy RHS (exponent) tile into DEST at slot i*2+1 (odd slots for input B)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Slot i*2+1 = 1 for first tile of cycle (exponent goes to DEST[1])
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // If POST activations exist, re-init SFPU constants here (they may have been
            // overwritten by the post-activation SFPU function from the previous iteration)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU binary power operation:
            // BINARY_SFPU_OP expands to power_binary_tile(i*2, i*2+1, i*2)
            // This reads base from DEST[0], exponent from DEST[1], writes result to DEST[0]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Apply post-activations if any (e.g., relu, abs, etc.)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        // Signal that DEST writes are complete -- hand off to packer
        tile_regs_commit();

        // Wait for DEST to be ready for reading (packer side)
        tile_regs_wait();

        // Pack result tile from DEST[i*2] into output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        // Release DEST registers for the next iteration
        tile_regs_release();

        // Push completed output tile to output CB (makes it visible to writer kernel)
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Free consumed input tiles from LHS and RHS CBs
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### Other Compute Kernel Variants

| Variant | File | When Used |
|---|---|---|
| Scalar (one tensor, one scalar) | `eltwise_binary_sfpu_scalar.cpp` | `b` is a scalar, not a tensor |
| Broadcast (COL, SCALAR subtile) | `eltwise_binary_sfpu.cpp` | One input needs subtile-level broadcast |
| Row broadcast | `eltwise_binary_sfpu_row_bcast.cpp` (ng) | ROW_A or ROW_B broadcast with LLK bcast |
| Row-col broadcast | `eltwise_binary_sfpu_row_col_bcast.cpp` (ng) | Mixed row-col broadcast |

All variants share the same fundamental pattern: copy LHS to DEST[even], copy RHS to DEST[odd], call `BINARY_SFPU_OP`, pack result. The differences are in how tiles are re-used across iterations (scalar variant reads RHS once; broadcast variant re-reads the broadcast input).

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
(Wormhole variant at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h` -- identical implementation)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"  // provides _float_to_int32_positive_
#include "ckernel_sfpu_exp.h"          // provides _sfpu_exp_f32_accurate_ (used in fp32 path)
#include "sfpu/ckernel_sfpu_polyval.h" // provides PolynomialEvaluator::eval (used in fp32 path)
#include "sfpi.h"                      // SFPI programming interface (vFloat, vInt, dst_reg, etc.)

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ============================================================================
// BFloat16 / Default Path: _sfpu_binary_power_21f_
// ============================================================================
//
// Implements base**pow using the "exp_21f" algorithm from:
//   Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
//   https://doi.org/10.1109/MSP.2022.3157460
//
// Two-step approach:
//   1) Compute log2(base) using a 3rd-order polynomial approximation
//   2) Compute 2^(pow * log2(base)) using the exp_21f fast exponentiation
//
// This path is selected when is_fp32_dest_acc_en == false (bfloat16 inputs).
// It uses fewer instructions than the fp32 path but has lower precision.
//
// Requires programmable constants set by sfpu_binary_pow_init():
//   vConstFloatPrgm0 = 1.442695f  (1/ln(2), used for log base conversion)
//   vConstFloatPrgm1 = -127.0f    (clamping threshold for overflow prevention)
//   vConstFloatPrgm2 = NaN        (returned for undefined cases like 0^(-1))
//
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base) via polynomial approximation
    //
    // IEEE 754 float = (-1)^s * 2^(exp-127) * 1.mantissa
    // log2(x) = (exp - 127) + log2(1.mantissa)
    // We approximate log2(1.mantissa) with a polynomial over [1, 2)

    // Take absolute value by clearing the sign bit (SFPSETSGN instruction)
    sfpi::vFloat absbase = setsgn(base, 0);
    // Normalize to range [1, 2) by setting exponent to 127 (bias) (SFPSETEXP instruction)
    sfpi::vFloat x = sfpi::setexp(absbase, 127);

    // 3rd order polynomial approximation for log2(x) over [1, 2):
    // Coefficients determined using rminimax (rational minimax) optimization.
    // Horner form: x * (x * (x * c3 + c2) + c1) + c0
    // Hex float literals for exact coefficient representation:
    //   0x2.44734p-4f  ~= 0.14193
    //   0xd.e712ap-4f  ~= 0.86866
    //   0x2.4f5388p+0f ~= 2.30991
    //   0x1.952992p+0f ~= 1.58325
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Extract the integer exponent from the original base (SFPEXEXP instruction)
    sfpi::vInt exp = sfpi::exexp(base);
    // Handle negative exponents: two's complement negation
    // v_if/v_endif are SFPU conditional execution (predicated lanes)
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    // Convert integer exponent to float for addition (SFPCAST instruction)
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);

    // Combine: log2(base) = exponent + polynomial(mantissa) * (1/ln(2))
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // 1/ln(2) = 1.442695
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // Step 2: Compute base**pow = 2^(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;

    // Clamp z_f32 to -127 to prevent overflow when the result should approach 0.
    // Without clamping, the exp_21f algorithm wraps around for very negative inputs.
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0f
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // exp_21f algorithm from Moroz et al. 2022, Section 5:
    // 2^x ~ reinterpret_as_float( (bias + x) * 2^23 )
    // where bias = 0x3f800000 / 2^23 = 1.0 in the integer domain
    //
    // Multiply by 2^23 using SFPDIVP2 (add 23 to exponent), which is a single-cycle
    // instruction with immediate operand -- more efficient than a multiply.
    z_f32 = addexp(z_f32, 23);  // z_f32 *= 2^23 (SFPDIVP2 with imm=23)
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);  // 1.0f reinterpreted as int = 0x3f800000
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    // Decompose z into integer exponent part and fractional mantissa part
    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // exponent bits (SFPEXEXP)
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // mantissa bits (SFPEXMAN)

    // Horner-form correction polynomial for improved accuracy:
    // d1 = 0.40196114e-7 (small correction factor)
    // d2 and d3 incorporate the fractional part (zif) for interpolation
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore the exponent by setting it to 127 + zii (re-biasing)
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Post-processing: handle special cases
    // Convert pow to integer for sign/integer checks
    sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);  // int16 is sufficient (large powers -> 0/inf)
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // Special case: 0^(negative) = NaN (division by zero)
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN
    }
    v_endif;

    // Special case: negative base
    v_if(base < 0.0f) {
        // Odd integer power -> negative result, even integer power -> positive result
        // Shift LSB of pow_int to sign bit position to set result sign
        y = setsgn(y, pow_int << 31);

        // Non-integer power of negative base -> NaN (complex result)
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // When DEST is bfloat16, SFPSTORE will truncate float32 LReg values.
        // Explicit round-to-nearest-even conversion prevents accuracy loss.
        // Example: 9^2 = 80.8 (truncation) vs 81.0 (rounding)
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// ============================================================================
// Float32 Path: _sfpu_binary_power_f32_
// ============================================================================
//
// Higher-precision implementation for float32 inputs.
// Uses improved log2 calculation with range reduction and Newton-Raphson reciprocal,
// plus Cody-Waite + Taylor expansion for exp.
//
sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // Step 1: Compute log2(base) using improved log with range reduction

    sfpi::vFloat abs_base = sfpi::abs(base);
    // Normalize mantissa to [1, 2)
    sfpi::vFloat m = sfpi::setexp(abs_base, 127);
    sfpi::vInt exp = sfpi::exexp(abs_base);

    // Range reduction: ensure m in [sqrt(2)/2, sqrt(2)] for better polynomial convergence
    constexpr float SQRT2 = 1.4142135381698608f;
    v_if(m >= SQRT2) {
        m = m * 0.5f;  // SFPMUL: divide by 2
        exp = exp + 1;  // compensate exponent
    }
    v_endif;

    // Transform to z = (m-1)/(m+1) for logarithm series
    // This transformation converges faster than direct series on [1,2]
    sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
    sfpi::vFloat m_minus_1 = m - sfpi::vConst1;

    // Compute 1/(m+1) using Newton-Raphson with linear initial guess
    // Initial guess: 1.0 - 0.2426...*t (linear interpolation on [1.7, 2.4])
    sfpi::vFloat recip = sfpi::vConst1 - 0.2426406871192851f * m_plus_1;
    recip = recip * (2.0f - m_plus_1 * recip);  // 1st Newton-Raphson iteration
    recip = recip * (2.0f - m_plus_1 * recip);  // 2nd NR iteration for float32 precision
    sfpi::vFloat z = m_minus_1 * recip;

    // Polynomial approximation of atanh(z) using odd powers via Horner's method
    // ln(m) = 2 * atanh(z) = 2 * z * (1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
    sfpi::vFloat z2 = z * z;
    sfpi::vFloat p = PolynomialEvaluator::eval(
        z2, sfpi::vConst1,
        0.3333333333333333f,   // 1/3
        0.2f,                   // 1/5
        0.14285714285714285f,   // 1/7
        0.1111111111111111f,    // 1/9
        0.09090909090909091f);  // 1/11
    sfpi::vFloat ln_m = 2.0f * (z * p);

    // Convert integer exponent to float, handling sign manually for SFPU compatibility
    sfpi::vInt sign_bit = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(exp) >> 31);
    sfpi::vInt exp_sign = sfpi::vInt(0) - sign_bit;     // 0 or 0xFFFFFFFF
    sfpi::vInt exp_abs = (exp ^ exp_sign) - exp_sign;   // absolute value via XOR trick
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(sfpi::setsgn(exp_abs, exp_sign), 0);

    // log2(base) = exponent + ln(mantissa) / ln(2)
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;  // 1/ln(2)
    sfpi::vFloat log2_result = exp_f32 + ln_m * vConst1Ln2;

    // Step 2: base**pow = 2^(pow * log2(base))
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0f
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // Use Cody-Waite range reduction + Taylor expansion for accurate exp
    // This achieves <1 ULP error for float32
    constexpr float LN2 = 0.693147180559945309f;
    sfpi::vFloat y = _sfpu_exp_f32_accurate_(z_f32 * LN2);

    // Same special-case handling as the 21f path
    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // NaN
    }
    v_endif;

    v_if(base < 0.0f) {
        sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
        sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);
        y = sfpi::setsgn(y, pow_int << 31);
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;  // NaN for non-integer power of negative base
        }
        v_endif;
    }
    v_endif;

    return y;
}

// ============================================================================
// Template Dispatch: _sfpu_binary_power_
// ============================================================================
// Selects between 21f (bfloat16) and f32 paths based on dest accumulator mode.

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);  // bfloat16 path
}

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_f32_(base, pow);  // float32 path
}

// ============================================================================
// Entry Point: calculate_sfpu_binary_pow
// ============================================================================
// Called from the LLK layer (_llk_math_eltwise_binary_sfpu_params_).
// Iterates over all 8 rows within a face (ITERATIONS=8) of the DEST register.
//
// The outer LLK params function (_llk_math_eltwise_binary_sfpu_params_) handles
// face iteration (4 faces for RC mode), calling this function once per face.
// Each face has 8 rows of 4 elements each (32 elements per face, 128 per tile).
// Total: 4 faces * 8 iterations = 32 SFPU passes over a tile.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // DEST tile size in SFPI units: 64 rows / SFP_DESTREG_STRIDE = 32 rows per tile
        // (SFPU processes 4 elements per row, so 32 rows * 4 = 128 elements = 4 faces * 32)
        constexpr uint dst_tile_size_sfpi = 32;

        // Load base and exponent vectors from their respective DEST tile slots
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // SFPLOAD
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // SFPLOAD

        // Compute power using the appropriate precision path
        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        // Store result back to output DEST slot
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;  // SFPSTORE
        // Advance the DEST register pointer to the next row
        sfpi::dst_reg++;  // SETRWC to advance by 1 row
    }
}

// ============================================================================
// Initialization: sfpu_binary_pow_init
// ============================================================================
// Sets the three programmable SFPU constants used throughout the power computation.
// Called once at the start of computation (or re-called when activations overwrite them).

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;                          // 1/ln(2) for log base conversion
    sfpi::vConstFloatPrgm1 = -127.0f;                            // clamping threshold
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();  // NaN for special cases
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| SFPI Intrinsic / Instruction | Underlying SFPU Instruction | Description |
|---|---|---|
| `sfpi::dst_reg[idx]` (read) | `SFPLOAD` | Load 4-element vector from DEST register row |
| `sfpi::dst_reg[idx]` (write) | `SFPSTORE` | Store 4-element vector to DEST register row |
| `sfpi::dst_reg++` | `SETRWC` | Advance DEST register read/write pointer by 1 row |
| `sfpi::setsgn(v, s)` | `SFPSETSGN` | Set sign bit of float vector |
| `sfpi::setexp(v, e)` | `SFPSETEXP` | Set exponent field of float vector |
| `sfpi::exexp(v)` | `SFPEXEXP` | Extract exponent from float as integer |
| `sfpi::exman9(v)` | `SFPEXMAN` | Extract 9-bit mantissa from float |
| `sfpi::abs(v)` | `SFPABS` | Absolute value |
| `sfpi::addexp(v, n)` | `SFPDIVP2` | Add immediate to exponent (multiply by 2^n) |
| `sfpi::int32_to_float(v, 0)` | `SFPCAST` | Convert int32 to float |
| `sfpi::float_to_int16(v, 0)` | `SFPCAST` | Convert float to int16 (truncation) |
| `sfpi::float_to_fp16b(v, 0)` | `SFPCAST` | Convert float to bfloat16 (round-to-nearest-even) |
| `sfpi::reinterpret<T>(v)` | `SFPNOP` (reinterpret) | Bitwise reinterpret between vFloat/vInt/vUInt |
| `v_if(...) { } v_endif;` | `SFPSETCC` / `SFPENCC` | Predicated execution (conditional per-lane) |
| `operator*`, `operator+`, `operator-` | `SFPMAD` / `SFPMUL` | Fused multiply-add or standalone multiply |
| `operator<<` (vInt) | `SFPSHFT` | Bitwise left shift |
| `operator~` (vInt) | `SFPNOT` | Bitwise NOT |
| `operator^` (vInt) | `SFPXOR` | Bitwise XOR |
| `_float_to_int32_positive_` | Multiple (SFPEXEXP, SFPSETMAN, SFPSHFT, SFPSETCC) | Custom float-to-positive-int32 conversion |
| `_sfpu_exp_f32_accurate_` | Multiple (SFPMAD, SFPDIVP2, SFPCAST, etc.) | Cody-Waite + Taylor exp for fp32 path |
| `PolynomialEvaluator::eval` | Multiple SFPMAD | Horner-form polynomial evaluation |

#### SFPU Register Usage

**DEST Registers (DST)**:
- `dst_index_in0 * 32` (base): Input A tile occupies even-indexed DEST slot. For `num_tiles_per_cycle=1`, this is DEST slot 0.
- `dst_index_in1 * 32` (exponent): Input B tile occupies odd-indexed DEST slot. For `num_tiles_per_cycle=1`, this is DEST slot 1.
- `dst_index_out * 32` (result): Output overwrites the base slot (slot 0), so the operation is in-place w.r.t. the base tile.

**SFPU Local Registers (LRegs)**:
- The SFPU has 8 local registers (LRegs), each holding a 4-element vector.
- The power computation is register-intensive, using all available LRegs for intermediate values (absbase, x, series_result, exp, exp_f32, log2_result, z_f32, z, zii, zif, d1, d2, d3, y, pow_int, pow_rounded).
- The compiler (SFPI) manages register allocation automatically.

**Programmable Constants**:
- `vConstFloatPrgm0`: 1.442695f (1/ln(2))
- `vConstFloatPrgm1`: -127.0f (overflow clamp)
- `vConstFloatPrgm2`: NaN (special case return)
- `vConst1`: 1.0f (built-in constant, used in fp32 path)

#### SFPU Execution Flow

1. **Initialization** (`sfpu_binary_pow_init`): Sets three programmable constants in the SFPU constant registers. This is called once per tile batch (or re-called if activations overwrite them).

2. **LLK Layer Dispatch** (`_llk_math_eltwise_binary_sfpu_params_`):
   - Asserts that DEST indices are valid
   - Calls `_llk_math_eltwise_binary_sfpu_start_` to configure SFPU for operation
   - In RC mode (default), loops over 4 faces of the 32x32 tile
   - For each face, calls `calculate_sfpu_binary_pow` which processes 8 rows
   - After each face, advances the DEST read/write pointer by 2 SETRWC steps (8 rows each)
   - Calls `_llk_math_eltwise_binary_sfpu_done_` to finalize

3. **Per-Row Processing** (`calculate_sfpu_binary_pow`):
   - Iterates 8 times per face (ITERATIONS=8)
   - Each iteration processes one row of 4 elements:
     a. `SFPLOAD` base vector from DEST[in0 * 32]
     b. `SFPLOAD` exponent vector from DEST[in1 * 32]
     c. Call `_sfpu_binary_power_<is_fp32_dest_acc_en>` to compute power
     d. `SFPSTORE` result to DEST[out * 32]
     e. `SETRWC` (dst_reg++) to advance to next row

4. **Power Computation** (bf16 path -- `_sfpu_binary_power_21f_`):
   a. Compute `|base|` and normalize mantissa to [1, 2)
   b. Evaluate 3rd-order polynomial for log2 of the mantissa
   c. Extract and convert the integer exponent
   d. Combine to get `log2(base)`
   e. Compute `z = pow * log2(base)`, clamp to [-127, ...]
   f. Apply exp_21f: multiply by 2^23, add bias, decompose, polynomial correction
   g. Handle special cases (0^negative -> NaN, negative base handling)
   h. Convert result to bfloat16 with round-to-nearest-even

5. **Packing**: After the SFPU completes all faces, DEST contents are packed back to the output CB via `pack_tile`.

#### SFPU Configuration

**Compile-Time Defines** (set by `OpConfig::as_defines()`):
- `BINARY_SFPU_INIT` = `power_binary_tile_init();`
- `BINARY_SFPU_OP` = `power_binary_tile`
- `BCAST_INPUT` = `""` (empty for no-bcast, `"0"` or `"1"` for bcast variants)

**Template Parameters**:
- `APPROX`: Set based on `fast_and_approximate_mode` -- currently unused by the power kernel (both paths use the same algorithm regardless of APPROX)
- `DST_ACCUM_MODE`: Controls `is_fp32_dest_acc_en` template parameter, which selects between the 21f and f32 algorithm paths

**Math Fidelity**: Not directly applicable to SFPU operations (math fidelity controls the FPU matrix engine, not the SFPU vector unit). The precision is controlled by `is_fp32_dest_acc_en`.

**UnpackToDestMode**: For POWER specifically:
- FLOAT32 inputs: `UnpackToDestFp32` -- tiles are unpacked to 32-bit float in DEST
- Other inputs (bfloat16): `Default` -- tiles remain in their native 16-bit format in DEST

#### Hardware Compatibility Notes

The Blackhole and Wormhole implementations are **identical** for the binary power kernel. Both `ckernel_sfpu_binary_pow.h` files contain the same code. This is because:

1. The SFPI programming interface abstracts away hardware differences between the two architectures
2. The power kernel uses only standard SFPI intrinsics (no architecture-specific instructions)
3. The polynomial coefficients and algorithm structure are the same

The only difference is in the LLK infrastructure layer (`_llk_math_eltwise_binary_sfpu_params_`), which has minor differences in face iteration and SETRWC patterns between architectures, but the functional behavior is identical.

### Dataflow Kernels

#### Reader Kernel (Two Tensor Inputs)

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

This reader handles the case where both base and exponent are full tensors. It reads tiles from both input tensors using the TensorAccessor API, handling multi-dimensional broadcasting and sharded layouts. Key aspects:

- Reads tiles one at a time from DRAM/L1 using `noc_async_read_page`
- Supports both interleaved and sharded memory layouts (compile-time `SRC_SHARDED`/`SRC_SHARDED_B` defines)
- For sharded inputs, simply makes existing shard visible via `cb_reserve_back`/`cb_push_back`
- For interleaved inputs, iterates through the multi-dimensional tile space (nD, D, N, C, Ht, Wt) with stride calculations for broadcasting
- Runtime args include per-dimension strides that encode broadcasting (stride=0 means broadcast along that dimension)

#### Writer Kernel (Scalar Variant)

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`

Used when one operand is a scalar. The writer:
1. Fills a single tile in CB c_1 with the scalar value (using `fill_with_val` helpers)
2. Writes computed output tiles from CB c_2 to DRAM using `noc_async_write_page`
3. Iterates through the output tile space with the same multi-dimensional loop structure

#### Writer Kernel (Two Tensor Inputs)

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`

Simpler writer that only writes output tiles from CB c_2 to DRAM. The reader handles both input tiles.

## Work Distribution

The program factory distributes output tiles across available cores using `split_work_to_cores`:

1. **Total tiles**: `c_num_tiles = c.physical_volume() / tile_hw`
2. **Core splitting**: Creates two core groups -- group 1 gets `num_tiles_per_core_group_1` tiles, group 2 gets `num_tiles_per_core_group_2` tiles (handles uneven division)
3. **Per-core runtime args**: Each core receives `c_num_tiles` (its tile count) and `c_start_id` (its starting tile offset)
4. **Sharded mode**: When inputs/outputs are sharded, each core processes its local shard directly from L1, avoiding DRAM reads

Compute runtime args per core: `{c_num_tiles, freq, counter, compute_scalar_value}`
- `freq` and `counter` are for broadcast iteration patterns (both 1/0 for NONE broadcast)
- `compute_scalar_value` is 0 for POWER (no quantization)

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Binary_ng program factory structure, kernel path resolution, OpConfig mapping for POWER, UnpackToDestMode handling
- `tenstorrent/tt-metal`: SFPU power operation -- algorithm details (21f vs f32 paths), special case handling, ckernel_sfpu_binary_pow.h location

### Confluence References
Not consulted for this analysis -- DeepWiki provided sufficient detail on the SFPU instruction set intrinsics used by the power kernel.

### Glean References
Not consulted for this analysis -- the binary power kernel implementation is fully available in the open-source codebase.

## Key Design Decisions

1. **Two-path precision strategy**: The POWER operation provides distinct algorithm implementations for bfloat16 (21f) and float32 inputs, rather than always using the higher-precision path. This is because the 21f path uses fewer SFPU instructions (faster), and bfloat16 output cannot benefit from float32-level intermediate precision anyway. The conditional `UnpackToDestMode` reinforces this -- it avoids wasting DEST capacity on fp32 when the 21f path is sufficient.

2. **exp_21f vs Cody-Waite+Taylor**: The bfloat16 path uses the Moroz et al. exp_21f algorithm which exploits IEEE 754 bit manipulation for fast 2^x computation. The float32 path uses the more accurate (but slower) Cody-Waite range reduction with Taylor series expansion, achieving <1 ULP error.

3. **Log2 implementation difference**: The bfloat16 path uses a simple 3rd-order polynomial for log2 over [1,2). The float32 path uses range reduction to [sqrt(2)/2, sqrt(2)] with an atanh-based series (6 terms) and Newton-Raphson reciprocal, providing much higher accuracy.

4. **Programmable constants**: Using `vConstFloatPrgm0/1/2` for frequently-used constants (1/ln(2), -127, NaN) avoids repeated `SFPLOADI` instructions, saving instruction slots and cycles.

5. **In-place DEST operation**: The result overwrites the base tile slot (`odst = idst0`), which avoids needing a third DEST slot and is natural since the base is consumed after computation.

6. **Shared kernel binary**: All binary SFPU operations share the same `.cpp` compute kernel file. The specific operation is selected entirely through compile-time defines (`BINARY_SFPU_INIT`, `BINARY_SFPU_OP`), which means the kernel binary is specialized per operation at compile time with zero runtime dispatch overhead.
