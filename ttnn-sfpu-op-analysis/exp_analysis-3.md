# TTNN SFPU Operation Analysis: Exp

## Overview

| Property | Value |
|---|---|
| **Operation Name** | `exp` (element-wise exponential) |
| **Operation Type** | Unary SFPU |
| **UnaryOpType Enum** | `UnaryOpType::EXP` |
| **Program Factory** | `UnaryProgramFactory` (also `UnarySubCoreGridProgramFactory` for sub-core grids) |
| **Program Factory File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` |
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Reader Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Writer Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **SFPU Kernel (tt-metal)** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` |
| **SFPU Kernel (tt-llk)** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h` |
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` |
| **Math Fidelity** | `HiFi4` |
| **Math Approx Mode** | `false` (from `get_op_approx_mode`; `fast_and_approx` controlled by param0) |

---

## Program Factory Analysis

### Factory Selection

The EXP operation uses the shared `UnaryProgramFactory::create` (or `UnarySubCoreGridProgramFactory::create` for sub-core grid configurations). It is not a dedicated single-op factory -- it handles all unary SFPU operations via compile-time defines that select the specific SFPU function.

### Work Distribution

```
split_work_to_cores(compute_with_storage_grid_size, num_pages)
  -> core_group_1: gets num_pages_per_core_group_1 tiles each
  -> core_group_2: gets num_pages_per_core_group_2 tiles each (remainder distribution)
```

Tiles are distributed across all available compute cores. Two core groups handle the uneven division of tiles -- group 1 gets one more tile per core than group 2 (or they are equal if tiles divide evenly).

### Circular Buffer Configuration

| CB Index | Name | Size (pages) | Data Format | Purpose |
|---|---|---|---|---|
| `c_0` | Input CB | 2 | Input tensor format | Double-buffered input tiles from reader |
| `c_1` | Temp CB | 2 | Input tensor format | **Not used for EXP** (only HARDSHRINK, CBRT, LOGIT) |
| `c_2` | Output CB | 2 | Output tensor format | Double-buffered output tiles to writer |

The double-buffering (2 pages) allows the reader to fill one page while compute processes the other, enabling pipeline overlap.

### Compile-Time Defines

For EXP with param0 = 1 (the default `fast_and_approx = true`), the following defines are generated:

| Define | Value | Source |
|---|---|---|
| `SFPU_OP_EXP_INCLUDE` | `1` | `get_macro_definition(EXP)` -- gates `#include "api/compute/eltwise_unary/exp.h"` |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | `get_block_defines` -- concatenates init and func calls |
| `SFPU_OP_CHAIN_0_INIT_0` | `exp_tile_init<1u>();` | From `get_defines_impl` for EXP with param0=1 |
| `SFPU_OP_CHAIN_0_FUNC_0` | `exp_tile<1u>(0);` | From `get_defines_impl` for EXP with param0=1 |
| `INP_FLOAT` or `INP_FLOAT32` etc. | `1` | Based on input dtype |

The template parameter `1u` corresponds to `fast_and_approx = true`. When param0 = 0 (precise mode), the defines become `exp_tile_init<0u>()` and `exp_tile<0u>(0)`.

### ComputeConfig

| Field | Value | Rationale |
|---|---|---|
| `math_fidelity` | `MathFidelity::HiFi4` | Hardcoded for all unary ops in this factory |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | Controls whether DEST accumulator is FP32 |
| `math_approx_mode` | `false` | `get_op_approx_mode(EXP)` returns false for all ops (default case) |
| `bfp8_pack_precise` | From `args.bfp8_pack_precise` | For BFP8 output precision |
| `unpack_to_dest_mode` | Default or `UnpackToDestFp32` | FP32 mode if `preserve_fp32_precision` is set |

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start tile ID) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start tile ID) |
| Compute | `packed_scalar1` (0 for EXP) | `packed_scalar2` (0 for EXP) | -- |

EXP does not use packed scalar runtime arguments (they are both 0). The scalars are only used for ops like HARDSHRINK, WHERE_TSS, and LOGIT.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

Simple page-at-a-time reader. Iterates from `start_id` to `start_id + num_pages`, reading each page from DRAM via NoC async read into CB `c_0`. Uses `TensorAccessor` for address computation (supports interleaved layout). The `cb_reserve_back` / `cb_push_back` pattern produces pages for the compute kernel.

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Mirrors the reader. Consumes pages from CB `c_2` via `cb_wait_front` / `cb_pop_front` and writes them to DRAM via NoC async write. Supports sharded output (`OUT_SHARDED` define) and backwards iteration (`BACKWARDS` define).

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the generic SFPU compute kernel shared by all unary SFPU operations. The specific SFPU function is injected via the `SFPU_OP_CHAIN_0` preprocessor define.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"              // Common compute API (cb_wait_front, cb_pop_front, etc.)
#include "api/compute/tile_move_copy.h"       // copy_tile: moves tile from CB to DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"  // init_sfpu, tile_regs_acquire/commit/wait/release, pack_tile
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes exp.h when SFPU_OP_EXP_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    // Initialize SFPU pipeline: configures unpack (CB c_0 -> SRC), math (SFPU), and pack (DST -> CB c_2)
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for one block of tiles before processing
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST register file (prevents pack from reading while math writes)
            tile_regs_acquire();

            // Wait for reader kernel to produce one tile in input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Unpack tile from CB c_0 slot 0 into DST register index 0
            // This triggers the unpacker RISC-V to move data from L1 (CB) into SRC registers,
            // then the FPU copies from SRC to DST
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // --- SFPU_OP_CHAIN_0 expands to: ---
            // For EXP with fast_and_approx=true (param0=1):
            //   exp_tile_init<1u>();  -- initializes SFPU constants and macro instructions for fast exp
            //   exp_tile<1u>(0);      -- computes exp() on tile at DST index 0
            //
            // For EXP with fast_and_approx=false (param0=0):
            //   exp_tile_init<0u>();  -- initializes reciprocal tables for precise exp
            //   exp_tile<0u>(0);      -- computes exp() on tile at DST index 0 using Horner series + squaring
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are written and ready for packing
            tile_regs_commit();

            // Wait for pack engine to be ready to read from DST
            tile_regs_wait();

            // Pack tile from DST index 0 into output CB c_2
            pack_tile(0, tt::CBIndex::c_2);

            // Release input tile from CB c_0 (consumer side: frees space for reader to write more)
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers for next iteration
            tile_regs_release();
        }
        // Push the completed block to the output CB (makes it visible to writer kernel)
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a deep dive into the SFPU kernel functions that implement the exponential computation. The exp operation has multiple implementation paths depending on template parameters.

#### API Layer

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`

This header defines the user-facing `exp_tile_init` and `exp_tile` functions. These are thin wrappers that dispatch to the underlying `ckernel::sfpu` functions via the LLK macro infrastructure.

```cpp
// exp_tile_init: Initializes SFPU state for exponential computation.
// Template parameters control which algorithm variant is used.
template <
    bool approx = false,           // Enable approximation mode
    bool fast_and_approx = true,   // If approx, use fast LOADMACRO-based path
    uint32_t scale = 0x3F800000,   // Scale factor (1.0f by default)
    InputClamping input_clamping = InputClamping::ClampToNegative>  // Clamp inputs < -88.5
ALWI void exp_tile_init() {
    // MATH macro ensures this runs only on the math RISC-V (TRISC_MATH)
    // Dispatches to sfpu::exp_init<approx, fast_and_approx, scale, clamp_negative>()
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential,
        sfpu::exp_init,
        approx, fast_and_approx, scale,
        (input_clamping == InputClamping::ClampToNegative)));
}

// exp_tile: Computes exp() on a tile in the DST register.
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8>           // 8 iterations = 8 SFPU lanes x 4 rows = 32x32 tile face
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    // Dispatches to sfpu::calculate_exponential<approx, fast_and_approx, DST_ACCUM_MODE, ...>()
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(
        calculate_exponential,
        approx, fast_and_approx, DST_ACCUM_MODE,
        scale_en, skip_positive_check,
        (input_clamping == InputClamping::ClampToNegative),
        iterations, idst, vector_mode, scale));
}
```

The `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macro expands to a call to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` which handles DST register indexing and then calls `ckernel::sfpu::calculate_exponential<...>()`.

#### Architecture-Specific Dispatch Layer (tt-metal)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` (identical structure for Blackhole)

This layer bridges the API to the shared LLK implementation. For the non-approximation path (used by the standard EXP via `UnaryProgramFactory`), it provides the `_sfpu_exp_improved_` specializations that select between `_sfpu_exp_21f_` (BF16 dest) and `_sfpu_exp_f32_accurate_` (FP32 dest).

##### Annotated Source (tt-metal layer, Wormhole)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // Shared LLK exp implementation (_calculate_exponential_, _init_exponential_)
#include "sfpu/ckernel_sfpu_polyval.h"    // PolynomialEvaluator::eval (Horner-form polynomial evaluation)
#include "sfpu/ckernel_sfpu_converter.h"  // Converter::as_float (bit-cast utilities)
#include "ckernel_sfpu_conversions.h"     // float_to_fp16b conversion
#include "sfpi.h"                         // SFPI programming interface (vFloat, vInt, dst_reg, etc.)

namespace ckernel {
namespace sfpu {

// Wrapper that calls the shared _sfpu_exp_ (Horner series + repeated squaring)
sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

// Branch-free float-to-int32 optimized for exp21f algorithm.
// Constraint: 0 <= val < 128.0f. Assumes val has been pre-divided by 2^23.
// Returns integer value scaled by 2^23 (the scaling is implicit from the IEEE 754 representation).
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);      // Extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);      // Extract mantissa with implicit leading 1 (8-bit mantissa)
    // Shift mantissa left by exponent amount -- this converts the float to a fixed-point integer
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

// ============================================================================
// exp_21f: Moroz et al. 2022 algorithm (2-ULP accuracy for BF16)
// Used when is_fp32_dest_acc_en = false
// ============================================================================
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // Compute exp(x) = 2^(x/ln2) by:
    // 1. Scale x by 1/ln(2) and add bias 127 to prepare IEEE 754 exponent encoding
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // 2. Clamp to [0, 255] to prevent overflow/underflow in intermediate computations
    //    (corresponds to exp input range approximately [-88.5, 88.5])
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);   // xlog2 = max(0, xlog2)
    sfpi::vec_min_max(xlog2, threshold_high);   // xlog2 = min(255, xlog2)

    // 3. Convert to integer (implicitly scaled by 2^23)
    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    // 4. Split into integer exponent and fractional mantissa parts
    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // 2^(integer part)
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));      // fractional [0, 1)

    // 5. Convert fractional part to float for polynomial evaluation
    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);

    // 6. Polynomial approximation of 2^frac on [0, 2^23] (2nd degree, Moroz coefficients)
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // 7. Recombine: result = 2^(integer) * 2^(fractional) by setting exponent bits
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // When DST is BF16, explicitly round to BF16 using round-to-nearest-even
        // to avoid truncation artifacts from SFPSTORE
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// ============================================================================
// exp_61f: Moroz et al. 2022 algorithm (higher accuracy, 6th degree polynomial)
// Not used by default EXP path but available for specialized use
// ============================================================================
sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    // Same clamping as exp_21f
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    frac = sfpi::addexp(frac, -23);  // Scale down by 2^-23 to get fractional in [0, 1]

    // 6th degree polynomial approximation of 2^x on [0, 1]
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

// ============================================================================
// Cody-Waite accurate exp: < 1 ULP accuracy for FP32
// Used when is_fp32_dest_acc_en = true
// ============================================================================
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;    // ~89 in original x domain
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;  // ~-88 in original x domain

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;

    sfpi::vInt exp_bits = sfpi::exexp(z);

    // Handle special cases: overflow -> inf, underflow -> 0, NaN -> NaN
    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_else {
        // Cody-Waite range reduction: split ln(2) into high and low parts
        // to maintain precision when computing r = x - k*ln(2)
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Negated constants to enable single SFPMAD instruction optimization
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;   // Compiles to single SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi;     // Compiles to single SFPMAD

        // 7th order Taylor series for exp(r) where |r| < ln(2)/2
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,   // 1
            sfpi::vConst1,   // 1
            0.5f,            // 1/2!
            1.0f / 6.0f,     // 1/3!
            1.0f / 24.0f,    // 1/4!
            1.0f / 120.0f,   // 1/5!
            1.0f / 720.0f,   // 1/6!
            1.0f / 5040.0f   // 1/7!
        );

        // Scale by 2^k via exponent manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

// Template specializations that select the algorithm based on FP32 dest mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);  // BF16 dest: use Moroz exp_21f
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // FP32 dest: use Cody-Waite accurate path
}

// ============================================================================
// Top-level dispatch: called from exp_tile via SFPU_TEMPLATE_PARAMS_KERNEL_FN
// ============================================================================
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
        // Delegates to shared LLK _calculate_exponential_ which has
        // FAST_APPROX (LOADMACRO), piecewise, and standard approximation paths
        _calculate_exponential_<
            APPROXIMATION_MODE, SCALE_EN, ITERATIONS,
            FAST_APPROX, SKIP_POSITIVE_CHECK, CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximation: iterate over SFPU lanes, apply improved exp to each
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;  // Advance to next SFPU lane (32 elements)
        }
    }
}

// Initialization dispatcher
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Shared LLK Layer (tt-llk)

**File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`

This is the shared implementation used by both Wormhole and Blackhole. It contains the traditional SFPU exp algorithm (Horner series + repeated squaring), the approximation paths, and the fast LOADMACRO-based path.

##### Annotated Source (shared LLK layer)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

// ============================================================================
// Traditional SFPU exp: Horner series + repeated squaring
// This is the "precise" non-approximation fallback used when APPROXIMATION_MODE = false
// ============================================================================
sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // Step 1: If exponent >= 0 (i.e., |val| >= 1.0), extract the exponent
    // and replace it with -1 (i.e., scale val into [-1, 1) range).
    // The extracted exponent tells us how many times to square the result later.
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);  // Set exponent to -1 (biased 126), keeping sign and mantissa
    }
    v_endif;

    // Step 2: Compute exp(val) for val in [-1, 1) using 2nd order Horner approximation
    // exp(x) ~ (x * 0.8373 + 0.863281) * x + 1
    // This is a rough approximation valid only for small |x|
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    // Step 3: If original exponent was >= 0, square the result repeatedly
    // Since we reduced to [-1, 1), we need to compute exp(2^n * x) = exp(x)^(2^n)
    // by squaring n times where n = original exponent
    v_if (exp >= 0)
    {
        val = val * val;  // First squaring (always needed when exp >= 0)
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0);   // Narrow predication: only lanes where exp still >= 0 continue
            val = val * val;   // Square again
        }
    }
    v_endif;

    return val;
}

// ============================================================================
// Approximation body: two paths (approx via bit manipulation, precise via Horner+reciprocal)
// ============================================================================
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // Fast bit-manipulation approximation:
        // exp(x) ~ reinterpret_as_float((int)(x / ln2) << shift + bias)
        // This exploits the IEEE 754 encoding where the integer part maps to the exponent bits
        constexpr int FRAC_BITS         = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;  // 1016

        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;  // 1/ln(2) loaded during init
        sfpi::vFloat conv           = in * vConstLn2Recip;

        sfpi::vInt c23_73 = p_exp::C23_73;  // Conversion constant for FP to fixed-point
        sfpi::vInt tmp    = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73;

        tmp += SP_BIAS;  // Add IEEE 754 exponent bias

        // Shift left to position integer bits as the exponent field of the result float
        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Precise path: compute exp(|x|) using Horner series, then take reciprocal for negative x
        out = _sfpu_exp_(sfpi::setsgn(in, 0));  // exp(|x|)

        v_if (in < 0)
        {
            out = _sfpu_reciprocal_<2>(out);  // 1/exp(|x|) = exp(-|x|) = exp(x) for x < 0
        }
        v_endif;
    }

    return out;
}

// ============================================================================
// Piecewise approximation with boundary checks
// ============================================================================
template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{
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
                result = std::numeric_limits<float>::infinity();  // Overflow saturation
            }
            v_elseif (in < -42)
            {
                result = 0.0f;  // Underflow saturation
            }
            v_else
            {
                result = _calculate_exponential_approx_(in);
            }
            v_endif;
        }
        else
        {
            // Skip positive check -- caller guarantees inputs <= 89
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
        // Non-approximation: Horner + reciprocal
        result = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if (in < 0)
        {
            result = _sfpu_reciprocal_<2>(result);
        }
        v_endif;
    }

    return result;
}

// ============================================================================
// Main dispatch function: _calculate_exponential_
// This is the function called by SFPU_TEMPLATE_PARAMS_KERNEL_FN for APPROXIMATION_MODE=true
// ============================================================================
template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // === LOADMACRO-based fast path with input clamping ===
        // Phase 1: Sanitize inputs -- clamp values below -88.5 to -88.5
        // Uses Macro Sequence Register 1 (LD, SWAP, STORE) to compare each DEST element
        // against -88.5 (stored in LREG[14]) and keep the larger value.
        // 8 SFPLOADMACRO calls cover all 8 dest offsets (0,2,4,6,8,10,12,14)
        // corresponding to even/odd columns across 4 row groups of a 32x32 tile face.
        TTI_SFPLOADMACRO(4, 0, 3, 0);   // LREG[0], dest offset 0 (even cols, rows 0-3)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, 3, 2);   // LREG[1], dest offset 2 (odd cols, rows 0-3)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 4);   // LREG[2], dest offset 4 (even cols, rows 4-7)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 6);   // LREG[3], dest offset 6 (odd cols, rows 4-7)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, 3, 8);   // LREG[0], dest offset 8 (even cols, rows 8-11)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, 3, 10);  // LREG[1], dest offset 10 (odd cols, rows 8-11)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 12);  // LREG[2], dest offset 12 (even cols, rows 12-15)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 14);  // LREG[3], dest offset 14 (odd cols, rows 12-15)

        // Phase 2: Compute exp() on sanitized values
        // Uses Macro Sequence Register 0 (LD, MAD, ROUND, SHIFT, STORE)
        // MAD: i = A * y + (B-C) where A = 256/ln(2), B-C = 32500.818
        // ROUND: convert FP32 to UINT16
        // SHIFT: shift left by 15 to place in IEEE 754 exponent position
        TTI_SFPLOADMACRO(0, 0, 3, 0);
        TTI_SFPLOADMACRO(1, 0, 3, 2);
        TTI_SFPLOADMACRO(2, 0, 3, 4);
        TTI_SFPLOADMACRO(3, 0, 3, 6);
        TTI_SFPLOADMACRO(0, 0, 3, 8);
        TTI_SFPLOADMACRO(1, 0, 3, 10);
        TTI_SFPLOADMACRO(2, 0, 3, 12);
        TTI_SFPLOADMACRO(3, 0, 3, 14);
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8)
    {
        // === Replay-buffer based fast path (no clamping), 8 elements ===
        // ~20 cycles for 8 elements = 2.5 cycles/element
        // Uses recorded LOADMACRO+SHFT2 pairs replayed from instruction replay buffer
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},  // Auto-increment dest by 2 per LOADMACRO
        }.set(ADDR_MOD_7);

        lltt::replay(0, 16);  // Replay 16 recorded instructions

        // Drain pipeline: process final 2 SHFT2 operations
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 32)
    {
        // === Replay-buffer based fast path (no clamping), 32 elements ===
        // ~68 cycles for 32 elements = 2.125 cycles/element
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }.set(ADDR_MOD_7);

        lltt::replay(0, 32);
        lltt::replay(0, 32);

        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else
    {
        // === Software loop path (piecewise or precise) ===
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vFloat val    = sfpi::dst_reg[0];
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(val, exp_base_scale_factor);
            sfpi::dst_reg[0]    = result;
            sfpi::dst_reg++;
        }
    }
}

// ============================================================================
// Initialization function: _init_exponential_
// Sets up SFPU constants, macro instructions, and replay buffers
// ============================================================================
template <bool APPROXIMATION_MODE, bool FAST_APPROX, std::uint32_t scale, bool CLAMP_NEGATIVE = true>
inline void _init_exponential_()
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // Schraudolph fast exp with input clamping
        // Algorithm: exp(x) ~ reinterpret_as_float(int(A * x + (B-C))) where:
        //   A = 256/ln(2) = 369.33 (scale factor to convert to 8.3 fixed-point)
        //   B = 127 * 256 = 32512 (IEEE 754 bias in 8-bit format)
        //   C ~ 11.18 (error minimization adjustment)
        //   B-C = 32500.818

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A         = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;
        constexpr float THRESHOLD = -88.5f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);
        constexpr float A_scaled         = A * scale_fp32;
        constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

        // Load constants into special LREG registers (12, 13, 14)
        // LREG[14] = threshold (-88.5) for input sanitization
        TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));
        TTI_SFPCONFIG(0, 14, 0);

        // LREG[12] = A (scale factor 369.33)
        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);

        // LREG[13] = B-C (bias offset 32500.818)
        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);

        // Program macro instruction 0: SWAP (for input sanitization)
        TTI_SFPLOADI(0, 0xA, 0x00E1);
        TTI_SFPLOADI(0, 0x8, 0x9200);
        TTI_SFPCONFIG(0, 0, 0);  // Install as Macro Instruction 4
        TTI_SFPNOP;

        // Program macro instruction 1: MAD (A * y + (B-C))
        TTI_SFPMAD(12, 0, 13, 13, 0);  // Backdoor load to Macro Instruction 5

        // Program macro instruction 2: STOCHRND (FP32 -> UINT16 rounding)
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14);  // Backdoor load to Macro Instruction 6

        // Program macro instruction 3: SHIFT (left shift by 15 to position exponent bits)
        TTI_SFPSHFT(15, 0, 15, 1);  // Backdoor load to Macro Instruction 7

        // Configure Macro Sequence Register 1: LD -> SWAP -> STORE (sanitization)
        TTI_SFPLOADI(0, 0xA, 0x0004);
        TTI_SFPLOADI(0, 0x8, 0x1300);
        TTI_SFPCONFIG(0, 5, 0);

        // Configure Macro Sequence Register 0: LD -> MAD -> ROUND -> SHIFT -> STORE (computation)
        TTI_SFPLOADI(0, 0xA, 0x85DF);
        TTI_SFPLOADI(0, 0x8, 0x6316);
        TTI_SFPCONFIG(0, 4, 0);

        // Reset LoadMacroConfig
        TTI_SFPCONFIG(0, 8, 1);
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE)
    {
        // Schraudolph fast exp WITHOUT input clamping
        // Uses LOADMACRO + SFPSHFT2 pipeline with replay buffer
        // Similar constants but adds LREG[14] = 15 (shift amount for SFPSHFT2)
        // and uses SETSGN to handle sign bits correctly

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A         = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;
        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);
        constexpr float A_scaled   = A * scale_fp32;

        // Load A into LREG[12], (B-C) into LREG[13], shift=15 into LREG[14]
        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);

        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);

        TTI_SFPLOADI(0, 0xA, 15);
        TTI_SFPLOADI(0, 0x8, 0);
        TTI_SFPCONFIG(0, 14, 0);

        // Backdoor-load macro instructions: MAD, STOCHRND (INT16 mode), SETSGN
        TTI_SFPMAD(12, 0, 13, 13, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);
        TTI_SFPSETSGN(0, 4, 15, 0);

        // Configure Macro Sequence Register 0 with SETSGN, MAD, STOCHRND, STORE
        TTI_SFPLOADI(0, 0xA, 0x85EF);
        TTI_SFPLOADI(0, 0x8, 0x731E);
        TTI_SFPCONFIG(0, 4, 0);

        // Reset LoadMacroConfig
        TTI_SFPCONFIG(0xF00, 0x8, 0x1);

        // Record 32 instructions into replay buffer (16 LOADMACRO+SHFT2 pairs)
        lltt::record(0, 32);
        // ... 16 pairs of TTI_SFPLOADMACRO + TTI_SFPSHFT2 with cycling LREG pattern ...
        // (LREG0, LREG1, LREG2, LREG3 cycling, SHFT2 offset by 2)
        TTI_SFPLOADMACRO(0, 0, 3, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        // ... (14 more pairs) ...
        TTI_SFPNOP;
    }
    else if constexpr (APPROXIMATION_MODE)
    {
        // Standard approximation mode (software loop, no LOADMACRO)
        // Initialize programmable constants used by _calculate_exponential_approx_
        sfpi::vConstFloatPrgm0 = 1.442695f;                       // 1/ln(2)
        sfpi::vConstFloatPrgm1 = sfpi::s2vFloat16b(p_exp::C23_73);  // Conversion constant
        sfpi::vConstFloatPrgm2 = sfpi::s2vFloat16b(p_exp::ADJ_EXP); // Exponent adjustment
    }
    else
    {
        // Non-approximation mode: initialize reciprocal tables
        // (used by _sfpu_reciprocal_<2> when computing exp(x) for x < 0)
        _init_sfpu_reciprocal_<false>();
    }
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description | Used In |
|---|---|---|
| `sfpi::exexp(val)` | Extract biased exponent from IEEE 754 float | `_sfpu_exp_`, `_float_to_int32_for_exp21f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::exexp_nodebias(val)` | Extract exponent without removing bias | `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::exman8(val)` | Extract 8-bit mantissa with implicit leading 1 | `_float_to_int32_for_exp21f_` |
| `sfpi::exman9(val)` | Extract 9-bit mantissa | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::setexp(val, exp)` | Set the exponent field of a float | `_sfpu_exp_`, `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::setsgn(val, sign)` | Set the sign bit of a float | `_calculate_exponential_body_`, `_calculate_exponential_piecewise_` |
| `sfpi::shft(val, amount)` | Shift operation | `_float_to_int32_for_exp21f_` |
| `sfpi::addexp(val, delta)` | Add to exponent field (multiply by 2^delta) | `_sfpu_exp_61f_` |
| `sfpi::int32_to_float(val, mode)` | Integer to float conversion | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::float_to_fp16b(val, mode)` | Float32 to BFloat16 conversion with rounding | `_sfpu_exp_21f_` |
| `sfpi::vec_min_max(a, b)` | Vectorized min/max (sorts a,b so a <= b) | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::reinterpret<T>(val)` | Bit-level type reinterpretation | Throughout |
| `sfpi::s2vFloat16b(val)` | Scalar to vector BF16 broadcast | Coefficient loading |
| `sfpi::dst_reg[idx]` | Read/write DEST register at SFPU lane offset | `calculate_exponential`, `_calculate_exponential_` |
| `TTI_SFPLOADMACRO(...)` | Execute a macro sequence on DEST data | Fast approx path |
| `TTI_SFPMAD(...)` | Multiply-Accumulate-Add | Macro instruction programming |
| `TTI_SFP_STOCH_RND(...)` | Stochastic/deterministic rounding (FP32->INT) | Macro instruction programming |
| `TTI_SFPSHFT(...)` | Shift operation | Macro instruction programming |
| `TTI_SFPSHFT2(...)` | Shift operation variant 2 (used with replay) | Replay-buffer fast path |
| `TTI_SFPLOADI(...)` | Load immediate into LREG | Constant and macro setup |
| `TTI_SFPCONFIG(...)` | Configure SFPU registers and macro sequences | Constant and macro setup |
| `TTI_SFPNOP` | No-operation (pipeline timing) | Throughout fast paths |
| `TTI_SFPSETSGN(...)` | Set sign bit instruction | Non-clamping fast path |
| `lltt::replay(start, count)` | Replay recorded instructions from buffer | Non-clamping fast path |
| `lltt::record(start, count)` | Record instructions into replay buffer | Non-clamping fast path init |
| `PolynomialEvaluator::eval(...)` | Horner-form polynomial evaluation | `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_` |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` through `dst_reg[7]` | DEST register lanes; each holds a 32-element vector from the tile. The `dst_reg++` advances through 8 iterations to cover a full 32x32 tile face (256 elements) |
| `LREG[0..3]` | Working registers used by LOADMACRO sequences for loading DEST values and intermediate computation |
| `LREG[4]` | Staging register for SFPSHFT2 output in non-clamping fast path |
| `LREG[12]` | Constant A = 256/ln(2) = 369.33 (fast approx) |
| `LREG[13]` | Constant (B-C) = 32500.818 (fast approx) |
| `LREG[14]` | Threshold -88.5 (clamping path) or shift amount 15 (non-clamping path) |
| `LREG[16]` | Staging register used by SETSGN in non-clamping path to avoid write port conflicts |
| `vConstFloatPrgm0` | Programmable constant: 1/ln(2) = 1.442695 (standard approx mode) |
| `vConstFloatPrgm1` | Programmable constant: C23_73 conversion factor (standard approx mode) |
| `vConstIntPrgm2` | Programmable constant: ADJ_EXP exponent adjustment (standard approx mode) |
| `vConst0p8373` | Hardware constant 0.8373 (Horner series coefficient) |
| `vConst1` | Hardware constant 1.0 |

#### SFPU Execution Flow

The execution flow depends on which algorithm path is taken. Here is the flow for the **default EXP operation** as invoked from TTNN (param0 = 1, meaning `fast_and_approx = true`, but `APPROXIMATION_MODE = false` because `get_op_approx_mode` returns false for all ops, and `math_approx_mode` in `ComputeConfig` is set from that):

**Important clarification**: Despite `fast_and_approx` being passed as a template parameter, the actual algorithm selection depends on `APPROXIMATION_MODE` (from `math_approx_mode` in ComputeConfig). Since `get_op_approx_mode(EXP)` returns `false`, the standard unary factory sets `math_approx_mode = false`, which means `APPROXIMATION_MODE = false` in the template. This routes to the **non-approximation path** in `calculate_exponential`.

However, the template parameter `fast_and_approx` (from param0) is `true` (value `1u`). Looking at the template instantiation in `exp_tile<1u>(0)`: the first template parameter is `approx` which maps to `APPROXIMATION_MODE`. So `exp_tile<1u>` means `approx = true`, `fast_and_approx = true` (default).

**Corrected flow for `exp_tile<1u>(0)` (approx=true, fast_and_approx=true)**:

1. **Initialization** (`exp_tile_init<1u>()`):
   - Routes to `_init_exponential_<true, true, 0x3F800000, true>()`
   - This is the `FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE` path
   - Loads constants A, B-C, threshold into LREG[12], LREG[13], LREG[14]
   - Programs 4 macro instructions (SWAP, MAD, STOCHRND, SHIFT) via backdoor load
   - Configures 2 macro sequences (sanitize + compute)

2. **Per-tile computation** (`exp_tile<1u>(0)`):
   - Routes to `calculate_exponential<true, true, DST_ACCUM_MODE, false, false, true, 8>`
   - Since `APPROXIMATION_MODE && FAST_APPROX && CLAMP_NEGATIVE`, enters LOADMACRO path
   - **Phase 1 (Sanitization)**: 8 SFPLOADMACRO calls using Macro Sequence 1
     - Each loads a DEST slice, compares against -88.5 (LREG[14]), keeps the larger value (SWAP), stores back
     - Covers all 16 DEST offsets (0 through 14, step 2) = 8 groups of 32 elements = 256 elements
   - **Phase 2 (Computation)**: 8 SFPLOADMACRO calls using Macro Sequence 0
     - Each loads the sanitized value, computes MAD (A*y + (B-C)), rounds to UINT16, shifts left by 15, stores
     - The result is a float whose exponent encodes the approximate exp() value

3. **Tile lifecycle in compute kernel**:
   - `tile_regs_acquire()` -- lock DST registers
   - `cb_wait_front(c_0, 1)` -- wait for reader to produce input tile
   - `copy_tile(c_0, 0, 0)` -- unpack input tile from CB to DST[0]
   - `exp_tile_init<1u>()` -- initialize SFPU (runs once, though called per tile -- init is idempotent)
   - `exp_tile<1u>(0)` -- compute exp in-place on DST[0]
   - `tile_regs_commit()` -- release DST for packer
   - `tile_regs_wait()` -- wait for packer ready
   - `pack_tile(0, c_2)` -- pack DST[0] to output CB
   - `cb_pop_front(c_0, 1)` -- free input CB slot
   - `tile_regs_release()` -- release DST for next iteration

#### SFPU Configuration

| Configuration | Value | Description |
|---|---|---|
| `SFPU_OP_EXP_INCLUDE` | `1` | Compile-time define that gates `#include "exp.h"` in `sfpu_split_includes.h` |
| `math_approx_mode` | `false` | From `get_op_approx_mode(EXP)` which returns false |
| `fast_and_approx` | Controlled by param0 (default `true`) | Template parameter from `UnaryWithParam(EXP, 1.0f)` |
| `InputClamping` | `ClampToNegative` (default) | Clamps inputs below -88.5 to prevent incorrect outputs |
| `iterations` | 8 (default) | 8 SFPU lanes x 32 elements each = 256 elements per tile face |
| `scale` | `0x3F800000` (1.0f) | Default scale factor (no scaling) |
| `DST_ACCUM_MODE` | From `fp32_dest_acc_en` | Selects between BF16 and FP32 DEST accumulation |

#### Hardware Compatibility Notes

**Wormhole vs. Blackhole differences**:

1. **ADDR_MOD usage**: Blackhole uses `ADDR_MOD_7` as a named constant in SFPLOADMACRO calls, while Wormhole uses the literal value `3`. Functionally identical.

2. **SFPMAD optimization**: The Cody-Waite accurate path (`_sfpu_exp_f32_accurate_`) has slightly different comments:
   - Wormhole: "SFPMAD on Wormhole can only do VD = VA * VB + VC"
   - Blackhole: "On Blackhole, SFFPMAD has SFPMAD_MOD1_NEGATE_VA and SFPMAD_MOD1_NEGATE_VC"
   - Both use the same negated-constant optimization to ensure a single SFPMAD instruction

3. **Shared LLK layer**: The `_calculate_exponential_` and `_init_exponential_` implementations in `tt_llk_wormhole_b0` and `tt_llk_blackhole` are functionally identical. The LOADMACRO-based fast paths use the same instruction sequences on both architectures.

4. **tt-metal layer**: The `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and `calculate_exponential` functions in the tt-metal `ckernel_sfpu_exp.h` are identical between Wormhole and Blackhole.

---

## Algorithm Summary

The EXP operation supports four distinct algorithmic paths, selected at compile time:

### Path 1: Fast Approximate with Clamping (FAST_APPROX=true, APPROX=true, CLAMP=true)
- **Default for TTNN `ttnn.exp()`** (param0=1)
- Schraudolph algorithm: `exp(x) ~ reinterpret_float(int(A*x + (B-C)))`
- Input clamped to [-88.5, +inf] before computation
- Uses SFPU LOADMACRO sequences for parallel computation
- Accuracy: ~2-3% relative error

### Path 2: Fast Approximate without Clamping (FAST_APPROX=true, APPROX=true, CLAMP=false)
- Same Schraudolph algorithm but without input sanitization
- Uses SFPSHFT2 + replay buffer for higher throughput
- Outputs negative values for inputs < -88.5 (requires downstream ReLU)
- ~2.5 cycles/element (8-element) or ~2.125 cycles/element (32-element)

### Path 3: Standard Approximation (FAST_APPROX=false, APPROX=true)
- Uses software loop with `_calculate_exponential_piecewise_`
- Piecewise: saturates to inf for x >= 89, to 0 for x < -42, uses bit-manipulation approx in between
- Better accuracy than fast paths but slower

### Path 4: Non-Approximation Precise (APPROX=false)
Two sub-paths based on `fp32_dest_acc_en`:
- **BF16 dest** (`_sfpu_exp_21f_`): Moroz et al. 2022 exp_21f algorithm with 2nd-degree polynomial refinement
- **FP32 dest** (`_sfpu_exp_f32_accurate_`): Cody-Waite range reduction with 7th-order Taylor series, < 1 ULP accuracy

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: exp_tile SFPU operation call chain, LLK API patterns, file locations for compute kernel and SFPU headers
- `tenstorrent/tt-llk`: `_calculate_exponential_` dispatch structure, `_sfpu_exp_` Horner series algorithm, shared LLK implementation paths

### Confluence References
- Not consulted for this analysis. The SFPU instruction details were sufficiently covered by the source code inline comments and DeepWiki.

### Glean References
- Not consulted for this analysis.

---

## File Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory (CB setup, kernel registration, work distribution) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header (shared_variables_t, cached_program_t) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Defines generation (`get_block_defines`, `get_compute_kernel_path`, `get_op_approx_mode`) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel |
| `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` | API layer: `exp_tile_init`, `exp_tile` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include gating for SFPU ops |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` | Architecture-specific exp (Wormhole): `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, `calculate_exponential` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` | Architecture-specific exp (Blackhole): identical structure |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | SFPU dispatch macros (`SFPU_TEMPLATE_PARAMS_KERNEL_FN`, etc.) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h` | Shared LLK: `_sfpu_exp_`, `_calculate_exponential_`, `_init_exponential_` (Wormhole) |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` | Shared LLK: identical (Blackhole) |
