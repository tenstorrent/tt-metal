# TTNN Operation Analysis: LOG (Natural Logarithm)

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | LOG (Natural Logarithm) |
| **Operation Type** | Unary Element-wise SFPU |
| **UnaryOpType Enum** | `UnaryOpType::LOG` |
| **Program Factory** | `UnaryProgramFactory` (shared with all standard unary ops) |
| **Compute Kernel** | `eltwise_sfpu.cpp` (generic SFPU dispatch kernel) |
| **SFPU Kernel** | `ckernel_sfpu_log.h` |
| **Math Fidelity** | `HiFi4` |
| **Approx Mode** | `false` (always exact -- `get_op_approx_mode` returns false for all ops by default) |
| **Data Types** | Float32, BFloat16, Float16_b |

The LOG operation computes the natural logarithm (ln) of each element in the input tensor. It is implemented as a SFPU (Special Function Processing Unit) operation on the Tensix vector unit, using polynomial approximation techniques to evaluate ln(x) for IEEE 754 floating-point inputs.

---

## Program Factory Analysis

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Type
`UnaryProgramFactory` -- a generic program factory shared by all standard unary element-wise operations (exp, log, sigmoid, tanh, relu, etc.). The LOG operation is not special-cased; it flows through the default path.

### Program Structure

The program factory creates a three-kernel program (Reader, Compute, Writer) that runs across multiple Tensix cores in SPMD fashion.

#### Work Distribution
```cpp
auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```
Tiles are distributed across the compute grid using `split_work_to_cores`, which creates two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets the remainder. Each group gets its own compute kernel instance with the appropriate tile count as a compile-time argument.

#### Circular Buffers

| CB Index | Name | Purpose | Page Count | Data Format |
|---|---|---|---|---|
| `c_0` | Input CB | Holds input tiles from DRAM | 2 | Input data format |
| `c_2` | Output CB | Holds output tiles for writing to DRAM | 2 | Output data format |

CB `c_1` (temporary buffer) is NOT allocated for LOG -- it is only used for `HARDSHRINK` and `LOGIT` operations.

The double-buffering (2 pages per CB) enables pipelining: the reader can fill one page while the compute kernel processes the other.

#### Compile-Time Defines for LOG

The program factory generates SFPU operation chain defines via `utils::get_block_defines()`. For LOG, these resolve to:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_CHAIN_0` | `"SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"` | Macro expansion chain |
| `SFPU_OP_CHAIN_0_INIT_0` | `"log_tile_init();"` | SFPU init call |
| `SFPU_OP_CHAIN_0_FUNC_0` | `"log_tile(0);"` | SFPU compute call |
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `"1"` | Include guard for `compute_kernel_api.h` |

LOG falls through to the `default` case in `get_macro_definition()`, which maps to `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`. This causes the generic `compute_kernel_api.h` to be included (which contains `log_tile_init` and `log_tile`), rather than a dedicated split-include header.

When LOG is called with parameters (parameterized path), the `fast_and_approx` template parameter is passed:
- `log_tile_init<{param0}u>()` and `log_tile<{param0}u>(idst)`
- Default parameter value is `true` (from `string_to_unary_with_param`: `UnaryWithParam(UnaryOpType::LOG, static_cast<float>(true))`)

When called without parameters (default path):
- `log_tile_init()` and `log_tile(idst)` -- template defaults to `fast_and_approx = false`

#### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,  // Default unless preserve_fp32_precision
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = math_approx_mode,  // false for LOG
    .compile_args = compute_kernel_args,
    .defines = unary_defines
}
```

#### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|---|---|---|---|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start tile ID) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start tile ID) |
| Compute | `0` (packed_scalar1, unused) | `0` (packed_scalar2, unused) | -- |

LOG does not use any runtime scalar parameters. The `packed_scalar1` and `packed_scalar2` arguments are always 0.

### Program Caching
The factory implements `override_runtime_arguments` for efficient program reuse. When buffer addresses change (but shapes stay the same), only the buffer address runtime arguments are updated -- no recompilation needed.

---

## Kernel Implementations

### Reader Kernel
**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);      // DRAM buffer address for source tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);      // Number of tiles this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);        // Starting tile ID for this core

    constexpr auto src_args = TensorAccessorArgs<0>();          // Compile-time tensor accessor config (from factory)

    constexpr uint32_t cb_id_in0 = 0;                          // CB index 0 = input circular buffer

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // ublocks size defined in pages (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;                            // Process one tile at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);  // Create accessor for DRAM reads

// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {            // Reverse iteration (for backward pass ops)
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {             // Forward iteration: tile start_id to end_id
#endif
        cb_reserve_back(cb_id_in0, onepage);                    // Wait for space in input CB (producer side)
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);     // Get L1 write address for next CB slot
        noc_async_read_page(i, s, l1_write_addr);              // Async DMA read: DRAM tile i -> L1
        noc_async_read_barrier();                               // Wait for DMA to complete
        cb_push_back(cb_id_in0, onepage);                       // Signal compute kernel: tile is ready
    }
}
```

### Writer Kernel
**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);        // DRAM buffer address for destination tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);        // Number of tiles this core writes
    const uint32_t start_id = get_arg_val<uint32_t>(2);          // Starting tile ID for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2 = 2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // Compile-time tensor accessor config

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);                         // Sharded: wait for all tiles, no DRAM write needed
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {              // Forward iteration over output tiles
#endif
        cb_wait_front(cb_id_out, onepage);                       // Wait for compute kernel to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);        // Get L1 read address for completed tile
        noc_async_write_page(i, s, l1_read_addr);               // Async DMA write: L1 -> DRAM tile i
        noc_async_writes_flushed();                              // Ensure write is in flight
        cb_pop_front(cb_id_out, onepage);                        // Free the CB slot for reuse
    }
    noc_async_write_barrier();                                   // Final barrier: all writes complete
#endif
}
```

### Compute Kernel
This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the generic SFPU compute kernel shared by all standard unary SFPU operations. The specific operation (LOG in this case) is injected at compile time via the `SFPU_OP_CHAIN_0` preprocessor define.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"                             // Common compute API (cb_reserve_back, cb_push_back, etc.)
#include "api/compute/tile_move_copy.h"                     // copy_tile: FPU copy from CB to DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"        // Base unary eltwise includes
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditional SFPU op includes based on defines
#include "api/compute/eltwise_unary/trigonometry.h"         // Trig functions (included unconditionally)
#include "api/compute/mul_int_sfpu.h"                       // Integer multiply SFPU
#include "api/compute/eltwise_unary/rpow.h"                 // Reverse power
#include "api/compute/eltwise_unary/rdiv.h"                 // Reverse division
#include "api/compute/eltwise_unary/fill.h"                 // Fill operation

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for LOG)

    // Initialize the SFPU for the specific operation. For LOG, this expands to:
    //   init_sfpu(c_0, c_2) followed by SFPU_OP_CHAIN_0_INIT_0 which is log_tile_init()
    // log_tile_init() loads programmable constants:
    //   vConstFloatPrgm0 = ln(2), vConstFloatPrgm1 and vConstFloatPrgm2 = polynomial coefficients
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for the block (1 tile). This blocks if output CB is full.
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST registers. The math RISC-V must own DST
            // before performing any compute. This synchronizes with the pack engine.
            tile_regs_acquire();

            // Wait for reader kernel to produce one input tile in CB c_0
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from input CB (c_0) slot 0 to DST register 0.
            // This uses the FPU matrix engine to move data, not the SFPU.
            // The tile data is now in DST[0] ready for SFPU processing.
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU operation chain. For LOG, this expands to:
            //   log_tile_init(); log_tile(0);
            // which calls:
            //   llk_math_eltwise_unary_sfpu_log_init() -- loads constants into SFPU programmable registers
            //   llk_math_eltwise_unary_sfpu_log(0)     -- runs SFPU log kernel on DST[0]
            // The SFPU reads from DST[0], computes ln(x) element-wise, writes result back to DST[0].
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Release DST to pack engine. The math RISC-V signals that DST contains valid results.
            tile_regs_commit();

            // Wait for pack engine to be ready to consume DST data
            tile_regs_wait();

            // Pack tile from DST register 0 to output CB (c_2).
            // The pack engine converts from DST accumulator format to the output data format.
            pack_tile(0, tt::CBIndex::c_2);

            // Free the input tile from CB c_0 so reader can reuse the slot
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers back to math RISC-V for the next iteration
            tile_regs_release();
        }
        // Push the completed block to the output CB so the writer kernel can consume it
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h`
- **LLK Bridge**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h`
- **Compute API**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (contains `log_tile` and `log_tile_init`)

#### Call Chain

```
log_tile<fast_and_approx>(idst)                                    [compute_kernel_api.h]
  -> MATH(llk_math_eltwise_unary_sfpu_log<APPROX, fast_and_approx, DST_ACCUM_MODE>(idst))
       -> _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(       [tt_llk submodule]
              ckernel::sfpu::calculate_log<APPROX, FAST_APPROX, false, is_fp32_dest_acc_en>,
              dst_index, vector_mode, 0)
            -> Sets DST write address, iterates over 4 faces (16 rows each)
            -> Calls calculate_log() for each face
               -> Loops 8 iterations (one per pair of rows in a 16-row face)
               -> Each iteration: read DST[0], compute log, write DST[0], advance DST pointer
```

#### Annotated SFPU Kernel Source (Wormhole B0)

The SFPU log kernel has two code paths controlled by `is_fp32_dest_acc_en`:
1. **BFloat16/Float16_b path** (`calculate_log_body`): Uses a degree-5 minimax polynomial over [1,2]
2. **Float32 path** (`calculate_log_f32_body`): Uses a higher-accuracy approach with range reduction to [sqrt(2)/2, sqrt(2)] and a degree-5 odd-power series via z=(m-1)/(m+1) transformation

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"  // PolynomialEvaluator::eval -- Horner's method polynomial evaluation

namespace ckernel {
namespace sfpu {

// ============================================================================
// BFloat16 / Float16_b path: Minimax polynomial approximation
// ============================================================================
// This function computes ln(x) for non-FP32 accumulator modes.
// Algorithm:
//   1. Decompose x = 2^n * m where m is in [1, 2)
//   2. Compute ln(m) using a degree-5 minimax polynomial (Sollya-generated)
//   3. Result = n * ln(2) + ln(m)
template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {

    // Step 1: Normalize input to [1, 2) range by replacing the exponent with bias (127).
    // setexp preserves the mantissa bits but forces exponent = 127, so the value becomes 1.mantissa.
    // This is equivalent to extracting the mantissa as a float in [1, 2).
    sfpi::vFloat x = sfpi::setexp(in, 127);

    // Step 2: Evaluate minimax polynomial for ln(x) over [1, 2].
    // Generated using Sollya: fpminimax(log(x), 5, [|single...|], [1+2^(-20); 2], relative)
    // The polynomial is: c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
    // where c1 = vConstFloatPrgm1 (loaded during init), c0 = vConstFloatPrgm2 (loaded during init),
    // and c2..c5 are hardcoded literal constants.
    // PolynomialEvaluator::eval uses Horner's method for numerical stability:
    //   result = ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,       // Coefficient loaded at init: -2.0069785118103027
        sfpi::vConstFloatPrgm2,       // Coefficient loaded at init: 3.767500400543213
        -2.800232410430908,           // Hardcoded polynomial coefficient
        1.3681391477584839,           // Hardcoded polynomial coefficient
        -0.3706687390804291,          // Hardcoded polynomial coefficient
        0.04224011301994324);         // Hardcoded polynomial coefficient (highest degree term)

    // Step 3: Extract the original (debiased) exponent as an integer.
    // exexp returns the exponent field minus the bias (127), so for x=4.0 it returns 2.
    sfpi::vInt exp = sfpi::exexp(in);

    // The SFPU uses sign-magnitude representation for integers, but exexp returns two's complement.
    // For negative exponents (e.g., x < 1.0), convert from two's complement to sign-magnitude:
    //   ~exp + 1 = absolute value (two's complement negation)
    //   setsgn(..., 1) = set the sign bit to negative
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    // Step 4: Convert integer exponent to float for the final calculation.
    // int32_to_float converts a sign-magnitude integer to IEEE 754 float.
    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

    // Step 5: Combine: ln(x) = exponent * ln(2) + ln(mantissa)
    // vConstFloatPrgm0 was loaded with ln(2) = 0.69314718246459961 during init.
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;

    // Optional base scaling: for log10 and log2, multiply by 1/ln(base).
    // For plain LOG, HAS_BASE_SCALING is false, so this is compiled out.
    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    // Special case: ln(0) = -infinity
    v_if(in == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    // When not in fast_approx mode, handle additional edge cases:
    //   - ln(+inf) = +inf
    //   - ln(NaN) = NaN
    //   - ln(negative) = NaN
    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            // Positive infinity: return +inf
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {
            // exp==128 means exponent field is all 1s (inf/NaN), or input is negative
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }

    // For non-FP32 dest accumulator mode, convert result back to BFloat16 format.
    // This truncates the lower 16 mantissa bits to match the BFloat16 representation.
    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

// ============================================================================
// Float32 path: Higher-accuracy polynomial using z = (m-1)/(m+1) transform
// ============================================================================
// This function is used when fp32_dest_acc_en is true, providing higher precision.
// Algorithm:
//   1. Handle special cases (0, inf, NaN, negative)
//   2. Decompose x = 2^n * m where m in [1, 2)
//   3. Range reduction: if m >= sqrt(2), set m = m/2 and n = n+1
//      so m is now in [sqrt(2)/2, sqrt(2)] ~ [0.707, 1.414]
//   4. Transform: z = (m-1)/(m+1), which maps [0.707, 1.414] to [-0.172, 0.172]
//   5. Compute ln(m) = 2*z*(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
//   6. Result = n * ln(2) + ln(m)
template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    sfpi::vFloat result;

    // Extract debiased exponent for special case detection
    sfpi::vInt exp = sfpi::exexp(val);

    // Handle special cases first using conditional execution (v_if/v_elseif/v_else)
    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        // Positive infinity input: ln(+inf) = +inf
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < 0.f) {
        // NaN input (exp==128 with nonzero mantissa) or negative input: return NaN
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == 0.0f) {
        // Zero input: ln(0) = -inf
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        // Normal case: compute ln(val)

        // Step 1: Extract mantissa as float in [1, 2)
        sfpi::vFloat m = sfpi::setexp(val, 127);

        // Step 2: Range reduction -- narrow m to [sqrt(2)/2, sqrt(2)]
        // If m >= sqrt(2), halve it and compensate by incrementing the exponent.
        // This tighter range improves polynomial convergence.
        // NOTE: On Wormhole, sqrt(2) is a literal constant (1.4142135381698608f).
        //        On Blackhole, it's loaded from vConstFloatPrgm1 (set during init).
        constexpr float SQRT2 = 1.4142135381698608f;
        v_if(m >= SQRT2) {
            m = m * 0.5f;
            exp = exp + 1;
        }
        v_endif;

        // Step 3: Transform to z = (m-1)/(m+1)
        // This centers the approximation around z=0, where the Taylor series converges fast.
        // For m in [0.707, 1.414], z is in [-0.172, 0.172] -- a very narrow range.
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

        // Compute reciprocal of (m+1) using Newton-Raphson with 2 iterations.
        // _sfpu_reciprocal_<2> provides sufficient precision for FP32.
        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

        sfpi::vFloat z2 = z * z;

        // Step 4: Evaluate ln(m) = 2*z*(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
        // Using Horner's method on the even-power polynomial in z^2:
        //   p(z^2) = 1 + (1/3)*z^2 + (1/5)*z^4 + (1/7)*z^6 + (1/9)*z^8 + (1/11)*z^10
        // These are the coefficients of the Taylor series for atanh(z)/z.
        sfpi::vFloat p = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,              // 1.0 (constant term)
            0.3333333333333333f,        // 1/3
            0.2f,                       // 1/5
            0.14285714285714285f,       // 1/7
            0.1111111111111111f,        // 1/9
            .09090909090909091f);       // 1/11

        sfpi::vFloat ln_m = 2.0f * (z * p);

        // Step 5: Convert exponent to float (sign-magnitude conversion for negatives)
        v_if(exp < 0) {
            sfpi::vInt exp_abs = ~exp + 1;
            exp = sfpi::setsgn(exp_abs, 1);
        }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        // Step 6: Combine: ln(x) = exponent * ln(2) + ln(mantissa)
        // NOTE: On Wormhole, LN2 is a literal constant.
        //        On Blackhole, it's loaded from vConstFloatPrgm2 (set during init).
        constexpr float LN2 = 0.69314718246459961f;
        result = expf * LN2 + ln_m;

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }
    }
    v_endif;

    return result;
}

// ============================================================================
// Top-level SFPU log kernel: dispatches per-element over all rows in a face
// ============================================================================
// ITERATIONS = 8 by default, processing 8 rows per face (each vFloat is a SIMD vector
// across the 32 columns of a tile). Since there are 4 faces per tile and 4 rows per
// face-half, 8 iterations * 4 faces = 32 rows = one full tile.
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8                                        // Compiler hint to fully unroll the loop
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];                 // Read current row from DST register
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            // BFloat16 path: minimax polynomial
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            // FP32 path: z-transform series
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;                          // Write result back to same DST row
        sfpi::dst_reg++;                                    // Advance to next row in the face
    }
}

// ============================================================================
// Initialization: load programmable constants into SFPU registers
// ============================================================================
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        // BFloat16 path: load polynomial coefficients and ln(2)
        sfpi::vConstFloatPrgm0 = 0.69314718246459961f;     // ln(2) -- used in exponent correction
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;      // Minimax polynomial coefficient (highest-order term passed to eval)
        sfpi::vConstFloatPrgm2 = 3.767500400543213;        // Minimax polynomial coefficient
    } else {
        // FP32 path: initialize reciprocal (Newton-Raphson seed) and load constants
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
        // NOTE: _init_reciprocal_ sets vConstFloatPrgm0 internally (to 2.0f for Newton-Raphson).
        // The FP32 path uses literal constants for LN2 and SQRT2 on Wormhole,
        // but on Blackhole it loads them into vConstFloatPrgm1 and vConstFloatPrgm2.
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

The LOG SFPU kernel uses the following SFPI intrinsics, which compile to SFPU hardware instructions:

| SFPI Intrinsic | SFPU Instruction | Description |
|---|---|---|
| `sfpi::setexp(val, 127)` | `SFPSETEXP` | Replace the exponent field of a float with a specified value. Used to normalize input to [1, 2) range. |
| `sfpi::exexp(val)` | `SFPEXEXP` with `SFPEXEXP_MOD1_DEBIAS` | Extract the debiased exponent from a float (exponent - 127). Returns an integer. |
| `sfpi::setsgn(val, 1)` | `SFPSETSGN` | Set the sign bit of a value. Used to convert two's complement negative to sign-magnitude. |
| `sfpi::int32_to_float(val, 0)` | `SFPCAST` | Convert a sign-magnitude integer to IEEE 754 float. |
| `sfpi::float_to_fp16b(val, 0)` | `SFPCAST` | Convert FP32 to BFloat16 (truncate lower 16 mantissa bits). |
| `sfpi::reinterpret<vFloat>(vInt)` | (no instruction -- register reinterpret) | Bitwise reinterpretation between float and int views of the same register. |
| `sfpi::dst_reg[0]` | `SFPLOAD` / `SFPSTORE` | Read from / write to DST accumulator registers. |
| `sfpi::dst_reg++` | Address increment | Advance the DST register pointer to the next row. |
| Multiply (`*`), Add (`+`), Subtract (`-`) | `SFPMAD` | Fused multiply-add: `a * b + c`. All arithmetic in the polynomial evaluation compiles to chains of SFPMAD. |
| `v_if` / `v_elseif` / `v_else` / `v_endif` | `SFPPUSHC`, `SFPSETCC`, `SFPCOMPC`, `SFPPOPC` | Predicated execution using condition codes. Enables per-lane branching without divergence. |
| `_sfpu_reciprocal_<2>(val)` (FP32 path only) | `SFPMAD` chain | Newton-Raphson reciprocal approximation with 2 refinement iterations. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `sfpi::vConstFloatPrgm0` | Stores `ln(2) = 0.69314718...` for the BF16 path. For FP32, set by `_init_reciprocal_` (value 2.0f for Newton-Raphson). |
| `sfpi::vConstFloatPrgm1` | BF16: polynomial coefficient `-2.0069785118103027`. BH FP32: `sqrt(2) = 1.4142135381698608`. WH FP32: unused (literal constant). |
| `sfpi::vConstFloatPrgm2` | BF16: polynomial coefficient `3.767500400543213`. BH FP32: `ln(2) = 0.69314718246459961`. WH FP32: unused (literal constant). |
| `sfpi::vConst1` | Built-in constant `1.0f` (used in FP32 path for polynomial and arithmetic). |
| `sfpi::dst_reg[0]` | Source and destination for tile data. Each access reads/writes one row of 32 elements. |
| LReg[0-6] (implicit) | Temporary SFPU local registers used by SFPMAD chains for intermediate polynomial terms. |

#### SFPU Execution Flow

1. **Initialization** (`log_init`):
   - For BF16 mode: Load `ln(2)`, and two polynomial coefficients into `vConstFloatPrgm0/1/2`.
   - For FP32 mode: Initialize reciprocal lookup tables (Newton-Raphson seed). On Blackhole, also load `sqrt(2)` and `ln(2)` into programmable constants.

2. **Per-tile execution** (called via `_llk_math_eltwise_unary_sfpu_params_`):
   - The LLK params wrapper sets the DST base address to the target tile.
   - Iterates over 4 faces of the 32x32 tile (each face = 16 rows x 32 cols).
   - For each face, calls `calculate_log` which processes 8 iterations (8 rows, each a 32-wide SIMD vector).

3. **Per-row execution** (`calculate_log` inner loop):
   - **Read**: `sfpi::vFloat in = sfpi::dst_reg[0]` -- load one row (32 elements) from DST.
   - **Compute** (BF16 path):
     a. `setexp(in, 127)` -- normalize mantissa to [1, 2)
     b. Evaluate degree-5 minimax polynomial via `PolynomialEvaluator::eval` (chain of SFPMAD instructions)
     c. `exexp(in)` -- extract debiased exponent
     d. Convert negative exponent to sign-magnitude: `setsgn(~exp + 1, 1)`
     e. `int32_to_float(exp, 0)` -- convert exponent to float
     f. Combine: `result = expf * ln(2) + series_result`
     g. Handle edge cases: `in == 0 -> -inf`, `in == +inf -> +inf`, `in < 0 -> NaN`
     h. Convert to BF16: `float_to_fp16b(result, 0)`
   - **Compute** (FP32 path):
     a. Check special cases first (inf, NaN, negative, zero)
     b. `setexp(val, 127)` -- extract mantissa m in [1, 2)
     c. Range reduction: if `m >= sqrt(2)`, set `m = m/2`, `exp++`
     d. Transform: `z = (m-1) / (m+1)` using `_sfpu_reciprocal_<2>(m+1)`
     e. Evaluate: `p(z^2) = 1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11`
     f. `ln_m = 2 * z * p`
     g. Convert exponent to float (with sign-magnitude handling)
     h. Combine: `result = expf * ln(2) + ln_m`
   - **Write**: `sfpi::dst_reg[0] = result` -- store result back to DST row.
   - **Advance**: `sfpi::dst_reg++` -- move to next row.

4. **Post-SFPU**: The compute kernel calls `tile_regs_commit()` and `pack_tile()` to pack DST data back into the output circular buffer in the target data format.

#### SFPU Configuration

| Configuration | Value | Notes |
|---|---|---|
| `APPROXIMATION_MODE` | Template parameter from `APPROX` define | Controls approximation vs exact mode in LLK layer |
| `FAST_APPROX` | Template parameter (default: `false` for `log_tile()`, `true` when parameterized as `log_tile<1u>()`) | When true, skips inf/NaN/negative edge case handling |
| `HAS_BASE_SCALING` | `false` for LOG (hardcoded in `llk_math_eltwise_unary_sfpu_log`) | `true` for LOG10 and LOG2 variants |
| `is_fp32_dest_acc_en` | From `DST_ACCUM_MODE` compile-time define | Selects between BF16 and FP32 code paths |
| `ITERATIONS` | `8` (default) | Number of rows processed per SFPU invocation (one face-half) |
| `MathFidelity` | `HiFi4` | Set by program factory; highest fidelity mode |

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations are functionally identical but differ in constant management for the FP32 path:

| Aspect | Wormhole B0 | Blackhole |
|---|---|---|
| **BF16 path** | Identical | Identical |
| **FP32 init** | `_init_reciprocal_<false, false>()` (2 template params) | `_init_sfpu_reciprocal_<false>()` (1 template param) + explicit `vConstFloatPrgm1 = sqrt(2)`, `vConstFloatPrgm2 = ln(2)` |
| **FP32 sqrt(2) constant** | Literal `constexpr float SQRT2 = 1.4142135381698608f` | `sfpi::vConstFloatPrgm1` (loaded during init) |
| **FP32 ln(2) constant** | Literal `constexpr float LN2 = 0.69314718246459961f` | `sfpi::vConstFloatPrgm2` (loaded during init) |
| **FP32 zero check** | `val == 0.0f` | `val == 0.f` (identical semantics) |

The Blackhole version uses programmable constant registers (`vConstFloatPrgm1/2`) for `sqrt(2)` and `ln(2)` instead of literal constants. This is because Blackhole's reciprocal init function (`_init_sfpu_reciprocal_`) only consumes `vConstFloatPrgm0`, leaving the other two available. On Wormhole, the reciprocal init uses a different API (`_init_reciprocal_` with `legacy_compat` parameter), and the FP32 log body uses literal constants instead.

Both architectures use the same SFPI instruction set for the core computation. The SFPU hardware is architecturally compatible between Wormhole and Blackhole for all instructions used by the log kernel.

---

## Data Flow Summary

```
DRAM                    L1 (per core)                    SFPU
 |                       |                                |
 |  noc_async_read       |                                |
 |--------------------->| CB c_0 (input, 2 tiles)        |
 |                       |                                |
 |                       |  cb_wait_front(c_0, 1)        |
 |                       |  copy_tile(c_0, 0, 0)          |
 |                       |         |                      |
 |                       |         v                      |
 |                       |  DST reg[0] ------------------>| log_tile_init()
 |                       |                                | log_tile(0)
 |                       |  DST reg[0] <------------------| (result)
 |                       |         |                      |
 |                       |  pack_tile(0, c_2)             |
 |                       |         |                      |
 |                       | CB c_2 (output, 2 tiles)       |
 |                       |  cb_pop_front(c_0, 1)          |
 |                       |                                |
 |  noc_async_write      |                                |
 |<---------------------| cb_wait_front(c_2, 1)          |
 |                       | cb_pop_front(c_2, 1)           |
```

---

## External Knowledge Sources

### DeepWiki References
- **tenstorrent/tt-metal**: Queried for unary program factory architecture, SFPU compute kernel dispatch mechanism, and `SFPU_OP_CHAIN_0` define system.
- **tenstorrent/tt-llk**: Queried for `ckernel` namespace details, `calculate_log` implementation, `_llk_math_eltwise_unary_sfpu_params_` wrapper behavior.
- **tenstorrent/tt-isa-documentation**: Queried for SFPU instruction set details (SFPLUT, SFPSETEXP, SFPEXEXP, SFPMAD).
- **tenstorrent/sfpi**: Queried for SFPI intrinsic usage in logarithm implementation (setexp, exexp, setsgn, int32_to_float, v_if/v_endif).

### Confluence References
Not consulted for this analysis. The DeepWiki responses provided sufficient detail on SFPU instructions used by the log kernel.

### Glean References
Not consulted for this analysis. The open-source codebase contained all necessary implementation details.

---

## File Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory -- creates 3-kernel program |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header -- shared_variables_t, cached_program_t |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Maps `UnaryOpType::LOG` to SFPU defines and kernel path |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | Utility function declarations |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader dataflow kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer dataflow kernel |
| `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | `log_tile()` and `log_tile_init()` API |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h` | LLK bridge: Wormhole |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h` | LLK bridge: Blackhole |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h` | SFPU kernel implementation: Wormhole |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h` | SFPU kernel implementation: Blackhole |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` | SFPU init wrapper |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | SFPU dispatch macros |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include system for SFPU ops |
