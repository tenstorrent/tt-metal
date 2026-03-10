# EXP Operation Analysis

## Overview

The EXP operation computes the element-wise exponential function `exp(x)` on each element of an input tensor. It is a unary SFPU operation that uses the generic unary program factory infrastructure shared by all unary element-wise operations in TTNN.

**Operation type**: `UnaryOpType::EXP`
**Category**: Eltwise Unary
**Execution unit**: SFPU (Special Function Processing Unit)
**Program factory**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

---

## Program Factory Structure

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Variants

The unary program factory has two variants defined in `unary_program_factory.hpp`:

1. **`UnaryProgramFactory`** -- Used for standard interleaved memory layouts. Splits work across the full compute grid using `split_work_to_cores`, creating two core groups (group 1 and group 2) to handle uneven tile distribution.

2. **`UnarySubCoreGridProgramFactory`** -- Used when an explicit sub-core grid is specified via `sub_core_grids`. Requires uniform tile distribution across the selected cores.

Both factories share the same kernel files and CB configuration patterns; they differ only in how cores are selected and how work is distributed.

### Shared Variables (for program caching)

**UnaryProgramFactory**:
```cpp
struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    tt::tt_metal::KernelHandle unary_writer_kernel_id;
    uint32_t num_cores;
    uint32_t num_cores_y;
};
```

**UnarySubCoreGridProgramFactory**:
```cpp
struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    std::vector<CoreCoord> cores_with_rtargs;
};
```

---

## Circular Buffer Configuration

| CB Index | Symbolic Name | Purpose | Page Count | Data Format | Notes |
|----------|--------------|---------|------------|-------------|-------|
| `c_0` | `src0_cb_index` | Input buffer | 2 tiles | Input tensor format | Double-buffered for pipeline overlap |
| `c_1` | `tmp0_cb_index` | Temporary intermediate | 2 tiles | Input tensor format | Only allocated for HARDSHRINK and LOGIT ops; NOT used by EXP |
| `c_2` | `output_cb_index` | Output buffer | 2 tiles | Output tensor format | Double-buffered for pipeline overlap |

**For EXP**: Only `c_0` (input) and `c_2` (output) are used. The temporary buffer `c_1` is not allocated because EXP does not require intermediate storage.

**Page size determination**: For TILE layout, page size equals `tile_size(data_format)`. For ROW_MAJOR layout, page size equals `buffer->page_size()`.

---

## Kernel Registration

### Compute Kernel Path Resolution

The compute kernel path for EXP is resolved via `utils::get_compute_kernel_path(UnaryOpType::EXP, input.dtype())`. Since EXP does not have a special-case entry in the switch statement, it falls through to the `default` case, which returns `"eltwise_sfpu.cpp"`.

Full path: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

### Preprocessor Defines for EXP

The define system uses `get_block_defines` which calls `update_macro_defines` to set `SFPU_OP_EXP_INCLUDE = 1`. This causes `sfpu_split_includes.h` to include `api/compute/eltwise_unary/exp.h`.

The `SFPU_OP_CHAIN_0` macro is defined to expand to:
- **With parameter (param0 set)**: `exp_tile_init<(uint32_t)param0>(); exp_tile<(uint32_t)param0>(idst);`
- **Without parameter**: `exp_tile_init(); exp_tile(idst);`

The default TTNN `exp` operation uses `UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true))`, meaning `param0 = 1.0f` which casts to `(uint32_t)1`. This makes the template parameter `approx = true` (since `(uint32_t)1.0f == 1`). However, `math_approx_mode` is always `false` because `get_op_approx_mode` returns `false` for all ops.

**Important distinction**: The `approx` template parameter on `exp_tile` comes from the op chain parameter, while `math_approx_mode` in ComputeConfig is a separate flag from `get_op_approx_mode`. For the default EXP invocation, `approx=true` and `fast_and_approx=true` (default), meaning the Schraudolph-based fast approximation path is used.

### Kernel Registrations Table

| Kernel Type | File Path | Config Type | Core Assignment |
|-------------|-----------|-------------|-----------------|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `ReaderDataMovementConfig` | `all_cores` |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `WriterDataMovementConfig` | `all_cores` |
| Compute (group 1) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `ComputeConfig` | `core_group_1` |
| Compute (group 2) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `ComputeConfig` | `core_group_2` |

### Compute Config

```cpp
ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,  // Default unless preserve_fp32_precision
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,  // get_op_approx_mode returns false for EXP
    .compile_args = {num_pages_per_core_group, 1},
    .defines = unary_defines  // includes SFPU_OP_EXP_INCLUDE=1, SFPU_OP_CHAIN_0, INP_FLOAT/INP_FLOAT32/etc.
}
```

### Runtime Arguments

| Kernel | Arg 0 | Arg 1 | Arg 2 |
|--------|-------|-------|-------|
| Reader | `src_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Writer | `dst_buffer->address()` | `num_pages_per_core` | `num_pages_written` (start offset) |
| Compute | `packed_scalar1` (0 for EXP) | `packed_scalar2` (0 for EXP) | -- |

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of input tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor config (interleaved layout info)

    constexpr uint32_t cb_id_in0 = 0;                      // Circular buffer index 0 = input CB (c_0)

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // ublocks size defined in pages (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;                        // Process one page at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);  // Create accessor for NoC reads

// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;                // Backwards iteration for reverse ops
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;                // Forward iteration (default for EXP)
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);               // Wait for space in input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write address for this CB slot
        noc_async_read_page(i, s, l1_write_addr);          // Issue async NoC read from DRAM to L1
        noc_async_read_barrier();                           // Wait for NoC read to complete
        cb_push_back(cb_id_in0, onepage);                  // Signal compute kernel that one page is ready
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM address of output tensor buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2 = 2)
    constexpr auto dst_args = TensorAccessorArgs<1>();     // Compile-time tensor accessor config

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);                   // For sharded output: wait for all pages, then done
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);  // Create accessor for NoC writes

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;                // Forward iteration (default for EXP)
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);                 // Wait for compute to produce one output page
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);   // Get L1 read address for this CB slot
        noc_async_write_page(i, s, l1_read_addr);          // Issue async NoC write from L1 to DRAM
        noc_async_writes_flushed();                        // Ensure write is flushed (not necessarily completed)
        cb_pop_front(cb_id_out, onepage);                  // Free the CB slot for compute to reuse
    }
    noc_async_write_barrier();                             // Final barrier to ensure all writes complete
#endif
}
```

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"                          // Common compute API: tile_regs_acquire/commit/wait/release
#include "api/compute/tile_move_copy.h"                  // copy_tile: moves tile from CB to DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"     // init_sfpu, unary_op_init_common
#include "api/compute/eltwise_unary/sfpu_split_includes.h" // Conditionally includes exp.h when SFPU_OP_EXP_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"      // Trigonometric SFPU ops (not used by EXP)
#include "api/compute/mul_int_sfpu.h"                    // Integer multiply SFPU ops (not used by EXP)
#include "api/compute/eltwise_unary/rpow.h"              // Reverse power op (not used by EXP)
#include "api/compute/eltwise_unary/rdiv.h"              // Reverse division op (not used by EXP)
#include "api/compute/eltwise_unary/fill.h"              // Fill op (not used by EXP)

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    // Initialize SFPU pipeline: configures unpack for CB c_0, pack for CB c_2,
    // sets up math HW config, and initializes pack sync
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for one block of tiles before processing
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire DST register file -- blocks until DST is available for writing
            tile_regs_acquire();

            // Wait for reader kernel to push one tile into input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from input CB slot 0 to DST register 0
            // This triggers the unpacker to convert the tile from its storage format
            // (e.g., bfloat16) into the internal DST register format
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU_OP_CHAIN_0 expands to the init+compute calls for the operation.
            // For EXP with approx=true: exp_tile_init<1u>(); exp_tile<1u>(0);
            // For EXP without params: exp_tile_init(); exp_tile(0);
            // This runs the SFPU exponential computation on DST register 0
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST register writes are complete; hand off to packer
            tile_regs_commit();

            // Wait for packer to be ready to read from DST
            tile_regs_wait();

            // Pack tile from DST register 0 into output CB c_2
            // Converts from internal format back to output storage format
            pack_tile(0, tt::CBIndex::c_2);

            // Release the input tile from the input CB so reader can reuse the slot
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST register so next iteration can acquire it
            tile_regs_release();
        }
        // Push the completed block of tiles to the output CB for the writer kernel
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **API header**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
- **Arch-specific implementation (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Arch-specific implementation (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Shared implementation**: `tt_metal/third_party/tt_llk/.../sfpu/ckernel_sfpu_exp.h` (in the tt_llk submodule; contains `_calculate_exponential_`, `_init_exponential_`, `_calculate_exponential_approx_`, `_sfpu_exp_`)

#### Annotated SFPU Kernel Source (Wormhole/Blackhole arch-specific)

The arch-specific `ckernel_sfpu_exp.h` files are identical for both Wormhole and Blackhole. They contain the improved non-approximation algorithms and the dispatch function `calculate_exponential`. The approximation-mode implementations (`_calculate_exponential_` and `_init_exponential_`) reside in the `tt_llk` submodule.

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // Shared tt_llk implementation: _calculate_exponential_, _init_exponential_,
                                          // _calculate_exponential_approx_, _sfpu_exp_
#include "sfpu/ckernel_sfpu_polyval.h"    // PolynomialEvaluator::eval -- Horner-form polynomial evaluation
#include "sfpu/ckernel_sfpu_converter.h"  // Converter::as_float -- reinterpret uint32 bits as float
#include "ckernel_sfpu_conversions.h"     // float_to_fp16b, int32_to_float conversions
#include "sfpi.h"                         // SFPI programming interface: vFloat, dst_reg, exexp, setexp, etc.

namespace ckernel {
namespace sfpu {

// Simple wrapper that delegates to the shared tt_llk _sfpu_exp_ function.
// This is the legacy non-approximation exp using Horner series.
sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

/*
 * Optimized float-to-int32 conversion for the exp21f algorithm.
 * Constraint: 0 <= val < 128.0f
 * The input is assumed to have already been divided by 2^23,
 * so the output is scaled by 2^23 compared to the actual integer value.
 * This saves one SFPADDI instruction vs. a general-purpose conversion.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);       // Extract exponent (unbiased)
    sfpi::vInt man = sfpi::exman8(val);       // Extract mantissa with implicit leading 1 bit (value in [1, 2))
    // Shift mantissa left by exponent bits to produce the integer representation.
    // reinterpret is used to treat the mantissa as unsigned for the shift operation.
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

/*
 * exp_21f algorithm from Moroz et al. 2022:
 * "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * https://doi.org/10.1109/MSP.2022.3157460
 *
 * Computes exp(x) = 2^(x/ln2) by splitting into integer and fractional parts,
 * then refining with a 2nd-degree polynomial on the fractional part.
 * This is the bfloat16 variant used when fp32_dest_acc is disabled.
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // Step 1: Convert x to base-2 representation and add IEEE 754 bias
    // xlog2 = x * (1/ln2) + 127, where 127 is the float32 exponent bias
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // Step 2: Clamp to [0, 255] to prevent overflow/underflow in intermediate values.
    // Values outside [-88.5, 88.5] in the original domain map to outside [0, 255] here.
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);   // xlog2 = max(xlog2, 0)
    sfpi::vec_min_max(xlog2, threshold_high);   // xlog2 = min(xlog2, 255)

    // Step 3: Convert to integer representation (scaled by 2^23)
    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    // Step 4: Split z into integer exponent and fractional mantissa
    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // 2^(integer part of x/ln2)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // Fractional part in [0, 1)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);

    // Step 5: Polynomial approximation of 2^(fractional) on [0, 2^23]
    // Uses 2nd-degree polynomial: frac = c0 + c1*frac + c2*frac^2
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Step 6: Recombine: result = 2^(integer) * 2^(fractional) via exponent manipulation
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // When DST is bfloat16, SFPSTORE would truncate. Explicitly round to nearest even
        // to avoid accuracy loss (e.g., 80.8 rounding to 80.5 instead of 81).
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

/*
 * exp_61f algorithm from Moroz et al. 2022 (Section 5).
 * Higher accuracy variant that uses a 6th-degree polynomial on [0, 1]
 * for the fractional part approximation.
 */
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
    // Scale fractional part down by 2^-23 to normalize to [0, 1]
    frac = sfpi::addexp(frac, -23);

    // 6th-degree polynomial approximation of 2^x on [0, 1]
    // Coefficients are essentially Taylor series of 2^x around 0
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    return y;
}

/*
 * Utility: round-to-nearest-even for float-to-int32 conversion.
 * Uses the "magic number" trick from Hacker's Delight:
 * Adding 2^23 + 2^22 to a float forces rounding to integer,
 * then subtracting recovers the rounded value.
 */
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);

    return k;
}

/*
 * High-accuracy exp(x) for fp32 using Cody-Waite range reduction.
 * Target: < 1 ULP error for float32.
 *
 * Algorithm:
 * 1. Handle overflow (x > ~89), underflow (x < ~-88), NaN
 * 2. k = round(x / ln2)
 * 3. Cody-Waite: r = x - k*ln2_hi - k*ln2_lo (extended precision subtraction)
 * 4. exp(r) via 7th-order Taylor series
 * 5. Result = 2^k * exp(r) via exponent addition
 */
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;    // In base-2: corresponds to x ~ 89
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;  // In base-2: corresponds to x ~ -88

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;

    sfpi::vInt exp_bits = sfpi::exexp(z);

    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();  // exp(large) = +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;                           // exp(very negative) = 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN(); // NaN input -> NaN output
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Cody-Waite range reduction: split ln(2) into high and low parts
        // for extended-precision subtraction. Constants are negated so that
        // the expression r = val + k*(-LN2_HI) + k*(-LN2_LO) maps naturally
        // to SFPMAD instructions (VA * VB + VC).
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;     // r_hi = val - k * ln2_hi (exact for integer k)
        sfpi::vFloat r = k * LN2_LO + r_hi;        // r = r_hi - k * ln2_lo

        // 7th-order Taylor series for exp(r) where |r| < ln(2)/2
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,     // 1
            sfpi::vConst1,     // 1
            0.5f,              // 1/2!
            1.0f / 6.0f,      // 1/3!
            1.0f / 24.0f,     // 1/4!
            1.0f / 120.0f,    // 1/5!
            1.0f / 720.0f,    // 1/6!
            1.0f / 5040.0f    // 1/7!
        );

        // Scale by 2^k: add k to the float exponent bits
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

// Dispatch: select algorithm based on fp32_dest_acc_en
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);   // bfloat16 path: Moroz exp_21f (2nd-degree poly)
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // fp32 path: Cody-Waite + 7th-order Taylor
}

/*
 * Main entry point called by exp_tile.
 * Dispatches between approximation mode (Schraudolph-based, from tt_llk submodule)
 * and non-approximation mode (improved algorithms above).
 */
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,        // 8 iterations for 32 SFPU lanes (4 elements per lane)
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Delegates to _calculate_exponential_ in tt_llk submodule.
        // Uses Schraudolph's algorithm: exploits IEEE 754 float layout to compute
        // exp(x) by treating x * (1/ln2) + bias as a float bit pattern.
        // When FAST_APPROX=true, uses macro instructions (SFPMAD + SFPSHFT/SFPSTOCHRND + SFPSETSGN).
        // When CLAMP_NEGATIVE=true, clamps inputs below -88.5 to prevent incorrect outputs.
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximation mode: iterate over SFPU lanes
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];             // Load value from DST register
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // Optional input scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);  // Compute exp
            sfpi::dst_reg[0] = result;                       // Store result back to DST

            sfpi::dst_reg++;                                  // Advance to next SFPU lane (next 4 rows)
        }
    }
}

// Initialization entry point called by exp_tile_init.
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    // Delegates to _init_exponential_ in tt_llk submodule.
    // For APPROXIMATION_MODE + FAST_APPROX: loads LN2_RECIP, A, B_minus_C constants into LREGs,
    // programs SFPMAD/SFPSTOCHRND/SFPSETSGN macro instructions via SFPCONFIG.
    // For non-approximation mode: may initialize reciprocal function constants.
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description | Used In |
|----------------------|-------------|---------|
| `sfpi::exexp(val)` | Extracts the unbiased exponent field from a float | `_float_to_int32_for_exp21f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::exexp_nodebias(val)` | Extracts the raw exponent field (with bias) | `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::exman8(val)` | Extracts 8-bit mantissa with implicit leading 1 | `_float_to_int32_for_exp21f_` |
| `sfpi::exman9(val)` | Extracts 9-bit mantissa | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::setexp(val, exp)` | Sets the exponent field of a float to a specified value | `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_` |
| `sfpi::addexp(val, n)` | Adds an integer to the exponent field (multiply by 2^n) | `_sfpu_exp_61f_` |
| `sfpi::shft(val, n)` | Logical/arithmetic bit shift | `_float_to_int32_for_exp21f_` |
| `sfpi::reinterpret<T>(val)` | Reinterpret-cast between vector types without bit changes | Throughout |
| `sfpi::int32_to_float(val, 0)` | Convert integer to float | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::float_to_fp16b(val, 0)` | Convert float32 to bfloat16 with round-to-nearest-even | `_sfpu_exp_21f_` (when bfloat16 mode) |
| `sfpi::vec_min_max(a, b)` | Vectorized min/max: swaps elements so a <= b per lane | `_sfpu_exp_21f_`, `_sfpu_exp_61f_` |
| `sfpi::s2vFloat16b(val)` | Convert scalar uint16 to vFloat (bfloat16 broadcast) | `calculate_exponential` (scale mode) |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Horner-form polynomial evaluation | All non-approx paths |
| `Converter::as_float(bits)` | Reinterpret uint32 as float | `_sfpu_round_nearest_int32_` |
| **Approximation mode only (from tt_llk):** | | |
| `TTI_SFPMAD` | Vectorized multiply-add: VD = VA * VB + VC | Schraudolph fast approx init |
| `TTI_SFP_STOCH_RND` | Stochastic rounding (used as bit-shift in this context) | Schraudolph fast approx |
| `TTI_SFPSETSGN` | Set sign bit of result | Schraudolph fast approx |
| `TTI_SFPLOADI` | Load immediate constant into LREG | `_init_exponential_` |
| `TTI_SFPCONFIG` | Configure macro sequence registers | `_init_exponential_` |
| `TTI_SFPLOADMACRO` | Execute a macro sequence (load + chained instructions) | `_calculate_exponential_` (fast path) |

#### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| `dst_reg[0]` | Source and destination for tile data. Each SFPU iteration operates on one 32-element lane of the DST register. The `dst_reg++` advances to the next lane. |
| `LReg[0..3]` | Local registers used for intermediate computations within SFPI expressions. The compiler maps `vFloat` temporaries to LRegs. |
| `LReg[3]` | In approximation mode, used as the primary operand for SFPLUT-based operations. |
| `vConstFloatPrgm0/1/2` | Programmable constants loaded during `_init_exponential_` for approximation mode (1/ln2, C23_73, ADJ_EXP). |
| Macro sequence registers 0, 1 | In fast-approx mode, pre-programmed instruction sequences loaded via `TTI_SFPCONFIG` and executed via `TTI_SFPLOADMACRO`. |

#### SFPU Execution Flow

1. **Initialization** (`init_sfpu` + `exp_tile_init`):
   - `init_sfpu(c_0, c_2)` configures the unpack pipeline (reads from `c_0`), the pack pipeline (writes to `c_2`), and initializes math HW.
   - `exp_tile_init<approx, fast_and_approx, scale, clamp_negative>()` calls `_init_exponential_` which:
     - **Approx + fast path**: Loads constants (LN2_RECIP, A, B_minus_C) into LREGs via `TTI_SFPLOADI`, programs macro instructions (SFPMAD, SFPSTOCHRND, SFPSETSGN) via `TTI_SFPCONFIG`, and configures macro sequence registers.
     - **Approx (non-fast) path**: Sets `vConstFloatPrgm0 = 1/ln2`, `vConstFloatPrgm1 = C23_73`, `vConstFloatPrgm2 = ADJ_EXP`.
     - **Non-approx path**: Initializes the reciprocal function for handling negative inputs.

2. **Tile acquisition**:
   - `cb_wait_front(c_0, 1)` blocks until the reader kernel has pushed one tile into the input CB.
   - `tile_regs_acquire()` acquires exclusive access to the DST register file.

3. **Unpack**:
   - `copy_tile(c_0, 0, 0)` triggers the unpacker RISC-V to read the tile from CB `c_0` slot 0 and write it into DST register 0. Data format conversion (e.g., bfloat16 to float32 internal) happens during unpack.

4. **SFPU compute** (`exp_tile` -> `calculate_exponential`):
   - The function iterates 8 times (ITERATIONS=8), processing 4 rows of the 32x32 tile per iteration (32 rows total = 8 iterations x 4 rows).
   - Each iteration:
     a. Loads the value from `dst_reg[0]` (current lane).
     b. Computes `exp(x)` using the selected algorithm (see below).
     c. Stores the result back to `dst_reg[0]`.
     d. Advances `dst_reg++` to the next lane.
   - **Default path (approx=true, fast_and_approx=true)**: Uses Schraudolph's algorithm with pre-programmed macro instructions. Exploits the IEEE 754 float bit layout to compute exp(x) by treating `x * (1/ln2) + bias` as a float bit pattern, then using bit shifts and MAD operations.
   - **Non-approx bfloat16 path**: `_sfpu_exp_21f_` -- Moroz exp_21f with 2nd-degree polynomial refinement.
   - **Non-approx fp32 path**: `_sfpu_exp_f32_accurate_` -- Cody-Waite range reduction + 7th-order Taylor series.

5. **Pack**:
   - `tile_regs_commit()` signals that SFPU writes to DST are complete.
   - `tile_regs_wait()` waits for the packer to be ready.
   - `pack_tile(0, c_2)` triggers the packer RISC-V to read from DST register 0 and write the tile into output CB `c_2`. Data format conversion (e.g., float32 internal to bfloat16 storage) happens during pack.

6. **CB management**:
   - `cb_pop_front(c_0, 1)` frees the input CB slot.
   - `tile_regs_release()` releases the DST register.
   - After the inner loop completes, `cb_push_back(c_2, per_core_block_dim)` makes the output tiles available to the writer kernel.

#### SFPU Configuration

| Configuration | Value | Notes |
|--------------|-------|-------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity; does not affect SFPU ops directly but affects FPU operations |
| `math_approx_mode` | `false` | `get_op_approx_mode` returns false for EXP |
| `approx` template param | `true` (default from TTNN) | The TTNN `exp` op passes `param0=1.0f` -> `(uint32_t)1` -> `approx=true` |
| `fast_and_approx` | `true` (default) | Enables Schraudolph-based fast exponential |
| `fp32_dest_acc_en` | Caller-controlled | When true, uses `_sfpu_exp_f32_accurate_` (Cody-Waite); when false, uses `_sfpu_exp_21f_` |
| `ITERATIONS` | `8` (default) | 8 iterations x 4 rows = 32 rows of the 32x32 tile |
| `InputClamping` | `ClampToNegative` (default) | Clamps inputs below ~-88.5 to prevent incorrect outputs in fast-approx mode |
| `SFPU_OP_EXP_INCLUDE` | `1` | Preprocessor define that enables inclusion of `exp.h` |

#### Hardware Compatibility Notes

- **Wormhole vs Blackhole**: The arch-specific `ckernel_sfpu_exp.h` files are identical. The underlying Schraudolph approximation in the tt_llk submodule has minor differences:
  - **Blackhole** uses `ADDR_MOD_7` directly in `TTI_SFPLOADMACRO` calls and includes `ckernel_addrmod.h`.
  - **Wormhole** passes `0` for the `addr_mod` parameter in `TTI_SFPLOADMACRO`.
  - **Blackhole** additionally has a hardware-accelerated `approx_exp` intrinsic (`__builtin_rvtt_sfparecip` with `SFPARECIP_MOD1_EXP` mode) available in the SFPI library, though it is not used by the standard exp operation path.
- **SFPMAD differences**: On Wormhole, SFPMAD computes `VD = VA * VB + VC`. On Blackhole, SFPMAD supports negation modifiers (`SFPMAD_MOD1_NEGATE_VA`, `SFPMAD_MOD1_NEGATE_VC`) allowing `VD = -(VA * VB) + VC` directly. The Cody-Waite implementation in `_sfpu_exp_f32_accurate_` pre-negates constants to ensure the expression maps to a single SFPMAD on both architectures.
- **Tile size**: Both architectures use 32x32 tiles with the same SFPU lane structure (32 lanes, 8 iterations).

---

## Work Distribution

The program factory uses `split_work_to_cores(compute_with_storage_grid_size, num_pages)` to distribute tiles across the compute grid:

- **Core group 1**: Gets `num_pages_per_core_group_1` tiles each.
- **Core group 2**: Gets `num_pages_per_core_group_2` tiles each (handles remainder).
- Core iteration: Column-major `(i / num_cores_y, i % num_cores_y)`.
- Each core processes its tiles sequentially, starting from `num_pages_written` as the start offset.

For the `UnarySubCoreGridProgramFactory`, tiles must divide evenly across cores. If not, the number of cores is reduced until even division is achieved.

---

## Program Caching

The `override_runtime_arguments` method enables program caching by updating only buffer addresses without recompilation:

```cpp
void UnaryProgramFactory::override_runtime_arguments(...) {
    // Only updates arg[0] (buffer address) for both reader and writer kernels
    // Compute kernel runtime args (packed_scalar1, packed_scalar2) don't change for EXP
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        GetRuntimeArgs(program, unary_reader_kernel_id, core)[0] = src_buffer->address();
        GetRuntimeArgs(program, unary_writer_kernel_id, core)[0] = dst_buffer->address();
    }
}
```

This allows the same compiled program to be reused with different input/output tensors as long as the tensor shapes, data types, and memory layouts match.

---

## Data Type Handling

| Input dtype | CB Format | Preprocessor Define | Notes |
|------------|-----------|-------------------|-------|
| `FLOAT32` | Float32 | `INP_FLOAT32=1` | Uses `_sfpu_exp_f32_accurate_` when fp32_dest_acc_en=true |
| `BFLOAT16` | Float16_b | `INP_FLOAT=1` | Default path, uses `_sfpu_exp_21f_` |
| `INT32` | Int32 | `INP_INT32=1` | Unusual for EXP but supported by the factory |
| `UINT32` | UInt32 | `INP_UINT32=1` | Unusual for EXP but supported by the factory |

When `preserve_fp32_precision` is set, the unpack-to-dest mode is changed to `UnpackToDestFp32`, which keeps full float32 precision in DST registers throughout computation.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Queried for unary program factory structure, kernel registration patterns, circular buffer configuration, and compute kernel file locations.
- **tenstorrent/tt-llk**: Queried for the `ckernel::sfpu` namespace, `_calculate_exponential_` call chain, approximation modes (Schraudolph, piecewise), and `_init_exponential_` constant loading.
- **tenstorrent/tt-isa-documentation**: Queried for SFPU instructions (SFPMAD, SFPLOAD, SFPSTORE, SFPSETCC, SFPLUT), register file layout (32 lanes of 32-bit LRegs), and instruction encoding.
- **tenstorrent/sfpi**: Queried for SFPI programming model (vFloat, dst_reg, exexp, setexp, addexp), Blackhole `approx_exp` intrinsic, and the v_if/v_else predicated execution model.

### Confluence References

Not consulted for this analysis. The DeepWiki and source code provided sufficient SFPU instruction details.

### Glean References

Not consulted for this analysis. The DeepWiki and source code provided sufficient hardware specification details.
