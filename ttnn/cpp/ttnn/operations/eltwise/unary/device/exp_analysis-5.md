# EXP Operation Implementation Analysis

## Overview

The EXP operation computes the element-wise natural exponential (`e^x`) of each element in an input tensor. It is dispatched through the **unary program factory** (`UnaryProgramFactory::create`), which is the shared program factory for most unary SFPU operations. The EXP operation has one of the most complex SFPU kernel implementations in the codebase, with multiple algorithm paths depending on approximation mode and destination accumulator precision.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) or row (for ROW_MAJOR layout) |
| **Unit size** | 1 page (1 tile or 1 row) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` (always 1) tiles per block |

Each core processes its assigned pages one at a time. The compute kernel loops over `per_core_block_cnt` iterations, processing 1 tile per iteration (since `per_core_block_dim = 1`).

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | NHWC (standard TTNN) |
| **Tensor layout** | TILE_LAYOUT (typical) or ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or different for certain op chains) |

### Layout Transformations

No layout transformations are performed. Input and output must have the same layout (TILE or ROW_MAJOR). For TILE layout, the CB page size equals `tile_size(data_format)`. For ROW_MAJOR, it equals `buffer->page_size()`.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (via NoC) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU exp, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (via NoC) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The reader reads one page at a time from the source buffer into CB c_0. The compute kernel waits for one page, copies the tile from CB c_0 into DST registers via `copy_tile`, executes the SFPU EXP operation, packs the result to CB c_2, and pops the input. The writer waits for one page in CB c_2, writes it to the destination buffer, and pops.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Block |
| c_2 | cb_output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Block |

**Note**: CB c_1 (tmp0) is **not** allocated for EXP. It is only created for HARDSHRINK, CBRT, or LOGIT operations.

- Capacity = `num_input_tiles * page_size` = `2 * page_size` for both input and output
- Both CBs use double-buffering (capacity = 2x block size), enabling overlap between reader/compute and compute/writer

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 are double-buffered (capacity = 2 pages, block size = 1 page). This allows:
- Reader can fill the next page in c_0 while compute processes the current page
- Compute can fill the next page in c_2 while writer drains the current page

## Index Calculations

The reader and writer both use `TensorAccessor` for index-to-address translation. The accessor is constructed from `TensorAccessorArgs` which encodes the buffer's interleaved layout and bank mapping.

- **Page index**: Linear page ID from `start_id` to `start_id + num_pages`
- **Physical address**: `noc_async_read_page(page_id, accessor, l1_addr)` translates the page ID to a physical NoC address using the tensor accessor's bank mapping
- No special dimension remapping is needed; pages are processed sequentially

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in ascending order from `start_id` to `start_id + num_pages - 1`
- **Access type**: DRAM or L1 via NoC0 (reader kernel)
- **Barrier**: `noc_async_read_barrier()` after each page read ensures completion before pushing to CB

### Write Pattern
- **Sequential**: Pages are written in ascending order from `start_id` to `start_id + num_pages - 1`
- **Access type**: DRAM or L1 via NoC1 (writer kernel)
- **Flush**: `noc_async_writes_flushed()` after each page, final `noc_async_write_barrier()` at end

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g. 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages / num_cores` (approximately equal) |
| **Load balancing** | Two-group: core_group_1 gets `num_pages_per_core_group_1`, core_group_2 gets `num_pages_per_core_group_2` |

The `split_work_to_cores` function divides `num_pages` across available cores. If `num_pages` doesn't divide evenly, a second core group handles the remainder (1 fewer page per core). Cores are indexed in column-major order: `core = {i / num_cores_y, i % num_cores_y}`.

Two separate compute kernels are created if core_group_2 is non-empty, differing only in `per_core_block_cnt`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes source buffer layout (interleaved bank mapping) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes destination buffer layout |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of pages (tiles/rows) to process on this core |
| 1 | per_core_block_dim | uint32_t | Pages per block (always 1 for standard unary) |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_pages | uint32_t | Number of pages to read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of pages to write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (set to 0) |

### Preprocessor Defines (Compile-Time)

The program factory sets these defines for the compute kernel:

| Define | Value for EXP | Description |
|--------|--------------|-------------|
| `SFPU_OP_EXP_INCLUDE` | `"1"` | Enables EXP-specific includes in sfpu_split_includes.h |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro chain for the SFPU operation |
| `SFPU_OP_CHAIN_0_INIT_0` | `exp_tile_init<{fast_approx}u>();` | Init call (template param from op's param0) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `exp_tile<{fast_approx}u>(0);` | Compute call on DST tile 0 |
| `INP_FLOAT` / `INP_FLOAT32` / etc. | `"1"` | Input data type indicator |

For EXP, `param0` encodes the `fast_and_approx` flag. The default EXP operation (`softmax` path) passes `param0 = 1` (fast_and_approx = true).

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read pages via TensorAccessor |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | copy_tile, SFPU exp, pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write pages via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequentially reads pages from source buffer using `noc_async_read_page` with TensorAccessor. Supports both forward and backward iteration (via `BACKWARDS` define, unused for EXP). One page is read per iteration with a full barrier between reads.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequentially writes pages to destination buffer using `noc_async_write_page`. Supports sharded output (via `OUT_SHARDED` define, unused here). Uses `noc_async_writes_flushed()` per page and a final `noc_async_write_barrier()`.

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes exp.h when SFPU_OP_EXP_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of blocks (pages) to process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for standard unary EXP)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack (from c_0) and pack (to c_2) pipelines for SFPU
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one block (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DST register bank

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // wait until reader has pushed 1 tile into input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: exp_tile_init<1u>(); exp_tile<1u>(0);
                             // This initializes the SFPU exp pipeline and then executes exp on DST[0]
#endif

            tile_regs_commit();  // signal that DST registers are ready for packing

            tile_regs_wait();  // wait for pack pipeline to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DST register 0 into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from CB c_0

            tile_regs_release();  // release DST register bank for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // signal writer that 1 tile is ready in output CB
    }
}
```

### SFPU Kernel Implementation

The EXP SFPU implementation is one of the most complex in the codebase, with **four distinct algorithm paths** selected at compile time based on template parameters. The code is split across two layers:

1. **API layer** (`exp.h`): Provides `exp_tile_init<>()` and `exp_tile<>()` which expand LLK macros
2. **SFPU kernel layer** (`ckernel_sfpu_exp.h`): Contains the actual SFPU implementations

#### SFPU Kernel File (API Layer)
`tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`

#### SFPU Kernel File (Architecture-specific: Wormhole B0)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`

#### SFPU Kernel File (Architecture-specific: Blackhole)
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`

#### SFPU Kernel File (Shared LLK: Wormhole B0)
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`

#### Annotated API Layer Source

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/exp.h

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"          // architecture-specific SFPU exp implementations
#include "llk_math_eltwise_unary_sfpu_macros.h"  // macro dispatch infrastructure
#endif

namespace ckernel {

// Controls whether fast approximate exp clamps very negative inputs to prevent incorrect outputs
enum class InputClamping : uint8_t {
    ClampToNegative = 1,  // Inputs below ~-88.5 are clamped. Safer but slightly slower.
    None = 0,             // No clamping. Faster, but needs external ReLU for very negative inputs.
};

// Initializes SFPU pipeline for exp computation
// Template params: approx (use approx mode), fast_and_approx (use fast Schraudolph algorithm),
//                  scale (scaling constant in FP32 bits), input_clamping (clamp control)
template <
    bool approx = false,
    bool fast_and_approx = true,
    uint32_t scale = 0x3F800000,           // 1.0f in FP32
    InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    // Expands to llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, approx>(
    //     exp_init<approx, fast_and_approx, scale, (input_clamping == InputClamping::ClampToNegative)>)
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential,
        sfpu::exp_init,
        approx,
        fast_and_approx,
        scale,
        (input_clamping == InputClamping::ClampToNegative)));
}

// Executes element-wise exp on a tile in DST register buffer
// Template params: approx, fast_and_approx, scale_en, skip_positive_check, input_clamping, iterations
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8>                     // 8 iterations = 8 SFPU lanes per tile face (8 * 64 elements = 512 = 16x32)
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    // Expands to _llk_math_eltwise_unary_sfpu_params_<approx>(
    //     ckernel::sfpu::calculate_exponential<approx, fast_and_approx, DST_ACCUM_MODE,
    //         scale_en, iterations, skip_positive_check,
    //         (input_clamping == InputClamping::ClampToNegative)>,
    //     idst, vector_mode, scale)
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(
        calculate_exponential,
        approx,
        fast_and_approx,
        DST_ACCUM_MODE,     // from ComputeConfig: fp32_dest_acc_en
        scale_en,
        skip_positive_check,
        (input_clamping == InputClamping::ClampToNegative),
        iterations,
        idst,
        vector_mode,
        scale));
}

}  // namespace ckernel
```

#### Annotated Architecture-Specific SFPU Kernel Source (Wormhole B0 metal layer)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // shared LLK layer (Wormhole B0)
#include "sfpu/ckernel_sfpu_polyval.h"   // PolynomialEvaluator utility
#include "sfpu/ckernel_sfpu_converter.h" // Converter utility for bit-casting
#include "ckernel_sfpu_conversions.h"    // float_to_fp16b etc.
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Legacy _sfpu_exp_ wrapper -- delegates to shared LLK implementation
sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

// Branch-free float-to-int32 conversion optimized for exp21f algorithm
// Constraint: 0 <= val < 128.0f (assumes val already divided by 2^23)
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);        // SFPEXEXP: extract unbiased exponent
    sfpi::vInt man = sfpi::exman8(val);       // SFPEXMAN: extract mantissa with hidden bit (8-bit mode)
    // SFPSHFT via reinterpret: shift mantissa left by exponent amount
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

// exp_21f: 2nd-order polynomial approximation based on Moroz et al. 2022
// Algorithm: exp(x) = 2^(x/ln2) = 2^(x_i) * 2^(x_f)
// Uses 2nd degree polynomial for 2^(x_f) on fractional part
// Accuracy: ~21 faithful bits for bfloat16
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);   // SFPMAD: scale by 1/ln2 and add IEEE754 bias

    // Clamp to [0, 255] to prevent overflow/underflow in intermediate calculations
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);          // SFPSWAP: ensures xlog2 >= 0
    sfpi::vec_min_max(xlog2, threshold_high);          // SFPSWAP: ensures xlog2 <= 255

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2); // Convert to fixed-point integer

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXEXP: extract biased exponent = 2^(integer part)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // SFPEXMAN: extract 9-bit mantissa = fractional part

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // Convert fractional part to float

    // 2nd degree polynomial: p(x) = c0 + c1*x + c2*x^2
    // Coefficients chosen to approximate 2^x on [0, 2^23]
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: result = mantissa * 2^exponent
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);  // SFPSETEXP: set exponent field

    if constexpr (!is_fp32_dest_acc_en) {
        // For bfloat16 destination: explicitly round to avoid truncation errors from SFPSTORE
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// exp_61f: 6th-order polynomial for higher accuracy
// Same structure as exp_21f but uses degree-6 polynomial on [0, 1] range
sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    frac = sfpi::addexp(frac, -23);  // Scale fractional part to [0, 1] by multiplying by 2^-23

    // 6th degree polynomial approximation of 2^x on [0, 1]
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

// Round-to-nearest-even for float -> int32 conversion (used in Cody-Waite method)
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22 (magic constant)
    sfpi::vFloat tmp = z + c231;         // SFPMAD: add magic constant to round
    sfpi::vFloat k = tmp - c231;         // SFPMAD: subtract to get rounded float
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);  // SFPIADD: get integer value
    return k;
}

// High-accuracy exp for FP32 destination using Cody-Waite range reduction + 7th order Taylor series
// Target: < 1 ULP accuracy for float32
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;             // SFPMAD: convert to base-2 exponent

    sfpi::vInt exp_bits = sfpi::exexp(z);        // SFPEXEXP: extract exponent for range checking

    v_if(z >= OVERFLOW_THRESHOLD) {              // SFPSETCC: conditional execution
        result = std::numeric_limits<float>::infinity();  // SFPLOADI: load +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;                  // underflow to 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();  // NaN passthrough
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);  // round z to nearest int

        // Cody-Waite range reduction: r = x - k*ln(2) in extended precision
        // Split ln(2) into high and low parts to maintain precision
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;   // SFPMAD: r_hi = val + k * (-LN2_HI)
        sfpi::vFloat r = k * LN2_LO + r_hi;     // SFPMAD: r = r_hi + k * (-LN2_LO)

        // 7th order Taylor series: exp(r) ~= sum_{i=0}^{7} r^i / i!
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

        // Scale by 2^k: add k to exponent of polynomial result
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);  // SFPEXEXP: get current biased exponent
        sfpi::vInt new_exp = p_exp + k_int;           // SFPIADD: add k
        result = sfpi::setexp(p, new_exp);            // SFPSETEXP: set new exponent
    }
    v_endif;

    return result;
}

// Dispatch: selects exp_21f (bf16 dest) or exp_f32_accurate (fp32 dest)
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);     // bf16 dest: use Moroz exp_21f (2nd order polynomial)
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);   // fp32 dest: use Cody-Waite + 7th order Taylor
}

// Top-level calculate_exponential: called by exp_tile via LLK macro dispatch
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
        // Delegates to shared LLK _calculate_exponential_ which uses
        // SFPLOADMACRO-based hardware-accelerated approach or piecewise Schraudolph approximation
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximate mode: iterate over 8 SFPU lanes (each lane = 64 elements)
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];       // SFPLOAD: load from DEST register 0
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // optional input scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);  // dispatch to appropriate algorithm
            sfpi::dst_reg[0] = result;                 // SFPSTORE: write result back to DEST register 0

            sfpi::dst_reg++;                           // advance DEST pointer to next 64-element lane
        }
    }
}

// Initialize exp constants and SFPU pipeline state
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Annotated Shared LLK Layer Source (Wormhole B0)

```cpp
// tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

#pragma once
#include <cstdint>
#include <limits>
#include "ckernel_sfpu_recip.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

// Legacy precise exp using Horner-form series approximation with repeated squaring
// This is the original _sfpu_exp_ used when APPROXIMATION_MODE=false in the shared LLK layer
sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // Extract exponent; if >= 0, normalize to exponent = -1 (bias 126)
    sfpi::vInt exp = exexp(val);         // SFPEXEXP: extract unbiased exponent
    v_if (exp >= 0)                      // SFPSETCC: conditional on sign of exponent
    {
        val = setexp(val, 126);          // SFPSETEXP: set exponent to 126 (= -1 unbiased)
    }
    v_endif;

    // Horner-form 2nd degree polynomial approximation of exp on [-1, 0]
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);  // SFPMAD
    val              = val * tmp + sfpi::vConst1;                                 // SFPMAD

    // Repeated squaring: val = val^(2^(exp+1)) to reconstruct full exp range
    v_if (exp >= 0)
    {
        val = val * val;                 // SFPMUL: first squaring
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;              // SFPIADD: decrement exponent counter
            v_and(exp >= 0);            // SFPSETCC + SFPPUSHC: narrow predication
            val = val * val;            // SFPMUL: conditional squaring
        }
    }
    v_endif;

    return val;
}

// Approximate exponential body (used in fallback paths)
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;
    if constexpr (APPROXIMATION_MODE)
    {
        // Schraudolph-style bit manipulation
        constexpr int FRAC_BITS = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;

        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;  // 1/ln(2) loaded in init
        sfpi::vFloat conv = in * vConstLn2Recip;                // SFPMAD: x / ln(2)

        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73;  // SFPIADD: clear exp bits
        tmp += SP_BIAS;                                                  // SFPIADD: add bias
        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS)); // SFPSHFT: shift to exponent position
    }
    else
    {
        // Precise: compute exp(|x|) then reciprocal for negative
        out = _sfpu_exp_(sfpi::setsgn(in, 0));   // SFPSETSGN + _sfpu_exp_: exp of absolute value
        v_if (in < 0)                             // SFPSETCC
        {
            out = _sfpu_reciprocal_<2>(out);      // Newton-Raphson reciprocal (2 iterations)
        }
        v_endif;
    }
    return out;
}

// Standalone approximate exp using programmable constants
inline sfpi::vFloat _calculate_exponential_approx_(sfpi::vFloat in)
{
    sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;  // 1/ln(2), loaded during init
    sfpi::vFloat c23_73         = sfpi::vConstFloatPrgm1;  // FP conversion constant
    sfpi::vInt adj_exp          = sfpi::vConstIntPrgm2;    // exponent adjustment
    in = in * vConstLn2Recip + c23_73;                      // SFPMAD: scale and bias

    sfpi::vInt in_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(in);  // SFPIADD: adjust exponent
    in_short <<= 10 - p_exp::FRAC_BITS;                                 // SFPSHFT: move to exponent bits
    return sfpi::reinterpret<sfpi::vFloat>(in_short);
}

// Piecewise exponential with boundary checking
template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{
    sfpi::vFloat result = 0.0f;
    if constexpr (SCALE_EN) {
        in = in * sfpi::s2vFloat16b(exp_base_scale_factor);
    }
    if constexpr (APPROXIMATION_MODE)
    {
        if constexpr (!SKIP_POSITIVE_CHECK) {
            v_if (in >= 89) {
                result = std::numeric_limits<float>::infinity();  // saturate large positive
            }
            v_elseif (in < -42) {
                result = 0.0f;                                    // saturate large negative
            }
            v_else {
                result = _calculate_exponential_approx_(in);     // Schraudolph approximation
            }
            v_endif;
        } else {
            v_if (in < -42) { result = 0.0f; }
            v_else { result = _calculate_exponential_approx_(in); }
            v_endif;
        }
    }
    else
    {
        result = _sfpu_exp_(sfpi::setsgn(in, 0));     // exp of absolute value
        v_if (in < 0) {
            result = _sfpu_reciprocal_<2>(result);     // reciprocal for negative inputs
        }
        v_endif;
    }
    return result;
}

// Main dispatch for _calculate_exponential_ -- shared LLK entry point
// This is called when APPROXIMATION_MODE=true from the metal layer's calculate_exponential
template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // PATH 1: SFPLOADMACRO-based hardware-accelerated exp
        // Two-phase approach:
        // Phase 1 (Sanitize): LOADMACRO Sequence 1 loads values from DEST,
        //   SWAPs against -88.5 threshold, stores sanitized values back
        // Phase 2 (Compute): LOADMACRO Sequence 0 loads sanitized values,
        //   MAD computes A*y + (B-C), ROUND converts to int16, SHIFT moves to exponent,
        //   STORE writes result back to DEST
        // Processes all 16 dest offsets (8 pairs of even/odd columns for rows 0-15)

        // Phase 1: Sanitize (clamp to >= -88.5)
        TTI_SFPLOADMACRO(4, 0, 3, 0);   // Seq1, LREG0, dest_offset=0 (even cols, rows 0-3)
        TTI_SFPNOP;                       // SWAP takes 2 cycles, needs NOP
        TTI_SFPLOADMACRO(5, 0, 3, 2);   // Seq1, LREG1, dest_offset=2 (odd cols, rows 0-3)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 4);   // Seq1, LREG2, dest_offset=4 (even cols, rows 4-7)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 6);   // Seq1, LREG3, dest_offset=6 (odd cols, rows 4-7)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, 3, 8);   // Seq1, LREG0, dest_offset=8 (even cols, rows 8-11)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, 3, 10);  // Seq1, LREG1, dest_offset=10 (odd cols, rows 8-11)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, 3, 12);  // Seq1, LREG2, dest_offset=12 (even cols, rows 12-15)
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, 3, 14);  // Seq1, LREG3, dest_offset=14 (odd cols, rows 12-15)

        // Phase 2: Compute (Schraudolph exp via LOADMACRO Sequence 0)
        TTI_SFPLOADMACRO(0, 0, 3, 0);   // Seq0, LREG0: LD->MAD->ROUND->SHIFT->STORE
        TTI_SFPLOADMACRO(1, 0, 3, 2);   // Seq0, LREG1
        TTI_SFPLOADMACRO(2, 0, 3, 4);   // Seq0, LREG2
        TTI_SFPLOADMACRO(3, 0, 3, 6);   // Seq0, LREG3
        TTI_SFPLOADMACRO(0, 0, 3, 8);   // Seq0, LREG0
        TTI_SFPLOADMACRO(1, 0, 3, 10);  // Seq0, LREG1
        TTI_SFPLOADMACRO(2, 0, 3, 12);  // Seq0, LREG2
        TTI_SFPLOADMACRO(3, 0, 3, 14);  // Seq0, LREG3
        TTI_SFPNOP;                       // Pipeline drain
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8)
    {
        // PATH 2: Replay-buffer based exp (8-element, ~2.5 cycles/element)
        // Uses LOADMACRO + SFPSHFT2 pairs recorded in replay buffer during init
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},           // auto-increment dest by 2 per LOADMACRO
        }.set(ADDR_MOD_7);

        lltt::replay(0, 16);               // replay 16 recorded instructions (8 LM + 8 SHFT2)

        // Drain final 2 SFPSHFT2 operations
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 32)
    {
        // PATH 3: Replay-buffer based exp (32-element, ~2.125 cycles/element)
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
        // PATH 4: Software loop using piecewise or precise exp
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vFloat in     = sfpi::dst_reg[0];     // SFPLOAD: read from DEST
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(in, exp_base_scale_factor);
            sfpi::dst_reg[0]    = result;                // SFPSTORE: write back to DEST
            sfpi::dst_reg++;                             // advance DEST pointer
        }
    }
}

// Initialization: loads constants, programs LOADMACRO registers, records replay buffers
// This is a very large function -- see source for complete LOADMACRO setup
template <bool APPROXIMATION_MODE, bool FAST_APPROX, std::uint32_t scale, bool CLAMP_NEGATIVE = true>
inline void _init_exponential_()
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // Sets up LOADMACRO infrastructure for hardware-accelerated Schraudolph exp:
        // - Loads constants: LREG[14]=-88.5 (threshold), LREG[12]=A=256/ln2, LREG[13]=B-C=32500.8
        // - Programs macro instruction registers via SFPCONFIG and backdoor loads:
        //   Macro 4: SFPSWAP (sanitize: clamp to >= -88.5)
        //   Macro 5: SFPMAD (compute: A * y + (B-C))
        //   Macro 6: SFP_STOCH_RND (round FP32 to uint16)
        //   Macro 7: SFPSHFT (shift left by 15 to place in exponent field)
        // - Programs sequence register 1 (sanitize: LD -> SWAP -> STORE)
        // - Programs sequence register 0 (compute: LD -> MAD -> ROUND -> SHIFT -> STORE)
        // - Resets LoadMacroConfig misc register
        // [Full source omitted for brevity -- see shared LLK file]
        // ...
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE)
    {
        // Replay-buffer based init (no clamping):
        // - Loads constants into LREG[12], LREG[13], LREG[14]
        // - Programs macro instructions 5 (MAD), 6 (STOCHRND), 7 (SETSGN) via backdoor
        // - Programs macro sequence register 0 with SETSGN+MAD+STOCHRND+STORE timing
        // - Records 32 instructions into replay buffer (16 LOADMACRO+SFPSHFT2 pairs)
        // [Full source omitted for brevity -- see shared LLK file]
        // ...
    }
    else if constexpr (APPROXIMATION_MODE)
    {
        // Simple approximate mode: load programmable constants
        sfpi::vConstFloatPrgm0 = 1.442695f;                  // 1/ln(2)
        sfpi::vConstFloatPrgm1 = sfpi::s2vFloat16b(p_exp::C23_73);  // conversion constant
        sfpi::vConstFloatPrgm2 = sfpi::s2vFloat16b(p_exp::ADJ_EXP); // exponent adjustment
    }
    else
    {
        // Precise mode: initialize reciprocal (needed for negative inputs)
        _init_sfpu_reciprocal_<false>();
    }
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction | Opcode | Description | Used In |
|-------------|--------|-------------|---------|
| **SFPEXEXP** | 0x77 | Extracts 8-bit exponent field from FP32 register. Can return biased or unbiased exponent. Updates CC based on sign. | `exexp()`, `exexp_nodebias()` -- used in all exp paths for range decomposition |
| **SFPSETEXP** | 0x82 | Sets the exponent field of a float from a register or immediate. Preserves sign and mantissa from source. | `setexp()` -- recombines integer/fractional parts in exp_21f, exp_f32_accurate |
| **SFPEXMAN** | 0x78 | Extracts mantissa field (with or without hidden bit). | `exman8()`, `exman9()` -- extracts fractional part in exp_21f |
| **SFPMAD** | 0x84 | Fused multiply-add: `(A * B) + C`. Core operation for polynomial evaluation and scaling. Latency: 2 cycles. | Used extensively in polynomial evaluation (Horner form), Cody-Waite reduction, LOADMACRO compute |
| **SFPMUL** | - | Floating-point multiply. | Repeated squaring in `_sfpu_exp_()`, polynomial evaluation |
| **SFPIADD** | - | Integer add/subtract on register values. | Exponent manipulation, loop counters in `_sfpu_exp_()` |
| **SFPSHFT** | - | Shift integer value left/right by immediate or register amount. | Bit manipulation in Schraudolph approximation, exponent positioning |
| **SFPSHFT2** | - | Two-operand shift with configurable mode. Mode 5: shift by VC register amount. Latency: 4 cycles for global modes. | Replay-buffer based fast exp (shift int16 result to exponent position) |
| **SFPSETSGN** | 0x89 | Sets sign bit of FP32 from register or immediate. Preserves exponent and mantissa. | `setsgn()` -- force positive for precise exp, restore sign in non-clamping fast path |
| **SFPLOADI** | - | Loads immediate value into lower or upper 16 bits of LREG[0]. | Loading constants during init (threshold, A, B-C coefficients) |
| **SFPCONFIG** | - | Configures SFPU shared state: constant registers (LREG[12-15]), macro instruction registers, macro sequence registers, and control registers. | Programming LOADMACRO infrastructure during init |
| **SFPLOADMACRO** | - | Composite instruction that executes a configurable sequence of up to 5 operations (Load, Simple, MAD, Round, Store) in a pipelined manner on DEST data. | Hardware-accelerated exp compute and sanitize phases |
| **SFPSWAP** | - | Conditional swap: places the larger of two values into the destination register. Latency: 4 cycles. | Clamping input values to >= -88.5 threshold (sanitize phase) |
| **SFP_STOCH_RND** | - | Stochastic rounding: converts FP32 to integer format (int16/uint16). Multiple modes for signed/unsigned output. | Converting MAD result to int16 for exponent construction |
| **SFPSTORE** | - | Stores LREG value back to DEST register file. | Writing exp results back to DEST (via LOADMACRO sequences) |
| **SFPNOP** | - | No-operation. Required for pipeline spacing between certain SFPU instructions. | Spacing between SFPSWAP pairs, SFPSHFT2 pairs, and LOADMACRO drain |
| **SFPPUSHC/SFPSETCC/SFPPOPC** | - | Condition code stack manipulation for conditional execution (`v_if`/`v_endif`). | All conditional paths (range checking, sign handling) |

#### SFPU Register Usage

| Register(s) | Type | Usage |
|-------------|------|-------|
| **LREG[0-3]** | Working | Loaded from DEST by LOADMACRO; used as intermediate computation registers |
| **LREG[4]** | Working | SFPSHFT2 output destination in replay-buffer path |
| **LREG[12]** | Constant | `A = 256/ln(2) = 369.33` (Schraudolph scaling factor) |
| **LREG[13]** | Constant | `B - C = 32500.818` (Schraudolph bias minus error correction) |
| **LREG[14]** | Constant | `-88.5` (clamping threshold) or `15` (shift amount for SFPSHFT2) |
| **LREG[16]** | Staging | Used by LOADMACRO sequence to avoid write-port conflicts in non-clamping path |
| **DEST[0-15]** | Destination | 16 addressable offsets in DEST register file, each holding 64 elements. Tile face processed by iterating over all 16 offsets. |
| **dst_reg[0]** | DEST accessor | SFPI interface to current DEST offset; incremented via `dst_reg++` |
| **vConstFloatPrgm0** | Programmable const | `1/ln(2)` for approximate mode |
| **vConstFloatPrgm1** | Programmable const | `C23_73` conversion constant for approximate mode |
| **vConstIntPrgm2** | Programmable const | `ADJ_EXP` exponent adjustment for approximate mode |
| **CC (Condition Code)** | Control | Used for conditional execution via `v_if`/`v_and`/`v_endif` |

#### SFPU Execution Flow

**Default path (APPROXIMATION_MODE=false, the standard non-approximate path):**

1. **Init** (`exp_init` -> `_init_exponential_<false, ...>`): Initializes the reciprocal function (needed for negative input handling)
2. **Per-tile execution** (`calculate_exponential<false, true, is_fp32_dest_acc_en>`):
   - Loop over 8 iterations (one per DEST lane, each lane = 64 elements):
     a. Load value from `dst_reg[0]` (SFPLOAD from current DEST offset)
     b. Call `_sfpu_exp_improved_<is_fp32_dest_acc_en>(val)`:
        - If **bf16 dest** (`is_fp32_dest_acc_en=false`): Uses `_sfpu_exp_21f_` (Moroz 2022):
          - Scale input by 1/ln(2) and add IEEE bias 127 -> `xlog2`
          - Clamp xlog2 to [0, 255] using `vec_min_max` (SFPSWAP)
          - Convert to fixed-point integer
          - Decompose into integer exponent part and fractional mantissa
          - Evaluate 2nd-degree polynomial on fractional part
          - Recombine with `setexp` (SFPSETEXP)
          - Round to bf16 with `float_to_fp16b`
        - If **fp32 dest** (`is_fp32_dest_acc_en=true`): Uses `_sfpu_exp_f32_accurate_`:
          - Scale by 1/ln(2) to get z
          - Check overflow (z >= 128) -> infinity
          - Check underflow (z <= -127) -> 0
          - Check NaN (exp == 255) -> NaN
          - Round z to nearest integer k using magic-constant method
          - Cody-Waite range reduction: r = x - k*LN2_HI - k*LN2_LO
          - 7th-order Taylor polynomial for exp(r)
          - Scale by 2^k via exponent addition (SFPEXEXP + SFPIADD + SFPSETEXP)
     c. Store result to `dst_reg[0]` (SFPSTORE to current DEST offset)
     d. Advance DEST pointer: `dst_reg++`

**Fast approximate path (APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=true):**

1. **Init** (`_init_exponential_<true, true, scale, true>`):
   - Load constants into LREG[12-14]: A, B-C, threshold
   - Program 4 macro instruction registers via SFPCONFIG and backdoor loads
   - Program 2 macro sequence registers (sanitize and compute sequences)
   - Reset LOADMACRO config misc
2. **Per-tile execution** (`_calculate_exponential_`):
   - Phase 1 (Sanitize): 8 SFPLOADMACRO instructions with Sequence 1 (LD -> SWAP -> STORE). Each clamps one DEST offset against -88.5 threshold, with SFPNOP spacing for SWAP latency.
   - Phase 2 (Compute): 8 SFPLOADMACRO instructions with Sequence 0 (LD -> MAD -> ROUND -> SHIFT -> STORE). Each computes `result = shift_left(round(A * x + (B-C)), 15)` which creates an IEEE 754 float whose bit pattern represents exp(x).

**Fast approximate path without clamping (APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=false):**

1. **Init**: Same as clamping path but also programs SFPSETSGN macro, SFPSHFT2 shift amount in LREG[14]=15, and records a 32-instruction replay buffer containing interleaved LOADMACRO + SFPSHFT2 pairs.
2. **Per-tile execution**:
   - For 8 iterations: 1 replay of 16 instructions + 2 drain SFPSHFT2
   - For 32 iterations: 2 replays of 32 instructions + 2 drain SFPSHFT2
   - Uses ADDR_MOD_7 for auto-increment of DEST offset by 2 per LOADMACRO

#### SFPU Configuration

| Configuration | Default EXP Value | Description |
|--------------|-------------------|-------------|
| **math_approx_mode** | `false` | `get_op_approx_mode(EXP)` returns false by default |
| **fast_and_approx** | `true` (template default) | Second template parameter of `exp_tile` |
| **MathFidelity** | `HiFi4` | Highest fidelity mode set in ComputeConfig |
| **fp32_dest_acc_en** | From `args.fp32_dest_acc_en` | Controls whether DEST accumulator uses FP32 precision |
| **unpack_to_dest_mode** | `Default` or `UnpackToDestFp32` | FP32 mode if `preserve_fp32_precision` is set |
| **DST_ACCUM_MODE** | Compile-time from ComputeConfig | Selects between exp_21f (bf16) and exp_f32_accurate (fp32) algorithms |
| **iterations** | 8 | Default: 8 lanes per tile face (8 * 64 = 512 elements = 16x32 half-tile) |

When EXP is called with a parameter (e.g., from softmax), `param0` controls `fast_and_approx`. The define becomes `exp_tile_init<{param0}u>()` and `exp_tile<{param0}u>(0)`.

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of the shared LLK layer (`ckernel_sfpu_exp.h`) are **functionally identical**. Both contain the same:
- `_sfpu_exp_()` legacy precise function
- `_calculate_exponential_body_()`, `_calculate_exponential_approx_()`, `_calculate_exponential_piecewise_()`
- `_calculate_exponential_()` with 4 paths (LOADMACRO clamping, replay 8, replay 32, software loop)
- `_init_exponential_()` with full LOADMACRO setup

Minor differences:
- Blackhole uses `ADDR_MOD_7` symbolic constant for SFPLOADMACRO address modifier fields, while Wormhole uses the literal `3`
- Both support the same 4 algorithm paths and the same constants

The **metal layer** (`ckernel_sfpu_exp.h` under `hw/ckernels/{arch}/`) is also identical between WH and BH, providing the same `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and `calculate_exponential` dispatcher. The `_sfpu_exp_improved_` template specialization selects `_sfpu_exp_21f_` for bf16 and `_sfpu_exp_f32_accurate_` for fp32 on both architectures.

## Implementation Notes

1. **Algorithm selection hierarchy**: The EXP operation has a 4-level dispatch:
   - `APPROXIMATION_MODE=false` (default): Metal layer's `calculate_exponential` directly loops over DEST lanes calling `_sfpu_exp_improved_` (exp_21f or exp_f32_accurate)
   - `APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=true`: LOADMACRO-based hardware pipeline with input sanitization
   - `APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=false`: Replay-buffer based pipeline with SFPSETSGN sign restoration
   - `APPROXIMATION_MODE=true, FAST_APPROX=false`: Software loop with piecewise boundary checking

2. **Moroz et al. 2022 reference**: The `exp_21f` and `exp_61f` algorithms are based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]" (https://doi.org/10.1109/MSP.2022.3157460), Section 5.

3. **Schraudolph reference**: The fast approximate mode is based on "A Fast, Compact Approximation of the Exponential Function" by Nicol N. Schraudolph, which exploits the linear relationship between IEEE 754 bit patterns and logarithmic values.

4. **LOADMACRO pipelining**: The hardware-accelerated path is highly optimized. Each SFPLOADMACRO instruction triggers a 5-stage sequence (Load, Simple/MAD, Round, Shift, Store) that executes across multiple SFPU functional units in parallel. The sequence register encoding packs 4 unit configurations into 32 bits, with each 8-bit field specifying the macro instruction to execute, pipeline delay, and source/destination overrides.

5. **Replay buffer optimization**: For the non-clamping fast path, a 32-instruction pattern is recorded once during init and replayed during compute. This avoids instruction fetch overhead and achieves ~2.125-2.5 cycles per element.

6. **Double-buffering strategy**: Both CB c_0 and c_2 use 2-page capacity with 1-page block size. This allows the reader to pre-fetch the next tile while compute processes the current one, and compute to produce the next result while the writer drains the current one.

7. **Cody-Waite range reduction**: The fp32-accurate path uses extended-precision subtraction by splitting ln(2) into high and low parts. This maintains full float32 precision through the range reduction step, enabling < 1 ULP accuracy in the final result.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary_program_factory.cpp work for SFPU operations like exp?"
   **Reason**: Needed to understand the overall program factory structure and kernel dispatch
   **Key Findings**: Confirmed reader/writer/compute kernel paths, CB configuration (c_0, c_1, c_2), `split_work_to_cores` for core distribution, and double-buffering with 2 tiles

2. **Query**: "How does SFPI implement exponential (exp)? What instructions and registers are used?"
   **Reason**: Needed to understand the SFPU-level implementation details
   **Key Findings**: Identified the four algorithm paths (high-precision series, fast bit-manipulation), the `vFloat`/`vInt`/`dst_reg` register model, key SFPU instructions (SFPEXEXP, SFPSETEXP, SFPMAD, SFPSHFT, SFPSETSGN), and the iteration model (8 iterations over 64-element lanes)

### Confluence References

1. **Source**: Tensix SFPU Instruction Set Architecture (page 1170505767)
   **Sections consulted**: SFPEXEXP, SFPSETEXP, SFPMAD, SFPSETSGN, SFPLOADMACRO Registers
   **Key Information**:
   - SFPEXEXP (opcode 0x77): Extracts biased or unbiased exponent, IPC=1, latency=1, sets CC.Res based on sign
   - SFPSETEXP (opcode 0x82): Sets exponent from register bits [7:0] or immediate, preserves sign/mantissa from VC
   - SFPMAD (opcode 0x84): Fused multiply-add with source select and sign inversion modes, IPC=1, latency=2, flushes subnormals
   - SFPSETSGN (opcode 0x89): Sets sign from VD or immediate, IPC=1, latency=1
   - LOADMACRO Sequence Registers: 4 configurable slots (Simple, MAD, Round, Store) with per-slot instruction select, delay, staging register, and source override bits

### Glean References

Not required for this analysis -- the Confluence SFPU ISA page and DeepWiki provided sufficient detail for all SFPU instructions used.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Understand the API-level template parameters and documentation
   **Key Information**: Template parameters for `exp_tile` (approx, fast_and_approx, scale_en, skip_positive_check, input_clamping, iterations), default values, and the `InputClamping` enum

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Understand how EXP is mapped to compute kernel path and defines
   **Key Information**: EXP uses `eltwise_sfpu.cpp` (default path), macro definition is `SFPU_OP_EXP_INCLUDE`, `get_op_approx_mode` returns `false` for EXP, the SFPU_OP_CHAIN defines expand to `exp_tile_init<param0u>(); exp_tile<param0u>(0);`

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Understand how `SFPU_TEMPLATE_INIT_KERNEL` and `SFPU_TEMPLATE_PARAMS_KERNEL_FN` dispatch to LLK functions
   **Key Information**: `SFPU_TEMPLATE_INIT_KERNEL` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROX>(INIT_CB<...>)`, `SFPU_TEMPLATE_PARAMS_KERNEL_FN` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::FN<...>, DST_IDX, VECTOR_MODE, SCALE)`
