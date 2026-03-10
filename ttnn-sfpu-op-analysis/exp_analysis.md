# EXP (Exponential) Implementation Analysis

## Overview

The EXP operation computes the element-wise exponential function `exp(x)` for each element of the input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which provides a generic framework for all unary element-wise operations. The EXP-specific behavior is injected via preprocessor defines (`SFPU_OP_EXP_INCLUDE`, `SFPU_OP_CHAIN_0`) that select the exponential compute kernel path and SFPU kernel function at compile time.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block. For the standard factory, `per_core_block_dim = 1`, so the inner loop processes 1 tile at a time. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (flattened to total tile count) |
| **Dimension convention** | N/A (treated as flat sequence of tiles) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or potentially different if output dtype is configured) |

### Layout Transformations

No explicit tilize/untilize or reshard operations are performed within the program factory. The input is expected to already be in TILE_LAYOUT, and the output is written in TILE_LAYOUT. The data format conversion between input and output (if dtypes differ) is handled implicitly by the unpack and pack stages.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (RISCV_0) | DRAM/L1 (src_buffer) | CB c_0 (input) | `cb_reserve_back(c_0, 1)`, NoC async read, `cb_push_back(c_0, 1)` |
| 2 | Compute (RISCV_2) | CB c_0 (input) | CB c_2 (output) | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU exp, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer (RISCV_1) | CB c_2 (output) | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, NoC async write, `cb_pop_front(c_2, 1)` |

Step-by-step flow:
1. The **reader kernel** iterates over its assigned tile range (`start_id` to `start_id + num_pages`). For each tile, it reserves space in CB c_0, issues an asynchronous NoC read from the source buffer, waits for the read to complete, and pushes the tile into CB c_0.
2. The **compute kernel** iterates over blocks. For each block, it reserves output space in CB c_2 for `per_core_block_dim` tiles (which is 1). For each tile in the block, it acquires tile registers, waits for a tile in CB c_0, copies the tile from CB c_0 into DEST registers via `copy_tile`, executes `exp_tile_init()` and `exp_tile(0)` on DEST register 0, commits the tile registers, waits for them, packs DEST[0] into CB c_2, pops the input tile from CB c_0, and releases the tile registers. After the inner loop, it pushes the completed tiles in CB c_2.
3. The **writer kernel** iterates over its assigned tile range. For each tile, it waits for a tile in CB c_2, reads the tile address from CB c_2, issues an asynchronous NoC write to the destination buffer, flushes writes, and pops the tile from CB c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- `num_input_tiles = 2` and `num_output_tiles = 2` are hardcoded in the program factory, giving double-buffered CBs for both input and output.
- CB c_1 (temporary buffer) is NOT created for the EXP operation. It is only created for HARDSHRINK, CBRT, or LOGIT operations.
- The page size for each CB is determined by the tile size of its respective data format.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity of 2 tiles with a block size of 1 tile, providing **double-buffering**. This means the reader can fill one tile slot in CB c_0 while the compute kernel processes the other, and similarly the compute kernel can write to one slot in CB c_2 while the writer drains the other. This enables overlap between all three pipeline stages.

## Index Calculations

The program factory uses `TensorAccessor` for index-to-address mapping. The reader and writer kernels both use a linear tile index (`i`) that ranges from `start_id` to `start_id + num_pages`. The `TensorAccessor` object (constructed from `TensorAccessorArgs`) translates this linear page index into the physical DRAM/L1 address for interleaved memory layout. For interleaved buffers, this involves computing: `bank_address = buffer_base + page_index * page_size` with bank interleaving across DRAM channels.

## Memory Access Patterns

### Read Pattern
**Sequential**: The reader kernel reads tiles in strictly sequential order from `start_id` to `end_id`. Each tile is read via a single `noc_async_read_page` call with an immediate barrier (`noc_async_read_barrier`), meaning tiles are read one at a time with no pipelining of NoC reads within the reader.

### Write Pattern
**Sequential**: The writer kernel writes tiles in strictly sequential order from `start_id` to `end_id`. Each tile is written via `noc_async_write_page` followed by `noc_async_writes_flushed`, with a final `noc_async_write_barrier` after all tiles. Writes are one at a time with flush-per-tile semantics.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D iteration via column-major indexing) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Determined by `split_work_to_cores` based on total tiles |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

The `split_work_to_cores` utility divides the total number of tiles across the available compute grid. If the tiles do not divide evenly, two core groups are created: `core_group_1` processes one more tile per core than `core_group_2`. Each core group gets its own compute kernel instance with a different `per_core_block_cnt` compile-time argument. The reader and writer kernels are shared across all cores with per-core runtime arguments.

Core iteration order is column-major: `core = {i / num_cores_y, i % num_cores_y}`.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Buffer descriptor for source tensor (address mode, bank info, page size) |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Buffer descriptor for destination tensor |

**Compute Kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for standard factory) |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles to read |
| 2 | start_id | uint32_t | Starting tile index for this core |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index for this core |

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 src_buffer | CB c_0 | Sequential tile read via TensorAccessor |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | copy_tile + SFPU exp_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 dst_buffer | Sequential tile write via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Uses `TensorAccessor` for address calculation. Reads one tile at a time in a tight loop with `noc_async_read_page` + barrier. Supports optional `BACKWARDS` mode for reverse iteration (not used by default for EXP).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Uses `TensorAccessor` for address calculation. Writes one tile at a time with `noc_async_write_page` + flush. Supports `OUT_SHARDED` mode (simple `cb_wait_front` of all pages, no write loop) and `BACKWARDS` mode. Neither is used by default for EXP.

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
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of blocks (= number of tiles for standard factory)
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (= 1 for standard factory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack (from c_0) and pack (to c_2) hardware for SFPU operation
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve output space for per_core_block_dim tiles in CB c_2
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for this tile operation

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // block until reader has produced 1 tile in CB c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DEST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: exp_tile_init(); exp_tile(0);
            // exp_tile_init() configures SFPU pipeline for exponential computation (calls exp_init -> _init_exponential_)
            // exp_tile(0) dispatches SFPU exp on tile in DEST[0]; calls calculate_exponential which iterates over 8 faces
#endif

            tile_regs_commit();  // signal that DEST registers have been written and are ready for pack

            tile_regs_wait();  // wait for pack to be ready to consume DEST registers

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] result into CB c_2 output buffer

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from CB c_0

            tile_regs_release();  // release DEST register lock so next iteration can acquire
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish per_core_block_dim tiles to writer via CB c_2
    }
}
```

### SFPU Kernel Implementation

The EXP operation has a multi-level dispatch chain. The compute kernel calls `exp_tile(0)`, which invokes `calculate_exponential` in the architecture-specific ckernel layer, which then dispatches to one of several SFPU kernel implementations depending on template parameters.

#### SFPU Kernel File

Two layers of SFPU kernel files:
1. **API layer**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` -- defines `exp_tile()` and `exp_tile_init()`
2. **Architecture-specific LLK layer** (Blackhole): `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` -- defines `calculate_exponential`, `_sfpu_exp_improved_`, `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`
3. **Lower-level LLK layer** (tt_llk): `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` -- defines `_calculate_exponential_`, `_init_exponential_`, `_sfpu_exp_`, `_calculate_exponential_approx_`, `_calculate_exponential_piecewise_`

#### Annotated SFPU Kernel Source (Architecture-specific layer -- Blackhole)

```cpp
// From: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // lower-level LLK layer with _calculate_exponential_, _init_exponential_
#include "sfpu/ckernel_sfpu_polyval.h"    // PolynomialEvaluator for polynomial approximations
#include "sfpu/ckernel_sfpu_converter.h"  // Converter utilities for bit-level float manipulation
#include "ckernel_sfpu_conversions.h"     // float_to_fp16b and related conversion functions
#include "sfpi.h"                         // SFPI programming interface: vFloat, vInt, dst_reg, etc.

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }
// Wrapper around the legacy _sfpu_exp_ function from the lower-level LLK layer.
// This uses a Horner-form polynomial series with iterative squaring.

/*
 * Branch-free float-to-int32 conversion optimized for exp21f algorithm.
 * Constraint: 0 <= val < 128.0f
 * The value is assumed to already be divided by 2^23, so the output is
 * scaled by 2^23 relative to the logical value. This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);       // extract biased exponent field
    sfpi::vInt man = sfpi::exman8(val);       // extract mantissa with implicit leading 1 bit (8-bit extraction)
    // shift mantissa left by exponent amount: this converts the float to a fixed-point integer
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

/*
 * exp_21f algorithm from Moroz et al. 2022.
 * Computes exp(x) = 2^(x/ln2) by separating into integer and fractional parts:
 *   exp(x) = 2^(x_i) * 2^(x_f)
 * Uses a 2nd-degree polynomial to approximate 2^(x_f).
 * This is the default path when fp32_dest_acc is disabled (bfloat16 mode).
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;  // 1/ln(2)
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);      // scale to base-2 and add IEEE754 bias

    // Clamp xlog2 to [0, 255] to prevent overflow/underflow in intermediate computations
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);  // xlog2 = max(xlog2, 0)
    sfpi::vec_min_max(xlog2, threshold_high);  // xlog2 = min(xlog2, 255)

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);  // convert biased result to integer (scaled by 2^23)

    // Separate integer (exponent) and fractional (mantissa) parts of the biased value
    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // extract exponent without debiasing = 2^(integer part)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));     // extract 9-bit mantissa = fractional part in [0, 1)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // convert fractional bits to float

    // 2nd-degree polynomial approximation of 2^(x_f) over [0, 2^23]
    // Coefficients chosen to minimize error over the valid range
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Reconstruct result: set exponent of frac to exponential_part = 2^(x_i) * 2^(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // When DEST is bfloat16, explicitly round to bfloat16 using round-to-nearest-even
        // to avoid truncation artifacts from SFPSTORE
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

/*
 * exp_61f algorithm from Moroz et al. 2022.
 * Higher accuracy than exp_21f, using a 6th-degree polynomial for 2^(x_f).
 * The fractional part is first scaled by 2^-23 to normalize to [0, 1].
 */
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
    frac = sfpi::addexp(frac, -23);  // scale fractional part by 2^-23 to normalize to [0, 1]

    // 6th-degree polynomial approximation of 2^x on [0, 1]
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

/*
 * Round-to-nearest-even using Hacker's Delight method.
 * Returns both the rounded float and the integer value.
 */
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22 (magic constant)
    sfpi::vFloat tmp = z + c231;        // adding magic constant forces rounding
    sfpi::vFloat k = tmp - c231;        // subtract to get rounded float
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);  // integer result
    return k;
}

/*
 * High-accuracy exp(x) using Cody-Waite range reduction.
 * Target: < 1 ULP accuracy for float32.
 * Used when fp32_dest_acc_en = true.
 */
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;    // ~89 in original x domain
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;  // ~-88 in original x domain

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;  // convert to base-2

    sfpi::vInt exp_bits = sfpi::exexp(z);  // extract exponent for NaN detection

    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();  // saturate to +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;  // saturate to 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();  // propagate NaN
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);  // k = round(x/ln2)

        // Cody-Waite range reduction: r = x - k*ln2 in extended precision
        // ln(2) split into high and low parts to minimize cancellation error
        // Constants are negated so compiler can optimize to single SFPMAD instructions
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;   // first reduction step (exact for small k)
        sfpi::vFloat r = k * LN2_LO + r_hi;     // second reduction step (adds low-order correction)

        // 7th-order Taylor series for exp(r), where |r| < ln(2)/2
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,    // 1
            sfpi::vConst1,    // 1
            0.5f,             // 1/2!
            1.0f / 6.0f,     // 1/3!
            1.0f / 24.0f,    // 1/4!
            1.0f / 120.0f,   // 1/5!
            1.0f / 720.0f,   // 1/6!
            1.0f / 5040.0f   // 1/7!
        );

        // Scale by 2^k: add k to the exponent of the polynomial result
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

// Template dispatch: selects exp_21f for bfloat16 mode, exp_f32_accurate for fp32 mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);  // bfloat16 path: fast 2nd-degree polynomial approximation
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // fp32 path: Cody-Waite + 7th-order Taylor series
}

/*
 * Main entry point called by exp_tile() via SFPU_TEMPLATE_PARAMS_KERNEL_FN macro.
 * Dispatches to either the approximate path (_calculate_exponential_ from LLK layer)
 * or the improved path (_sfpu_exp_improved_) based on APPROXIMATION_MODE.
 * Default: APPROXIMATION_MODE=false, so the improved path is used.
 */
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,          // 8 iterations = 8 sub-tile faces (4 rows x 2 columns per face = 32x32 tile)
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Approximate mode: dispatches to lower-level LLK _calculate_exponential_ which uses
        // Schraudolph's fast exp algorithm with SFPLOADMACRO-based execution
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximate (improved) mode: iterates over ITERATIONS faces of the tile
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // load current face element from DEST register 0
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // optional pre-scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);  // compute exp
            sfpi::dst_reg[0] = result;  // write result back to DEST register 0
            sfpi::dst_reg++;  // advance to next face (each face = 32 elements in SFPU lane)
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
    // For non-approximate mode: initializes SFPU reciprocal tables (used by _sfpu_exp_ fallback)
    // For approximate mode with FAST_APPROX: loads Schraudolph constants into LREGs,
    //   programs macro instruction registers, and records replay buffer patterns
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Annotated SFPU Kernel Source (Lower-level LLK layer -- key functions)

```cpp
// From: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h

namespace ckernel::sfpu {

/*
 * Legacy exponential approximation using Horner-form polynomial with iterative squaring.
 * Used by _calculate_exponential_body_<false> and _calculate_exponential_piecewise_<false>.
 * Algorithm:
 *   1. If exponent >= 0, clamp exponent to -1 (reduce range to [-1, 0))
 *   2. Evaluate 2nd-degree Horner polynomial: val = val*(val*0.8373 + 0.863281) + 1
 *   3. For original exponent >= 0, iteratively square the result (up to 8 times)
 *      using predicated narrowing to handle varying exponent magnitudes
 */
sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val) {
    sfpi::vInt exp = exexp(val);       // extract biased exponent
    v_if (exp >= 0) {
        val = setexp(val, 126);        // force exponent to -1 (val in [-1, 0) range)
    }
    v_endif;

    // Horner polynomial for exp(x) on [-1, 0): exp(x) ~ x*(x*0.8373 + 0.863281) + 1
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val = val * tmp + sfpi::vConst1;

    // Repeated squaring to reconstruct exp for larger exponents
    v_if (exp >= 0) {
        val = val * val;               // first squaring (accounts for exponent bit 0)
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            v_and(exp >= 0);           // narrow predication: only continue for lanes with remaining exponent
            val = val * val;           // subsequent squarings
        }
    }
    v_endif;

    return val;
}

/*
 * Fast approximate exponential using Schraudolph's algorithm.
 * Exploits IEEE 754 bit layout: reinterpreting exp(x) as integer manipulation.
 * i = A*x + (B-C) interpreted as float gives exp(x).
 */
inline sfpi::vFloat _calculate_exponential_approx_(sfpi::vFloat in) {
    sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;  // 1/ln(2) loaded during init
    sfpi::vFloat c23_73 = sfpi::vConstFloatPrgm1;          // bias constant loaded during init
    sfpi::vInt adj_exp = sfpi::vConstIntPrgm2;              // adjustment loaded during init
    in = in * vConstLn2Recip + c23_73;                      // scale and bias

    sfpi::vInt in_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(in);  // integer-domain bias adjust
    in_short <<= 10 - p_exp::FRAC_BITS;                    // shift to IEEE exponent position
    return sfpi::reinterpret<sfpi::vFloat>(in_short);       // reinterpret as float = exp(x)
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[0]` (SFPLOAD/SFPSTORE) | Load/store element from/to DEST register file at current face offset |
| `sfpi::dst_reg++` | Advance DEST register pointer to next face (32 elements) |
| `sfpi::exexp(val)` (SFPEXEXP) | Extract biased exponent field from float |
| `sfpi::exexp_nodebias(val)` | Extract exponent without removing IEEE754 bias |
| `sfpi::exman8(val)` (SFPEXMAN) | Extract 8-bit mantissa with implicit leading 1 bit |
| `sfpi::exman9(val)` | Extract 9-bit mantissa |
| `sfpi::setexp(val, exp)` (SFPSETEXP) | Set/replace exponent field of float, keeping sign and mantissa |
| `sfpi::setsgn(val, sign)` (SFPSETSGN) | Set/replace sign bit of float |
| `sfpi::shft(val, amount)` (SFPSHFT) | Shift integer value left/right |
| `sfpi::reinterpret<T>(val)` | Bitwise reinterpret between vFloat/vInt/vUInt without conversion |
| `sfpi::int32_to_float(val, mode)` | Convert integer to float |
| `sfpi::float_to_fp16b(val, mode)` | Convert float32 to bfloat16 with rounding |
| `sfpi::addexp(val, offset)` (SFPIADD) | Add integer offset to exponent field (multiply by power of 2) |
| `sfpi::vec_min_max(a, b)` (SFPSWAP) | Vectorized min/max: after call, `a = min(a_old, b_old)`, `b = max(a_old, b_old)` |
| `sfpi::s2vFloat16b(val)` (SFPLOADI) | Load scalar constant as bfloat16 vector |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (per-lane conditional) |
| `v_and(cond)` | Narrow predication mask (AND with current mask) |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Evaluate polynomial using Horner's method via SFPMAD chains |
| `TTI_SFPLOADMACRO` | Trigger macro sequence execution from programmed macro registers |
| `TTI_SFPSHFT2` | Two-operand shift instruction used in fast approx replay pattern |
| `TTI_SFPMAD` | Multiply-add: VD = VA * VB + VC |
| `TTI_SFP_STOCH_RND` | Stochastic rounding: convert FP32 to integer format |
| `TTI_SFPCONFIG` | Configure SFPU constant registers (LREG[12-14]) and macro sequence registers |
| `lltt::record / lltt::replay` | Record and replay SFPU instruction sequences for amortized setup cost |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[0]** | Current tile face element; loaded via `dst_reg[0]`, result written back to `dst_reg[0]` |
| **DEST pointer** | Advanced via `dst_reg++` after each face iteration |
| **LREG[0-3]** | Working registers used by SFPLOADMACRO for loading, computation, and storing (fast approx mode) |
| **LREG[4]** | Staging register for SFPSHFT2 results (fast approx mode) |
| **LREG[12]** | Constant A = 256/ln(2) (Schraudolph constant, fast approx mode) |
| **LREG[13]** | Constant B-C = 32500.818 (Schraudolph bias adjustment, fast approx mode) |
| **LREG[14]** | Threshold value -88.5 (clamping, fast approx CLAMP mode) or shift amount 15 (non-clamp fast approx) |
| **LREG[16]** | Staging register for SETSGN output (non-clamp fast approx mode) |
| **vConstFloatPrgm0** | 1/ln(2) = 1.442695 (slow approx mode) |
| **vConstFloatPrgm1** | Bias constant C23_73 (slow approx mode) |
| **vConstIntPrgm2** | Exponent adjustment ADJ_EXP (slow approx mode) |

#### SFPU Execution Flow

1. **Initialization** (`exp_tile_init()` -> `exp_init()` -> `_init_exponential_()`):
   - For default non-approximate mode: initializes the reciprocal table (`_init_sfpu_reciprocal_<false>()`) needed by the `_sfpu_exp_` fallback path.
   - For approximate + fast mode with clamping: loads Schraudolph constants (A, B-C, threshold) into LREG[12-14], programs macro instructions for SWAP, MAD, STOCHRND, and SHFT operations, and configures two macro sequences (sanitize + compute).
   - For approximate + fast mode without clamping: loads constants, programs MAD/STOCHRND/SETSGN macros, configures macro sequence with SHFT2 pipeline, and records replay buffer.

2. **Per-tile execution** (`exp_tile(0)` -> `calculate_exponential()`):
   - **Non-approximate mode** (default for EXP without params):
     - Iterates `ITERATIONS=8` times (one per face of the 32x32 tile).
     - Each iteration: loads `dst_reg[0]`, optionally scales, calls `_sfpu_exp_improved_<is_fp32_dest_acc_en>(val)`, stores result back to `dst_reg[0]`, advances `dst_reg++`.
     - If `is_fp32_dest_acc_en=false` (bfloat16): calls `_sfpu_exp_21f_<false>()` which uses a 2nd-degree polynomial approximation of 2^(x_f) with explicit bfloat16 rounding.
     - If `is_fp32_dest_acc_en=true` (float32): calls `_sfpu_exp_f32_accurate_()` which uses Cody-Waite range reduction + 7th-order Taylor series for <1 ULP accuracy.
   - **Approximate mode** (when EXP is called with approx param = true):
     - Fast + clamping: uses SFPLOADMACRO sequences to process all 8 faces through programmed macro pipeline (sanitize inputs, MAD, ROUND, SHIFT).
     - Fast + no clamping: uses SFPLOADMACRO + SFPSHFT2 replay buffer pipeline with SETSGN for sign restoration.
     - Slow approximate: iterates with `_calculate_exponential_piecewise_` which clamps to [-42, 89] and uses `_calculate_exponential_approx_`.

3. **Tile face structure**: A 32x32 tile is processed as 8 "faces" of 32 elements each (or 4 rows x 2 columns of 4x16 sub-tiles, depending on DEST layout). The SFPU operates on one face per iteration via the `dst_reg` pointer.

#### SFPU Configuration

| Configuration | Value | Description |
|--------------|-------|-------------|
| **math_approx_mode** | `false` | EXP returns `false` from `get_op_approx_mode()`, so `APPROXIMATION_MODE=false` |
| **math_fidelity** | `MathFidelity::HiFi4` | Highest fidelity setting for FPU operations |
| **fp32_dest_acc_en** | Configurable | When true, DEST registers use float32; selects `_sfpu_exp_f32_accurate_` |
| **fast_and_approx** | `true` (default template) | When approx mode is enabled, uses fast Schraudolph-based algorithm |
| **ITERATIONS** | 8 | Processes 8 faces per tile (standard for 32x32 tiles) |
| **InputClamping** | `ClampToNegative` (default) | Clamps inputs below -88.5 to prevent incorrect outputs in fast approx mode |
| **Defines** | `SFPU_OP_EXP_INCLUDE=1` | Enables conditional inclusion of `exp.h` via `sfpu_split_includes.h` |
| **Defines** | `SFPU_OP_CHAIN_0` = `exp_tile_init(); exp_tile(0);` | Injected into compute kernel loop body |
| **Defines** | `INP_FLOAT=1` or `INP_FLOAT32=1` | Input data type indicator |

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations of the architecture-specific layer (`ckernel_sfpu_exp.h`) are **identical** -- both files contain the same `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and `calculate_exponential` implementations.

The lower-level LLK layer (`tt_llk_blackhole` vs `tt_llk_wormhole_b0`) differs in the SFPLOADMACRO-based fast approximate implementation, as the macro instruction encoding and pipeline timing may vary between architectures. The comments in `_sfpu_exp_f32_accurate_` explicitly note that Wormhole SFPMAD can only do `VD = VA * VB + VC`, while Blackhole has `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers. The code works around this by pre-negating constants.

## Implementation Notes

1. **Operation chaining**: The unary program factory supports chaining multiple operations via the `SFPU_OP_CHAIN_0` macro. For standalone EXP, the chain contains only one operation. When EXP is called with a parameter (e.g., `UnaryOpType::EXP` with `param0`), the parameter controls the `approx` template argument: `exp_tile_init<param0u>(); exp_tile<param0u>(0);`.

2. **Precision trade-offs**: The operation offers three distinct accuracy levels:
   - **Default** (`approx=false`, `fp32_dest_acc=false`): `_sfpu_exp_21f_` with 2nd-degree polynomial -- good for bfloat16 precision.
   - **High accuracy** (`approx=false`, `fp32_dest_acc=true`): `_sfpu_exp_f32_accurate_` with Cody-Waite + 7th-order Taylor -- <1 ULP for float32.
   - **Fast approximate** (`approx=true`): Schraudolph's algorithm via SFPLOADMACRO pipeline -- fastest but lowest accuracy. Achieves ~2.125-2.5 cycles/element using replay buffer.

3. **Program caching**: The `override_runtime_arguments` method enables reuse of compiled programs by updating only the buffer addresses. The kernel structure (core groups, tile counts) is fixed after initial compilation.

4. **Double buffering**: Both input and output CBs use 2-tile capacity with 1-tile block size, enabling full pipeline overlap between reader, compute, and writer.

5. **Tile register lifecycle**: The compute kernel follows the standard acquire-commit-wait-release pattern for DEST registers, ensuring correct synchronization between the unpack (copy_tile), SFPU compute (exp_tile), and pack (pack_tile) stages.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations like exp?"
   **Reason**: Initial understanding of the program factory architecture, kernel selection, and work distribution.
   **Key Findings**: Confirmed three-kernel pattern (reader/compute/writer), `split_work_to_cores` for distribution, CB configuration with 2-tile double buffering, and the `SFPU_OP_CHAIN` macro injection mechanism.

2. **Query**: "What is the SFPU exp operation implementation? Where is the exp compute kernel and the underlying SFPU kernel function located?"
   **Reason**: Locate the specific kernel files and understand the dispatch chain from `exp_tile` to `calculate_exponential` to the actual SFPU implementations.
   **Key Findings**: Identified three implementation variants (`_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`), the template dispatch via `_sfpu_exp_improved_`, and the file locations across the API, architecture-specific, and LLK layers.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Understand how `get_compute_kernel_path`, `get_block_defines`, and `get_op_approx_mode` configure the EXP operation.
   **Key Information**: EXP uses the default compute kernel path `eltwise_sfpu.cpp`, returns `false` for approx mode, and defines `SFPU_OP_CHAIN_0` as `exp_tile_init(); exp_tile(0);` (without params) or with template parameters (with params).

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Understand the API-level `exp_tile` and `exp_tile_init` function signatures and template parameters.
   **Key Information**: Documented the full template parameter list including `approx`, `fast_and_approx`, `scale_en`, `skip_positive_check`, `input_clamping`, and `iterations`. Confirmed the dispatch via `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macro.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Read the architecture-specific SFPU kernel implementations.
   **Key Information**: Found `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, `_sfpu_exp_improved_`, and `calculate_exponential` with detailed algorithm documentation and references to Moroz et al. 2022.

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Read the lower-level LLK implementation including `_calculate_exponential_`, `_init_exponential_`, `_sfpu_exp_`, and the SFPLOADMACRO-based fast approximate path.
   **Key Information**: Found the complete Schraudolph fast approximation with SFPLOADMACRO macro programming, replay buffer optimization, and the legacy `_sfpu_exp_` Horner polynomial implementation. Also found the three operating modes (fast+clamp, fast+no-clamp, slow approx) and their respective SFPU instruction sequences.
