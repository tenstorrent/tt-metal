# LOG (Natural Logarithm) Implementation Analysis

## Overview

The LOG operation computes the element-wise natural logarithm (ln) of each element in the input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which dispatches the generic `eltwise_sfpu.cpp` compute kernel with LOG-specific preprocessor defines injected via the operation chain mechanism.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The LOG operation has two distinct compute paths depending on whether FP32 destination accumulation is enabled:
- **Non-FP32 path**: Uses a degree-5 minimax polynomial approximation of ln(x) over [1, 2], then corrects via the extracted exponent.
- **FP32 path**: Uses a higher-precision approach based on the identity ln(x) = 2z(1 + z^2/3 + z^4/5 + ...) where z = (m-1)/(m+1), with range reduction to [sqrt(2)/2, sqrt(2)].

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for this factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages/tiles |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Flattened to pages/tiles |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or may differ for BITCAST; not applicable for LOG) |

### Layout Transformations

No tilize/untilize or reshard operations are performed. The output has the same layout and shape as the input.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved pages) | CB c_0 | `cb_reserve_back`, `noc_async_read_page`, `cb_push_back` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front`, `copy_tile`, SFPU log, `pack_tile`, `cb_pop_front`, `cb_push_back` |
| 3 | Writer | CB c_2 | DRAM (interleaved pages) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

The reader streams tiles one at a time from DRAM into CB c_0. The compute kernel acquires tile registers, waits for one tile in c_0, copies it to the DEST register, applies the SFPU LOG chain, packs the result into CB c_2, and pops the input. The writer drains CB c_2 one tile at a time back to DRAM.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is only allocated for HARDSHRINK, CBRT, or LOGIT operations. It is **not** used by LOG.

Both input and output CBs are double-buffered (capacity = 2 * page_size), allowing the reader to fill one tile slot while compute processes the other, and similarly for compute/writer overlap.

## Pipeline Pattern Summary

- **CB c_0**: Double-buffered (2 tiles capacity, 1 tile block). Reader and compute can overlap.
- **CB c_2**: Double-buffered (2 tiles capacity, 1 tile block). Compute and writer can overlap.

The pipeline supports a 3-stage overlap: while the reader fills the next tile in c_0, compute processes the current tile and writes to c_2, and the writer drains the previous tile from c_2.

## Index Calculations

The reader and writer use `TensorAccessor` for page-to-physical-address mapping. Each core is assigned a contiguous range of page IDs:
- `start_id` = sum of pages assigned to all lower-numbered cores
- `end_id` = `start_id + num_pages_per_core`

Pages are iterated sequentially from `start_id` to `end_id`. The `TensorAccessor` converts the logical page index to a physical DRAM bank address via the interleaved buffer's bank mapping.

## Memory Access Patterns

### Read Pattern
Sequential page reads: each core reads its assigned page range in order, one page at a time, using `noc_async_read_page`. Each read is followed by `noc_async_read_barrier` before pushing to the CB. This is a simple sequential streaming pattern.

### Write Pattern
Sequential page writes: each core writes its output pages in order, one page at a time, using `noc_async_write_page`. Writes are flushed after each page (`noc_async_writes_flushed`), with a final `noc_async_write_barrier` at the end.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets ceil(num_pages/num_cores), group 2 gets floor |

Core enumeration is column-major: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides tiles across the available compute grid, creating up to two core groups to handle remainder tiles. Group 1 cores process one extra tile compared to group 2 cores.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for the source buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for the destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

Additionally, the compute kernel receives preprocessor defines:
- `SFPU_OP_CHAIN_0` = `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` which expands to `log_tile_init(); log_tile(0);`
- `INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, or `INP_FLOAT` depending on input dtype

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages this core should read |
| 2 | start_id | uint32_t | First page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages this core should write |
| 2 | start_id | uint32_t | First page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for LOG (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for LOG (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB c_0 | Read tiles sequentially via TensorAccessor |
| compute | RISCV_2 (math) | N/A | CB c_0 | CB c_2 | copy_tile, SFPU log, pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM | Write tiles sequentially via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Iterates from `start_id` to `start_id + num_pages`, reading one page at a time into CB c_0. Uses `TensorAccessor` for address translation. Supports optional `BACKWARDS` define (not used by LOG).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Drains CB c_2 one page at a time. Supports `OUT_SHARDED` define (not used by LOG in interleaved mode). Uses `TensorAccessor` for address translation.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of blocks (= number of tiles for LOG since block_dim=1)
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for this factory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initialize SFPU pipeline with input CB c_0 and output CB c_2;
                                                      // configures unpack (from c_0) and pack (to c_2) hardware
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for per_core_block_dim tiles
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers (16 tile slots)

            cb_wait_front(tt::CBIndex::c_0, 1);  // block until reader has produced 1 tile in CB c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DEST register slot 0;
                                                  // this moves data from L1 (CB) through the unpacker into DEST

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: log_tile_init(); log_tile(0);
                             // log_tile_init() configures SFPU programmable constants (ln(2), polynomial coefficients)
                             // log_tile(0) dispatches SFPU to compute ln() on tile in DEST[0]
#endif

            tile_regs_commit();  // signal that DEST registers are ready for packing (math->pack handoff)

            tile_regs_wait();  // wait for pack stage to be ready to consume DEST registers

            pack_tile(0, tt::CBIndex::c_2);  // pack tile from DEST[0] into CB c_2 output buffer;
                                              // converts from DEST format back to CB data format

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed tile slot in CB c_0 for reader to reuse

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish per_core_block_dim tiles to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h`
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h`
- **LLK wrapper**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h`

#### Annotated SFPU Kernel Source (Blackhole variant)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"  // provides PolynomialEvaluator::eval for Horner's method polynomial evaluation

namespace ckernel {
namespace sfpu {

// Non-FP32 path: minimax polynomial approximation of ln(x) over [1, 2]
// Uses a degree-5 polynomial computed via Sollya's fpminimax.
// The identity ln(x) = ln(2^n * m) = n*ln(2) + ln(m) is used,
// where m is the mantissa normalized to [1, 2) and n is the exponent.
template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {

    // Step 1: Normalize mantissa to [1, 2) range by forcing the exponent to 127 (IEEE bias)
    sfpi::vFloat x = sfpi::setexp(in, 127);  // setexp replaces the 8-bit exponent field;
                                               // this extracts the mantissa as a float in [1, 2)

    // Step 2: Evaluate degree-5 minimax polynomial for ln(x) over [1, 2]
    // Coefficients found via Sollya: fpminimax(log(x), 5, [|single...|], [1+2^(-20); 2], relative)
    // The polynomial is: c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    // evaluated via Horner's method through PolynomialEvaluator::eval
    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,   // c0 = -2.0069785118103027 (loaded into programmable constant register 1)
        sfpi::vConstFloatPrgm2,   // c1 = 3.767500400543213 (loaded into programmable constant register 2)
        -2.800232410430908,       // c2
        1.3681391477584839,       // c3
        -0.3706687390804291,      // c4
        0.04224011301994324);     // c5

    // Step 3: Extract the debiased exponent (n) from the original input
    sfpi::vInt exp = sfpi::exexp(in);  // exexp extracts the exponent field and debiases it (subtracts 127)
                                        // result is a signed integer: e.g., for 2.0 -> exp=1, for 0.5 -> exp=-1

    // Convert negative exponent from two's complement to sign-magnitude for int32_to_float
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }  // ~exp+1 = abs(exp), setsgn sets sign bit to 1
    v_endif;

    // Convert exponent integer to float
    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);  // int32_to_float converts sign-magnitude int to float
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;   // ln(2) = 0.693147..., preloaded in log_init()

    // Step 4: Combine: ln(x) = n * ln(2) + ln(m)
    sfpi::vFloat result = expf * vConstLn2 + series_result;

    // Optional base scaling for log2/log10: multiply by 1/ln(base)
    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    // Step 5: Handle special case: ln(0) = -infinity
    v_if(in == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    // Step 6: Handle NaN/infinity cases (only in non-fast-approx mode)
    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            // +infinity input -> +infinity output
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {
            // NaN input (exp==128 with non-zero mantissa) or negative input -> NaN
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }

    // Step 7: Convert result to bfloat16 if not using FP32 dest accumulation
    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

// FP32 path: higher-precision computation using the identity
// ln(x) = 2z(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
// where z = (m - 1) / (m + 1) and m is the mantissa in [sqrt(2)/2, sqrt(2)]
template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    sfpi::vFloat result;

    // Handle special cases first
    sfpi::vInt exp = sfpi::exexp(val);  // extract debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        result = std::numeric_limits<float>::infinity();     // +inf -> +inf
    }
    v_elseif(exp == 128 || val < 0.f) {
        result = std::numeric_limits<float>::quiet_NaN();    // NaN or negative -> NaN
    }
    v_elseif(val == 0.0f) {
        result = -std::numeric_limits<float>::infinity();    // 0 -> -inf
    }
    v_else {
        // Step 1: Extract mantissa into [1, 2) by setting exponent to bias (127)
        sfpi::vFloat m = sfpi::setexp(val, 127);

        // Step 2: Range reduction - if m >= sqrt(2), halve it and increment exponent
        // This narrows the range to [sqrt(2)/2, sqrt(2)] ~ [0.707, 1.414]
        // which improves polynomial convergence around m=1
        constexpr float SQRT2 = 1.4142135381698608f;
        v_if(m >= SQRT2) {
            m = m * 0.5f;
            exp = exp + 1;
        }
        v_endif;

        // Step 3: Variable substitution z = (m-1)/(m+1)
        // Maps m in [0.707, 1.414] to z in [-0.172, 0.172] - a very narrow range
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;    // vConst1 = 1.0f (hardware constant)
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

        // Division via Newton-Raphson reciprocal (2 iterations for FP32 precision)
        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

        sfpi::vFloat z2 = z * z;  // z^2 for Horner's evaluation

        // Step 4: Polynomial approximation using odd-power series
        // ln(m) = 2z(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
        // Evaluated as: p = 1 + z2*(1/3 + z2*(1/5 + z2*(1/7 + z2*(1/9 + z2*1/11))))
        sfpi::vFloat p = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,           // 1.0
            0.3333333333333333f,     // 1/3
            0.2f,                    // 1/5
            0.14285714285714285f,    // 1/7
            0.1111111111111111f,     // 1/9
            .09090909090909091f);    // 1/11

        sfpi::vFloat ln_m = 2.0f * (z * p);  // ln(m) = 2 * z * p

        // Convert exponent from two's complement to sign-magnitude for int32_to_float
        v_if(exp < 0) {
            sfpi::vInt exp_abs = ~exp + 1;
            exp = sfpi::setsgn(exp_abs, 1);
        }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        // Step 5: Combine: ln(x) = n * ln(2) + ln(m)
        constexpr float LN2 = 0.69314718246459961f;
        result = expf * LN2 + ln_m;

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }
    }
    v_endif;

    return result;
}

// Main entry point: processes one tile face (ITERATIONS elements) per call
// Called 8 times per tile (8 faces of 4 rows each = 32 rows, or arch-specific)
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8  // request full unrolling for all 8 iterations
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];    // load current face element from DEST register
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            // Non-FP32: use minimax polynomial path
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            // FP32: use higher-precision odd-power series path
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;   // write result back to DEST register
        sfpi::dst_reg++;              // advance to next face element (stride through tile)
    }
}

// Initialization: load programmable constants used by the polynomial
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        // Non-FP32 path constants for the minimax polynomial
        sfpi::vConstFloatPrgm0 = 0.69314718246459961f;  // ln(2) - used in exponent correction
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;   // polynomial coefficient c0
        sfpi::vConstFloatPrgm2 = 3.767500400543213;     // polynomial coefficient c1
    } else {
        // FP32 path: initialize reciprocal LUT (needed for z = (m-1)/(m+1) division)
        _init_reciprocal_</*approximation_mode*/ false, /*legacy_compat*/ false>();
        // Note: _init_reciprocal_ sets vConstFloatPrgm0 = 2.0f (used internally by reciprocal)
        // The FP32 path uses inline LN2 constant instead of vConstFloatPrgm0
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::setexp(v, imm)` | Replaces the 8-bit exponent field of each float element with `imm`. Used to normalize mantissa to [1, 2) by setting exponent to 127 (IEEE bias). |
| `sfpi::exexp(v)` | Extracts and debiases the exponent field of each float element (subtracts 127). Returns a signed integer. |
| `sfpi::setsgn(v, bit)` | Sets the sign bit of each element. Used for two's complement to sign-magnitude conversion. |
| `sfpi::int32_to_float(v, 0)` | Converts a sign-magnitude integer to floating point. Used to convert the extracted exponent to float for the n*ln(2) term. |
| `sfpi::float_to_fp16b(v, 0)` | Converts FP32 to BFloat16 format. Used in non-FP32 path to truncate result precision. |
| `sfpi::reinterpret<T>(v)` | Bitwise reinterpretation between vFloat/vInt/vUInt types (no data conversion). |
| `sfpi::dst_reg[i]` | Access to DEST register file. Reads/writes tile face elements. |
| `sfpi::dst_reg++` | Advances the DEST register pointer to the next face element. |
| `sfpi::vConst1` | Hardware constant 1.0f. |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers (3 available). Loaded during `log_init()`. |
| `_sfpu_reciprocal_<N>(v)` | Newton-Raphson reciprocal with N iterations. Used in FP32 path for division. |
| `PolynomialEvaluator::eval(...)` | Horner's method polynomial evaluation. Evaluates a polynomial at a given point using the provided coefficients. |
| `v_if / v_elseif / v_else / v_endif` | SFPU conditional execution (predicated operations). Sets per-lane condition codes. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` | Input: current tile face element loaded from DEST. Output: result written back. |
| `dst_reg++` | Pointer increment to iterate through 8 face elements per call. |
| `vConstFloatPrgm0` | Non-FP32: ln(2) = 0.693147... / FP32: set to 2.0 by reciprocal init (unused directly) |
| `vConstFloatPrgm1` | Non-FP32: polynomial coeff c0 = -2.00698... / FP32: unused |
| `vConstFloatPrgm2` | Non-FP32: polynomial coeff c1 = 3.76750... / FP32: unused |
| `vConst1` | Hardware constant 1.0f (used in FP32 path for m-1 and m+1) |
| Local `vFloat` regs | `x`, `series_result`, `expf`, `result`, `m`, `z`, `z2`, `p`, `ln_m` (compiler-managed SFPU registers) |

#### SFPU Execution Flow

1. **Initialization** (`log_init`): Called once before tile processing begins. Loads programmable constants into `vConstFloatPrgm0/1/2`. For FP32 mode, initializes the reciprocal lookup table instead.

2. **Per-tile dispatch** (`log_tile(0)`): The LLK wrapper `llk_math_eltwise_unary_sfpu_log` calls `_llk_math_eltwise_unary_sfpu_params_` which:
   - Selects the DEST register for the tile at index 0
   - Calls `calculate_log` with `ITERATIONS=8` to process one tile face at a time
   - The SFPU processes all faces of the tile (the LLK framework manages face iteration)

3. **Per-element computation** (inside `calculate_log` loop, 8 iterations):
   - **Load**: Read element from `dst_reg[0]`
   - **Path selection**: Compile-time branch on `is_fp32_dest_acc_en`
     - **Non-FP32 path** (`calculate_log_body`):
       a. Normalize mantissa via `setexp(in, 127)` to get m in [1, 2)
       b. Evaluate degree-5 minimax polynomial on m
       c. Extract exponent via `exexp(in)`, convert to float
       d. Combine: result = n * ln(2) + poly(m)
       e. Handle edge cases (zero -> -inf, NaN/negative -> NaN)
       f. Convert to bfloat16 via `float_to_fp16b`
     - **FP32 path** (`calculate_log_f32_body`):
       a. Check special cases first (inf, NaN, negative, zero)
       b. Normalize mantissa via `setexp(val, 127)`
       c. Range reduction: if m >= sqrt(2), halve m and increment exponent
       d. Compute z = (m-1)/(m+1) using Newton-Raphson reciprocal
       e. Evaluate 6-term odd-power polynomial in z^2
       f. Combine: result = n * ln(2) + 2*z*p
   - **Store**: Write result to `dst_reg[0]`, advance pointer

4. **Pack**: After SFPU completes, `tile_regs_commit()` hands DEST to the packer, which writes the result tile to CB c_2.

#### SFPU Configuration

| Configuration | Value |
|---------------|-------|
| **Math fidelity** | HiFi4 |
| **Math approx mode** | `false` (LOG returns false from `get_op_approx_mode`) |
| **FP32 dest accumulation** | Configurable via `args.fp32_dest_acc_en` |
| **FAST_APPROX template** | `false` by default (LOG is called as `log_tile<false>()` unless param overrides) |
| **Unpack to DEST mode** | Default (or `UnpackToDestFp32` if `preserve_fp32_precision` is set) |
| **BFP8 pack precise** | Configurable via `args.bfp8_pack_precise` |

The `FAST_APPROX` template parameter, when true, skips the NaN/infinity/negative edge case handling in the non-FP32 path for better performance at the cost of correctness on edge inputs.

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations are **nearly identical** in algorithm and structure. Key differences:

1. **FP32 path initialization**: Blackhole uses `_init_reciprocal_<false, false>()` while Wormhole uses `_init_sfpu_reciprocal_<false>()`. These are architecture-specific names for the same operation (initializing the reciprocal lookup table).

2. **FP32 path range reduction**: In Blackhole, the `calculate_log_f32_body` uses a local `constexpr float SQRT2` constant. In Wormhole (Blackhole variant shown in the worktree), `sfpi::vConstFloatPrgm1` is loaded with sqrt(2) instead.

3. **FP32 path ln(2)**: Blackhole uses a local `constexpr float LN2`, while Wormhole stores ln(2) in `sfpi::vConstFloatPrgm2`.

4. Both architectures use the same polynomial coefficients and the same mathematical approach. The algorithmic behavior is equivalent.

## Implementation Notes

1. **Two algorithm variants**: The non-FP32 path uses a 5th-degree minimax polynomial (Sollya-optimized) directly on the mantissa in [1, 2). The FP32 path uses a different approach: variable substitution z = (m-1)/(m+1) with range reduction to [sqrt(2)/2, sqrt(2)], then a 6-term odd-power series. The FP32 path is more numerically stable for single-precision computation.

2. **Programmable constant pressure**: The SFPU has only 3 programmable constant registers (vConstFloatPrgm0/1/2). The non-FP32 path uses all 3 (ln(2), and 2 polynomial coefficients), with the remaining 4 coefficients as immediates. The FP32 path uses vConstFloatPrgm0 implicitly (for reciprocal init), and uses inline constants for everything else.

3. **Division via reciprocal**: The FP32 path avoids hardware division by computing `z = (m-1) * reciprocal(m+1)`, where `_sfpu_reciprocal_<2>` performs 2 Newton-Raphson iterations for FP32-adequate precision.

4. **GCC unroll pragma**: The `#pragma GCC unroll 8` in `calculate_log` requests full loop unrolling for the 8-iteration face processing loop, avoiding branch overhead and enabling better register allocation.

5. **Cached program reuse**: The `override_runtime_arguments` method allows reusing the compiled program with different buffer addresses, avoiding kernel recompilation when only tensor locations change.

6. **No LOG-specific scalar parameters**: Unlike some unary ops (e.g., HARDSHRINK, WHERE_TSS), LOG does not use packed scalar runtime arguments. Both `packed_scalar1` and `packed_scalar2` are set to 0.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary SFPU program factory implemented? What kernels does it use?"
   **Reason**: Needed to understand the overall program factory structure before reading the code.
   **Key Findings**: Confirmed that the unary factory uses three kernels (reader, compute, writer), that the compute kernel is `eltwise_sfpu.cpp` for most operations, and that CBs c_0 and c_2 are used with double-buffering.

2. **Query**: "How does the LOG unary SFPU operation work? What compute kernel file is used?"
   **Reason**: Needed to identify the specific SFPU kernel file and function names for LOG.
   **Key Findings**: LOG uses `ckernel_sfpu_log.h`, dispatches `log_tile_init()` and `log_tile(0)`, and has both `calculate_log_body` (non-FP32) and `calculate_log_f32_body` (FP32) implementations.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how LOG's compute defines are generated and which kernel path is selected.
   **Key Information**: LOG uses `log_tile_init()` / `log_tile(0)` as SFPU_OP_CHAIN defines, falls through to `eltwise_sfpu.cpp` in `get_compute_kernel_path`, and `get_op_approx_mode` returns false for all ops (default case).

2. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h`
   **Reason**: Needed to understand the LLK-level wrapper between compute API and SFPU kernel.
   **Key Information**: `llk_math_eltwise_unary_sfpu_log` calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_log<APPROXIMATE, FAST_APPROX, false, is_fp32_dest_acc_en>`, passing `HAS_BASE_SCALING=false` and `log_base_scale_factor=0` for natural log.
