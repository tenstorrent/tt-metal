# EXP (Exponential) Implementation Analysis

## Overview

The EXP operation computes element-wise `exp(x)` for every element of the input tensor. It is implemented as a unary SFPU operation through the shared `UnaryProgramFactory`, which dispatches the `eltwise_sfpu.cpp` compute kernel with EXP-specific defines injected via the `SFPU_OP_CHAIN_0` macro.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The factory supports two program factory variants:
- `UnaryProgramFactory` -- standard interleaved path (primary, analyzed here)
- `UnarySubCoreGridProgramFactory` -- sub-core-grid variant for restricted core sets

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `input.buffer()->num_pages()` (total tiles in tensor) |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt` tiles), inner loop per tile within block (always 1 tile per block) |

Each core processes `per_core_block_cnt` tiles sequentially, one tile at a time. The block size (`per_core_block_dim`) is hardcoded to 1 in `UnaryProgramFactory`.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (any rank) | Same as input |
| **Dimension convention** | N/A (flattened to pages) | Same as input |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16 / FLOAT32 / INT32 / UINT32 | Same as input (or different for BITCAST) |

### Layout Transformations
None. The operation is a pure element-wise transformation; no tilize/untilize or reshard occurs within the program factory.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

Step-by-step:
1. **Reader** fetches one tile from DRAM using `TensorAccessor`-based `noc_async_read_page`, pushing it into CB c_0.
2. **Compute** waits for a tile in CB c_0, copies it to DST register via `copy_tile`, executes `exp_tile_init()` + `exp_tile(0)` on DST[0], packs the result into CB c_2.
3. **Writer** waits for a tile in CB c_2, writes it to DRAM via `noc_async_write_page`, then pops it.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_tmp0 | Scratchpad (conditional) | 2 tiles | 1 tile | Double | Compute | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes:**
- CB c_1 is only created for `HARDSHRINK`, `CBRT`, and `LOGIT` operations. For EXP, it is **not** allocated.
- Both input and output CBs have capacity for 2 tiles (double-buffered), enabling overlap between producer and consumer.
- Page size is `tile_size(cb_data_format)` for tile layout, or `buffer->page_size()` for row-major.

## Pipeline Pattern Summary

All CBs use double-buffering (capacity = 2 x block_size). This enables:
- Reader can write tile N+1 while Compute processes tile N
- Compute can write result N while Writer outputs result N-1
- Full 3-stage pipeline overlap is possible

## Index Calculations

The program factory uses `TensorAccessor` for both reader and writer kernels. The accessor is constructed from compile-time `TensorAccessorArgs` (encoding buffer layout metadata) and runtime arguments (buffer address, page size).

Index mapping is sequential: each core is assigned a contiguous range of page IDs starting at `start_id = num_pages_written` (accumulated from previous cores). Within each kernel, pages are accessed sequentially from `start_id` to `start_id + num_pages`.

## Memory Access Patterns

### Read Pattern
- **Sequential page reads**: Pages are read in ascending order from `start_id` to `end_id`
- **One page per NoC transaction**: Each `noc_async_read_page` reads exactly one tile
- **Barrier per tile**: `noc_async_read_barrier()` is called after each read (no batching)
- Access pattern is DRAM-interleaved via `TensorAccessor`

### Write Pattern
- **Sequential page writes**: Pages are written in ascending order from `start_id` to `end_id`
- **One page per NoC transaction**: Each `noc_async_write_page` writes exactly one tile
- **Flush per tile**: `noc_async_writes_flushed()` after each write, final `noc_async_write_barrier()` at end

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages/num_cores)`, group 2 gets `floor(num_pages/num_cores)` |

Core enumeration is column-major: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides total pages into two groups to handle remainder tiles. If `num_pages` divides evenly, `core_group_2` is empty.

Two separate compute kernel instances are created (one per group) with different `per_core_block_cnt` compile-time arguments.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded buffer metadata (layout, bank info, page size) for src_buffer |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded buffer metadata for dst_buffer |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tiles (blocks) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1) |

#### Compute Kernel Defines (Compile-Time)
| Define | Value for EXP | Description |
|--------|---------------|-------------|
| `SFPU_OP_EXP_INCLUDE` | `1` | Enables EXP header inclusion in `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro expanding to init + func calls |
| `SFPU_OP_CHAIN_0_INIT_0` | `exp_tile_init<Pu>();` | Init call (P=param0, typically 1 for approx) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `exp_tile<Pu>(0);` | Compute call on DST[0] |
| `INP_FLOAT` / `INP_FLOAT32` | `1` | Input data type indicator |

#### Compute Config
| Field | Value | Description |
|-------|-------|-------------|
| `math_fidelity` | `HiFi4` | Highest fidelity |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | Enables FP32 DST accumulation |
| `math_approx_mode` | `false` | `get_op_approx_mode` returns false for EXP |
| `bfp8_pack_precise` | From `args.bfp8_pack_precise` | Precise BFP8 packing |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | First page ID for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | First page ID for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0 | Sequential tile reads via TensorAccessor |
| compute | TRISC (RISCV_2, math) | N/A | CB c_0 | CB c_2 | copy_tile -> exp_tile (SFPU) -> pack_tile |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Sequential tile writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Creates a `TensorAccessor` from compile-time args, then loops from `start_id` to `start_id + num_pages`, reading one page at a time into CB c_0 with a read barrier per page.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Waits for each page in CB c_2, writes it to DRAM via `noc_async_write_page`, flushes, and pops. Supports `OUT_SHARDED` define (not used for interleaved path). Final `noc_async_write_barrier()` ensures all writes complete.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // ANALYSIS: conditionally includes exp.h when SFPU_OP_EXP_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // ANALYSIS: number of tiles this core must process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // ANALYSIS: tiles per block, always 1 for UnaryProgramFactory

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // ANALYSIS: initializes unpack (from c_0) and pack (to c_2) pipelines for SFPU
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // ANALYSIS: reserves output space; for block_dim=1, reserves 1 tile in c_2
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {  // ANALYSIS: inner loop runs once (block_dim=1)
            tile_regs_acquire();  // ANALYSIS: acquires exclusive access to DST registers for writing

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // ANALYSIS: blocks until reader has produced 1 tile in CB c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // ANALYSIS: unpacks tile 0 from CB c_0 into DST[0] via unpacker

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // ANALYSIS: expands to "exp_tile_init<P>(); exp_tile<P>(0);" -- initializes SFPU for exp, then computes exp on DST[0]
#endif

            tile_regs_commit();  // ANALYSIS: signals DST write complete, hands off to pack stage

            tile_regs_wait();  // ANALYSIS: waits for pack stage to be ready to read from DST

            pack_tile(0, tt::CBIndex::c_2);  // ANALYSIS: packs DST[0] into CB c_2 output buffer

            cb_pop_front(tt::CBIndex::c_0, 1);  // ANALYSIS: frees the consumed input tile from CB c_0

            tile_regs_release();  // ANALYSIS: releases DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // ANALYSIS: publishes the block (1 tile) in CB c_2 to writer
    }
}
```

### SFPU Kernel Implementation

The EXP operation has two distinct implementation paths selected at compile time:

1. **Metal-level improved path** (`tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`) -- used when `APPROXIMATION_MODE=false` in `calculate_exponential`. Contains `_sfpu_exp_21f_` (bfloat16) and `_sfpu_exp_f32_accurate_` (float32).
2. **LLK-level legacy path** (`tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h`) -- used when `APPROXIMATION_MODE=true`. Contains `_calculate_exponential_`, `_sfpu_exp_`, and fast approximate implementations.

The dispatch is:
- `exp_tile<approx>(0)` -> `calculate_exponential<approx, true, DST_ACCUM_MODE, ...>()`
  - If `approx=true`: calls `_calculate_exponential_<true, ...>()` (LLK fast approx path)
  - If `approx=false`: loops over 8 SFPU iterations, calling `_sfpu_exp_improved_<DST_ACCUM_MODE>(val)` per iteration
    - If `DST_ACCUM_MODE=false`: `_sfpu_exp_21f_<false>()` (Moroz 2022, 2nd-degree polynomial)
    - If `DST_ACCUM_MODE=true`: `_sfpu_exp_f32_accurate_()` (Cody-Waite + 7th-order Taylor)

#### SFPU Kernel File (Metal-level, non-approximate path)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
(Identical for `blackhole`)

#### Annotated SFPU Kernel Source (Metal-level)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // ANALYSIS: includes the LLK-level base exp implementations
#include "sfpu/ckernel_sfpu_polyval.h"    // ANALYSIS: PolynomialEvaluator for polynomial approximations
#include "sfpu/ckernel_sfpu_converter.h"  // ANALYSIS: Converter utility for bit-casting floats
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"                         // ANALYSIS: SFPI programming interface (vFloat, vInt, dst_reg, etc.)

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }
// ANALYSIS: thin wrapper around LLK _sfpu_exp_ (Horner series + repeated squaring). Not used by improved path.

/*
 * Both _float_to_int32_ and _float_to_int32_positive_ use branch to handle special cases
 * With exp21f function, some of these cases never happen (e.g. negative exponent, overflow)
 * This allow for a branch free (and much smaller algorithm) to compute integer value
 *
 * The constraint on `val` is: 0 <= val < 128.0f
 * Note: Unlike _float_to_int32_ and _float_to_int32_positive, this function assumes that
 * value has been been divided by 2^23. Output value will be scaled by 2^23 compared to 'val'.
 * If that was not the case, we would have had to shift by `exp - 23` instead of `exp`
 * This saves 1 SFPADDI instruction.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);      // ANALYSIS: extracts biased exponent from float
    sfpi::vInt man = sfpi::exman8(val);      // ANALYSIS: extracts 8-bit mantissa with implicit leading 1 (value in [1,2))
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    // ANALYSIS: shifts mantissa left by exponent amount, converting float to fixed-point integer
    // This is a branch-free float-to-int conversion optimized for the constrained input range [0, 128)
    return man;
}

/*
 * exp_21f algorithm from Moroz et al. 2022 "Simple Multiple Precision Algorithms
 * for Exponential Functions" - Section 5.
 *
 * Computes exp(x) = 2^(x/ln2) by splitting into integer and fractional parts,
 * then using a 2nd-degree polynomial to approximate 2^(fractional_part).
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // ANALYSIS: Step 1 - Convert x to base-2: xlog2 = x / ln(2) + 127 (IEEE bias)
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);
    // ANALYSIS: The +127 incorporates the IEEE-754 exponent bias so the integer part
    // directly becomes a valid biased exponent when recombined

    // ANALYSIS: Step 2 - Clamp to valid range [0, 255] to prevent overflow/underflow
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);   // ANALYSIS: SFPU min/max swap; after this, threshold_low <= xlog2
    sfpi::vec_min_max(xlog2, threshold_high);   // ANALYSIS: after this, xlog2 <= threshold_high (255)

    // ANALYSIS: Step 3 - Convert clamped float to integer (branch-free)
    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    // ANALYSIS: Step 4 - Split into exponent (integer) and mantissa (fractional) parts
    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // ANALYSIS: extracts exponent bits = 2^(integer part)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // ANALYSIS: extracts 9-bit mantissa = fractional part in [0,1)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    // ANALYSIS: converts integer mantissa bits to float for polynomial evaluation

    // ANALYSIS: Step 5 - Polynomial approximation of 2^(frac) using 2nd-degree polynomial
    // Coefficients optimized for input range [0, 2^23]
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // ANALYSIS: Step 6 - Recombine: result = 2^(integer) * 2^(fractional)
    // setexp sets the exponent field of frac to exponential_part
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // ANALYSIS: When DST is bfloat16, explicitly round to bf16 using round-to-nearest-even
        // to avoid truncation errors from SFPSTORE's default truncation behavior
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

/*
 * exp_61f algorithm - higher-order polynomial variant from same Moroz 2022 paper.
 * Uses 6th-degree polynomial for 2^(frac) approximation on [0, 1] range.
 */
sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    // ANALYSIS: Same clamping as exp_21f
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    // ANALYSIS: Scale fractional part by 2^-23 to bring into [0, 1] range
    frac = sfpi::addexp(frac, -23);

    // ANALYSIS: 6th-degree polynomial approximation of 2^x on [0, 1]
    // Higher accuracy than exp_21f but more SFPU instructions
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

// ANALYSIS: Utility for Cody-Waite path: round-to-nearest-even float -> int32
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    // ANALYSIS: Hacker's Delight trick: add 2^23+2^22, then subtract to get rounded integer
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;                                    // ANALYSIS: float result = round(z)
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);  // ANALYSIS: integer result
    return k;
}

/*
 * High-accuracy exp(x) using Cody-Waite range reduction + 7th-order Taylor series.
 * Target: < 1 ULP error for float32.
 * Used when fp32_dest_acc_en = true.
 */
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;    // ANALYSIS: ~exp(88.7)
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;  // ANALYSIS: ~exp(-88)

    // ANALYSIS: Step 1 - Scale input: z = x / ln(2)
    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;

    // ANALYSIS: Check for special cases using predicated execution (v_if/v_elseif/v_else)
    sfpi::vInt exp_bits = sfpi::exexp(z);

    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();   // ANALYSIS: overflow -> +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;                             // ANALYSIS: underflow -> 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();   // ANALYSIS: NaN passthrough
    }
    v_else {
        // ANALYSIS: Step 2 - Round z to nearest integer k
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // ANALYSIS: Step 3 - Cody-Waite range reduction
        // Split ln(2) into high + low parts for extended precision subtraction
        // r = x - k*ln(2) computed as r = k*(-LN2_HI) + val, then r += k*(-LN2_LO)
        // Negated constants enable SFPMAD optimization: VD = VA * VB + VC
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;  // ANALYSIS: compiles to single SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi;    // ANALYSIS: compiles to single SFPMAD

        // ANALYSIS: Step 4 - 7th-order Taylor polynomial: exp(r) ~= sum(r^n/n!, n=0..7)
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,  // 1
            sfpi::vConst1,  // 1
            0.5f,           // 1/2!
            1.0f / 6.0f,    // 1/3!
            1.0f / 24.0f,   // 1/4!
            1.0f / 120.0f,  // 1/5!
            1.0f / 720.0f,  // 1/6!
            1.0f / 5040.0f  // 1/7!
        );

        // ANALYSIS: Step 5 - Scale by 2^k via exponent manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);   // ANALYSIS: get current exponent of polynomial result
        sfpi::vInt new_exp = p_exp + k_int;            // ANALYSIS: add k to exponent = multiply by 2^k
        result = sfpi::setexp(p, new_exp);             // ANALYSIS: set new exponent
    }
    v_endif;

    return result;
}

// ANALYSIS: Dispatcher template - selects implementation based on FP32 dest accumulation mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);   // ANALYSIS: bfloat16 path: Moroz 2022 exp_21f with bf16 rounding
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // ANALYSIS: float32 path: Cody-Waite + Taylor series, < 1 ULP
}

/*
 * Main entry point called from exp_tile via SFPU_TEMPLATE_PARAMS_KERNEL_FN.
 * Template parameters determine which implementation path is taken.
 */
template <
    bool APPROXIMATION_MODE,     // ANALYSIS: true = use fast LLK approximate path
    bool FAST_APPROX,            // ANALYSIS: true = use fastest Schraudolph-based approximation
    bool is_fp32_dest_acc_en,    // ANALYSIS: true = FP32 DST registers, false = BF16
    bool SCALE_EN = false,       // ANALYSIS: enable input pre-scaling
    int ITERATIONS = 8,          // ANALYSIS: number of SFPU iterations (8 = one tile face, 32 = full tile)
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // ANALYSIS: Delegates to LLK-level _calculate_exponential_ which has multiple sub-paths:
        // - FAST_APPROX && CLAMP_NEGATIVE: SFPLOADMACRO-based pipeline (fastest, ~2.5 cycles/element)
        // - FAST_APPROX && !CLAMP_NEGATIVE: Replay buffer version with SFPSHFT2
        // - !FAST_APPROX: _calculate_exponential_piecewise_ (Horner series + reciprocal for negatives)
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // ANALYSIS: Non-approximate path. Iterates over 8 SFPU lanes (one tile face).
        // Each iteration processes one row-pair of the 32x32 tile.
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];   // ANALYSIS: load element from DST register
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            // ANALYSIS: dispatches to _sfpu_exp_21f_ (bf16) or _sfpu_exp_f32_accurate_ (fp32)
            sfpi::dst_reg[0] = result;  // ANALYSIS: write result back to DST register
            sfpi::dst_reg++;            // ANALYSIS: advance to next SFPU lane (next row-pair)
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
    // ANALYSIS: For approx+fast: loads constants into LREGs, programs SFPLOADMACRO sequences, records replay buffer
    // For approx+!fast: loads ln2_recip and conversion constants into programmable registers
    // For !approx: initializes reciprocal lookup table (for handling negative inputs in legacy path)
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Kernel File (LLK-level, approximate path)
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`

This file contains the legacy LLK implementations used when `APPROXIMATION_MODE=true`:

- `_sfpu_exp_()`: Horner-form polynomial + repeated squaring
- `_calculate_exponential_body_()`: wraps `_sfpu_exp_` with sign handling
- `_calculate_exponential_approx_()`: fast FxP conversion approximation
- `_calculate_exponential_piecewise_()`: boundary-checked piecewise dispatch
- `_calculate_exponential_()`: top-level dispatcher with SFPLOADMACRO-based fast paths
- `_init_exponential_()`: constant loading and macro sequence programming

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::exexp(v)` | Extracts biased exponent field from IEEE-754 float |
| `sfpi::exexp_nodebias(v)` | Extracts exponent without subtracting IEEE bias |
| `sfpi::exman8(v)` | Extracts 8-bit mantissa with implicit leading 1 bit |
| `sfpi::exman9(v)` | Extracts 9-bit mantissa |
| `sfpi::setexp(v, e)` | Sets the exponent field of float v to integer e |
| `sfpi::setsgn(v, s)` | Sets the sign bit of float v |
| `sfpi::addexp(v, n)` | Adds integer n to the exponent of float v (multiply by 2^n) |
| `sfpi::shft(v, n)` | Shifts mantissa/integer by n bits |
| `sfpi::int32_to_float(v, n)` | Converts integer to float with scaling |
| `sfpi::float_to_fp16b(v, n)` | Converts FP32 to BF16 with round-to-nearest-even |
| `sfpi::vec_min_max(a, b)` | Swaps a and b so a <= b (vectorized min/max) |
| `sfpi::reinterpret<T>(v)` | Bit-reinterpret between vFloat/vInt/vUInt |
| `sfpi::dst_reg[n]` | Access to DEST register n (read/write SFPU lanes) |
| `sfpi::dst_reg++` | Advances DEST register pointer to next lane |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Evaluates polynomial c0 + c1*x + c2*x^2 + ... using Horner's method |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (per-lane conditionals via condition codes) |
| `TTI_SFPLOADMACRO(lreg, seq, rows, offset)` | Programs and executes macro sequence on SFPU pipeline |
| `TTI_SFPMAD(a, b, c, d, mod)` | Fused multiply-add: d = a * b + c |
| `TTI_SFPSHFT2(src, vc, vd, mode)` | Barrel shift operation |
| `TTI_SFP_STOCH_RND(...)` | Stochastic rounding FP32 -> INT16 |
| `TTI_SFPNOP` | SFPU pipeline NOP (timing/synchronization) |
| `TTI_SFPLOADI(lreg, mode, imm)` | Loads immediate value into LREG |
| `TTI_SFPCONFIG(val, dest, mode)` | Configures SFPU registers/macro sequences |
| `TTI_SFPSETSGN(imm, vc, vd, mod)` | Sets sign bit from source register |
| `lltt::replay(start, count)` | Replays recorded SFPU instruction sequence |
| `lltt::record(start, count)` | Records SFPU instruction sequence into replay buffer |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` | Input/output tile element; read for input, written with exp result |
| `dst_reg++` | Advances through 8 SFPU lanes (each lane = 2 rows of a 32x32 tile face) |
| `LREG[0-3]` | Working registers for SFPLOADMACRO pipeline (fast approx path) |
| `LREG[12]` | Constant A = 256/ln(2) (fast approx path) |
| `LREG[13]` | Constant B-C = 32500.818 (fast approx path) |
| `LREG[14]` | Threshold -88.5 (CLAMP path) or shift amount 15 (non-clamp path) |
| `vConstFloatPrgm0` | 1/ln(2) (non-fast approx path) |
| `vConstFloatPrgm1` | C23_73 conversion constant (non-fast approx path) |
| `vConstFloatPrgm2` | ADJ_EXP adjustment constant (non-fast approx path) |

#### SFPU Execution Flow

1. **Initialization** (`exp_init` / `_init_exponential_`):
   - For fast approximate: loads A, B-C, threshold constants into LREGs; programs macro instruction registers (SWAP, MAD, STOCHRND, SHFT); configures macro sequence registers; records replay buffer pattern
   - For non-approximate: initializes reciprocal table (`_init_sfpu_reciprocal_`)

2. **Per-tile execution** (`calculate_exponential`):
   - For `APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=true`:
     - First pass: 8x SFPLOADMACRO (sequence 1) sanitizes inputs against -88.5 threshold via SWAP
     - Second pass: 8x SFPLOADMACRO (sequence 0) computes MAD -> STOCHRND -> SHFT -> STORE
   - For `APPROXIMATION_MODE=true, FAST_APPROX=true, CLAMP_NEGATIVE=false`:
     - Replays recorded instruction buffer (16 LOADMACRO+SHFT2 pairs), then drains final 2 SHFT2s
   - For `APPROXIMATION_MODE=false` (improved path):
     - Loops 8 iterations (one per SFPU lane / row-pair)
     - Each iteration: read `dst_reg[0]` -> call `_sfpu_exp_improved_` -> write `dst_reg[0]` -> advance

3. **Per-lane math** (non-approximate, `_sfpu_exp_21f_` for bf16):
   - Multiply input by 1/ln(2), add IEEE bias (127)
   - Clamp result to [0, 255]
   - Branch-free float-to-int conversion
   - Split into exponent (integer) and mantissa (fractional) parts
   - Evaluate 2nd-degree polynomial on fractional part
   - Recombine via `setexp`
   - Round to bf16

4. **Per-lane math** (non-approximate, `_sfpu_exp_f32_accurate_` for fp32):
   - Multiply by 1/ln(2)
   - Check overflow (>128), underflow (<-127), NaN
   - Round-to-nearest-even to get integer k
   - Cody-Waite range reduction: r = x - k*ln2_hi - k*ln2_lo
   - Evaluate 7th-order Taylor series on r
   - Scale by 2^k via exponent addition

#### SFPU Configuration

| Configuration | Value | Description |
|--------------|-------|-------------|
| `math_fidelity` | `HiFi4` | Highest FPU fidelity (not directly relevant for SFPU but set in ComputeConfig) |
| `math_approx_mode` | `false` | ComputeConfig's global approx flag (always false for EXP per `get_op_approx_mode`) |
| `APPROXIMATION_MODE` template | `true` (default from Python) or `false` | Controls which SFPU implementation path is used |
| `FAST_APPROX` template | `true` (default) | Enables SFPLOADMACRO-based fast path when approx is true |
| `fp32_dest_acc_en` | From operation params | Selects between bf16 (exp_21f) and fp32 (exp_f32_accurate) implementations |
| `ITERATIONS` | `8` (default) | Number of SFPU lanes processed per `exp_tile` call (8 = one tile face) |
| `InputClamping` | `ClampToNegative` (default) | Clamps very negative inputs to -88.5 to prevent incorrect results |

#### Hardware Compatibility Notes

- The metal-level `ckernel_sfpu_exp.h` files are **identical** for Wormhole B0 and Blackhole architectures. The improved implementations (`_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`) use only SFPI intrinsics that are supported on both architectures.
- The LLK-level fast approximate path uses `TTI_SFPLOADMACRO` which is a Wormhole/Blackhole common instruction. The Blackhole SFPU specification notes that `SFPMAD` supports `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers, but the implementation intentionally negates constants instead for consistency across architectures.
- The Cody-Waite path in `_sfpu_exp_f32_accurate_` is designed with SFPMAD optimization in mind: expressions like `k * LN2_HI + val` map to a single `SFPMAD` instruction (VD = VA * VB + VC).

## Implementation Notes

1. **Parameterized vs non-parameterized EXP**: When called from Python as `ttnn.exp(x)`, the default creates `UnaryWithParam(EXP, 1.0f)` where param0=1.0 (true) enables approximate mode. The `SFPU_OP_CHAIN_0` expands to `exp_tile_init<1u>(); exp_tile<1u>(0);`. Without parameters (internal calls), it defaults to `exp_tile_init<>(); exp_tile<>(0);` which uses `approx=false`.

2. **Dual implementation hierarchy**: There are effectively two layers of exp implementations:
   - **LLK layer** (tt_llk): Legacy implementations including the SFPLOADMACRO-based ultra-fast approximation, Horner polynomial, and piecewise approach
   - **Metal layer** (hw/ckernels): Improved implementations (exp_21f, exp_f32_accurate) that override the non-approximate path

3. **Program caching**: The `override_runtime_arguments` method allows reusing compiled programs by updating only buffer addresses, avoiding recompilation when tensor addresses change but shapes remain the same.

4. **The exp_61f variant** (`_sfpu_exp_61f_`) is defined but not currently used in any dispatch path. It exists as an intermediate accuracy option between exp_21f and exp_f32_accurate.

5. **The `per_core_block_dim = 1` hardcoding** means the inner tile loop always runs once. The `cb_reserve_back` / `cb_push_back` of the output CB wraps exactly one tile per outer iteration. This simplifies the pipeline but means the output CB double-buffering advantage is only between the compute-writer pair, not within compute itself.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary elementwise operation program factory implemented? What kernels does it use?"
   **Reason**: Initial reconnaissance of the unary operation framework and kernel organization
   **Key Findings**: Confirmed 3-kernel architecture (reader, compute, writer), interleaved vs sharded factory selection, CB configuration (c_0, c_1 conditional, c_2), and the generic eltwise_sfpu.cpp compute kernel with SFPU_OP_CHAIN defines.

2. **Query**: "What is the SFPU exp operation implementation? Where is exp_tile defined?"
   **Reason**: Understanding the SFPU dispatch chain from exp_tile API through LLK to hardware
   **Key Findings**: Identified three main implementations (_sfpu_exp_21f_, _sfpu_exp_61f_, _sfpu_exp_f32_accurate_), the dispatcher pattern via _sfpu_exp_improved_, the calculate_exponential entry point, and the exp_tile API location in compute_kernel_api.h.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Understanding how EXP maps to kernel path and SFPU defines
   **Key Information**: EXP defaults to `eltwise_sfpu.cpp`; SFPU_OP_EXP_INCLUDE enables exp header; parameterized version passes approx flag as template parameter to exp_tile_init/exp_tile.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Understanding the exp_tile and exp_tile_init API signatures and template parameters
   **Key Information**: exp_tile dispatches to calculate_exponential with approx, fast_and_approx, DST_ACCUM_MODE, scale_en, iterations as template args. Default: approx=false, fast_and_approx=true, iterations=8.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Understanding the improved (metal-level) SFPU exp implementations
   **Key Information**: _sfpu_exp_21f_ uses Moroz 2022 algorithm with 2nd-degree polynomial; _sfpu_exp_f32_accurate_ uses Cody-Waite + 7th-order Taylor for <1 ULP accuracy; _sfpu_exp_improved_ dispatches based on fp32_dest_acc_en.

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Understanding the LLK-level legacy and fast approximate implementations
   **Key Information**: _calculate_exponential_ has three major paths: SFPLOADMACRO-based (fastest, ~2.5 cycles/elem), replay-buffer based (fast, no clamping), and iterative piecewise (precise). _init_exponential_ programs all constants, macro instruction registers, and replay buffers.
