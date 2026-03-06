# EXP Implementation Analysis

## Overview
The EXP operation computes the element-wise exponential function `exp(x) = e^x` on each element of an input tensor. It is implemented as a **unary SFPU operation** using the shared `UnaryProgramFactory`, which is the common program factory for all standard unary elementwise operations. The EXP operation supports multiple computation modes: a fast approximate mode using SFPU LOADMACRO-based hardware acceleration, a standard approximation mode using SFPI, and a high-accuracy improved mode that dispatches either `_sfpu_exp_21f_` (bfloat16) or `_sfpu_exp_f32_accurate_` (float32) based on `fp32_dest_acc_en`.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total tiles in input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_dim` tiles (always 1 for standard factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Any (arbitrary rank) | Same as input |
| **Dimension convention** | NHWC (or any) | Same as input |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) | Same as input |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as input (or specified output dtype) |

### Layout Transformations
No layout transformations occur within the program factory. Input and output tensors must already be in the correct layout. The CB page size adapts to either tile size (for TILE_LAYOUT) or buffer page size (for ROW_MAJOR).

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU_OP_CHAIN_0 (exp), `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

**Key detail**: The compute kernel reserves output CB space once per block (`cb_reserve_back(c_2, per_core_block_dim)` before the inner tile loop), processes all tiles in the block, then pushes all at once (`cb_push_back(c_2, per_core_block_dim)`). Since `per_core_block_dim = 1` for the standard factory, this is equivalent to single-tile push.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_1 (tmp0) is **NOT** allocated for EXP. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.
- Both input and output CBs use 2-page capacity for double buffering, enabling reader/compute and compute/writer overlap.
- For BITCAST operations, the input CB uses the output data format to avoid unpacker conversion; this does not apply to EXP.

## Pipeline Pattern Summary
- **Input CB (c_0)**: Double-buffered (capacity=2, block=1). Reader can write one tile ahead while compute processes.
- **Output CB (c_2)**: Double-buffered (capacity=2, block=1). Compute can write one tile ahead while writer drains.

## Index Calculations
The program factory uses `TensorAccessor` for mapping logical page indices to physical DRAM addresses. The reader/writer kernels use a simple sequential page index starting from `start_id`:

```
page_index = start_id, start_id+1, ..., start_id+num_pages-1
```

Physical address resolution is handled by `TensorAccessor` which encodes the buffer's bank mapping (interleaved across DRAM banks) as compile-time args.

## Memory Access Patterns

### Read Pattern
**Sequential**: The reader kernel iterates pages from `start_id` to `start_id + num_pages` in ascending order, reading one page at a time via `noc_async_read_page`. Each page maps to an interleaved DRAM bank via TensorAccessor.

### Write Pattern
**Sequential**: The writer kernel iterates pages from `start_id` to `start_id + num_pages` in ascending order, writing one page at a time via `noc_async_write_page`. Same bank mapping as reader.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D assignment) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: Group 1 gets `ceil(num_pages/num_cores)` tiles, Group 2 gets `floor(num_pages/num_cores)` tiles |

Core indexing uses column-major order: `core = {i / num_cores_y, i % num_cores_y}`.

Two separate compute kernel instances are created (one per core group) with different `per_core_block_cnt` compile-time args. If all cores get equal work, core_group_2 is empty.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded buffer metadata (bank mapping, page size, etc.) for source buffer |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded buffer metadata for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (= number of tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages/tiles to read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages/tiles to write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (always 0) |

### Preprocessor Defines (Compile-Time)

For EXP, `get_block_defines` generates these key defines:

| Define | Value (parametrized EXP with `fast_and_approx=1`) | Purpose |
|--------|------|---------|
| `SFPU_OP_EXP_INCLUDE` | `1` | Gates `#include "api/compute/eltwise_unary/exp.h"` |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro chain executed in compute kernel |
| `SFPU_OP_CHAIN_0_INIT_0` | `exp_tile_init<1u>();` | SFPU initialization call |
| `SFPU_OP_CHAIN_0_FUNC_0` | `exp_tile<1u>(0);` | SFPU compute call on tile at DST[0] |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Input data type indicator |

**Parametrized vs non-parametrized EXP**: When called from Python as `ttnn.exp(x)`, the default creates `UnaryWithParam(EXP, 1.0f)` meaning `fast_and_approx=true`. The template parameter `1u` is baked into the defines. If called without parameters (non-parametrized path), the defines use default template args: `exp_tile_init<>()` / `exp_tile<>(0)`, which expand to `approx=false, fast_and_approx=true`.

### ComputeConfig Settings

| Setting | Value | Source |
|---------|-------|--------|
| `math_fidelity` | `MathFidelity::HiFi4` | Hardcoded |
| `math_approx_mode` | `false` | `get_op_approx_mode(EXP)` returns `false` (default case) |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | User-configurable |
| `bfp8_pack_precise` | From `args.bfp8_pack_precise` | User-configurable |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 (BRISC) | NOC0 | DRAM | CB c_0 | Read tiles sequentially via TensorAccessor |
| Compute | RISCV_2 (TRISC) | N/A | CB c_0 | CB c_2 | Unpack, copy_tile, exp SFPU chain, pack_tile |
| Writer | RISCV_1 (NCRISC) | NOC1 | CB c_2 | DRAM | Write tiles sequentially via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Uses `TensorAccessor` for address resolution. Reads one page at a time into CB c_0 with a barrier after each read (`noc_async_read_barrier`). Supports optional `BACKWARDS` mode for reverse-order reading.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Waits for compute to produce one page, reads it from CB c_2, writes via NoC. Uses `noc_async_writes_flushed()` per tile and a final `noc_async_write_barrier()`. Supports `OUT_SHARDED` mode (not used for standard interleaved EXP).

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
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // ANALYSIS: number of tile-blocks to process (= number of tiles for EXP since block_dim=1)
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // ANALYSIS: tiles per block (always 1 for standard UnaryProgramFactory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // ANALYSIS: initializes unpack/pack pipeline between input CB c_0 and output CB c_2; calls unary_op_init_common
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // ANALYSIS: reserve output CB space for the entire block (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // ANALYSIS: acquire DST register file for exclusive use by compute RISC

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // ANALYSIS: block until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // ANALYSIS: unpack tile 0 from CB c_0 into DST register 0 (copies input data to DEST for SFPU processing)

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // ANALYSIS: expands to "exp_tile_init<1u>(); exp_tile<1u>(0);" for parametrized EXP
                             // exp_tile_init configures SFPU constants/macros for the exp algorithm
                             // exp_tile dispatches calculate_exponential on DST[0] for 8 iterations (one tile face)
#endif

            tile_regs_commit();  // ANALYSIS: signal that DST registers are ready for packing (compute done, pack can begin)

            tile_regs_wait();  // ANALYSIS: wait for pack unit to be ready to accept DST data

            pack_tile(0, tt::CBIndex::c_2);  // ANALYSIS: pack DST[0] result into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // ANALYSIS: free the consumed input tile from CB c_0 (allows reader to refill)

            tile_regs_release();  // ANALYSIS: release DST registers (allows next iteration's acquire)
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // ANALYSIS: publish the block's output tiles to the writer kernel
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
- **Compute API header**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
- **Metal-layer kernel (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Metal-layer kernel (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **LLK-layer kernel (Wormhole)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
- **LLK-layer kernel (Blackhole)**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`

The metal-layer files contain the improved algorithms (`_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and the `calculate_exponential` dispatcher). The LLK-layer files contain the legacy/approximation algorithms (`_sfpu_exp_`, `_calculate_exponential_`, `_init_exponential_`).

#### Annotated SFPU Kernel Source (Compute API - exp.h)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"  // ANALYSIS: includes the metal-layer SFPU exp implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"  // ANALYSIS: provides SFPU_TEMPLATE_INIT_KERNEL and SFPU_TEMPLATE_PARAMS_KERNEL_FN macros
#endif

namespace ckernel {

// ANALYSIS: InputClamping enum controls whether fast approximate exp clamps negative inputs below -88.5
enum class InputClamping : uint8_t {
    ClampToNegative = 1,  // ANALYSIS: inputs below -88.5 clamped to -88.5 (safer, default)
    None = 0,             // ANALYSIS: no clamping (faster, but incorrect for very negative inputs)
};

// ANALYSIS: exp_tile_init configures the SFPU pipeline for exp computation.
// Template params control which algorithm variant is initialized.
// For default EXP: approx=false, fast_and_approx=true (but since approx=false, the non-approx init path runs).
// For parametrized EXP with param0=1: approx=true(cast from 1u), fast_and_approx=true => fast approx LOADMACRO init.
template <
    bool approx = false,
    bool fast_and_approx = true,
    uint32_t scale = 0x3F800000,           // ANALYSIS: 1.0f in IEEE 754 (no scaling by default)
    InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential,          // ANALYSIS: SfpuType::exponential used for LLK init dispatch
        sfpu::exp_init,       // ANALYSIS: calls _init_exponential_ which sets up constants and macro registers
        approx,
        fast_and_approx,
        scale,
        (input_clamping == InputClamping::ClampToNegative)));
}

// ANALYSIS: exp_tile dispatches the SFPU exp computation on tile at DST[idst].
// The SFPU_TEMPLATE_PARAMS_KERNEL_FN macro expands to call _llk_math_eltwise_unary_sfpu_params_
// which handles DEST register walking and calls calculate_exponential with all template params.
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8>         // ANALYSIS: 8 iterations = 8 faces of 4 rows each = 32 rows = full tile column
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(
        calculate_exponential,    // ANALYSIS: the main SFPU kernel function dispatched
        approx,
        fast_and_approx,
        DST_ACCUM_MODE,           // ANALYSIS: resolves to fp32_dest_acc_en at compile time
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

#### Annotated SFPU Kernel Source (Metal-layer - ckernel_sfpu_exp.h, Wormhole)

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"       // ANALYSIS: includes the LLK-layer legacy exp functions
#include "sfpu/ckernel_sfpu_polyval.h"    // ANALYSIS: PolynomialEvaluator used by exp_21f and exp_f32_accurate
#include "sfpu/ckernel_sfpu_converter.h"  // ANALYSIS: Converter::as_float for bit-cast constants
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// ANALYSIS: Legacy SFPU exp wrapper (Horner-form polynomial + repeated squaring)
sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

// ANALYSIS: Helper for exp_21f - converts float to int32 without branching.
// Constraint: 0 <= val < 128.0f. Assumes val has been divided by 2^23 conceptually.
// Uses SFPU exexp (extract exponent) and exman8 (extract 8-bit mantissa with implicit bit),
// then shifts mantissa left by exponent to produce integer representation.
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);        // ANALYSIS: extract biased exponent from float
    sfpi::vInt man = sfpi::exman8(val);       // ANALYSIS: extract 8-bit mantissa with implicit leading 1
    man = sfpi::reinterpret<sfpi::vInt>(
        sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));  // ANALYSIS: shift mantissa left by exponent value
    return man;
}

// ANALYSIS: Moroz et al. 2022 "exp_21f" algorithm. Computes exp(x) via:
//   exp(x) = 2^(x/ln2) = 2^(x_i) * 2^(x_f)
// Uses 2nd-degree polynomial for fractional part approximation.
// This is the DEFAULT path when fp32_dest_acc_en=false.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;   // ANALYSIS: 1/ln(2)
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);       // ANALYSIS: compute x/ln2 + bias (IEEE 754 exponent bias = 127)

    // ANALYSIS: Clamp to [0, 255] to prevent overflow in intermediate computation
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);  // ANALYSIS: SFPU min/max intrinsic - ensures xlog2 >= 0
    sfpi::vec_min_max(xlog2, threshold_high);  // ANALYSIS: ensures xlog2 <= 255

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);  // ANALYSIS: convert to integer representation

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // ANALYSIS: extract exponent without debiasing = 2^(integer part)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // ANALYSIS: extract 9-bit mantissa = fractional part in [0, 1)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // ANALYSIS: convert fractional part to float

    // ANALYSIS: 2nd-degree polynomial refinement: 2^(x_f) ~ c0 + c1*x + c2*x^2
    // Operating on the raw scaled fractional part (not divided by 2^23)
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // ANALYSIS: recombine integer and fractional parts via setexp: result = frac * 2^(exponential_part)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // ANALYSIS: When DST accumulator is bfloat16, explicitly round to bf16 to avoid truncation artifacts
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// ANALYSIS: Moroz et al. 2022 "exp_61f" algorithm. Higher accuracy version using 6th-degree polynomial.
// NOT used in the default EXP path - available for explicit selection.
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
    frac = sfpi::addexp(frac, -23);  // ANALYSIS: multiply by 2^-23 to normalize fractional part to [0, 1]

    // ANALYSIS: 6th-degree polynomial: 2^x ~ 1 + 0.693*x + 0.240*x^2 + 0.055*x^3 + ...
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

// ANALYSIS: Round-to-nearest-even for the f32-accurate path.
// Uses the "magic number" trick: adding 2^23+2^22 forces rounding in float representation.
sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // ANALYSIS: 2^23 + 2^22 = 12582912.0f
    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;              // ANALYSIS: k = round(z) as float
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);  // ANALYSIS: k_int = round(z) as int
    return k;
}

// ANALYSIS: High-accuracy exp(x) using Cody-Waite range reduction + 7th-order Taylor series.
// Target: < 1 ULP for float32. Used when fp32_dest_acc_en=true.
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;    // ANALYSIS: ~2^128 in base-2 domain => exp(88.7) in natural domain
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;  // ANALYSIS: ~2^-127 => exp(-88) in natural domain

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;  // ANALYSIS: convert to base-2: z = x / ln(2)

    sfpi::vInt exp_bits = sfpi::exexp(z);  // ANALYSIS: extract exponent for NaN detection

    // ANALYSIS: SFPU conditional execution via predicated v_if/v_elseif/v_else
    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();  // ANALYSIS: saturate to +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;  // ANALYSIS: saturate to 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();  // ANALYSIS: propagate NaN
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);  // ANALYSIS: k = round(x/ln2)

        // ANALYSIS: Cody-Waite range reduction for extended precision.
        // ln(2) is split into high and low parts to minimize floating-point error.
        // Constants are negated so the expression maps to a single SFPMAD instruction: VD = VA * VB + VC
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;  // ANALYSIS: r_hi = val - k * |LN2_HI| (maps to SFPMAD)
        sfpi::vFloat r = k * LN2_LO + r_hi;     // ANALYSIS: r = r_hi - k * |LN2_LO| (Cody-Waite remainder)

        // ANALYSIS: 7th-order Taylor series: exp(r) ~ 1 + r + r^2/2! + ... + r^7/7!
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,      // 1
            sfpi::vConst1,      // 1
            0.5f,               // 1/2!
            1.0f / 6.0f,        // 1/3!
            1.0f / 24.0f,       // 1/4!
            1.0f / 120.0f,      // 1/5!
            1.0f / 720.0f,      // 1/6!
            1.0f / 5040.0f      // 1/7!
        );

        // ANALYSIS: Scale by 2^k via exponent manipulation (ldexp equivalent)
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);  // ANALYSIS: get current exponent without bias subtraction
        sfpi::vInt new_exp = p_exp + k_int;            // ANALYSIS: add k to exponent
        result = sfpi::setexp(p, new_exp);             // ANALYSIS: set new exponent => result = p * 2^k
    }
    v_endif;

    return result;
}

// ANALYSIS: Template dispatcher - selects algorithm based on fp32_dest_acc_en
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);  // ANALYSIS: bfloat16 path -> Moroz exp_21f (fast, ~1 ULP for bf16)
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // ANALYSIS: float32 path -> Cody-Waite + Taylor (< 1 ULP for fp32)
}

// ANALYSIS: Main entry point called by exp_tile via SFPU_TEMPLATE_PARAMS_KERNEL_FN.
// When APPROXIMATION_MODE=false (the non-parametrized EXP default), runs the improved path.
// When APPROXIMATION_MODE=true (parametrized EXP with approx=1), delegates to _calculate_exponential_
// in the LLK layer which uses LOADMACRO-based hardware acceleration.
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
        // ANALYSIS: Approximate mode - delegates to LLK legacy implementation which uses
        // SFPLOADMACRO hardware for ultra-fast exp computation (~2.5 cycles/element)
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // ANALYSIS: Precise mode - uses improved SFPI-based algorithms
        // Iterates ITERATIONS times (default 8), processing one SFPU lane per iteration.
        // 8 iterations * 4 rows/iteration = 32 rows = one tile column (one face pair).
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // ANALYSIS: load element from DEST register 0
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // ANALYSIS: optional input scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);  // ANALYSIS: dispatch to exp_21f or exp_f32_accurate
            sfpi::dst_reg[0] = result;  // ANALYSIS: write result back to DEST register 0

            sfpi::dst_reg++;  // ANALYSIS: advance DEST register pointer to next SFPU lane (next 4 rows)
        }
    }
}

// ANALYSIS: Initialization function paired with calculate_exponential
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
    // ANALYSIS: For APPROX+FAST_APPROX+CLAMP: programs LREG constants, macro instructions, and macro sequences
    // For APPROX+FAST_APPROX (no clamp): programs LREG constants, macro instructions, macro sequences + replay buffer
    // For APPROX only: sets vConstFloatPrgm0/1/2 registers
    // For non-approx: initializes reciprocal for negative input handling
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::exexp(val)` | Extract biased exponent from IEEE 754 float |
| `sfpi::exexp_nodebias(val)` | Extract exponent without debiasing (raw exponent bits) |
| `sfpi::exman8(val)` | Extract 8-bit mantissa with implicit leading 1 bit |
| `sfpi::exman9(val)` | Extract 9-bit mantissa (higher precision than exman8) |
| `sfpi::shft(val, amt)` | Arithmetic/logical shift by variable amount |
| `sfpi::setexp(val, exp)` | Set the exponent field of a float (effectively ldexp) |
| `sfpi::setsgn(val, sign)` | Set sign bit of float |
| `sfpi::addexp(val, delta)` | Add immediate to exponent (multiply by 2^delta) |
| `sfpi::vec_min_max(a, b)` | SIMD min/max swap between two vector registers |
| `sfpi::int32_to_float(val, mode)` | Convert integer to float |
| `sfpi::float_to_fp16b(val, mode)` | Convert float32 to bfloat16 with rounding |
| `sfpi::reinterpret<T>(val)` | Bit-cast between vFloat/vInt/vUInt types |
| `sfpi::s2vFloat16b(val)` | Convert scalar to vector float16b |
| `sfpi::dst_reg[n]` | Read/write DEST register at SFPU lane offset n |
| `sfpi::dst_reg++` | Advance DEST register pointer by one SFPU lane |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Horner-form polynomial evaluation |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated conditional execution |
| `v_and(cond)` | Narrow predication mask (used in repeated squaring loop) |
| `TTI_SFPLOADMACRO(lreg, seq, addr_mod, dest_offset)` | Hardware macro instruction execution (fast approx path) |
| `TTI_SFPMAD(va, vb, vc, vd, mod)` | SFPU multiply-accumulate |
| `TTI_SFP_STOCH_RND(...)` | Stochastic rounding (FP32 to INT16 conversion) |
| `TTI_SFPSHFT(imm, vc, vd, mod)` | SFPU shift |
| `TTI_SFPSHFT2(va, vb, vc, mod)` | SFPU shift2 (used in replay buffer path) |
| `TTI_SFPSETSGN(imm, vc, vd, mod)` | Set sign bit (used in non-clamping fast approx) |
| `TTI_SFPLOADI(lreg, target, imm)` | Load immediate to SFPU register |
| `TTI_SFPCONFIG(val, dest, mode)` | Configure SFPU registers/macro instruction slots |
| `TTI_SFPNOP` | SFPU no-operation (pipeline delay) |
| `lltt::record(slot, count)` | Record instructions into SFPU replay buffer |
| `lltt::replay(slot, count)` | Replay recorded instructions from buffer |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` | Current DEST register - input/output for each SFPU lane iteration |
| `LREG[0-3]` | Working registers for LOADMACRO operations (fast approx path) |
| `LREG[4]` | Staging register for SFPSHFT2 output (non-clamping fast approx) |
| `LREG[12]` | Constant A = 256/ln(2) = 369.33 (fast approx path) |
| `LREG[13]` | Constant (B-C) = 32500.82 (fast approx path) |
| `LREG[14]` | Threshold -88.5 (clamping path) or shift amount 15 (non-clamping path) |
| `LREG[16]` | Staging register for SETSGN output (non-clamping fast approx) |
| `vConstFloatPrgm0` | 1/ln(2) = 1.442695 (standard approx path) |
| `vConstFloatPrgm1` | C23_73 constant (standard approx path) |
| `vConstFloatPrgm2` | ADJ_EXP adjustment constant (standard approx path) |

#### SFPU Execution Flow

The execution depends on the template parameters `APPROXIMATION_MODE` and `FAST_APPROX`:

**Path 1: Default parametrized EXP (`approx=true, fast_and_approx=true, CLAMP_NEGATIVE=true`)**

This is the path taken when `ttnn.exp(x)` is called (which passes `param0=1.0f` cast to `approx=true`):

1. **Init** (`_init_exponential_`): Programs LREG constants (A, B-C, threshold), macro instruction registers (SWAP, MAD, ROUND, SHIFT), and two macro sequences (sanitize + compute).
2. **Compute** (`_calculate_exponential_`): Executes two phases per tile face:
   - **Sanitize phase**: 8x SFPLOADMACRO with sequence 1 (LD, SWAP, STORE) - clamps each DEST element to >= -88.5
   - **Compute phase**: 8x SFPLOADMACRO with sequence 0 (LD, MAD, ROUND, SHIFT, STORE) - computes Schraudolph approximation
3. Each LOADMACRO operates on a 4-row, 2-column slice of the DEST tile face.

**Path 2: Default parametrized EXP without clamping (`approx=true, fast_and_approx=true, CLAMP_NEGATIVE=false`)**

1. **Init**: Programs LREG constants (A, B-C, shift=15), macro instructions (MAD, STOCHRND, SETSGN), macro sequence 0, and records 32-instruction replay buffer.
2. **Compute**: Uses replay buffer for pipelined execution:
   - Replays 16 LOADMACRO+SHFT2 pairs (first 2 SHFT2s are pipeline priming dummies)
   - Drain phase: 2 final SHFT2 instructions
   - ~2.5 cycles/element for 8-element version, ~2.125 cycles/element for 32-element version

**Path 3: Non-approximate improved EXP (`approx=false` - default template args)**

1. **Init**: Initializes reciprocal function (for negative input handling in legacy `_sfpu_exp_` path, though this path actually uses the improved algorithms)
2. **Compute** (`calculate_exponential` with `APPROXIMATION_MODE=false`):
   - Loops 8 times (ITERATIONS=8), once per SFPU lane
   - Each iteration: loads from `dst_reg[0]`, calls `_sfpu_exp_improved_`, stores back, advances `dst_reg`
   - If `fp32_dest_acc_en=false`: uses `_sfpu_exp_21f_` (Moroz 2022, 2nd-degree polynomial, bf16 accuracy)
   - If `fp32_dest_acc_en=true`: uses `_sfpu_exp_f32_accurate_` (Cody-Waite + 7th-order Taylor, < 1 ULP for fp32)

**Path 4: Standard approximation (`approx=true, fast_and_approx=false`)**

1. **Init**: Sets `vConstFloatPrgm0/1/2` programmable registers
2. **Compute**: Loops over DEST elements, calls `_calculate_exponential_piecewise_` which:
   - Saturates to +inf for inputs >= 89
   - Saturates to 0 for inputs < -42
   - Otherwise calls `_calculate_exponential_approx_` (bit-manipulation approximation)

#### SFPU Configuration

| Setting | Value | Effect |
|---------|-------|--------|
| `math_approx_mode` | `false` | `get_op_approx_mode(EXP)` always returns false. The SFPU approximation is controlled by the operation's template parameter, not this global flag. |
| `math_fidelity` | `HiFi4` | Highest math fidelity setting. Affects FPU matrix operations but not SFPU directly. |
| `fp32_dest_acc_en` | User-configurable | When true, DEST registers hold float32; selects `_sfpu_exp_f32_accurate_`. When false (default), DEST holds bf16; selects `_sfpu_exp_21f_`. |
| `unpack_to_dest_mode` | `Default` or `UnpackToDestFp32` | When `preserve_fp32_precision=true`, tiles are unpacked directly to DEST in fp32 format. |

#### Hardware Compatibility Notes

- **Wormhole vs Blackhole**: The metal-layer `ckernel_sfpu_exp.h` files are **functionally identical** between architectures. The only difference is a comment about SFPMAD: Wormhole supports `VD = VA * VB + VC` only, while Blackhole's `SFFPMAD` adds `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers. However, the code uses the same negated-constant approach for both architectures for consistency.
- **LLK-layer**: The `tt_llk_wormhole_b0` and `tt_llk_blackhole` versions of `ckernel_sfpu_exp.h` are **structurally identical**. The Blackhole version uses `ADDR_MOD_7` symbolic constants in SFPLOADMACRO calls instead of literal `3`, but the values are the same.
- Both architectures support the same SFPU instructions (SFPLOADMACRO, SFPMAD, SFP_STOCH_RND, SFPSHFT, SFPSHFT2, SFPSETSGN, replay buffers).

## Implementation Notes

1. **EXP is a parametrized unary op**: `is_parametrized_type(UnaryOpType::EXP)` returns true. The parameter (index 0) is a float encoding the `fast_and_approx` boolean. When `ttnn.exp(x)` is called with `fast_and_approximate_mode=True` (the default), `UnaryWithParam(EXP, 1.0f)` is created, and the defines expand to `exp_tile_init<1u>()` / `exp_tile<1u>(0)`. Since the first template parameter is `approx`, this means `approx=true`, `fast_and_approx=true` (default), triggering the LOADMACRO fast path.

2. **Runtime args unused for EXP**: `packed_scalar1` and `packed_scalar2` are both 0 because EXP doesn't fall into the HARDSHRINK, WHERE_TSS, or LOGIT special-casing branches. The EXP parameters are encoded entirely as compile-time template parameters in the preprocessor defines.

3. **Program caching**: The `override_runtime_arguments` method only updates `src_buffer->address()` and `dst_buffer->address()`, enabling efficient re-execution when tensor addresses change but shapes remain the same.

4. **Schraudolph fast exp algorithm**: The fast approximate path is based on the insight that IEEE 754 bit patterns are approximately linear in log2(value). The algorithm computes `i = A*x + (B-C)` and reinterprets the result as a float, where A = 256/ln(2) and B-C is an error-minimizing bias term. This achieves ~2.5 cycles/element throughput using SFPU hardware macro instructions.

5. **The SubCoreGrid variant**: `UnarySubCoreGridProgramFactory` is an alternative factory that uses a user-specified core grid instead of the full compute grid. It uses uniform work distribution (requires tiles evenly divisible by cores) and supports blocking with `ntiles_per_block > 1`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary SFPU operation (like exp) implemented end-to-end?"
   **Reason**: To understand the overall program factory structure, kernel dispatch, and CB setup for unary operations.
   **Key Findings**: Confirmed the 3-kernel pattern (reader/compute/writer), double-buffered CBs, SFPU_OP_CHAIN_0 macro injection pattern, and the generic eltwise_sfpu.cpp compute kernel shared by most unary SFPU ops.

2. **Query**: "What is the SFPU exp kernel implementation? Where is exp_tile defined?"
   **Reason**: To trace the full call chain from exp_tile through LLK APIs to the SFPU math functions.
   **Key Findings**: Identified the dispatch chain: `exp_tile` -> `SFPU_TEMPLATE_PARAMS_KERNEL_FN` -> `_llk_math_eltwise_unary_sfpu_params_` -> `calculate_exponential` -> `_sfpu_exp_improved_` (or `_calculate_exponential_`). Confirmed three algorithm variants: `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, and `_calculate_exponential_` (legacy/fast).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To determine how EXP maps to compute kernel path and preprocessor defines.
   **Key Information**: EXP defaults to `eltwise_sfpu.cpp`; SFPU_OP_EXP_INCLUDE enables exp header; parameterized version passes approx flag as template parameter to exp_tile_init/exp_tile. `get_op_approx_mode` returns false for all ops by default.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: To understand the compute API layer and template parameter flow.
   **Key Information**: `exp_tile_init` and `exp_tile` are thin wrappers that use SFPU macro dispatch. Template parameters control algorithm selection (approx, fast_and_approx, scale_en, input_clamping, iterations).

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: To document the improved SFPU exp algorithms (exp_21f, exp_61f, exp_f32_accurate).
   **Key Information**: Three accuracy tiers: exp_21f (bfloat16, ~2 ULP), exp_61f (higher accuracy, not default), exp_f32_accurate (Cody-Waite + 7th-order Taylor, < 1 ULP for fp32).

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: To document the legacy/fast approximation SFPU exp using LOADMACRO hardware.
   **Key Information**: The FAST_APPROX + APPROXIMATION_MODE + CLAMP path uses SFPLOADMACRO with two macro sequences (sanitize + compute). The non-clamping variant uses replay buffers for even higher throughput.
