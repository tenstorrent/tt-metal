# EXP (Exponential) Implementation Analysis

## Overview

The EXP operation computes element-wise `e^x` for each element of an input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory` program factory, which handles all standard unary element-wise operations through a common three-kernel pipeline (reader, compute, writer) with operation-specific behavior injected via preprocessor defines.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) or row (for ROW_MAJOR layout) |
| **Unit size** | 1 page (1 tile for TILE layout, 1 row for ROW_MAJOR) |
| **Total units** | `input.buffer()->num_pages()` |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt`), inner loop over tiles-per-block (`per_core_block_dim` = 1 for the standard factory) |

In the standard `UnaryProgramFactory`, `per_core_block_dim` is always 1, so the compute kernel processes one tile at a time. The `per_core_block_cnt` varies per core group and equals the number of pages assigned to that core.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Any (flattened to pages) | Same as input |
| **Dimension convention** | N/A (treated as flat page stream) | N/A |
| **Tensor layout** | TILE_LAYOUT or ROW_MAJOR | Same as input |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16 / FLOAT32 / INT32 / UINT32 | BFLOAT16 / FLOAT32 (output dtype may differ) |

### Layout Transformations

No explicit tilize/untilize or reshard operations are performed. The input and output are expected to already be in the correct layout. The program factory supports both TILE and ROW_MAJOR layouts transparently through page-based addressing. For TILE layout, page size equals tile size; for ROW_MAJOR, page size comes from the buffer.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (`reader_unary_interleaved_start_id.cpp`) | DRAM/L1 (src_buffer via NoC) | CB c_0 | `cb_reserve_back(c_0, 1)` -> NoC read -> `cb_push_back(c_0, 1)` |
| 2 | Compute (`eltwise_sfpu.cpp`) | CB c_0 | CB c_2 | `cb_reserve_back(c_2, 1)` -> `cb_wait_front(c_0, 1)` -> `copy_tile` -> SFPU EXP -> `pack_tile` -> `cb_pop_front(c_0, 1)` -> `cb_push_back(c_2, 1)` |
| 3 | Writer (`writer_unary_interleaved_start_id.cpp`) | CB c_2 | DRAM/L1 (dst_buffer via NoC) | `cb_wait_front(c_2, 1)` -> NoC write -> `cb_pop_front(c_2, 1)` |

The flow is strictly sequential per tile: read one tile from DRAM into CB c_0, compute EXP on it and write to CB c_2, then write from CB c_2 back to DRAM. The double-buffered CBs allow overlap between consecutive tile operations.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_1 | tmp0 | Intermediate (conditional) | 2 pages | 1 page | Double | N/A for EXP | N/A for EXP | N/A (not created for EXP) |
| c_2 | output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

**Note**: CB c_1 is only created for `HARDSHRINK`, `CBRT`, or `LOGIT` operations. For EXP, only c_0 and c_2 are allocated.

- For TILE layout: page size = `tile_size(data_format)` (e.g., 2048 bytes for BFLOAT16 32x32 tiles)
- For ROW_MAJOR: page size = `buffer->page_size()`
- Input CB data format may differ from input tensor format for BITCAST operations (not relevant to EXP)

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 pages and block size = 1 page, classifying them as **double-buffered**. This allows:
- Reader can write tile N+1 into c_0 while compute processes tile N
- Writer can write tile N to DRAM from c_2 while compute produces tile N+1

## Index Calculations

The reader and writer kernels use `TensorAccessor` for address computation. Pages are addressed by a linear page index starting from `start_id` and incrementing sequentially to `start_id + num_pages`.

- `TensorAccessorArgs` encodes the buffer's memory layout (interleaved bank mapping) as compile-time arguments
- `noc_async_read_page(i, accessor, l1_addr)` translates page index `i` to a physical (NoC x, y, address) tuple via the accessor
- For interleaved layout, page `i` maps to bank `i % num_banks` at offset `(i / num_banks) * page_size`

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in order from `start_id` to `start_id + num_pages - 1`
- **One page at a time**: Each iteration reads exactly one page via `noc_async_read_page` with a barrier after each read
- **DRAM or L1**: Source buffer location determines NoC access path

### Write Pattern
- **Sequential**: Pages are written in the same order as read
- **One page at a time**: Each iteration writes one page via `noc_async_write_page` with flush after each write and a final barrier
- **DRAM or L1**: Destination buffer location determines NoC access path

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent, e.g., 8x8 = 64 cores) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` pages |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` pages, group 2 gets `floor(num_pages / num_cores)` pages |

The `split_work_to_cores` utility divides the total page count across cores. If the pages don't divide evenly, `core_group_1` contains cores with one extra page, and `core_group_2` contains cores with one fewer. Each group gets its own compute kernel instance (with different `per_core_block_cnt` compile-time args).

Core linearization: `core = {i / num_cores_y, i % num_cores_y}` (column-major ordering).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs (src_buffer) | uint32_t[] | Encodes buffer type, memory layout, bank mapping for source tensor |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB ID for output (c_2 = 2) |
| 1+ | TensorAccessorArgs (dst_buffer) | uint32_t[] | Encodes buffer type, memory layout, bank mapping for destination tensor |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core will read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core will write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Packed scalar parameter (0 for EXP unless parameterized) |
| 1 | packed_scalar2 | uint32_t | Packed scalar parameter (0 for EXP) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader (`reader_unary_interleaved_start_id.cpp`) | BRISC (RISCV_0) | NOC0 | DRAM/L1 src_buffer | CB c_0 | Sequential page reads via TensorAccessor |
| compute (`eltwise_sfpu.cpp`) | TRISC (RISCV_2/3/4) | N/A | CB c_0 | CB c_2 | `copy_tile` + SFPU exp_tile + `pack_tile` |
| writer (`writer_unary_interleaved_start_id.cpp`) | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst_buffer | Sequential page writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple loop reading pages from DRAM to CB c_0. Uses `TensorAccessor` with compile-time args for address resolution. Supports optional `BACKWARDS` define for reverse iteration (not used for EXP). One `noc_async_read_barrier()` per page ensures data is available before pushing.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Waits for compute to produce each page in CB c_2, then writes to DRAM. Uses `noc_async_writes_flushed()` per page and a final `noc_async_write_barrier()`. Supports `OUT_SHARDED` define for sharded output (simple `cb_wait_front` for all pages) and `BACKWARDS` define.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: The generic unary SFPU compute kernel. Operation-specific behavior is injected via the `SFPU_OP_CHAIN_0` preprocessor macro, which expands to the init and function calls for the specific operation.

## SFPU Kernel Implementation

### SFPU Kernel File

The SFPU exp kernel has a multi-layered implementation:

1. **Compute API**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` - provides `exp_tile_init()` and `exp_tile()` user-facing APIs
2. **Metal LLK bridge (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` - Wormhole-specific wrapper calling the LLK
3. **Metal LLK bridge (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` - Blackhole-specific wrapper (identical to WH for the improved path)
4. **LLK core (Wormhole)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h` - Low-level SFPU kernel with approximation and fast-approx paths

### Compute Kernel Source Code

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp
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
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);

            copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, tt::CBIndex::c_2);

            cb_pop_front(tt::CBIndex::c_0, 1);

            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

For EXP, the `SFPU_OP_CHAIN_0` macro expands to (with `param0` being the `fast_and_approx` flag from the parametrized EXP op, typically `1`):

```cpp
// SFPU_OP_CHAIN_0 expands to:
exp_tile_init<param0>();   // e.g., exp_tile_init<1u>();
exp_tile<1u>(0);           // e.g., exp_tile<1u>(0);
```

This is generated by `get_op_init_and_func` in `unary_op_utils.cpp` (line 244-247):
```cpp
case UnaryOpType::EXP:
    return {
        fmt::format("exp_tile_init<{}u>();", (uint32_t)param0),
        fmt::format("exp_tile<{1}u>({0});", idst, (uint32_t)param0)};
```

Where `param0` is the first parameter of the `UnaryWithParam` for EXP, which controls `fast_and_approx`. The `idst` is `"0"` (tile index in DST register).

### SFPU Kernel Source Code (Metal LLK Bridge - Wormhole)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man = sfpi::exman8(val);
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

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
    frac = sfpi::addexp(frac, -23);

    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);
    return k;
}

sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;

    sfpi::vInt exp_bits = sfpi::exexp(z);

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
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;
        sfpi::vFloat r = k * LN2_LO + r_hi;

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

        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);
}

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
        _calculate_exponential_<
            APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX,
            SKIP_POSITIVE_CHECK, CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

The EXP implementation uses a rich set of SFPI (SFPU Programming Interface) instructions. The specific instructions used depend on which code path is taken:

#### Common SFPI Instructions (all paths)

| Instruction | Description |
|-------------|-------------|
| `sfpi::dst_reg[0]` | Load/store a 32-element vector from/to the DEST register file at the current write pointer |
| `sfpi::dst_reg++` | Advance the DEST register pointer by 32 elements (one SFPU lane width) |
| `sfpi::exexp(val)` | Extract the biased exponent field from a float |
| `sfpi::setexp(val, exp)` | Set the exponent field of a float to a given value |
| `sfpi::reinterpret<T>(val)` | Reinterpret bits between vFloat/vInt/vUInt without conversion |
| `sfpi::s2vFloat16b(val)` | Convert a scalar uint16 BF16 value to an SFPU vFloat |

#### `_sfpu_exp_21f_` Path (non-approx, BF16 dest) - Moroz et al. 2022 "exp_21f"

| Instruction | Description |
|-------------|-------------|
| `sfpi::vec_min_max(a, b)` | Swap a and b such that a = min(a,b), b = max(a,b) - used for clamping |
| `sfpi::exman8(val)` | Extract mantissa with implicit bit (8-bit mantissa + implicit 1) |
| `sfpi::shft(val, exp)` | Shift a value left/right by the amount in exp |
| `exexp_nodebias(val)` | Extract exponent without removing bias |
| `sfpi::exman9(val)` | Extract 9-bit mantissa |
| `sfpi::int32_to_float(val, 0)` | Convert integer to float |
| `PolynomialEvaluator::eval(...)` | Evaluate polynomial using Horner's method (2nd degree for 21f) |
| `sfpi::float_to_fp16b(val, 0)` | Convert float32 to BF16 with round-to-nearest-even |

#### `_sfpu_exp_f32_accurate_` Path (non-approx, FP32 dest) - Cody-Waite + Taylor series

| Instruction | Description |
|-------------|-------------|
| `v_if / v_elseif / v_else / v_endif` | SFPU conditional execution blocks (lane-level predication) |
| `sfpi::vConst0` / `sfpi::vConst1` | SFPU hardware constants (0.0f and 1.0f) |
| `sfpi::exexp_nodebias(val)` | Extract exponent without bias removal |
| `PolynomialEvaluator::eval(...)` | 7th-degree Taylor series polynomial evaluation |

#### `_calculate_exponential_` Approximation Paths (LLK layer, Wormhole-specific)

| Instruction | Description |
|-------------|-------------|
| `sfpi::vConstFloatPrgm0/1/2` | Programmable SFPU constants |
| `sfpi::setsgn(val, 0)` | Force sign bit to positive |
| `_sfpu_reciprocal_<2>(val)` | Compute 1/val with 2 Newton-Raphson iterations |
| `TTI_SFPLOADMACRO(...)` | Load and execute a macro instruction sequence from DEST |
| `TTI_SFPNOP` | No-operation (pipeline timing) |
| `TTI_SFPSHFT2(...)` | Shift operation (mode 5: shift by VC register amount) |
| `TTI_SFPMAD(...)` | Multiply-accumulate: VD = VA * VB + VC |
| `TTI_SFP_STOCH_RND(...)` | Stochastic rounding (FP32 to INT16 conversion) |
| `TTI_SFPSHFT(...)` | Shift left by immediate amount |
| `TTI_SFPLOADI(...)` | Load immediate value into LREG |
| `TTI_SFPCONFIG(...)` | Configure SFPU registers and macro sequences |
| `TTI_SFPSETSGN(...)` | Set sign bit from another register |
| `lltt::replay(offset, count)` | Replay instructions from the replay buffer |
| `lltt::record(offset, count)` | Record instructions into the replay buffer |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` | Current 32-element vector being processed; read input, write output |
| `LREG[0-3]` | Working registers for macro instruction sequences (fast approx path) |
| `LREG[12]` | Constant A = 256/ln(2) for Schraudolph approximation |
| `LREG[13]` | Constant (B-C) = 32500.818... for Schraudolph approximation |
| `LREG[14]` | Threshold value (-88.5 for clamping) or shift amount (15 for SFPSHFT2) |
| `LREG[16]` | Staging register for macro output (avoids write port conflicts) |
| `vConstFloatPrgm0` | 1/ln(2) = 1.442695f (standard approx path) |
| `vConstFloatPrgm1` | p_exp::C23_73 constant (standard approx path) |
| `vConstFloatPrgm2` | p_exp::ADJ_EXP adjustment constant (standard approx path) |

### SFPU Execution Flow

The execution flow depends on the template parameters. For the default EXP operation with `fast_and_approx=true` and `approx=false`:

**Step 1: Initialization (`exp_tile_init`)**
1. `exp_tile_init<false, true>()` is called (approx=false, fast_and_approx=true is the default but irrelevant when approx=false)
2. This calls `_init_exponential_<false, true, 0x3F800000, true>()`
3. Since `APPROXIMATION_MODE=false`, it calls `_init_sfpu_reciprocal_<false>()` (prepares for potential reciprocal in precise mode)

**Step 2: Tile Processing (per tile)**
1. `tile_regs_acquire()` - Acquire DST register bank for exclusive write access
2. `cb_wait_front(c_0, 1)` - Wait for reader to produce one tile in CB c_0
3. `copy_tile(c_0, 0, 0)` - Unpack tile from CB c_0 into DST register at index 0
4. `exp_tile<false, true>(0)` is called, which expands via `SFPU_TEMPLATE_PARAMS_KERNEL_FN` to call `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_exponential<false, true, DST_ACCUM_MODE, false, 8, false, true>, 0, VectorMode::RC, scale)`
5. Inside `_llk_math_eltwise_unary_sfpu_params_`:
   - Sets DST write address for tile index 0
   - Stalls until SFPU is ready
   - In RC vector mode: loops over 4 faces (each face = 8 iterations of 32 elements = 256 elements; 4 faces = 1024 elements = one 32x32 tile)
   - Each face calls `calculate_exponential(...)`:
     - Since `APPROXIMATION_MODE=false`: loops 8 iterations
     - Each iteration: reads `dst_reg[0]`, computes `_sfpu_exp_improved_<DST_ACCUM_MODE>(val)`, writes back to `dst_reg[0]`, increments `dst_reg++`
     - If `DST_ACCUM_MODE=false` (BF16): uses `_sfpu_exp_21f_<false>` (Moroz exp_21f algorithm with 2nd-degree polynomial)
     - If `DST_ACCUM_MODE=true` (FP32): uses `_sfpu_exp_f32_accurate_` (Cody-Waite + 7th-order Taylor series)
   - After each face, advances DST pointer by 16 rows via `TTI_SETRWC`
6. `tile_regs_commit()` - Signal that DST is ready for packing
7. `tile_regs_wait()` - Wait for packer to be ready
8. `pack_tile(0, c_2)` - Pack tile from DST register 0 into CB c_2
9. `cb_pop_front(c_0, 1)` - Free the input tile slot in CB c_0
10. `tile_regs_release()` - Release DST register bank

**Step 3: Output Push**
After all tiles in a block are processed: `cb_push_back(c_2, per_core_block_dim)` pushes the completed tiles to the writer.

### SFPU Configuration

| Configuration | Value | Description |
|--------------|-------|-------------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest math fidelity for FPU operations |
| `fp32_dest_acc_en` | `args.fp32_dest_acc_en` | If true, DST accumulator uses FP32; selects accurate exp path |
| `math_approx_mode` | `false` (for EXP) | `get_op_approx_mode(EXP)` returns `false` |
| `bfp8_pack_precise` | `args.bfp8_pack_precise` | Controls BFP8 packing precision |
| `SFPU_OP_EXP_INCLUDE` | `"1"` | Conditionally includes `api/compute/eltwise_unary/exp.h` |
| `unpack_to_dest_mode` | `Default` or `UnpackToDestFp32` | If `preserve_fp32_precision` is set, uses FP32 unpack |

### Hardware Compatibility Notes

The Wormhole and Blackhole Metal LLK bridge files (`ckernel_sfpu_exp.h`) are **identical** for the improved paths (`_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and `calculate_exponential`). The key differences are in the **LLK core layer** (`tt_llk_wormhole_b0` vs `tt_llk_blackhole`):

1. **Wormhole** uses `SFPMAD` which can only do `VD = VA * VB + VC`. The Cody-Waite step pre-negates `LN2_HI` and `LN2_LO` so that the subtraction becomes an addition, enabling single-instruction MAD.
2. **Blackhole** has `SFFPMAD` with `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers, but the implementation keeps negated constants for consistency.
3. The `_sfpu_exp_` legacy function (Horner-form with squaring loop) is identical on both architectures, defined in the LLK layer.
4. The fast approximation path with `TTI_SFPLOADMACRO` and replay buffers is Wormhole-specific (from the LLK core). Blackhole may have different macro instruction encoding.

#### Algorithm Selection Matrix

| `approx` | `fp32_dest_acc_en` | Algorithm | Accuracy |
|-----------|-------------------|-----------|----------|
| `false` | `false` | `_sfpu_exp_21f_<false>` (Moroz exp_21f) | ~21 bits (BF16-sufficient) |
| `false` | `true` | `_sfpu_exp_f32_accurate_` (Cody-Waite + Taylor-7) | < 1 ULP for FP32 |
| `true` + `fast_approx` + `clamp_neg` | any | Fast macro-based Schraudolph with input clamping | ~8-bit exponent accuracy |
| `true` + `fast_approx` + no clamp | any | Fast macro-based Schraudolph without clamping | ~8-bit, needs external ReLU |
| `true` + not `fast_approx` | any | `_calculate_exponential_piecewise_` | ~10 bits |

## Implementation Notes

1. **Shared program factory**: EXP shares `unary_program_factory.cpp` with all other unary SFPU operations. The operation-specific behavior is entirely determined by preprocessor defines injected at kernel compile time. This means the program structure (CBs, core distribution, data flow) is identical across all unary SFPU ops.

2. **Parameterized EXP**: The EXP operation is parameterized - `is_parametrized_type(UnaryOpType::EXP)` returns `true`. The parameter controls `fast_and_approx` mode. When called without parameters, the non-parameterized path is used with default template args.

3. **Operation chaining**: The `SFPU_OP_CHAIN_0` macro supports chaining multiple unary operations. For single EXP, only `SFPU_OP_CHAIN_0_INIT_0` and `SFPU_OP_CHAIN_0_FUNC_0` are defined.

4. **`math_approx_mode`**: Despite having approximation paths, `get_op_approx_mode(EXP)` returns `false`. The approximation mode is controlled by the operation's parameter, not the global math_approx_mode setting.

5. **Data format handling**: Input dtype affects defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, `INP_FLOAT`) which may influence unpacking/packing behavior in the compute pipeline.

6. **Program caching**: `override_runtime_arguments` only updates buffer addresses, not page counts or start IDs. This means the program can be reused across calls with different buffer allocations but the same tensor shape.

7. **Moroz exp_21f algorithm**: Based on "Simple Multiple Precision Algorithms for Exponential Functions" by Moroz et al. 2022. The key insight is computing `exp(x) = 2^(x/ln2)` by splitting into integer and fractional parts, then refining the fractional part with a 2nd-degree polynomial. This is the default path for BF16 operations.

8. **Cody-Waite range reduction**: Used in the FP32-accurate path. Splits `ln(2)` into high and low parts (`LN2_HI` + `LN2_LO`) to preserve precision when computing `r = x - k*ln(2)`. This is crucial for sub-ULP accuracy in FP32.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary eltwise SFPU operation program factory structured? What kernels does it use (reader, compute, writer), how are circular buffers configured, and how is core distribution handled? Specifically for operations like exp."
   **Reason**: Initial reconnaissance to understand the overall program factory architecture before reading source code.
   **Key Findings**: Confirmed three-kernel pipeline (reader, compute, writer), identified kernel file paths, documented CB configuration (c_0 input, c_2 output, c_1 conditional temp), and core distribution via `split_work_to_cores`.

2. **Query**: "What is the eltwise_sfpu compute kernel and how does it work? How does it invoke SFPU operations like exp? What are the key APIs: init_sfpu, process_sfpu?"
   **Reason**: Understanding the compute kernel's execution model and API usage.
   **Key Findings**: Documented the tile processing loop (`tile_regs_acquire` -> `copy_tile` -> SFPU op -> `tile_regs_commit` -> `tile_regs_wait` -> `pack_tile` -> `tile_regs_release`). Confirmed `SFPU_OP_CHAIN_0` macro injection mechanism.

3. **Query**: "How is the exponential (exp) SFPU operation implemented in the LLK layer? What functions are called, what SFPU instructions are used, and how does the tile processing work?"
   **Reason**: Deep understanding of the SFPU-level implementation for the exp function.
   **Key Findings**: Identified the `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, `_calculate_exponential_` functions and their algorithm selection based on `APPROXIMATION_MODE` and `is_fp32_dest_acc_en`. Documented SFPI instruction usage and the iteration model (8 iterations x 32 elements = 256 elements per face).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Understanding how EXP-specific defines and kernel path are generated.
   **Key Information**: EXP sets `SFPU_OP_EXP_INCLUDE=1`, uses `eltwise_sfpu.cpp` compute kernel (default path), generates `exp_tile_init<N>()` and `exp_tile<N>(0)` calls.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Understanding the compute API layer for exp.
   **Key Information**: `exp_tile_init` calls `SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, ...)` and `exp_tile` calls `SFPU_TEMPLATE_PARAMS_KERNEL_FN(calculate_exponential, ...)`. Documented template parameters and their effects.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Understanding the lowest-level SFPU implementation including fast approximation with macro sequences.
   **Key Information**: Documented the Schraudolph fast approximation algorithm, macro instruction programming, replay buffer usage, and the three-tier approximation strategy (fast macro, standard approx, precise).

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understanding how the SFPU function is dispatched across tile faces.
   **Key Information**: In `VectorMode::RC` mode, the SFPU function is called 4 times (once per face), with DST pointer advanced by 16 rows between faces via `TTI_SETRWC`.
