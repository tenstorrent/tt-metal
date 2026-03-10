# TANH Implementation Analysis

## Overview
The TANH operation computes the element-wise hyperbolic tangent function: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`. It is a unary SFPU operation implemented through the shared `UnaryProgramFactory`, which dispatches to the generic `eltwise_sfpu.cpp` compute kernel with TANH-specific defines injected at compile time.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition
One work unit is **one tile** (32x32 elements). The total number of tiles (`num_pages`) from the input tensor is distributed across available Tensix cores. Each core processes its assigned tiles sequentially, one tile at a time through the reader-compute-writer pipeline.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Value |
|---|---|
| **Dimension Convention** | NHWC (flattened to 1D page stream) |
| **Tensor Layout** | TILE (32x32) or ROW_MAJOR |
| **Memory Layout** | Interleaved |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Output Tensor(s)

| Property | Value |
|---|---|
| **Dimension Convention** | Same as input |
| **Tensor Layout** | Same as input |
| **Memory Layout** | Interleaved |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | Same as input (may differ for certain dtype combinations) |

### Layout Transformations
No layout transformations are performed. Input and output share the same layout. The compute kernel operates in-register on tiles unpacked from CB c_0 and packs results into CB c_2.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from the input buffer (DRAM/L1) via NoC into circular buffer `c_0` (CB input).
2. **Compute kernel** waits for a tile in `c_0`, copies it to DEST registers via `copy_tile`, executes the SFPU tanh operation via `tanh_tile<0u>(0)`, then packs the result from DEST into `c_2` (CB output).
3. **Writer kernel** waits for a tile in `c_2`, reads it from L1, and writes it to the output buffer (DRAM/L1) via NoC.

The pipeline processes tiles in a streaming fashion: reader pushes one tile into `c_0`, compute consumes it and produces one tile into `c_2`, writer drains `c_2`.

## Circular Buffer Configuration

| CB ID | Index | Purpose | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` | `CBIndex::c_0` | Input tiles from reader | `single_tile_size` (input dtype) | 2 | 2 * tile_size | Double-buffered | Reader | Compute |
| `c_2` | `CBIndex::c_2` | Output tiles to writer | `single_tile_size_output` (output dtype) | 2 | 2 * tile_size | Double-buffered | Compute | Writer |

**Note**: CB `c_1` (tmp0) is NOT created for TANH. It is only created for HARDSHRINK or LOGIT operations.

## Pipeline Pattern Summary
Both input and output circular buffers are configured with 2 pages and operate with a block size of 1 tile. This enables **double-buffering**: the reader can fill one slot in `c_0` while the compute kernel processes from the other slot, and similarly the compute kernel can fill one slot in `c_2` while the writer drains the other. This allows overlap between the three pipeline stages.

## Index Calculations
Tiles are addressed sequentially by a linear page index. Each core receives a `start_id` (the first tile index it is responsible for) and a `num_pages` count. The reader and writer iterate from `start_id` to `start_id + num_pages`, using `TensorAccessor` to map linear page indices to physical NoC addresses via `noc_async_read_page` and `noc_async_write_page`. The `TensorAccessor` handles bank interleaving automatically.

## Memory Access Patterns

### Read Pattern
**Sequential**: The reader kernel iterates linearly through tile indices from `start_id` to `start_id + num_pages - 1`. Each tile read is a full page-sized NoC read from the interleaved buffer, with a barrier after each read (`noc_async_read_barrier`). This is a simple sequential access pattern with no striding or reuse.

### Write Pattern
**Sequential**: The writer kernel iterates through the same linear tile index range. Each tile is written via `noc_async_write_page` followed by a flush (`noc_async_writes_flushed`), with a final barrier at the end. Same sequential pattern as the reader.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Device compute grid (`compute_with_storage_grid_size`) |
| **Work Splitting** | `split_work_to_cores(grid_size, num_pages)` |
| **Core Enumeration** | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| **Group 1 Cores** | `core_group_1`: each processes `num_pages_per_core_group_1` tiles |
| **Group 2 Cores** | `core_group_2`: each processes `num_pages_per_core_group_2` tiles (remainder handling) |
| **Load Balancing** | Two-group split: group 1 gets `ceil(num_pages/num_cores)` tiles, group 2 gets `floor(num_pages/num_cores)` tiles |

The `split_work_to_cores` utility divides `num_pages` across the available cores. If `num_pages` does not divide evenly, group 1 cores handle one extra tile each compared to group 2 cores. If it divides evenly, group 2 is empty.

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for the source buffer (bank mapping, page size, etc.) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (`CBIndex::c_2 = 2`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for the destination buffer |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks this core processes (equals num_pages_per_core for this core group) |
| 1 | per_core_block_size | uint32_t | Tiles per block, always 1 for unary ops |

**Compute Kernel Defines (compile-time):**

| Define | Value for TANH | Description |
|---|---|---|
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro-expanded init + compute chain |
| `SFPU_OP_CHAIN_0_INIT_0` | `tanh_tile_init<0u>();` | SFPU init call (0u = non-approximate mode by default) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `tanh_tile<0u>(0);` | SFPU compute call on tile in DEST[0] |
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | Includes compute_kernel_api.h where tanh_tile is defined |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Input data type indicator |

**Compute Config:**

| Property | Value | Description |
|---|---|---|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest math fidelity |
| `math_approx_mode` | `false` | `get_op_approx_mode(TANH)` returns false (default switch case) |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | Controls FP32 accumulation in DEST registers |

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Source buffer base address in device memory |
| 1 | num_pages | uint32_t | Number of tiles this core reads |
| 2 | start_id | uint32_t | First tile index for this core |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Destination buffer base address in device memory |
| 1 | num_pages | uint32_t | Number of tiles this core writes |
| 2 | start_id | uint32_t | First tile index for this core |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Always 0 for TANH (no scalar parameter) |
| 1 | packed_scalar2 | uint32_t | Always 0 for TANH (no scalar parameter) |

## Kernel Implementations

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequentially reads tiles from interleaved DRAM/L1 into CB `c_0`. Uses `TensorAccessor` for address translation. Supports `BACKWARDS` mode via compile-time define (not used for standard TANH). Each tile read uses `cb_reserve_back` / `noc_async_read_page` / `noc_async_read_barrier` / `cb_push_back` pattern.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequentially writes tiles from CB `c_2` to interleaved DRAM/L1. Supports `OUT_SHARDED` mode where it simply waits for all tiles. Standard path uses `cb_wait_front` / `noc_async_write_page` / `noc_async_writes_flushed` / `cb_pop_front` pattern with final `noc_async_write_barrier`.

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
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for unary TANH)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initialize SFPU pipeline: sets up unpack from c_0, pack to c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST register file

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // wait until reader has pushed at least 1 tile into input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from c_0 into DEST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: tanh_tile_init<0u>(); tanh_tile<0u>(0);
            // tanh_tile_init<0u>() initializes SFPU for tanh (loads polynomial coefficients or LUT values)
            // tanh_tile<0u>(0) computes tanh on tile in DEST[0], processing all 4 faces (RC vector mode)
#endif

            tile_regs_commit();  // signal that DEST registers are ready for packing

            tile_regs_wait();  // wait for pack engine to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] result into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish the packed tile(s) to the writer
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`
(Identical implementation exists at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"  // PolynomialEvaluator for Horner-scheme polynomial evaluation
#include "ckernel_sfpu_sigmoid.h"       // _sfpu_sigmoid_ used in fp32 accurate mode

namespace ckernel::sfpu {

/*
 * Accurate tanh for fp32 using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
 * For small |x| < 0.6, uses minimax polynomial for better accuracy
 *
 * Algorithm:
 * - For |x| < 0.6: Use minimax polynomial (Sollya-optimized)
 * - For |x| >= 0.6: Use 2*sigmoid(2x) - 1
 *
 * Target accuracy: < 5 ULP for float32 (0.5 ULP for bfloat16)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;  // initialize result to 0.0

    constexpr float POLYNOMIAL_THRESHOLD = 0.6f;

    sfpi::vFloat abs_val = sfpi::abs(val);  // compute |x| for range check

    v_if(abs_val < POLYNOMIAL_THRESHOLD) {  // SFPU conditional: per-lane predication for |x| < 0.6
        // Small |x|: Use minimax polynomial for better accuracy
        // tanh(x)/x approximated as polynomial in x^2, then multiplied by x
        // Coefficients from Sollya: fpminimax(tanh(x)/x, [|0,2,4,6,8|], ...)
        sfpi::vFloat x2 = val * val;  // x^2

        // Evaluate degree-8 even polynomial in x^2 via Horner's method
        sfpi::vFloat p = PolynomialEvaluator::eval(
            x2,
            0.999999940395355224609375f,        // c0: ~1.0
            -0.33332359790802001953125f,         // c2: ~-1/3
            0.13310669362545013427734375f,       // c4: ~2/15
            -5.21197654306888580322265625e-2f,   // c6
            1.5497927553951740264892578125e-2f); // c8

        result = val * p;  // tanh(x) = x * P(x^2)
    }
    v_else {
        // Normal region: Use tanh(x) = 2*sigmoid(2x) - 1
        sfpi::vFloat two_x = 2.f * val;  // compute 2x
        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(two_x);  // sigmoid(2x) = 1/(1+exp(-2x))

        // Compute 2*sigmoid(2x) - 1
        result = 2.f * sig - sfpi::vConst1;  // vConst1 = 1.0
    }
    v_endif;

    return result;
}

// Continued fraction approximation (Lambert's formula) -- NOT used in default TANH path
// Included for completeness; may be used by other operations or future modes
template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_continued_fraction_(sfpi::vFloat val) {
    // tanh(x) via Lambert's continued fraction
    sfpi::vFloat x = sfpi::abs(val);  // work with positive values

    // Compute numerator and denominator using Horner's method
    sfpi::vFloat x2 = x * x;
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));

    sfpi::vFloat denominator = PolynomialEvaluator::eval(x2, 135135.f, 62370.f, 3150.f, 28.f);

    // reciprocal with 2 Newton-Raphson iterations for accuracy
    sfpi::vFloat result = numerator * ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);

    // Clamp to [-1, 1] since continued fraction can overshoot
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);  // result = min(result, 1.0); threshold_value = max(result, 1.0)

    result = sfpi::setsgn(result, val);  // restore original sign: tanh(-x) = -tanh(x)

    return result;
}

// Polynomial approximation for bfloat16 / non-fp32 mode
template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    sfpi::vFloat val = sfpi::abs(x);  // work with positive values: tanh(-x) = -tanh(x)

    // Degree-6 polynomial approximation from Sollya
    // tanh(x) ~ x * (c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6)))))
    // vConstFloatPrgm0/1/2 are loaded during tanh_init()
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,              // constant term = 0 (absorbed into structure)
        0.999004364013671875,       // c1
        3.0897438526153564453125e-2,  // c2
        -0.4890659749507904052734375, // c3
        sfpi::vConstFloatPrgm2,     // c4 = 0.281917631626129150390625 (loaded in init)
        sfpi::vConstFloatPrgm1,     // c5 = -6.6649019718170166015625e-2 (loaded in init)
        sfpi::vConstFloatPrgm0);    // c6 = 5.876733921468257904052734375e-3 (loaded in init)

    // Clamp to [-1, 1] since polynomial can exceed bounds
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);  // clamp result to max 1.0

    result = sfpi::setsgn(result, x);  // restore sign

    return result;
}

// Main entry point: dispatched by _llk_math_eltwise_unary_sfpu_params_ for each tile face
// ITERATIONS = 8 means process 8 rows per face (each row = 4 floats in SIMD, so 32 elements per face)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // LUT-based fast approximation mode
        // Load LUT configuration from L-registers (pre-loaded by tanh_init)
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];  // LUT param 0: slope for region 0
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];  // LUT param 1: slope+offset for region 1
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];  // LUT param 2: saturation value

#pragma GCC unroll 8  // fully unroll the 8-iteration loop for performance
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];    // read element from DEST register (current row)
            val = sfpi::lut(val, l0, l1, l2);       // apply 3-segment piecewise-linear LUT approximation
            sfpi::dst_reg[0] = val;                  // write result back to DEST

            sfpi::dst_reg++;  // advance to next row within the face
        }

        // Restore L-registers (compiler may have spilled them)
        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;
    } else {  // Non-approximate (accurate) mode

        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // read element from DEST register

            sfpi::vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                // FP32 mode: use sigmoid-based accurate formula with minimax polynomial for small |x|
                result = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(val);
            } else {
                // BF16 mode: use degree-6 polynomial approximation
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
                // Convert result back to fp16b format for storage in DEST
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;  // write result back to DEST
            sfpi::dst_reg++;            // advance to next row
        }
    }
}

// Initialization function: configures SFPU state before compute loop
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        // Load 3-segment piecewise-linear LUT parameters into L-registers
        uint imm0 = 0x1DFF;  // encodes: slope=0.90625, applied for |x| in region 0
        uint imm1 = 0x481A;  // encodes: slope=0.09375, offset=0.8125, for |x| in region 1
        uint imm2 = 0xFF00;  // encodes: output=1.0 (saturation) for |x| in region 2
        _sfpu_load_imm16_(0, imm0);  // load into LReg0
        _sfpu_load_imm16_(1, imm1);  // load into LReg1
        _sfpu_load_imm16_(2, imm2);  // load into LReg2
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            // FP32 accurate mode needs sigmoid init (which inits reciprocal)
            sigmoid_init<false>();  // false = non-approximate sigmoid (uses Newton-Raphson reciprocal)
        } else {
            // BF16 polynomial mode: store polynomial tail coefficients in programmable constant registers
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;   // c6 coefficient
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;       // c5 coefficient
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;          // c4 coefficient
        }
    }
}

}  // namespace ckernel::sfpu
```

#### LLK Dispatch Layer

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    // Initialize SFPU for tanh: calls tanh_init which loads LUT or polynomial coefficients
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>(sfpu::tanh_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // Dispatch calculate_tanh across all 4 faces of the tile (RC mode)
    // _llk_math_eltwise_unary_sfpu_params_ iterates over faces, calling calculate_tanh<APPROX, FP32, 8> per face
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_tanh<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[0]` | Read/write element from/to DEST register file (current face row) |
| `sfpi::dst_reg++` | Advance DEST register pointer to next row within a face |
| `sfpi::lut(val, l0, l1, l2)` | 3-segment piecewise-linear LUT lookup (approximation mode only) |
| `sfpi::abs(val)` | Compute absolute value of SFPU vector register |
| `sfpi::setsgn(result, val)` | Copy sign bits from `val` to `result` |
| `sfpi::vConst0` / `sfpi::vConst1` | Hardware constant registers: 0.0 and 1.0 |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers (loaded during init) |
| `sfpi::vec_min_max(a, b)` | After call: `a = min(a,b)`, `b = max(a,b)` -- used for clamping |
| `sfpi::float_to_fp16b(val, 0)` | Convert fp32 to bfloat16 format in-register |
| `sfpi::reinterpret<vFloat>(val)` | Reinterpret cast between SFPU register types |
| `v_if` / `v_else` / `v_endif` | Per-lane predicated execution (SFPU condition codes) |
| `_sfpu_load_imm16_(reg, imm)` | Load 16-bit immediate value into SFPU L-register |
| `PolynomialEvaluator::eval(...)` | Horner-scheme polynomial evaluation using SFPU multiply-add |
| `_sfpu_sigmoid_(val)` | Compute sigmoid: `1/(1+exp(-x))` using exp + reciprocal |
| `_sfpu_reciprocal_<N>(val)` | Newton-Raphson reciprocal with N iterations |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` | DEST register: holds current tile face row data (input on read, output on write) |
| `LReg0` (`l_reg[LRegs::LReg0]`) | LUT parameter 0 (approximation mode): encodes slope for region 0 (0x1DFF) |
| `LReg1` (`l_reg[LRegs::LReg1]`) | LUT parameter 1 (approximation mode): encodes slope+offset for region 1 (0x481A) |
| `LReg2` (`l_reg[LRegs::LReg2]`) | LUT parameter 2 (approximation mode): saturation value (0xFF00 = 1.0) |
| `vConstFloatPrgm0` | Programmable constant: polynomial coefficient c6 (non-approx BF16 mode) |
| `vConstFloatPrgm1` | Programmable constant: polynomial coefficient c5 (non-approx BF16 mode) |
| `vConstFloatPrgm2` | Programmable constant: polynomial coefficient c4 (non-approx BF16 mode) |
| `vConst0` | Hardware constant 0.0 |
| `vConst1` | Hardware constant 1.0 |

#### SFPU Execution Flow

1. **Initialization** (`tanh_init`): Called once before the tile processing loop.
   - **Approximation mode**: Loads three 16-bit immediate values into L-registers (LReg0, LReg1, LReg2) that parameterize the 3-segment piecewise-linear LUT.
   - **Non-approx FP32 mode**: Calls `sigmoid_init<false>()` which initializes the reciprocal function needed by sigmoid.
   - **Non-approx BF16 mode**: Loads three polynomial coefficients into programmable constant registers.

2. **Tile acquisition**: The compute kernel calls `copy_tile(c_0, 0, 0)` which unpacks the tile from CB `c_0` into DEST register 0. This fills all 4 faces (Face0-Face3) of the tile in DEST.

3. **Face iteration**: `_llk_math_eltwise_unary_sfpu_params_` iterates over all 4 faces of the tile (VectorMode::RC). For each face, it calls `calculate_tanh<APPROX, FP32, 8>()`.

4. **Per-face computation** (`calculate_tanh`): Processes 8 rows per face (ITERATIONS=8, each row is 4 elements wide = SFPU SIMD width). For each row:
   - Reads the current row from `dst_reg[0]`
   - **Approx mode**: Applies `sfpi::lut()` -- a single-instruction 3-segment piecewise-linear approximation
   - **FP32 accurate mode**: Branches based on `|x| < 0.6`:
     - Small x: evaluates degree-8 minimax polynomial `tanh(x) = x * P(x^2)`
     - Large x: computes `2*sigmoid(2x) - 1` using the sigmoid helper
   - **BF16 polynomial mode**: Evaluates degree-6 polynomial, clamps to [-1,1], converts back to fp16b
   - Writes result back to `dst_reg[0]`, then advances `dst_reg++`

5. **Packing**: After all 4 faces are processed, control returns to the compute kernel which calls `pack_tile(0, c_2)` to pack the result from DEST into the output circular buffer.

#### SFPU Configuration

| Configuration | Value | Description |
|---|---|---|
| `APPROXIMATION_MODE` | Template parameter (default: `false` / `0u`) | When true, uses LUT; when false, uses polynomial/sigmoid |
| `is_fp32_dest_acc_en` | Template parameter from `DST_ACCUM_MODE` | When true, uses fp32 accurate sigmoid path; when false, uses bf16 polynomial |
| `ITERATIONS` | 8 | Rows per tile face (32 elements / 4 SIMD lanes = 8 iterations) |
| `math_fidelity` | `HiFi4` | Set in ComputeConfig; highest precision mode |
| `math_approx_mode` | `false` | `get_op_approx_mode(TANH)` always returns false |
| `SfpuType::tanh` | Enum value | Used by `llk_math_eltwise_unary_sfpu_init` to configure SFPU pipeline |

#### Hardware Compatibility Notes

The SFPU kernel implementation is **identical** between Blackhole and Wormhole B0 architectures for TANH. Both architectures:
- Share the same `ckernel_sfpu_tanh.h` source code
- Support the same three modes: LUT approximation, FP32 accurate (sigmoid-based), and BF16 polynomial
- Use the same LUT encoding format for approximation mode (0x1DFF, 0x481A, 0xFF00)
- Use the same polynomial coefficients

The `_llk_math_eltwise_unary_sfpu_params_` dispatch function may have minor differences in how it advances face addresses between architectures (using `TTI_SETRWC` instructions), but the mathematical computation is identical.

## Implementation Notes

1. **Default mode is non-approximate**: When TANH is constructed via `string_to_unary_with_param("tanh")`, `param0 = false` (i.e., `APPROXIMATION_MODE = 0u`). The `get_op_approx_mode` function also returns `false` for all ops by default. Therefore, the default path is the polynomial/sigmoid-based accurate computation, not the LUT approximation.

2. **Three computational paths**: The SFPU kernel provides three distinct algorithms:
   - **LUT approximation** (`APPROXIMATION_MODE=true`): Fastest, lowest accuracy. Uses a 3-segment piecewise-linear function via the hardware `lut` instruction.
   - **BF16 polynomial** (`APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=false`): Degree-6 Sollya-optimized polynomial with clamping and sign restoration. Result converted to fp16b.
   - **FP32 accurate** (`APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=true`): Hybrid approach using minimax polynomial for |x| < 0.6 and `2*sigmoid(2x)-1` for larger values. Target accuracy < 5 ULP for float32.

3. **Continued fraction function**: `_sfpu_tanh_continued_fraction_` is defined in the header but is NOT used by the default `calculate_tanh` dispatch. It may be available for alternative implementations or future use.

4. **No scalar parameters**: Unlike operations such as HARDSHRINK or MISH, TANH does not use `packed_scalar1` or `packed_scalar2` runtime arguments. They are always 0.

5. **CB c_1 not created**: The temporary circular buffer `c_1` is only created for HARDSHRINK and LOGIT operations. TANH does not need intermediate storage.

6. **Double-buffering**: Both input and output CBs have 2 pages, allowing the reader/compute/writer pipeline stages to overlap for improved throughput.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations like tanh? What kernels does it use (reader, compute, writer)? How are circular buffers configured and how is work distributed across cores?"
   **Reason**: Needed architectural overview of the UnaryProgramFactory to understand the overall structure before reading source code.
   **Key Findings**: Confirmed three-kernel structure (reader/compute/writer), double-buffered CBs, split_work_to_cores distribution, and the dynamic kernel path selection based on op type.

2. **Query**: "What is the tanh SFPU kernel implementation? Where is the compute kernel for unary operations like tanh located? How does the eltwise_unary compute kernel dispatch to specific SFPU functions?"
   **Reason**: Needed to locate the SFPU kernel files and understand the dispatch chain from compute kernel to SFPU function.
   **Key Findings**: Located `ckernel_sfpu_tanh.h` in both blackhole and wormhole_b0 directories. Confirmed the three-mode implementation (LUT, polynomial, sigmoid-based). Identified `llk_math_eltwise_unary_sfpu_tanh.h` as the LLK dispatch layer.

3. **Query**: "What does _llk_math_eltwise_unary_sfpu_params_ do? How does it iterate over tile faces and call the SFPU compute function?"
   **Reason**: Needed to understand how the LLK layer dispatches the SFPU function across tile faces.
   **Key Findings**: Confirmed that `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces in RC mode, calling the SFPU function once per face with 8 iterations (rows) each. It handles SFPU start/done synchronization.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how TANH operation type maps to compute kernel path, macro defines, and init/func strings.
   **Key Information**: TANH uses default `eltwise_sfpu.cpp` kernel, gets `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` macro, generates `tanh_tile_init<Xu>()` / `tanh_tile<Xu>(dst)` where X is the param0 (approx mode flag).

2. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Needed to verify the compute API layer that bridges from `tanh_tile<>()` to `llk_math_eltwise_unary_sfpu_tanh<>()`.
   **Key Information**: `tanh_tile<fast_and_approx>(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)))`.
