# GELU Operation Analysis

## Overview

**Operation**: GELU (Gaussian Error Linear Unit)
**Category**: Eltwise Unary
**SFPU Operation**: Yes
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

GELU is a smooth approximation of the rectifier function used as an activation in neural networks. Mathematically:
```
GELU(x) = x * Phi(x)
```
where Phi(x) is the CDF of the standard normal distribution. The Tenstorrent implementation provides two modes: an approximate mode using a piecewise linear LUT (look-up table), and an accurate mode using a polynomial CDF approximation. Additionally, the metal-level kernel in `ckernel_sfpu_gelu.h` contains a third path using a 15th-degree Chebyshev polynomial for a narrow input range (|x| < 3).

---

## Program Factory Architecture

### Factory Selection

The GELU operation uses the standard `UnaryProgramFactory` (or `UnarySubCoreGridProgramFactory` when sub-core grids are specified). Factory selection is managed by `UnaryDeviceOperation`, which uses a `std::variant` to choose between:
1. `UnaryProgramFactory` -- default, uses full compute grid
2. `UnarySubCoreGridProgramFactory` -- when `sub_core_grids` is specified
3. `UnaryShardedProgramFactory` -- for sharded tensors (not used for GELU in the interleaved case)

### Program Structure

The program factory creates three kernels per Tensix core:
- **Reader kernel**: Reads input tiles from DRAM via NoC
- **Compute kernel**: Executes the GELU SFPU operation on each tile
- **Writer kernel**: Writes output tiles back to DRAM via NoC

### Work Distribution

Work is distributed across all available compute cores using `split_work_to_cores()`. The total number of tiles (pages) is divided into two core groups:
- **Core group 1**: Gets `num_pages_per_core_group_1` tiles each
- **Core group 2**: Gets `num_pages_per_core_group_2` tiles each (remainder distribution)

Each core processes tiles sequentially, one at a time (block size = 1).

---

## Circular Buffer Configuration

| CB Index | Identifier | Purpose | Page Count | Data Format |
|----------|-----------|---------|------------|-------------|
| `c_0` | `src0_cb_index` | Input buffer | 2 | Input tensor data format |
| `c_2` | `output_cb_index` | Output buffer | 2 | Output tensor data format |

GELU does **not** require the temporary buffer `c_1` (used only by HARDSHRINK, CBRT, and LOGIT).

The double-buffering (2 pages) allows the reader to fill one page while the compute kernel processes the other, enabling pipeline overlap.

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
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of the input tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor configuration

    constexpr uint32_t cb_id_in0 = 0;                      // CB index c_0 for input

    // Page size is read from the CB interface -- works for both tile and row-major layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                        // Process one page (tile) at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

// Read tiles one by one from DRAM into the input circular buffer
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);               // Wait for space in the CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);          // Initiate async NoC read from DRAM
        noc_async_read_barrier();                           // Wait for the read to complete
        cb_push_back(cb_id_in0, onepage);                  // Signal the compute kernel that a tile is ready
    }
}
```

**Runtime Arguments**: `{src_buffer_address, num_pages_per_core, start_tile_id}`
**Compile-Time Arguments**: `TensorAccessorArgs` (encoding buffer type, page size, bank mapping)

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM address of the output tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile index for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // Compile-time tensor accessor for output

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);                   // For sharded output, wait for all pages at once
#else

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);                 // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);   // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);          // Initiate async NoC write to DRAM
        noc_async_writes_flushed();                        // Ensure write is in flight
        cb_pop_front(cb_id_out, onepage);                  // Free the CB slot for compute to reuse
    }
    noc_async_write_barrier();                             // Final barrier: all writes complete
#endif
}
```

**Runtime Arguments**: `{dst_buffer_address, num_pages_per_core, start_tile_id}`
**Compile-Time Arguments**: `{output_cb_index, TensorAccessorArgs}`

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

GELU uses the generic `eltwise_sfpu.cpp` compute kernel (the `default` case in `get_compute_kernel_path()`). The GELU-specific behavior is injected via preprocessor defines.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes gelu.h when SFPU_OP_GELU_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tiles assigned to this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for GELU)

    // Initialize the SFPU for unary operation -- sets up unpack/pack for input CB c_0 and output CB c_2
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in the output CB for the entire block (1 tile)
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to the destination register file (DST)
            tile_regs_acquire();

            // Wait for one tile to be available in the input CB (produced by reader)
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from input CB to DST register at index 0 -- this invokes the unpacker
            copy_tile(tt::CBIndex::c_0, 0, 0);

// The SFPU_OP_CHAIN_0 macro expands to the GELU init + compute calls:
//   For GELU without param: "gelu_tile_init(); gelu_tile(0);"
//   For GELU with param:    "gelu_tile_init<Pu>(); gelu_tile<Pu>(0);"
// where P is the approximation mode parameter (0=accurate, 1=approximate)
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are ready for packing
            tile_regs_commit();

            // Wait for pack to be ready to read from DST
            tile_regs_wait();

            // Pack tile from DST register 0 into the output CB
            pack_tile(0, tt::CBIndex::c_2);

            // Release the input tile slot so the reader can reuse it
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers for the next iteration
            tile_regs_release();
        }
        // Push the completed block to the writer kernel
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

**Compile-Time Arguments**: `{per_core_block_cnt, per_core_block_dim (=1)}`
**Runtime Arguments**: `{packed_scalar1 (=0), packed_scalar2 (=0)}` -- unused for GELU

#### Compute Kernel Defines for GELU

The program factory generates these preprocessor defines via `get_block_defines()` and `update_macro_defines()`:

| Define | Value | Purpose |
|--------|-------|---------|
| `SFPU_OP_GELU_INCLUDE` | `1` | Gates the `#include "api/compute/eltwise_unary/gelu.h"` in `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro chain that expands to init + compute |
| `SFPU_OP_CHAIN_0_INIT_0` | `gelu_tile_init();` (default) or `gelu_tile_init<Pu>();` (parameterized) | Initialization call |
| `SFPU_OP_CHAIN_0_FUNC_0` | `gelu_tile(0);` (default) or `gelu_tile<Pu>(0);` (parameterized) | Per-tile computation call |

When the GELU operation is created with a parameter (e.g., `UnaryWithParam(UnaryOpType::GELU, 1.0f)` for approximate mode), the parameterized path is used. When created without a parameter, the default path is used (which defaults to `fast_and_approx = true`).

#### Math Approximation Mode

The `get_op_approx_mode()` function returns `false` for GELU (default case). This means `math_approx_mode` in the `ComputeConfig` is `false` unless all operations in the chain request approximate mode. However, the GELU-specific approximation is controlled by the template parameter `fast_and_approx`, not by the global `math_approx_mode` flag.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel Files

The GELU SFPU implementation spans multiple layers:

1. **API Layer**: `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h` -- public API (`gelu_tile_init`, `gelu_tile`)
2. **Metal ckernel Layer**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` -- metal-level dispatch with Chebyshev fallback
3. **LLK Layer**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_gelu.h` -- core SFPU implementation with LUT-based and CDF-based paths
4. **CDF Helper**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_cdf.h` -- CDF polynomial approximation
5. **Polynomial Helper**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_polyval.h` -- POLYVAL5 Horner's method

#### Annotated SFPU Kernel Source

**API Layer** (`gelu.h`):

```cpp
#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_gelu.h"           // Metal-level GELU implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"  // SFPU dispatch macros
#endif

namespace ckernel {

// Initialization: calls _init_gelu_ from the LLK layer to load LUT coefficients into SFPU local registers
template <bool fast_and_approx = true>
ALWI void gelu_tile_init() {
    // SFPU_INIT_KERNEL_CALL expands to:
    //   llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, fast_and_approx>(sfpu::gelu_init<fast_and_approx>)
    // This configures the SFPU math pipeline and calls gelu_init to load LUT constants
    MATH(SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, fast_and_approx));
}

// Per-tile computation: applies GELU to all elements in DST register at tile_index
template <bool fast_and_approx = true>
ALWI void gelu_tile(uint32_t idst) {
    // SFPU_UNARY_NO_PARAM_KERNEL_FN expands to:
    //   _llk_math_eltwise_unary_sfpu_params_<fast_and_approx>(
    //       ckernel::sfpu::calculate_gelu<fast_and_approx>, idst, (int)VectorMode::RC)
    // This iterates over all 4 faces of the tile, calling calculate_gelu on each face
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, fast_and_approx, idst));
}

}  // namespace ckernel
```

**Metal ckernel Layer** (`ckernel_sfpu_gelu.h` in `tt_metal/hw/ckernels/{arch}/`):

```cpp
#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

namespace ckernel {
namespace sfpu {

// POLYVAL15: 15th-degree polynomial evaluation using Horner's method
// This evaluates p(x) = c15*x^15 + c14*x^14 + ... + c1*x + c0
// Used by the Chebyshev approximation path for |x| < 3
#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)  \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + \
    (c10)) * (x) + (c9)) * (x) + (c8)) * (x) + (c7)) * (x) + (c6)) * (x) + (c5)) * (x) +   \
    (c4)) * (x) + (c3)) * (x) + (c2)) * (x) + (c1)) * (x) + (c0)

// Chebyshev polynomial approximation of GELU for inputs in [-5.5, +inf)
// For inputs < -5.5, the result is 0.0 (GELU approaches 0 for very negative inputs)
// The polynomial coefficients approximate x * Phi(x) directly
inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {                      // Conditional: only compute for val >= -5.5
        result = POLYVAL15(
            -1.81205228163e-09,               // c15
            -4.59055119276e-08,               // c14
            -3.74540617693e-07,               // c13
            -2.29754133825e-07,               // c12
            1.19076782913e-05,                // c11
            4.25116466215e-05,                // c10
            -0.000138391838381,               // c9
            -0.000862052441087,               // c8
            0.000768340223025,                // c7
            0.0092074331601,                  // c6
            -0.00208478037614,                // c5
            -0.0656369476513,                 // c4
            0.00244542739174,                 // c3
            0.398579460781,                   // c2  (close to 0.5 * 2/sqrt(pi))
            0.499174645395,                   // c1  (close to 0.5 for GELU midpoint)
            2.98325768482e-05,                // c0  (close to 0 for GELU at origin)
            val);

        // Correct the sign: GELU is an odd-like function around the origin
        // setsgn copies the sign bit of val onto result
        result = setsgn(result, val);
    }
    v_endif;
    return result;
}

// Initialization wrapper: delegates to LLK _init_gelu_ to load LUT coefficients
template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

// Main calculate_gelu function: dispatches between three paths
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        // APPROXIMATE path: uses LUT-based piecewise linear approximation (fast)
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
        // ACCURATE path in metal layer: uses Chebyshev polynomial for |x| < 3,
        // identity (pass-through) for |x| >= 3 (where GELU(x) ~ x)
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];           // Load element from DST register
            sfpi::vFloat result = in;                      // Default: pass through (for |x| >= 3)
            v_if(in == 0.0f) { result = 0.0f; }           // Special case: GELU(0) = 0
            v_elseif(in < 3.0f) {                          // For |x| < 3: use Chebyshev polynomial
                result = calculate_gelu_chebyshev(in);
            }
            v_endif;
            sfpi::dst_reg[0] = result;                     // Store result back to DST
            sfpi::dst_reg++;                               // Advance to next row in the face
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**LLK Layer** (`ckernel_sfpu_gelu.h` in `tt_metal/third_party/tt_llk/`):

```cpp
#pragma once
#include <cstdint>
#include "ckernel_sfpu_cdf.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

// Core GELU transformation shared by gelu_derivative (accurate mode)
// In approximate mode: identity (pass-through to LUT)
// In accurate mode: computes sqrt(2/pi) * (x + 0.044715 * x^3)
//   This is the argument to tanh() in the GELU approximation formula:
//   GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_gelu_core_(sfpi::vFloat in)
{
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        result = in;    // In approx mode, the LUT handles everything
    }
    else
    {
        // f = 0.044715 * x^3 + x, then scaled by sqrt(2/pi) ~ 0.79788
        result = (in * in) * (in * sfpi::s2vFloat16b(0.044715f)) + in;
        result *= sfpi::s2vFloat16b(0.79788f);
    }
    return result;
}

// APPROXIMATE GELU: uses a 6-entry piecewise linear LUT loaded during init
// For each element: result = 0.5 * x + lut2_sign(x)
// where lut2_sign approximates 0.5 * x * erf(x/sqrt(2)) using the preloaded coefficients
template <int ITERATIONS>
inline void _calculate_gelu_appx_()
{
    // Cache LUT coefficients from local registers into vUInt variables
    // This avoids repeated SFPU register reads in the inner loop
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];   // Slopes for ranges [0,0.5) and [0.5,1.0)
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];   // Slopes for ranges [1.0,1.5) and [1.5,2.0)
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];   // Slopes for ranges [2.0,3.0) and [3.0,+inf)
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];   // Intercepts for ranges [0,0.5) and [0.5,1.0)
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];   // Intercepts for ranges [1.0,1.5) and [1.5,2.0)
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];   // Intercepts for ranges [2.0,3.0) and [3.0,+inf)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in      = sfpi::dst_reg[0];          // Load element from DST
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;     // 0.5, loaded during init
        sfpi::vFloat half_in = in * half;                  // half_in = 0.5 * x
        // lut2_sign: evaluates a piecewise linear function on |x|, preserving sign
        // Each segment: result_segment = slope_i * |x| + intercept_i
        // The function approximates 0.5 * erf(x/sqrt(2)) (the signed CDF offset)
        sfpi::vFloat result  = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result               = half_in + result;           // GELU(x) = 0.5*x + lut_result

        sfpi::dst_reg[0] = result;                         // Store result
        sfpi::dst_reg++;                                   // Move to next row

        // Historical comment showing the underlying SFPU instruction sequence:
        // TTI_SFPLOAD(3, 0, 1, 0);       -- load from DEST into LReg3
        // TTI_SFPLUTFP32(7, 2);           -- LReg7 = LUT(LReg3)
        // TTI_SFPMAD(3, 12, 7, 3, 0);    -- LReg3 = 0.5*LReg3 + LReg7
        // TTI_SFPSTORE(3, 0, 3, 0);      -- store and increment write counter
    }

    // Restore LUT coefficients to local registers (required for correct state management)
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

// ACCURATE GELU: uses polynomial CDF approximation
// GELU(x) = x * CDF(x) where CDF is approximated by a 5th-degree polynomial
template <int ITERATIONS>
inline void _calculate_gelu_accurate_()
{
    constexpr bool scaled = true;     // "scaled" means CDF result is multiplied by x
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];
        // _calculate_cdf_appx_(in, scaled=true) computes:
        //   CDF(x) * x  where CDF is a piecewise polynomial approximation
        sfpi::vFloat result = _calculate_cdf_appx_(in, scaled);
        sfpi::dst_reg[0]    = result;
        sfpi::dst_reg++;
    }
}

// Dispatcher: selects approximate or accurate path based on template parameter
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_gelu_()
{
    if constexpr (APPROXIMATION_MODE)
    {
        _calculate_gelu_appx_<ITERATIONS>();
    }
    else
    {
        _calculate_gelu_accurate_<ITERATIONS>();
    }
}

// Initialization: loads piecewise linear LUT coefficients into SFPU local registers
// These coefficients define 6 segments of the GELU approximation for |x| in [0, +inf):
//   [0.0, 0.5):   slope=0.1928, intercept=-0.0150
//   [0.5, 1.0):   slope=0.4939, intercept=-0.1605
//   [1.0, 1.5):   slope=0.6189, intercept=-0.2797
//   [1.5, 2.0):   slope=0.6099, intercept=-0.2635
//   [2.0, 3.0):   slope=0.5402, intercept=-0.1194
//   [3.0, +inf):  slope=0.50,   intercept=0.0 (saturates to 0.5*x)
template <bool APPROXIMATION_MODE>
inline void _init_gelu_()
{
    sfpi::vConstFloatPrgm0 = 0.5f;     // Store 0.5 in programmable constant register

    // LReg0: slopes for segments [0.5,1.0) in hi-16 and [0,0.5) in lo-16
    _sfpu_load_imm32_(0, 0x37E7322B);  // hi=0.4939(FP16), lo=0.1928(FP16)
    // LReg4: intercepts for segments [0.5,1.0) in hi-16 and [0,0.5) in lo-16
    _sfpu_load_imm32_(4, 0xB12286D8);  // hi=-0.1605(FP16), lo=-0.0150(FP16)

    // LReg1: slopes for segments [1.5,2.0) in hi-16 and [1.0,1.5) in lo-16
    _sfpu_load_imm32_(1, 0x38E138F3);  // hi=0.6099(FP16), lo=0.6189(FP16)
    // LReg5: intercepts for segments [1.5,2.0) in hi-16 and [1.0,1.5) in lo-16
    _sfpu_load_imm32_(5, 0xB437B479);  // hi=-0.2635(FP16), lo=-0.2797(FP16)

    // LReg2: slopes for segments [3.0,+inf) in hi-16 and [2.0,3.0) in lo-16
    _sfpu_load_imm32_(2, 0x38003852);  // hi=0.50(FP16), lo=0.5402(FP16)
    // LReg6: intercepts for segments [3.0,+inf) in hi-16 and [2.0,3.0) in lo-16
    _sfpu_load_imm32_(6, 0x7c00afa4);  // hi=+inf(used as 0 intercept), lo=-0.1194(FP16)
}

} // namespace ckernel::sfpu
```

**CDF Approximation** (`ckernel_sfpu_cdf.h`):

```cpp
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Positive CDF: polynomial approximation of Phi(x) for x >= 0
// Uses two ranges:
//   [0, 2.5): 4th-degree polynomial
//   [2.5, 5): linear approximation
//   [5, +inf): saturates to 1.0
inline sfpi::vFloat _calculate_pos_cdf_appx_(sfpi::vFloat val)
{
    sfpi::vFloat result;
    v_if (val < 2.5f)
    {
        // 4th-degree polynomial fit for CDF in [0, 2.5)
        // Coefficients: [0.0122792, -0.05281024, -0.03048313, 0.41314081, 0.49866379]
        result = POLYVAL5<sfpi::vFloat>(
            0.0122792f, -0.05281024f, -0.03048313f, 0.41314081f, 0.49866379f, val);
    }
    v_else
    {
        // Linear approximation for [2.5, 5): slope * x + intercept
        result = 0.44656975f * val + 0.58216001f;
    }
    v_endif;

    // Clamp to [0, 1]
    v_if (result > 1.0f)
    {
        result = 1.0f;
    }
    v_endif;
    return result;
}

// Full CDF: handles negative inputs by symmetry CDF(-x) = 1 - CDF(x)
// When scaled=true (used by GELU): returns CDF(x) * x  (= GELU(x))
inline sfpi::vFloat _calculate_cdf_appx_(sfpi::vFloat val, bool scaled = false)
{
    sfpi::vFloat result = 0.0f;

    v_if (val < 0.0f)
    {
        result = 1.0f - _calculate_pos_cdf_appx_(-val);   // Symmetry: CDF(-x) = 1 - CDF(x)
    }
    v_else
    {
        result = _calculate_pos_cdf_appx_(val);
    }
    v_endif;

    if (scaled)
    {
        result *= val;    // GELU(x) = x * CDF(x)
    }
    return result;
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` / `sfpi::dst_reg[0]` (read) | Loads a vector element from the DEST register file into an SFPU local register (LReg3) |
| `SFPSTORE` / `sfpi::dst_reg[0]` (write) | Stores an SFPU local register value back to the DEST register file |
| `SFPLOADI` / `_sfpu_load_imm32_` | Loads a 32-bit immediate value into an SFPU local register (used during init to set LUT coefficients) |
| `SFPMUL` / `*` operator | Vectorized floating-point multiplication (e.g., `in * half`) |
| `SFPMAD` / `*` + `+` operators | Multiply-accumulate: computes `a * b + c` in a single operation |
| `SFPLUTFP32` / `lut2_sign()` | Evaluates a 6-entry piecewise linear function using coefficients in LRegs; the `_sign` variant preserves the input sign |
| `SFPSETCC` / `v_if`, `v_elseif` | Sets per-lane condition flags for conditional execution (e.g., `val < 3.0f`) |
| `SFPENCC` / `v_if` block entry | Enables conditional execution mode |
| `SFPCOMPC` / `v_else`, `v_elseif` | Complements condition flags for else branches |
| `SFPPOPC` / `v_endif` | Restores condition flags from the stack (ends conditional block) |
| `SFPPUSHC` / `v_if` nesting | Pushes condition flags onto the stack for nested conditionals |
| `setsgn()` | Copies the sign bit from one value to another (used in Chebyshev path) |
| `sfpi::s2vFloat16b()` | Converts a scalar float to a BF16 SFPU vector constant |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` | Primary working register -- input is loaded here, result is stored here |
| `dst_reg++` | Advances to the next row within a face (32 elements per face, 8 rows of 4) |
| `LReg0` | Piecewise linear slopes for ranges [0.5, 1.0) and [0.0, 0.5) packed as FP16 hi/lo |
| `LReg1` | Piecewise linear slopes for ranges [1.5, 2.0) and [1.0, 1.5) |
| `LReg2` | Piecewise linear slopes for ranges [3.0, +inf) and [2.0, 3.0) |
| `LReg4` | Piecewise linear intercepts for ranges [0.5, 1.0) and [0.0, 0.5) |
| `LReg5` | Piecewise linear intercepts for ranges [1.5, 2.0) and [1.0, 1.5) |
| `LReg6` | Piecewise linear intercepts for ranges [3.0, +inf) and [2.0, 3.0) |
| `vConstFloatPrgm0` | Programmable constant register, loaded with 0.5 during init |
| `LReg3` (implicit) | Used internally by `SFPLOAD`/`SFPSTORE` as the data transfer register |
| `LReg7` (implicit) | Used internally by `SFPLUTFP32` as the LUT output register |

#### SFPU Execution Flow

The full execution flow for processing one tile through the GELU SFPU operation:

1. **Initialization** (`gelu_tile_init`):
   - `llk_math_eltwise_unary_sfpu_init` configures the SFPU math pipeline for the `gelu` SfpuType
   - `_init_gelu_()` loads 0.5 into `vConstFloatPrgm0` and loads 6 piecewise linear coefficients (slopes + intercepts) into LReg0-6 via `_sfpu_load_imm32_` (which compiles to `SFPLOADI` instructions)

2. **Tile acquisition**: `tile_regs_acquire()` locks the DEST register file for exclusive math RISC-V access

3. **Unpack**: `cb_wait_front(c_0, 1)` blocks until the reader has produced a tile; `copy_tile(c_0, 0, 0)` invokes the unpacker to decompress the tile from the input CB into DEST register 0

4. **SFPU dispatch** (`gelu_tile`):
   - `_llk_math_eltwise_unary_sfpu_start_` positions the SFPU to operate on the correct DST tile index
   - For `VectorMode::RC` (full tile), the dispatch iterates over all 4 faces (each face = 16x16 = 256 elements)
   - For each face, `calculate_gelu<APPROX_MODE>()` is called with `ITERATIONS = 8` (8 rows of 4 elements = 32 elements per face half)

5. **Per-face SFPU computation** (approximate path):
   - For each of 8 iterations:
     a. `dst_reg[0]` loads the current 4-element vector from DEST
     b. `half_in = in * 0.5` computes half the input
     c. `lut2_sign(in, l0..l6)` evaluates the piecewise linear LUT on `|in|` and applies the sign of `in`
     d. `result = half_in + lut_result` combines to produce GELU(x)
     e. `dst_reg[0] = result` stores the result back
     f. `dst_reg++` advances to the next row

6. **Per-face SFPU computation** (accurate path -- LLK layer):
   - For each of 8 iterations:
     a. `dst_reg[0]` loads the current vector
     b. `_calculate_cdf_appx_(in, scaled=true)` computes:
        - If `in >= 0`: `POLYVAL5(coeffs, in) * in` (4th-degree polynomial scaled by x)
        - If `in < 0`: `(1 - POLYVAL5(coeffs, -in)) * in`
        - Clamps CDF to [0, 1] before scaling
     c. Stores result and advances

7. **Per-face SFPU computation** (Chebyshev path -- metal layer, accurate mode):
   - For each of 8 iterations:
     a. Loads the current vector
     b. If `in == 0`: result = 0
     c. If `in < 3`: evaluates 15th-degree Chebyshev polynomial, then `setsgn(result, in)`
     d. If `in >= 3`: result = in (GELU(x) ~ x for large positive x)
     e. Stores result and advances

8. **Face transition**: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` moves the SFPU to the next face

9. **Completion**: `_llk_math_eltwise_unary_sfpu_done_()` finalizes SFPU state

10. **Pack**: `tile_regs_commit()` signals the packer; `tile_regs_wait()` waits for readiness; `pack_tile(0, c_2)` packs the result from DEST into the output CB

11. **Release**: `cb_pop_front(c_0, 1)` frees the input CB slot; `tile_regs_release()` unlocks DEST; `cb_push_back(c_2, 1)` signals the writer

#### SFPU Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity -- full precision FP multiply |
| `math_approx_mode` | `false` | GELU returns `false` from `get_op_approx_mode()` |
| `fp32_dest_acc_en` | Depends on `args.fp32_dest_acc_en` | When true, DEST uses FP32 accumulation |
| `preserve_fp32_precision` | Depends on `args.preserve_fp32_precision` | When true, unpack goes directly to DEST in FP32 |
| `fast_and_approx` template param | Default `true`, overridable via `UnaryWithParam` | Controls which GELU algorithm is used |
| `ITERATIONS` | 8 (default) | 8 iterations x 4 elements = 32 elements per face half |
| `VectorMode` | `RC` (Row-Column) | All 4 faces of the 32x32 tile are processed |
| `SfpuType` | `gelu` | Used for SFPU init configuration |

#### Hardware Compatibility Notes

The GELU implementation is **identical** between Wormhole B0 and Blackhole architectures:
- Both `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h` and `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h` contain the same source code
- Both `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` and the Blackhole variant contain the same source code
- The LUT coefficients, polynomial coefficients, and control flow are all shared
- The `SFPLUTFP32` instruction used by `lut2_sign` is available on both architectures
- Any differences would arise from the underlying FMA precision model (`fma_model_bh` vs `fma_model_wh`), which could cause minor numerical differences in the polynomial evaluation results, but the algorithm and coefficients are identical

---

## Compile-Time vs Runtime Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Value for GELU |
|-------|------|---------------|
| 0 | `per_core_block_cnt` | Number of tiles per core |
| 1 | `per_core_block_dim` | 1 (one tile per block) |

### Runtime Arguments

| Kernel | Index | Name | Description |
|--------|-------|------|-------------|
| Reader | 0 | `src_addr` | Input buffer DRAM address |
| Reader | 1 | `num_pages` | Tiles to process |
| Reader | 2 | `start_id` | Starting tile index |
| Writer | 0 | `dst_addr` | Output buffer DRAM address |
| Writer | 1 | `num_pages` | Tiles to process |
| Writer | 2 | `start_id` | Starting tile index |
| Compute | 0 | `packed_scalar1` | 0 (unused for GELU) |
| Compute | 1 | `packed_scalar2` | 0 (unused for GELU) |

---

## Operation Registration and Dispatch Chain

The GELU operation follows this dispatch chain from Python to SFPU:

1. **Python API**: `ttnn.gelu(input_tensor)` or `ttnn.gelu(input_tensor, fast_and_approx=True)`
2. **C++ binding**: Maps to `UnaryWithParam(UnaryOpType::GELU, param)` where `param = 0.0f` (accurate) or `1.0f` (approximate)
3. **Device operation**: `UnaryDeviceOperation` selects the appropriate program factory
4. **Program factory**: `UnaryProgramFactory::create()` builds the program:
   - `get_macro_definition(GELU)` returns `"SFPU_OP_GELU_INCLUDE"`
   - `get_compute_kernel_path(GELU)` returns `"eltwise_sfpu.cpp"` (default case)
   - `get_op_init_and_func(GELU)` returns `{"gelu_tile_init();", "gelu_tile(0);"}` (default) or parameterized variants
5. **Compute kernel**: `eltwise_sfpu.cpp` with `SFPU_OP_CHAIN_0` expanding to `gelu_tile_init(); gelu_tile(0);`
6. **API layer**: `gelu.h` calls `SFPU_INIT_KERNEL_CALL` and `SFPU_UNARY_NO_PARAM_KERNEL_FN`
7. **LLK macros**: Expand to `llk_math_eltwise_unary_sfpu_init` and `_llk_math_eltwise_unary_sfpu_params_`
8. **SFPU kernel**: `calculate_gelu<APPROX>()` dispatches to `_calculate_gelu_appx_` or `_calculate_gelu_accurate_`

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: GELU operation architecture, program factory dispatch, compute kernel paths, `UnaryOpType` enum
- `tenstorrent/tt-llk`: `ckernel::sfpu` namespace details, `_init_gelu_`, `_calculate_gelu_`, LUT and CDF functions
- `tenstorrent/tt-isa-documentation`: SFPU instruction details (`SFPLOADI`, `SFPMAD`, `SFPMUL`, `SFPLUTFP32`, `SFPSETCC`, `SFPENCC`, `SFPCOMPC`, `SFPPOPC`, `SFPPUSHC`), piecewise linear function evaluation, conditional execution

### Confluence References
Not consulted for this analysis. The DeepWiki sources provided sufficient detail on SFPU instructions.

### Glean References
Not consulted for this analysis. The open-source codebase provided all necessary implementation details.

---

## Key Design Decisions

1. **Three computation paths**: The GELU implementation provides three different algorithms with different accuracy/speed tradeoffs:
   - **Approximate (LUT)**: Fastest, uses precomputed piecewise linear segments via `SFPLUTFP32`. Requires only ~5 SFPU instructions per element.
   - **Accurate (CDF polynomial)**: Medium speed, uses a 5th-degree polynomial CDF approximation. More operations but better accuracy.
   - **Accurate (Chebyshev, metal-layer override)**: Slowest but most accurate for |x| < 3, using a 15th-degree polynomial. The metal layer overrides the LLK accurate path with this for non-approximate mode.

2. **LUT coefficient packing**: The 6-segment LUT coefficients are packed as pairs of FP16 values in 32-bit immediates, fitting into just 6 LRegs. This efficient packing is critical because the SFPU has a limited number of local registers (LReg0-7).

3. **Generic compute kernel**: GELU shares the `eltwise_sfpu.cpp` kernel with dozens of other unary operations. Specialization happens entirely through preprocessor defines, avoiding kernel code duplication.

4. **Default to approximate mode**: The `gelu_tile_init` and `gelu_tile` templates default `fast_and_approx = true`, reflecting that most use cases prioritize throughput over precision.

5. **No temporary CB required**: Unlike HARDSHRINK or CBRT, GELU does not need the temporary CB `c_1`, which keeps its memory footprint minimal.
