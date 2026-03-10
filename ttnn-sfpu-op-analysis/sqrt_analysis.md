# SQRT Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | SQRT |
| **Operation Type** | Unary Eltwise SFPU |
| **UnaryOpType Enum** | `UnaryOpType::SQRT` |
| **Program Factory** | `UnaryProgramFactory` (shared with all unary eltwise ops) |
| **Program Factory Path** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` |
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **SFPU Kernel** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_sqrt.h` |
| **Math Description** | Computes element-wise square root: `y = sqrt(x)` for each element in a tile |
| **Approximation Mode** | `false` (always uses high-fidelity path; `get_op_approx_mode` returns `false` for SQRT) |
| **Math Fidelity** | `MathFidelity::HiFi4` |

## Program Factory Structure

### Factory Selection

The SQRT operation uses `UnaryProgramFactory::create`, the shared program factory for all standard unary eltwise operations. There is also a `UnarySubCoreGridProgramFactory` variant for sub-core-grid dispatch, but the standard factory is the primary path.

The factory is parameterized at compile time via preprocessor defines injected through the `get_block_defines` and `get_defines_impl` utility functions, which generate the SFPU operation chain macros.

### Factory: `UnaryProgramFactory`

**Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

**Header**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp`

#### Cached Program Variables

```cpp
struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    tt::tt_metal::KernelHandle unary_writer_kernel_id;
    uint32_t num_cores;
    uint32_t num_cores_y;
};
```

These are cached across invocations so that `override_runtime_arguments` can update buffer addresses without recreating the entire program.

### Work Distribution

Work is split across cores using `tt::tt_metal::split_work_to_cores`, which divides the total number of pages (tiles) across the compute grid. This produces two core groups:
- **core_group_1**: Gets `num_pages_per_core_group_1` tiles each
- **core_group_2**: Gets `num_pages_per_core_group_2` tiles each (handles remainder)

Each core group gets its own compute kernel instance with the appropriate `per_core_block_cnt` compile-time argument.

### Circular Buffer Configuration

| CB Index | Name | Size (pages) | Data Format | Purpose |
|---|---|---|---|---|
| `c_0` (0) | Input | 2 | Input tensor format | Holds input tiles from reader; double-buffered |
| `c_2` (2) | Output | 2 | Output tensor format | Holds result tiles for writer; double-buffered |

Note: CB `c_1` (tmp0) is NOT allocated for SQRT. It is only allocated for HARDSHRINK, CBRT, and LOGIT operations.

The double-buffering (2 pages per CB) allows the reader to fill the next tile while the compute kernel processes the current one, enabling pipeline overlap.

### Compile-Time Defines for SQRT

The program factory generates these defines for SQRT via `get_block_defines` -> `get_defines_impl`:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Top-level SFPU chain macro expanded in compute kernel |
| `SFPU_OP_CHAIN_0_INIT_0` | `sqrt_tile_init();` | Initializes SFPU constants for sqrt |
| `SFPU_OP_CHAIN_0_FUNC_0` | `sqrt_tile(0);` | Executes sqrt on tile at DST index 0 |
| `SFPU_OP_SQRT_INCLUDE` | `1` | Gates inclusion of `api/compute/eltwise_unary/sqrt.h` |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Set based on input dtype (bfloat16 vs float32) |

When SQRT is called with a parameter (e.g., `FAST_APPROX`), the parameterized path generates:
- Init: `sqrt_tile_init();`
- Func: `sqrt_tile<{param0_raw}>(0);` where `param0_raw` is the fast-approx flag

### Compute Kernel Path Resolution

The compute kernel path is determined by `utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype())`. For `UnaryOpType::SQRT`, this falls through to the `default` case, returning `"eltwise_sfpu.cpp"`. The full path becomes:
```
ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp
```

### Runtime Arguments

| Kernel | Argument | Description |
|---|---|---|
| Reader | `src_buffer->address()` | Source buffer DRAM address |
| Reader | `num_pages_per_core` | Number of tiles to read |
| Reader | `num_pages_written` | Start tile ID offset |
| Writer | `dst_buffer->address()` | Destination buffer DRAM address |
| Writer | `num_pages_per_core` | Number of tiles to write |
| Writer | `num_pages_written` | Start tile ID offset |
| Compute | `packed_scalar1` | 0 (unused for SQRT) |
| Compute | `packed_scalar2` | 0 (unused for SQRT) |

## Kernel Implementations

### Reader Kernel

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM address of input buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor config

    constexpr uint32_t cb_id_in0 = 0;                      // Input circular buffer index (c_0)

    // Page size comes from CB interface -- works for both tile and row-major layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                         // Process one page at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

// Iterate over all assigned tiles, reading each from DRAM into the input CB
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {        // Forward iteration (default for SQRT)
#endif
        cb_reserve_back(cb_id_in0, onepage);               // Wait for space in input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);          // Issue async NoC read from DRAM
        noc_async_read_barrier();                           // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);                  // Signal tile is available for compute
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
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM address of output buffer
    const uint32_t num_pages = get_arg_val<uint32_t>(1);    // Number of tiles this core writes
    const uint32_t start_id = get_arg_val<uint32_t>(2);     // Starting tile index for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0); // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();           // Compile-time tensor accessor config

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);                    // For sharded output: wait for all pages at once
#else

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {         // Forward iteration (default for SQRT)
#endif
        cb_wait_front(cb_id_out, onepage);                  // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);    // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);           // Issue async NoC write to DRAM
        noc_async_writes_flushed();                         // Ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);                   // Free the CB slot for compute
    }
    noc_async_write_barrier();                              // Final barrier: all writes complete
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
#include "api/compute/common.h"                              // Common compute API (cb_wait_front, cb_pop_front, etc.)
#include "api/compute/tile_move_copy.h"                      // copy_tile: unpacks a tile from CB into DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"         // init_sfpu, tile_regs_acquire/commit/wait/release, pack_tile
#include "api/compute/eltwise_unary/sfpu_split_includes.h"   // Conditional includes gated by SFPU_OP_*_INCLUDE defines
                                                              // For SQRT: includes api/compute/eltwise_unary/sqrt.h
                                                              // which provides sqrt_tile_init() and sqrt_tile<>()
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);             // Initialize SFPU: configures unpack (c_0) and pack (c_2) pipelines

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim); // Reserve space in output CB for this block

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();                                // Acquire exclusive access to DST registers
                                                                // (blocks until pack from previous iteration completes)

            cb_wait_front(tt::CBIndex::c_0, 1);                // Wait for reader to produce one input tile

            copy_tile(tt::CBIndex::c_0, 0, 0);                 // Unpack tile from input CB slot 0 into DST register 0
                                                                // This invokes the unpacker RISC-V to load tile data
                                                                // from L1 (CB c_0) into the DST register file

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0                                     // Expands to: sqrt_tile_init(); sqrt_tile(0);
                                                                // sqrt_tile_init() loads magic constants into SFPU programmable registers
                                                                // sqrt_tile(0) computes sqrt on all elements in DST[0]
#endif

            tile_regs_commit();                                 // Signal that DST registers are ready for packing

            tile_regs_wait();                                   // Wait for the commit to be acknowledged by the packer

            pack_tile(0, tt::CBIndex::c_2);                     // Pack DST register 0 into output CB c_2
                                                                // Converts from DST format back to output tensor format

            cb_pop_front(tt::CBIndex::c_0, 1);                 // Release the consumed input tile slot

            tile_regs_release();                                // Release DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);    // Signal writer that output tiles are ready
    }
}
```

The key design pattern here is the **tile-at-a-time pipeline**: for each tile, the compute kernel acquires DST registers, unpacks input data into them, applies the SFPU operation, then packs the result back. The `SFPU_OP_CHAIN_0` macro is the injection point -- it gets expanded at compile time to the specific SFPU init+compute calls for whatever unary operation was requested.

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h`
(identical file at `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h`)

The arch-specific wrappers at `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h` are thin forwarders that simply call the shared implementation.

#### LLK API Wrapper

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH                                           // Only compiled for the Math RISC-V core
#include "ckernel_sfpu_sqrt.h"                              // Arch-specific sqrt header (thin wrapper)
#include "llk_math_eltwise_unary_sfpu_macros.h"             // Macro definitions for SFPU dispatch
#endif

namespace ckernel {

// Initialization: loads magic constants into SFPU programmable registers
ALWI void sqrt_tile_init() {
    MATH(                                                    // Executes only on the Math RISC-V
        SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX) // Expands to:
        // llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROX>(sfpu::sqrt_init<APPROX>())
    );
}

// Compute: applies sqrt to all elements of tile at DST index idst
template <bool FAST_APPROX = false>
ALWI void sqrt_tile(uint32_t idst) {
    MATH(
        SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN(                // Expands to:
            calculate_sqrt,                                   // Function name: ckernel::sfpu::calculate_sqrt
            APPROX,                                           // Template param 1: approximation mode (false for SQRT)
            8,                                                // Template param 2: ITERATIONS = 8 (processes 8 rows of 32 elements)
            DST_ACCUM_MODE,                                   // Template param 3: fp32_dest_acc_en
            FAST_APPROX,                                      // Template param 4: fast approximation flag
            idst,                                             // DST register index
            RC                                                // VectorMode: Row-Column (processes full tile)
        )
        // Final expansion: _llk_math_eltwise_unary_sfpu_params_<APPROX>(
        //     ckernel::sfpu::calculate_sqrt<APPROX, 8, DST_ACCUM_MODE, FAST_APPROX>, idst, (int)VectorMode::RC)
    );
}

}  // namespace ckernel
```

#### Annotated SFPU Kernel Source

**Full source of the shared SFPU sqrt implementation**:

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"                      // Legacy sqrt/rsqrt compat implementations
#include "sfpi.h"                                            // SFPI vector programming interface
#include "sfpi_fp16.h"                                       // float_to_fp16b conversion utility

namespace ckernel
{
namespace sfpu
{

// Reference: Kokosinski, Z., Gepner, P., Moroz, L. et al.
// "Fast and accurate approximation algorithms for computing floating point square root."
// Numerical Algorithms (2024). https://doi.org/10.1007/s11075-024-01932-7
//
// This function computes sqrt(x) or 1/sqrt(x) for positive float x.
// The algorithm is based on the "fast inverse square root" trick (manipulating IEEE 754
// bit representation) followed by Newton-Raphson refinement iterations.
template <bool APPROXIMATE = false, bool RECIPROCAL = false, bool FAST_APPROX = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x)
{
    // Step 1: Initial approximation via IEEE 754 bit manipulation.
    // Right-shift the bit pattern by 1 (effectively halving the exponent),
    // then subtract from a magic constant to get an initial reciprocal sqrt estimate.
    // This is the classic "fast inverse square root" trick (Quake III style).
    sfpi::vInt i   = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);
    // At this point, y is approximately 1/sqrt(x).

    if constexpr (APPROXIMATE)
    {
        // Algorithm SQRT_10-bits: gives ~10 bits of mantissa accuracy.
        // Single refinement iteration using the relationship sqrt(x) = x * (1/sqrt(x)).
        sfpi::vFloat c           = x * y;                    // c = x * y ~ sqrt(x) (crude estimate)
        sfpi::vFloat negative_y  = -y;
        sfpi::vFloat infinity    = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::reinterpret<sfpi::vInt>(infinity);
        sfpi::vFloat t           = sfpi::vConstFloatPrgm1 + negative_y * c;
        // t = vConstFloatPrgm1 - y*c is the refinement correction factor.
        // vConstFloatPrgm1 is preloaded with 1.89099014875 in _init_sqrt_.
        if constexpr (RECIPROCAL)
        {
            // For reciprocal sqrt: refine y directly
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            v_if (infinity_minus_x_bits != 0 && x_bits != 0) // x is neither inf nor zero
            {
                y = y * t;                                    // Apply refinement: y = y * t
            }
            v_else                                            // Handle edge cases
            {
                // x=0 -> inf_bits - 0 = inf_bits, so y=inf (1/sqrt(0) = inf)
                // x=inf -> inf_bits - inf_bits = 0, so y=0 (1/sqrt(inf) = 0)
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            // For forward sqrt: use c = x*y as the base estimate
            y = c;                                            // y = x * (1/sqrt(x)) ~ sqrt(x)
            v_if (sfpi::reinterpret<sfpi::vInt>(x) != infinity_bits) // x is not inf
            {
                y = y * t;                                    // Apply refinement: y = c * t
            }
            v_endif;                                          // If x=inf, c=inf which is correct for sqrt(inf)
        }
    }
    else
    {
        // Algorithm SQRT_23-bits: gives ~23 bits of mantissa accuracy (full single precision).
        // Two refinement iterations for higher accuracy.
        sfpi::vFloat xy            = x * y;                   // xy ~ sqrt(x)
        sfpi::vFloat negative_y    = -y;
        sfpi::vFloat c             = negative_y * xy;         // c = -y * xy = -(1/sqrt(x)) * sqrt(x) = -1 + error
        sfpi::vFloat infinity      = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits   = sfpi::reinterpret<sfpi::vInt>(infinity);

        // First refinement: improve the reciprocal sqrt estimate y
        // y = y * (vConstFloatPrgm1 + c * (vConstFloatPrgm2 + c))
        // vConstFloatPrgm1 = 2.2825186, vConstFloatPrgm2 = 2.2533049
        // These constants are from the Kokosinski et al. paper, optimized for this iteration scheme.
        y                          = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));

        // Second refinement: compute new xy product with improved y
        xy                         = x * y;
        negative_y                 = -y;
        sfpi::vFloat one_minus_xyy = sfpi::vConst1 + (negative_y * xy);
        // one_minus_xyy = 1 - x*y^2, which measures how far y is from the true 1/sqrt(x).
        // If y were exactly 1/sqrt(x), then x*y^2 = 1, and this residual would be 0.

        if constexpr (RECIPROCAL)
        {
            // For reciprocal sqrt: Newton-Raphson correction on y
            sfpi::vFloat half_y              = sfpi::addexp(y, -1);  // half_y = y/2 via exponent decrement
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            v_if (infinity_minus_x_bits != 0 && x_bits != 0) // x is finite and nonzero
            {
                y = one_minus_xyy * half_y + y;              // Newton-Raphson: y_new = y + (1-xy^2)*y/2
            }
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits); // Edge cases as in APPROXIMATE
            }
            v_endif;
        }
        else
        {
            // For forward sqrt: refine xy (the sqrt estimate) directly
            sfpi::vFloat half_xy = 0.5f * xy;                // half_xy = sqrt(x)/2
            v_if (sfpi::reinterpret<sfpi::vInt>(x) < infinity_bits) // x is not inf
            {
                // Newton-Raphson on sqrt: sqrt_new = xy + (1-xy^2/x) * xy/2
                // Simplified: y = one_minus_xyy * half_xy + xy
                y = one_minus_xyy * half_xy + xy;
            }
            v_endif;                                          // If x=inf, xy is already inf
        }
    }

    if constexpr (!FAST_APPROX)
    {
        // Safety check: sqrt of negative numbers returns NaN (or inf for bf16)
        v_if (x < 0.0F)
        {
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }

    return y;
}

// Iteration wrapper: processes multiple rows of a tile through the SFPU.
// Each iteration processes one row of 32 elements (the SFPU vector width).
// For a standard 32x32 tile, iterations=8 processes the full tile
// (the SFPU processes 4 elements per cycle across 8 sub-iterations).
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_(const int iterations)
{
#pragma GCC unroll 8                                         // Compiler hint to unroll the loop for performance
    for (int d = 0; d < iterations; d++)
    {
        // Read the current element from DST register, compute sqrt, write back
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = tmp;                          // Write back in fp32 when DST accumulator is fp32
        }
        else
        {
            // Convert result to fp16b before writing back to DST
            // This is needed because the DST register is in bf16 format when fp32_dest_acc is disabled
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0));
        }
        sfpi::dst_reg++;                                     // Advance to next row in DST register
    }
}

// Top-level dispatch: selects between new algorithm and legacy compat
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat = false>
inline void _calculate_sqrt_(int iterations)
{
    if constexpr (legacy_compat)
    {
        // Legacy path: uses the older _sqrt_compat_ function from ckernel_sfpu_rsqrt_compat.h
        // which uses a simpler magic-number + Newton-Raphson approach
        return _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(iterations);
    }
    else
    {
        // New path (default): uses the Kokosinski et al. optimized algorithms
        // RECIPROCAL=false for sqrt (as opposed to rsqrt)
        return _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>(iterations);
    }
}

// Initialization: loads magic constants into SFPU programmable registers.
// These constants are specific to the Kokosinski et al. algorithm and differ
// between approximate and high-precision modes.
template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_sqrt_()
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            // SQRT_10-bits constants (from Kokosinski et al.)
            sfpi::vConstIntPrgm0   = 0x5f0b3892;            // Magic constant for initial reciprocal sqrt approximation
            sfpi::vConstFloatPrgm1 = 1.89099014875f;         // Refinement multiplier for 10-bit accuracy
        }
        else
        {
            // SQRT_23-bits constants (from Kokosinski et al.)
            sfpi::vConstIntPrgm0   = 0x5f1110a0;            // Magic constant (different from 10-bit variant)
            sfpi::vConstFloatPrgm1 = 2.2825186f;             // First refinement constant
            sfpi::vConstFloatPrgm2 = 2.2533049f;             // Second refinement constant
        }
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### Legacy Compat Implementation

The `_calculate_sqrt_compat_` function (from `ckernel_sfpu_rsqrt_compat.h`) provides an older, simpler implementation:

```cpp
// Legacy sqrt using simpler magic-number approach
template <bool APPROXIMATION_MODE, int RECIPROCAL_ITERATIONS>
sfpi_inline sfpi::vFloat _sqrt_compat_(sfpi::vFloat val)
{
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        // Simple bit manipulation: add bias and right-shift
        sfpi::vUInt magic = (127 << 7) << 16;               // 0x3F800000 >> 1 rearranged
        sfpi::vUInt val_s = magic + sfpi::reinterpret<sfpi::vUInt>(val);
        val_s >>= 1;                                         // Halve the exponent
        result = sfpi::reinterpret<sfpi::vFloat>(val_s);     // No refinement -- very rough
    }
    else
    {
        // Newton-Raphson reciprocal sqrt with 2 iterations, then multiply by x
        v_if (val != 0.0f)
        {
            sfpi::vUInt magic   = 0x5f37 << 16;             // Classic magic number
            sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
            }
            result = approx * val;                           // Convert 1/sqrt(x) to sqrt(x)
        }
        v_else { result = val; }                             // sqrt(0) = 0
        v_endif;
    }
    return result;
}

// Legacy wrapper iterating over DST rows
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void _calculate_sqrt_compat_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::dst_reg[0] = _sqrt_compat_<APPROXIMATION_MODE, 2>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}
```

The legacy implementation is NOT used by default. The `legacy_compat` template parameter defaults to `false`, so the new Kokosinski et al. algorithm is always selected for standard SQRT dispatch.

#### SFPU Instructions Used

The SFPU sqrt kernel does not use dedicated hardware sqrt instructions (none exist in the Wormhole or Blackhole ISA). Instead, it is implemented entirely using general-purpose SFPI instructions:

| Instruction / Intrinsic | SFPU Opcode | Description |
|---|---|---|
| `sfpi::reinterpret<vInt/vUInt/vFloat>()` | `SFPCAST` / bitwise reinterpret | Reinterprets vector register between int/uint/float types without conversion |
| `>> 1` (right shift) | `SFPSHFT` | Shifts integer bit pattern right by 1 (halves the IEEE 754 exponent) |
| `vConstIntPrgm0 - i` | `SFPIADD` | Integer subtraction of bit patterns (magic constant minus shifted value) |
| `x * y` (float multiply) | `SFPMUL` | Vector floating-point multiplication |
| `a + b` (float add) | `SFPMAD` / `SFPIADD` | Vector floating-point addition (often fused with multiply as MAD) |
| `-y` (negate) | `SFPSETSGN` | Sets sign bit of floating-point value |
| `v_if / v_else / v_endif` | `SFPSETCC` / `SFPENCC` / `SFPCOMPC` | Conditional vector execution using SFPU condition codes |
| `sfpi::s2vFloat16b()` | `SFPLOADI` | Loads an immediate scalar into a vector register |
| `sfpi::addexp(y, -1)` | `SFPIADD` with exponent mode | Decrements the exponent by 1 (effectively divides by 2) |
| `sfpi::vConstFloatPrgm1/2` | `SFPLOAD` from programmable registers | Reads preloaded constants from SFPU programmable constant registers |
| `float_to_fp16b(tmp, 0)` | Software conversion | Converts fp32 result to bf16 format for DST writeback |
| `sfpi::dst_reg[0]` read/write | `SFPLOAD` / `SFPSTORE` to DEST | Reads from / writes to the destination register file |
| `sfpi::dst_reg++` | Increment DEST pointer | Advances the DST register row pointer |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` | Source and destination: reads input value, writes sqrt result |
| `dst_reg++` | Advances row pointer after each iteration |
| `vConstIntPrgm0` | Programmable integer constant register: holds the magic number for initial approximation (0x5f0b3892 or 0x5f1110a0) |
| `vConstFloatPrgm1` | Programmable float constant register: holds first refinement constant (1.89099... or 2.2825186) |
| `vConstFloatPrgm2` | Programmable float constant register: holds second refinement constant (2.2533049, only used in 23-bit mode) |
| `vConst1` | Hardware constant register: holds 1.0f |
| SFPU LRegs (implicit) | Temporary vector registers used for intermediate calculations (i, y, c, xy, etc.) |

#### SFPU Execution Flow

1. **Initialization** (`sqrt_tile_init` -> `_init_sqrt_`):
   - Loads magic constants into SFPU programmable registers (`vConstIntPrgm0`, `vConstFloatPrgm1`, optionally `vConstFloatPrgm2`)
   - Constants depend on approximation mode: 10-bit vs 23-bit algorithm
   - This initialization is called once before processing tiles

2. **Tile acquisition** (in compute kernel):
   - `cb_wait_front(c_0, 1)` blocks until reader has produced a tile
   - `tile_regs_acquire()` acquires exclusive DST register access
   - `copy_tile(c_0, 0, 0)` unpacks the tile from L1 (CB c_0) into DST register 0

3. **SFPU computation** (`sqrt_tile` -> `calculate_sqrt` -> `_calculate_sqrt_internal_`):
   - Iterates 8 times (ITERATIONS=8), processing one row of the tile per iteration
   - Each iteration reads `dst_reg[0]`, computes `_calculate_sqrt_body_`, writes result back
   - The body function:
     a. Reinterprets float as integer, right-shifts by 1 (halves exponent)
     b. Subtracts from magic constant to get initial 1/sqrt(x) estimate
     c. **23-bit path** (default): Two Newton-Raphson refinement iterations using optimized Kokosinski constants
     d. Multiplies by x to convert from 1/sqrt(x) to sqrt(x)
     e. Handles edge cases: x=inf (result=inf), x<0 (result=NaN)
   - After computing, converts to fp16b if `fp32_dest_acc_en` is false

4. **Result packing** (in compute kernel):
   - `tile_regs_commit()` signals DST registers are ready
   - `tile_regs_wait()` waits for packer acknowledgment
   - `pack_tile(0, c_2)` packs DST register 0 into output CB c_2
   - `cb_pop_front(c_0, 1)` releases the input CB slot
   - `tile_regs_release()` releases DST registers

5. **Output** (in compute kernel):
   - `cb_push_back(c_2, per_core_block_dim)` signals the writer that output tiles are ready

#### SFPU Configuration

| Configuration | Value | Notes |
|---|---|---|
| `APPROX` (approximation mode) | `false` | `get_op_approx_mode(UnaryOpType::SQRT)` returns `false` -- always uses 23-bit high-precision path |
| `ITERATIONS` | 8 | Processes 8 rows per tile (SFPU processes 4 elements per lane, 8 iterations x 4 = 32 rows for full tile) |
| `DST_ACCUM_MODE` | Depends on `fp32_dest_acc_en` | When true, DST stores fp32; when false, results are truncated to bf16 before writeback |
| `FAST_APPROX` | `false` by default | When true, skips the negative-input NaN check. Can be set to true via parameterized SQRT |
| `legacy_compat` | `false` | New Kokosinski algorithm is always used; legacy path exists but is not selected |
| `RECIPROCAL` | `false` | Hardcoded to false for SQRT (true for RSQRT) |
| `MathFidelity` | `HiFi4` | Set in ComputeConfig; highest fidelity mode |
| `math_approx_mode` | `false` | Determined by `get_op_approx_mode`, always false for SQRT |

#### Algorithm Details

The SQRT implementation uses a two-stage approach based on the 2024 Kokosinski et al. paper:

**Stage 1 - Initial Approximation (Fast Inverse Square Root trick):**
Given input x, compute an initial estimate of 1/sqrt(x) using IEEE 754 bit manipulation:
```
i = reinterpret_as_int(x) >> 1
y = reinterpret_as_float(MAGIC - i)
```
This exploits the fact that for IEEE 754 floats, the exponent and mantissa together form a rough logarithm, so halving the bit pattern and subtracting from a magic constant yields an approximate reciprocal square root.

**Stage 2 - Newton-Raphson Refinement (23-bit path):**
Two iterations refine the estimate to full single-precision accuracy:
1. First refinement: `y = y * (C1 + c * (C2 + c))` where `c = -y * (x * y)` and C1=2.2825186, C2=2.2533049
2. Second refinement: `y_final = (1 - x*y^2) * (x*y)/2 + x*y`

The final result is sqrt(x), not 1/sqrt(x), because the last step computes `x*y` (which equals `x/sqrt(x) = sqrt(x)`).

#### Hardware Compatibility Notes

- **Wormhole B0 and Blackhole**: The SFPU sqrt kernel source code is **identical** for both architectures. Both use the same software-based approach since neither architecture has a dedicated hardware sqrt instruction.
- **No hardware sqrt instruction**: Neither Wormhole nor Blackhole SFPU ISA includes a dedicated sqrt instruction. The Baby RISC-V cores also do not implement `fsqrt.s` from the RISC-V F extension.
- **Blackhole SFPARECIP**: Blackhole introduces an approximate reciprocal instruction (`SFPARECIP`) that could theoretically be used as a building block for sqrt (via reciprocal sqrt), but the current implementation does not use it -- it relies entirely on SFPI arithmetic instructions.
- **SFPLUT**: Both architectures have `SFPLUT` (lookup-table-based fused multiply-add), but it is not used for sqrt computation.

## Data Flow Summary

```
DRAM Input Buffer
      |
      | (NoC async read, one tile at a time)
      v
  CB c_0 (Input, 2 tiles double-buffered)
      |
      | cb_wait_front -> copy_tile (unpack into DST)
      v
  DST Register [0]
      |
      | sqrt_tile_init() -> load magic constants
      | sqrt_tile(0) -> _calculate_sqrt_internal_ (8 iterations)
      |   Per iteration:
      |     1. Read dst_reg[0]
      |     2. Bit-manipulate for initial 1/sqrt(x) estimate
      |     3. Two Newton-Raphson refinements (23-bit accuracy)
      |     4. Convert 1/sqrt(x) to sqrt(x) via multiplication by x
      |     5. Write result back to dst_reg[0]
      |     6. Advance dst_reg++
      |
      | pack_tile (pack from DST into output CB)
      v
  CB c_2 (Output, 2 tiles double-buffered)
      |
      | cb_wait_front -> noc_async_write_page
      v
DRAM Output Buffer
```

## File Inventory

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory: creates program, allocates CBs, creates kernels, sets runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header: defines cached_program_t and shared_variables_t |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Utility functions: `get_compute_kernel_path`, `get_op_init_and_func`, `get_block_defines`, `get_op_approx_mode` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | Utility header |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Defines `UnaryOpType::SQRT` enum |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Compute kernel: tile-at-a-time SFPU dispatch |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel: reads tiles from DRAM into input CB |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel: writes tiles from output CB to DRAM |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h` | LLK API: `sqrt_tile_init()` and `sqrt_tile()` functions |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include system: gates `sqrt.h` inclusion via `SFPU_OP_SQRT_INCLUDE` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h` | Arch wrapper (WH): thin forwarder to shared implementation |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h` | Arch wrapper (BH): thin forwarder to shared implementation |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h` | Shared SFPU sqrt implementation (Kokosinski et al. algorithm) |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h` | Shared SFPU sqrt implementation (identical to WH) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h` | Legacy compat: `_sqrt_compat_` and `_calculate_sqrt_compat_` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | SFPU dispatch macros: `SFPU_INIT_KERNEL_CALL`, `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN` |

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Confirmed compute kernel path resolution, SFPU define injection mechanism, and program factory dispatch flow
- `tenstorrent/tt-llk`: Detailed `_calculate_sqrt_body_` implementation, iteration structure, and SFPI usage patterns
- `tenstorrent/tt-isa-documentation`: Confirmed no dedicated SQRT instruction exists in Wormhole or Blackhole SFPU ISA; documented SFPARECIP availability in Blackhole
- `tenstorrent/sfpi`: Confirmed SFPI intrinsics used (reinterpret, shft, arithmetic ops, v_if/v_endif, dst_reg access)

### Confluence References
Not consulted for this analysis. The DeepWiki sources provided sufficient detail on SFPU instruction capabilities and the sqrt algorithm.

### Glean References
Not consulted for this analysis. The source code and DeepWiki provided complete coverage of the sqrt implementation across both hardware architectures.
