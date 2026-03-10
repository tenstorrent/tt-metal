# GELU Implementation Analysis

## Overview
The GELU (Gaussian Error Linear Unit) operation computes `GELU(x) = x * Phi(x)` where `Phi(x)` is the cumulative distribution function of the standard normal distribution. This is a widely used activation function in transformer models. The operation supports two modes: an approximation mode using a hardware LUT-based piecewise linear function, and an accurate mode using a polynomial CDF approximation.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The GELU operation uses the shared `UnaryProgramFactory` which dispatches to `eltwise_sfpu.cpp` as the compute kernel. The operation-specific behavior is injected via preprocessor defines (`SFPU_OP_GELU_INCLUDE` and `SFPU_OP_CHAIN_0`) that configure the generic kernel to call `gelu_tile_init()` and `gelu_tile()`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt`), inner loop over tiles within block (`per_core_block_dim` = 1). Each iteration processes one tile. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to linear tile sequence |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, or other supported types |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations
No layout transformations are performed. Input and output must both be in TILE_LAYOUT.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU GELU op, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT allocated for GELU since it is only created for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary
Both input (c_0) and output (c_2) circular buffers use double buffering (capacity = 2 tiles, block size = 1 tile). This allows the reader to fill one slot while compute processes the other, and similarly for compute/writer overlap.

## Index Calculations
The program factory uses `TensorAccessor` for both reader and writer kernels, which handles the mapping from a linear page index to the physical DRAM/L1 address. The reader kernel iterates sequentially from `start_id` to `start_id + num_pages`, reading one tile per iteration using `noc_async_read_page(i, s, l1_write_addr)`.

The writer mirrors this pattern with `noc_async_write_page(i, s, l1_read_addr)`.

## Memory Access Patterns

### Read Pattern
Sequential tile-by-tile reads. Each core reads a contiguous range of tile IDs from `start_id` to `start_id + num_pages_per_core`. Within each iteration, a single tile is read from DRAM/L1 via NoC into the input CB.

### Write Pattern
Sequential tile-by-tile writes. Each core writes its computed tiles in the same order to contiguous tile IDs in the output buffer.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Core indexing uses column-major order: `core = {i / num_cores_y, i % num_cores_y}`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor configuration for src_buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor configuration for dst_buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles for this core to read |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles for this core to write |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for GELU (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for GELU (set to 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0 | Read tiles via TensorAccessor |
| compute | TRISC (math RISCV) | N/A | CB c_0 | CB c_2 | Unpack, SFPU GELU, pack |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write tiles via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Reads one page at a time from `start_id` to `end_id`, using `TensorAccessor` to resolve physical addresses. Supports optional `BACKWARDS` mode via define.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Waits for compute to produce one page in CB c_2, then writes it out via NoC. Also supports `OUT_SHARDED` mode (not used in this factory) and `BACKWARDS` mode.

### Compute Kernel
This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source
```cpp
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes gelu.h when SFPU_OP_GELU_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (always 1 for GELU)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initialize unpack (c_0 -> SRC) and pack (DEST -> c_2) pipelines
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for 1 tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers

            cb_wait_front(tt::CBIndex::c_0, 1);  // wait until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from c_0 into DEST[0] via the unpacker

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: gelu_tile_init<P>(); gelu_tile<P>(0);
                             // where P is the approx mode parameter (0=accurate, 1=approx)
                             // gelu_tile_init configures LUT registers for SFPU GELU
                             // gelu_tile(0) runs the SFPU GELU kernel on DEST[0]
#endif

            tile_regs_commit();  // signal that DEST registers are ready for packing

            tile_regs_wait();  // wait for packer to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish produced tile(s) to writer
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **API layer**: `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h`
- **LLK implementation (Blackhole)**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h`
- **LLK implementation (Wormhole B0)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h`
- **CDF helper**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_cdf.h`
- **Dispatch layer**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`

#### Annotated SFPU Kernel Source

**API layer** (`gelu.h`):
```cpp
#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_gelu.h"       // LLK-layer GELU SFPU functions
#include "llk_math_eltwise_unary_sfpu_macros.h"  // macro definitions for dispatching SFPU ops
#endif

namespace ckernel {

// Initializes the SFPU pipeline for GELU computation.
// When fast_and_approx=true (default), loads LUT coefficients into SFPU local registers.
// When fast_and_approx=false, no special LUT setup needed (uses polynomial CDF).
template <bool fast_and_approx = true>
ALWI void gelu_tile_init() {
    // SFPU_INIT_KERNEL_CALL expands to:
    //   llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, fast_and_approx>(sfpu::gelu_init<fast_and_approx>)
    // This calls gelu_init which loads piecewise-linear LUT coefficients into l_reg[0..6]
    MATH(SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, fast_and_approx));
}

// Performs element-wise GELU on tile at DEST[tile_index].
// DST register buffer must be acquired via tile_regs_acquire().
// fast_and_approx: true = LUT-based approximation, false = polynomial CDF
template <bool fast_and_approx = true>
ALWI void gelu_tile(uint32_t idst) {
    // SFPU_UNARY_NO_PARAM_KERNEL_FN expands to:
    //   _llk_math_eltwise_unary_sfpu_params_<fast_and_approx>(
    //       ckernel::sfpu::calculate_gelu<fast_and_approx>, idst, (int)VectorMode::RC)
    // VectorMode::RC means process all 4 faces of the 32x32 tile
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, fast_and_approx, idst));
}

}  // namespace ckernel
```

**LLK dispatch layer** (`llk_math_eltwise_unary_sfpu_params.h`):
```cpp
// This function orchestrates SFPU execution across all 4 faces of a tile.
// For VectorMode::RC (the mode used by GELU), it iterates over all 4 faces:
//   Face 0: rows 0-15, cols 0-15
//   Face 1: rows 0-15, cols 16-31
//   Face 2: rows 16-31, cols 0-15
//   Face 3: rows 16-31, cols 16-31
// Each face call to sfpu_func processes 8 rows (ITERATIONS=8), each row having 16 elements.
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index, int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(dst_index);  // set DEST register base address

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::RC)
    {
        // Process all four 16x16 faces of the 32x32 tile
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            // Advance DEST face address pointer to next 16x16 face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    _llk_math_eltwise_unary_sfpu_done_();  // finalize SFPU execution
}
```

**LLK SFPU kernel** (`ckernel_sfpu_gelu.h` - Blackhole, identical to Wormhole B0):
```cpp
#pragma once

#include <cstdint>

#include "ckernel_sfpu_cdf.h"          // _calculate_cdf_appx_ for accurate mode
#include "ckernel_sfpu_exp.h"          // exponential helpers (used by derivative, not main GELU)
#include "ckernel_sfpu_load_config.h"  // _sfpu_load_imm32_ for loading LUT constants
#include "sfpi.h"                      // SFPU programming interface (vFloat, dst_reg, l_reg, etc.)
#include "sfpi_fp16.h"                 // FP16 conversion utilities

namespace ckernel::sfpu
{

// Initialization function: loads piecewise-linear LUT coefficients into SFPU local registers.
// These coefficients implement erf(x/sqrt(2)) as a 6-piece linear approximation
// over the ranges [0,0.5], [0.5,1.0], [1.0,1.5], [1.5,2.0], [2.0,3.0], [3.0,+inf).
// The function is symmetric (sign is handled by lut2_sign).
template <bool APPROXIMATION_MODE>
inline void _init_gelu_()
{
    sfpi::vConstFloatPrgm0 = 0.5f;  // store 0.5 in programmable constant register for use in compute

    // Load piecewise-linear slope (A) and intercept (B) coefficients into local registers.
    // Each 32-bit immediate packs two FP16b values: [hi_range_coeff | lo_range_coeff].
    // LReg0: A for ranges [0.5,1.0] (hi=0.4939) and [0.0,0.5] (lo=0.1928)
    _sfpu_load_imm32_(0, 0x37E7322B);
    // LReg4: B for ranges [0.5,1.0] (hi=-0.1605) and [0.0,0.5] (lo=-0.0150)
    _sfpu_load_imm32_(4, 0xB12286D8);

    // LReg1: A for ranges [1.5,2.0] (hi=0.6099) and [1.0,1.5] (lo=0.6189)
    _sfpu_load_imm32_(1, 0x38E138F3);
    // LReg5: B for ranges [1.5,2.0] (hi=-0.2635) and [1.0,1.5] (lo=-0.2797)
    _sfpu_load_imm32_(5, 0xB437B479);

    // LReg2: A for ranges [3.0,+inf) (hi=0.50) and [2.0,3.0] (lo=0.5402)
    _sfpu_load_imm32_(2, 0x38003852);
    // LReg6: B for ranges [3.0,+inf) (hi=0.0/inf marker) and [2.0,3.0] (lo=-0.1194)
    _sfpu_load_imm32_(6, 0x7c00afa4);
}

// Approximation mode GELU computation.
// For each of 8 rows in a face (ITERATIONS=8, each row = 16 SIMD elements):
//   result = 0.5*x + lut2_sign(x)
// where lut2_sign evaluates the piecewise-linear LUT on |x| and applies the sign of x.
// This computes x * 0.5 * (1 + erf(x/sqrt(2))) via the identity:
//   GELU(x) = 0.5*x + x_sign * LUT(|x|)
template <int ITERATIONS>
inline void _calculate_gelu_appx_()
{
    // Cache local registers into vUInt variables for efficient repeated access
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];  // slopes for ranges [0.5,1.0] and [0.0,0.5]
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];  // slopes for ranges [1.5,2.0] and [1.0,1.5]
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];  // slopes for ranges [3.0,inf) and [2.0,3.0]
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];  // intercepts for ranges [0.5,1.0] and [0.0,0.5]
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];  // intercepts for ranges [1.5,2.0] and [1.0,1.5]
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];  // intercepts for ranges [3.0,inf) and [2.0,3.0]

#pragma GCC unroll 8  // fully unroll: one iteration per row in a 16x16 face
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in      = sfpi::dst_reg[0];           // load current row from DEST register
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;      // 0.5 from programmable constant
        sfpi::vFloat half_in = in * half;                    // compute 0.5 * x
        // lut2_sign: evaluates 6-piece piecewise-linear function on |x|, then applies sign of x.
        // Uses SFPLUTFP32 instruction with FP16_6ENTRY_TABLE1 mode and SGN_UPDATE.
        // The LUT computes: sign(x) * (A_i * |x| + B_i) for the appropriate range i.
        // This approximates 0.5 * erf(x/sqrt(2)) * |x| with the sign correction.
        sfpi::vFloat result  = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result               = half_in + result;             // GELU(x) = 0.5*x + lut_result

        sfpi::dst_reg[0] = result;                           // write result back to DEST register

        sfpi::dst_reg++;                                     // advance to next row in face
    }

    // Restore local registers (they may have been clobbered by SFPU pipeline)
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

// Accurate mode GELU computation using polynomial CDF approximation.
// Computes GELU(x) = x * CDF(x) where CDF is approximated by a degree-4 polynomial
// for |x| < 2.5 and a linear function for 2.5 <= |x| < 5.
template <int ITERATIONS>
inline void _calculate_gelu_accurate_()
{
    constexpr bool scaled = true;  // tells _calculate_cdf_appx_ to multiply result by x
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];              // load current row from DEST
        sfpi::vFloat result = _calculate_cdf_appx_(in, scaled);  // CDF(x) * x = GELU(x)
        sfpi::dst_reg[0]    = result;                        // write back to DEST
        sfpi::dst_reg++;                                     // advance to next row
    }
}

// Top-level dispatcher: selects approximation or accurate mode at compile time.
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

} // namespace ckernel::sfpu
```

**CDF helper** (`ckernel_sfpu_cdf.h`, used by accurate mode):
```cpp
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_polyval.h"  // POLYVAL5 polynomial evaluation
#include "sfpi.h"

namespace ckernel::sfpu
{

// Computes CDF for positive values using two polynomial segments.
inline sfpi::vFloat _calculate_pos_cdf_appx_(sfpi::vFloat val)
{
    // Two-piece polynomial approximation:
    // For val in [0, 2.5): degree-4 polynomial with coefficients optimized for CDF fit
    // For val in [2.5, 5): linear approximation 0.44656975 * val + 0.58216001
    sfpi::vFloat result;
    v_if (val < 2.5f)
    {
        // POLYVAL5 evaluates: c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
        result = POLYVAL5<sfpi::vFloat>(0.0122792f, -0.05281024f, -0.03048313f, 0.41314081f, 0.49866379f, val);
    }
    v_else
    {
        // Linear approximation for larger values (approaching 1.0)
        result = 0.44656975f * val + 0.58216001f;
    }
    v_endif;

    // Clamp to [0, 1] range
    v_if (result > 1.0f)
    {
        result = 1.0f;
    }
    v_endif;
    return result;
}

// Computes CDF of the standard normal distribution.
// Uses symmetry: CDF(-x) = 1 - CDF(x).
// When scaled=true, returns CDF(x) * x (i.e., GELU(x)).
inline sfpi::vFloat _calculate_cdf_appx_(sfpi::vFloat val, bool scaled = false)
{
    sfpi::vFloat result = 0.0f;

    v_if (val < 0.0f)
    {
        result = 1.0f - _calculate_pos_cdf_appx_(-val);  // symmetry for negative inputs
    }
    v_else
    {
        result = _calculate_pos_cdf_appx_(val);           // direct computation for positive
    }
    v_endif;

    if (scaled)
    {
        result *= val;  // GELU(x) = x * CDF(x)
    }
    return result;
}

} // namespace ckernel::sfpu
```

**Metal-layer GELU wrapper** (`ckernel_sfpu_gelu.h` in `tt_metal/hw/ckernels/blackhole/`):
```cpp
#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

namespace ckernel {
namespace sfpu {

// Horner's method evaluation of a 15th-degree Chebyshev polynomial.
// Used only in the non-approximation path of the metal layer's calculate_gelu.
#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x) \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + (c10)) * (x) + (c9)) * \
                (x) + (c8)) * (x) + (c7)) * (x) + (c6)) * (x) + (c5)) * (x) + (c4)) * (x) + \
       (c3)) * (x) + (c2)) * (x) + (c1)) * (x) + (c0)

// Alternative Chebyshev-based GELU for the metal layer's non-approx path.
// Computes GELU(x) directly via a 15th-degree polynomial for |x| < 5.5.
inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            -1.81205228163e-09, -4.59055119276e-08, -3.74540617693e-07,
            -2.29754133825e-07,  1.19076782913e-05,  4.25116466215e-05,
            -0.000138391838381, -0.000862052441087,  0.000768340223025,
             0.0092074331601,   -0.00208478037614,  -0.0656369476513,
             0.00244542739174,   0.398579460781,     0.499174645395,
             2.98325768482e-05, val);
        result = setsgn(result, val);  // ensure result has same sign as input
    }
    v_endif;
    return result;
}

// Initialization: delegates to LLK _init_gelu_ which loads LUT coefficients.
template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

// Main GELU computation entry point.
// In approx mode: delegates to _calculate_gelu_ (LUT-based).
// In non-approx mode: uses Chebyshev polynomial for |x| < 3, passthrough for |x| >= 3.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();  // LLK appx path
    } else {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = in;           // default: passthrough for |x| >= 3
        v_if(in == 0.0f) { result = 0.0f; }  // special case: GELU(0) = 0
        v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }  // polynomial for |x| < 3
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[0]` (SFPLOAD/SFPSTORE) | Reads/writes a row of 16 elements from/to the DEST register file |
| `sfpi::dst_reg++` (INCRWC) | Advances the DEST register row pointer by one row (16 elements) |
| `sfpi::vConstFloatPrgm0` | Programmable constant register, loaded with 0.5f during init |
| `_sfpu_load_imm32_` (SFPLOADI) | Loads a 32-bit immediate value into an SFPU local register |
| `sfpi::l_reg[N]` (LREG access) | Reads from SFPU local registers 0-6 holding LUT coefficients |
| `lut2_sign` (SFPLUTFP32) | 6-piece piecewise-linear LUT evaluation with sign update. Uses `__builtin_rvtt_sfplutfp32_6r` with `SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 \| SFPLUTFP32_MOD0_SGN_UPDATE` |
| `v_if` / `v_elseif` / `v_else` / `v_endif` (SFPSETCC/SFPENCC) | SIMD predicated execution - enables conditional per-element branching |
| `setsgn` (SFPSETSGN) | Copies the sign bit from one value to another |
| Multiply `*` (SFPMUL/SFPMAD) | SFPU floating-point multiply |
| Add `+` (SFPMAD/SFPADD) | SFPU floating-point add |

#### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **DEST[0]** (dst_reg[0]) | Input/output for each row; holds tile data being processed |
| **vConstFloatPrgm0** | Stores 0.5f constant used in approx mode |
| **LReg0** | Slopes for ranges [0.5,1.0] (hi) and [0.0,0.5] (lo) - packed FP16b |
| **LReg1** | Slopes for ranges [1.5,2.0] (hi) and [1.0,1.5] (lo) |
| **LReg2** | Slopes for ranges [3.0,inf) (hi) and [2.0,3.0] (lo) |
| **LReg4** | Intercepts for ranges [0.5,1.0] (hi) and [0.0,0.5] (lo) |
| **LReg5** | Intercepts for ranges [1.5,2.0] (hi) and [1.0,1.5] (lo) |
| **LReg6** | Intercepts for ranges [3.0,inf) (hi) and [2.0,3.0] (lo) |

Note: LReg3 and LReg7 are not used by GELU init. The local registers are cached into `vUInt` variables at the start of the loop and restored after to preserve state across iterations.

#### SFPU Execution Flow

1. **Initialization** (`gelu_tile_init` -> `_init_gelu_`):
   - Sets `vConstFloatPrgm0 = 0.5f`
   - Loads 6 pairs of (slope, intercept) coefficients as packed FP16b values into LReg0, LReg1, LReg2, LReg4, LReg5, LReg6 via `_sfpu_load_imm32_`

2. **Tile dispatch** (`gelu_tile` -> `_llk_math_eltwise_unary_sfpu_params_`):
   - Sets DEST register base address to the target tile
   - Iterates over 4 faces (each 16x16 sub-block of the 32x32 tile)
   - For each face, calls `calculate_gelu<APPROX_MODE>()`

3. **Per-face computation** (8 iterations per face, one per row of 16 elements):
   - **Approximation mode** (`_calculate_gelu_appx_`):
     - Loads row from DEST
     - Computes `half_in = 0.5 * x`
     - Evaluates `lut2_sign(x, l0..l6)` which performs: `sign(x) * (A_i * |x| + B_i)` for the appropriate piecewise segment
     - Computes `result = half_in + lut_result`
     - Stores result back to DEST, advances row pointer
   - **Accurate mode** (`_calculate_gelu_accurate_`):
     - Loads row from DEST
     - Calls `_calculate_cdf_appx_(x, scaled=true)` which computes `x * CDF(x)` using a degree-4 polynomial for |x| < 2.5 and a linear fit for 2.5 <= |x| < 5
     - Stores result back to DEST, advances row pointer

4. **Completion**: After all 4 faces processed, `_llk_math_eltwise_unary_sfpu_done_()` is called to finalize SFPU state.

#### SFPU Configuration

| Configuration | Value |
|--------------|-------|
| **Math fidelity** | HiFi4 |
| **Math approx mode** | `false` (from `get_op_approx_mode` which returns false for GELU) |
| **fp32_dest_acc_en** | Configurable via `args.fp32_dest_acc_en` |
| **Unpack to dest mode** | Default (or UnpackToDestFp32 if `preserve_fp32_precision`) |
| **SFPU_OP_GELU_INCLUDE** | Set to "1" - includes `gelu.h` |
| **SFPU_OP_CHAIN_0** | Set to `gelu_tile_init<P>(); gelu_tile<P>(0);` where P depends on parameter |
| **APPROXIMATION_MODE template** | Controlled by param0 of UnaryOpType::GELU (0=accurate, 1=approx) |

The GELU operation's approximation mode is controlled by the `param0` parameter passed when constructing `UnaryWithParam(UnaryOpType::GELU, static_cast<float>(bool))`. This is encoded as a template parameter: `gelu_tile_init<P>()` and `gelu_tile<P>(idst)` where P is 0 (accurate) or 1 (fast approximate).

Importantly, `get_op_approx_mode` returns `false` for GELU, meaning the global `math_approx_mode` flag in `ComputeConfig` is set to `false`. The per-operation approximation is instead controlled by the template parameter.

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations of the GELU SFPU kernel are **identical** (confirmed by diff). Both architectures:
- Use the same LUT coefficients and piecewise-linear approximation
- Use the same `lut2_sign` / `SFPLUTFP32` instruction interface
- Use the same CDF polynomial approximation for accurate mode
- Share the same `_sfpu_load_imm32_` instruction for loading constants

The Chebyshev polynomial path in the metal-layer `calculate_gelu` (non-approx, non-LLK) uses `setsgn` which is available on both architectures.

## Implementation Notes

1. **Dual non-approx paths**: There are two different "accurate" implementations. The metal-layer `calculate_gelu` (in `ckernel_sfpu_gelu.h` under `tt_metal/hw/ckernels/`) uses a 15th-degree Chebyshev polynomial for |x| < 3. The LLK-layer `_calculate_gelu_accurate_` (in `tt_metal/third_party/tt_llk/`) uses a 4th-degree polynomial CDF approximation. In approximation mode, both paths converge to the same LUT-based `_calculate_gelu_appx_`.

2. **No packed scalars for GELU**: The program factory sets `packed_scalar1 = 0` and `packed_scalar2 = 0` since GELU does not use runtime scalar parameters (unlike HARDSHRINK or WHERE_TSS).

3. **No temporary CB**: CB c_1 (tmp0) is not allocated for GELU, reducing L1 memory usage.

4. **LUT coefficient packing**: Each `_sfpu_load_imm32_` call packs two FP16b coefficients into a single 32-bit word. The SFPLUTFP32 instruction hardware unpacks these based on which range the input magnitude falls into, selecting the appropriate (slope, intercept) pair for the piecewise-linear evaluation.

5. **Sign handling via lut2_sign**: The `lut2_sign` variant of `lut2` automatically applies the sign of the input to the LUT output via the `SFPLUTFP32_MOD0_SGN_UPDATE` flag. This exploits the symmetry of erf(x) being an odd function, allowing the LUT to store only positive-side coefficients.

6. **Face-level parallelism**: Each `calculate_gelu` call processes 8 rows of 16 elements (128 elements per face), and the dispatch layer calls it 4 times per tile for a total of 512 elements (one 16x16 face x 4 = full 32x32 tile). The loop is fully unrolled (`#pragma GCC unroll 8`) for maximum instruction-level parallelism.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How is the GELU unary SFPU operation implemented? What compute kernel files are used, and how does the unary program factory dispatch GELU?"
   **Reason**: Initial reconnaissance to understand the GELU dispatch chain and locate all relevant source files.
   **Key Findings**: Identified the full file hierarchy: program factory -> eltwise_sfpu.cpp -> gelu.h -> ckernel_sfpu_gelu.h. Learned about the SFPU_OP_GELU_INCLUDE define mechanism and the two-mode (approx/accurate) implementation.

2. **Query**: "How is the GELU SFPU kernel implemented in LLK? What functions like gelu_tile_init and gelu_tile are defined, and what SFPU instructions do they use?" (tenstorrent/tt-llk)
   **Reason**: Needed detailed LLK-layer implementation information including SFPU instructions and register usage.
   **Key Findings**: Confirmed the `_init_gelu_` / `_calculate_gelu_appx_` / `_calculate_gelu_accurate_` function structure. Learned about `lut2_sign`, `vConstFloatPrgm0`, `_sfpu_load_imm32_`, and the 6-piece piecewise-linear approximation scheme.

### Documentation References
1. **Source**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Needed to understand the `lut2_sign` function implementation and which SFPU hardware instruction it maps to.
   **Key Information**: `lut2_sign` with 6 vUInt arguments maps to `__builtin_rvtt_sfplutfp32_6r` with `SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | SFPLUTFP32_MOD0_SGN_UPDATE` mode flags.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how GELU's macro defines are generated, how approx mode is controlled, and which compute kernel path is selected.
   **Key Information**: GELU maps to `SFPU_OP_GELU_INCLUDE` define, uses `eltwise_sfpu.cpp` as default kernel path, generates `gelu_tile_init<P>(); gelu_tile<P>(idst);` where P is the approx mode from param0.
