# SQRT Implementation Analysis

## Overview

The SQRT operation computes the element-wise square root of each element in an input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which dispatches a reader, compute, and writer kernel across multiple Tensix cores. The SFPU kernel uses a software-based Newton-Raphson-style iterative algorithm (based on a fast inverse square root seed) rather than a hardware LUT, with two accuracy modes: a 10-bit approximate mode and a 23-bit full-precision mode.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `input.buffer()->num_pages()` (total tiles in tensor) |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt`), inner loop over tiles per block (`per_core_block_dim` = 1) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to tiles |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (or specified output dtype) |

### Layout Transformations

No layout transformations are performed. The input and output share the same layout. If the input is in TILE_LAYOUT, pages are tiles; if ROW_MAJOR, pages are rows. The CB page size is set accordingly.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU sqrt, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 pages | 1 page | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 pages | 1 page | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT created for SQRT. It is only allocated for HARDSHRINK, CBRT, or LOGIT operations.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 pages with block size = 1 page, enabling **double-buffered** operation. This allows the reader to fill a second tile into c_0 while compute processes the first, and compute to write a result into c_2 while the writer drains the previous result. This provides overlap between all three pipeline stages.

## Index Calculations

The program factory uses `TensorAccessor` for both reader and writer kernels. The accessor encapsulates the mapping from a linear page index to a physical DRAM address, handling interleaved bank distribution transparently. Each core receives a contiguous range of page indices defined by `(start_id, start_id + num_pages_per_core)`.

- Reader: iterates `i` from `start_id` to `start_id + num_pages`, calling `noc_async_read_page(i, s, l1_write_addr)`.
- Writer: iterates `i` from `start_id` to `start_id + num_pages`, calling `noc_async_write_page(i, s, l1_read_addr)`.

## Memory Access Patterns

### Read Pattern
Sequential page reads. Each core reads its assigned contiguous range of pages one at a time from DRAM via NoC0, with a barrier after each read (`noc_async_read_barrier`).

### Write Pattern
Sequential page writes. Each core writes its assigned contiguous range of pages one at a time to DRAM via NoC1, with a flush after each write (`noc_async_writes_flushed`) and a final barrier at the end (`noc_async_write_barrier`).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Cores are indexed in column-major order: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides the total page count across available cores, creating two core groups to handle the remainder. If the total tiles divide evenly, group 2 is empty.

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encapsulated tensor accessor parameters for src_buffer (bank mapping, page size metadata) |

**Writer Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encapsulated tensor accessor parameters for dst_buffer |

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process on this core (equals num_pages_per_core for this core group) |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for SQRT) |

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages this core should read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages this core should write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for SQRT (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for SQRT (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB c_0 | Read pages sequentially via TensorAccessor |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | copy_tile, sqrt_tile (SFPU), pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM | Write pages sequentially via TensorAccessor |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Iterates from `start_id` to `start_id + num_pages`, reading one page at a time into CB c_0. Uses `noc_async_read_barrier()` after each page to ensure completion before pushing to the CB.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page writer. Waits for compute to produce one page in CB c_2, reads it, writes to DRAM via NoC, flushes, and pops. Has `OUT_SHARDED` conditional path (not used for interleaved SQRT).

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes sqrt.h when SFPU_OP_SQRT_INCLUDE=1
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile-blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block; always 1 for SQRT

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack from c_0, pack to c_2, configures HW
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for 1 tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // block until reader has pushed 1 tile into c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from c_0 into DEST[0] via A2D datacopy

// For SQRT, SFPU_OP_CHAIN_0 expands to: sqrt_tile_init(); sqrt_tile(0);
// sqrt_tile_init() loads magic constants into SFPU programmable registers
// sqrt_tile(0) runs the SFPU sqrt algorithm on DEST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // signal that DEST registers are ready for pack stage

            tile_regs_wait();  // wait for pack stage to be ready to consume DEST

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish 1 tile in c_2 for writer to consume
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h`
(Wormhole B0 uses an identical implementation at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h`)

The hardware-specific wrappers that delegate to these are at:
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h`

The public API is at `tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h`.

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel
{
namespace sfpu
{

// Based on: Kokosinski, Z., Gepner, P., Moroz, L. et al.
// "Fast and accurate approximation algorithms for computing floating point square root."
// Numerical Algorithms (2024). https://doi.org/10.1007/s11075-024-01932-7
//
// This function computes sqrt(x) (or optionally 1/sqrt(x) when RECIPROCAL=true).
// The algorithm uses a fast inverse square root seed (similar to the "Quake III" trick)
// followed by Newton-Raphson refinement iterations.

template <bool APPROXIMATE = false, bool RECIPROCAL = false, bool FAST_APPROX = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x)
{
    // Step 1: Compute initial estimate of 1/sqrt(x) using integer bit manipulation.
    // Right-shift the float bits by 1 (approximately halving the exponent),
    // then subtract from a magic constant to get an initial 1/sqrt(x) estimate.
    sfpi::vInt i   = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);
    // y is now an initial approximation of 1/sqrt(x).

    if constexpr (APPROXIMATE)
    {
        // SQRT_10-bits algorithm: single Newton-Raphson refinement step.
        // Gives approximately 10 bits of precision.
        sfpi::vFloat c           = x * y;           // c = x * y ~ x / sqrt(x) = sqrt(x) initial estimate
        sfpi::vFloat negative_y  = -y;
        sfpi::vFloat infinity    = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::reinterpret<sfpi::vInt>(infinity);
        sfpi::vFloat t           = sfpi::vConstFloatPrgm1 + negative_y * c;  // refinement correction factor
        if constexpr (RECIPROCAL)
        {
            // For rsqrt: refine 1/sqrt(x) estimate directly.
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)  // if x != inf and x != 0
            {
                y = y * t;  // one Newton-Raphson step for 1/sqrt(x)
            }
            v_else  // special cases: sqrt(0)=0 handled as 1/inf, sqrt(inf)=inf handled as 1/0
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            // For sqrt: use c = x * (1/sqrt(x)) as initial sqrt estimate, refine it.
            y = c;  // y = sqrt(x) estimate
            v_if (sfpi::reinterpret<sfpi::vInt>(x) != infinity_bits)  // if x != inf
            {
                y = y * t;  // refine: y = c * t
            }
            v_endif;
            // If x == inf, c is already inf, which is correct.
        }
    }
    else
    {
        // SQRT_23-bits algorithm: two Newton-Raphson refinement steps.
        // Gives approximately 23 bits of precision (full float32 mantissa).
        sfpi::vFloat xy            = x * y;          // xy ~ sqrt(x) initial estimate
        sfpi::vFloat negative_y    = -y;
        sfpi::vFloat c             = negative_y * xy; // c = -y * x * y = -(x * y^2); measures error from 1
        sfpi::vFloat infinity      = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits   = sfpi::reinterpret<sfpi::vInt>(infinity);

        // First refinement: improve y (1/sqrt(x) estimate) using quadratic correction
        // y = y * (P1 + c * (P2 + c)) where P1, P2 are tuned polynomial coefficients
        y                          = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));

        // Recompute products with improved y
        xy                         = x * y;           // improved sqrt(x) estimate
        negative_y                 = -y;
        sfpi::vFloat one_minus_xyy = sfpi::vConst1 + (negative_y * xy);  // residual: 1 - x*y^2

        if constexpr (RECIPROCAL)
        {
            // For rsqrt: final Householder refinement of 1/sqrt(x)
            sfpi::vFloat half_y              = sfpi::addexp(y, -1);  // y/2 via exponent decrement
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)
            {
                y = one_minus_xyy * half_y + y;  // Householder step: y += (1-x*y^2)*(y/2)
            }
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            // For sqrt: final refinement of sqrt(x) = x * (1/sqrt(x))
            sfpi::vFloat half_xy = 0.5f * xy;  // (sqrt(x) estimate) / 2
            // Skip if x == inf to avoid inf - inf = nan; xy is already inf.
            v_if (sfpi::reinterpret<sfpi::vInt>(x) < infinity_bits)
            {
                y = one_minus_xyy * half_xy + xy;  // y = xy + (1 - x*y^2) * xy/2
            }
            v_endif;
        }
    }

    if constexpr (!FAST_APPROX)
    {
        // Handle negative inputs: sqrt(negative) = NaN
        v_if (x < 0.0f)
        {
            y = std::numeric_limits<float>::quiet_NaN();  // NaN for fp32, inf for bf16
        }
        v_endif;
    }

    return y;
}

// Iterates over tile faces (sub-tile rows of 16 elements processed by SFPU).
// ITERATIONS=8 means 8 faces are processed, covering a full 32x32 tile
// (each iteration processes one row of 16 elements; 8 iterations * 4 faces = 32 rows,
// but the SFPU processes 4 elements per SIMD lane with 16 lanes = 64 elements per face).
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_(const int iterations)
{
#pragma GCC unroll 8  // hint to fully unroll the face iteration loop
    for (int d = 0; d < iterations; d++)
    {
        // Read current element from DEST register, compute sqrt, write back
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = tmp;  // write back as fp32 when DEST is in fp32 mode
        }
        else
        {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0));  // convert to bf16 and write back
        }
        sfpi::dst_reg++;  // advance to next face (next 16-element row)
    }
}

// Entry point: dispatches to internal implementation (or legacy compat path)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat = false>
inline void _calculate_sqrt_(int iterations)
{
    if constexpr (legacy_compat)
    {
        return _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(iterations);
    }
    else
    {
        // For SQRT: RECIPROCAL=false, dispatches to _calculate_sqrt_internal_
        return _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>(iterations);
    }
}

// Initialization: loads magic constants into SFPU programmable registers.
// These constants are derived from the paper cited above and are critical
// for the accuracy of the initial 1/sqrt(x) seed and the refinement polynomial.
template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_sqrt_()
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            // 10-bit accuracy mode constants
            sfpi::vConstIntPrgm0   = 0x5f0b3892;  // magic constant for initial 1/sqrt(x) seed
            sfpi::vConstFloatPrgm1 = 1.89099014875f;  // Newton-Raphson refinement coefficient
        }
        else
        {
            // 23-bit accuracy mode constants
            sfpi::vConstIntPrgm0   = 0x5f1110a0;  // magic constant for initial 1/sqrt(x) seed
            sfpi::vConstFloatPrgm1 = 2.2825186f;   // first polynomial coefficient
            sfpi::vConstFloatPrgm2 = 2.2533049f;   // second polynomial coefficient
        }
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[0]` | Reads/writes elements from DEST register file; SFPU operates on 16 elements (one face) at a time |
| `sfpi::dst_reg++` | Advances the DEST register pointer to the next face |
| `sfpi::reinterpret<vInt>(vFloat)` | Bit-cast reinterpretation between float and integer vector types (no conversion) |
| `sfpi::reinterpret<vUInt>(vFloat)` | Bit-cast to unsigned integer for logical shift |
| `>> 1` | Logical right shift by 1 on integer vector; approximately halves the float exponent |
| `sfpi::vConstIntPrgm0` | Programmable integer constant register; loaded with magic seed for 1/sqrt(x) |
| `sfpi::vConstFloatPrgm1` | Programmable float constant register 1; loaded with refinement coefficient |
| `sfpi::vConstFloatPrgm2` | Programmable float constant register 2; loaded with second polynomial coefficient (23-bit mode only) |
| `sfpi::vConst1` | Hardware constant register holding 1.0f |
| `sfpi::s2vFloat16b(...)` | Converts a scalar to a bf16 vector broadcast |
| `sfpi::addexp(y, -1)` | Decrements the exponent of float y by 1, effectively computing y/2 |
| `float_to_fp16b(tmp, 0)` | Converts fp32 vector to bf16 format (truncation) |
| `v_if / v_else / v_endif` | SFPU conditional execution (predicated lanes); enables per-element branching |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` | Input: holds one face (16 elements) of the current tile from DEST. Output: result is written back here. |
| `vConstIntPrgm0` | Magic constant for the fast inverse square root seed (0x5f0b3892 in approx mode, 0x5f1110a0 in full precision) |
| `vConstFloatPrgm1` | Newton-Raphson coefficient (1.89099 in approx mode, 2.2825 in full precision) |
| `vConstFloatPrgm2` | Second polynomial coefficient (only used in 23-bit mode: 2.2533) |
| `vConst1` | Constant 1.0f, used in residual calculation `1 - x*y^2` |
| Local vFloat/vInt | Intermediate values: `y` (estimate), `xy`, `c`, `t`, `negative_y`, `one_minus_xyy`, `half_xy` |

#### SFPU Execution Flow

1. **Initialization** (`sqrt_tile_init()` -> `_init_sqrt_()`): Loads magic constants into SFPU programmable registers (`vConstIntPrgm0`, `vConstFloatPrgm1`, and optionally `vConstFloatPrgm2`). These constants depend on whether `APPROXIMATION_MODE` is true or false.

2. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for the reader to produce a tile, then `copy_tile(c_0, 0, 0)` to unpack the tile from CB c_0 into DEST register 0 via the A2D datacopy path (unpack engine).

3. **SFPU dispatch** (`sqrt_tile(0)` -> `calculate_sqrt()` -> `_calculate_sqrt_internal_()`): The SFPU iterates over 8 faces of the tile. For each face:
   a. Read 16 elements from `dst_reg[0]`.
   b. Compute initial 1/sqrt(x) estimate via integer bit manipulation: `y = magic_constant - (x_bits >> 1)`.
   c. In **approximate mode** (10-bit): one Newton-Raphson step refines the estimate.
   d. In **full precision mode** (23-bit): two refinement steps using polynomial coefficients, followed by a Householder-like correction.
   e. Handle special cases: x = 0, x = infinity, x < 0 (returns NaN).
   f. Write result back to `dst_reg[0]`, converting to bf16 if `fp32_dest_acc_en` is false.
   g. Advance to next face with `dst_reg++`.

4. **Pack**: After SFPU completes, `tile_regs_commit()` signals readiness, `tile_regs_wait()` waits for pack availability, and `pack_tile(0, c_2)` packs DEST[0] into CB c_2.

5. **Cleanup**: `cb_pop_front(c_0, 1)` frees the input tile, `tile_regs_release()` releases DEST. After all tiles in the block, `cb_push_back(c_2, per_core_block_dim)` publishes the output.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (set by program factory, though SFPU ops typically do not use the FPU fidelity setting).
- **Approximation mode**: `math_approx_mode` is determined by `get_op_approx_mode(UnaryOpType::SQRT)`. The current implementation returns `false` for all ops (the `switch` has only a `default: return false` case), so SQRT uses the **23-bit full precision** algorithm by default.
- **fp32_dest_acc_en**: Configurable per operation invocation; when true, DEST registers hold fp32 values and the result is not truncated to bf16.
- **FAST_APPROX template parameter**: Defaults to `false` in the non-parameterized `sqrt_tile(idst)` call; when `false`, the kernel includes the negative-input check that returns NaN.
- **Compile-time defines**: `SFPU_OP_SQRT_INCLUDE=1` is set to include `sqrt.h`. `SFPU_OP_CHAIN_0` is defined as `sqrt_tile_init(); sqrt_tile(0);`.

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `_calculate_sqrt_body_` are **identical** in this codebase. Both architectures use the same SFPI-based software algorithm. The only minor difference is in the Wormhole B0 version: when `FAST_APPROX` is false, the NaN comment notes "returns nan for fp32 and inf for bf16" while the Blackhole version simply says the same. The actual code is byte-for-byte identical.

Both architectures share the same `_init_sqrt_` magic constants and the same programmable register interface (`vConstIntPrgm0`, `vConstFloatPrgm1`, `vConstFloatPrgm2`).

The hardware wrapper files (`tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h`) are thin templates that simply forward to `_calculate_sqrt_` and `_init_sqrt_`.

## Implementation Notes

1. **Parameterized vs non-parameterized invocation**: SQRT is listed as a parameterized op type in `is_parametrized_type()`. When called with a parameter, the init/func strings include a template parameter for `FAST_APPROX` (e.g., `sqrt_tile<{param0}>({idst})`). When called without parameters, the non-parameterized path uses `sqrt_tile({idst})` which defaults `FAST_APPROX=false`.

2. **No temporary CB needed**: Unlike HARDSHRINK, CBRT, or LOGIT, SQRT does not require the temporary circular buffer c_1. The computation is done entirely in DEST registers.

3. **Algorithm reference**: The implementation is based on Kokosinski et al. (2024), which extends the classic fast inverse square root technique with tuned polynomial refinement coefficients for higher accuracy.

4. **Special value handling**: The algorithm explicitly handles x = 0 (returns 0 via the infinity subtraction trick), x = infinity (returns infinity by skipping refinement), and x < 0 (returns NaN unless FAST_APPROX is true).

5. **bf16 truncation**: When `fp32_dest_acc_en` is false, the SFPU result is truncated from fp32 to bf16 via `float_to_fp16b()` before writing back to DEST. This truncation happens per-face, not per-tile.

6. **Runtime args for scalars unused**: The compute kernel receives `packed_scalar1` and `packed_scalar2` as runtime args (both 0 for SQRT), but these are not consumed by the sqrt kernel. They exist because the same `eltwise_sfpu.cpp` kernel is shared across all unary SFPU ops, some of which need scalar parameters.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations like sqrt? What kernels does it use, how are circular buffers configured, and how is work distributed across cores?"
   **Reason**: To understand the overall program factory architecture and identify all kernel files.
   **Key Findings**: Confirmed the three-kernel pattern (reader/compute/writer), double-buffered CBs with 2 tiles capacity, `split_work_to_cores` for distribution, and the `SFPU_OP_CHAIN_0` macro mechanism.

2. **Query**: "How does the SFPU sqrt operation work in tt-metal? What is the LLK compute kernel path and the underlying SFPU kernel function for sqrt?"
   **Reason**: To locate the SFPU kernel files and understand the call chain from `sqrt_tile()` to `_calculate_sqrt_`.
   **Key Findings**: Identified the LLK file paths for both Blackhole and Wormhole, confirmed the `calculate_sqrt` -> `_calculate_sqrt_` -> `_calculate_sqrt_internal_` -> `_calculate_sqrt_body_` call chain, and the `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN` macro dispatch.

### Confluence References

Not consulted. The SFPU kernel for SQRT uses standard SFPI vector intrinsics (reinterpret casts, arithmetic, conditional execution) that are well-documented in the source code and DeepWiki. No low-level SFPU ISA instruction details were needed beyond what the source code provides.

### Glean References

Not consulted. The implementation is fully available in open-source tt_llk files.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To determine the compile-time defines and compute kernel path for SQRT.
   **Key Information**: SQRT maps to `SFPU_OP_SQRT_INCLUDE`, uses default `eltwise_sfpu.cpp` kernel, `get_op_approx_mode` returns `false` (full precision mode), and the SFPU_OP_CHAIN_0 define is `sqrt_tile_init(); sqrt_tile(0);`.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h`
   **Reason**: To understand the public API layer between compute kernel and SFPU implementation.
   **Key Information**: `sqrt_tile_init()` calls `SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX)` and `sqrt_tile(idst)` calls `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN(calculate_sqrt, APPROX, 8, DST_ACCUM_MODE, FAST_APPROX, idst, RC)`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: To understand macro expansion for SFPU dispatch.
   **Key Information**: `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_sqrt<APPROX, 8, DST_ACCUM_MODE, FAST_APPROX>, idst, VectorMode::RC)`.
