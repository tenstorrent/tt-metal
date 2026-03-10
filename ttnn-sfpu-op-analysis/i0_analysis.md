# I0 (Modified Bessel Function of the First Kind, Order 0) Implementation Analysis

## Overview

The I0 operation computes the element-wise modified Bessel function of the first kind, order 0, on each element of the input tensor. Mathematically, I0(x) is defined as the integral `(1/pi) * integral(0, pi, exp(x*cos(t)) dt)`. The implementation uses a degree-10 polynomial approximation in `x^2` (Horner's method) to evaluate I0.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

This operation uses the shared `UnaryProgramFactory` -- the same program factory used by all standard unary SFPU operations. The I0-specific behavior is injected via preprocessor defines (`SFPU_OP_I0_INCLUDE`, `SFPU_OP_CHAIN_0`) that select the `i0_tile_init()` and `i0_tile(idst)` functions at compile time.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for I0) |

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (any rank) | Same as input |
| **Dimension convention** | N/A (arbitrary) | Same as input |
| **Tensor layout** | TILE (32x32) | TILE (32x32) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16 / FLOAT32 | BFLOAT16 / FLOAT32 |

### Layout Transformations

No layout transformations are performed. Input and output must both be in TILE layout. Data format conversion between input and output types is handled by the unpacker/packer hardware.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (interleaved) | CB c_0 | `cb_reserve_back(c_0, 1)`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, per_core_block_dim)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM (interleaved) | `cb_wait_front(c_2, 1)`, `cb_pop_front(c_2, 1)` |

The reader fetches one tile at a time from DRAM into CB c_0. The compute kernel copies the tile from CB c_0 into the DEST register, applies the I0 SFPU operation, and packs the result into CB c_2. The writer drains one tile at a time from CB c_2 back to DRAM.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is NOT allocated for I0 -- it is only created for HARDSHRINK, CBRT, and LOGIT operations.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 tiles and block size = 1 tile, enabling **double-buffering**. This allows the reader to fill one tile slot while the compute kernel processes another, and similarly the compute kernel can fill one output slot while the writer drains another.

## Index Calculations

The reader and writer use `TensorAccessor` with interleaved addressing. The mapping is straightforward: each tile is identified by a linear page index. The reader starts at `start_id` (assigned per core) and reads `num_pages_per_core` consecutive tiles. The writer uses the same linear indexing scheme for output.

## Memory Access Patterns

### Read Pattern
Sequential tile reads from DRAM. Each core reads a contiguous range of tile pages starting from `start_id`. One page is read per iteration using `noc_async_read_page`, followed by a barrier.

### Write Pattern
Sequential tile writes to DRAM. Each core writes a contiguous range of tile pages. One page is written per iteration using `noc_async_write_page`, with a final barrier after all writes.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages/num_cores)` tiles, group 2 gets `floor(num_pages/num_cores)` tiles |

Cores are enumerated column-major: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides the total tile count across available cores, creating two core groups when tiles do not divide evenly. Group 2 may have fewer tiles (or be empty).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for source buffer (address mode, bank info) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for I0) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of tile pages this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of tile pages this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for I0 (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for I0 (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM | CB c_0 | Read tiles via `noc_async_read_page` |
| compute | TRISC (math RISC) | N/A | CB c_0 | CB c_2 | `copy_tile` + `i0_tile` SFPU op + `pack_tile` |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM | Write tiles via `noc_async_write_page` |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Creates a `TensorAccessor` from compile-time args and iterates from `start_id` to `start_id + num_pages`, reading one page per iteration into CB c_0. Uses `noc_async_read_barrier()` per page.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page writer. Waits for one page in CB c_2, reads the L1 pointer, issues an async write to DRAM, flushes, and pops. Final `noc_async_write_barrier()` after all pages.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes i0.h when SFPU_OP_I0_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (1 for I0)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // Initialize SFPU pipeline: configure unpacker for c_0 input, packer for c_2 output
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // Reserve output space for one block (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // Acquire exclusive access to DEST registers for math operations

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // Wait for reader to produce 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // Unpack tile 0 from CB c_0 into DEST register 0

// For I0, SFPU_OP_CHAIN_0 expands to: i0_tile_init(); i0_tile(0);
// i0_tile_init() calls llk_math_eltwise_unary_sfpu_init<SfpuType::i0, APPROX>() to configure the SFPU pipeline
// i0_tile(0) dispatches calculate_i0<APPROX>() on all 4 faces of the tile in DEST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // Signal that DEST registers are ready for pack stage

            tile_regs_wait();  // Wait for pack stage to be ready to consume DEST

            pack_tile(0, tt::CBIndex::c_2);  // Pack DEST[0] result into CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // Free the consumed input tile from CB c_0

            tile_regs_release();  // Release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // Publish the output block to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
- Wormhole: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h`
- Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h`

Both files are identical.

#### Annotated SFPU Kernel Source
```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;  // SFPI: the SFPU Programming Interface providing vFloat, dst_reg, etc.

namespace ckernel {
namespace sfpu {

// POLYVAL10: Evaluates a degree-10 polynomial using Horner's method.
// Given coefficients c10..c0 and variable t4, computes:
//   c0 + c1*t4 + c2*t4^2 + ... + c10*t4^10
// Horner's form is numerically stable and minimizes multiplications:
//   c0 + t4*(c1 + t4*(c2 + ... + t4*(c9 + c10*t4)...))
#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)

// calculate_i0: Computes I0(x) for one face (8 rows) of a tile.
// Template parameters:
//   APPROXIMATION_MODE: not used in this implementation (no alternate path)
//   ITERATIONS: number of rows to process per face (default 8 = 16 elements per row x 8 rows = 128 elements per face)
// The SFPU processes elements in SIMD fashion: each vFloat holds one row of 16 elements.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
#pragma GCC unroll 0  // Prevent compiler from unrolling -- saves instruction memory

    for (int d = 0; d < ITERATIONS; d++) {  // Iterate over 8 rows within the current face
        vFloat result = 0.0f;               // Initialize result register to 0
        vFloat input = dst_reg[0];          // Load current row from DEST register 0 (16 elements SIMD)
        vFloat x = input * input;           // Compute x^2 -- I0's Taylor series is in terms of (x/2)^2

        // Evaluate the polynomial approximation of I0(x):
        // I0(x) ~= 1 + P10(x^2) where P10 is a degree-10 polynomial in x^2
        // The coefficients approximate the Taylor series: I0(x) = sum_{k=0}^{inf} ((x/2)^2)^k / (k!)^2
        // Truncated and fitted to 10 terms for the SFPU:
        //   coef0  = 0.25       ~= 1/(1!)^2 * (1/4)  [absorbed scaling]
        //   coef1  = 0.015625   ~= 1/(2!)^2 * (1/16)
        //   coef2  = 0.000434.. ~= 1/(3!)^2 * ...
        //   ... and so on for higher-order terms
        result = 1.0f + POLYVAL10(
                            1.50E-22f,           // coef10: highest-order coefficient
                            7.24E-20f,           // coef9
                            2.90E-17f,           // coef8
                            9.39E-15f,           // coef7
                            2.40E-12f,           // coef6
                            4.71E-10f,           // coef5
                            6.78E-08f,           // coef4
                            0.000006781684028f,  // coef3
                            0.0004340277778f,    // coef2
                            0.015625f,           // coef1
                            0.25f,               // coef0
                            x);                  // variable: x^2

        dst_reg[0] = result;  // Write result back to current row in DEST register 0
        dst_reg++;            // Advance to the next row within the face (pointer increment)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `dst_reg[0]` (read) | Loads a SIMD row (16 elements as `vFloat`) from the DEST register file at the current row offset |
| `dst_reg[0]` (write) | Stores a SIMD row back to the DEST register file at the current row offset |
| `dst_reg++` | Advances the DEST register row pointer to the next row within the face |
| `vFloat * vFloat` | SFPU multiply -- element-wise multiplication of two SIMD vectors |
| `vFloat + vFloat` | SFPU add -- element-wise addition of two SIMD vectors |
| Scalar-to-vFloat promotion | Implicit broadcast of float literal to all 16 SIMD lanes |

The I0 kernel uses only basic SFPU arithmetic (multiply, add) via the SFPI programming interface. It does not use LUT-based approximations, special transcendental instructions, or condition codes.

#### SFPU Register Usage

- **DEST register `dst_reg[0]`**: Used as both input source and output destination. Each iteration reads one row (16 elements), computes I0, and writes back.
- **`dst_reg++`**: Row pointer auto-increment to walk through the 8 rows of a face.
- **`vFloat` temporaries (`result`, `input`, `x`)**: Mapped to SFPU local registers (LREGs). The compiler allocates these to the 4 available SFPU local registers.

#### SFPU Execution Flow

1. The `_llk_math_eltwise_unary_sfpu_params_<APPROX>` dispatcher is invoked with `calculate_i0<APPROX>` as the function pointer, `dst_index=0`, and `vector_mode=RC`.
2. The dispatcher sets the DEST write address to the target tile, configures address modes, and stalls until the SFPU is ready (`TTI_STALLWAIT`).
3. In RC mode, the dispatcher calls `calculate_i0()` once per face, for all 4 faces of the 32x32 tile. Between faces, `TTI_SETRWC` instructions advance the DEST row counter by 16 rows (2 increments of 8).
4. Within each `calculate_i0()` call (one face):
   - The loop iterates 8 times (ITERATIONS=8), one per row of the 16x16 face.
   - Each iteration: load row from DEST -> square it -> evaluate degree-10 polynomial via Horner's method -> add 1.0 -> store back to DEST -> advance row pointer.
5. After all 4 faces are processed, `math::clear_dst_reg_addr()` resets the pointer and `TTI_STALLWAIT` ensures SFPU completion before the pack stage proceeds.

#### SFPU Configuration

- **Math fidelity**: `HiFi4` (set in ComputeConfig, highest fidelity)
- **Math approx mode**: `false` (I0 returns false from `get_op_approx_mode`; the `APPROXIMATION_MODE` template parameter is false)
- **fp32_dest_acc_en**: Configurable per call (from `args.fp32_dest_acc_en`)
- **unpack_to_dest_mode**: Default (or `UnpackToDestFp32` if `preserve_fp32_precision` is set)
- **Preprocessor defines**: `SFPU_OP_I0_INCLUDE=1` causes `ckernel_sfpu_i0.h` to be included; `SFPU_OP_CHAIN_0` expands to `i0_tile_init(); i0_tile(0);`

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `calculate_i0` are **identical** -- the same source file with the same polynomial coefficients and the same SFPI code. Since this kernel uses only basic SFPU arithmetic (multiply, add) without hardware-specific instructions, it is fully portable across both architectures.

## Implementation Notes

1. **Polynomial approximation quality**: The degree-10 polynomial in x^2 effectively gives a degree-20 approximation of I0(x). The coefficients closely follow the Taylor series pattern for I0, which converges as `sum_{k=0}^{inf} ((x/2)^2)^k / (k!)^2`. This provides good accuracy for moderate values of x but may lose precision for very large |x| values where I0 grows exponentially.

2. **No approximation mode branching**: Unlike many SFPU operations (e.g., exp, log) that have both accurate and approximate implementations, `calculate_i0` has a single code path regardless of the `APPROXIMATION_MODE` template parameter. The parameter is accepted but unused.

3. **No runtime parameters**: I0 does not use `packed_scalar1` or `packed_scalar2` -- they are both 0. The operation is purely element-wise with no configurable parameters.

4. **Pure arithmetic implementation**: The I0 SFPU kernel achieves its computation entirely through multiply-add chains (Horner's method). It does not use LUTs, special function hardware, or transcendental instruction support. This makes it straightforward but potentially slower than operations that can leverage hardware LUT acceleration.

5. **Double-buffering**: Both input and output CBs have 2-tile capacity with 1-tile block size, allowing overlap between reader-compute and compute-writer stages.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary program factory implemented for SFPU operations?"
   **Reason**: Needed to understand the overall structure of `UnaryProgramFactory` before reading source code.
   **Key Findings**: Confirmed the factory sets up reader/writer/compute kernels, uses `split_work_to_cores` for distribution, and configures CBs with double-buffering. The compute kernel path is determined by `get_compute_kernel_path()`.

2. **Query**: "How is the i0 (modified Bessel function) SFPU operation implemented in the LLK/ckernel layer?"
   **Reason**: Needed to understand if there was a dedicated LLK-level implementation beyond the ckernel_sfpu_i0.h file.
   **Key Findings**: DeepWiki confirmed I0 is not a dedicated SFPU hardware instruction -- it is implemented as a software polynomial approximation using basic SFPU arithmetic primitives.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/i0.h`
   **Reason**: API-level wrapper that defines `i0_tile()` and `i0_tile_init()`.
   **Key Information**: `i0_tile()` uses `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_i0, RC, APPROX, idst)` -- dispatches to `calculate_i0` in RC vector mode with no extra parameters.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understanding how `_llk_math_eltwise_unary_sfpu_params_` dispatches the SFPU function across tile faces.
   **Key Information**: In RC mode, the function is called 4 times (once per face), with `TTI_SETRWC` advancing the DEST pointer between faces. Each call processes 8 rows of 16 elements.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Mapping from `UnaryOpType::I0` to kernel paths and defines.
   **Key Information**: I0 maps to macro `SFPU_OP_I0_INCLUDE`, uses default compute kernel `eltwise_sfpu.cpp`, init/func pair is `i0_tile_init()` / `i0_tile(idst)`, and `get_op_approx_mode` returns false.
