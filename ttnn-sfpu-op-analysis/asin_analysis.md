# ASIN (Arcsine) Implementation Analysis

## Overview
The ASIN operation computes the element-wise arcsine (inverse sine) of each element in the input tensor. The mathematical domain is [-1, 1]; inputs outside this range produce NaN. The implementation uses a Maclaurin series polynomial approximation up to the x^11 term, executed on the SFPU (Special Function Processing Unit).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

This operation uses the shared `UnaryProgramFactory` with the generic `eltwise_sfpu.cpp` compute kernel. The specific SFPU behavior is injected at compile time via preprocessor defines (`SFPU_OP_CHAIN_0`) that expand to `asin_tile_init(); asin_tile(0);`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `input.buffer()->num_pages()` (total tiles in the tensor) |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block. For the standard factory, `per_core_block_dim` is always 1, so effectively a single loop of `per_core_block_cnt` tiles. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank, flattened to pages) |
| **Dimension convention** | N/A (page-based iteration) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16 (typical), FLOAT32, or other supported types |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or as specified by output dtype) |

### Layout Transformations
No layout transformations are performed. Input and output share the same layout. The operation is purely element-wise.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (input buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU asin, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)` |
| 3 | Writer | CB c_2 | DRAM/L1 (output buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

**Detailed flow for one tile:**
1. Reader reads one page from DRAM via NoC0 into CB c_0.
2. Compute acquires tile registers, waits for a tile in c_0, copies it to DEST register 0 via `copy_tile`.
3. Compute executes `asin_tile_init()` (once, at program start via `init_sfpu`) and `asin_tile(0)` which dispatches `calculate_asin<true>` across all 4 faces of the tile.
4. Compute commits tile registers, waits for pack, packs DEST[0] into CB c_2, pops c_0, releases tile registers.
5. Writer waits for a tile in c_2, writes it to DRAM via NoC1, pops c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0_cb | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output_cb | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes:**
- CB c_1 (tmp0) is NOT allocated for ASIN. It is only created for HARDSHRINK, CBRT, or LOGIT operations.
- Both input and output CBs are double-buffered (capacity = 2 * page_size), allowing overlap between producer and consumer.

## Pipeline Pattern Summary
- **CB c_0**: Double-buffered. Reader can produce the next tile while Compute processes the current tile.
- **CB c_2**: Double-buffered. Compute can produce the next result while Writer drains the current result.
- This enables a 3-stage pipeline where Reader, Compute, and Writer can operate concurrently on different tiles.

## Index Calculations
Tile indexing is linear. Each core receives a `start_id` and `num_pages` count. The reader and writer iterate sequentially from `start_id` to `start_id + num_pages`. The `TensorAccessor` maps linear page indices to physical NoC addresses based on the buffer's interleaved bank distribution.

## Memory Access Patterns

### Read Pattern
Sequential page reads. The reader iterates `i = start_id` to `start_id + num_pages - 1`, reading one page per iteration via `noc_async_read_page`. Each read is followed by `noc_async_read_barrier()` ensuring completion before pushing to CB. This is a simple sequential scan with no striding.

### Write Pattern
Sequential page writes. The writer iterates in the same linear order, writing one page per iteration via `noc_async_write_page`. Writes are flushed per page (`noc_async_writes_flushed()`), with a final `noc_async_write_barrier()` after the loop.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major linearization) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` based on total pages |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

**Core linearization**: `core = {i / num_cores_y, i % num_cores_y}` -- column-major order. Two separate compute kernels are created if the core groups differ in tile count, each with the appropriate `per_core_block_cnt` compile-time argument.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, and bank mapping for NoC reads |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, and bank mapping for NoC writes |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tiles (blocks) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard unary factory) |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | First page index for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | First page index for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ASIN (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ASIN (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0 | Sequential page reads via TensorAccessor |
| compute | MATH (RISCV_2) | N/A | CB c_0 | CB c_2 | copy_tile + SFPU asin (Maclaurin series) |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Sequential page writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple linear scan. Creates a `TensorAccessor` from compile-time args and base address, then loops `start_id..end_id` reading one page per iteration. Supports `BACKWARDS` mode (unused for ASIN). Each page read blocks on `noc_async_read_barrier()` before pushing to CB.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Mirrors the reader in reverse. Waits on CB c_2, reads the L1 pointer, writes to DRAM via NoC1. Supports `OUT_SHARDED` mode (unused for interleaved ASIN). Final `noc_async_write_barrier()` ensures all writes complete.

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
#include "api/compute/eltwise_unary/trigonometry.h"       // provides asin_tile_init() and asin_tile()
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // total number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block; always 1 for standard unary

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes SFPU pipeline: configures unpack from c_0 and pack to c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve output space for one block (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // block until reader has produced 1 tile in c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from c_0 into DEST register 0

// For ASIN, SFPU_OP_CHAIN_0 expands to: asin_tile_init(); asin_tile(0);
// asin_tile_init() calls llk_math_eltwise_unary_sfpu_init<SfpuType::asin, true>() -- no custom init callback
// asin_tile(0) dispatches calculate_asin<true> across all 4 faces of the tile in DEST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // signal that DEST registers are ready for packing

            tile_regs_wait();  // wait for pack pipeline to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish the block of output tiles to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h`
(Blackhole variant: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h`)

#### Annotated SFPU Kernel Source
```cpp
// From ckernel_sfpu_trigonometry.h -- only the asin-relevant functions are shown below.
// The full file contains sin, cos, tan, atan, acos, cosh, sinh as well.

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel::sfpu {

static const float PI = 3.1415927f;
static const float PI_2 = 1.5707964f;       // pi/2, used by acos (not asin directly)
static const float PI_4 = 0.7853982f;
static const float FRAC_1_PI = 0.31830987f;

// Maclaurin series approximation for arcsin(x), valid for x in [-1, 1].
// Uses 6 terms: x + (1/6)x^3 + (3/40)x^5 + (5/112)x^7 + (35/1152)x^9 + (63/2816)x^11
// The APPROXIMATION_MODE template parameter is accepted but unused -- only one implementation exists.
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_asine_maclaurin_series(vFloat val) {
    // input for [-1:1]
    // Mclauren series
    // arcsin(x) = x + [(1/2)*x^3/3] + [(1*3)/(2*4)*x^5/5] + [(1*3*5)/(2*4*6)*x^7/7] + ...
    // arcsin(x) ~ x + (1/6)*x^3 + (3/40)*x^5 + (5/112)*x^7 + (35/1152)*x^9 + (63/2816)*x^11

    vFloat tmp = val;                        // tmp accumulates x^(2k+1) powers
    vFloat val_square = val * val;           // x^2, reused for successive power computation
    // x  (first term)
    vFloat output = tmp;                     // output = x
    // (1/6) * x^3
    tmp = tmp * val_square;                  // tmp = x^3
    output += 0.166666666 * tmp;             // output += (1/6)*x^3
    // (3/40) * x^5
    tmp = tmp * val_square;                  // tmp = x^5
    output += 0.075 * tmp;                   // output += (3/40)*x^5

    // (5/112) * x^7
    tmp = tmp * val_square;                  // tmp = x^7
    output += 0.044642857 * tmp;             // output += (5/112)*x^7

    // (35/1152) * x^9
    tmp = tmp * val_square;                  // tmp = x^9
    output += 0.03038194 * tmp;              // output += (35/1152)*x^9

    // (63/2816) * x^11
    tmp = tmp * val_square;                  // tmp = x^11
    output += 0.02237216 * tmp;              // output += (63/2816)*x^11

    // Write out output
    return output;                           // returns arcsin(x) approximation
}

// Main SFPU kernel entry point for asin.
// ITERATIONS defaults to 8, meaning 8 rows of a 16x16 face are processed per call.
// The _llk_math_eltwise_unary_sfpu_params_ dispatcher calls this 4 times (once per face)
// in VectorMode::RC mode, advancing the DEST write pointer between faces.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asin() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {   // iterate over 8 rows of the current face
        vFloat v = dst_reg[0];               // load the current row's element from DEST register

        // Domain check: asin is only defined for [-1, 1]
        v_if(v < vConstNeg1 || v > vConst1) {
            dst_reg[0] = std::numeric_limits<float>::quiet_NaN();  // out-of-range -> NaN
        }
        v_else {
            dst_reg[0] = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);  // compute asin via polynomial
        }
        v_endif;

        dst_reg++;                           // advance to next row in the face
    }
}

}  // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `dst_reg[0]` (read) | Loads a vector of elements from the current DEST register row. Each SFPU lane processes one element. |
| `dst_reg[0] = ...` (write) | Writes a vector result back to the current DEST register row. |
| `dst_reg++` | Advances the DEST register pointer to the next row within the face. |
| `vFloat` arithmetic (`*`, `+`) | SFPU vector multiply and add operations on float32 values. Each operates on all SFPU lanes in parallel. |
| `v_if` / `v_else` / `v_endif` | SFPU predicated execution: sets per-lane condition codes and conditionally applies operations. |
| `vConst1`, `vConstNeg1` | Built-in SFPU constants for 1.0f and -1.0f. |
| `std::numeric_limits<float>::quiet_NaN()` | Produces a NaN constant loaded into SFPU lanes. |

#### SFPU Register Usage

- **DEST registers**: The tile is loaded into DEST by `copy_tile`. The SFPU reads from and writes to DEST[0] (current row), advancing via `dst_reg++`. A 32x32 tile occupies 4 faces of 16x16 elements each. Each face has 16 rows, but ITERATIONS=8 processes 8 rows per call (the dispatcher handles the full face by calling the function and advancing the DEST write pointer).
- **SFPU vector registers (`vFloat`)**: Temporary variables `tmp`, `val_square`, `output`, and `v` are held in SFPU vector registers. These are allocated by the compiler from the SFPU's register file.
- **Condition code registers**: Used implicitly by `v_if`/`v_else`/`v_endif` for predicated execution of the NaN vs. polynomial paths.

#### SFPU Execution Flow

1. **Tile acquisition**: `tile_regs_acquire()` grants exclusive access to the DEST register file.
2. **Unpack**: `copy_tile(c_0, 0, 0)` unpacks the tile from CB c_0 position 0 into DEST register 0. This converts from the wire format (e.g., BFLOAT16) to the DEST format (FP32 in DEST registers).
3. **SFPU init**: `asin_tile_init()` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::asin, true>()`. This calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::asin>()` which configures the SFPU pipeline. No custom init callback is needed for asin.
4. **SFPU dispatch**: `asin_tile(0)` expands to `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_asin, RC, true, 0)`, which calls `_llk_math_eltwise_unary_sfpu_params_<true>(ckernel::sfpu::calculate_asin<true>, 0, (int)VectorMode::RC)`.
5. **Face iteration**: The `_llk_math_eltwise_unary_sfpu_params_` function in VectorMode::RC mode:
   - Sets the DEST write address to the target tile index.
   - Stalls until SFPU is ready (`TTI_STALLWAIT`).
   - Iterates over all 4 faces. For each face:
     - Calls `calculate_asin<true>()` which processes 8 rows (ITERATIONS=8).
     - Advances the DEST pointer by 16 rows (`TTI_SETRWC` x2, each advancing by 8).
6. **Per-row computation** (inside `calculate_asin`): For each of the 8 rows:
   - Reads `dst_reg[0]` -- the vector of elements at the current row.
   - Checks if any lane value is outside [-1, 1]. For those lanes, writes NaN.
   - For in-range lanes, computes the Maclaurin series polynomial (6 terms, up to x^11).
   - Writes the result back to `dst_reg[0]`.
   - Increments `dst_reg` to the next row.
7. **Pack**: After all 4 faces, `pack_tile(0, c_2)` converts the result from DEST back to the output wire format and writes it into CB c_2.
8. **Cleanup**: `cb_pop_front(c_0, 1)` frees the input tile. `tile_regs_release()` releases DEST.

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (set in `ComputeConfig`).
- **Math approx mode**: `true` for ASIN. The `get_op_approx_mode` function returns `false` by default (ASIN is not in any special case), but the `APPROXIMATION_MODE` template parameter is hardcoded to `true` in the macro expansion (`SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_asin, RC, true, idst)`). Note: The `APPROXIMATION_MODE` parameter has no effect on the asin implementation since `sfpu_asine_maclaurin_series` has only one implementation (not specialized for `true` vs `false`).
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` from the operation attributes. When enabled, DEST registers hold full FP32 precision.
- **Unpack to dest mode**: Default unless `preserve_fp32_precision` is set, in which case `UnpackToDestFp32` is used for CB c_0.
- **Include define**: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default fallback, since ASIN is not listed in `get_macro_definition`). However, `trigonometry.h` is directly included by `eltwise_sfpu.cpp`.

#### Hardware Compatibility Notes
- The `calculate_asin` and `sfpu_asine_maclaurin_series` functions are **identical** between Wormhole B0 and Blackhole implementations. Both files contain the same Maclaurin series with the same coefficients.
- The `_llk_math_eltwise_unary_sfpu_params_` dispatch function exists in both architectures' LLK libraries with the same face iteration pattern.
- The `APPROXIMATION_MODE` template parameter is accepted but has no effect on the asin kernel -- there is no specialized high-precision variant. This differs from operations like `tan` or `sinpi` which have separate `<true>` and `<false>` template specializations.

## Implementation Notes

1. **Polynomial accuracy**: The 6-term Maclaurin series (up to x^11) provides reasonable accuracy near x=0 but accuracy degrades as |x| approaches 1, where the true asin function has a vertical slope. For |x| close to 1, the approximation error increases significantly. A Chebyshev or minimax polynomial, or a range-reduction technique (as used for atan), would provide better accuracy across the full domain.

2. **No approximation mode differentiation**: Unlike `tan`, `sinpi`, or `atan` which have separate `APPROXIMATION_MODE=true` and `APPROXIMATION_MODE=false` implementations, `sfpu_asine_maclaurin_series` has only one implementation. The template parameter is accepted for API consistency but ignored.

3. **NaN handling**: The domain check `v < vConstNeg1 || v > vConst1` uses predicated execution (`v_if`/`v_else`/`v_endif`) to write NaN for out-of-range inputs and the polynomial result for in-range inputs, all within the same SFPU instruction stream.

4. **No scalar parameters**: ASIN takes no parameters beyond the input tensor. The `packed_scalar1` and `packed_scalar2` runtime args for the compute kernel are always 0.

5. **Shared program factory**: ASIN shares the `UnaryProgramFactory` with dozens of other unary operations. The only differentiation is through:
   - The `SFPU_OP_CHAIN_0` define (set to `asin_tile_init(); asin_tile(0);`)
   - The `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` define (for conditional includes)
   - The compute kernel path (default `eltwise_sfpu.cpp`)

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations? What is the structure of unary_program_factory.cpp and how does it set up reader, compute, and writer kernels for elementwise unary operations like asin?"
   **Reason**: Needed to understand the overall program factory structure before reading the source code.
   **Key Findings**: Confirmed that UnaryProgramFactory uses a shared eltwise_sfpu.cpp compute kernel with operation-specific behavior injected via `SFPU_OP_CHAIN_0` defines. Reader/writer use interleaved start_id kernels. Two core groups handle remainder tile distribution.

2. **Query**: "How does the SFPU asin (arcsine) operation work in tt-metal? What compute kernel and SFPU kernel files implement asin?"
   **Reason**: Needed to locate the specific SFPU kernel files and understand the mathematical implementation.
   **Key Findings**: Confirmed that asin uses `ckernel_sfpu_trigonometry.h` with a Maclaurin series approximation up to x^11. Identified the compute kernel as `eltwise_sfpu.cpp` and the SFPU function as `calculate_asin`.

### Documentation References

1. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand how `_llk_math_eltwise_unary_sfpu_params_` dispatches the SFPU function across tile faces.
   **Key Information**: In VectorMode::RC, the function calls the SFPU kernel 4 times (once per face), advancing the DEST pointer by 16 rows between faces using `TTI_SETRWC`. Each call processes ITERATIONS=8 rows.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h`
   **Reason**: Needed to see the compute API layer that bridges `asin_tile()` calls to the underlying LLK/SFPU functions.
   **Key Information**: `asin_tile(idst)` expands via `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_asin, RC, true, idst)` -- always uses APPROXIMATION_MODE=true. `asin_tile_init()` uses `SFPU_UNARY_KERNEL_INIT(asin, true)` with no custom init callback.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how the ASIN operation type maps to defines, kernel paths, and approximation mode.
   **Key Information**: ASIN maps to init=`asin_tile_init()` and func=`asin_tile(0)`. Falls through to default compute kernel path `eltwise_sfpu.cpp`. Falls through to default macro define `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`. `get_op_approx_mode` returns `false` for ASIN (default case).
