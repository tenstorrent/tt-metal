# ERFINV Implementation Analysis

## Overview
The ERFINV operation computes the element-wise inverse error function (`erfinv(x)`) on each element of an input tensor. For a given input `x` in the range `(-1, 1)`, it returns the value `y` such that `erf(y) = x`. The implementation uses a Winitzki (2008) approximation that combines logarithm and square root sub-computations on the SFPU.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` (shared by all standard unary SFPU operations; ERFINV hits the `default` path in `get_compute_kernel_path`, selecting `eltwise_sfpu.cpp`).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total tiles in the input tensor |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt` iterations), inner loop over tiles within each block (`per_core_block_dim` = 1 tile). Effectively one tile per iteration. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (flattened to total tiles) |
| **Dimension convention** | N/A (elementwise, shape-agnostic) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR; CB page size adjusts accordingly) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 (typical), FLOAT32, INT32, UINT32 supported |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (or as specified by output dtype) |

### Layout Transformations
None. The operation is a pure elementwise unary; no tilize/untilize or reshard occurs within the program factory.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_addr) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU op chain (erfinv), `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)` |
| 3 | Writer | CB c_2 | DRAM (dst_addr) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The reader fetches one tile at a time from DRAM into CB c_0. The compute kernel unpacks the tile from c_0 into DEST registers, applies the SFPU erfinv operation, and packs the result into CB c_2. The writer drains one tile at a time from c_2 back to DRAM.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

ERFINV does not require the temporary CB c_1 (that buffer is only allocated for HARDSHRINK, CBRT, and LOGIT).

## Pipeline Pattern Summary
Both c_0 and c_2 are double-buffered (capacity = 2x block size). This allows the reader to write the next tile into c_0 while the compute kernel processes the current tile, and the compute kernel to write into c_2 while the writer drains the previous result. This enables a 3-stage pipelined overlap of reader, compute, and writer.

## Index Calculations
Tiles are indexed linearly by a flat page ID. The program factory computes a `start_id` for each core (the cumulative number of tiles assigned to preceding cores) and a `num_pages` count. The reader and writer iterate `i` from `start_id` to `start_id + num_pages`, using `TensorAccessor` to translate the flat page index into a physical DRAM NoC address via `noc_async_read_page(i, ...)` and `noc_async_write_page(i, ...)`.

## Memory Access Patterns

### Read Pattern
Sequential page reads. The reader iterates through tiles in ascending page order (`start_id` to `start_id + num_pages`). Each tile read is a single NoC transaction followed by a read barrier, resulting in one outstanding read at a time per core.

### Write Pattern
Sequential page writes. The writer iterates in the same ascending order. Writes are flushed after each page (`noc_async_writes_flushed`), with a final write barrier at the end.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (compute_with_storage_grid_size) |
| **Grid dimensions** | Device-dependent (e.g., 8x8 on Wormhole) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

Core linearization uses column-major order: `core = {i / num_cores_y, i % num_cores_y}`. The `split_work_to_cores` utility divides the total tiles as evenly as possible, with remainder tiles distributed to group 1 cores.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, bank mapping for the source buffer |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, page size, bank mapping for the destination buffer |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

Additionally, the compute kernel receives preprocessor defines:
- `SFPU_OP_ERFINV_INCLUDE=1` -- gates inclusion of `erfinv.h`
- `SFPU_OP_CHAIN_0` -- expands to `erfinv_tile_init(); erfinv_tile(0);` (init + execute macros)
- `INP_FLOAT` or `INP_FLOAT32` etc. -- input dtype flag

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles this core reads |
| 2 | start_id | uint32_t | First tile page index for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of tiles this core writes |
| 2 | start_id | uint32_t | First tile page index for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ERFINV (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ERFINV (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM | CB c_0 | Sequential tile reads via TensorAccessor |
| compute | TRISC_MATH (RISCV_2) | N/A | CB c_0 | CB c_2 | Unpack tile, SFPU erfinv, pack tile |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM | Sequential tile writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Reads tiles one at a time from DRAM using `noc_async_read_page` with a `TensorAccessor` constructed from compile-time args. Each tile read is individually barriered before pushing to CB c_0.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Writes tiles one at a time to DRAM using `noc_async_write_page` with a `TensorAccessor`. Flushes after each page write. Supports an `OUT_SHARDED` mode (not used here) that simply waits for all tiles in CB.

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
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of blocks (tiles) to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block; always 1 for ERFINV

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initialize unpack/pack hardware for CB c_0 -> c_2 path
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for 1 tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST register file

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // wait for reader to produce 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpack tile 0 from CB c_0 into DEST[0]

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: erfinv_tile_init(); erfinv_tile(0);
            // erfinv_tile_init() calls erfinv_init<APPROX>() which sets up log LUT constants
            // erfinv_tile(0) dispatches calculate_erfinv<APPROX>() on DEST[0] via SFPU
#endif

            tile_regs_commit();  // signal that DEST registers are ready for pack stage

            tile_regs_wait();  // wait for pack stage to be ready to consume DEST

            pack_tile(0, tt::CBIndex::c_2);  // pack DEST[0] result into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed tile in input CB

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish 1 tile to writer
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
(Identical implementation exists for Blackhole at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`)

#### Annotated SFPU Kernel Source
```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"  // provides calculate_log_body and log_init

#include "sfpi.h"  // SFPI intrinsic programming interface for SFPU

namespace ckernel {
namespace sfpu {

// Custom square root using the fast inverse square root (Quake III) algorithm.
// Used instead of the standard sqrt to avoid dependency on the general sqrt SFPU path.
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    v_if(val != 0.0f) {  // SFPU conditional: skip sqrt for zero inputs
        // 0x5f37 is the bfloat16 version of the "magic number" 0x5f3759df from the
        // fast inverse square root algorithm. It provides an initial approximation.
        sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
        // Initial approximation of 1/sqrt(val) via bit manipulation
        sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        // Two Newton-Raphson iterations to refine the inverse square root approximation
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;  // iteration 1
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;  // iteration 2
        // Convert from 1/sqrt(val) to sqrt(val) by multiplying by val
        out = approx * val;
    }
    v_else { out = val; }  // sqrt(0) = 0
    v_endif;
    return out;
}

// Core erfinv computation body using Winitzki's 2008 approximation.
// Formula: erfinv(x) = sign(x) * sqrt( -2/(pi*a) - ln(1-x^2)/2
//                        + sqrt( (2/(pi*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a ) )
// where a = 0.147 is the approximation constant.
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    // Compute log(1 - x^2)
    sfpi::vFloat log_value = in * in;           // x^2
    log_value = 1 - log_value;                  // 1 - x^2
    // Use calculate_log_body with fp32 accumulation (third template param = true)
    // to avoid precision loss in intermediate steps
    log_value = calculate_log_body<false, false, true>(log_value, 0);  // ln(1 - x^2)

    sfpi::vFloat temp = log_value * 0.5;  // ln(1 - x^2) / 2

    // Precomputed constants from a = 0.147:
    constexpr float TwoPiA = 4.330746750799873f;   // 2 / (pi * 0.147)
    constexpr float OneDivA = 6.802721088435375f;   // 1 / 0.147

    // temp = -(2/(pi*a) + ln(1-x^2)/2)
    temp = TwoPiA + temp;
    temp = -temp;

    // Discriminant: temp^2 - ln(1-x^2)/a
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    // Inner sqrt
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    // temp + sqrt(discriminant)
    calculated_value = temp + intermediate_result;

    // Outer sqrt gives the final unsigned result
    sfpi::vFloat result = calculate_sqrt_custom<false>(calculated_value);

    return result;
}

// Main SFPU microcode entry point for erfinv.
// Processes all 8 rows (faces) of a tile column in DEST.
template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    // SFPU microcode: 8 iterations process 8 rows of 4 elements each (one tile face column)
    // A 32x32 tile has 4 faces of 16x16; each face has 16 rows; the SFPU processes
    // 4 elements per SIMD lane, so 8 iterations cover 32 rows (one column of all faces).
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];  // load 4 elements from current DEST row
        sfpi::vFloat result;

        // erfinv is an odd function: erfinv(-x) = -erfinv(x)
        // Compute on |x| to simplify edge-case handling, then restore sign.
        sfpi::vFloat abs_v = sfpi::abs(v);  // |x|

        v_if(abs_v == 1.0f) {
            result = std::numeric_limits<float>::infinity();  // erfinv(+/-1) = +/-inf
        }
        v_elseif(abs_v > 1.0f) {
            result = std::numeric_limits<float>::quiet_NaN();  // erfinv outside [-1,1] is NaN
        }
        v_else {
            result = calculate_erfinv_body<true>(abs_v);  // main approximation
        }
        v_endif;

        result = sfpi::setsgn(result, v);  // restore the original sign of the input

        sfpi::dst_reg[0] = result;  // write result back to DEST
        sfpi::dst_reg++;            // advance to next row (4-element SIMD group)
    }
}

// Initialization function: sets up logarithm LUT constants in SFPU programmable registers.
template <bool APPROXIMATION_MODE>
void erfinv_init() {
    // log_init<false, false, false>() loads ln(2) and polynomial coefficients
    // into vConstFloatPrgm0/1/2 for use by calculate_log_body.
    log_init<false, false, false>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### LLK API Wrapper
**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h`

```cpp
// erfinv_tile dispatches calculate_erfinv on the tile at DEST[idst].
// SFPU_UNARY_NO_PARAM_KERNEL_FN expands to:
//   _llk_math_eltwise_unary_sfpu_params_<APPROX>(
//       ckernel::sfpu::calculate_erfinv<APPROX>, idst, (int)VectorMode::RC)
ALWI void erfinv_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, idst));
}

// erfinv_tile_init calls erfinv_init which loads log LUT constants.
ALWI void erfinv_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(erfinv, sfpu::erfinv_init, APPROX));
}
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[0]` (load) | Loads 4 elements from the current DEST register row into an SFPU vector register |
| `sfpi::dst_reg[0]` (store) | Writes 4 elements back to the current DEST register row |
| `sfpi::dst_reg++` | Advances the DEST register row pointer by one (next 4-element group) |
| `sfpi::abs(v)` | Computes element-wise absolute value via sign-bit manipulation |
| `sfpi::setsgn(result, v)` | Copies the sign bit of `v` onto `result`, restoring the original sign |
| `sfpi::reinterpret<vUInt>(vFloat)` | Bit-casts a float vector to unsigned integer vector (no conversion) |
| `sfpi::reinterpret<vFloat>(vUInt)` | Bit-casts an unsigned integer vector back to float vector |
| `sfpi::s2vFloat16b(0x5f37)` | Loads an immediate bfloat16 scalar into a vector float register |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (condition codes set per SIMD lane) |
| Arithmetic: `*`, `+`, `-`, `>>` | SFPU vector multiply, add, subtract, right-shift (on vUInt) |
| `calculate_log_body<false,false,true>()` | Computes natural logarithm using polynomial approximation with fp32 DEST accumulation |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[0]** (and successive rows) | Input tile data loaded by `copy_tile`; overwritten with erfinv result |
| **dst_reg pointer** | Iterates through 8 rows per `calculate_erfinv` call (32 rows = full tile column) |
| **vConstFloatPrgm0** | Holds `ln(2) = 0.693147...` for log computation (loaded by `log_init`) |
| **vConstFloatPrgm1** | Holds polynomial coefficient `-2.0069785...` for log computation |
| **vConstFloatPrgm2** | Holds polynomial coefficient `3.7675004...` for log computation |
| **SFPU local vFloat registers** | `v`, `abs_v`, `result`, `log_value`, `temp`, `calculated_value`, `approx`, etc. -- all temporary vector registers used during computation |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to ensure a tile is available.
2. **Unpack to DEST**: `copy_tile(c_0, 0, 0)` unpacks the tile from CB c_0 into DEST register 0, converting from the CB data format to the DEST accumulator format.
3. **SFPU init**: `erfinv_tile_init()` calls `log_init<false, false, false>()`, which loads `ln(2)` and two polynomial coefficients into SFPU programmable constant registers (`vConstFloatPrgm0/1/2`). These are needed by the internal `calculate_log_body` function.
4. **SFPU dispatch**: `erfinv_tile(0)` calls `_llk_math_eltwise_unary_sfpu_params_` which sets up the DEST tile index and invokes `calculate_erfinv<APPROX>()` in `VectorMode::RC` (row-column mode, processing all faces).
5. **Per-row loop (8 iterations)**: Each iteration processes one 4-element SIMD row:
   - Load 4 elements from `dst_reg[0]`
   - Compute `|x|`; handle edge cases (`|x| == 1` yields infinity, `|x| > 1` yields NaN)
   - For valid inputs, call `calculate_erfinv_body`:
     - Compute `ln(1 - x^2)` via `calculate_log_body`
     - Compute intermediate terms using Winitzki's formula
     - Two calls to `calculate_sqrt_custom` (fast inverse sqrt + Newton-Raphson)
   - Restore original sign via `setsgn`
   - Write result back to `dst_reg[0]`; advance `dst_reg++`
6. **Pack**: `pack_tile(0, c_2)` packs the transformed DEST data into output CB c_2.
7. **Release**: `cb_pop_front(c_0, 1)` frees the input tile; `cb_push_back(c_2, 1)` publishes the output tile.

#### SFPU Configuration

| Setting | Value | Effect |
|---------|-------|--------|
| **math_fidelity** | HiFi4 | Highest fidelity FPU mode (not directly used by SFPU, but set in ComputeConfig) |
| **math_approx_mode** | `false` | `get_op_approx_mode` returns `false` for all ops by default; ERFINV uses exact mode. The `APPROX` template parameter passed to `calculate_erfinv` will be `false`. |
| **fp32_dest_acc_en** | Configurable | When enabled, DEST registers use fp32 accumulation; `calculate_log_body` is called with `is_fp32_dest_acc_en=true` for higher precision |
| **unpack_to_dest_mode** | Default (or UnpackToDestFp32 if `preserve_fp32_precision`) | Controls whether unpack converts to fp32 in DEST |
| **SFPU_OP_ERFINV_INCLUDE** | `1` | Preprocessor define that gates inclusion of `erfinv.h` via `sfpu_split_includes.h` |
| **Log LUT constants** | Loaded by `log_init` into `vConstFloatPrgm0/1/2` | Required by the internal `calculate_log_body` call |

#### Hardware Compatibility Notes
The SFPU kernel implementations for Wormhole B0 and Blackhole are **identical** (byte-for-byte the same `ckernel_sfpu_erfinv.h`). The erfinv operation uses only standard SFPI intrinsics (`vFloat` arithmetic, `reinterpret`, `abs`, `setsgn`, conditional execution) and the shared `calculate_log_body` function, all of which are available on both architectures. No architecture-specific instructions or divergent behavior is present.

## Implementation Notes

1. **Algorithm choice**: The Winitzki (2008) approximation was chosen for its balance of accuracy and computational simplicity on the SFPU. It requires only log, sqrt, and basic arithmetic -- no iterative root-finding or lookup tables beyond the log constants.

2. **Custom sqrt**: The implementation uses a dedicated `calculate_sqrt_custom` function based on the fast inverse square root trick (adapted for bfloat16 with magic number `0x5f37`) followed by two Newton-Raphson refinement iterations, rather than calling the standard SFPU sqrt. This avoids additional SFPU pipeline configuration overhead and keeps the erfinv kernel self-contained (only depending on log).

3. **Precision strategy**: The `calculate_log_body` is called with `is_fp32_dest_acc_en=true` (third template parameter) to use fp32 accumulation during the logarithm computation, reducing intermediate rounding errors that would otherwise accumulate through the multi-step erfinv formula.

4. **Edge case handling**: The kernel explicitly handles `|x| == 1` (returns infinity) and `|x| > 1` (returns NaN) using SFPU predicated execution (`v_if`/`v_elseif`). The odd symmetry of erfinv is exploited by computing on `|x|` and restoring the sign via `setsgn`.

5. **No runtime parameters**: Unlike parameterized unary ops (e.g., HARDSHRINK with a threshold), ERFINV takes no scalar parameters. The `packed_scalar1` and `packed_scalar2` runtime args are always 0.

6. **Double buffering**: Both input and output CBs have capacity for 2 tiles, enabling overlap between the reader producing the next tile and the compute kernel processing the current one.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the unary program factory work for SFPU operations?"
   **Reason**: To understand the overall structure of `unary_program_factory.cpp` and how it sets up kernels.
   **Key Findings**: The factory creates reader/compute/writer kernels, uses `split_work_to_cores` for distribution, and sets SFPU operation defines via `get_block_defines`. ERFINV uses the default `eltwise_sfpu.cpp` compute kernel path.

2. **Query**: "How is the erfinv (inverse error function) SFPU operation implemented?"
   **Reason**: To identify the kernel files, LLK macros, and initialization functions involved.
   **Key Findings**: The core SFPU implementation is in `ckernel_sfpu_erfinv.h`, wrapped by `erfinv.h` using `SFPU_UNARY_NO_PARAM_KERNEL_FN` and `SFPU_INIT_KERNEL_CALL` macros. Initialization calls `log_init` to set up logarithm constants.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To confirm the compute kernel path selection, macro definitions, and approx mode for ERFINV.
   **Key Information**: ERFINV maps to macro `SFPU_OP_ERFINV_INCLUDE`, init/func pair `erfinv_tile_init()`/`erfinv_tile(0)`, default compute path `eltwise_sfpu.cpp`, and `get_op_approx_mode` returns `false`.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: To understand the `SFPU_UNARY_NO_PARAM_KERNEL_FN` and `SFPU_INIT_KERNEL_CALL` macro expansions.
   **Key Information**: `SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROX, DST_IDX)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::FN<APPROX>, DST_IDX, (int)VectorMode::MODE)`.
