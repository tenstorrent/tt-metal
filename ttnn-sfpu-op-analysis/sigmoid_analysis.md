# Sigmoid Implementation Analysis

## Overview
The sigmoid operation computes the element-wise sigmoid activation function: `sigmoid(x) = 1 / (1 + exp(-x))`. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory`, which orchestrates reader, compute, and writer kernels across Tensix cores. The SFPU kernel computes sigmoid by composing exponential and reciprocal primitives, with separate code paths for FP32 and BF16 precision modes.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `input.buffer()->num_pages()` (total tiles in tensor) |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for sigmoid) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Last-dim-contiguous |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32 |

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
No layout transformations are performed. Input and output must both be in TILE_LAYOUT. The program factory also supports ROW_MAJOR layout (page size comes from buffer rather than tile size), but the SFPU sigmoid path requires TILE_LAYOUT.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src buffer) | CB c_0 (input) | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 (input) | CB c_2 (output) | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU sigmoid, `pack_tile`, `cb_pop_front(c_0, 1)` |
| 3 | Writer | CB c_2 (output) | DRAM/L1 (dst buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The compute kernel reserves `per_core_block_dim` tiles (always 1) in CB c_2 at the start of each block iteration, then pushes them all at the end. This means each block iteration processes exactly 1 tile.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output tiles staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Notes:
- CB c_1 (tmp0) is NOT allocated for sigmoid. It is only created for HARDSHRINK, CBRT, or LOGIT operations.
- Both input and output CBs use double buffering (capacity = 2 * page_size), enabling overlap between reader/compute and compute/writer.
- For BITCAST operations, the input CB uses the output data format. This does not apply to sigmoid.

## Pipeline Pattern Summary
- **Input CB (c_0)**: Double-buffered -- reader can fill one tile while compute processes another.
- **Output CB (c_2)**: Double-buffered -- compute can produce one tile while writer drains another.
- The double buffering on both sides enables a 3-stage pipeline where reader, compute, and writer can operate concurrently on different tiles.

## Index Calculations
The reader and writer kernels use `TensorAccessor` for index-to-address mapping. The factory constructs `TensorAccessorArgs` from the source and destination buffers and passes them as compile-time arguments. At runtime, each core receives:
- `src_addr` / `dst_addr`: base address of the buffer
- `num_pages`: number of tiles this core processes
- `start_id`: the global tile index where this core begins

The reader/writer iterate sequentially from `start_id` to `start_id + num_pages`, calling `noc_async_read_page(i, accessor, l1_addr)` / `noc_async_write_page(i, accessor, l1_addr)`. The `TensorAccessor` internally maps the page index to the correct DRAM bank and address offset based on the interleaved memory layout.

## Memory Access Patterns

### Read Pattern
Sequential tile-by-tile reads. The reader iterates `i` from `start_id` to `start_id + num_pages`, reading one tile per iteration via `noc_async_read_page`. Each read is followed by `noc_async_read_barrier()` before pushing to the CB, so reads are non-pipelined within the reader (one outstanding read at a time).

### Write Pattern
Sequential tile-by-tile writes. The writer iterates the same range, waiting for one tile in CB c_2, reading it, and writing via `noc_async_write_page`. Writes use `noc_async_writes_flushed()` per tile (not a full barrier), with a final `noc_async_write_barrier()` after the loop. This allows some write pipelining.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (logical 1D enumeration over 2D grid) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages / num_cores` tiles (with remainder handling) |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

The `split_work_to_cores` utility divides the total tile count across all available compute cores. Cores are enumerated column-major: `core = {i / num_cores_y, i % num_cores_y}`. If the division is uneven, two separate compute kernels are created -- one for `core_group_1` (more tiles) and one for `core_group_2` (fewer tiles) -- with different `per_core_block_cnt` compile-time arguments.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Serialized tensor accessor parameters for source buffer (bank info, page size, etc.) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Serialized tensor accessor parameters for destination buffer |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for sigmoid) |

#### Compute Kernel Defines
| Define | Value | Description |
|--------|-------|-------------|
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | Includes `compute_kernel_api.h` for sigmoid_tile |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro chain for init + func |
| `SFPU_OP_CHAIN_0_INIT_0` | `sigmoid_tile_init<0u>();` | Init call (0u = not approximate) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `sigmoid_tile<4, 0u>(0);` | Func call (VecMode::RC=4, not approx, dst=0) |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Input data type indicator |

#### Compute Config
| Setting | Value | Description |
|---------|-------|-------------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest math fidelity |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | FP32 accumulation in DEST registers |
| `math_approx_mode` | `false` | Approximation mode disabled for sigmoid |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_pages | uint32_t | Number of tiles this core reads |
| 2 | start_id | uint32_t | Global starting tile index |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of tiles this core writes |
| 2 | start_id | uint32_t | Global starting tile index |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for sigmoid (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for sigmoid (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_interleaved_start_id | RISCV_0 | NOC0 | DRAM/L1 src buffer | CB c_0 | Sequential tile reads via TensorAccessor |
| eltwise_sfpu (compute) | RISCV_2 (math) | N/A | CB c_0 | CB c_2 | copy_tile + sigmoid SFPU + pack_tile |
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 dst buffer | Sequential tile writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Reads tiles one at a time from interleaved DRAM/L1 using `TensorAccessor`. Supports both forward and backward iteration via `BACKWARDS` define (not used for sigmoid). The page size is obtained dynamically from the CB interface, making this kernel layout-agnostic (works for both TILE and ROW_MAJOR).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Writes tiles one at a time to interleaved DRAM/L1. Supports sharded output via `OUT_SHARDED` define (not used in this factory). Uses `noc_async_writes_flushed()` per tile for partial pipelining, with a final barrier at the end.

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
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of blocks (tiles) this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block, always 1 for sigmoid

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // Initialize SFPU pipeline: sets up unpack from c_0, pack to c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // Reserve space for 1 output tile in c_2
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // Acquire exclusive access to DEST registers for this tile

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // Wait until reader has produced 1 tile in c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // Unpack tile 0 from c_0 into DEST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // Expands to: sigmoid_tile_init<0u>(); sigmoid_tile<4, 0u>(0);
                             // This initializes the SFPU for sigmoid (sets up reciprocal LUT)
                             // then executes sigmoid on the tile in DEST[0]
#endif

            tile_regs_commit();  // Signal that DEST registers are ready for packing

            tile_regs_wait();  // Wait for pack engine to be ready

            pack_tile(0, tt::CBIndex::c_2);  // Pack DEST[0] result into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // Free the consumed input tile from c_0

            tile_regs_release();  // Release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // Publish 1 output tile to writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` (Blackhole)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` (Wormhole B0)

Both architectures share the same source code for the sigmoid SFPU kernel (identical implementations).

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid_appx.h"  // Provides calculate_sigmoid_appx() for approximation mode
#include "ckernel_sfpu_recip.h"          // Provides _sfpu_reciprocal_() Newton-Raphson reciprocal

namespace ckernel {
namespace sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function (_sfpu_exp_f32_accurate_ with Cody-Waite range reduction)
    // Otherwise, use exp_21f (~1 ULP on bfloat16, based on Moroz et al. 2022 algorithm)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);  // Dispatches to _sfpu_exp_f32_accurate_ for FP32
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);  // Uses exp_21f algorithm: polynomial approx of 2^(x/ln2)
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;  // denominator = 1.0 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);  // 2 Newton-Raphson iterations for FP32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator);  // 1 Newton-Raphson iteration for BF16 precision
    }

    return result;  // result = 1 / (1 + exp(-x))
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE) {
        // Precise mode: iterate over 8 rows of the tile face (32 elements per row)
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {  // ITERATIONS=8: one per row of a 16x16 face
            sfpi::vFloat val = sfpi::dst_reg[0];  // Load current row from DEST register

            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);  // Compute sigmoid

            if constexpr (!is_fp32_dest_acc_en) {
                // When DEST is BF16, explicitly convert to BF16 with round-to-nearest-even
                // to avoid truncation artifacts from SFPSTORE
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;  // Write result back to DEST register
            sfpi::dst_reg++;            // Advance to next row in the face
        }
    } else {
        // Approximation mode: uses LUT-based sigmoid approximation
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        // For precise mode on Blackhole: calls _init_sfpu_reciprocal_<false>() which sets vConstFloatPrgm0 = 2.0
        // For precise mode on Wormhole: calls _init_reciprocal_<false, false>() which sets up
        //   quadratic approximation constants for the reciprocal initial estimate
        _init_sfpu_reciprocal_<false>();  // Blackhole version
        // _init_reciprocal_<false, false>();  // Wormhole version
    } else {
        // For approx mode: loads LUT constants for sigmoid approximation
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Note on init difference**: The Blackhole version calls `_init_sfpu_reciprocal_<false>()` which simply sets `vConstFloatPrgm0 = 2.0f`. The Wormhole version calls `_init_reciprocal_<false, false>()` which loads quadratic approximation coefficients into programmable constant registers for the reciprocal initial estimate.

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[0]` (read) | Load a vector of elements from DEST register row 0 |
| `sfpi::dst_reg[0]` (write) | Store a vector of elements back to DEST register row 0 |
| `sfpi::dst_reg++` | Advance the DEST register pointer to the next row |
| `sfpi::vConst1` | Built-in constant 1.0f |
| `sfpi::float_to_fp16b(val, 0)` | Convert FP32 to BF16 with round-to-nearest-even |
| `sfpi::reinterpret<vFloat>(val)` | Reinterpret-cast between SFPU vector types |
| `sfpi::approx_recip(x)` (via `_sfpu_reciprocal_`) | Hardware approximate reciprocal (SFPARECIP instruction) |
| `sfpi::setman` (via reciprocal on WH) | Set mantissa field of float |
| `sfpi::exexp` / `sfpi::exman8` / `sfpi::exman9` (via exp) | Extract exponent/mantissa fields |
| `sfpi::setexp` (via exp) | Set exponent field of float |
| `sfpi::shft` (via exp) | Barrel shift |
| `sfpi::int32_to_float` (via exp) | Integer to float conversion |
| `sfpi::addexp` (via exp_61f) | Add to exponent field |
| `sfpi::vec_min_max` (via exp) | Vector min/max for clamping |
| `sfpi::lut` (via sigmoid_appx) | LUT-based function evaluation (approximation mode only) |
| `PolynomialEvaluator::eval` (via exp) | Polynomial evaluation using Horner's method |

#### SFPU Register Usage

- **DEST registers (`dst_reg`)**: The primary I/O for the SFPU. Each tile face has 8 rows of 32 elements. The kernel reads from `dst_reg[0]`, computes sigmoid, writes back to `dst_reg[0]`, then advances with `dst_reg++`. In RC vector mode, all 4 faces (4 x 8 = 32 rows total) are processed.
- **LREG registers (L-registers)**: Used internally by sub-functions:
  - In approximation mode: `LReg0`, `LReg1`, `LReg2` hold the LUT coefficients loaded by `sigmoid_appx_init()`
  - In the reciprocal function (Wormhole): `vConstFloatPrgm0`, `vConstFloatPrgm1`, `vConstFloatPrgm2` hold quadratic approximation constants
  - In the reciprocal function (Blackhole): `vConstFloatPrgm0` holds 2.0f for Newton-Raphson
- **Temporary vFloat variables**: `exp_neg_x`, `denominator`, `result` -- all map to SFPU local registers during execution

#### SFPU Execution Flow

1. **Initialization** (`sigmoid_init`):
   - In precise mode: initializes the reciprocal function by setting `vConstFloatPrgm0 = 2.0f` (Blackhole) or loading quadratic estimate constants (Wormhole).
   - In approximation mode: loads 3 LUT constants (0x3DFF, 0x21D8, 0xFF10) into L-registers.

2. **Per-tile execution** (`calculate_sigmoid`, called via `_llk_math_eltwise_unary_sfpu_params_`):
   - The LLK params wrapper iterates over all 4 faces of the tile (RC vector mode).
   - For each face, `calculate_sigmoid` is called with `ITERATIONS=8` (one per row).
   - For each row:
     a. Read the vector from `dst_reg[0]`
     b. Negate: compute `-x`
     c. Compute `exp(-x)`:
        - **FP32 mode** (`_sfpu_exp_f32_accurate_`): Cody-Waite range reduction + 7th-order Taylor polynomial
        - **BF16 mode** (`_sfpu_exp_21f_`): Moroz et al. 2022 algorithm with 2nd-degree polynomial refinement
     d. Compute `1 + exp(-x)`
     e. Compute reciprocal of denominator:
        - **FP32 mode**: `_sfpu_reciprocal_<2>` -- `approx_recip` + 2 Newton-Raphson iterations
        - **BF16 mode**: `_sfpu_reciprocal_<1>` -- `approx_recip` + 1 Newton-Raphson iteration
     f. If BF16 mode: explicitly convert result to BF16 with round-to-nearest-even
     g. Write result back to `dst_reg[0]`
     h. Advance `dst_reg++`

3. **Post-execution**: The LLK wrapper calls `_llk_math_eltwise_unary_sfpu_done_()` to finalize.

#### SFPU Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `APPROXIMATION_MODE` | `false` (default) | Uses precise exp + reciprocal; can be `true` for LUT-based approximation |
| `is_fp32_dest_acc_en` | From `ComputeConfig.fp32_dest_acc_en` | Controls whether FP32 or BF16 code paths are used |
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity (affects FPU, not directly SFPU) |
| `math_approx_mode` | `false` | `get_op_approx_mode()` returns false for all ops |
| `vector_mode` | `VectorMode::RC` (4) | Process all 4 faces of the tile |

The `SFPU_OP_CHAIN_0` macro expands to:
```
sigmoid_tile_init<0u>(); sigmoid_tile<4, 0u>(0);
```
Where `0u` = not approximate, `4` = VectorMode::RC, `0` = dst_index.

#### Hardware Compatibility Notes

- **Blackhole vs Wormhole reciprocal initialization**: Blackhole's `_init_sfpu_reciprocal_<false>()` sets `vConstFloatPrgm0 = 2.0f`. Wormhole's `_init_reciprocal_<false, false>()` loads 3 programmable constants for a quadratic initial reciprocal estimate.
- **Reciprocal algorithm**: Both architectures use the same `_sfpu_reciprocal_` template, but the implementation differs. Blackhole uses `sfpi::approx_recip()` (SFPARECIP hardware instruction) as the starting point for Newton-Raphson. Wormhole uses a software quadratic approximation (`setman` + polynomial) followed by Newton-Raphson.
- **Exp algorithm**: Both use the same `_sfpu_exp_21f_` for BF16 and `_sfpu_exp_f32_accurate_` for FP32 (with Cody-Waite range reduction and 7th-order Taylor series).
- **Sigmoid init difference**: The Blackhole sigmoid_init calls `_init_sfpu_reciprocal_<false>()` while the Wormhole version calls `_init_reciprocal_<false, false>()`. This is because the reciprocal sub-functions have different initialization requirements on each architecture.
- **BF16 conversion**: Both architectures require explicit `float_to_fp16b` conversion when DEST is in BF16 mode, because SFPSTORE would otherwise truncate rather than round.
- **LLK params wrapper**: Wormhole's `llk_math_eltwise_unary_sfpu_sigmoid` uses `ITERATIONS = 8` explicitly, while Blackhole's version uses the default template parameter `ITERATIONS = 8`. The behavior is identical.

## Implementation Notes

1. **Op chain mechanism**: The compute kernel uses a macro-based op chain (`SFPU_OP_CHAIN_0`) that expands to init + function calls. This allows the same kernel source to serve all unary SFPU operations -- the specific operation is injected via preprocessor defines.

2. **Parametrized sigmoid**: When created via `string_to_unary_with_param("sigmoid")`, the op gets params `{VecMode::RC, false}`. The `false` means non-approximate. A `sigmoid_approx` variant exists with `{VecMode::RC, true}` that uses the LUT-based path.

3. **Approximation mode**: The LUT-based approximation (`calculate_sigmoid_appx`) uses 3 pre-loaded constants and the `lut()` SFPI intrinsic to evaluate `lut(val, l0, l1, l2) + 0.5`. This is significantly faster but less accurate than the precise path.

4. **Precision trade-offs in precise mode**:
   - FP32: Uses Cody-Waite exp with 7th-order Taylor + 2 Newton-Raphson reciprocal iterations (sub-ULP accuracy)
   - BF16: Uses Moroz exp_21f with 2nd-degree polynomial + 1 Newton-Raphson reciprocal iteration (~1 ULP accuracy)

5. **Runtime argument override**: The `override_runtime_arguments` method only updates buffer addresses (arg[0]) for reader and writer kernels, allowing efficient tensor reuse without recompilation.

6. **SubCoreGrid variant**: The file also contains `UnarySubCoreGridProgramFactory` which supports running on a subset of cores. The sigmoid logic is identical; only the core allocation differs.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How is the unary program factory implemented for SFPU operations like sigmoid?"
   **Reason**: Needed to understand the overall structure of the program factory and identify kernel files.
   **Key Findings**: Confirmed the three-kernel pattern (reader/compute/writer), the use of `split_work_to_cores`, and the `eltwise_sfpu.cpp` compute kernel with define-based op injection.

2. **Query**: "Where is the SFPU sigmoid kernel implementation located? What LLK/ckernel functions does it call?"
   **Reason**: Needed to locate the SFPU kernel source files and understand the call chain.
   **Key Findings**: Located `ckernel_sfpu_sigmoid.h` for both architectures. Confirmed sigmoid uses `_sfpu_exp_improved_`/`_sfpu_exp_21f_` for exp and `_sfpu_reciprocal_` for the final division.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to determine which compute kernel path sigmoid uses and what defines are generated.
   **Key Information**: Sigmoid uses default `eltwise_sfpu.cpp` kernel, gets `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` macro, and the parametrized version generates `sigmoid_tile_init<approx>()` and `sigmoid_tile<vecmode, approx>(idst)`.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand how the LLK params wrapper orchestrates face iteration.
   **Key Information**: In VectorMode::RC, the wrapper calls the SFPU function 4 times (once per face), advancing the DEST face address between calls. Each call processes 8 rows (ITERATIONS=8) of 32 elements.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Needed to understand the reciprocal implementation used by sigmoid.
   **Key Information**: `_sfpu_reciprocal_<N>` uses `approx_recip` (SFPARECIP) as initial estimate, then N iterations of Newton-Raphson refinement. The Blackhole version handles NaN edge cases (0*inf) via sign-based detection.
