# SILU (SiLU / Swish) Implementation Analysis

## Overview

SiLU (Sigmoid Linear Unit), also known as Swish, computes `silu(x) = x * sigmoid(x)` element-wise on a tensor. It is a smooth, non-monotonic activation function commonly used in modern neural network architectures. The operation is implemented as a unary SFPU operation through the shared unary program factory.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles per block (always 1 for this factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Any shape (flattened to total tile count) |
| **Dimension convention** | N/A (shape-agnostic, operates on flat tile stream) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16, FLOAT32 supported |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (or as specified by output tensor) |

### Layout Transformations

None. The operation is a pure element-wise unary; input and output share the same layout and shape.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, SFPU ops, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The reader fetches one tile at a time from DRAM into CB c_0. The compute kernel waits for a tile in c_0, unpacks it into DEST registers, applies the SFPU silu operation (x * sigmoid(x)), packs the result into CB c_2, then frees c_0. The writer waits for a tile in c_2, writes it to DRAM, and frees c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_1 (tmp0) is NOT created for SILU. It is only created for HARDSHRINK, CBRT, and LOGIT operations.
- Both CBs use double-buffering (capacity = 2 * page_size), enabling overlap between the producer and consumer of each buffer.
- Page size is `single_tile_size` for TILE_LAYOUT (determined by `tt::tile_size(cb_data_format)`).

## Pipeline Pattern Summary

Both circular buffers (c_0 and c_2) are double-buffered, allowing the reader to fill one tile slot while the compute kernel processes another, and similarly for compute/writer overlap. This enables a 3-stage pipelined execution across reader, compute, and writer.

## Index Calculations

Index mapping uses the `TensorAccessor` abstraction. The reader and writer kernels receive:
- `src_addr` / `dst_addr`: Base address of the buffer in DRAM
- `start_id`: The first page (tile) index this core should process
- `num_pages`: How many pages this core processes

Page indices are sequential starting from `start_id`. The `TensorAccessor` translates a linear page index into the correct DRAM bank and offset, handling the interleaved memory layout where tiles are distributed round-robin across DRAM banks.

## Memory Access Patterns

### Read Pattern
Sequential tile reads. Each core reads a contiguous range of tile indices `[start_id, start_id + num_pages)`. Within that range, tiles are read one at a time in order. Each `noc_async_read_page` fetches a single tile from its interleaved DRAM bank location into L1 CB c_0.

### Write Pattern
Sequential tile writes. Mirrors the read pattern. Each core writes tiles in the same order they were computed, one at a time via `noc_async_write_page` from CB c_2 to the output buffer's interleaved DRAM banks.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (linearized as 1D) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores` based on total tile count |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split with remainder handling |

The `split_work_to_cores` utility divides `num_pages` tiles across available cores. If tiles divide evenly, all cores get the same count (core_group_1 only). If not, `core_group_1` gets `ceil(num_pages / num_cores)` tiles per core and `core_group_2` gets one fewer tile per core. Separate compute kernels are created for each group with different `per_core_block_cnt` compile-time arguments.

Core linearization: Core index `i` maps to `CoreCoord{i / num_cores_y, i % num_cores_y}`, filling columns first (column-major order within the grid).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for the source buffer (encodes memory layout, bank mapping, page size) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core (equals number of tiles since block_size=1) |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block, always 1 |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for the destination buffer |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to read |
| 2 | start_id | uint32_t | First page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for SILU (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for SILU (always 0) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to write |
| 2 | start_id | uint32_t | First page index for this core |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB c_0 | Read tiles via `noc_async_read_page` |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | `copy_tile` (unpack to DEST), `silu_tile` (SFPU silu), `pack_tile` (DEST to CB) |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM | Write tiles via `noc_async_write_page` |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Creates a `TensorAccessor` from compile-time args and the runtime source address. Loops through `[start_id, start_id + num_pages)`, reading one page at a time into CB c_0. Uses `noc_async_read_barrier()` after each page to ensure the read completes before publishing.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Creates a `TensorAccessor` from compile-time args. Loops through `[start_id, start_id + num_pages)`, waiting for one tile in CB c_2, writing it to DRAM via `noc_async_write_page`, flushing, then popping. Final `noc_async_write_barrier()` ensures all writes complete.

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
// For SILU, the define SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1 is set,
// which causes sfpu_split_includes.h to include compute_kernel_api.h,
// providing silu_tile() and silu_tile_init().

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    // per_core_block_cnt: number of blocks (= number of tiles for SILU, since block_dim=1)
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    // per_core_block_dim: tiles per block, always 1 for standard unary factory

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    // Initializes the SFPU pipeline: configures unpack for CB c_0 as input,
    // and pack for CB c_2 as output. Also calls the SFPU_OP_CHAIN_0_INIT_0
    // macro which expands to silu_tile_init() for SILU.
    // silu_tile_init() -> llk_math_eltwise_unary_sfpu_silu_init<APPROX>()
    //   -> sigmoid_init<false>() on Blackhole (non-approx path)
    //   -> _init_sfpu_reciprocal_<false>() which sets vConstFloatPrgm0 = 2.0f

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);
        // Reserve space in output CB for per_core_block_dim tiles (1 tile)

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            // Acquire exclusive access to DEST registers for the math engine

            cb_wait_front(tt::CBIndex::c_0, 1);
            // Block until reader has pushed at least 1 tile into CB c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);
            // Unpack tile 0 from CB c_0 into DEST register 0
            // This triggers the unpacker to convert from the CB data format
            // (e.g., bfloat16) into the DEST register format (float32 if fp32_dest_acc_en)

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            // This macro expands to:
            //   SFPU_OP_CHAIN_0_INIT_0  ->  silu_tile_init();
            //   SFPU_OP_CHAIN_0_FUNC_0  ->  silu_tile(0);
            // silu_tile(0) calls llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(0)
            // which dispatches calculate_silu<is_fp32_dest_acc_en, 8>() across all 4 faces
            // of the tile via _llk_math_eltwise_unary_sfpu_params_
#endif

            tile_regs_commit();
            // Signal that math engine is done writing to DEST; hand off to packer

            tile_regs_wait();
            // Wait for packer to be ready to read from DEST

            pack_tile(0, tt::CBIndex::c_2);
            // Pack DEST register 0 into CB c_2, converting from DEST format
            // back to the output data format

            cb_pop_front(tt::CBIndex::c_0, 1);
            // Free the consumed tile in CB c_0, allowing reader to write more

            tile_regs_release();
            // Release DEST registers for the next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
        // Publish per_core_block_dim tiles (1) in CB c_2 for the writer to consume
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`

(Wormhole version: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_sigmoid.h"
// Includes the sigmoid helper _sfpu_sigmoid_ which computes 1/(1+exp(-x))

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    // Unroll hint: process up to 8 row-pairs per face.
    // ITERATIONS is 8 by default, corresponding to 8 row-pairs in a 16-row face.
    // Each iteration processes one row-pair (2 rows of 16 elements = 32 elements)
    // via the SFPU's SIMD-32 datapath.
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Load the current row-pair from DEST register into SFPU local register.
        // dst_reg[0] accesses the current DEST row-pair (32 elements).

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        // _sfpu_sigmoid_ computes sigmoid(x) = 1/(1+exp(-x)):
        //   1. Computes exp(-x) using either _sfpu_exp_improved_ (fp32 mode)
        //      or _sfpu_exp_21f_ (bfloat16 mode)
        //   2. Adds 1.0 to get denominator = 1 + exp(-x)
        //   3. Computes reciprocal using _sfpu_reciprocal_ with Newton-Raphson
        //      (2 iterations for fp32, 1 for bfloat16)
        // The result sigmoid(x) is then multiplied by x.

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            // Convert float32 result to bfloat16 using round-to-nearest-even.
            // This is necessary because SFPU local registers are always float32,
            // but if DEST is bfloat16, we must round before storing back to avoid
            // truncation errors when SFPSTORE writes to DEST.
        }

        sfpi::dst_reg[0] = result;
        // Write the silu result back to the current DEST row-pair.

        sfpi::dst_reg++;
        // Advance the DEST register pointer to the next row-pair.
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    // Blackhole version: always calls sigmoid_init<false>() regardless of APPROXIMATION_MODE.
    // This is because calculate_silu uses _sfpu_sigmoid_ which always takes the non-approx path.
    sigmoid_init<false>();
    // sigmoid_init<false>() calls _init_sfpu_reciprocal_<false>()
    // which sets vConstFloatPrgm0 = 2.0f (used by Newton-Raphson reciprocal).
}

}  // namespace ckernel::sfpu
```

#### SFPU Instructions Used

The SILU SFPU kernel is composed of several sub-operations. Here are the key SFPU instructions and intrinsics used throughout the call chain:

1. **`sfpi::dst_reg[0]`** (SFPLOAD/SFPSTORE) -- Loads a row-pair (32 elements) from the DEST register file into an SFPU local register (vFloat), or stores back.

2. **`sfpi::dst_reg++`** (SFPINCRWC) -- Increments the DEST register pointer to the next row-pair.

3. **Negation (`-x`)** -- SFPU arithmetic negation, used to compute `-x` for `exp(-x)`.

4. **`sfpi::float_to_fp16b(result, 0)`** -- Converts float32 to bfloat16 with round-to-nearest-even. Used when `is_fp32_dest_acc_en` is false.

5. **`sfpi::reinterpret<vFloat>()`** -- Bitwise reinterpretation between vFloat/vInt/vUInt types without any data conversion.

6. **Arithmetic multiply (`x * sigmoid_result`)** (SFPMAD) -- Multiplied-add operation used for the final `x * sigmoid(x)`.

**Within `_sfpu_sigmoid_` (sigmoid sub-kernel):**

7. **`_sfpu_exp_21f_` / `_sfpu_exp_improved_`** -- Polynomial-based exponential approximation. Uses:
   - `sfpi::exexp()` / `sfpi::exexp_nodebias()` -- Extract exponent field from float
   - `sfpi::exman8()` / `sfpi::exman9()` -- Extract mantissa with implicit bit
   - `sfpi::shft()` -- Barrel shift for integer conversion
   - `sfpi::int32_to_float()` -- Integer to float conversion
   - `sfpi::setexp()` -- Set exponent field of a float
   - `sfpi::vec_min_max()` (SFPSWAP) -- Clamping via min/max swap
   - `PolynomialEvaluator::eval()` -- Horner's method polynomial evaluation using SFPMAD chains

8. **`sfpi::vConst1`** -- Constant 1.0f loaded from the SFPU constant register file.

9. **Addition (`vConst1 + exp_neg_x`)** (SFPMAD with one operand = 1) -- Computes `1 + exp(-x)`.

10. **`_sfpu_reciprocal_<N>(denominator)`** -- Newton-Raphson reciprocal:
    - `sfpi::approx_recip()` (SFPARECIP) -- Hardware approximate reciprocal seed (~7-bit precision)
    - SFPMAD chains for Newton-Raphson refinement: `y = y * (2 - x*y)` iterated N times
    - `sfpi::vConstFloatPrgm0` -- Programmable constant register loaded with 2.0f during init
    - `v_if` / `v_endif` -- SFPU conditional execution for NaN detection

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[0..3]** | Four faces of a 32x32 tile. Each face is 16x16 elements, processed as 8 row-pairs of 32 elements. `_llk_math_eltwise_unary_sfpu_params_` iterates over all 4 faces. |
| **dst_reg pointer** | Auto-incremented through 8 row-pairs per face (ITERATIONS=8). |
| **SFPU LRegs (vFloat)** | Local registers used for intermediate values: `x`, `exp_neg_x`, `denominator`, `result`, and Newton-Raphson temporaries `y`, `t`. |
| **vConstFloatPrgm0** | Programmable constant set to 2.0f during `silu_init()`, used by `_sfpu_reciprocal_` for Newton-Raphson. |
| **vConst0** | Constant 0.0f |
| **vConst1** | Constant 1.0f, used in `1 + exp(-x)` |

#### SFPU Execution Flow

1. **Initialization** (`silu_tile_init`):
   - Calls `silu_init<APPROX>()` which calls `sigmoid_init<false>()`
   - `sigmoid_init<false>()` calls `_init_sfpu_reciprocal_<false>()` which sets `vConstFloatPrgm0 = 2.0f`
   - On Blackhole, this specifically does NOT initialize the LOADMACRO-based fast reciprocal since `_sfpu_sigmoid_` uses the inline `_sfpu_reciprocal_` function rather than the standalone `calculate_reciprocal`

2. **Tile acquisition** (in `eltwise_sfpu.cpp`):
   - `cb_wait_front(c_0, 1)` blocks until a tile is available
   - `copy_tile(c_0, 0, 0)` unpacks tile from CB c_0 into DEST register 0

3. **SFPU dispatch** (`silu_tile(0)`):
   - Calls `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(0)`
   - `_llk_math_eltwise_unary_sfpu_start_` sets the DEST base address for tile index 0
   - In RC (full tile) mode, iterates over all 4 faces (0-3)
   - For each face, calls `calculate_silu<is_fp32_dest_acc_en, 8>()`

4. **Per-face SFPU computation** (`calculate_silu`):
   - For each of the 8 row-pairs (ITERATIONS=8):
     a. Load row-pair from DEST: `x = dst_reg[0]`
     b. Compute `sigmoid(x)`:
        - Compute `exp(-x)` via polynomial approximation
        - Compute `1 + exp(-x)`
        - Compute reciprocal of denominator via `approx_recip` + Newton-Raphson
     c. Multiply: `result = x * sigmoid(x)`
     d. If bfloat16 mode: round result to bfloat16
     e. Store back: `dst_reg[0] = result`
     f. Advance to next row-pair: `dst_reg++`

5. **Pack and publish** (back in `eltwise_sfpu.cpp`):
   - `tile_regs_commit()` / `tile_regs_wait()` synchronize MATH and PACK engines
   - `pack_tile(0, c_2)` packs DEST[0] into CB c_2
   - `cb_pop_front(c_0, 1)` frees the input tile
   - `cb_push_back(c_2, 1)` publishes the output tile for the writer

#### SFPU Configuration

- **Math fidelity**: `MathFidelity::HiFi4` (set in `ComputeConfig`)
- **Math approx mode**: `false` (SILU's `get_op_approx_mode` returns false by default, so no approximation)
- **fp32_dest_acc_en**: Configurable per-call (affects which exp and reciprocal paths are used)
- **unpack_to_dest_mode**: Default (or `UnpackToDestFp32` if `preserve_fp32_precision` is set)
- **Preprocessor defines**: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1` enables including `compute_kernel_api.h` which provides `silu_tile()` and `silu_tile_init()`
- **SFPU_OP_CHAIN_0**: Expands to `silu_tile_init(); silu_tile(0);`

#### Hardware Compatibility Notes

**Blackhole vs. Wormhole differences in the SILU kernel:**

1. **`silu_init()` implementation differs**:
   - **Blackhole**: Always calls `sigmoid_init<false>()` regardless of `APPROXIMATION_MODE`. The comment explains this is because `calculate_silu` always uses the non-approx `_sfpu_sigmoid_` path.
   - **Wormhole**: Calls `_init_sfpu_reciprocal_<false>()` or `_init_sfpu_reciprocal_<true>()` based on `APPROXIMATION_MODE`. This is a subtle difference in initialization, though in practice SILU always has `APPROXIMATION_MODE=false`.

2. **`sigmoid_init<false>()` implementation differs**:
   - **Blackhole**: Calls `_init_sfpu_reciprocal_<false>()` which sets `vConstFloatPrgm0 = 2.0f`
   - **Wormhole**: Calls `_init_reciprocal_<false, false>()` which initializes the LOADMACRO-based fast reciprocal pipeline

3. **`_sfpu_reciprocal_` implementation**:
   - **Blackhole**: Uses `approx_recip` + Newton-Raphson iterations with conditional NaN handling using `v_if(t < 0)`. The Blackhole-specific comment notes that when `x=0` and `y=infinity`, `t=+NaN` regardless of operand signs.
   - **Wormhole**: May use a different reciprocal implementation from the LLK library with different register allocation and pipeline scheduling.

4. **`_sfpu_exp_21f_` and `_sfpu_exp_improved_`**: Both architectures share the same high-level algorithm (polynomial approximation based on Moroz et al. 2022), but the underlying SFPU instruction encodings and timing differ.

5. **`calculate_silu()` function**: Identical between Blackhole and Wormhole -- both use the same `x * _sfpu_sigmoid_(x)` formulation with the same template parameters.

## Implementation Notes

1. **No scalar parameters**: Unlike operations like HARDSHRINK or WHERE_TSS, SILU has no scalar parameters. The runtime args `packed_scalar1` and `packed_scalar2` are always 0 for SILU.

2. **Op chain mechanism**: The SFPU operation is injected via preprocessor defines rather than being hard-coded in the compute kernel. The program factory calls `get_block_defines()` which generates `SFPU_OP_CHAIN_0` expanding to `silu_tile_init(); silu_tile(0);`. This allows the same `eltwise_sfpu.cpp` kernel to be reused for many different unary SFPU operations.

3. **Compute kernel path**: SILU uses the default `eltwise_sfpu.cpp` compute kernel (falls through the `default` case in `get_compute_kernel_path`).

4. **Include mechanism**: SILU falls into the `default` case of `get_macro_definition`, which returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"`. This causes `sfpu_split_includes.h` to include the full `compute_kernel_api.h`, which provides `silu_tile()` and `silu_tile_init()`.

5. **Sigmoid reuse**: SILU's SFPU kernel is remarkably concise because it delegates to the pre-existing `_sfpu_sigmoid_` helper. The entire mathematical complexity (exp, reciprocal, Newton-Raphson) is encapsulated in the sigmoid sub-kernel.

6. **Precision modes**: Two distinct precision paths exist:
   - **bfloat16 mode** (`is_fp32_dest_acc_en=false`): Uses `_sfpu_exp_21f_` (2nd degree polynomial, ~1 ULP on bfloat16) and 1-iteration Newton-Raphson reciprocal. Result is explicitly rounded to bfloat16.
   - **float32 mode** (`is_fp32_dest_acc_en=true`): Uses `_sfpu_exp_f32_accurate_` (7th order Taylor series with Cody-Waite range reduction, <1 ULP) and 2-iteration Newton-Raphson reciprocal. No rounding.

7. **Program caching**: The `override_runtime_arguments` method enables program caching. When the operation is called again with the same configuration but different tensors, only the buffer addresses need to be updated in the runtime args, avoiding full program recreation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations like silu? What kernels does it use (reader, compute, writer)? How are circular buffers configured and how is work distributed across cores?"
   **Reason**: Needed architectural overview of the unary program factory before diving into source code.
   **Key Findings**: Confirmed three-kernel pattern (reader/compute/writer), double-buffered CBs (c_0 for input, c_2 for output), and two-group core distribution via `split_work_to_cores`. Identified the key source files and the SFPU_OP_CHAIN define mechanism.

2. **Query**: "What is the SILU (SiLU / swish) SFPU kernel implementation? Where is the silu compute kernel and SFPU kernel source code located in the repository?"
   **Reason**: Needed to locate the SFPU kernel files and understand the silu computation approach.
   **Key Findings**: Confirmed `ckernel_sfpu_silu.h` location for both architectures, the `x * sigmoid(x)` formulation, and the LLK API wrapper in `llk_math_eltwise_unary_sfpu_silu.h`. Also identified that `silu_tile` is in `compute_kernel_api.h`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how SILU's compute kernel path, macro defines, and SFPU op chain are determined.
   **Key Information**: SILU uses default `eltwise_sfpu.cpp` kernel, `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` define, `silu_tile_init()` / `silu_tile(0)` op chain, and `get_op_approx_mode` returns false.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand how the SFPU kernel function is dispatched across tile faces.
   **Key Information**: `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces in RC mode, calling the provided SFPU function for each face. Each face processes 8 row-pairs (ITERATIONS=8).

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Needed to understand the `_sfpu_reciprocal_` implementation used by sigmoid.
   **Key Information**: Uses `approx_recip` hardware instruction as seed, then Newton-Raphson refinement with configurable iterations (0, 1, or 2). Includes NaN handling for edge cases (0 * inf).

4. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Needed to understand the exponential functions used by sigmoid (`_sfpu_exp_21f_` and `_sfpu_exp_improved_`).
   **Key Information**: `_sfpu_exp_21f_` uses Moroz et al. 2022 algorithm with 2nd degree polynomial (fast, ~1 ULP on bfloat16). `_sfpu_exp_improved_` dispatches to `_sfpu_exp_21f_` for bfloat16 or `_sfpu_exp_f32_accurate_` (7th order Taylor) for float32 mode.
