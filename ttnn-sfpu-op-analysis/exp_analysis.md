# Exp Implementation Analysis

## Overview

The **exp** (natural exponential) operation computes `e^x` element-wise on each element of an input tensor. It is implemented as a unary SFPU operation that flows through the shared `UnaryProgramFactory`, which orchestrates reader, compute, and writer kernels across multiple Tensix cores.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The factory is shared by all unary SFPU operations. The exp-specific behavior is injected via preprocessor defines (`SFPU_OP_EXP_INCLUDE` and `SFPU_OP_CHAIN_0`) that configure the compute kernel to call `exp_tile_init()` and `exp_tile(0)`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `input.buffer()->num_pages()` (total tiles in the tensor) |
| **Loop structure** | Outer loop over blocks (`per_core_block_cnt` iterations), inner loop over tiles within a block (`per_core_block_dim` = 1 tile). Effectively a flat tile-by-tile loop. |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened into pages/tiles |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR, but TILE_LAYOUT is the standard path for SFPU) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | BFLOAT16 (typical), FLOAT32, or INT32/UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Same as input (may differ in bitcast scenarios, but not for exp) |

### Layout Transformations

No layout transformations (tilize/untilize) are performed within this program factory. The input and output are expected to already be in the same layout.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (via NoC) | CB c_0 (input) | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 (input) | CB c_2 (output) | `cb_reserve_back(c_2, block_dim)`, `cb_wait_front(c_0, 1)`, `copy_tile`, `exp_tile_init(); exp_tile(0)`, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, block_dim)` |
| 3 | Writer | CB c_2 (output) | DRAM (via NoC) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)`, `noc_async_write_barrier` |

**Step-by-step flow:**

1. The **reader** kernel iterates over its assigned pages, reading one tile at a time from DRAM into CB c_0 using the TensorAccessor for address calculation.
2. The **compute** kernel iterates over blocks. For each block, it reserves output space in CB c_2, then processes each tile: acquires DEST registers, waits for a tile in CB c_0, copies it to DEST via `copy_tile`, runs the SFPU exp operation via `SFPU_OP_CHAIN_0`, commits DEST, waits for pack, packs the result into CB c_2, pops the input tile from c_0, and releases DEST.
3. The **writer** kernel iterates over its assigned pages, waiting for one tile at a time in CB c_2, writing it to DRAM via NoC, and popping the tile.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | Input CB | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | Output CB | Output tile staging | 2 tiles | 1 tile (per_core_block_dim) | Double | Compute | Writer | Program |

**Notes:**
- CB c_1 (tmp0) is only created for HARDSHRINK, CBRT, or LOGIT operations -- not for exp.
- Both input and output CBs are double-buffered (capacity = 2 * page_size), allowing overlap between producer and consumer.

## Pipeline Pattern Summary

- **CB c_0**: Double-buffered (capacity 2 tiles, block size 1 tile). The reader can write the next tile while compute processes the current one.
- **CB c_2**: Double-buffered (capacity 2 tiles, block size 1 tile). The compute kernel can produce the next tile while the writer drains the current one.
- This enables a 3-stage pipelined execution: reader, compute, and writer can all be active simultaneously on different tiles.

## Index Calculations

- The **TensorAccessor** abstraction handles mapping from a linear page index to physical DRAM addresses. The reader and writer use `TensorAccessorArgs` (compile-time) and `TensorAccessor` (runtime) to translate page indices to NoC addresses.
- Each core receives a `start_id` (the first page index it should process) and `num_pages` (how many pages to process). The core iterates linearly: `for (i = start_id; i < start_id + num_pages; i++)`.
- Page index `i` is mapped to a physical address by the TensorAccessor, which accounts for the interleaved bank layout.

## Memory Access Patterns

### Read Pattern

Sequential tile reads. Each core reads a contiguous range of tile indices `[start_id, start_id + num_pages)`. Within each iteration, one tile is read via `noc_async_read_page`, which translates the page index to a (bank_id, offset) pair and issues a NoC read to the corresponding DRAM bank. A barrier is issued after each read to ensure completion before pushing to the CB.

### Write Pattern

Sequential tile writes. Each core writes tiles in the same order as reads: `[start_id, start_id + num_pages)`. One tile is written per iteration via `noc_async_write_page`. A flush is issued per-tile, and a final barrier is issued after the loop completes.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (flattened to 1D indexing) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | Determined by `split_work_to_cores()` based on total pages |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: `core_group_1` gets `ceil(num_pages / num_cores)` tiles, `core_group_2` gets `floor(num_pages / num_cores)` tiles (or is empty if tiles divide evenly) |

**Details:**
- `split_work_to_cores(compute_with_storage_grid_size, num_pages)` divides the total tile count across available cores.
- Cores are indexed in column-major order: `core = {i / num_cores_y, i % num_cores_y}`.
- `core_group_1` and `core_group_2` may have different tile counts to handle the remainder. Each group gets a separately compiled compute kernel with its own `per_core_block_cnt`.
- Runtime args for reader and writer are set per-core with the running `num_pages_written` offset to ensure non-overlapping work.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for the source buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encoded tensor accessor parameters for the destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core reads |
| 2 | start_id | uint32_t | First page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_pages | uint32_t | Number of pages (tiles) this core writes |
| 2 | start_id | uint32_t | First page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for exp (set to 0) |
| 1 | packed_scalar2 | uint32_t | Unused for exp (set to 0) |

### Preprocessor Defines (Compile-Time Configuration for Compute)

For the exp operation, the following defines are injected into the compute kernel:

| Define | Value | Purpose |
|--------|-------|---------|
| `SFPU_OP_EXP_INCLUDE` | `"1"` | Causes `sfpu_split_includes.h` to include `exp.h` |
| `SFPU_OP_CHAIN_0` | `"SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"` | Expands to init + func calls |
| `SFPU_OP_CHAIN_0_INIT_0` | `"exp_tile_init<Pu>();"` where `P` = `(uint32_t)param0` | Initializes SFPU for exp (param0 is approx mode flag) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `"exp_tile<Pu>(0);"` where `P` = `(uint32_t)param0` | Calls SFPU exp on tile at DST index 0 |
| `INP_FLOAT` or `INP_FLOAT32` | `"1"` | Indicates input data type |

The `param0` value for exp is the **approximation mode flag**. When `exp` is created via `string_to_unary_with_param("exp")`, `param0 = true` (i.e., approximate mode is enabled by default).

### Compute Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity math mode |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | Whether DEST accumulator uses FP32 |
| `math_approx_mode` | `false` (for exp, `get_op_approx_mode` returns `false` for all ops) | Global approx mode flag (note: the per-op approx is controlled via template params) |
| `unpack_to_dest_mode` | `Default` (or `UnpackToDestFp32` if `preserve_fp32_precision`) | How unpacker writes to DEST |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_interleaved_start_id | RISCV_0 (BRISC) | NOC0 | DRAM | CB c_0 | Read tiles via TensorAccessor |
| eltwise_sfpu (compute) | RISCV_2 (MATH/TRISC) | N/A | CB c_0 | CB c_2 | copy_tile, exp_tile (SFPU), pack_tile |
| writer_unary_interleaved_start_id | RISCV_1 (NCRISC) | NOC1 | CB c_2 | DRAM | Write tiles via TensorAccessor |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Uses `TensorAccessor` to resolve page indices to physical NoC addresses. Reads one page at a time with a barrier per read (no batching). Supports both forward and backward iteration via `BACKWARDS` define (not used for exp).

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page writer. Uses `TensorAccessor` for address resolution. Writes one page at a time, flushing after each write. Supports sharded output via `OUT_SHARDED` define (not used in this factory). Final `noc_async_write_barrier` ensures all writes complete.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // When SFPU_OP_EXP_INCLUDE=1, this includes exp.h
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tiles this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    // Initialize SFPU pipeline: configures unpack (c_0 -> SRC), math, and pack (DEST -> c_2)
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space for per_core_block_dim tiles in the output CB before starting the block
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DEST registers for math operations
            tile_regs_acquire();

            // Wait until the reader has produced at least 1 tile in the input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from CB c_0 position 0 into DEST register 0 (unpack + move)
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU_OP_CHAIN_0 expands to:
            //   exp_tile_init<1u>();  -- initialize SFPU for exp (approx mode = true by default)
            //   exp_tile<1u>(0);     -- compute exp on tile at DEST[0]
            // (The template parameter 1u indicates approximate mode is enabled)
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DEST register writes are complete, hand off to packer
            tile_regs_commit();

            // Wait for packer to be ready to read from DEST
            tile_regs_wait();

            // Pack tile from DEST[0] into output CB c_2
            pack_tile(0, tt::CBIndex::c_2);

            // Release the input tile from CB c_0 so reader can reuse the slot
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DEST registers for next iteration
            tile_regs_release();
        }
        // Push the entire block of tiles to the output CB, making them visible to the writer
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

- **API header**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
- **Architecture-specific implementation (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Architecture-specific implementation (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Shared LLK SFPU exp (Wormhole)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
- **Shared LLK SFPU exp (Blackhole)**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`

The call chain is:
1. `exp_tile<approx=true, fast_and_approx=true>(0)` (from `exp.h`)
2. -> `SFPU_TEMPLATE_PARAMS_KERNEL_FN(calculate_exponential, ...)` (macro in `llk_math_eltwise_unary_sfpu_macros.h`)
3. -> `_llk_math_eltwise_unary_sfpu_params_<true>(ckernel::sfpu::calculate_exponential<true, true, ...>, 0, VectorMode::RC, scale)`
4. -> `calculate_exponential<APPROXIMATION_MODE=true, FAST_APPROX=true, ...>()` (in arch-specific `ckernel_sfpu_exp.h`)
5. -> `_calculate_exponential_<true, false, 8, true, false, true>()` (in LLK `ckernel_sfpu_exp.h`)

Similarly, `exp_tile_init<approx=true, fast_and_approx=true>()` calls `_init_exponential_<true, true, 0x3F800000, true>()`.

#### Annotated SFPU Kernel Source -- API Layer (exp.h)

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/exp.h

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"       // Architecture-specific SFPU exp implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"  // Macro wrappers for LLK dispatch
#endif

namespace ckernel {

// Controls whether the fast approximate exponential clamps very negative inputs.
// ClampToNegative (default): Inputs below ~-88.5 are clamped to -88.5. Safer but slightly slower.
// None: No input clamping. Faster, but inputs below ~-88.5 will produce incorrect outputs.
enum class InputClamping : uint8_t {
    ClampToNegative = 1,
    None = 0,
};

// Initializes the SFPU pipeline for the exp operation.
// When approx=true and fast_and_approx=true (the default for exp), this sets up
// LOADMACRO registers, constants (A, B-C, threshold), and replay buffers.
template <
    bool approx = false,
    bool fast_and_approx = true,
    uint32_t scale = 0x3F800000,           // 1.0f in IEEE 754
    InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    // MATH() macro ensures this only executes on the math RISC-V (TRISC_MATH)
    // SFPU_TEMPLATE_INIT_KERNEL dispatches to:
    //   llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, approx>(
    //       sfpu::exp_init<approx, fast_and_approx, scale, clamp_negative>)
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential,
        sfpu::exp_init,
        approx,
        fast_and_approx,
        scale,
        (input_clamping == InputClamping::ClampToNegative)));
}

// Computes element-wise exp on a tile in the DST register buffer.
// DST must be acquired via tile_regs_acquire() before calling.
// Default: approx=false, fast_and_approx=true, iterations=8 (processes 8 sub-tile faces)
template <
    bool approx = false,
    bool fast_and_approx = true,
    bool scale_en = false,
    bool skip_positive_check = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    // SFPU_TEMPLATE_PARAMS_KERNEL_FN dispatches to:
    //   _llk_math_eltwise_unary_sfpu_params_<approx>(
    //       ckernel::sfpu::calculate_exponential<approx, fast_and_approx, DST_ACCUM_MODE, scale_en, ...>,
    //       idst, vector_mode, scale)
    MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(
        calculate_exponential,
        approx,
        fast_and_approx,
        DST_ACCUM_MODE,
        scale_en,
        skip_positive_check,
        (input_clamping == InputClamping::ClampToNegative),
        iterations,
        idst,
        vector_mode,
        scale));
}

}  // namespace ckernel
```

#### Annotated SFPU Kernel Source -- Architecture-Specific Layer (Wormhole B0)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"      // Shared LLK exp implementations
#include "sfpu/ckernel_sfpu_polyval.h"   // Polynomial evaluator
#include "sfpu/ckernel_sfpu_converter.h" // Float bit manipulation utilities
#include "ckernel_sfpu_conversions.h"    // FP format conversion helpers
#include "sfpi.h"                        // SFPI (SFPU Programming Interface) types and intrinsics

namespace ckernel {
namespace sfpu {

// Wrapper for the legacy _sfpu_exp_ function (Horner-form approximation)
sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

// Optimized float-to-int32 conversion for the exp21f algorithm.
// Constraint: 0 <= val < 128.0f
// Unlike generic float_to_int32, this is branch-free because exp21f guarantees
// non-negative inputs with bounded magnitude.
// The input is assumed to already be scaled by 1/2^23, so the output is implicitly
// scaled by 2^23 (saving one SFPADDI instruction).
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);      // Extract unbiased exponent from FP32
    sfpi::vInt man = sfpi::exman8(val);      // Extract 8-bit mantissa with implicit leading 1
    // Shift mantissa left by exponent amount to produce the integer value
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}

// exp_21f algorithm: ~21 fractional bits of accuracy for BF16 output.
// Based on Moroz et al. 2022: "Simple Multiple Precision Algorithms for Exponential Functions"
// Computes exp(x) = 2^(x/ln2) by splitting into integer and fractional parts
// and using a 2nd-degree polynomial to approximate 2^(fractional_part).
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;  // 1/ln(2)
    // Compute x/ln(2) + 127 (the IEEE 754 exponent bias)
    // This maps the input into a range where the integer part becomes the IEEE exponent
    // and the fractional part can be approximated with a polynomial
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);

    // Clamp to [0, 255] to prevent overflow in intermediate computations
    // Values outside this range correspond to exp(x) = 0 or exp(x) = +inf
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);   // xlog2 = max(xlog2, 0)
    sfpi::vec_min_max(xlog2, threshold_high);   // xlog2 = min(xlog2, 255)

    // Convert the floating-point value to a fixed-point integer representation
    // This integer encodes both the exponent (integer part) and mantissa (fractional part)
    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    // Extract the exponent bits (integer part of x/ln2) -- this gives 2^floor(x/ln2)
    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    // Extract the mantissa bits (fractional part of x/ln2) -- in [0, 1)
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    // Convert fractional part to float for polynomial evaluation
    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);

    // 2nd-degree polynomial approximation of 2^(frac) on [0, 2^23]
    // Coefficients chosen to minimize error over the range
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: result = 2^(integer_part) * 2^(fractional_part)
    // setexp replaces the exponent field of frac with exponential_part
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // When DEST is BF16, explicitly round to BF16 to avoid truncation artifacts
        // from the implicit FP32->BF16 conversion in SFPSTORE
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// exp_61f algorithm: higher accuracy (~61 fractional bits) using a 6th-degree polynomial
sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    // Scale down by 2^-23 to normalize fractional part to [0, 1]
    frac = sfpi::addexp(frac, -23);

    // 6th-degree polynomial approximation of 2^x on [0, 1]
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    return y;
}

// Accurate FP32 exp using Cody-Waite range reduction + 7th-order Taylor series
// Target: < 1 ULP error for float32
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;
    constexpr float INV_LN2 = 1.4426950408889634f;

    sfpi::vFloat z = val * INV_LN2;
    sfpi::vInt exp_bits = sfpi::exexp(z);

    // Handle special cases: overflow, underflow, NaN
    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();  // NaN input
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Cody-Waite range reduction: r = x - k*ln(2)
        // Split ln(2) into high and low parts for extended precision
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;  // Optimizes to single SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi;    // Optimizes to single SFPMAD

        // 7th-order Taylor series: exp(r) = 1 + r + r^2/2! + ... + r^7/7!
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,   // 1
            sfpi::vConst1,   // 1
            0.5f,            // 1/2!
            1.0f / 6.0f,     // 1/3!
            1.0f / 24.0f,    // 1/4!
            1.0f / 120.0f,   // 1/5!
            1.0f / 720.0f,   // 1/6!
            1.0f / 5040.0f   // 1/7!
        );

        // Scale by 2^k: add k to the exponent of p
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

// Dispatch: selects exp_21f (BF16) or exp_f32_accurate (FP32) based on dest accumulator mode
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);  // BF16 path: 2nd-degree polynomial
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);  // FP32 path: Cody-Waite + 7th-order Taylor
}

// Main entry point called by exp_tile() via the LLK macro dispatch
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Approximate mode: uses Schraudolph's algorithm via LOADMACRO + replay
        // (dispatched to _calculate_exponential_ which uses hardware macros)
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Precise mode: software loop over 8 SFPU lanes (tile faces)
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // Load from DEST register
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;  // Store back to DEST register
            sfpi::dst_reg++;            // Advance to next SFPU lane (next 4 rows of tile face)
        }
    }
}

// Initialization entry point called by exp_tile_init()
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Annotated SFPU Kernel Source -- Shared LLK Layer (tt_llk)

The `_calculate_exponential_` and `_init_exponential_` functions reside in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`. The Wormhole and Blackhole versions are structurally identical for the `_sfpu_exp_` base function and differ primarily in LOADMACRO encoding and pipeline timing for the fast approximate path.

```cpp
// tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h (key excerpts)

namespace ckernel::sfpu {

// Legacy Horner-form exp approximation (used by non-approximate, non-improved paths)
// Computes exp(x) for x in [-1, 0] using a 2nd-degree Horner polynomial,
// then repeated squaring to handle larger exponents.
sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val) {
    sfpi::vInt exp = exexp(val);         // Extract unbiased exponent
    v_if (exp >= 0) {
        val = setexp(val, 126);          // Clamp exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Horner polynomial: val = val * (val * 0.8373 + 0.863281) + 1.0
    // Approximates exp(x) for x in [-1, 0]
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val = val * tmp + sfpi::vConst1;

    // Repeated squaring: for each bit of the original exponent,
    // square the result. This computes exp(2^n * x) = exp(x)^(2^n)
    v_if (exp >= 0) {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            v_and(exp >= 0);             // Narrow predication: only active lanes continue
            val = val * val;
        }
    }
    v_endif;

    return val;
}

// Main dispatch function for _calculate_exponential_
// When FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE (the default for exp):
//   Uses SFPLOADMACRO-based hardware acceleration with input clamping to -88.5
// When FAST_APPROX && APPROXIMATION_MODE && !CLAMP_NEGATIVE:
//   Uses replay buffer + SFPSHFT2 pipeline (Schraudolph without clamping)
// When !FAST_APPROX || !APPROXIMATION_MODE:
//   Falls back to software loop with _calculate_exponential_piecewise_
template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE>
void _calculate_exponential_(const uint16_t exp_base_scale_factor) {
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE) {
        // Hardware macro path with input clamping:
        // Uses SFPLOADMACRO to execute a fused sequence on each DEST element:
        //   1. Load value from DEST
        //   2. SWAP against threshold (-88.5) -- clamps negative inputs
        //   3. Store sanitized value back to DEST
        // Then a second LOADMACRO sequence:
        //   1. Load sanitized value from DEST
        //   2. MAD: compute A*x + (B-C) where A=256/ln2, B-C=32500.818
        //   3. STOCHRND: round FP32 to unsigned INT16
        //   4. SHFT: shift left by 15 to place into exponent field
        //   5. Store result back to DEST
        // This processes all 16 DEST offsets (8 pairs of even/odd columns)
        // using LREGs 0-3 in rotation across 4 SFPU lanes.

        // [16 SFPLOADMACRO instructions with SFPNOP interleavings]
        // Sanitize pass: 8 LOADMACRO calls (Macro Sequence Register 1)
        // Compute pass: 8 LOADMACRO calls (Macro Sequence Register 0)
        // ... (see full source in the file)
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8) {
        // Replay buffer path for 8 iterations:
        // Configures ADDR_MOD_7 for auto-increment (dest += 2 per LOADMACRO)
        // Replays 16 pre-recorded instruction pairs (LOADMACRO + SFPSHFT2)
        // Then drains the pipeline with 2 final SFPSHFT2 + NOPs
        // Total: ~20 cycles for 8 elements = 2.5 cycles/element
        // ... (see full source in the file)
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 32) {
        // Replay buffer path for 32 iterations (full tile face):
        // Same pattern but with 2 replays of 32-instruction sequences
        // Total: ~68 cycles for 32 elements = 2.125 cycles/element
        // ... (see full source in the file)
    }
    else {
        // Software fallback: iterate over ITERATIONS SFPU lanes
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                val, exp_base_scale_factor);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

// Initialization function
template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale, bool CLAMP_NEGATIVE>
inline void _init_exponential_() {
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE) {
        // Sets up:
        //   LREG[14] = -88.5 (clamping threshold)
        //   LREG[12] = A = 256/ln2 = 369.33 (multiply factor)
        //   LREG[13] = B-C = 32500.818 (bias minus correction)
        // Programs macro instructions:
        //   Macro 4: SWAP instruction (compare against threshold)
        //   Macro 5: MAD (A*x + (B-C))
        //   Macro 6: STOCHRND (FP32 -> unsigned INT16)
        //   Macro 7: SHFT (shift left by 15)
        // Programs macro sequences:
        //   Sequence 0: LD, MAD, ROUND, SHIFT, STORE (compute path)
        //   Sequence 1: LD, SWAP, STORE (sanitize path)
        // ... (see full source in the file)
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE) {
        // Non-clamping fast approximate path:
        // Sets up LREG[12]=A, LREG[13]=B-C, LREG[14]=15 (shift amount)
        // Programs macro instructions for MAD, STOCHRND, SETSGN
        // Programs Macro Sequence Register 0
        // Records 32 instructions into replay buffer
        // ... (see full source in the file)
    }
    else if constexpr (APPROXIMATION_MODE) {
        // Slow approximate path: sets up programmable float constants
        sfpi::vConstFloatPrgm0 = 1.442695f;  // 1/ln2
        sfpi::vConstFloatPrgm1 = sfpi::s2vFloat16b(p_exp::C23_73);
        sfpi::vConstFloatPrgm2 = sfpi::s2vFloat16b(p_exp::ADJ_EXP);
    }
    else {
        // Precise mode: initialize reciprocal for negative input handling
        _init_sfpu_reciprocal_<false>();
    }
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::exexp(val)` | Extracts the unbiased exponent field from an FP32 value in an LREG |
| `sfpi::exexp_nodebias(val)` | Extracts the raw (biased) exponent field without subtracting the IEEE bias |
| `sfpi::exman8(val)` | Extracts the 8-bit mantissa with the implicit leading 1 bit |
| `sfpi::exman9(val)` | Extracts the 9-bit mantissa with the implicit leading 1 bit |
| `sfpi::setexp(val, exp)` | Replaces the exponent field of a float with a new value |
| `sfpi::setsgn(val, sign)` | Forces the sign bit of a float to the given value |
| `sfpi::shft(val, amount)` | Shifts a value left or right by the given amount |
| `sfpi::addexp(val, delta)` | Adds a delta to the exponent field (equivalent to multiply by 2^delta) |
| `sfpi::int32_to_float(val, mode)` | Converts an integer to a floating-point value |
| `sfpi::float_to_fp16b(val, mode)` | Converts FP32 to BF16 with round-to-nearest-even |
| `sfpi::vec_min_max(a, b)` | Performs element-wise min/max swap: after call, a=min(a,b), b=max(a,b) |
| `sfpi::reinterpret<T>(val)` | Reinterprets bits between vFloat/vInt/vUInt without conversion |
| `sfpi::s2vFloat16b(val)` | Converts a scalar to a vFloat broadcast value in BF16 format |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Evaluates polynomial c0 + c1*x + c2*x^2 + ... using Horner's method |
| `sfpi::dst_reg[0]` | Reads/writes the current DEST register element (SFPLOAD/SFPSTORE) |
| `sfpi::dst_reg++` | Advances the DEST register pointer to the next SFPU lane |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (sets condition codes per lane) |
| `v_and(cond)` | Narrows the active lane mask (AND with new condition) |
| `TTI_SFPLOADMACRO(lreg, seq, rows, offset)` | Issues a LOADMACRO instruction that executes a pre-programmed macro sequence |
| `TTI_SFPMAD(a, b, c, d, mod)` | Multiply-accumulate: d = a * b + c (also used for backdoor macro loading) |
| `TTI_SFP_STOCH_RND(...)` | Stochastic rounding: converts FP32 to integer format |
| `TTI_SFPSHFT(imm, src, dst, mod)` | Shift operation (also used for backdoor macro loading) |
| `TTI_SFPSHFT2(a, b, c, mod)` | Shift-2 operation used in the replay buffer path |
| `TTI_SFPCONFIG(val, dest, mode)` | Configures SFPU registers (stores constants, programs macro sequences) |
| `TTI_SFPLOADI(lreg, mod, imm)` | Loads an immediate value into an LREG |
| `TTI_SFPNOP` | No-operation (pipeline timing placeholder) |
| `TTI_SFPSETSGN(imm, src_c, dst, mod)` | Sets the sign bit of a value |
| `TTI_SFPSWAP(imm, src_c, dst, mod)` | Swaps values based on comparison (used for clamping) |
| `lltt::replay(start, count)` | Replays pre-recorded instructions from the replay buffer |
| `lltt::record(start, count)` | Begins recording instructions into the replay buffer |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers [0..15]** | Hold tile face data (16 elements per face, 8 faces per tile in RC mode). The SFPU reads from and writes to DEST via `sfpi::dst_reg`. |
| **LREG[0..3]** | Working registers used by LOADMACRO for loading values from DEST and intermediate computations. Rotated across 4 SFPU lanes. |
| **LREG[12]** | Constant A = 256/ln(2) = 369.33 (fast approx multiply factor) |
| **LREG[13]** | Constant B-C = 32500.818 (fast approx bias-correction) |
| **LREG[14]** | Threshold = -88.5 (clamping, in CLAMP_NEGATIVE path) or shift amount = 15 (in non-clamping path) |
| **LREG[16]** | Staging register (used by SETSGN in non-clamping fast approx to avoid write port conflict) |
| **Macro Instruction Registers [4..7]** | Programmable macro instructions: 4=SWAP, 5=MAD, 6=STOCHRND, 7=SHFT (or SETSGN in non-clamping path) |
| **Macro Sequence Register 0** | Compute sequence: LD->MAD->ROUND->SHIFT->STORE |
| **Macro Sequence Register 1** | Sanitize sequence: LD->SWAP->STORE (only in CLAMP_NEGATIVE path) |
| **vConstFloatPrgm0** | 1/ln(2) = 1.442695 (used in slow approximate path) |
| **vConstFloatPrgm1** | C23_73 constant (slow approximate path) |
| **vConstFloatPrgm2** | ADJ_EXP adjustment (slow approximate path) |
| **Replay buffer** | Stores 32 instructions for the non-clamping fast approx path |

#### SFPU Execution Flow

**Default path (approx=true, fast_and_approx=true, clamp_negative=true):**

1. **Initialization** (`exp_tile_init` -> `_init_exponential_`):
   - Load constants into LREGs: threshold (-88.5) into LREG[14], multiply factor A into LREG[12], bias B-C into LREG[13].
   - Program macro instructions into slots 4-7: SWAP (slot 4), MAD (slot 5), STOCHRND (slot 6), SHFT (slot 7).
   - Program macro sequences: Sequence 1 = sanitize (LD, SWAP, STORE), Sequence 0 = compute (LD, MAD, ROUND, SHIFT, STORE).
   - Reset LOADMACRO configuration.

2. **Tile acquisition** (in compute kernel):
   - `cb_wait_front(c_0, 1)` blocks until a tile is available.
   - `copy_tile(c_0, 0, 0)` unpacks the tile from CB c_0 into DEST register 0.

3. **SFPU computation** (`exp_tile` -> `calculate_exponential` -> `_calculate_exponential_`):
   - **Sanitize pass**: 8 SFPLOADMACRO instructions (using Macro Sequence Register 1) iterate over all 16 DEST offsets in pairs (even/odd columns). Each LOADMACRO loads a value from DEST, compares it against -88.5 via SWAP, and stores the clamped value back. SFPNOP instructions provide pipeline timing.
   - **Compute pass**: 8 SFPLOADMACRO instructions (using Macro Sequence Register 0) iterate over the same 16 DEST offsets. Each LOADMACRO executes: (a) LD - load sanitized value, (b) MAD - compute A*x + (B-C), (c) STOCHRND - round to unsigned INT16, (d) SHFT - shift left by 15 to place into FP32 exponent field, (e) STORE - write result back to DEST.

4. **Result packing** (in compute kernel):
   - `tile_regs_commit()` signals that DEST writes are complete.
   - `tile_regs_wait()` waits for the packer to be ready.
   - `pack_tile(0, c_2)` packs DEST[0] into CB c_2.
   - `cb_pop_front(c_0, 1)` frees the input tile slot.

5. **Looping**: Steps 2-4 repeat for each tile assigned to this core.

**Alternative paths:**

- **approx=true, fast_and_approx=true, clamp_negative=false**: Uses the replay buffer path. Init pre-records 32 LOADMACRO+SFPSHFT2 instruction pairs. At runtime, `lltt::replay()` replays them with auto-incrementing DEST offsets. This path uses SETSGN instead of SWAP for sign handling and does not clamp negative inputs.
- **approx=false (precise mode)**: Software loop over 8 SFPU lanes. For BF16 DEST: uses `_sfpu_exp_21f_` (Moroz 2nd-degree polynomial). For FP32 DEST: uses `_sfpu_exp_f32_accurate_` (Cody-Waite range reduction + 7th-order Taylor series).

#### SFPU Configuration

| Configuration | Value | Description |
|---------------|-------|-------------|
| **Math fidelity** | HiFi4 | Highest fidelity (set in ComputeConfig) |
| **Math approx mode** | false | `get_op_approx_mode(EXP)` returns false (the switch has only a `default: return false`) |
| **Per-op approx mode** | true (by default) | Controlled via template parameter `param0` on `exp_tile_init<P>()` and `exp_tile<P>(0)`. When the op is created from `string_to_unary_with_param("exp")`, param0 = `true`. |
| **fast_and_approx** | true (default) | Second template parameter in exp_tile, always true by default |
| **InputClamping** | ClampToNegative (default) | Clamps inputs below -88.5 to prevent incorrect outputs |
| **Iterations** | 8 (default) | Number of SFPU lanes to process per tile face (standard RC mode) |
| **fp32_dest_acc_en** | From operation attributes | When true, selects the `_sfpu_exp_f32_accurate_` path in non-approximate mode |
| **unpack_to_dest_mode** | Default (or UnpackToDestFp32 if preserve_fp32_precision) | Controls how unpacker writes data into DEST |

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The `_sfpu_exp_` base function (Horner form) and the `_sfpu_exp_21f_`/`_sfpu_exp_61f_`/`_sfpu_exp_f32_accurate_` improved functions are **identical** between Wormhole and Blackhole architectures. They share the same source in `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`.
- **LOADMACRO and replay buffer paths**: The `_calculate_exponential_` and `_init_exponential_` functions in the shared LLK layer (`tt_metal/third_party/tt_llk/`) are also identical between architectures for the fast approximate path. Both use the same Schraudolph-based algorithm with SFPLOADMACRO sequences.
- **SFPMAD optimization**: The Cody-Waite implementation notes that on Wormhole, SFPMAD can only do `VD = VA * VB + VC`, so constants are negated to avoid extra subtraction instructions. On Blackhole, `SFFPMAD` has `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers, but the code uses negated constants anyway for consistency.
- **Blackhole includes**: The Blackhole version additionally includes `ckernel_addrmod.h` and `ckernel_ops.h` headers not present in the Wormhole version.

## Implementation Notes

1. **Default approximation mode**: When `exp` is created via the standard Python API path (`string_to_unary_with_param("exp")`), `param0 = true` (approximate mode). This means the fast LOADMACRO-based Schraudolph algorithm is used by default, trading some accuracy for significant performance gains (~2.5 cycles per element).

2. **Three accuracy tiers**:
   - **Fast approximate** (approx=true, fast_and_approx=true): Schraudolph algorithm via hardware LOADMACRO. ~2.5 cycles/element for 8 iterations. Valid range: [-88.5, +inf) with clamping.
   - **Slow approximate** (approx=true, fast_and_approx=false): Software implementation using FP32-to-fixed-point conversion. More accurate than fast approximate but slower.
   - **Precise** (approx=false): Two sub-paths depending on DEST format:
     - BF16 DEST: Moroz exp_21f (2nd-degree polynomial, ~21 bits accuracy)
     - FP32 DEST: Cody-Waite + 7th-order Taylor series (< 1 ULP for float32)

3. **Shared program factory**: The `UnaryProgramFactory` is shared by all unary SFPU operations. The exp-specific behavior is entirely injected via preprocessor defines. This makes it easy to add new unary operations without modifying the factory.

4. **No temporary CB needed**: Unlike HARDSHRINK, CBRT, or LOGIT, the exp operation does not require the temporary circular buffer CB c_1.

5. **Packed scalars unused**: The exp operation does not use `packed_scalar1` or `packed_scalar2` runtime args (they are set to 0). These are only used by operations like HARDSHRINK and WHERE_TSS.

6. **Program caching**: The factory supports `override_runtime_arguments` for efficient program reuse. Only buffer addresses need updating between invocations; the kernel binaries and core distribution remain cached.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary SFPU operation program factory work? Specifically, how does the exp operation flow through the unary_program_factory.cpp?"
   **Reason**: Needed to understand the overall program factory structure and how exp fits into it.
   **Key Findings**: Confirmed the 3-kernel architecture (reader/compute/writer), CB configuration, core distribution via `split_work_to_cores`, and that the compute kernel path for exp defaults to `eltwise_sfpu.cpp`.

2. **Query**: "What compute kernel file is used for unary SFPU operations like exp? How does the compute kernel dispatch to the SFPU?"
   **Reason**: Needed to trace the kernel file path and understand the SFPU dispatch mechanism.
   **Key Findings**: Confirmed `eltwise_sfpu.cpp` is the compute kernel, `SFPU_OP_EXP_INCLUDE` enables the exp header, and `SFPU_OP_CHAIN_0` macro expands to `exp_tile_init(); exp_tile(0)`. Also identified `ckernel_sfpu_exp.h` as the SFPU implementation file.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how exp-specific defines are generated.
   **Key Information**: `get_macro_definition(EXP)` returns `"SFPU_OP_EXP_INCLUDE"`. `get_op_init_and_func` returns `exp_tile_init<P>()` and `exp_tile<P>(idst)` with the approx param. `get_op_approx_mode` returns false for all ops. `get_compute_kernel_path` defaults to `eltwise_sfpu.cpp` for exp.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Needed to confirm how `SFPU_OP_EXP_INCLUDE` triggers the exp header inclusion.
   **Key Information**: `#if SFPU_OP_EXP_INCLUDE` -> `#include "api/compute/eltwise_unary/exp.h"`.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand the `SFPU_TEMPLATE_INIT_KERNEL` and `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macro dispatch.
   **Key Information**: These macros expand to `llk_math_eltwise_unary_sfpu_init` and `_llk_math_eltwise_unary_sfpu_params_` calls respectively, passing through all template parameters to the underlying SFPU functions.

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Core SFPU exp implementation with LOADMACRO-based hardware acceleration.
   **Key Information**: Contains `_calculate_exponential_` (the main dispatch with 4 code paths) and `_init_exponential_` (hardware macro setup). The fast approximate path uses Schraudolph's algorithm with SFPLOADMACRO for ~2.5 cycles/element throughput.
