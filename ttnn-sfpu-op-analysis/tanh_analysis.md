# TANH Operation Analysis

## Operation Overview

**Operation**: `ttnn::tanh`
**Type**: Unary element-wise SFPU operation
**UnaryOpType enum**: `UnaryOpType::TANH`
**Mathematical definition**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
**Output range**: `[-1, 1]`

The TANH operation computes the hyperbolic tangent of each element in the input tensor. It is a standard activation function used widely in neural networks. On Tenstorrent hardware, it is implemented as a unary SFPU operation dispatched through the shared `UnaryProgramFactory`, which provides a common program factory for all standard unary element-wise operations.

---

## Program Factory

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Structure

The TANH operation uses the shared `UnaryProgramFactory` (and optionally `UnarySubCoreGridProgramFactory` for sub-core-grid dispatch). The factory is parameterized by the `UnaryParams` struct which contains the operation chain (`op_chain`) identifying the specific unary operation type.

Two factory variants exist:
1. **`UnaryProgramFactory`** -- Standard variant that uses `split_work_to_cores` to distribute tiles across the full compute grid, creating two core groups to handle uneven tile distribution.
2. **`UnarySubCoreGridProgramFactory`** -- Variant that operates on a caller-specified sub-core grid rather than the full device compute grid.

### Header File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp`

```cpp
struct UnaryProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id;
        tt::tt_metal::KernelHandle unary_writer_kernel_id;
        uint32_t num_cores;
        uint32_t num_cores_y;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UnaryParams& operation_attributes,
        const UnaryInputs& tensor_args,
        Tensor& output);
};
```

---

## Circular Buffer Configuration

| CB Index | Variable | Data Format | Page Count | Purpose |
|----------|----------|-------------|------------|---------|
| `c_0` | `src0_cb_index` | Input tensor data format | 2 | Input tiles from reader |
| `c_2` | `output_cb_index` | Output tensor data format | 2 | Output tiles to writer |

Notes:
- CB `c_1` (tmp0) is only allocated for `HARDSHRINK` and `LOGIT` operations, not for TANH.
- Page size is `single_tile_size` for tiled layout, or `buffer->page_size()` for row-major layout.
- Double-buffering (2 pages) is used for both input and output CBs, allowing one tile to be processed while the next is being read/written.

---

## Work Distribution

The factory uses `tt::tt_metal::split_work_to_cores` to distribute tiles (or pages for row-major) across all available compute cores:

```cpp
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_pages_per_core_group_1, num_pages_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
```

- **Core group 1**: Gets `num_pages_per_core_group_1` tiles per core (the larger share)
- **Core group 2**: Gets `num_pages_per_core_group_2` tiles per core (handles remainder)
- Separate compute kernel instances are created for each core group with different compile-time `per_core_block_cnt` arguments.
- Both groups run the same compute kernel source, differing only in tile count.

---

## Kernel Registrations

### Reader Kernel
- **Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Config**: `ReaderDataMovementConfig` with `TensorAccessorArgs` from the source buffer
- **Runtime args**: `{src_buffer->address(), num_pages_per_core, num_pages_written}`

### Writer Kernel
- **Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Config**: `WriterDataMovementConfig` with compile-time args `{output_cb_index}` plus `TensorAccessorArgs`
- **Runtime args**: `{dst_buffer->address(), num_pages_per_core, num_pages_written}`

### Compute Kernel
- **Path resolution**: `get_compute_kernel_path(UnaryOpType::TANH, input.dtype())` returns `"eltwise_sfpu.cpp"` (the default case)
- **Full path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Config**: `ComputeConfig` with:
  - `math_fidelity = MathFidelity::HiFi4`
  - `fp32_dest_acc_en` from operation params
  - `math_approx_mode = false` (TANH returns false from `get_op_approx_mode`)
  - Defines include the SFPU op chain and `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE=1`
- **Compile-time args**: `{per_core_block_cnt, 1}` (block count, block size of 1 tile)
- **Runtime args**: `{packed_scalar1=0, packed_scalar2=0}` (TANH uses no scalar parameters)

### SFPU Op Chain Defines

For TANH with `param0 = false` (approximation mode off), the defines generated are:

```
SFPU_OP_CHAIN_0_INIT_0 = "tanh_tile_init<0u>();"
SFPU_OP_CHAIN_0_FUNC_0 = "tanh_tile<0u>(0);"
SFPU_OP_CHAIN_0 = "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"
SFPU_OP_COMPUTE_KERNEL_API_INCLUDE = 1
```

The `<0u>` template argument corresponds to `fast_and_approx = false`, meaning the non-LUT polynomial/sigmoid-based implementation is used.

---

## Kernel Implementations

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
#include "api/compute/common.h"           // Common compute API (init_sfpu, tile_regs_acquire/commit/wait/release)
#include "api/compute/tile_move_copy.h"   // copy_tile API for moving data from CB to DST registers
#include "api/compute/eltwise_unary/eltwise_unary.h"  // Base unary eltwise declarations
#include "api/compute/eltwise_unary/sfpu_split_includes.h" // Conditional includes based on SFPU_OP_*_INCLUDE defines
#include "api/compute/eltwise_unary/trigonometry.h"  // Trigonometric SFPU ops (also includes tanh via compute_kernel_api.h)
#include "api/compute/mul_int_sfpu.h"     // Integer multiply SFPU operations
#include "api/compute/eltwise_unary/rpow.h"  // Reciprocal power operations
#include "api/compute/eltwise_unary/rdiv.h"  // Reciprocal division operations
#include "api/compute/eltwise_unary/fill.h"  // Fill operations

void kernel_main() {
    // Compile-time arguments set by the program factory
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    // Initialize SFPU for unary operation: sets up unpacker for CB c_0, packer for CB c_2
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    // Outer loop: iterate over tile blocks assigned to this core
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in the output CB for the entire block (1 tile)
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        // Inner loop: iterate over tiles within each block
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST register file for math operations
            tile_regs_acquire();

            // Wait for one tile to be available in input CB (pushed by reader kernel)
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from input CB (c_0, tile 0) into DST register 0
            // This performs the unpack operation: data moves from SRAM through unpacker into DST
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // Execute the SFPU operation chain -- for TANH this expands to:
            //   tanh_tile_init<0u>();  -- one-time init (called each tile but idempotent after first)
            //   tanh_tile<0u>(0);      -- compute tanh on DST register 0
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are ready for packing (math -> pack handoff)
            tile_regs_commit();

            // Wait for pack engine to be ready to consume DST register data
            tile_regs_wait();

            // Pack tile from DST register 0 into output CB c_2
            pack_tile(0, tt::CBIndex::c_2);

            // Release the input tile from CB c_0 (allows reader to push more data)
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers for next iteration
            tile_regs_release();
        }
        // Push the completed block to the output CB (allows writer to consume)
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

### Reader Kernel

#### Reader Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments set per-core by the program factory
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // Base address of source tensor in DRAM
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core must read
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page ID for this core

    // Compile-time TensorAccessor configuration (interleaved addressing, bank mapping)
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;  // Input circular buffer index (c_0)

    // Page size comes from the CB configuration, works for both tile and row-major layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;

    // Construct TensorAccessor for NoC read operations
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Sequential read loop: read one tile at a time from DRAM into CB c_0
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);          // Wait for space in CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0); // Get L1 write pointer
        noc_async_read_page(i, s, l1_write_addr);     // Initiate async NoC read from DRAM
        noc_async_read_barrier();                      // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);              // Signal tile is ready for compute
    }
}
```

### Writer Kernel

#### Writer Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

#### Annotated Writer Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // Base address of output tensor in DRAM
    const uint32_t num_pages = get_arg_val<uint32_t>(1);    // Number of pages this core must write
    const uint32_t start_id = get_arg_val<uint32_t>(2);     // Starting page ID for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0); // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // For sharded output, just wait for all tiles to arrive in the output CB
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    // Sequential write loop: write one tile at a time from CB c_2 to DRAM
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);              // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out); // Get L1 read pointer
        noc_async_write_page(i, s, l1_read_addr);       // Initiate async NoC write to DRAM
        noc_async_writes_flushed();                     // Flush write (non-blocking barrier)
        cb_pop_front(cb_id_out, onepage);                // Release tile from CB
    }
    noc_async_write_barrier();  // Final barrier: ensure all writes complete before kernel exits
#endif
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`
(Identical implementation exists at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"   // PolynomialEvaluator::eval() -- Horner's method polynomial evaluation
#include "ckernel_sfpu_sigmoid.h"        // _sfpu_sigmoid_<> -- used by the fp32 accurate path

namespace ckernel::sfpu {

/*
 * Accurate tanh for fp32 using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
 * For small |x| < 0.6, uses minimax polynomial for better accuracy
 *
 * This function implements a piecewise approach:
 *   - For |x| < 0.6: minimax polynomial optimized by Sollya for relative error
 *   - For |x| >= 0.6: identity tanh(x) = 2*sigmoid(2x) - 1 using the sigmoid SFPU kernel
 *
 * The polynomial approach for small x avoids catastrophic cancellation that occurs
 * when computing 2*sigmoid(2x) - 1 near zero (where sigmoid(2x) is near 0.5).
 *
 * Target accuracy: < 5 ULP for float32 (0.5 ULP for bfloat16)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;  // Initialize result to 0.0

    constexpr float POLYNOMIAL_THRESHOLD = 0.6f;  // Boundary between poly and sigmoid paths

    sfpi::vFloat abs_val = sfpi::abs(val);  // SFPABS instruction: clear sign bit

    // SFPI conditional: per-lane predication via condition codes
    // Lanes where |x| < 0.6 take the polynomial path
    v_if(abs_val < POLYNOMIAL_THRESHOLD) {
        // Minimax polynomial for tanh(x)/x on [-0.6, 0.6], then multiply by x
        // Polynomial: p(x^2) where p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
        // Result = x * p(x^2), which gives tanh(x) directly
        // Coefficients found with Sollya:
        //   fpminimax(tanh(x)/x, [|0,2,4,6,8|], [|single...|], [-0.6;-2^(-40)]+[2^(-40);0.6], relative)
        sfpi::vFloat x2 = val * val;  // SFPMAD: x * x + 0

        // Horner evaluation of degree-4 polynomial in x^2
        // Each step is an SFPMAD: coeff * x2 + next_coeff
        sfpi::vFloat p = PolynomialEvaluator::eval(
            x2,
            0.999999940395355224609375f,           // c0: ~1.0 (tanh(x) ~ x for small x)
            -0.33332359790802001953125f,           // c1: ~-1/3 (from Taylor series: -x^3/3)
            0.13310669362545013427734375f,         // c2: ~2/15
            -5.21197654306888580322265625e-2f,     // c3
            1.5497927553951740264892578125e-2f);   // c4

        result = val * p;  // SFPMAD: val * p + 0
    }
    v_else {
        // Normal region: use the identity tanh(x) = 2*sigmoid(2x) - 1
        // This is accurate for |x| >= 0.6 because sigmoid is well-behaved there
        sfpi::vFloat two_x = 2.f * val;                           // SFPMAD: 2.0 * val + 0
        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(two_x); // Calls sigmoid kernel (exp + reciprocal)

        // Compute 2*sigmoid(2x) - 1
        result = 2.f * sig - sfpi::vConst1;  // SFPMAD: 2.0 * sig + (-1.0)
    }
    v_endif;

    return result;
}

/*
 * Continued fraction approximation of tanh (currently unused in the main path).
 * Based on Lambert's continued fraction formula.
 * Kept for reference/alternative implementations.
 */
template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_continued_fraction_(sfpi::vFloat val) {
    sfpi::vFloat x = sfpi::abs(val);  // SFPABS: work with positive values

    // Compute numerator and denominator of Pade-like rational approximation
    // Using Horner's method for efficient evaluation
    sfpi::vFloat x2 = x * x;  // x^2
    // numerator = x * (135135 + x^2 * (17326 + x^2 * (378 + x^2)))
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));

    // denominator = 135135 + x^2 * (62370 + x^2 * (3150 + 28 * x^2))
    sfpi::vFloat denominator = PolynomialEvaluator::eval(x2, 135135.f, 62370.f, 3150.f, 28.f);

    // Division via reciprocal: result = numerator * (1/denominator)
    // _sfpu_reciprocal_<2> uses 2 Newton-Raphson iterations for fp32 precision
    sfpi::vFloat result = numerator * ckernel::sfpu::_sfpu_reciprocal_<2>(denominator);

    // Clamp to [0, 1] since tanh is bounded -- use SFPSWAP with VEC_MIN_MAX mode
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);  // result = min(result, 1.0)

    result = sfpi::setsgn(result, val);  // SFPSETSGN: restore original sign

    return result;
}

/*
 * Polynomial approximation of tanh for bfloat16/fp16b precision.
 * This is the DEFAULT path when APPROXIMATION_MODE=false and fp32 dest accumulation is disabled.
 *
 * Uses a degree-5 polynomial found via Sollya optimization, evaluated on |x|.
 * Three of the six coefficients are stored in programmable constant registers
 * (vConstFloatPrgm0/1/2) to reduce instruction count.
 */
template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    sfpi::vFloat val = sfpi::abs(x);  // SFPABS: compute on positive values

    // Degree-5 polynomial in |x|: tanh(|x|) ~ sum(c_i * |x|^i, i=0..5)
    // vConst0 = 0.0 (c0 coefficient -- tanh(0) = 0)
    // Remaining coefficients:
    //   c1 = 0.999004364013671875        (loaded as immediate)
    //   c2 = 3.0897438526153564453125e-2 (loaded as immediate)
    //   c3 = -0.4890659749507904052734375 (loaded as immediate)
    //   c4 = 0.281917631626129150390625   (from vConstFloatPrgm2, set in tanh_init)
    //   c5 = -6.6649019718170166015625e-2 (from vConstFloatPrgm1, set in tanh_init)
    //   c6 = 5.876733921468257904052734375e-3 (from vConstFloatPrgm0, set in tanh_init)
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,               // c0 = 0.0
        0.999004364013671875,         // c1 ~ 1.0
        3.0897438526153564453125e-2,  // c2
        -0.4890659749507904052734375, // c3
        sfpi::vConstFloatPrgm2,       // c4 = 0.281917... (from programmable constant reg)
        sfpi::vConstFloatPrgm1,       // c5 = -0.06664... (from programmable constant reg)
        sfpi::vConstFloatPrgm0);      // c6 = 0.005876... (from programmable constant reg)

    // Clamp to [0, 1] via SFPSWAP VEC_MIN_MAX
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, x);  // SFPSETSGN: restore original sign

    return result;
}

/*
 * Main entry point: calculate_tanh
 *
 * Template parameters:
 *   APPROXIMATION_MODE: if true, uses LUT-based SFPLUT instruction (fast, less accurate)
 *   is_fp32_dest_acc_en: if true, uses fp32 sigmoid-based accurate path
 *   ITERATIONS: number of SFPU rows to process (always 8 for a full 32x32 tile face)
 *
 * Each tile has 4 faces (16x16 sub-tiles). The LLK infrastructure calls this function
 * once per face, with ITERATIONS=8. Each iteration processes one row of 16 elements
 * (one SFPU lane processes 2 elements, and there are 8 lanes per SFPU slice).
 * Total: 4 faces x 8 iterations x 2 elements x 8 lanes = 512 elements = 32*16 (one half of tile).
 * The full tile (32x32 = 1024 elements) is processed across 2 calls of the 4-face loop.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // ---- FAST/APPROXIMATE PATH: LUT-based ----
        // Load LUT coefficients from LRegs into local variables to avoid repeated memory access
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];  // Coefficients for |x| < 1.0
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];  // Coefficients for 1.0 <= |x| < 2.0
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];  // Coefficients for |x| >= 2.0

        // Unroll hint for the compiler -- process 8 rows per face
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];       // Load value from DST register
            val = sfpi::lut(val, l0, l1, l2);          // SFPLUT instruction: piecewise linear approx
            sfpi::dst_reg[0] = val;                    // Store result back to DST register

            sfpi::dst_reg++;  // Advance DST register pointer to next row
        }

        // Restore LReg values (SFPLUT may modify them during execution)
        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;

    } else {  // APPROXIMATION_MODE is false

        // ---- ACCURATE PATH ----
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];       // Load from DST register
            sfpi::vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                // FP32 path: use sigmoid-based accurate computation
                // tanh(x) = 2*sigmoid(2x) - 1 for |x| >= 0.6
                // minimax polynomial for |x| < 0.6
                result = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(val);
            } else {
                // BFloat16 path: use Sollya-optimized degree-5 polynomial
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
                // Convert result back to fp16b format (truncate mantissa)
                // This uses SFPSTOCHRND instruction with FP32_TO_FP16B mode
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }

            sfpi::dst_reg[0] = result;  // Store result to DST register
            sfpi::dst_reg++;            // Advance to next row
        }
    }
}

/*
 * Initialization function: tanh_init
 *
 * Called once before processing tiles. Sets up constants needed by the compute path.
 *
 * For APPROXIMATION_MODE (LUT path):
 *   Loads 16-bit packed LUT coefficients into LRegs 0-2 via _sfpu_load_imm16_.
 *   These encode piecewise linear coefficients for three input ranges.
 *
 * For non-approximation with fp32:
 *   Initializes the sigmoid kernel (which needs reciprocal initialization).
 *
 * For non-approximation with bfloat16:
 *   Loads polynomial coefficients into programmable constant registers
 *   (vConstFloatPrgm0/1/2), which are special SFPU registers accessible
 *   via SFPCONFIG instruction.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        // LUT mode: load piecewise linear coefficients
        // imm0 = 0x1DFF: for |x| < 1.0 -> slope=0.90625, intercept=~0
        // imm1 = 0x481A: for 1.0 <= |x| < 2.0 -> slope=0.09375, intercept=0.8125
        // imm2 = 0xFF00: for |x| >= 2.0 -> slope=0, intercept=1.0 (saturation)
        uint imm0 = 0x1DFF;
        uint imm1 = 0x481A;
        uint imm2 = 0xFF00;
        _sfpu_load_imm16_(0, imm0);  // Load into LReg0
        _sfpu_load_imm16_(1, imm1);  // Load into LReg1
        _sfpu_load_imm16_(2, imm2);  // Load into LReg2
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            // FP32 accurate path: initialize sigmoid (which initializes reciprocal)
            sigmoid_init<false>();
        } else {
            // BFloat16 polynomial path: store coefficients in programmable constant registers
            // These are accessed via SFPCONFIG writes, shared across all lanes in an SFPU slice
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;    // c6 coefficient
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;         // c5 coefficient
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;           // c4 coefficient
        }
    }
}

}  // namespace ckernel::sfpu
```

#### LLK API Wrapper

The LLK API layer provides the bridge between the compute kernel API (`tanh_tile`/`tanh_tile_init`) and the SFPU kernel functions. Located at:
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h`

```cpp
#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    // Initialize SFPU state machine, then call tanh-specific init
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>(
        sfpu::tanh_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // Dispatch to the SFPU params infrastructure which handles:
    // 1. Setting the destination register address
    // 2. Iterating over tile faces (4 faces per tile in RC mode)
    // 3. Calling calculate_tanh<APPROXIMATE, is_fp32_dest_acc_en, 8> for each face
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_tanh<APPROXIMATE, is_fp32_dest_acc_en, 8>,
        dst_index, vector_mode);
}

}  // namespace ckernel
```

#### Sigmoid Dependency (for FP32 accurate path)

The FP32 accurate tanh path delegates to `_sfpu_sigmoid_`, located at:
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`

```cpp
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);  // High-accuracy exp for fp32
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);       // ~1 ULP accuracy exp for bfloat16
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;  // 1 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);  // 2 Newton-Raphson iterations
    } else {
        result = _sfpu_reciprocal_<1>(denominator);  // 1 Newton-Raphson iteration
    }

    return result;
}
```

#### SFPU Instructions Used

| SFPU Instruction/Intrinsic | Mapped From | Description |
|---------------------------|-------------|-------------|
| `SFPLUT` (opcode 0x73) | `sfpi::lut(val, l0, l1, l2)` | Piecewise linear LUT interpolation. Used in approximation mode. Selects coefficients from LRegs based on input range (<1.0, <2.0, >=2.0) and computes `A * abs(x) + B`. |
| `SFPMAD` (opcode 0x84) | `*`, `+`, `-` on `vFloat` | Fused multiply-add: `(A * B) + C`. Core arithmetic instruction used extensively for polynomial evaluation (Horner's method), scaling, and offset computation. |
| `SFPABS` | `sfpi::abs(val)` | Clears the sign bit of a floating-point value. Used to work with `|x|` in all non-LUT paths. |
| `SFPSETSGN` | `sfpi::setsgn(result, val)` | Copies the sign bit from one value to another. Used to restore the sign of the result after computing on `|x|`. |
| `SFPSWAP` (VEC_MIN_MAX mode) | `sfpi::vec_min_max(result, threshold)` | Per-lane min/max operation. Used to clamp polynomial results to [0, 1] before sign restoration. |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load value from DST register into SFPU local register. |
| `SFPSTORE` | `sfpi::dst_reg[0] = result` (write) | Store value from SFPU local register back to DST register. |
| `SFPSTOCHRND` (FP32_TO_FP16B) | `sfpi::float_to_fp16b(result, 0)` | Convert FP32 to FP16b (bfloat16) by truncating lower 16 mantissa bits. Used in the bfloat16 polynomial path. |
| `SFPCONFIG` | `sfpi::vConstFloatPrgm0 = ...` | Write to programmable constant registers. Used in `tanh_init` to set polynomial coefficients. |
| `SFPLOADI` | `_sfpu_load_imm16_(reg, imm)` | Load 16-bit immediate into an LReg. Used in LUT mode initialization. |
| `SFPSETCC` / `SFPENCC` | `v_if` / `v_else` / `v_endif` | Conditional execution via per-lane predication. Used in the fp32 accurate path to select between polynomial and sigmoid paths based on `|x| < 0.6`. |

#### SFPU Register Usage

**DST Registers (Destination Register File)**:
- `dst_reg[0]` is used for both input and output of each SFPU row computation
- `dst_reg++` advances the pointer to the next row in the tile face
- 8 iterations process all rows in one face of a 32x32 tile

**Local Registers (LRegs)**:
- **LReg0, LReg1, LReg2**: Used in approximation mode to hold packed LUT coefficients for three input ranges
- **LReg3**: Implicitly used by `SFPLUT` as the input value register

**Programmable Constant Registers**:
- **vConstFloatPrgm0**: Stores c6 coefficient (5.877e-3) for the bfloat16 polynomial path
- **vConstFloatPrgm1**: Stores c5 coefficient (-6.665e-2) for the bfloat16 polynomial path
- **vConstFloatPrgm2**: Stores c4 coefficient (0.2819) for the bfloat16 polynomial path

**Fixed Constant Registers**:
- **vConst0**: 0.0 -- used as the c0 coefficient in the polynomial
- **vConst1**: 1.0 -- used for clamping and in the `2*sigmoid - 1` computation

#### SFPU Execution Flow

1. **Initialization** (`tanh_init`):
   - For LUT mode: Load three 16-bit packed coefficient words into LRegs 0-2 via `_sfpu_load_imm16_`
   - For fp32 mode: Initialize the sigmoid kernel (which initializes reciprocal tables)
   - For bfloat16 mode: Write three polynomial coefficients to programmable constant registers via `SFPCONFIG`

2. **Per-tile processing** (in the compute kernel):
   - `tile_regs_acquire()`: Lock DST register bank for exclusive SFPU access
   - `cb_wait_front(c_0, 1)`: Block until a tile is available from the reader
   - `copy_tile(c_0, 0, 0)`: Unpack tile from CB c_0 into DST register 0 (unpacker decompresses the data format)
   - `tanh_tile<0u>(0)`: Dispatch SFPU tanh computation on DST[0]

3. **Per-face SFPU dispatch** (inside `_llk_math_eltwise_unary_sfpu_params_`):
   - Set destination register base address for the current face
   - Call `calculate_tanh<false, is_fp32_dest_acc_en, 8>()` for each of the 4 faces (in RC vector mode)
   - Each call processes 8 rows of the 16x16 face

4. **Per-row SFPU computation** (inside `calculate_tanh`):
   - Load value from `dst_reg[0]` (SFPLOAD)
   - **If approximation mode**: Apply `SFPLUT` instruction with preloaded coefficients
   - **If fp32 accurate mode**:
     - Compute `|x|` via `SFPABS`
     - If `|x| < 0.6`: evaluate minimax polynomial via chain of `SFPMAD` instructions
     - If `|x| >= 0.6`: compute `2 * sigmoid(2x) - 1` (involving exp, reciprocal, and arithmetic)
   - **If bfloat16 polynomial mode**:
     - Compute `|x|` via `SFPABS`
     - Evaluate degree-5 polynomial via chain of `SFPMAD` instructions (Horner's method)
     - Clamp via `SFPSWAP` (VEC_MIN_MAX)
     - Restore sign via `SFPSETSGN`
     - Convert to fp16b via `SFPSTOCHRND`
   - Store result to `dst_reg[0]` (SFPSTORE)
   - Advance `dst_reg++` (TTINCRWC)

5. **Post-SFPU processing** (back in compute kernel):
   - `tile_regs_commit()`: Release DST from math, signal pack engine
   - `tile_regs_wait()`: Wait for pack engine readiness
   - `pack_tile(0, c_2)`: Pack DST[0] into output CB c_2 (packer compresses to output data format)
   - `cb_pop_front(c_0, 1)`: Free input tile slot
   - `tile_regs_release()`: Release DST registers

#### SFPU Configuration

| Configuration | Value | Source |
|---------------|-------|--------|
| `APPROXIMATION_MODE` | `false` (default) | `get_op_approx_mode(TANH)` returns `false`; overridden to `true` if param0 is set |
| `math_fidelity` | `MathFidelity::HiFi4` | Hardcoded in program factory |
| `fp32_dest_acc_en` | From operation params | Controls whether fp32 or bfloat16 path is selected |
| `math_approx_mode` | `false` | Passed to `ComputeConfig`; controls `APPROX` compile-time constant |
| Programmable Constants | c4, c5, c6 coefficients | Set in `tanh_init` for bfloat16 polynomial mode |
| LUT Coefficients | `0x1DFF`, `0x481A`, `0xFF00` | Set in `tanh_init` for approximation mode |

The `APPROXIMATION_MODE` template parameter is controlled by the `param0` field of the `UnaryWithParam` struct. By default, `string_to_unary_with_param("tanh")` creates `UnaryWithParam(UnaryOpType::TANH, static_cast<float>(false))`, meaning the accurate (non-LUT) path is selected.

#### Hardware Compatibility Notes

The SFPU kernel implementation (`ckernel_sfpu_tanh.h`) is **identical** between Wormhole B0 and Blackhole architectures. Both use the same:
- Polynomial coefficients
- LUT coefficients
- Conditional paths (fp32 accurate vs bfloat16 polynomial vs LUT)
- SFPI intrinsic functions

Key architectural differences that may affect TANH behavior:
- **SFPLUT precision**: Both architectures use the same 8-bit coefficient packing scheme (`unpackCoeff`), but Blackhole has improved NaN and zero handling in the underlying FMA model.
- **SFPMAD (FMA) precision**: The FMA sub-unit uses a partially-fused multiply-add that maintains higher precision than FP32 for the intermediate product. Blackhole may have minor improvements in edge-case handling (subnormal flushing, NaN propagation).
- **Reciprocal accuracy**: The `_sfpu_reciprocal_` function used by the sigmoid path may have different convergence behavior on Blackhole due to potential hardware reciprocal approximation instructions (`SFPARECIP`).

---

## Compile-Time Defines Summary

| Define | Value | Purpose |
|--------|-------|---------|
| `SFPU_OP_CHAIN_0` | `"SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"` | Main SFPU op chain macro expanded in compute kernel |
| `SFPU_OP_CHAIN_0_INIT_0` | `"tanh_tile_init<0u>();"` | Initialization call |
| `SFPU_OP_CHAIN_0_FUNC_0` | `"tanh_tile<0u>(0);"` | Compute call on DST[0] |
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | Includes `api/compute/compute_kernel_api.h` which contains `tanh_tile` |
| `INP_FLOAT32` or `INP_FLOAT` | `1` | Set based on input data type |

---

## Runtime Arguments Summary

### Reader Kernel (per core)
| Index | Value | Description |
|-------|-------|-------------|
| 0 | `src_buffer->address()` | Source tensor DRAM address |
| 1 | `num_pages_per_core` | Number of tiles to read |
| 2 | `num_pages_written` | Starting tile ID |

### Writer Kernel (per core)
| Index | Value | Description |
|-------|-------|-------------|
| 0 | `dst_buffer->address()` | Destination tensor DRAM address |
| 1 | `num_pages_per_core` | Number of tiles to write |
| 2 | `num_pages_written` | Starting tile ID |

### Compute Kernel (per core)
| Index | Value | Description |
|-------|-------|-------------|
| 0 | `0` | packed_scalar1 (unused by TANH) |
| 1 | `0` | packed_scalar2 (unused by TANH) |

---

## Override Runtime Arguments

The `override_runtime_arguments` method updates only the buffer addresses (arg index 0) for both reader and writer kernels when tensor buffers are reallocated. Tile counts and starting IDs remain unchanged from the initial `create` call.

---

## External Knowledge Sources

### DeepWiki References
- **tenstorrent/tt-metal**: UnaryProgramFactory structure, kernel registration patterns, SFPU op chain macro system
- **tenstorrent/tt-llk**: ckernel namespace organization, `_llk_math_eltwise_unary_sfpu_params_` dispatch pattern, `calculate_tanh` function signatures, face iteration model
- **tenstorrent/tt-isa-documentation**: SFPLUT instruction specification (opcode 0x73, piecewise linear interpolation, coefficient packing), SFPMAD FMA instruction details
- **tenstorrent/sfpi**: SFPI C++ API mapping to hardware intrinsics (`lut()` -> `SFPLUT`, `abs()` -> `SFPABS`, `setsgn()` -> `SFPSETSGN`, `vec_min_max()` -> `SFPSWAP`, `dst_reg` -> `SFPLOAD/SFPSTORE`, programmable constants -> `SFPCONFIG`)

### Confluence References
- **Tensix SFPU Instruction Set Architecture** (Page ID: 1170505767):
  - SFPLUT section: Full algorithmic implementation showing 3-range piecewise linear interpolation with `unpackCoeff` function that converts 8-bit LUT entries to FP32
  - SFPMAD section: FMA operation details `(A * B) + C` with sign inversion modes and subnormal flushing behavior
  - Programmable Constant Registers: Reset values and SFPCONFIG write mechanism
  - SFPLUTFP32 section: Extended LUT instruction with FP32/FP16 coefficient modes (not used by TANH but available for higher-fidelity approximations)

### Glean References
Not consulted. The combination of source code, DeepWiki, and Confluence provided sufficient detail for this analysis.

---

## File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory (host-side program construction) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Op chain define generation, kernel path resolution, approx mode |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Compute kernel (shared by most unary SFPU ops) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h` | SFPU tanh kernel (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h` | SFPU tanh kernel (Wormhole B0) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` | LLK API wrapper for tanh |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` | Sigmoid SFPU kernel (dependency for fp32 path) |
| `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | Compute API: `tanh_tile` / `tanh_tile_init` declarations |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include system for SFPU ops |
