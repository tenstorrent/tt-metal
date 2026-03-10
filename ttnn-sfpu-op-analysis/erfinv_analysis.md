# ERFINV Operation Analysis

## Overview

**Operation**: ERFINV (Inverse Error Function)
**Type**: Unary SFPU Operation
**Mathematical Definition**: `erfinv(x)` computes the inverse of the error function `erf`, such that `erf(erfinv(x)) = x` for `x` in `(-1, 1)`.

**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

ERFINV is implemented as a standard unary SFPU operation that shares the generic `UnaryProgramFactory` and `UnarySubCoreGridProgramFactory` program factories with all other unary SFPU operations. The operation uses the Winitzki (2008) approximation formula, which relies internally on logarithm and square root subfunctions.

---

## Program Factory Analysis

### Factory Type
- **Primary Factory**: `UnaryProgramFactory` (interleaved memory layout)
- **Secondary Factory**: `UnarySubCoreGridProgramFactory` (sub-core grid variant)
- **Namespace**: `ttnn::prim`

### Program Structure

The program factory creates a standard 3-kernel program (reader, compute, writer) distributed across available Tensix cores:

1. **Work Distribution**: `split_work_to_cores()` divides tiles across the compute grid, producing two core groups (group 1 and group 2) that may have different tile counts to handle remainders.
2. **Circular Buffers**:
   - `c_0` (CB index 0): Input buffer, 2 tiles, input data format
   - `c_2` (CB index 2): Output buffer, 2 tiles, output data format
   - `c_1` (CB index 1): Temporary buffer -- NOT used for ERFINV (only allocated for HARDSHRINK and LOGIT)
3. **Data Formats**: Input and output data formats are derived from the tensor dtypes via `datatype_to_dataformat_converter()`.

### Kernel Registration

Three kernels are registered via `CreateKernel`:

| Kernel Type | Source Path | Config |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `ReaderDataMovementConfig` with TensorAccessorArgs |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `WriterDataMovementConfig` with output CB index and TensorAccessorArgs |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `ComputeConfig` with HiFi4 fidelity, defines including `SFPU_OP_ERFINV_INCLUDE` |

### Compile-Time Defines

The operation dispatch chain sets the following defines:

| Define | Value | Purpose |
|---|---|---|
| `SFPU_OP_ERFINV_INCLUDE` | `1` | Triggers conditional inclusion of `erfinv.h` in `sfpu_split_includes.h` |
| `SFPU_OP_INIT_0` | `erfinv_tile_init();` | SFPU initialization call |
| `SFPU_OP_FUNC_0` | `erfinv_tile(0);` | SFPU per-tile function call |
| `SFPU_OP_CHAIN_0` | Composed init+func | Expanded into the compute kernel's `#ifdef SFPU_OP_CHAIN_0` block |
| `INP_FLOAT32` / `INP_FLOAT` / etc. | `1` | Selects the input data format path |

### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,   // caller-controlled
    .unpack_to_dest_mode = unpack_to_dest_mode,   // Default (or Fp32 if preserve_fp32_precision)
    .bfp8_pack_precise = args.bfp8_pack_precise,
    .math_approx_mode = false,                    // ERFINV returns false from get_op_approx_mode
    .compile_args = {num_tiles_per_core, 1},      // per_core_block_cnt, per_core_block_size
    .defines = unary_defines
}
```

**Key observation**: `math_approx_mode` is always `false` for ERFINV because `get_op_approx_mode()` returns `false` for all unary operations by default. However, the SFPU kernel template parameter `APPROXIMATION_MODE` is propagated through the `APPROX` macro from the LLK layer, which maps to the `math_approx_mode` field.

### Runtime Arguments

| Kernel | Runtime Args |
|---|---|
| Reader | `{src_buffer_address, num_pages_per_core, start_page_id}` |
| Writer | `{dst_buffer_address, num_pages_per_core, start_page_id}` |
| Compute | `{packed_scalar1=0, packed_scalar2=0}` (ERFINV uses no scalar params) |

### Program Caching

The factory supports program caching via `override_runtime_arguments()`, which updates only the source and destination buffer addresses on subsequent invocations, avoiding full program reconstruction.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM source buffer address
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages (tiles) this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();      // Compile-time tensor accessor config (from slot 0)

    constexpr uint32_t cb_id_in0 = 0;                      // Input circular buffer index (c_0)

    // Get page size from CB interface -- works for both TILE and ROW_MAJOR layouts
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // Each iteration reads one page (one tile for tiled layout)
    constexpr uint32_t onepage = 1;

    // Construct TensorAccessor from compile-time args and runtime address
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

// Read pages sequentially from DRAM into the input circular buffer
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onepage);           // Wait for space in the CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);  // Get L1 write pointer
        noc_async_read_page(i, s, l1_write_addr);     // Issue async NoC read from DRAM
        noc_async_read_barrier();                       // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);              // Signal page is ready for consumer (compute)
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM destination buffer address
    const uint32_t num_pages = get_arg_val<uint32_t>(1);    // Number of pages this core writes
    const uint32_t start_id = get_arg_val<uint32_t>(2);     // Starting page index

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // Compile-time tensor accessor config (slot 1)

    // Get page size from CB interface
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);  // For sharded output, just wait for all pages
#else

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onepage);             // Wait for compute to produce a page
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);    // Get L1 read pointer
        noc_async_write_page(i, s, l1_read_addr);     // Issue async NoC write to DRAM
        noc_async_writes_flushed();                     // Ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);              // Free the CB slot for compute to reuse
    }
    noc_async_write_barrier();  // Final barrier to ensure all writes complete
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
#include "api/compute/common.h"              // Common compute API (tile_regs_acquire/commit/wait/release)
#include "api/compute/tile_move_copy.h"      // copy_tile: unpacks from CB to DST register
#include "api/compute/eltwise_unary/eltwise_unary.h"  // Base unary eltwise infrastructure
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditional SFPU op includes -- SFPU_OP_ERFINV_INCLUDE triggers #include of erfinv.h
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process on this core
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Number of tiles per block (always 1 for standard unary)

    // Initialize the SFPU pipeline: sets up unpack (c_0) and pack (c_2) configuration,
    // and calls erfinv_tile_init() which in turn calls log_init<false, false, false>()
    // to load logarithm constants into SFPU programmable constant registers.
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in the output CB for the entire block before processing
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DST registers for this tile computation.
            // This prevents the packer from reading DST while the math engine is writing.
            tile_regs_acquire();

            // Wait for the reader kernel to push one tile into the input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Unpack tile from CB c_0, slot 0, into DST register 0.
            // This calls the unpacker hardware to convert data from the CB format
            // into the internal DST register format (typically FP32 in DEST).
            copy_tile(tt::CBIndex::c_0, 0, 0);

// The SFPU_OP_CHAIN_0 macro expands to:
//   erfinv_tile_init();   -- one-time SFPU configuration (log constants)
//   erfinv_tile(0);       -- apply erfinv to DST register tile 0
// erfinv_tile() calls SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, 0)
// which expands to _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_erfinv<APPROX>, 0, VectorMode::RC)
// This iterates over all 4 faces of the 32x32 tile and applies calculate_erfinv() to each face.
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST registers are ready for the packer to read
            tile_regs_commit();

            // Wait for the packer to finish reading the previous tile before reusing DST
            tile_regs_wait();

            // Pack the result from DST register 0 into output CB c_2
            pack_tile(0, tt::CBIndex::c_2);

            // Free the consumed tile from the input CB, allowing the reader to reuse the slot
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST registers so the next iteration can acquire them
            tile_regs_release();
        }
        // Push the completed block of tiles to the output CB for the writer kernel
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

The SFPU kernel is defined identically for both supported architectures:
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`

Both files are byte-identical. The implementation also depends on:
- `ckernel_sfpu_log.h` for the `calculate_log_body` function and `log_init` initialization

#### API Layer

The high-level API is in `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h`:

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
// Only included when compiling for the TRISC math processor (RISC-V core 3, the math engine)
#include "ckernel_sfpu_erfinv.h"              // The actual SFPU implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"  // Macro helpers for SFPU dispatch
#endif

namespace ckernel {

// erfinv_tile: applies inverse error function to tile at DST register index idst.
// SFPU_UNARY_NO_PARAM_KERNEL_FN expands to:
//   _llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_erfinv<APPROX>, idst, (int)VectorMode::RC)
// RC mode processes all 4 faces (16x16 sub-tiles) of the 32x32 tile.
ALWI void erfinv_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, idst)); }

// erfinv_tile_init: initializes SFPU configuration for erfinv.
// SFPU_INIT_KERNEL_CALL expands to:
//   llk_math_eltwise_unary_sfpu_init<SfpuType::erfinv, APPROX>(sfpu::erfinv_init<APPROX>)
// This first calls _llk_math_eltwise_unary_sfpu_init_<SfpuType::erfinv>() to configure
// SFPU address modifiers, then calls erfinv_init() which delegates to log_init<false, false, false>()
// to load ln(2) and polynomial coefficients into SFPU programmable constant registers.
ALWI void erfinv_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(erfinv, sfpu::erfinv_init, APPROX)); }
}  // namespace ckernel
```

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"          // Core ckernel infrastructure
#include "ckernel_defs.h"     // Definitions for ckernel constants and types
#include "ckernel_sfpu_log.h" // calculate_log_body and log_init -- erfinv depends on natural log

#include "sfpi.h"             // SFPI programming interface: vFloat, vUInt, dst_reg, v_if, etc.

namespace ckernel {
namespace sfpu {

// Custom square root function using the fast inverse square root algorithm (Quake III method).
// This is used instead of the hardware sqrt to avoid dependency on HW sqrt configuration
// and to maintain consistency when sqrt is called as a subroutine within erfinv.
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    v_if(val != 0.0f) {
        // The magic constant 0x5f37 is the BFloat16b equivalent of the famous 0x5f3759df
        // used in the fast inverse square root algorithm. It provides an initial approximation
        // for 1/sqrt(x) by exploiting the IEEE floating-point bit representation:
        // the exponent bits encode an approximate log2, so subtracting the shifted bits
        // from a magic constant gives an approximate -0.5 * log2(x), i.e., log2(1/sqrt(x)).
        sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));

        // Two iterations of Newton-Raphson refinement for 1/sqrt(x):
        // f(y) = 1/(y^2) - x = 0  =>  y_{n+1} = y_n * (1.5 - 0.5 * x * y_n^2)
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;  // First Newton-Raphson iteration
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;  // Second Newton-Raphson iteration

        // Convert from 1/sqrt(x) to sqrt(x) by multiplying by x
        out = approx * val;
    }
    v_else { out = val; }  // sqrt(0) = 0
    v_endif;
    return out;
}

// Core erfinv computation body using the Winitzki (2008) approximation.
// Reference: "A handy approximation for the error function and its inverse" by Sergei Winitzki
//
// The approximation defines:
//   erfinv(x) = sign(x) * sqrt( -c - ln(1-x^2)/2 + sqrt( (c + ln(1-x^2)/2)^2 - ln(1-x^2)/a ) )
// where:
//   a = 0.147 (approximation constant from the paper)
//   c = 2/(pi*a) = 4.330746750799873
//   1/a = 6.802721088435375
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    // Step 1: Compute log(1 - x^2)
    // This is the key intermediate value used throughout the formula.
    sfpi::vFloat log_value = in * in;         // x^2
    log_value = 1 - log_value;                // 1 - x^2
    // Use the LLK log implementation in FP32 mode (third template param = true)
    // to avoid intermediate rounding errors that would degrade accuracy.
    // The second argument (0) is the log base scale factor -- 0 means natural log (ln).
    log_value = calculate_log_body<false, false, true>(log_value, 0);

    // Step 2: Compute temp = -(2/(pi*a) + log(1-x^2)/2)
    sfpi::vFloat temp = log_value * 0.5;      // log(1-x^2) / 2

    // Pre-computed constants from the paper's coefficient a = 0.147:
    constexpr float TwoPiA = 4.330746750799873f;   // 2 / (pi * a) = 2 / (3.14159... * 0.147)
    constexpr float OneDivA = 6.802721088435375f;   // 1 / a = 1 / 0.147

    temp = TwoPiA + temp;    // 2/(pi*a) + log(1-x^2)/2
    temp = -temp;            // -(2/(pi*a) + log(1-x^2)/2)

    // Step 3: Compute the discriminant and take its square root
    // discriminant = temp^2 - log(1-x^2) / a
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);

    // Step 4: Add temp to the square root of the discriminant
    calculated_value = temp + intermediate_result;

    // Step 5: Take the outer square root to get the final result
    sfpi::vFloat result = calculate_sqrt_custom<false>(calculated_value);

    return result;
}

// Main SFPU entry point for erfinv.
// This function is called once per face (16x16 sub-tile) by the LLK dispatcher.
// It processes 8 rows of the face (ITERATIONS = 8), where each row contains
// a SIMD vector of elements accessed via dst_reg[0].
template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    // SFPU microcode: process 8 row-vectors per face.
    // Each face is 16x16 elements. The SFPU operates on one row at a time,
    // so 8 iterations process 8 rows. The LLK params wrapper handles
    // the face iteration (4 faces for RC mode), and within each face
    // call, SETRWC advances the DST pointer by 2 rows after each face.
    // With ITERATIONS=8 and dst_reg++ per iteration, this covers
    // 8 rows, and the SETRWC in the params wrapper advances past the
    // remaining 8 rows to reach the next face.
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];   // Read current row from DST register
        sfpi::vFloat result;

        // Exploit the odd symmetry of erfinv: erfinv(-x) = -erfinv(x)
        // By working with |x|, we simplify edge-case handling and only need
        // to restore the sign at the end.
        sfpi::vFloat abs_v = sfpi::abs(v);

        // Edge case: erfinv(+/-1) = +/-infinity
        v_if(abs_v == 1.0f) { result = std::numeric_limits<float>::infinity(); }
        // Edge case: erfinv(x) is undefined for |x| > 1 -- return NaN
        v_elseif(abs_v > 1.0f) {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        // Normal case: compute erfinv using the Winitzki approximation
        v_else { result = calculate_erfinv_body<true>(abs_v); }
        v_endif;

        // Restore the original sign: erfinv(-x) = -erfinv(x)
        result = sfpi::setsgn(result, v);

        sfpi::dst_reg[0] = result;  // Write result back to DST register
        sfpi::dst_reg++;            // Advance to the next row in the face
    }
}

// Initialization function for erfinv.
// Delegates to log_init because erfinv internally uses calculate_log_body,
// which requires programmable constant registers to be loaded with ln(2)
// and polynomial coefficients for the log approximation.
template <bool APPROXIMATION_MODE>
void erfinv_init() {
    // Template args: <APPROXIMATION_MODE=false, FAST_APPROX=false, is_fp32_dest_acc_en=false>
    // This loads:
    //   vConstFloatPrgm0 = 0.693147182464599609375  (ln(2))
    //   vConstFloatPrgm1 = -2.0069785118103027       (polynomial coeff)
    //   vConstFloatPrgm2 = 3.767500400543213          (polynomial coeff)
    log_init<false, false, false>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[0]` | Read/write access to the current row of the SFPU destination register file |
| `sfpi::dst_reg++` | Advance the destination register pointer to the next row |
| `sfpi::abs(v)` | Compute absolute value of a SIMD vector |
| `sfpi::setsgn(result, v)` | Copy the sign bits from `v` onto `result` |
| `sfpi::reinterpret<vUInt>(vFloat)` | Bitwise reinterpretation between float and unsigned integer SIMD types |
| `sfpi::reinterpret<vFloat>(vUInt)` | Bitwise reinterpretation from unsigned integer to float |
| `sfpi::s2vFloat16b(0x5f37)` | Load a scalar immediate as a BFloat16b constant into a SIMD register |
| `v_if` / `v_elseif` / `v_else` / `v_endif` | SFPU conditional execution (predicated per-lane) |
| `std::numeric_limits<float>::infinity()` | Generate +infinity constant |
| `std::numeric_limits<float>::quiet_NaN()` | Generate quiet NaN constant |
| Arithmetic: `*`, `+`, `-`, `>>` | SFPU ALU operations on SIMD vectors (multiply, add, subtract, right shift) |
| `calculate_log_body<false, false, true>()` | Internal LLK log function operating in FP32 mode |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `dst_reg[0]` | Input tile row (read), output tile row (write) -- operation is in-place |
| `dst_reg` pointer | Incremented 8 times per face via `dst_reg++`, covering 8 of 16 rows per face |
| `vConstFloatPrgm0` | Loaded with `ln(2) = 0.693147...` during `erfinv_init()` via `log_init()` |
| `vConstFloatPrgm1` | Loaded with polynomial coefficient `-2.0069785...` during init |
| `vConstFloatPrgm2` | Loaded with polynomial coefficient `3.767500...` during init |
| SFPU condition codes | Used by `v_if`/`v_elseif`/`v_else` for per-lane predication |
| Local SFPU registers | Temporary `vFloat`/`vUInt` variables are allocated to SFPU local registers by the compiler |

#### SFPU Execution Flow

1. **Initialization** (`erfinv_tile_init` -> `erfinv_init` -> `log_init`):
   - Loads natural logarithm constants (`ln(2)` and polynomial coefficients) into SFPU programmable constant registers (`vConstFloatPrgm0/1/2`).
   - Configures SFPU address modifiers and base registers via `_llk_math_eltwise_unary_sfpu_init_<SfpuType::erfinv>()`.

2. **Tile acquisition** (`tile_regs_acquire` in compute kernel):
   - Acquires exclusive access to DST registers, blocking the packer.

3. **Unpack** (`copy_tile(c_0, 0, 0)`):
   - The unpacker hardware reads one tile from CB `c_0` and converts it to the internal DST format (FP32 in DEST).
   - Data is placed in DST register index 0.

4. **SFPU dispatch** (`erfinv_tile(0)` -> `_llk_math_eltwise_unary_sfpu_params_`):
   - Issues `TTI_STALLWAIT(STALL_SFPU, MATH)` to wait for any prior math operations to complete.
   - Sets DST write address to tile index 0.
   - Iterates over all 4 faces in `VectorMode::RC` mode:
     - For each face, calls `calculate_erfinv()` which processes 8 rows.
     - After each face, advances the DST pointer by 2 increments of 8 rows (via `TTI_SETRWC`) to reach the next 16x16 face.
   - Issues `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` to wait for SFPU completion.

5. **Per-row SFPU computation** (within `calculate_erfinv`):
   - Reads input from `dst_reg[0]`.
   - Takes absolute value for odd symmetry exploitation.
   - Handles edge cases: `|x| == 1` -> infinity, `|x| > 1` -> NaN.
   - For normal inputs, calls `calculate_erfinv_body`:
     a. Computes `log(1 - x^2)` using `calculate_log_body` (FP32 precision).
     b. Computes `temp = -(2/(pi*a) + log(1-x^2)/2)`.
     c. Computes discriminant `temp^2 - log(1-x^2)/a`.
     d. Takes square root of discriminant using fast inverse sqrt + Newton-Raphson.
     e. Adds `temp` to discriminant sqrt.
     f. Takes outer square root for final result.
   - Restores sign via `setsgn`.
   - Writes result back to `dst_reg[0]`, advances to next row.

6. **Pack** (`pack_tile(0, c_2)`):
   - The packer reads the computed tile from DST register 0 and writes it to output CB `c_2`, converting from internal format to the output data format.

7. **CB management**:
   - `cb_pop_front(c_0, 1)` frees the input tile slot.
   - `cb_push_back(c_2, per_core_block_dim)` signals the writer kernel that output tiles are ready.

#### SFPU Configuration

| Configuration | Value | Notes |
|---|---|---|
| `APPROXIMATION_MODE` | `false` (via `math_approx_mode = false`) | ERFINV always runs in non-approximate mode |
| `VectorMode` | `RC` (Row-Column) | Processes all 4 faces of the 32x32 tile |
| `ITERATIONS` | 8 | Processes 8 of 16 rows per face; the params wrapper SETRWC handles the other 8 |
| `MathFidelity` | `HiFi4` | Highest fidelity setting |
| Log mode | `<false, false, true>` -- non-approx, no base scaling, FP32 | FP32 mode for log to minimize rounding errors in the intermediate computation |
| `SFPU_OP_ERFINV_INCLUDE` | `1` (compile-time define) | Triggers inclusion of erfinv.h via sfpu_split_includes.h |
| Scalar parameters | None | ERFINV is a no-parameter unary op (packed_scalar1 = packed_scalar2 = 0) |

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations of `ckernel_sfpu_erfinv.h` are **byte-identical**. This is because erfinv is implemented entirely using high-level SFPI intrinsics (`vFloat`, `vUInt`, `v_if`, `dst_reg`, etc.) which are portable across both architectures.

The only architecture-dependent code is in the `_llk_math_eltwise_unary_sfpu_params_` wrapper (in the tt_llk submodule), which handles DST address setup and face iteration differently between Wormhole B0 and Blackhole. However, from the SFPU kernel's perspective, the interface is identical.

Key architecture differences that are transparent to erfinv:
- **DST register file size**: Blackhole has a larger DEST, but the tile layout (4 faces of 16x16) is the same.
- **SFPU pipeline depth**: May differ, but the stall/wait logic in the params wrapper handles this.
- **FP32 accumulation**: When `fp32_dest_acc_en` is true, the unpack-to-dest mode and log body may take different code paths, but erfinv itself does not specialize for this.

---

## Circular Buffer Configuration

| CB Index | Name | Page Count | Data Format | Purpose |
|---|---|---|---|---|
| `c_0` | Input | 2 | Input tensor dtype | Double-buffered input from reader |
| `c_2` | Output | 2 | Output tensor dtype | Double-buffered output to writer |
| `c_1` | Temp | Not allocated for ERFINV | N/A | Only used by HARDSHRINK and LOGIT |

The double-buffering (2 pages) allows the reader and writer to overlap with compute, providing pipeline parallelism.

---

## Data Flow Summary

```
DRAM --> [Reader: NoC async read] --> CB c_0 --> [Compute: unpack -> SFPU erfinv -> pack] --> CB c_2 --> [Writer: NoC async write] --> DRAM
```

1. Reader reads one tile at a time from DRAM into CB `c_0`.
2. Compute waits for a tile in `c_0`, unpacks to DST, runs SFPU erfinv, packs to `c_2`.
3. Writer waits for a tile in `c_2`, writes to DRAM.

All three kernels run concurrently on different RISC-V processors within the same Tensix core, with circular buffers providing synchronization.

---

## File Inventory

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory: creates Program with 3 kernels and CBs |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header: defines `shared_variables_t` and cached program type |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Maps `UnaryOpType::ERFINV` to macro define, init/func strings, and kernel path |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Defines `UnaryOpType::ERFINV` enum value |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic compute kernel for all SFPU unary ops |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel: DRAM -> CB c_0 |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel: CB c_2 -> DRAM |
| `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h` | API layer: `erfinv_tile()` and `erfinv_tile_init()` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include gate for erfinv.h |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h` | SFPU kernel: `calculate_erfinv()` and `calculate_erfinv_body()` (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h` | SFPU kernel: identical copy (Wormhole B0) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h` | Dependency: `calculate_log_body()` and `log_init()` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | LLK macros: `SFPU_UNARY_NO_PARAM_KERNEL_FN`, `SFPU_INIT_KERNEL_CALL` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` | LLK init: `llk_math_eltwise_unary_sfpu_init<>()` |

---

## External Knowledge Sources

### DeepWiki References
- **tenstorrent/tt-metal**: Located all relevant source files, understood the unary operation dispatch flow from `UnaryOpType::ERFINV` through macro defines to the compute kernel.
- **tenstorrent/tt-llk**: Obtained the complete implementation of `_llk_math_eltwise_unary_sfpu_params_<>()` which handles DST address setup, face iteration in VectorMode::RC, and SFPU stall management.

### Confluence References
Not consulted -- DeepWiki provided sufficient detail for this operation's SFPU instructions (all are high-level SFPI intrinsics, not raw ISA instructions).

### Glean References
Not consulted -- the operation uses standard SFPI intrinsics documented in open-source repositories.

---

## Key Design Decisions

1. **Winitzki approximation over lookup tables**: The erfinv implementation uses an analytical approximation rather than a lookup table. This is because erfinv has a singularity at +/-1 (goes to infinity), making polynomial/table approaches impractical across the full domain. The Winitzki formula provides uniform accuracy with bounded computational cost.

2. **Custom sqrt instead of hardware sqrt**: The `calculate_sqrt_custom` function uses the fast inverse square root algorithm with two Newton-Raphson iterations, rather than calling a hardware sqrt instruction or the standard LLK sqrt. This avoids SFPU pipeline configuration conflicts (erfinv calls sqrt as a subroutine, and reconfiguring the SFPU for a different operation mid-computation would be complex).

3. **FP32 log mode**: The log computation within erfinv uses `is_fp32_dest_acc_en=true` (third template parameter) to `calculate_log_body`. This is critical because erfinv involves subtracting nearly-equal quantities (e.g., `1 - x^2` when `x` is close to 0), which amplifies rounding errors. Using FP32 precision for the intermediate log computation maintains accuracy.

4. **Odd symmetry exploitation**: By computing `erfinv(|x|)` and restoring the sign afterward, the implementation avoids duplicating edge-case logic for negative inputs and simplifies the conditional branching.

5. **8 iterations per face**: The SFPU processes 8 rows per `calculate_erfinv()` call, while each face has 16 rows. The remaining 8 rows are covered by the `SETRWC` instructions in the `_llk_math_eltwise_unary_sfpu_params_` wrapper, which advances the DST pointer by 16 rows (2 SETRWC increments of 8) between faces.

6. **No approximation mode**: Despite the template parameter `APPROXIMATION_MODE`, the `get_op_approx_mode()` function always returns `false` for ERFINV, meaning the operation always runs in full-precision mode. This is because the Winitzki approximation already introduces some error, and further approximation of the sub-operations (log, sqrt) would compound inaccuracy to unacceptable levels.
