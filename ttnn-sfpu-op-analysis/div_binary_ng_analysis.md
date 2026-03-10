# SFPU Operation Analysis: DIV (binary_ng)

## Operation Overview

The DIV operation in the `binary_ng` framework computes element-wise floating-point division: `c = a / b`. It supports both tensor-tensor and tensor-scalar variants, with specialized SFPU and FPU code paths.

**Operation Type**: `BinaryOpType::DIV`
**Framework**: `binary_ng` (next-generation binary operation infrastructure)
**Namespace**: `ttnn::operations::binary_ng`

### SFPU vs FPU Path Selection

DIV is unique among binary_ng operations because it has **dual dispatch paths**:

- **SFPU path** (`is_sfpu == true`): Maps directly to `SfpuBinaryOp::DIV`, using the SFPU vector unit to compute `a * reciprocal(b)` with edge-case handling (div-by-zero, NaN, inf).
- **FPU path** (`is_sfpu == false`): Decomposes into `process_rhs = RECIP` + `FpuBinaryOp::MUL`. The RHS is first passed through a unary reciprocal activation, then the FPU matrix unit performs multiplication.

The SFPU path is selected when `utils::is_binary_sfpu_op(BinaryOpType::DIV, a_dtype, b_dtype)` returns true. For INT32 inputs, the operation uses a dedicated `div_int32_tile` kernel that converts integers to float, divides, and returns a float result.

---

## Program Factory

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

### Program Structure

The `BinaryNgDeviceOperation::ProgramFactory::create` function builds a program with three kernels:

1. **Reader kernel** -- reads input tensors A and B from DRAM/L1 into circular buffers
2. **Compute kernel** -- performs the SFPU division on tiles in the DST register
3. **Writer kernel** -- writes results from the output circular buffer back to DRAM/L1

### Circular Buffer Configuration

| CB Index | Purpose | Double-Buffered | Notes |
|----------|---------|-----------------|-------|
| `c_0` | Input A (LHS) | Yes (2 tiles or shard size) | Source for dividend |
| `c_1` | Input B (RHS) | Yes (2 tiles, or 1 for scalar) | Source for divisor |
| `c_2` | Output C | Yes (2 tiles or shard size) | Destination for quotient |
| `c_3` | LHS intermediate | No (1 tile) | Only allocated if LHS activations are present |
| `c_4` | RHS intermediate | No (1 tile) | Only allocated if RHS activations are present |

### Kernel Selection Logic

The compute kernel is selected based on the `SubtileBroadcastType` and whether the operation is SFPU:

- **No broadcast** (both tensors same shape): `eltwise_binary_sfpu_no_bcast.cpp`
- **Broadcast** (scalar or column broadcast): `eltwise_binary_sfpu.cpp`
- **Scalar B** (B is a scalar value): `eltwise_binary_sfpu_scalar.cpp`

The kernel file path is resolved by `get_kernel_file_path(kernel_name, is_sfpu=true, is_where_op=false)`.

### Compile-Time Defines for DIV

When `SfpuBinaryOp::DIV` is selected, the `OpConfig::as_defines()` function calls `get_sfpu_init_fn(SfpuBinaryOp::DIV, dtype)` which returns:

- **Float types**: `BINARY_SFPU_INIT = "div_binary_tile_init();"` and `BINARY_SFPU_OP = "div_binary_tile"`
- **INT32**: `BINARY_SFPU_INIT = "div_int32_tile_init();"` and `BINARY_SFPU_OP = "div_int32_tile"`

### FP32 Dest Accumulation

For the SFPU DIV path (non-POWER ops), `UnpackToDestMode::UnpackToDestFp32` is set on all source CBs (`c_0`, `c_1`, `c_3`, `c_4`). This ensures tiles are unpacked to full FP32 precision in the DST register before the SFPU operates on them. The `fp32_dest_acc_en` flag is set when output or both inputs are Float32, Int32, or UInt32.

### Runtime Arguments

**Compute kernel** receives 4 runtime arguments:
1. `num_tiles` -- total number of output tiles to process on this core
2. `freq` -- broadcast frequency (tiles between broadcast input reloads)
3. `counter` -- starting tile offset within the broadcast cycle
4. `compute_scalar_value` -- unused for standard DIV (used for quantization zero-point)

### Work Distribution

Output tiles are split across cores using `split_work_to_cores()`. Each core processes a contiguous range of output tiles. For sharded tensors, the shard spec directly determines per-core tile counts.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`

The reader kernel reads input tensor A tiles from DRAM into CB `c_0`. It traverses the tensor in ND-CHW order using stride-based addressing to handle broadcasting across dimensions. For sharded inputs, it simply calls `cb_reserve_back` / `cb_push_back` to expose the pre-loaded shard data.

When B is a tensor (not scalar), the reader also reads B tiles into CB `c_1` using a separate set of strides and broadcast flags. The reader kernel variant is selected based on `SubtileBroadcastType` (no_bcast, row_bcast, col_bcast, scalar_bcast, etc.).

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` (scalar variant)

For the scalar variant, the writer first fills a single tile in CB `c_1` with the packed scalar value, then loops over output tiles writing them from CB `c_2` to DRAM. For the tensor-tensor variant, `writer_interleaved_no_bcast.cpp` is used, which only writes output tiles without handling scalar fill.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File (no-broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source (no-broadcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// SFPU split includes provide per-operation SFPU function declarations
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Binary SFPU API: provides div_binary_tile, div_binary_tile_init, etc.
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
// INT32 division API: provides div_int32_tile, div_int32_tile_init
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

// Common utilities: defines PREPROCESS macro for activation preprocessing
#include "eltwise_utils_common.hpp"
// SFPU utilities: defines HAS_ACTIVATIONS, PROCESS_ACTIVATIONS, BCAST_OP macros
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime arg 0: total number of output tiles this core must process
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: number of tiles processed per compute cycle (always 1 for binary_ng)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB indices for the three data streams
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;   // raw LHS input from reader
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;   // raw RHS input from reader
    constexpr auto cb_out = tt::CBIndex::c_2;        // output to writer

    // If LHS/RHS activations are defined, use intermediate CBs; otherwise alias to raw CBs
    // For plain DIV, no activations are present, so cb_post_lhs == cb_pre_lhs
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize unpack and pack engines for the LHS->output data path
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    // If RELU post-activation is fused into pack, configure hardware RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Initialize the SFPU for division ONCE if no activations require re-init per tile.
    // For DIV, this expands to: div_binary_tile_init();
    // which calls llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>()
    // That in turn calls _sfpu_binary_init_<APPROX, BinaryOp::DIV>() which sets up
    // the reciprocal constants (vConstFloatPrgm0 = 2.0f for Newton-Raphson).
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // Main tile processing loop: one tile per iteration
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS: if LHS has activations, unpack from cb_pre_lhs, apply activation,
        // pack to cb_post_lhs. For plain DIV, this is a no-op (PREPROCESS_0).
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Wait for LHS tile(s) to be available in the post-activation CB
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        // Same for RHS
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in output CB for the result tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // If activations are present but no post-activation, re-init SFPU here
        // (activation preprocessing may have clobbered SFPU state)
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire DST register file for writing
        tile_regs_acquire();

        // Unpack LHS tile(s) to even DST slots (0, 2, 4, ...)
        // First configure unpack for the LHS data format
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Copy tile i from cb_post_lhs into DST register slot i*2
            copy_tile(cb_post_lhs, i, i * 2);
        }

        // Unpack RHS tile(s) to odd DST slots (1, 3, 5, ...)
        // Reconfigure unpack for the RHS data format
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // Copy tile i from cb_post_rhs into DST register slot i*2+1
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // If post-activations exist, re-init SFPU before each tile
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // Execute the SFPU binary operation: DST[i*2] = div(DST[i*2], DST[i*2+1])
            // For float DIV, this expands to: div_binary_tile(i*2, i*2+1, i*2)
            // which calls the SFPU to compute: in0 * reciprocal(in1) with edge-case handling
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // Apply any post-activations (e.g., TYPECAST for mixed dtypes)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        // Signal that DST register writes are complete
        tile_regs_commit();

        // Wait for DST data to be ready for packing
        tile_regs_wait();

        // Pack result tile(s) from DST into the output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        // Release DST registers for the next iteration
        tile_regs_release();

        // Push completed output tile(s) to writer and free input CB slots
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### Compute Kernel File (broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp`

The broadcast variant adds a two-level loop structure: the outer loop holds the broadcast input constant while the inner loop iterates over the non-broadcast input. The `BCAST_INPUT` compile-time define selects which input (LHS or RHS) is the broadcast source. The core SFPU dispatch logic (`BINARY_SFPU_OP`, `BINARY_SFPU_INIT`) is identical to the no-broadcast variant.

#### Compute Kernel File (scalar variant)
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`

The scalar variant loads the RHS tile once (the scalar value pre-filled by the writer into CB `c_1`) and then iterates over all LHS tiles. The RHS CB is only popped after all tiles are processed.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File (Float Division)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`
(Identical implementation in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`)

#### Annotated SFPU Kernel Source (Float Division)

```cpp
// ckernel_sfpu_binary.h -- Wormhole B0 / Blackhole shared implementation

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Utility: Convert float32 to bfloat16 with Round-to-Nearest-Even (RNE).
// Used when fp32 dest accumulation is disabled to maintain bf16 precision parity with FPU.
// The "0x7fff + LSB" trick handles the IEEE 754 tie-breaking rule:
//   - Ties (lower 16 bits == 0x8000) round to even mantissa
//   - Non-ties round normally
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1;       // bit 16 = LSB of bf16 mantissa
    bits = bits + 0x7fffU + lsb;               // rounding bias + tie-breaker
    bits = bits & 0xFFFF0000U;                 // truncate lower 16 bits
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// Float binary division: computes a / b element-wise across a tile.
// Template params:
//   APPROXIMATION_MODE: if true, uses fewer Newton-Raphson iterations (faster, less precise)
//   BINOP: always BinaryOp::DIV for this function
//   ITERATIONS: 8 (processes 8 rows of 4 floats each = 32 rows per tile face)
//   is_fp32_dest_acc_en: if true, skips bf16 rounding (keeps full fp32 precision)
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(
    const uint dst_index_in0,    // DST register index for input A (dividend)
    const uint dst_index_in1,    // DST register index for input B (divisor)
    const uint dst_index_out     // DST register index for output (quotient)
) {
    // Each tile occupies 32 SFPI-addressable rows in the DST register.
    // The SFPU processes 4 elements (one row) per SFPI instruction.
    // 8 iterations * 4 elements = 32 rows = one tile face.
    // The full tile (4 faces) is handled by the LLK params wrapper which
    // calls this function 4 times with adjusted dst_reg offsets.
    constexpr uint dst_tile_size_sfpi = 32;

    for (int d = 0; d < ITERATIONS; d++) {
        // Load 4 float elements from the dividend tile (row d)
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        // Load 4 float elements from the divisor tile (row d)
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Core division: multiply dividend by reciprocal of divisor.
        // _sfpu_reciprocal_<2> uses approx_recip() as initial estimate
        // followed by 2 Newton-Raphson iterations for high precision.
        // Newton-Raphson formula: y_{n+1} = y_n * (2 - x * y_n)
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        // Edge case handling using SFPU conditional execution (v_if/v_else/v_endif):
        // These compile to SFPU condition code instructions that mask lanes.
        v_if(in1 == 0) {
            // Division by zero: 0/0 = NaN, nonzero/0 = +/-inf
            v_if(in0 == 0) {
                result = std::numeric_limits<float>::quiet_NaN();
            }
            v_else {
                result = std::numeric_limits<float>::infinity();
                // Preserve the sign of the dividend in the infinity result
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        // Identity case: x/x = 1.0 (avoids reciprocal rounding errors)
        v_elseif(in0 == in1) {
            result = sfpi::vConst1;  // hardware constant 1.0f
        }
        v_endif;

        // If not in fp32 accumulation mode, round result to bf16 precision.
        // This ensures SFPU division matches FPU multiplication precision
        // when the output format is bf16.
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        // Write 4 result elements back to the output DST register (row d)
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        // Advance the SFPI row pointer to the next row
        sfpi::dst_reg++;
    }
}

// Initialization for binary SFPU operations.
// For DIV, this delegates to _sfpu_binary_init_<APPROX, BinaryOp::DIV>()
// which calls _init_sfpu_reciprocal_<false>() to set up:
//   - vConstFloatPrgm0 = 2.0f (used in Newton-Raphson: y * (2 - x*y))
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Kernel File (INT32 Division)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_div_int32.h`
(Identical implementation in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_div_int32.h`)

#### Annotated SFPU Kernel Source (INT32 Division)

```cpp
// ckernel_sfpu_div_int32.h -- Integer division via float conversion

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Integer division: converts int32 operands to float, divides, returns float result.
// Note: the result is a FLOAT, not an integer. For integer results, use div_int32_floor
// or div_int32_trunc which additionally call _float_to_int32_.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(
    const uint dst_index_in0,    // DST register index for input A (dividend, int32)
    const uint dst_index_in1,    // DST register index for input B (divisor, int32)
    const uint dst_index_out     // DST register index for output (quotient, float)
) {
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8   // hint to fully unroll the 8-iteration loop
    for (int d = 0; d < ITERATIONS; d++) {
        // Load as vInt (reinterpret DST bits as signed 32-bit integers)
        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        // Fast path: if both operands are equal and nonzero, result is 1.0
        v_if(in0 != 0 && in0 == in1) {
            result = sfpi::vConst1;
        }
        v_else {
            // Convert int32 to float using SFPU int-to-float conversion.
            // The second parameter (0) is the shift amount for fixed-point scaling.
            sfpi::vFloat float_in0 = sfpi::int32_to_float(in0, 0);
            sfpi::vFloat float_in1 = sfpi::int32_to_float(in1, 0);
            // Divide using reciprocal with 2 Newton-Raphson iterations
            result = float_in0 * _sfpu_reciprocal_<2>(float_in1);
        }
        v_endif;

        // Store result as float in DST
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Init for int32 division: sets up reciprocal constants.
// _init_sfpu_reciprocal_<false>() configures vConstFloatPrgm0 = 2.0f
// for the Newton-Raphson reciprocal refinement.
template <bool APPROXIMATION_MODE>
inline void div_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
```

#### LLK Dispatch Layer
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`

```cpp
// LLK layer that connects the compute API to the SFPU kernel functions.

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel {

// Init: configures SFPU state for the binary op.
// For DIV, calls sfpu_binary_init<APPROX, BinaryOp::DIV>()
// which calls _sfpu_binary_init_<APPROX, BinaryOp::DIV>()
// which calls _init_sfpu_reciprocal_<false>() to set vConstFloatPrgm0 = 2.0f
template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>);
}

// Execution: dispatches to calculate_sfpu_binary_div with params wrapper.
// The _llk_math_eltwise_binary_sfpu_params_ function:
//   1. Sets the DST write address
//   2. Waits for SFPU readiness
//   3. Calls the SFPU function 4 times (once per tile face)
//   4. Clears the DST address when done
template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_div(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_div<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
```

#### Compute API Layer
`tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`

```cpp
// Top-level compute API that device kernels call directly.

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_binop.h"
#include "llk_math_eltwise_binary_sfpu_binary_pow.h"
#include "llk_math_eltwise_binary_sfpu_binary_comp.h"
#endif

namespace ckernel {

// div_binary_tile: the function that BINARY_SFPU_OP expands to for float DIV.
// Parameters:
//   idst0: DST index of dividend tile
//   idst1: DST index of divisor tile
//   odst:  DST index for quotient output
// Runs on the MATH RISC-V processor (TRISC2).
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop_div<APPROX, ckernel::BinaryOp::DIV, DST_ACCUM_MODE>(
        idst0, idst1, odst)));
}

// div_binary_tile_init: the function that BINARY_SFPU_INIT expands to for float DIV.
// Sets up the reciprocal Newton-Raphson constants in SFPU programmable registers.
ALWI void div_binary_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::DIV>()));
}

}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|-------------------------|-------------|
| `sfpi::dst_reg[index]` | Load/store 4 float elements from/to DST register at the given SFPI row |
| `sfpi::dst_reg++` | Advance the SFPI row pointer by one row (4 elements) |
| `sfpi::approx_recip(x)` | Hardware initial reciprocal approximation (called inside `_sfpu_reciprocal_`) |
| `sfpi::vConstFloatPrgm0` | Programmable constant register, set to 2.0f for Newton-Raphson |
| `sfpi::vConst1` | Hardware constant 1.0f |
| `sfpi::setsgn(value, sign_source)` | Copy sign bit from `sign_source` to `value` |
| `sfpi::reinterpret<vUInt>(x)` | Bitwise reinterpret vFloat as vUInt (no conversion) |
| `sfpi::int32_to_float(x, shift)` | Convert int32 to float with optional fixed-point shift |
| `v_if / v_else / v_elseif / v_endif` | SFPU conditional execution (lane masking via condition codes) |
| Arithmetic: `*`, `+`, `>>`, `&`, `\|\|`, `==`, `!=` | SFPU vector arithmetic and comparison operations on vFloat/vInt/vUInt |

#### SFPU Register Usage

- **DST registers**: Three tile slots used simultaneously:
  - `dst_index_in0 * 32` through `dst_index_in0 * 32 + 31`: Dividend tile (input A)
  - `dst_index_in1 * 32` through `dst_index_in1 * 32 + 31`: Divisor tile (input B)
  - `dst_index_out * 32` through `dst_index_out * 32 + 31`: Quotient tile (output, overlaps with in0)
  - In practice, `dst_index_out == dst_index_in0` (slot 0), so the dividend is overwritten in-place
- **Programmable constant registers**:
  - `vConstFloatPrgm0` = 2.0f (set during init, used by Newton-Raphson in reciprocal)
  - On Wormhole B0: `vConstFloatPrgm1`, `vConstFloatPrgm2` may also be used for the quadratic initial estimate
- **SFPI row pointer** (`dst_reg++`): auto-incremented through 32 rows per tile face

#### SFPU Execution Flow

1. **Initialization** (`div_binary_tile_init` / `BINARY_SFPU_INIT`):
   - Calls `_sfpu_binary_init_<APPROX, BinaryOp::DIV>()`
   - For DIV and POW operations, this calls `_init_sfpu_reciprocal_<false>()`
   - Sets `vConstFloatPrgm0 = 2.0f` (Newton-Raphson constant)
   - On Wormhole B0, may also set quadratic estimate coefficients

2. **Tile acquisition from circular buffers**:
   - `cb_wait_front(cb_post_lhs, 1)`: Block until dividend tile is ready
   - `cb_wait_front(cb_post_rhs, 1)`: Block until divisor tile is ready
   - `cb_reserve_back(cb_out, 1)`: Reserve output slot

3. **Unpack to DST registers**:
   - `tile_regs_acquire()`: Acquire exclusive access to DST register file
   - `copy_tile(cb_post_lhs, 0, 0)`: Unpack LHS tile to DST slot 0 (via UNPACK RISC-V)
   - `copy_tile(cb_post_rhs, 0, 1)`: Unpack RHS tile to DST slot 1

4. **SFPU math operation** (`div_binary_tile(0, 1, 0)`):
   - The LLK params wrapper calls the SFPU function 4 times (once per tile face: top-left, top-right, bottom-left, bottom-right)
   - Each call processes 8 iterations of 4 elements = 32 rows
   - Per element: `result = in0 * reciprocal(in1)` with edge-case handling for 0/0, x/0, and x/x
   - Result overwrites DST slot 0 (same as input A)

5. **Pack to output circular buffer**:
   - `tile_regs_commit()`: Signal DST writes complete
   - `tile_regs_wait()`: Wait for data to be packable
   - `pack_tile(0, cb_out)`: Pack DST slot 0 to output CB (via PACK RISC-V)
   - `tile_regs_release()`: Release DST for next tile

6. **CB management**:
   - `cb_push_back(cb_out, 1)`: Make output tile available to writer
   - `cb_pop_front(cb_post_lhs, 1)`: Free LHS input slot
   - `cb_pop_front(cb_post_rhs, 1)`: Free RHS input slot

#### SFPU Configuration

- **`APPROX` (compile-time)**: Controls approximation mode. When true, `_sfpu_reciprocal_<0>` uses only the hardware initial estimate with no Newton-Raphson refinement. When false (default for DIV), `_sfpu_reciprocal_<2>` applies 2 Newton-Raphson iterations.
- **`DST_ACCUM_MODE` (compile-time)**: Corresponds to `fp32_dest_acc_en`. When true, results stay in fp32; when false, `float32_to_bf16_rne()` rounds to bf16 after computation.
- **`UnpackToDestFp32`**: For SFPU DIV (non-POWER ops), all source CBs use `UnpackToDestMode::UnpackToDestFp32` to ensure the SFPU operates on full-precision fp32 values regardless of the input tensor's storage format.
- **`num_tiles_per_cycle = 1`**: Always 1 for binary_ng, meaning one tile pair is processed per loop iteration.

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The `ckernel_sfpu_binary.h` file is **identical** between Wormhole B0 and Blackhole architectures. The `calculate_sfpu_binary_div` function uses the same algorithm on both.
- **Reciprocal implementation differs**: The underlying `_sfpu_reciprocal_` function (from tt_llk) has architecture-specific implementations:
  - **Blackhole**: Uses `sfpi::approx_recip(x)` as the initial estimate, then applies Newton-Raphson.
  - **Wormhole B0**: Uses a quadratic initial estimate with architecture-specific coefficients stored in `vConstFloatPrgm0/1/2`, then applies Newton-Raphson.
  - Both use 2 Newton-Raphson iterations for DIV (`max_iter = 2`), providing high precision.
- **INT32 division**: Same implementation on both architectures. Converts to float, divides, returns float. The `#pragma GCC unroll 8` hint is used for the INT32 variant but not the float variant.

---

## Operation Configuration Details

### OpConfig for DIV

From `binary_ng_utils.cpp`, the `OpConfig` constructor for DIV:

```cpp
case BinaryOpType::DIV:
    if (is_sfpu_op()) {
        binary_op = SfpuBinaryOp::DIV;
    } else {
        process_rhs = unary::UnaryOpType::RECIP;
        binary_op = FpuBinaryOp::MUL;
    }
    break;
```

- **SFPU path**: No pre/post processing. Pure SFPU division.
- **FPU path**: Applies reciprocal to RHS as a preprocessing step, then multiplies. This is because the FPU matrix unit only supports ADD, SUB, MUL natively.

### Integer Division Special Case

When `a_dtype == DataType::INT32 && b_dtype == DataType::INT32`, the program factory detects this as `is_integer_division` and skips adding a TYPECAST post-activation (line 600-604 of the program factory). The INT32 division kernel (`div_int32_tile`) produces a float output, so no typecast is needed.

### SubtileBroadcastType Handling

The `SubtileBroadcastType` enum controls how broadcasting is handled at the kernel level:

| Type | Description | Compute Kernel | Broadcast Frequency |
|------|-------------|----------------|---------------------|
| `NONE` | Both tensors same shape | `_no_bcast` | N/A |
| `SCALAR_A` | A is scalar | `_sfpu` (bcast) | Ht * Wt |
| `SCALAR_B` | B is scalar | `_sfpu_scalar` | 1 (B loaded once) |
| `COL_A` | A is column vector | `_sfpu` (bcast) | Wt |
| `COL_B` | B is column vector | `_sfpu` (bcast) | Wt |
| `ROW_A` | A is row vector | `_sfpu_no_bcast` | 1 |
| `ROW_B` | B is row vector | `_sfpu_no_bcast` | 1 |

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: binary_ng program factory architecture, kernel selection, OpConfig for DIV
- `tenstorrent/tt-llk`: `_sfpu_reciprocal_` implementation details, Newton-Raphson iterations, architecture differences between Wormhole B0 and Blackhole

### Confluence References
Not consulted for this analysis. The SFPU division implementation was fully documented in source code and DeepWiki.

### Glean References
Not consulted for this analysis. No confidential hardware specs were needed beyond what was available in source code and DeepWiki.

---

## File Inventory

### Host-Side (Program Factory)
| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` | Program factory: creates kernels, CBs, sets runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` | OpConfig, KernelName enum, utility declarations |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` | OpConfig implementation, get_sfpu_init_fn, kernel path resolution |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/types.hpp` | Type aliases (BinaryOpType from binary::BinaryOpType) |

### Device-Side (Kernels)
| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | Compute kernel: no broadcast SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp` | Compute kernel: broadcast SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp` | Compute kernel: scalar SFPU binary |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp` | Macro utilities: HAS_ACTIVATIONS, BCAST_OP |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp` | PREPROCESS macro for activation preprocessing |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp` | Reader kernel: reads A (and B) tiles from DRAM |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp` | Writer kernel: fills scalar tile, writes output |

### SFPU / LLK Layer
| File | Purpose |
|------|---------|
| `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` | Compute API: div_binary_tile, div_binary_tile_init |
| `tt_metal/hw/inc/api/compute/div_int32_sfpu.h` | Compute API: div_int32_tile, div_int32_tile_init |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` | LLK dispatch: connects API to SFPU functions |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_div_int32.h` | LLK dispatch for INT32 division |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` | SFPU kernel: calculate_sfpu_binary_div (float) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_div_int32.h` | SFPU kernel: calculate_div_int32 |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h` | Reciprocal wrapper: sfpu_reciprocal |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` | SFPU init infrastructure |
