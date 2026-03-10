# POWER (Binary SFPU) -- Program Factory Analysis

**Operation**: `BinaryOpType::POWER` (element-wise `base ** exponent`)
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`
**Namespace**: `ttnn::operations::binary`
**Class**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`

---

## Operation Overview

The POWER operation computes `output[i] = input_a[i] ** input_b[i]` element-wise across two input tensors. It is a binary SFPU operation, meaning both operands are full tensors (not a tensor + scalar). The computation is performed entirely on the SFPU (Special Function Processing Unit) using a polynomial-approximation-based `log2 + exp2` identity: `base^pow = 2^(pow * log2(base))`.

This is the "legacy" element-wise binary SFPU variant, as opposed to the newer `binary_ng` path.

---

## Program Structure

### Program Factory Methods

| Method | Purpose |
|--------|---------|
| `create(...)` | Builds the full `Program` object: creates CBs, compiles kernels, sets initial runtime args |
| `override_runtime_arguments(...)` | Updates runtime args (buffer addresses, tile counts) for cached program reuse |

### Kernel Registration Summary

| Kernel Type | Path | Config Type |
|-------------|------|-------------|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | `ReaderDataMovementConfig` |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (or block-sharded variant) | `WriterDataMovementConfig` |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | `ComputeConfig` |

---

## Circular Buffer Configuration

| CB Index | Symbol | Purpose | Size (non-sharded) | Size (sharded) |
|----------|--------|---------|---------------------|----------------|
| `c_0` | `cb_src0` | Input A (base) | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_1` | `cb_src1` | Input B (exponent) | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_2` | `cb_out0` | Output | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_3` | (interim0) | Pre-scaled input A | `max_block_size * tile_size` (only if `SFPU_OP_INIT_PRE_IN0_0` defined) | Same |
| `c_4` | (interim1) | Pre-scaled input B | `max_block_size * tile_size` (only if `SFPU_OP_INIT_PRE_IN1_0` defined) | Same |

For POWER specifically, neither `SFPU_OP_INIT_PRE_IN0_0` nor `SFPU_OP_INIT_PRE_IN1_0` is defined (no input pre-scaling is needed). Therefore, `c_3` and `c_4` are NOT created, and `cb_inp0 = cb_in0` (c_0), `cb_inp1 = cb_in1` (c_1) in the compute kernel.

### Sharding Support

The factory supports all sharding modes:
- **Height sharded**: `max_block_size = 1`
- **Width/Block sharded**: `block_or_width_sharded = true`, affects reader tiling and writer kernel selection
- **Interleaved**: Standard double-buffered CBs with `2 * max_block_size` pages

When an input is sharded, its CB is globally allocated to the tensor buffer via `set_globally_allocated_address`, and the reader kernel uses `cb_reserve_back + cb_push_back` to "publish" the already-present data rather than performing NoC reads.

---

## Compile-Time Defines for POWER

The `get_defines_fp32` function in `binary_op_utils.cpp` generates these defines for `BinaryOpType::POWER`:

| Define | Value | Purpose |
|--------|-------|---------|
| `BINOP_INIT` | `power_binary_tile_init();` | Initializes SFPU programmable constants for power computation |
| `BINARY_SFPU_OP` | `power_binary_tile(i*2, i*2+1, i*2);` | The per-tile SFPU dispatch call. Args: `(dst_in0, dst_in1, dst_out)` |

No pre-scaling defines (`SFPU_OP_INIT_PRE_IN0_0`, `SFPU_OP_INIT_PRE_IN1_0`) are generated for POWER.
No fused activation defines (`SFPU_OP_INIT_0`, `SFPU_OP_FUNC_0`, `SFPU_OP_CHAIN_0`) are generated unless explicitly requested.

### POWER-Specific UnpackToDestMode

POWER has unique `UnpackToDestMode` handling in the program factory. Unlike other binary SFPU ops (ADD, SUB, MUL, etc.) which always force `UnpackToDestFp32`, POWER conditionally sets the unpack mode based on the actual input dtype:

```cpp
if (op_type != BinaryOpType::POWER) {
    // Other ops: always FP32 unpack
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    // ...
} else {
    // POWER: only FP32 unpack when input is actually FLOAT32
    unpack_to_dest_mode[src0_cb_index] =
        (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    // ...
}
```

**Why**: The POWER SFPU kernel internally handles bfloat16 precision via explicit `float_to_fp16b` rounding in the `_sfpu_binary_power_21f_` path. Forcing FP32 unpack for bfloat16 inputs would be wasteful and could interfere with the kernel's own precision management. The `_sfpu_binary_power_61f_` path (used when `is_fp32_dest_acc_en = true`) naturally requires FP32 data.

### FP32 Dest Accumulation

```cpp
bool fp32_dest_acc_en = (dst_cb_data_format == tt::DataFormat::Float32) ||
                        (dst_cb_data_format == tt::DataFormat::Int32) ||
                        (dst_cb_data_format == tt::DataFormat::UInt32);
```

This flag selects between two SFPU kernel variants:
- `fp32_dest_acc_en = false`: Uses `_sfpu_binary_power_21f_` (3rd-order polynomial log2 approximation)
- `fp32_dest_acc_en = true`: Uses `_sfpu_binary_power_61f_` (5th-degree polynomial log2 approximation, higher accuracy)

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

Reads tiles from two input tensors into CBs `c_0` and `c_1`. For each tile:
1. `cb_reserve_back` to claim write space
2. `noc_async_read_tile` to DMA the tile from DRAM
3. `noc_async_read_barrier` to wait for completion
4. `cb_push_back` to publish to the compute kernel

Supports sharded inputs via `IN0_SHARDED` / `IN1_SHARDED` defines (bulk reserve+push) and block/width sharded tiling patterns.

**Runtime Args**: `(src0_addr, src1_addr, num_tiles, start_id, block_height, block_width, num_cores_y)`

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Writes output tiles from CB `c_2` to DRAM. For each tile:
1. `cb_wait_front` to wait for compute kernel output
2. `noc_async_write_page` to DMA the tile to DRAM
3. `cb_pop_front` to free the CB slot

Supports sharded output via `OUT_SHARDED` define (just `cb_wait_front` with no write).

**Runtime Args**: `(dst_addr, num_pages, start_id)`

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

This is a generic binary SFPU compute kernel shared by all binary SFPU operations. The specific operation is selected via compile-time defines.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"        // provides power_binary_tile / power_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true when input pre-scaling is enabled (not the case for POWER)
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime args: number of tile blocks and tiles per block for this core
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // how many blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // input B circular buffer

    // For POWER: no PRE_IN0/PRE_IN1 defines, so cb_inp0 = cb_in0, cb_inp1 = cb_in1
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;  // pre-scaled input A (not used for POWER)
#else
    constexpr auto cb_inp0 = cb_in0;             // POWER takes this path: read directly from c_0
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;  // pre-scaled input B (not used for POWER)
#else
    constexpr auto cb_inp1 = cb_in1;             // POWER takes this path: read directly from c_1
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // output circular buffer

    // Initialize unpack/pack hardware for the data formats of cb_in0 and cb_out0
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // optional ReLU on pack (not typical for POWER)
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE-SCALING SECTION (skipped for POWER) ---
        // SFPU_OP_INIT_PRE_IN0_0 and SFPU_OP_INIT_PRE_IN1_0 blocks are not compiled for POWER.
        // These would copy tiles to DST, apply a unary SFPU op (e.g. exp for LOGADDEXP),
        // then pack results into intermediate CBs c_3/c_4.

        // --- MAIN BINARY SFPU SECTION ---
        // Wait for both input tile blocks to be available in their CBs
        cb_wait_front(cb_inp0, per_core_block_size);    // wait for base tiles in c_0
        cb_wait_front(cb_inp1, per_core_block_size);    // wait for exponent tiles in c_1
        cb_reserve_back(cb_out0, per_core_block_size);  // reserve output space in c_2

        tile_regs_acquire();  // acquire DST register file for writing
        tile_regs_wait();     // wait for DST to be ready

        // Set up unpack for input A (base) and copy all A tiles into even DST slots
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // configure unpack for cb_inp0's format
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);     // copy base tile i from c_0 into DST[i*2]
        }

        // Switch unpack format to input B (exponent) and copy+process each tile
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // configure unpack for cb_inp1's format
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1); // copy exponent tile i from c_1 into DST[i*2+1]

            // For POWER, BINOP_INIT expands to: power_binary_tile_init();
            // This sets SFPU programmable constants:
            //   vConstFloatPrgm0 = 1.442695 (1/ln(2))
            //   vConstFloatPrgm1 = -127.0   (underflow clamp threshold)
            //   vConstFloatPrgm2 = NaN       (for special case handling)
#ifdef BINOP_INIT
            BINOP_INIT           // power_binary_tile_init();
#endif

            // For POWER, BINARY_SFPU_OP expands to:
            //   power_binary_tile(i*2, i*2+1, i*2);
            // This computes: DST[i*2] = DST[i*2] ** DST[i*2+1]
            // i.e., base^exponent, result overwrites the base tile slot
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP       // power_binary_tile(i*2, i*2+1, i*2);
#endif

            // Pack result tile from DST[i*2] into output CB c_2
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();   // signal DST writes are complete
        tile_regs_release();  // release DST register file

        // Release consumed input tiles and publish output tiles
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

**Key observations for POWER**:
- Tiles from both inputs are interleaved in DST: base at even indices (0, 2, 4, ...), exponent at odd indices (1, 3, 5, ...)
- `BINOP_INIT` is called inside the inner loop (per-tile), not once before the loop. This is because the compute kernel is generic and some ops may need re-initialization per tile.
- The `pack_tile(i * 2, cb_out0)` packs the result from the same DST slot as the base, since `power_binary_tile` overwrites `idst0`.

---

### SFPU Kernel Implementation

This section provides a deep dive into the underlying SFPU kernel that implements the power function.

#### SFPU Kernel File

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`
(Identical file exists at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h`)

#### API Layer Files

- `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` -- Defines `power_binary_tile()` and `power_binary_tile_init()`
- `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_pow.h` -- LLK dispatch layer

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// --------------------------------------------------------------------------
// _sfpu_binary_power_21f_: Lower-precision variant for bfloat16 dest mode
//
// Implements base^pow = 2^(pow * log2(base)) using the "exp_21f" algorithm
// from Moroz et al. 2022 ("Simple Multiple Precision Algorithms for
// Exponential Functions").
//
// The "21f" name refers to the paper's exp_21f: a 2-multiply, 1-float
// coefficient scheme using bit manipulation techniques (BMT) for the
// exponential step.
//
// Log2 approximation: 3rd-order polynomial via rminimax over [1,2]
// Exp2 approximation: Horner-form BMT with integer+float constants
// --------------------------------------------------------------------------
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // === STEP 1: Compute log2(base) ===

    // Take absolute value of base for logarithm computation.
    // Negative bases are handled in post-processing.
    sfpi::vFloat absbase = setsgn(base, 0);       // clear sign bit => |base|

    // Normalize |base| to [1, 2) by setting exponent field to bias (127).
    // This extracts the mantissa part for polynomial evaluation.
    sfpi::vFloat x = sfpi::setexp(absbase, 127);   // SFPSETEXP: x in [1, 2)

    // Evaluate 3rd-order polynomial approximation of ln(x) for x in [1,2]
    // Coefficients determined using the rminimax algorithm.
    // Result: series_result ~ ln(x) where x is the normalized mantissa
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Extract the original exponent of base as an integer.
    // exexp returns the biased exponent minus 127 (i.e., the actual exponent).
    sfpi::vInt exp = sfpi::exexp(base);             // SFPEXEXP: extract exponent
    // Handle negative exponents: negate and set sign bit
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    // Convert integer exponent to float for addition with series result
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);  // SFPCAST: int -> float

    // Combine: log2(base) = exponent + ln(mantissa) * (1/ln(2))
    // vConstFloatPrgm0 = 1.442695 = 1/ln(2), converting ln to log2
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // === STEP 2: Compute 2^(pow * log2(base)) ===

    sfpi::vFloat z_f32 = pow * log2_result;

    // Clamp to prevent overflow: when z < -127, intermediaries overflow
    // and produce incorrect non-zero results instead of converging to 0.
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;  // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // Apply the exp_21f formula from the paper:
    // z = (bias + z_f32) * 2^23, where bias = 0x3f800000 (1.0f as int)
    // addexp(z_f32, 23) multiplies by 2^23 using a single SFPDIVP2 instruction
    // (SFPDIVP2 with positive immediate adds to the exponent field).
    z_f32 = addexp(z_f32, 23);                      // SFPDIVP2: z_f32 *= 2^23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);  // 1.0f reinterpreted
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);  // truncate to integer

    // Split z into exponent field (zii) and mantissa field (zif)
    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // exponent bits
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // mantissa bits (9-bit)

    // Evaluate 2-multiply Horner form using BMT (bit manipulation technique).
    // The paper uses integer constants added to mantissa bits to efficiently
    // compute polynomial terms without separate float multiplications.
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;                                    // SFPMUL
    zif = _float_to_int32_positive_(d2 * d3);         // SFPMUL + truncate

    // Restore the exponent: set the exponent of the mantissa result
    // to (127 + extracted_exponent), reconstructing the full float
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // === POST-PROCESSING: Handle special cases ===

    // Convert exponent to integer for parity check
    sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);  // SFPCAST: float -> int16
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // Special case: base=0, pow<0 => NaN (division by zero, e.g. 0^(-2))
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;                  // NaN from programmable constant
    }
    v_endif;

    // Special case: negative base
    v_if(base < 0.0f) {
        // For integer powers: sign = (-1)^pow
        // Shift LSB of pow_int to sign position to determine odd/even
        y = setsgn(y, pow_int << 31);                // SFPSETSGN: set sign from parity

        // Non-integer power with negative base => complex result => NaN
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;              // NaN
        }
        v_endif;
    }
    v_endif;

    // For bfloat16 destination: explicitly round to bfloat16 using round-to-nearest-even.
    // Without this, SFPSTORE would truncate, causing errors like 9^2 = 80.5 instead of 81.
    if constexpr (!is_fp32_dest_acc_en) {
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));  // round to bf16
    }

    return y;
}

// --------------------------------------------------------------------------
// _sfpu_binary_power_61f_: Higher-precision variant for FP32 dest mode
//
// Uses a 5th-degree Remez polynomial for log2 approximation (vs 3rd-order
// in 21f) and a degree-6 Chebyshev polynomial for exp2 approximation
// (vs BMT in 21f). All coefficients are pure floats.
// --------------------------------------------------------------------------
sfpi_inline sfpi::vFloat _sfpu_binary_power_61f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // === STEP 1: Compute log2(base) ===
    sfpi::vFloat abs_base = sfpi::abs(base);
    sfpi::vFloat x = sfpi::setexp(abs_base, 127);    // normalize to [1, 2)

    // 5th-degree Remez polynomial for ln(x), x in [1,2]
    // Higher degree than 21f => more accurate for FP32 results
    sfpi::vFloat series_result =
        x * (x * (x * (x * (x * 0.03101577f - 0.28807408f) + 1.1286426f) - 2.45830873f) + 3.5271965f) - 1.94046315f;

    sfpi::vInt exp = sfpi::exexp(base);
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);

    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;     // 1/ln(2)
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;

    // === STEP 2: Compute 2^(pow * log2(base)) ===
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;   // -127.0
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    z_f32 = sfpi::addexp(z_f32, 23);                             // *= 2^23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    // Normalize mantissa to fractional value in [0,1) for polynomial evaluation
    sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.1920929e-7f);  // / 2^23

    // Degree-6 Chebyshev polynomial for 2^frac, frac in [0,1)
    // All float coefficients (no BMT tricks needed for FP32 precision)
    sfpi::vFloat poly =
        ((((((0.0002170391f * frac) + 0.001243946f) * frac + 0.0096788315f) * frac + 0.055483369f) * frac +
          0.24022982f) *
             frac +
         0.69314699f) *
            frac +
        1.0000000018f;

    // Restore exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // === POST-PROCESSING (same logic as 21f) ===
    sfpi::vInt pow_int = sfpi::float_to_int16(pow, 0);
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;
    }
    v_endif;

    v_if(base < 0.0f) {
        y = sfpi::setsgn(y, pow_int << 31);
        v_if(pow_rounded != pow) {
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    // No bf16 rounding needed: FP32 dest preserves full precision
    return y;
}

// --------------------------------------------------------------------------
// Template specialization: select 21f or 61f based on dest accumulation mode
// --------------------------------------------------------------------------
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);  // bfloat16 path
}

template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_61f_(base, pow);          // FP32 path
}

// --------------------------------------------------------------------------
// calculate_sfpu_binary_pow: Top-level iteration function
//
// Called by _llk_math_eltwise_binary_sfpu_params_ once per face (4 faces
// per tile in RC mode). ITERATIONS=8 processes 8 rows per face call,
// so 4 faces * 8 rows = 32 rows = one full tile.
// --------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // Each tile occupies 32 "SFPI rows" in DST (64 actual rows / stride of 2)
        constexpr uint dst_tile_size_sfpi = 32;

        // Load one row of base and exponent from their respective DST tile slots
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // SFPLOAD from DST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // SFPLOAD from DST

        // Compute base^exponent
        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        // Store result back to DST at the output tile slot
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;  // SFPSTORE to DST
        sfpi::dst_reg++;  // advance row pointer for next iteration
    }
}

// --------------------------------------------------------------------------
// sfpu_binary_pow_init: Set programmable SFPU constants
// Called once during BINOP_INIT => power_binary_tile_init()
// --------------------------------------------------------------------------
template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;                            // 1/ln(2) for log2 conversion
    sfpi::vConstFloatPrgm1 = -127.0f;                              // underflow clamp threshold
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN(); // NaN for special cases
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| SFPI Intrinsic / Instruction | Underlying SFPU Instruction | Description |
|------------------------------|---------------------------|-------------|
| `sfpi::setsgn(v, bit)` | `SFPSETSGN` | Set or clear the sign bit of a float vector |
| `sfpi::setexp(v, exp)` | `SFPSETEXP` | Set the exponent field of a float to a constant value |
| `sfpi::exexp(v)` | `SFPEXEXP` | Extract the exponent field as a signed integer |
| `sfpi::exman9(v)` | `SFPEXMAN` | Extract the 9-bit mantissa field as an integer |
| `sfpi::abs(v)` | `SFPABS` | Absolute value (clear sign bit) |
| `sfpi::addexp(v, n)` | `SFPDIVP2` | Add `n` to the exponent field (multiply by `2^n`) |
| `sfpi::int32_to_float(v, n)` | `SFPCAST` | Convert integer to float with optional shift |
| `sfpi::float_to_int16(v, n)` | `SFPCAST` | Convert float to 16-bit integer (truncation) |
| `sfpi::float_to_fp16b(v, n)` | `SFPCAST` | Convert float32 to bfloat16 with rounding mode |
| `_float_to_int32_positive_(v)` | `SFPCAST` | Convert positive float to int32 (helper) |
| `sfpi::reinterpret<T>(v)` | (no instruction) | Reinterpret bits between vFloat and vInt |
| `v_if(...) { } v_endif` | `SFPSETCC` / `SFPENCC` | Predicated execution using SFPU condition codes |
| `operator*` | `SFPMUL` / `SFPMAD` | Floating-point multiply (MAD with zero addend) |
| `operator+` | `SFPMAD` / `SFPIADD` | Floating-point or integer add |
| `operator<<` | `SFPSHFT` | Bit shift left |
| `sfpi::dst_reg[idx]` | `SFPLOAD` / `SFPSTORE` | Load from / store to DST register at given row |
| `sfpi::dst_reg++` | `SFPINCRWC` | Increment the DST row write counter |
| `sfpi::vConstFloatPrgm0/1/2` | `SFPLOADI` (to LReg) | Programmable float constants loaded into SFPU LRegs |

#### SFPU Register Usage

**DST (Destination) Registers**:
- `dst_index_in0 * 32` through `dst_index_in0 * 32 + 31`: Base tile (input A), loaded by `copy_tile` before SFPU execution
- `dst_index_in1 * 32` through `dst_index_in1 * 32 + 31`: Exponent tile (input B), loaded by `copy_tile` before SFPU execution
- `dst_index_out * 32` through `dst_index_out * 32 + 31`: Output tile (same as in0 for POWER since `idst_out = idst_in0 = i*2`)

**SFPU LRegs (Local Registers)**:
- The SFPU has a small set of local registers (LRegs) that hold intermediate values during computation. Each SFPI instruction reads from / writes to these implicitly.
- `vConstFloatPrgm0/1/2`: Three programmable constant registers pre-loaded during init

**Key point**: The SFPU processes one row (32 elements in bfloat16, 16 in float32) per iteration. With `ITERATIONS=8` and 4 face calls from the params function (RC mode), all 32 rows of a 32x32 tile are processed.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to claim the DST register file. Then `copy_tile(cb_inp0, i, i*2)` unpacks base tiles from CB c_0 into even DST slots, and `copy_tile(cb_inp1, i, i*2+1)` unpacks exponent tiles from CB c_1 into odd DST slots.

2. **SFPU init**: `power_binary_tile_init()` is called, which invokes `sfpu_binary_pow_init<APPROX>()`. This sets three SFPU programmable constants:
   - `vConstFloatPrgm0 = 1.442695` (1/ln(2), used to convert ln to log2)
   - `vConstFloatPrgm1 = -127.0` (underflow clamp for exponent)
   - `vConstFloatPrgm2 = NaN` (special case marker)

3. **SFPU dispatch**: `power_binary_tile(i*2, i*2+1, i*2)` is called, which chains through:
   - `llk_math_eltwise_binary_sfpu_binary_pow<APPROX, DST_ACCUM_MODE>(i*2, i*2+1, i*2)`
   - `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary_pow<APPROX, 8, is_fp32_dest_acc_en>, i*2, i*2+1, i*2, VectorMode::RC)`

4. **Per-face iteration**: The params function iterates over 4 faces (RC mode). For each face, it calls `calculate_sfpu_binary_pow` which loops 8 times (8 rows per face). Each iteration:
   a. Loads one row of base from `DST[in0 * 32 + current_row]`
   b. Loads one row of exponent from `DST[in1 * 32 + current_row]`
   c. Computes `base^pow` via the `_sfpu_binary_power_` function (21f or 61f variant)
   d. Stores the result to `DST[out * 32 + current_row]`
   e. Increments the DST row pointer via `dst_reg++`
   Between faces, `TTI_SETRWC` instructions advance the DST pointer by 16 rows (2 increments of 8).

5. **Pack**: After the SFPU completes, `pack_tile(i*2, cb_out0)` reads the result from DST[i*2] and packs it into output CB c_2.

6. **CB release**: `cb_pop_front` frees the consumed input pages, `cb_push_back` publishes the output pages.

#### SFPU Configuration

| Configuration | Value | Purpose |
|--------------|-------|---------|
| `APPROXIMATION_MODE` (APPROX) | Compile-time, typically `true` | Enables fast approximation paths in SFPU init |
| `DST_ACCUM_MODE` | Set by `fp32_dest_acc_en` from program factory | Selects 21f (bfloat16) vs 61f (FP32) kernel variant |
| `ITERATIONS` | `8` (hardcoded) | 8 rows per face, 4 faces = 32 rows = full tile |
| `VectorMode::RC` | Default | Process all 4 faces (full tile), not just row or column |
| `vConstFloatPrgm0` | `1.442695f` | 1/ln(2) for log-base conversion |
| `vConstFloatPrgm1` | `-127.0f` | Exponent underflow clamp |
| `vConstFloatPrgm2` | `NaN` | Special case output value |

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations of `ckernel_sfpu_binary_pow.h` are **identical** (verified by diff). Both architectures share the same SFPU instruction set for this operation. The key architectural differences that could affect behavior:

- **DST register size**: Blackhole has larger DST, allowing more tiles to be resident simultaneously. The `get_dest_max_tiles` check in the params function enforces this limit.
- **SFPU pipeline depth**: May differ between architectures, but this is transparent to the kernel code.
- **Data format support**: Both support bfloat16 and float32 for POWER; the `UnpackToDestMode` handling in the program factory adapts accordingly.

---

## Runtime Arguments

### Reader Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `src0_addr` | DRAM address of input tensor A (base) |
| 1 | `src1_addr` | DRAM address of input tensor B (exponent) |
| 2 | `num_tiles` | Total tiles to read for this core |
| 3 | `start_id` | Starting tile ID for this core's work partition |
| 4 | `block_height` | Shard block height in tiles (0 if interleaved) |
| 5 | `block_width` | Shard block width in tiles (0 if interleaved) |
| 6 | `num_cores_y` | Number of cores in Y dimension (for row stride in block sharding) |

### Compute Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `per_core_block_cnt` | Number of tile blocks to process |
| 1 | `per_core_block_size` | Number of tiles per block |

### Writer Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | DRAM address of output tensor |
| 1 | `num_pages` | Number of tiles to write |
| 2 | `start_id` | Starting tile ID for writes |

---

## Work Distribution

The factory uses `split_work_to_cores` for interleaved tensors to divide tiles across available cores. For sharded tensors, each core processes its local shard. The work is split into two groups:
- **Core group 1**: Cores that each process `num_tiles_per_core_group_1` tiles
- **Core group 2**: Cores that each process `num_tiles_per_core_group_2` tiles (handles remainder)

The `find_max_block_size` function determines the optimal blocking factor for sharded data.

---

## Mathematical Algorithm Summary

The POWER SFPU kernel computes `base^pow` using the identity:

```
base^pow = 2^(pow * log2(base))
```

This is decomposed into:
1. **log2(base)**: Decompose base into `mantissa * 2^exponent`, compute `ln(mantissa)` via polynomial approximation over [1,2], then `log2(base) = exponent + ln(mantissa) / ln(2)`
2. **2^z** where `z = pow * log2(base)`: Use the Moroz et al. 2022 exp_21f (or 61f for FP32) algorithm, which decomposes z into integer and fractional parts and evaluates a polynomial

Two precision levels:
- **21f** (bfloat16 mode): 3rd-order rminimax log polynomial + 2-multiply BMT exponential. Faster but lower accuracy. Includes explicit bf16 rounding at the end.
- **61f** (FP32 mode): 5th-degree Remez log polynomial + degree-6 Chebyshev exponential. Slower but higher accuracy for float32 output.

Special case handling:
- `0^(negative)` => NaN
- `(negative)^(integer)` => proper sign based on parity
- `(negative)^(non-integer)` => NaN (complex result)
- Underflow clamping at `z < -127` to prevent overflow wraparound

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Binary SFPU operation structure, `get_defines_fp32` for POWER, compute kernel dispatch chain
- `tenstorrent/tt-llk`: `_llk_math_eltwise_binary_sfpu_params_` iteration pattern, face-based DST processing, `SfpuType` dispatch
- `tenstorrent/sfpi`: SFPI instruction mapping (SFPMUL, SFPSETEXP, SFPEXEXP, etc.), `dst_reg` access patterns

### Confluence References
Not consulted for this analysis. The SFPU instruction details were sufficiently covered by the inline source documentation and DeepWiki.

### Glean References
Not consulted for this analysis. The Wormhole/Blackhole implementations are identical, and no confidential hardware specs were needed beyond what is available in the source code.
