# LOGADDEXP Operation Analysis (binary_ng SFPU variant)

## Operation Overview

**Operation**: `LOGADDEXP`
**Type**: Binary elementwise (binary_ng framework)
**Mathematical Definition**: `logaddexp(a, b) = log(exp(a) + exp(b))`
**SFPU Classification**: Composite SFPU operation -- decomposes into three SFPU stages (EXP + ADD + LOG) rather than using a single dedicated SFPU kernel.

### Key Design Decision: Composite Decomposition

LOGADDEXP is NOT implemented as a monolithic SFPU kernel. Instead, the `binary_ng` framework decomposes it into:
1. **LHS preprocessing**: `exp(a)` -- unary SFPU EXP applied to input A
2. **RHS preprocessing**: `exp(b)` -- unary SFPU EXP applied to input B
3. **Binary operation**: `exp(a) + exp(b)` -- SFPU binary ADD
4. **Postprocessing**: `log(exp(a) + exp(b))` -- unary SFPU LOG applied to the sum

This decomposition is defined in `OpConfig::OpConfig()` (see binary_ng_utils.cpp, lines 226-231) and leverages the binary_ng framework's activation chaining mechanism. The advantage is code reuse across operations that share sub-operations (e.g., LOGADDEXP2 uses EXP2+ADD+LOG2, LDEXP uses EXP2+MUL, HYPOT uses SQUARE+ADD+SQRT).

### SFPU vs FPU Path Selection

LOGADDEXP uses the SFPU path when both inputs are `FLOAT32` (see `is_binary_sfpu_op()` in binary_ng_device_operation.cpp, lines 32-36). For `BFLOAT16` inputs, it falls back to the FPU path where the binary ADD is executed on the matrix engine (FPU) instead of the SFPU, while the EXP and LOG pre/post-processing still run as SFPU unary operations.

---

## Program Factory

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

### Program Structure

The program factory creates a single program with three kernels (reader, compute, writer) distributed across a worker grid. The factory handles:

1. **Tensor shape analysis**: Extracts 5D shape dimensions (D, N, C, Ht, Wt) from padded shapes
2. **Sharding support**: Detects and configures native L1 sharding, block sharding, height/width sharding
3. **Work distribution**: Splits output tiles across cores using `split_work_to_cores`
4. **Broadcast support**: Handles scalar, row, column, and mixed broadcast types

### Circular Buffers

| CB Index | Name | Purpose | Tile Count |
|----------|------|---------|------------|
| c_0 | cb_src_a | Input tensor A | 2 (double-buffered) or shard volume |
| c_1 | cb_src_b | Input tensor B | 2 (double-buffered) or shard volume |
| c_2 | cb_out | Output tensor C | 2 (double-buffered) or shard volume |
| c_3 | cb_intermediate_a | LHS activation intermediate (EXP result) | 1 |
| c_4 | cb_intermediate_b | RHS activation intermediate (EXP result) | 1 |

For LOGADDEXP specifically, CBs c_3 and c_4 are allocated because `PROCESS_LHS_ACTIVATIONS` and `PROCESS_RHS_ACTIVATIONS` are non-empty (they contain the EXP init+op). The intermediate data format for SFPU operations matches the input data format (since `is_sfpu_op` is true), whereas for FPU operations with EXP it would use `Float16_b` (line 645-647).

### Compute Configuration

- **fp32_dest_acc_en**: Enabled when output is Float32, Int32, UInt32, or both inputs are Float32/Int32/UInt32
- **UnpackToDestMode**: For SFPU ops (non-POWER), all source CBs use `UnpackToDestFp32` -- this means data is unpacked directly into DEST registers in FP32 format, bypassing the FPU math pipeline entirely. This is critical for SFPU operations because the SFPU reads/writes DEST registers directly.
- **Compile-time args**: `{num_tiles_per_cycle}` where `num_tiles_per_cycle = 1`

### Compile-Time Defines Generated

For LOGADDEXP with SFPU path, the `OpConfig::as_defines()` method generates:

```
BINARY_SFPU_INIT = "add_binary_tile_init();"
BINARY_SFPU_OP   = "add_binary_tile"
```

Additionally, `add_activation_defines()` generates for the LHS, RHS, and POST activations:

```
PROCESS_LHS_ACTIVATIONS(i) = "exp_tile_init(); exp_tile(i);"
PROCESS_RHS_ACTIVATIONS(i) = "exp_tile_init(); exp_tile(i);"
PROCESS_POST_ACTIVATIONS(i) = "log_tile_init(); log_tile(i);"
```

The `HAS_ACTIVATIONS(LHS)`, `HAS_ACTIVATIONS(RHS)`, and `HAS_ACTIVATIONS(POST)` macros evaluate to `1` because these strings are non-empty.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
(Selected for `SubtileBroadcastType::NONE` via `get_reader_kernel_name_and_defines`)

The reader kernel reads tiles from both input tensors A and B from DRAM into L1 circular buffers c_0 and c_1. It handles:
- N-dimensional broadcasting with stride calculations per dimension
- Sharded memory layouts (bypasses NoC reads when data is already in L1)
- TensorAccessor-based addressing for correct page mapping

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
(Selected for two-tensor case)

The writer kernel reads computed tiles from output CB c_2 and writes them to DRAM. It mirrors the reader's N-dimensional iteration pattern for correct output tile placement.

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
(Selected for `SubtileBroadcastType::NONE` with `is_sfpu=true`)

This is the primary compute kernel for LOGADDEXP. It orchestrates the full SFPU pipeline.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Include headers for all SFPU unary operations (EXP, LOG, etc.)
// These are conditionally compiled based on SFPU_OP_*_INCLUDE defines
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Include headers for all SFPU binary operations (ADD, SUB, MUL, DIV, etc.)
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

// Macro utilities for activation chaining (PREPROCESS, HAS_ACTIVATIONS, etc.)
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    // Runtime arg 0: number of output tiles this core is responsible for
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Compile-time arg 0: tiles produced per read-compute-write cycle (always 1 for binary_ng)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    // CB assignments:
    // cb_pre_lhs/rhs = raw input CBs (c_0, c_1)
    // cb_post_lhs/rhs = after preprocessing (c_3, c_4 if activations exist, otherwise same as pre)
    // cb_out = output CB (c_2)
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // For LOGADDEXP: HAS_ACTIVATIONS(LHS) = 1, HAS_ACTIVATIONS(RHS) = 1
    // So cb_post_lhs = c_3 (intermediate after EXP), cb_post_rhs = c_4 (intermediate after EXP)
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    // Initialize the unary op common state (configures pack/unpack for the given CB pair)
    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // For LOGADDEXP: Both LHS and RHS have activations AND POST has activations.
    // So BINARY_SFPU_INIT is deferred to inside the loop (see #if HAS_ACTIVATIONS(POST) block below).
    // This is because the unary operations (EXP, LOG) reconfigure SFPU state,
    // so the binary ADD init must be re-run after each preprocessing step.
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT  // expands to: add_binary_tile_init();
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS LHS: Apply exp() to input A
        // Expands to the PREPROCESS_1 macro which:
        //   1. Reconfigures pack data format for intermediate CB
        //   2. Waits for input tile in cb_pre_lhs
        //   3. Acquires DEST registers
        //   4. Copies tile from cb_pre_lhs to DEST via copy_tile
        //   5. Runs PROCESS_LHS_ACTIVATIONS(i) = "exp_tile_init(); exp_tile(i);" on DEST
        //   6. Packs result to cb_post_lhs (c_3)
        //   7. Pops the input tile from cb_pre_lhs
        //   8. Reconfigures pack data format back to output format
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);

        // PREPROCESS RHS: Apply exp() to input B (same flow as LHS)
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);

        // Reserve space in output CB for the result tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // For LOGADDEXP: This condition is false because POST also has activations
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        // Acquire DEST register bank for the binary operation + postprocessing
        tile_regs_acquire();

        // Copy LHS (exp(a)) from cb_post_lhs into DEST at even indices (0, 2, 4, ...)
        // copy_tile_to_dst_init_short_with_dt reconfigures the unpacker for the source CB's data type
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // DEST[0] = exp(a)
        }

        // Copy RHS (exp(b)) from cb_post_rhs into DEST at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // DEST[1] = exp(b)

            // For LOGADDEXP: HAS_ACTIVATIONS(POST) = 1, so BINARY_SFPU_INIT runs here.
            // This re-initializes the SFPU for binary ADD after each copy_tile,
            // because the LOG postprocessing (which runs after the ADD) will have
            // reconfigured SFPU state on the previous iteration.
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT  // expands to: add_binary_tile_init();
#endif
            // Execute the binary ADD: DEST[0] = DEST[0] + DEST[1] = exp(a) + exp(b)
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);  // expands to: add_binary_tile(0, 1, 0);

            // Apply postprocessing: log(DEST[0]) = log(exp(a) + exp(b))
            PROCESS_POST_ACTIVATIONS(i * 2);  // expands to: log_tile_init(); log_tile(0);
        }
        // Signal that DEST registers are ready for packing
        tile_regs_commit();

        // Wait for DEST to be available for reading
        tile_regs_wait();

        // Pack result from DEST to output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        // Release DEST registers for next iteration
        tile_regs_release();

        // Push completed output tile and pop consumed input tiles
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
    }
}
```

#### PREPROCESS Macro Expansion

The `PREPROCESS` macro (from `eltwise_utils_sfpu.hpp`) is the mechanism that applies unary SFPU operations to input tiles before the binary operation. For LOGADDEXP, each PREPROCESS call runs exp() on one input:

```cpp
// From eltwise_utils_sfpu.hpp:
#define PREPROCESS_1(op, cb_pre, cb_post, cb_out, per_core_block_size)  \
    do {                                                                \
        using namespace ckernel;                                        \
        pack_reconfig_data_format(/*old*/ cb_out, /*new*/ cb_post);     \
        cb_wait_front(cb_pre, per_core_block_size);                     \
        cb_reserve_back(cb_post, per_core_block_size);                  \
        tile_regs_acquire();                                            \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            copy_tile_to_dst_init_short(cb_pre);                        \
            copy_tile(cb_pre, i, i);                                    \
            PROCESS_ACTIVATIONS(op, i);  /* exp_tile_init(); exp_tile(i); */ \
        }                                                               \
        tile_regs_commit();                                             \
        tile_regs_wait();                                               \
        for (uint32_t i = 0; i < per_core_block_size; ++i) {           \
            pack_tile(i, cb_post);                                      \
        }                                                               \
        tile_regs_release();                                            \
        cb_pop_front(cb_pre, per_core_block_size);                      \
        cb_push_back(cb_post, per_core_block_size);                     \
        pack_reconfig_data_format(/*old*/ cb_post, /*new*/ cb_out);     \
    } while (0)
```

This means for LOGADDEXP, each tile goes through 3 separate DEST acquire/release cycles:
1. PREPROCESS LHS: load A -> exp(A) -> pack to intermediate
2. PREPROCESS RHS: load B -> exp(B) -> pack to intermediate
3. Main compute: load exp(A) and exp(B) -> add -> log -> pack to output

---

### SFPU Kernel Implementation

This section provides a deep dive into the three SFPU kernels that compose the LOGADDEXP operation.

#### 1. SFPU Binary ADD Kernel

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
(Blackhole variant; Wormhole variant is structurally identical)

##### Annotated SFPU Binary ADD Source

```cpp
// The core binary SFPU dispatch function. For LOGADDEXP, BINOP = BinaryOp::ADD.
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    // SFPU processes 8 rows of 4 elements each per face.
    // ITERATIONS=8 means we process all 8 rows of one face per call.
    // The caller (_llk_math_eltwise_binary_sfpu_params_) handles face iteration.
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile in DEST occupies 32 rows when accessed via SFPI (64 / SFP_DESTREG_STRIDE=2).
        // dst_index_in0 * 32 gives the base row offset for tile 0.
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

        // Load a vector of 4 elements (one row) from each input tile in DEST
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        // Compile-time dispatch based on operation type.
        // For LOGADDEXP, only the ADD branch executes.
        if constexpr (BINOP == BinaryOp::ADD)
        {
            // SFPU vector addition: processes 4 FP32 elements in parallel.
            // Compiles to SFPADD instruction (or equivalent SFPI intrinsic).
            result = in0 + in1;
        }
        else if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1;
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            result = in0 * _sfpu_reciprocal_<2>(in1);
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        else if constexpr (BINOP == BinaryOp::POW)
        {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }
        else if constexpr (BINOP == BinaryOp::XLOGY)
        {
            v_if ((in1 < 0.0f) || (in1 == nan))
            {
                result = nan;
            }
            v_else
            {
                sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in1;
                _calculate_log_body_<false>(0, dst_index_out);
                result = sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] * in0;
            }
            v_endif;
        }

        // Write the result back to DEST at the output tile location
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the DEST register pointer to the next row (auto-increment)
        sfpi::dst_reg++;
    }
}

// Initialization for binary SFPU operations.
// For ADD, this is a no-op (no special init required).
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // For ADD, SUB, MUL, RSUB: no initialization needed
}
```

##### LLK Dispatch Layer

The binary SFPU operation flows through these LLK layers:

```
add_binary_tile(idst0, idst1, odst)
  -> llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(idst0, idst1, odst)
    -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(
         calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>,
         idst0, idst1, odst, VectorMode::RC)
      -> For each of 4 faces:
           sfpu_func(dst_index_in0, dst_index_in1, dst_index_out)
           // = calculate_sfpu_binary which calls _calculate_sfpu_binary_
           TTI_SETRWC(...)  // Advance DEST register pointer by 16 rows (2 blocks of 8)
```

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`

The params function handles the face iteration pattern for a 32x32 tile (4 faces of 16x16):

```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode,
    Args&&... args)
{
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0);

    // VectorMode::RC = process all 4 faces (Row-Column = full tile)
    if (mode == VectorMode::RC) {
        for (int face = 0; face < 4; face++) {
            sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, args...);
            // Advance DEST pointer by 16 rows (8 rows * 2 via SETRWC)
            TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D);
            TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D);
        }
    }

    _llk_math_eltwise_binary_sfpu_done_();
}
```

#### 2. SFPU Unary EXP Kernel

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`

The EXP kernel has multiple implementation paths:
- **FAST_APPROX + APPROXIMATION_MODE + CLAMP_NEGATIVE**: Uses SFPLOADMACRO-based pipeline with SWAP for input clamping and MAD+ROUND+SHIFT for Schraudolph's fast exp approximation. This is the highest-throughput path.
- **FAST_APPROX + APPROXIMATION_MODE (no clamp)**: Uses replay buffer with SFPSHFT2 instructions for pipelined execution at ~2.5 cycles/element.
- **APPROXIMATION_MODE (not fast)**: Uses the piecewise approximation with `_calculate_exponential_approx_` which converts to fixed-point, adds bias, and shifts to produce the IEEE 754 exponent.
- **Precise mode**: Uses `_sfpu_exp_` with Horner-form polynomial evaluation and iterative squaring.

##### Core EXP Function (Precise Mode)

```cpp
sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // Extract exponent; if >= 0, normalize input to [-1, 0) range
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);  // Set exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Horner-form polynomial approximation for exp(x) where x in [-1, 0):
    // exp(x) ~ x * (x * 0.8373 + 0.863281) + 1.0
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val = val * tmp + sfpi::vConst1;

    // For large exponents, iteratively square the result:
    // exp(x) = exp(mantissa) * 2^exp = exp(mantissa)^(2^exp)
    v_if (exp >= 0)
    {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0);  // Narrow predication: only continue if exp still >= 0
            val = val * val;
        }
    }
    v_endif;

    return val;
}
```

For negative inputs, the precise mode computes `exp(|x|)` then takes the reciprocal:

```cpp
sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(in, 0));  // exp(|x|)
v_if (in < 0)
{
    result = _sfpu_reciprocal_<2>(result);  // 1/exp(|x|) = exp(-|x|) = exp(x)
}
v_endif;
```

##### EXP Initialization

For precise mode, initializes the reciprocal LUT needed for negative input handling:
```cpp
_init_sfpu_reciprocal_<false>();
```

For approximation mode, loads constants into programmable float registers:
```cpp
sfpi::vConstFloatPrgm0 = 1.442695f;     // 1/ln(2)
sfpi::vConstFloatPrgm1 = C23_73;         // FP conversion constant
sfpi::vConstFloatPrgm2 = ADJ_EXP;        // Exponent adjustment
```

#### 3. SFPU Unary LOG Kernel

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h`

##### Annotated LOG Body

```cpp
template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(
    const std::uint32_t log_base_scale_factor,
    const std::uint32_t dst_idx = 0)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    // Load value from DEST register
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];

    // Normalize to [1, 2) by setting exponent to bias (127)
    // This extracts the mantissa portion for polynomial evaluation
    sfpi::vFloat x = setexp(in, 127);

    // 3rd-order Chebyshev polynomial approximation for ln(x) where x in [1, 2)
    // Uses Horner form on (x-1), with pre-computed coefficients:
    //   A' = 0.1058, B' = -0.7116, C' = 2.0871, D' = -1.4753
    // These are loaded from programmable constant registers
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;  // 0.1058
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;  // -0.7166
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

    // Extract and convert the original exponent to float
    // This gives us the integer part of log2(in)
    sfpi::vInt exp = sfpi::exexp(in);
    v_if (exp < 0)
    {
        exp = sfpi::setsgn(~exp + 1, 1);  // Negate: two's complement -> sign-magnitude
    }
    v_endif;

    sfpi::vFloat expf = int32_to_float(exp, 0);

    // Final result: ln(in) = ln(mantissa) + exponent * ln(2)
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;  // 0.692871
    sfpi::vFloat result = expf * vConstLn2 + series_result;

    // Optional base scaling for log2, log10, etc.
    if constexpr (HAS_BASE_SCALING)
    {
        result *= sfpi::s2vFloat16a(log_base_scale_factor);
    }

    // Handle special case: ln(0) = -infinity
    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}
```

##### LOG Initialization

```cpp
template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.692871f;   // ln(2)
    sfpi::vConstFloatPrgm1 = 0.1058f;     // Polynomial coefficient A'
    sfpi::vConstFloatPrgm2 = -0.7166f;    // Polynomial coefficient B'
}
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Used In | Description |
|----------------------|---------|-------------|
| `sfpi::dst_reg[]` | Binary ADD, EXP, LOG | Load/store 4 FP32 elements from/to DEST register bank |
| `sfpi::dst_reg++` | Binary ADD | Auto-increment DEST row pointer |
| `sfpi::vFloat + vFloat` | Binary ADD | SFPU vector floating-point addition (SFPADD) |
| `sfpi::vFloat * vFloat` | EXP, LOG | SFPU vector floating-point multiply (SFPMUL) |
| `sfpi::exexp()` | EXP, LOG | Extract exponent field from FP32 value |
| `sfpi::setexp()` | EXP, LOG | Set exponent field of FP32 value (normalize to range) |
| `sfpi::setsgn()` | EXP | Set sign bit of FP32 value |
| `sfpi::int32_to_float()` | EXP, LOG | Convert integer to float (for exponent as float) |
| `sfpi::float_to_int16()` | EXP (power path) | Convert float to 16-bit integer |
| `sfpi::s2vFloat16b()` | EXP | Load scalar constant as bfloat16 vector |
| `v_if / v_endif` | EXP, LOG | SFPU predicated execution (condition codes) |
| `v_and()` | EXP | Narrow predication (AND with existing predicate) |
| `_sfpu_reciprocal_<2>()` | EXP (precise) | 2-iteration Newton-Raphson reciprocal |
| `TTI_SETRWC` | LLK params | Set Read/Write Counter for DEST pointer advancement |
| `TTI_SFPLOADMACRO` | EXP (fast approx) | Execute macro instruction sequence from SFPU macro registers |
| `TTI_SFPSHFT2` | EXP (fast approx) | Shift operation for fast exp approximation |
| `TTI_SFPMAD` | EXP (fast approx) | Multiply-accumulate for Schraudolph algorithm |
| `TTI_SFP_STOCH_RND` | EXP (fast approx) | Stochastic rounding FP32 to INT16 |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| DEST[0] (tile index * 32) | Input A tile / exp(A) result / ADD result / log() result |
| DEST[1] (tile index * 32) | Input B tile / exp(B) operand for ADD |
| `sfpi::vConstFloatPrgm0` | LOG: ln(2) = 0.692871; EXP (approx): 1/ln(2) = 1.442695 |
| `sfpi::vConstFloatPrgm1` | LOG: polynomial coeff A' = 0.1058; EXP (approx): C23_73 constant |
| `sfpi::vConstFloatPrgm2` | LOG: polynomial coeff B' = -0.7166; EXP (approx): ADJ_EXP constant |
| `sfpi::vConst0p8373` | EXP (precise): Horner polynomial coefficient |
| `sfpi::vConst1` | EXP (precise): Constant 1.0 |
| LREG[12-14] | EXP (fast approx): Algorithm constants A, B-C, threshold/-88.5 |
| Macro registers 0-7 | EXP (fast approx): Programmed instruction sequences |

#### SFPU Execution Flow

The complete LOGADDEXP execution for one tile proceeds as follows:

1. **Reader kernel** reads tiles A and B from DRAM into CBs c_0 and c_1 via NoC.

2. **PREPROCESS LHS** (EXP on input A):
   a. `cb_wait_front(c_0, 1)` -- wait for input A tile
   b. `cb_reserve_back(c_3, 1)` -- reserve intermediate buffer
   c. `tile_regs_acquire()` -- acquire DEST registers
   d. `copy_tile(c_0, 0, 0)` -- unpack tile from c_0 to DEST[0] (uses UnpackToDestFp32 mode)
   e. `exp_tile_init()` -- initialize EXP SFPU state (loads constants into programmable registers)
   f. `exp_tile(0)` -- run EXP on DEST[0] in-place, processing all 4 faces (32 rows total)
   g. `tile_regs_commit()` / `tile_regs_wait()` -- synchronize
   h. `pack_tile(0, c_3)` -- pack DEST[0] to intermediate CB c_3
   i. `tile_regs_release()` -- release DEST
   j. `cb_pop_front(c_0, 1)` / `cb_push_back(c_3, 1)` -- advance CB pointers

3. **PREPROCESS RHS** (EXP on input B): Same flow as step 2 but with c_1 -> c_4.

4. **Main compute** (ADD + LOG):
   a. `cb_wait_front(c_3, 1)` and `cb_wait_front(c_4, 1)` -- wait for preprocessed tiles
   b. `cb_reserve_back(c_2, 1)` -- reserve output buffer
   c. `tile_regs_acquire()` -- acquire DEST
   d. `copy_tile(c_3, 0, 0)` -- unpack exp(A) to DEST[0]
   e. `copy_tile(c_4, 0, 1)` -- unpack exp(B) to DEST[1]
   f. `add_binary_tile_init()` -- initialize binary ADD SFPU (no-op for ADD)
   g. `add_binary_tile(0, 1, 0)` -- DEST[0] = DEST[0] + DEST[1] = exp(A) + exp(B)
   h. `log_tile_init()` -- initialize LOG SFPU (loads ln(2) and polynomial coefficients)
   i. `log_tile(0)` -- DEST[0] = log(DEST[0]) = log(exp(A) + exp(B))
   j. `tile_regs_commit()` / `tile_regs_wait()` -- synchronize
   k. `pack_tile(0, c_2)` -- pack final result to output CB c_2
   l. `tile_regs_release()` -- release DEST

5. **Writer kernel** reads tile from c_2 and writes to DRAM output via NoC.

#### SFPU Configuration

- **UnpackToDestMode**: `UnpackToDestFp32` for all source CBs -- data bypasses the FPU and goes directly to DEST in FP32 format. This is essential because the SFPU operates on DEST registers directly and needs FP32 precision.
- **fp32_dest_acc_en**: Enabled when both inputs are FLOAT32, allowing DEST to accumulate in full FP32 precision.
- **APPROXIMATION_MODE**: Controlled by the `APPROX` compile-time constant (typically set via `fast_and_approximate_mode` in operation attributes). When true, EXP uses the fast Schraudolph algorithm; when false, uses Horner polynomial with iterative squaring.
- **No special SFPU init for ADD**: The `_sfpu_binary_init_<APPROX, BinaryOp::ADD>()` function is a no-op -- ADD requires no LUT or constant preloading.

#### Hardware Compatibility Notes

- Both Wormhole B0 and Blackhole use the same `_calculate_sfpu_binary_` implementation for ADD (the file is in the `tt_llk` submodule with per-arch variants, but the ADD path is identical).
- The EXP kernel has significantly different fast-approximation paths between architectures:
  - Blackhole supports the `SFPLOADMACRO`-based pipeline and `SFPSHFT2` replay buffer for fast EXP.
  - The precise EXP path (`_sfpu_exp_` with Horner form) is shared across both architectures.
- The LOG kernel uses the same polynomial approximation on both architectures.
- DEST register layout (32 rows per tile via SFPI, 4 faces per tile) is consistent across both architectures.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Queried for binary_ng program factory structure, compute kernel selection, SFPU op type passing mechanism, and LOGADDEXP decomposition strategy.
- **tenstorrent/tt-llk**: Queried for ckernel namespace binary SFPU operation handling, LLK API dispatch chain, and `_calculate_sfpu_binary_` implementation details.

### Confluence References

Not consulted for this analysis. The SFPU instructions used (basic arithmetic: ADD, MUL, exponent manipulation) are well-documented via DeepWiki and source code comments.

### Glean References

Not consulted for this analysis. The operation uses standard SFPU primitives that are fully documented in the open-source codebase.

---

## Runtime Arguments

### Compute Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | num_tiles | Number of output tiles this core processes |
| 1 | freq | Broadcast frequency (1 for no-bcast) |
| 2 | counter | Broadcast start counter (0 for no-bcast) |
| 3 | compute_scalar_value | Unused for LOGADDEXP (0) |

### Reader Kernel Runtime Args (21 args)

| Index | Name | Description |
|-------|------|-------------|
| 0 | src_addr | Input A buffer address |
| 1 | start_tile_id | Starting output tile ID for this core |
| 2 | a_num_tiles | Number of A tiles in shard (0 if interleaved) |
| 3 | c_num_tiles | Number of output tiles for this core |
| 4 | c_shard_width | Output shard width in tiles (0 if interleaved) |
| 5-8 | strides | N-D, D, N, C stride multipliers for input A |
| 9-14 | dims | D, N, C, Ht, Wt, cND output dimensions |
| 15 | src_addr_b | Input B buffer address |
| 16-19 | strides_b | N-D, D, N, C stride multipliers for input B |
| 20 | b_num_tiles | Number of B tiles in shard |

### Writer Kernel Runtime Args (11 args)

| Index | Name | Description |
|-------|------|-------------|
| 0 | dst_addr | Output buffer address |
| 1 | start_tile_id | Starting output tile ID |
| 2 | num_tiles | Number of tiles to write |
| 3 | shard_width | Output shard width (0 if interleaved) |
| 4-10 | dims | D, N, C, Ht, Wt, cND, 0 |

---

## Key Design Decisions Explained

### Why Composite Decomposition Instead of Dedicated SFPU Kernel?

The LOGADDEXP operation could theoretically be implemented as a single SFPU kernel that computes `log(exp(a) + exp(b))` in one pass through DEST. However, the composite approach was chosen because:

1. **Code reuse**: The EXP, ADD, and LOG SFPU kernels are shared across many operations. A dedicated kernel would duplicate their logic.
2. **Maintainability**: Bug fixes or optimizations to EXP/LOG automatically benefit LOGADDEXP.
3. **Flexibility**: The binary_ng framework supports arbitrary activation chaining, so LOGADDEXP falls out naturally from the `process_lhs`/`process_rhs`/`postprocess` mechanism.
4. **Trade-off**: The cost is 3 DEST acquire/release cycles per tile instead of 1, plus 2 extra pack/unpack roundtrips through intermediate CBs (c_3, c_4). For a memory-bandwidth-bound operation this overhead may be negligible.

### Why BINARY_SFPU_INIT Is Re-executed Per Tile

When `HAS_ACTIVATIONS(POST)` is true (as for LOGADDEXP), the `BINARY_SFPU_INIT` macro runs inside the inner loop, right before `BINARY_SFPU_OP`. This is necessary because `log_tile_init()` (the POST activation) overwrites the SFPU programmable constant registers (`vConstFloatPrgm0/1/2`) with LOG-specific values. The binary ADD init (`add_binary_tile_init()`) is actually a no-op for ADD, so this re-initialization has zero cost for LOGADDEXP specifically. However, for other composite operations where the binary op requires SFPU state (e.g., if DIV were used), this re-initialization would be essential.

### Why UnpackToDestFp32 for SFPU Operations

SFPU operations read from and write to DEST registers directly, bypassing the FPU. The `UnpackToDestFp32` mode ensures that data is unpacked from the source CBs directly into DEST in FP32 format, rather than going through the normal unpack-to-SrcA/SrcB path that feeds the FPU. This is the fundamental architectural distinction between FPU and SFPU compute paths.
