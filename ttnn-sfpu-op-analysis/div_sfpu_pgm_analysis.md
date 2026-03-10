# DIV (Legacy Element-Wise Binary SFPU) -- Operation Analysis

## Overview

The DIV operation computes element-wise floating-point division of two input tensors: `output = input_a / input_b`. It is implemented as a binary SFPU operation in the legacy `element_wise_multi_core_sfpu` program factory, which loads both input tiles into the DEST register file and then invokes an SFPU kernel that computes division as `input_a * reciprocal(input_b)`.

There are two code paths for DIV depending on data types:
- **FP32/BF16 path** (this analysis): Uses `div_binary_tile` which dispatches to `calculate_sfpu_binary_div` on the SFPU. This is the primary path analyzed here.
- **INT32 path**: Uses `div_int32_tile` for integer division. Not covered in depth.

**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

---

## Program Factory Architecture

### Class and Method

- **Namespace**: `ttnn::operations::binary`
- **Class**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`
- **Methods**:
  - `create(...)` -- builds the `Program` with all kernels and circular buffers
  - `override_runtime_arguments(...)` -- updates runtime arguments for subsequent runs without rebuilding the program

### Operation Selection

The SFPU program factory is selected when the operation type (e.g., `BinaryOpType::DIV`) is identified as requiring SFPU execution. This decision is made in `binary_device_operation.cpp` where DIV (with float types) routes to `ElementWiseMultiCoreSfpu`.

### Compile-Time Defines for DIV

The function `get_defines_fp32()` in `binary_op_utils.cpp` generates the compile-time defines that configure the generic SFPU compute kernel for the DIV operation. For floating-point DIV:

```
BINOP_INIT  ->  "div_binary_tile_init();"
BINARY_SFPU_OP  ->  "div_binary_tile(i*2, i*2+1, i*2);"
```

- `i*2` is the DST index of input A (and also the output destination)
- `i*2+1` is the DST index of input B
- Both tiles are loaded into adjacent DST slots so the SFPU can operate on them in-place

---

## Circular Buffer Configuration

| CB Index | Name | Purpose | Page Count (interleaved) | Data Format |
|----------|------|---------|--------------------------|-------------|
| `c_0` | `cb_src0` | Input tensor A | `2 * max_block_size` (or `num_tiles_per_shard` if sharded) | Matches input A dtype |
| `c_1` | `cb_src1` | Input tensor B | `2 * max_block_size` (or `num_tiles_per_shard` if sharded) | Matches input B dtype |
| `c_2` | `cb_out0` | Output tensor | `2 * max_block_size` (or `num_tiles_per_shard` if sharded/block-sharded) | Matches output dtype |
| `c_3` | `cb_interm0` | Intermediate for pre-scaled input A | `max_block_size` (only if `SFPU_OP_INIT_PRE_IN0_0` defined) | Same as input A format |
| `c_4` | `cb_interm1` | Intermediate for pre-scaled input B | `max_block_size` (only if `SFPU_OP_INIT_PRE_IN1_0` defined) | Same as input B format |

For the basic DIV operation (no fused activations or input activations), `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are **not** defined, so `c_3` and `c_4` are not created.

### Sharding Support

The factory supports all three sharding modes:
- **Height-sharded**: Standard 1D partitioning along the height dimension
- **Width-sharded**: Partitioning along the width dimension
- **Block-sharded**: 2D partitioning

When a tensor is sharded, its circular buffer is globally allocated (backed by the tensor's L1 buffer directly) and the reader/writer kernels skip NoC transfers for that tensor.

### UnpackToDestMode

For all non-POWER binary SFPU operations (including DIV), all input circular buffers use `UnpackToDestMode::UnpackToDestFp32`. This ensures that regardless of the input data format (BF16, FP16, etc.), data is unpacked to full FP32 precision in the DEST register before SFPU processing. This is critical because the SFPU always operates in FP32.

### FP32 Dest Accumulation

`fp32_dest_acc_en` is set to `true` when the output data format is `Float32`, `Int32`, or `UInt32`. This affects whether the SFPU kernel applies BF16 rounding to the result before writing back to DEST.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

This reader kernel handles both interleaved and sharded inputs:

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments provided by the program factory
    uint32_t src0_addr = get_arg_val<uint32_t>(0);       // DRAM address of input tensor A
    uint32_t src1_addr = get_arg_val<uint32_t>(1);       // DRAM address of input tensor B
    uint32_t num_tiles = get_arg_val<uint32_t>(2);        // Total tiles to process on this core
    uint32_t start_id = get_arg_val<uint32_t>(3);         // Starting tile ID for this core
    uint32_t block_height = get_arg_val<uint32_t>(4);     // Block height for sharded layouts
    uint32_t block_width = get_arg_val<uint32_t>(5);      // Block width for sharded layouts
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);      // Number of cores along Y for sharded striding

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;     // CB for input A
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;     // CB for input B
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;

    // TensorAccessor compile-time args are conditionally present
    // depending on which inputs are sharded vs interleaved

    // For sharded inputs: simply reserve and push the entire shard
    // (data is already in L1, the CB is globally allocated on top of the tensor buffer)
#ifdef IN0_SHARDED
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#else
    // For interleaved inputs: read tiles one at a time from DRAM via NoC
    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif

#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, num_tiles);
    cb_push_back(cb_id_in1, num_tiles);
#else
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);
#endif

    // Main read loop -- two modes depending on sharding type
    // block_or_width_sharded: iterate (h, w) with strided tile IDs
    // otherwise: simple sequential tile iteration
    // Each iteration: cb_reserve_back -> noc_async_read_tile -> barrier -> cb_push_back
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

Standard unary writer: waits for output tiles in `cb_out0`, writes them to DRAM one at a time.

```cpp
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);   // DRAM address of output tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of pages to write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting page ID

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2)

#ifdef OUT_SHARDED
    // For sharded output: just wait for compute to finish writing all pages
    cb_wait_front(cb_id_out, num_pages);
#else
    // For interleaved output: drain one page at a time to DRAM
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);
        noc_async_writes_flushed();
        cb_pop_front(cb_id_out, 1);
    }
    noc_async_write_barrier();
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

This is the generic binary SFPU compute kernel. It is parameterized entirely through compile-time `#define` macros set by the program factory. For DIV, the active defines are `BINOP_INIT` and `BINARY_SFPU_OP`.

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
#include "api/compute/eltwise_binary_sfpu.h"     // Provides div_binary_tile() and div_binary_tile_init()
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

// PRE_SCALE is true if either input needs pre-processing (e.g., for LOGADDEXP, LOGICAL_OR)
// For basic DIV, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined,
// so PRE_SCALE evaluates to false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime arguments: how many blocks and tiles per block this core processes
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // Number of tile blocks
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);  // Tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // Input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // Input B circular buffer

    // For DIV: no pre-scaling, so cb_inp0 == cb_in0 and cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;  // Pre-processed input A
#else
    constexpr auto cb_inp0 = cb_in0;             // Direct: input A
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;  // Pre-processed input B
#else
    constexpr auto cb_inp1 = cb_in1;             // Direct: input B
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // Output circular buffer

    // Initialize unpack/pack hardware for the input/output CB pair
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- Pre-scaling phases (skipped for basic DIV) ---
        // SFPU_OP_INIT_PRE_IN0_0: would copy input A tiles to DST, apply a unary SFPU op,
        //   pack results to c_3, then use c_3 as the effective input A.
        // SFPU_OP_INIT_PRE_IN1_0: same pattern for input B -> c_4.
        // These are used by compound operations like LOGADDEXP (exp both inputs before adding).

        // --- Main compute phase ---
        // Wait for both inputs to be available in their respective CBs
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        // Acquire DEST registers for SFPU use
        tile_regs_acquire();
        tile_regs_wait();

        // Copy all input A tiles into even DST slots (0, 2, 4, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);       // CB page i -> DST[i*2]
        }

        // Copy all input B tiles into odd DST slots (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);   // CB page i -> DST[i*2+1]

            // For DIV, BINOP_INIT expands to: div_binary_tile_init();
            // This initializes the SFPU reciprocal polynomial constants.
#ifdef BINOP_INIT
            BINOP_INIT                           // div_binary_tile_init();
#endif
            // BINARY_SFPU_OP expands to: div_binary_tile(i*2, i*2+1, i*2);
            // Computes DST[i*2] = DST[i*2] / DST[i*2+1], i.e. A / B.
            // Result overwrites the input A slot in DST.
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP                       // div_binary_tile(i*2, i*2+1, i*2);
#endif

            // Pack the result tile from DST[i*2] to the output CB
            pack_tile(i * 2, cb_out0);
        }

        // Release DEST registers
        tile_regs_commit();
        tile_regs_release();

        // Signal that input tiles are consumed and output tiles are produced
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

---

## SFPU Kernel Implementation

This section provides a deep dive into the SFPU kernel function that the compute kernel dispatches to when `div_binary_tile()` is called.

### Call Chain

```
div_binary_tile(idst0, idst1, odst)
  -> MATH((llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>(idst0, idst1, odst)))
    -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(
           calculate_sfpu_binary_div<APPROX, BinaryOp::DIV, 8, is_fp32_dest_acc_en>,
           dst_index0, dst_index1, odst, VectorMode::RC)
      -> calculate_sfpu_binary_div<...>(dst_index_in0, dst_index_in1, dst_index_out)
```

The initialization chain:
```
div_binary_tile_init()
  -> MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>()))
    -> llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(sfpu_binary_init<APPROX, BinaryOp::DIV>)
      -> _llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()   // HW-level SFPU init
      -> _sfpu_binary_init_<APPROX, BinaryOp::DIV>()
        -> _init_sfpu_reciprocal_<false>()                          // Load polynomial constants
```

### SFPU Kernel File (Wormhole B0 -- metal overlay)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`

This file is the architecture-specific overlay. For DIV, it delegates to `calculate_sfpu_binary_div` which is defined in the same file.

### SFPU Kernel File (tt_llk common -- Wormhole B0)

**File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`

This contains the shared `_calculate_sfpu_binary_` template and `_sfpu_binary_init_`.

### Annotated SFPU Kernel Source (Wormhole B0 metal overlay -- `calculate_sfpu_binary_div`)

```cpp
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Each tile in DEST occupies 32 rows when accessed via SFPI (64/SFP_DESTREG_STRIDE).
    // The SFPU processes 32 elements at a time (one "row" = one SIMD lane per face).
    // 8 iterations cover all 8 sub-sections of a 32x32 tile (4 faces x 2 halves = 8).
    constexpr uint dst_tile_size_sfpi = 32;

    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row of input A from DEST register at the input A tile offset
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        // Load one row of input B from DEST register at the input B tile offset
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Core division: multiply input A by the reciprocal of input B.
        // _sfpu_reciprocal_<2> uses 2 Newton-Raphson iterations for FP32 precision.
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        // Handle special cases for division by zero
        v_if(in1 == 0) {
            // 0/0 = NaN (indeterminate form)
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                // nonzero/0 = +/-infinity, with sign matching input A
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        // Handle exact equality: a/a = 1.0 (avoids reciprocal rounding errors)
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_endif;

        // If not in FP32 dest accumulation mode, round the result to BF16 precision
        // using IEEE 754 Round-to-Nearest-Even (RNE) algorithm.
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        // Store the result back to DEST at the output tile offset (same as input A)
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        // Advance to the next row within the tile
        sfpi::dst_reg++;
    }
}
```

### BF16 RNE Helper (used when `!is_fp32_dest_acc_en`)

```cpp
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Reinterpret the FP32 value as raw bits
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    // Extract the LSB of the would-be BF16 mantissa (bit 16 of FP32)
    sfpi::vUInt lsb = (bits >> 16) & 1;
    // Add 0x7fff + lsb: implements correct RNE tie-breaking
    // - Ties (lower 16 bits = 0x8000) round to even: up if lsb=1, down if lsb=0
    bits = bits + 0x7fffU + lsb;
    // Truncate lower 16 bits to produce BF16-in-FP32 representation
    bits = bits & 0xFFFF0000U;
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}
```

### Reciprocal Implementation (Wormhole B0 -- `_sfpu_reciprocal_<2>`)

**File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`

```cpp
// Computes the reciprocal of a floating point value x.
// max_iter = 2: sufficient for float32 precision (<=1 ulps).
// max_iter = 1: sufficient for bfloat16/float16 precision (<=0.5 ulps).
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Step 1: Normalize input to [1.0, 2.0) range by replacing the exponent.
    // setman(vConstNeg1, mantissa_of_in) produces a value with:
    //   - sign = 1 (negative, from vConstNeg1)
    //   - exponent = 127 (from vConstNeg1, which is -1.0)
    //   - mantissa = mantissa of `in`
    // This effectively computes negative_x = -|in| * 2^(127 - in.Exp)
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Step 2: Quadratic initial estimate: y = k2 + k1*negative_x + k0*negative_x^2
    // The polynomial minimizes maximum relative error over [1,2], computed via Sollya.
    // vConstFloatPrgm0 = 0.3232325... (k0)
    // vConstFloatPrgm1 = 1.4545459... (k1 -- used negatively since negative_x is negative)
    // vConstFloatPrgm2 = 2.1212124... (k2)
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Step 3: Compute the scale factor for de-normalization.
    // 1/in = 1/x * 2^(127-in.Exp), so scale.Exp = 254 - in.Exp.
    // Using ~in gives 255-in.Exp (via SFPNOT), which is one off; corrected below with scale*0.5.
    // This also handles special cases: in.Exp=0 -> inf, in.Exp=255 -> 0.
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue quadratic estimate: y = k2 + (k1 + k0*negative_x) * negative_x
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Clear mantissa of scale factor (keep only sign+exponent)
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // Step 4: First Newton-Raphson iteration
    // t = 1 - x*y (using negative_x, so t = 1 + negative_x * y)
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Correct scale exponent: scale *= 0.5 adjusts from 255-in.Exp to 254-in.Exp
    // Also handles: inf*0.5 = inf, 0*0.5 = 0 (preserving special cases)
    scale *= 0.5f;

    // Complete first NR iteration: y = y + y*t = y * (1 + t) = y * (2 - x*y_old)
    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Step 5: Second Newton-Raphson iteration for full FP32 precision
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Step 6: Apply scaling and restore original sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}
```

### Reciprocal Initialization

```cpp
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Load the Sollya-optimized polynomial coefficients into SFPU programmable constants.
    // These are used by _sfpu_reciprocal_ for the quadratic initial estimate.
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0 (quadratic coeff)
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1 (linear coeff)
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2 (constant term)
}
```

### SFPU Instructions Used

The `calculate_sfpu_binary_div` function (and its reciprocal subroutine) compiles to the following SFPU instructions via the SFPI compiler:

| Instruction | Description | Usage in DIV |
|-------------|-------------|--------------|
| `SFPLOAD` | Load data from DEST register into SFPU LREG | Loading `in0`, `in1` from DST tile offsets |
| `SFPSTORE` | Store data from SFPU LREG back to DEST register | Writing `result` back to DST output offset |
| `SFPMUL` | Vectorized FP32 multiply | `in0 * reciprocal(in1)`, Newton-Raphson products |
| `SFPMAD` | Multiply-accumulate (a*b+c) | Newton-Raphson: `y + y*t`, polynomial evaluation |
| `SFPLOADI` | Load immediate value into LREG | Loading polynomial constants, infinity, NaN |
| `SFPNOT` | Bitwise NOT | Computing `~in` for scale factor exponent |
| `SFPSETMAN` | Set mantissa bits | Normalizing input, clearing mantissa of scale |
| `SFPSETSGN` | Set sign bit | Restoring sign of reciprocal, setting infinity sign |
| `SFPSETEXP` | Set exponent field | Not directly used in div, but referenced in SFPI |
| `SFPCOMPC` | Compare for conditional execution | `in1 == 0`, `in0 == 0`, `in0 == in1` checks |
| `SFPENCC` / `SFPSETCC` | Condition code manipulation | `v_if`/`v_else`/`v_elseif`/`v_endif` control flow |
| `SFPMOV` | Move between LREGs | Register shuffling during computation |
| `SFPIADD` | Integer add | BF16 RNE: `bits + 0x7fff + lsb` |
| `SFPAND` | Bitwise AND | BF16 RNE: `bits & 0xFFFF0000` |
| `SFPSHFT` | Bit shift | BF16 RNE: `bits >> 16` |

### SFPU Register Usage

- **DEST registers**: Tiles are loaded into DEST by the `copy_tile` calls. Input A occupies even slots (0, 2, 4, ...) and input B occupies odd slots (1, 3, 5, ...). The SFPU reads from and writes back to DEST via `dst_reg[]`.
- **LREG0-LREG7**: The SFPU has 8 local vector registers (LREGs). During reciprocal computation:
  - `negative_x`, `y`, `t`, `scale` each occupy one LREG
  - The compiler manages register allocation automatically through the SFPI abstraction
- **vConstFloatPrgm0-2**: Three programmable constant registers loaded during `_init_sfpu_reciprocal_` with the polynomial coefficients
- **vConst1**: Hardware constant register holding 1.0f
- **vConstNeg1**: Hardware constant register holding -1.0f

### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to claim exclusive access to DEST registers. `tile_regs_wait()` stalls until DEST is available.

2. **Unpack to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks tile `i` from CB `c_0` into DEST slot `i*2` (input A). `copy_tile(cb_inp1, i, i*2+1)` unpacks tile `i` from CB `c_1` into DEST slot `i*2+1` (input B). The `UnpackToDestFp32` mode ensures all data arrives in FP32.

3. **SFPU initialization**: `div_binary_tile_init()` initializes the reciprocal polynomial constants into programmable constant registers.

4. **SFPU math operation**: `div_binary_tile(i*2, i*2+1, i*2)` triggers the SFPU to:
   - Iterate 8 times over sub-tile sections (each iteration processes one row of 32 elements)
   - Load `in0` from DEST[i*2 offset] and `in1` from DEST[i*2+1 offset]
   - Compute `reciprocal(in1)` via quadratic estimate + 2 Newton-Raphson iterations
   - Multiply `in0 * reciprocal(in1)` to get the division result
   - Handle special cases (0/0 -> NaN, x/0 -> +/-inf, x/x -> 1.0)
   - Optionally round to BF16 via RNE
   - Store result back to DEST[i*2 offset]

5. **Pack to output CB**: `pack_tile(i*2, cb_out0)` reads from DEST[i*2] and packs the result into the output circular buffer.

6. **Release and signal**: `tile_regs_commit()` and `tile_regs_release()` free the DEST registers. `cb_pop_front` frees input CB pages; `cb_push_back` publishes output pages for the writer kernel.

### SFPU Configuration

- **APPROX**: The `APPROX` template parameter (derived from math fidelity settings) controls whether approximation mode is used. For `_sfpu_reciprocal_`, `APPROX=true` would use 0 Newton-Raphson iterations (faster, lower precision), while `APPROX=false` uses 2 iterations (slower, FP32-precise).
- **DST_ACCUM_MODE**: When `fp32_dest_acc_en` is true, the SFPU skips BF16 rounding and keeps full FP32 precision in DEST.
- **ITERATIONS = 8**: Each tile has 32x32 = 1024 elements. The SFPU processes 32 elements per cycle (SIMD width), across 4 faces. 8 iterations = 4 faces x 2 halves per face.

### Hardware Compatibility Notes

**Wormhole B0**:
- Reciprocal uses software Newton-Raphson with a Sollya-optimized quadratic polynomial initial estimate
- Initialization loads 3 polynomial constants into `vConstFloatPrgm0/1/2`
- The reciprocal function uses `setman`, `SFPNOT`, and manual exponent manipulation
- No hardware reciprocal instruction available

**Blackhole**:
- Reciprocal uses the hardware `SFPARECIP` instruction (`sfpi::approx_recip()`) for the initial estimate, which is significantly faster
- Newton-Raphson refinement is still applied (1 or 2 iterations depending on precision needs)
- Initialization sets `vConstFloatPrgm0 = 2.0f` (used in the NR formula `x*y - 2.0`)
- The NR formula is restructured: instead of `t = 1 - x*y; y = y + y*t`, it uses `t = x*y - 2.0; y = y * -t` (negated for NaN detection via sign check)
- Blackhole also has dedicated fast-path implementations using `SFPLOADMACRO` for bulk reciprocal:
  - `_calculate_reciprocal_fast_7b_`: ~7-bit precision, 1 cycle/32 elements throughput
  - `_calculate_reciprocal_fast_8b_3c_`: BF16 precision, 3 cycles/32 elements
  - `_calculate_reciprocal_fast_24b_5c_`: FP32 precision, 5 cycles/32 elements
  - These fast paths are used for standalone reciprocal operations, not directly in `calculate_sfpu_binary_div`

**Both architectures**: The `calculate_sfpu_binary_div` function in the metal overlay (`ckernel_sfpu_binary.h`) is identical for Wormhole and Blackhole -- the architectural difference is entirely encapsulated within `_sfpu_reciprocal_<2>()`. The `_sfpu_binary_init_` and `_calculate_sfpu_binary_` functions in the tt_llk common layer are also identical across both architectures.

---

## Runtime Arguments

### Reader Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `src0_addr` | DRAM address of input tensor A buffer |
| 1 | `src1_addr` | DRAM address of input tensor B buffer |
| 2 | `num_tiles` | Total number of tiles this core processes |
| 3 | `start_id` | Starting tile ID for this core's work partition |
| 4 | `block_height` | Block height in tiles (for sharded layouts) |
| 5 | `block_width` | Block width in tiles (for sharded layouts) |
| 6 | `num_shards_per_width` | Number of shards across the width dimension |

### Compute Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `per_core_block_cnt` | Number of tile blocks to process |
| 1 | `per_core_block_size` | Number of tiles per block |

### Writer Kernel Runtime Args (interleaved output)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | DRAM address of output tensor buffer |
| 1 | `num_pages` | Number of output pages to write |
| 2 | `start_id` | Starting page ID |

---

## Work Distribution

Work is split across cores using `split_work_to_cores()` for interleaved tensors. Each core gets a contiguous range of tile IDs. For sharded tensors, the shard spec determines the tile assignment per core.

The `max_block_size` (largest power of 2 that divides `num_tiles_per_shard`) controls how many tiles are processed per inner loop iteration, enabling more efficient CB double-buffering.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Program factory structure, kernel registration, `get_defines_fp32` function, binary operation routing
- **tenstorrent/tt-llk**: `ckernel::sfpu` namespace, `_calculate_sfpu_binary_` template, `BinaryOp::DIV` path, `_sfpu_reciprocal_` implementation details, Newton-Raphson iteration structure
- **tenstorrent/tt-isa-documentation**: SFPU instruction set -- `SFPARECIP` (Blackhole), `SFPDIVP2`, `SFPMUL`, `SFPLOADI`, `SFPSETMAN`, `SFPNOT`, `SFPSETSGN`
- **tenstorrent/sfpi**: SFPI programming interface -- `vFloat` operations, `setman`, `setsgn`, `setexp`, `exexp`, `approx_recip` (Blackhole), Newton-Raphson reciprocal implementation

### Confluence References

Not consulted for this analysis. The DeepWiki and source code provided sufficient detail on the SFPU instructions used.

### Glean References

Not consulted for this analysis. The open-source tt_llk implementations for both Wormhole and Blackhole provided complete reciprocal/division SFPU kernel source code.

---

## Key Design Decisions

1. **Division as multiplication-by-reciprocal**: The SFPU has no native division instruction. Division is implemented as `a * (1/b)`, where the reciprocal uses a quadratic polynomial estimate refined by Newton-Raphson iterations. This is a standard approach on hardware without dedicated dividers.

2. **Separate `calculate_sfpu_binary_div` vs `_calculate_sfpu_binary_<DIV>`**: The metal overlay defines a dedicated `calculate_sfpu_binary_div` function rather than using the shared `_calculate_sfpu_binary_` template from tt_llk. The dedicated version adds:
   - Division-by-zero handling (NaN for 0/0, signed infinity for x/0)
   - Exact equality handling (a/a = 1.0, avoiding reciprocal rounding artifacts)
   - BF16 RNE rounding when not in FP32 dest mode
   These are correctness improvements over the basic `in0 * reciprocal(in1)` in the shared template.

3. **Two Newton-Raphson iterations (`max_iter=2`)**: This provides FP32-level precision (<=1 ULP error) for the reciprocal. For BF16-only workloads, a single iteration would suffice, but the code uses 2 iterations regardless because the SFPU always computes in FP32 internally.

4. **`UnpackToDestFp32` for all inputs**: Regardless of input data format, inputs are unpacked to FP32 in DEST. This ensures the SFPU math operates at full precision, with format conversion happening only at the pack stage.

5. **`get_defines_fp32` vs `get_defines`**: There are two define-generation paths for DIV:
   - `get_defines` (non-SFPU path): Decomposes DIV into `RECIP(input_b)` as a pre-scale on input B, then uses `mul_tiles` via the FPU. This uses two separate operations.
   - `get_defines_fp32` (SFPU path): Uses `div_binary_tile` which performs the entire division in a single SFPU kernel pass. The SFPU path is selected for FP32-FP32 and INT32-INT32 type combinations.
