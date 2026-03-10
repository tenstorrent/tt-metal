# TTNN SFPU Operation Analysis: I0 (Modified Bessel Function of the First Kind, Order Zero)

## Operation Overview

**Operation Name**: `ttnn::i0`
**Operation Type**: `UnaryOpType::I0`
**Category**: Eltwise Unary
**Mathematical Function**: I0(x) = zeroth-order modified Bessel function of the first kind
**Python API**: `ttnn.i0(input_tensor)`

The modified Bessel function of the first kind of order zero, I0(x), is defined by the series:

```
I0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
```

This implementation uses a degree-10 polynomial approximation in x^2 (equivalently a degree-20 even polynomial in x), which closely follows the truncated Taylor series of I0 around zero.

## Program Factory

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Factory Type
`UnaryProgramFactory` (shared by all standard unary SFPU operations)

There is also a `UnarySubCoreGridProgramFactory` variant that supports sub-core grid execution with the same kernel structure.

### Program Factory Parameters

The factory receives `UnaryParams`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_chain` | `std::vector<EltwiseUnaryWithParam>` | Chain of unary operations to fuse (I0 is a single-op chain) |
| `output_dtype` | `DataType` | Output data type |
| `output_memory_config` | `MemoryConfig` | Output memory configuration |
| `fp32_dest_acc_en` | `bool` | Enable FP32 destination accumulation |
| `preserve_fp32_precision` | `bool` | Preserve FP32 precision through unpack-to-dest mode |
| `bfp8_pack_precise` | `bool` | Enable precise BFP8 packing |
| `sub_core_grids` | `optional<CoreRangeSet>` | Optional sub-core grid (selects SubCoreGrid factory variant) |

### Key Design Decisions

1. **Shared program factory**: I0 uses the same `UnaryProgramFactory` as all standard unary SFPU operations. The operation-specific behavior is injected via preprocessor defines (`SFPU_OP_CHAIN_0`) that expand to the `i0_tile_init()` and `i0_tile(idst)` calls.

2. **No special parameters**: I0 is a no-parameter SFPU operation. It does not pack any scalar runtime arguments (the `packed_scalar1` and `packed_scalar2` remain 0).

3. **Approximation mode**: `get_op_approx_mode` returns `false` for I0 (it falls through to the `default` case). This means `math_approx_mode` is `false`, and the `APPROX` template parameter in the SFPU kernel is `false`.

4. **Math fidelity**: Always `MathFidelity::HiFi4` for all unary SFPU operations dispatched through this factory.

## Circular Buffer Configuration

| CB Index | Name | Page Count | Data Format | Purpose |
|----------|------|------------|-------------|---------|
| `c_0` | Input | 2 | Input tensor dtype | Double-buffered input tiles from reader |
| `c_2` | Output | 2 | Output tensor dtype | Double-buffered output tiles for writer |

Note: CB `c_1` (tmp0) is only allocated for `HARDSHRINK` and `LOGIT` operations, not for I0.

The page size is set based on layout: `tile_size(cb_data_format)` for TILE layout, or `buffer->page_size()` for ROW_MAJOR layout.

## Core Distribution

Work is distributed across all compute cores using `split_work_to_cores`:
- Input tiles are evenly divided across cores
- Two core groups handle remainder tiles (group 1 gets `ceil(tiles/cores)`, group 2 gets `floor(tiles/cores)`)
- Each core processes tiles sequentially in a tile-at-a-time loop

## Kernel Implementations

### Reader Kernel

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

**Runtime Args**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | `src_addr` | Source buffer DRAM address |
| 1 | `num_pages` | Number of pages this core processes |
| 2 | `start_id` | Starting page ID for this core |

**Compile-Time Args**: `TensorAccessorArgs` from the source buffer.

The reader iterates over assigned pages, reading one page at a time from DRAM into CB `c_0` using `noc_async_read_page` with a barrier after each page.

### Writer Kernel

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Runtime Args**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | Destination buffer DRAM address |
| 1 | `num_pages` | Number of pages this core processes |
| 2 | `start_id` | Starting page ID for this core |

**Compile-Time Args**: `[output_cb_index, TensorAccessorArgs...]`

The writer iterates over assigned pages, waiting for each completed page in CB `c_2` and writing it to DRAM via `noc_async_write_page`.

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes i0.h when SFPU_OP_I0_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // Initialize SFPU with input and output CB indices
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // Reserve output space before processing the block
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // Acquire exclusive access to DST register file

            // Wait for one input tile, copy it from CB c_0 into DST register 0
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);  // Unpack tile from CB c_0, slot 0, into DST register 0

// For I0, this macro expands to:
//   i0_tile_init();    -- one-time SFPU init (calls llk_math_eltwise_unary_sfpu_init<SfpuType::i0, false>())
//   i0_tile(0);        -- execute I0 SFPU kernel on DST register 0
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // Signal that DST registers are ready for packing

            tile_regs_wait();  // Wait for packer to be ready

            pack_tile(0, tt::CBIndex::c_2);  // Pack DST register 0 into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // Release the consumed input tile

            tile_regs_release();  // Release DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // Publish completed output tiles to writer
    }
}
```

**Compile-Time Args**:
| Index | Name | Value for I0 |
|-------|------|--------------|
| 0 | `per_core_block_cnt` | Number of tiles assigned to this core |
| 1 | `per_core_block_dim` | 1 (tile-at-a-time processing) |

**Runtime Args**:
| Index | Name | Value for I0 |
|-------|------|--------------|
| 0 | `packed_scalar1` | 0 (unused) |
| 1 | `packed_scalar2` | 0 (unused) |

**Preprocessor Defines** (set by `get_block_defines` for I0):
- `SFPU_OP_I0_INCLUDE` = `1` -- triggers inclusion of `api/compute/eltwise_unary/i0.h`
- `SFPU_OP_CHAIN_0_INIT_0` = `i0_tile_init();`
- `SFPU_OP_CHAIN_0_FUNC_0` = `i0_tile(0);`
- `SFPU_OP_CHAIN_0` = `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0`

**Compute Config**:
- `math_fidelity`: `MathFidelity::HiFi4`
- `math_approx_mode`: `false`
- `fp32_dest_acc_en`: from operation params
- `bfp8_pack_precise`: from operation params

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU API Header
`tt_metal/hw/inc/api/compute/eltwise_unary/i0.h`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_i0.h"                    // Includes the actual SFPU kernel implementation
#include "llk_math_eltwise_unary_sfpu_macros.h"  // Provides the dispatch macros
#endif

namespace ckernel {

// i0_tile: dispatches the SFPU kernel calculate_i0 for a single tile in DST
// SFPU_UNARY_NO_PARAM_KERNEL_FN expands to:
//   _llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_i0<false>, idst, (int)VectorMode::RC)
// This configures the SFPU to process all rows and columns of the tile at DST[idst].
ALWI void i0_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_i0, RC, APPROX, idst)); }

// i0_tile_init: one-time initialization of SFPU state for I0 operation
// Expands to: llk_math_eltwise_unary_sfpu_init<SfpuType::i0, false>()
ALWI void i0_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(i0, APPROX)); }

}  // namespace ckernel
```

#### SFPU Kernel File (Wormhole B0)
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h`

#### SFPU Kernel File (Blackhole)
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h`

Both files are identical. The I0 SFPU kernel has no architecture-specific differences between Wormhole B0 and Blackhole.

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;  // Brings in SFPI types: vFloat, dst_reg, and SFPU intrinsics

namespace ckernel {
namespace sfpu {

// POLYVAL10: Evaluates a degree-10 polynomial using Horner's method.
// Given coefficients c10..c0 and variable t4, computes:
//   c0 + c1*t4 + c2*t4^2 + ... + c10*t4^10
// The nested form (Horner's method) minimizes the number of multiplications
// and is numerically more stable than naive power-sum evaluation.
// Each multiplication compiles to an SFPMAD (multiply-add) SFPU instruction.
#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)

// calculate_i0: Core SFPU kernel for the modified Bessel function I0.
//
// Template parameters:
//   APPROXIMATION_MODE (bool): Controls whether approximate math is used.
//     For I0, this is always false (set by get_op_approx_mode default case).
//     The parameter is present for interface consistency but does not alter
//     behavior in the current implementation.
//   ITERATIONS (int): Number of SFPU vector rows to process per call.
//     Default is 8, which processes 8 rows of 4 elements = 32 elements,
//     corresponding to one face of a 32x32 tile. The LLK dispatch layer
//     calls this function multiple times (controlled by VectorMode::RC) to
//     cover all faces of the tile.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
#pragma GCC unroll 0  // Prevent loop unrolling to reduce code size in SFPU instruction memory

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];  // Read current element from DST register file (SFPU lane-parallel read)
        vFloat x = input * input;   // Compute x^2; I0 is an even function, so the polynomial is in x^2
                                    // This SFPMUL computes all 4 lanes in parallel

        // Evaluate the degree-10 polynomial in x^2 using Horner's method.
        // The coefficients are derived from the Taylor series of I0(x):
        //   I0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
        //
        // Expanding and collecting terms in powers of x^2:
        //   I0(x) = 1 + (1/4)x^2 + (1/64)x^4 + (1/2304)x^6 + ...
        //
        // Coefficient verification (matching Taylor series coefficients of I0):
        //   coef0  = 0.25          = 1/(2^2)           = 1/((1!)^2 * 4^1) * 4     -- k=1 term: 1/4
        //   coef1  = 0.015625      = 1/(2^6)           = 1/64                      -- k=2 term: 1/64
        //   coef2  = 4.340277778e-4 ~ 1/2304           = 1/((3!)^2 * 4^0)         -- k=3 term: 1/2304
        //   coef3  = 6.781684028e-6 ~ 1/147456                                     -- k=4 term
        //   coef4  = 6.78e-08      ~ 1/14745600                                    -- k=5 term
        //   coef5  = 4.71e-10      ~ 1/2123366400                                  -- k=6 term
        //   coef6  = 2.40e-12                                                       -- k=7 term
        //   coef7  = 9.39e-15                                                       -- k=8 term
        //   coef8  = 2.90e-17                                                       -- k=9 term
        //   coef9  = 7.24e-20                                                       -- k=10 term
        //   coef10 = 1.50e-22                                                       -- k=11 term
        //
        // The +1.0f accounts for the k=0 term of the series (I0(0) = 1).
        result = 1.0f + POLYVAL10(
                            1.50E-22f,           // coef10: Taylor series k=11 term coefficient
                            7.24E-20f,           // coef9:  Taylor series k=10 term coefficient
                            2.90E-17f,           // coef8:  Taylor series k=9 term coefficient
                            9.39E-15f,           // coef7:  Taylor series k=8 term coefficient
                            2.40E-12f,           // coef6:  Taylor series k=7 term coefficient
                            4.71E-10f,           // coef5:  Taylor series k=6 term coefficient
                            6.78E-08f,           // coef4:  Taylor series k=5 term coefficient
                            0.000006781684028f,  // coef3:  Taylor series k=4 term coefficient
                            0.0004340277778f,    // coef2:  Taylor series k=3 term coefficient
                            0.015625f,           // coef1:  Taylor series k=2 term coefficient (1/64)
                            0.25f,               // coef0:  Taylor series k=1 term coefficient (1/4)
                            x);                  // Variable: x^2 (input squared)

        dst_reg[0] = result;  // Write the result back to DST register (lane-parallel write)
        dst_reg++;            // Advance the DST register pointer to the next row of 4 elements
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description | Usage in I0 |
|------------------------|-------------|-------------|
| `SFPMUL` | Floating-point multiply (lane-parallel, 4 lanes) | Used for `input * input` (x^2 computation) and each Horner step multiply |
| `SFPMAD` | Fused multiply-add: `A * B + C` | Each step of the Horner polynomial evaluation compiles to SFPMAD instructions. The nested `coef + (... * t4)` pattern maps directly to MAD operations |
| `SFPLOAD` | Load from DST register into SFPU LReg | Implicit in `dst_reg[0]` read -- loads tile element into SFPU vector register |
| `SFPSTORE` | Store from SFPU LReg back to DST register | Implicit in `dst_reg[0] = result` write -- stores computed result |
| `SFPLOADI` | Load immediate constant into SFPU register | Used for all floating-point literal coefficients (0.25f, 0.015625f, etc.) and the initial `0.0f` / `1.0f` values |

Note: The I0 kernel does NOT use:
- `SFPLUT` / `SFPLUTFP32` (no lookup table approximation)
- `SFPEXEXP` / `SFPSETEXP` (no exponent manipulation)
- `SFPSETCC` / `SFPPUSHC` / `SFPPOPC` (no conditional execution)

This makes I0 one of the simplest SFPU kernels -- it is a pure arithmetic polynomial evaluation with no branching or special instruction usage.

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` (DST register file) | Input: contains the element to transform. Output: receives the I0 result. The DST register file is shared between unpack, SFPU, and pack stages |
| SFPU LReg (implicit via `vFloat`) | Temporary storage for `input`, `x` (= input^2), `result`, and intermediate Horner evaluation values. The compiler allocates SFPU local registers (LReg[0]-LReg[3]) for these vFloat variables |
| Immediate constants | The 11 polynomial coefficients and constants (0.0f, 1.0f) are loaded via SFPLOADI into LRegs as needed during evaluation |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to gain exclusive access to the DST register file, then `copy_tile(c_0, 0, 0)` unpacks one tile from input CB into DST register 0.

2. **SFPU init**: `i0_tile_init()` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::i0, false>()` which configures the SFPU pipeline for unary elementwise operation (no special init callback needed for I0).

3. **SFPU dispatch**: `i0_tile(0)` expands to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_i0<false>, 0, (int)VectorMode::RC)`. The dispatch layer:
   - Sets up the DST register pointer to tile index 0
   - Calls `calculate_i0<false>()` repeatedly with `ITERATIONS=8` to cover all rows/columns (VectorMode::RC processes all 4 faces of the 32x32 tile)

4. **Per-iteration SFPU math** (repeated 8 times per face, 4 faces total = 32 rows of 4 elements = 1024 elements per tile):
   - **Load**: Read element from `dst_reg[0]` (SFPLOAD)
   - **Square**: Compute `x = input * input` (SFPMUL)
   - **Horner evaluation**: 10 nested multiply-add steps (each is SFPMAD or SFPMUL+SFPADD):
     - Start with innermost: `coef10 * x`
     - Add `coef9`, multiply by `x`
     - Continue outward through all 10 levels
     - Add `coef0` at the outermost level
   - **Add constant**: Add `1.0f` to the polynomial result
   - **Store**: Write result to `dst_reg[0]` (SFPSTORE)
   - **Advance**: Increment `dst_reg` pointer to next row

5. **Pack**: After SFPU completes, `tile_regs_commit()` signals packer, `pack_tile(0, c_2)` packs the result from DST register 0 into output CB `c_2`.

6. **Release**: `cb_pop_front(c_0, 1)` frees the input tile, `tile_regs_release()` releases DST registers. After all tiles in a block, `cb_push_back(c_2, per_core_block_dim)` publishes the output.

#### SFPU Configuration

- **APPROXIMATION_MODE**: `false` (template parameter). The I0 implementation does not vary between approximate and exact modes -- the same polynomial is used regardless. This parameter exists for interface consistency with other SFPU kernels.
- **ITERATIONS**: 8 (default). Processes 8 rows of 4 SFPU lanes = 32 elements per call. Combined with VectorMode::RC dispatch across 4 faces, this covers the full 32x32 = 1024 elements of a tile.
- **Math fidelity**: `MathFidelity::HiFi4` (set in ComputeConfig). This ensures maximum precision in the FPU/SFPU pipeline.
- **Math approx mode**: `false`. This is the global approximate math flag passed to the compute config. For I0, `get_op_approx_mode` returns `false`.
- **Unroll pragma**: `#pragma GCC unroll 0` prevents the compiler from unrolling the iteration loop, conserving SFPU instruction memory which is limited.
- **No special defines**: I0 does not require any additional compile-time defines beyond the standard `SFPU_OP_I0_INCLUDE` and `SFPU_OP_CHAIN_0` mechanism.

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `ckernel_sfpu_i0.h` are **identical**. This is because the I0 kernel uses only basic SFPU arithmetic (multiply, multiply-add, load, store) which behaves the same on both architectures.

Key differences between Wormhole and Blackhole SFPU that do NOT affect I0 but are relevant context:
- Blackhole's `fma_model_bh` has improved NaN handling (canonical NaN patterns) and corrected denormal rounding compared to Wormhole's `fma_model_wh`.
- Blackhole supports additional SFPU instructions not used by I0.
- The FMA precision model differs slightly (both are partially-fused, not fully IEEE754-compliant), but for the magnitude of I0's coefficients this difference is negligible.

For very large inputs where I0 grows exponentially, the polynomial approximation will lose accuracy regardless of architecture. The implementation is most accurate for small to moderate |x| values.

## Macro and Define Chain

The following chain shows how the I0 operation flows from the program factory to the SFPU kernel:

```
Program Factory (unary_program_factory.cpp)
  |
  +--> get_block_defines(op_chain, "0", "0", dtype)
  |      |
  |      +--> get_defines_impl(UnaryOpType::I0, ...)
  |             |
  |             +--> update_macro_defines(I0, defines)
  |             |      defines["SFPU_OP_I0_INCLUDE"] = "1"
  |             |
  |             +--> get_op_init_and_func_default(I0, "0")
  |                    returns: {"i0_tile_init();", "i0_tile(0);"}
  |
  |      Result defines:
  |        SFPU_OP_I0_INCLUDE = 1
  |        SFPU_OP_CHAIN_0_INIT_0 = i0_tile_init();
  |        SFPU_OP_CHAIN_0_FUNC_0 = i0_tile(0);
  |        SFPU_OP_CHAIN_0 = SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0
  |
  +--> get_compute_kernel_path(I0, dtype)
  |      returns: "eltwise_sfpu.cpp"  (default case)
  |
  +--> CreateKernel("ttnn/.../compute/eltwise_sfpu.cpp", ...)
         with defines above and ComputeConfig{math_approx_mode=false, ...}

Compile-time in eltwise_sfpu.cpp:
  |
  +--> sfpu_split_includes.h
  |      #if SFPU_OP_I0_INCLUDE   -->  #include "api/compute/eltwise_unary/i0.h"
  |
  +--> i0.h
  |      #include "ckernel_sfpu_i0.h"  (the SFPU kernel)
  |      defines i0_tile(idst) and i0_tile_init()
  |
  +--> SFPU_OP_CHAIN_0 expands to:
         i0_tile_init();   -->  llk_math_eltwise_unary_sfpu_init<SfpuType::i0, false>()
         i0_tile(0);       -->  _llk_math_eltwise_unary_sfpu_params_<false>(
                                    ckernel::sfpu::calculate_i0<false>, 0, (int)VectorMode::RC)
```

## Mathematical Analysis

### Polynomial Approximation Quality

The implementation uses a degree-10 polynomial in x^2 (equivalently degree-20 in x), based on the Taylor series of I0. The Taylor series of I0(x) converges for all x, but the rate of convergence slows for large |x|.

The polynomial coefficients match the first 12 terms (k=0 through k=11) of:
```
I0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
```

**Accuracy considerations**:
- For small |x| (roughly |x| < 5-6), the truncated Taylor series provides excellent accuracy
- For larger |x|, the missing higher-order terms cause increasing error
- The implementation does NOT use range reduction or asymptotic expansion for large |x|, which limits its effective range
- There is no input clamping or special handling for overflow/NaN

**Comparison to PyTorch**: PyTorch's `torch.i0` typically uses Chebyshev polynomial approximations with range-specific polynomial sets (one for small |x|, one for large |x|), providing better accuracy over a wider range. This SFPU implementation trades range for simplicity and execution speed.

### Numerical Properties

- **I0(0) = 1**: The kernel correctly returns 1.0 when input is 0 (the polynomial evaluates to 0 at x^2=0, plus the constant 1.0).
- **I0 is even**: I0(-x) = I0(x). The kernel exploits this by computing x^2 first, making the function naturally even without explicit absolute value handling.
- **I0(x) > 0 for all x**: The function is always positive. However, the polynomial approximation could theoretically go negative for very large |x| where truncation error dominates.

## File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory -- creates the TT-Metal program with reader, compute, and writer kernels |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp` | Program factory header -- defines `UnaryProgramFactory` and `UnarySubCoreGridProgramFactory` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation_types.hpp` | Defines `UnaryParams` and `UnaryInputs` structs |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Utility functions: `get_block_defines`, `get_compute_kernel_path`, `get_op_approx_mode`, `get_op_init_and_func_default` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Defines `UnaryOpType::I0` enum |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(i0, I0)` -- binds `ttnn::i0` to `UnaryOpType::I0` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | Python binding for `ttnn.i0` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Compute kernel -- shared by all standard SFPU unary ops |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel -- reads tiles from DRAM |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel -- writes tiles to DRAM |
| `tt_metal/hw/inc/api/compute/eltwise_unary/i0.h` | API header -- defines `i0_tile()` and `i0_tile_init()` |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include -- gates `i0.h` inclusion on `SFPU_OP_I0_INCLUDE` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h` | SFPU kernel (Wormhole B0) -- `calculate_i0` implementation |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h` | SFPU kernel (Blackhole) -- identical to Wormhole B0 |

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Located the I0 operation registration, program factory dispatch, compute kernel path resolution, and SFPU include mechanism
- `tenstorrent/tt-llk`: Confirmed ckernel namespace structure (sfpu::calculate_i0) and LLK dispatch patterns
- `tenstorrent/tt-isa-documentation`: Documented SFPMAD, SFPMUL, SFPLUT, SFPLUTFP32 instructions and FMA model differences between Wormhole and Blackhole
- `tenstorrent/sfpi`: Documented SFPI programming patterns for polynomial evaluation (Horner's method via SFPMAD), vFloat type, dst_reg access, and conditional execution primitives

### Confluence References
Not consulted. The I0 kernel uses only basic SFPU arithmetic instructions (multiply, multiply-add, load, store) which are well-documented in DeepWiki. No advanced SFPU ISA details were needed.

### Glean References
Not consulted. The I0 implementation is fully visible in open-source code with no architecture-specific variations requiring confidential hardware specs.
