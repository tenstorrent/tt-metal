# Exp Implementation Analysis

## Overview
The `exp` operation computes the element-wise natural exponential function (`e^x`) on an input tensor. It is implemented as a unary SFPU operation that uses the shared `UnaryProgramFactory` program factory, which hosts all standard unary elementwise operations via compile-time define injection.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition
One work unit is **one tile** (32x32 elements). The program factory divides the total number of tiles across cores, and each core processes its assigned tiles one at a time (`per_core_block_size = 1`). The outer loop iterates over assigned tiles (`per_core_block_cnt`), while the inner loop processes one tile per iteration.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Value |
|---|---|
| Dimension Convention | NHWC (standard ttnn) |
| Tensor Layout | TILE (32x32) or ROW_MAJOR |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, or other supported formats |

### Output Tensor(s)

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | Same as input |
| Memory Layout | Interleaved |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input (or potentially different if output dtype specified) |

### Layout Transformations
No layout transformations are performed. Input and output share the same layout.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from DRAM/L1 into circular buffer `c_0` via NoC.
2. **Compute kernel** waits for a tile in `c_0`, copies it to DST registers, executes the SFPU exp operation (injected via `SFPU_OP_CHAIN_0` define), and packs the result into circular buffer `c_2`.
3. **Writer kernel** waits for a tile in `c_2` and writes it to the output buffer in DRAM/L1 via NoC.

This is a simple streaming pipeline: Reader -> Compute -> Writer, one tile at a time.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` | Input CB | Holds input tiles from reader for compute | tile_size (input dtype) | 2 | 2 * tile_size | Double-buffered | Reader | Compute |
| `c_2` | Output CB | Holds computed tiles from compute for writer | tile_size (output dtype) | 2 | 2 * tile_size | Double-buffered | Compute | Writer |

**Note**: CB `c_1` (tmp0) is only allocated for `HARDSHRINK` or `LOGIT` operations, not for `exp`.

## Pipeline Pattern Summary
Both input and output CBs have capacity = 2 pages with block_size = 1 page, enabling **double-buffering**. This allows the reader to fill one slot while compute processes the other, and similarly compute can produce while the writer drains.

## Index Calculations
The reader and writer kernels use `TensorAccessor` with a simple page-index scheme. Each core receives a `start_id` (starting page/tile index) and `num_pages` count. Pages are accessed sequentially from `start_id` to `start_id + num_pages - 1`. The `TensorAccessor` handles the mapping from logical page index to physical DRAM bank address.

## Memory Access Patterns

### Read Pattern
Sequential tile reads. Each core reads its assigned contiguous range of tile indices from the interleaved buffer. One tile is read per iteration: `noc_async_read_page(i, s, l1_write_addr)` with a barrier after each read.

### Write Pattern
Sequential tile writes. Each core writes its computed tiles in the same contiguous order: `noc_async_write_page(i, s, l1_read_addr)` with a flush after each write and a final barrier.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Full compute grid (`compute_with_storage_grid_size`) |
| Work Splitting | `split_work_to_cores()` divides total tiles across available cores |
| Core Group 1 | Gets `num_pages_per_core_group_1` tiles (the "larger" share) |
| Core Group 2 | Gets `num_pages_per_core_group_2` tiles (handles remainder, may be empty) |
| Core Enumeration | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| Load Balancing | At most 1 tile difference between group 1 and group 2 |

If the number of tiles divides evenly across cores, group 2 is empty. Otherwise, group 1 cores get one extra tile compared to group 2 cores. Each group gets its own compute kernel instance (with different `per_core_block_cnt` compile-time arg).

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, memory layout, page size, and bank mapping for the source buffer |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (always `c_2` = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Encodes buffer type, memory layout, page size, and bank mapping for the destination buffer |

**Compute Kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of tiles this core must process |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for this factory) |

**Compute Kernel Defines** (injected at compile time):

| Define | Value for Exp | Description |
|---|---|---|
| `SFPU_OP_EXP_INCLUDE` | `1` | Triggers `#include "api/compute/eltwise_unary/exp.h"` |
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | Macro chain that expands to init + func calls |
| `SFPU_OP_CHAIN_0_INIT_0` | `exp_tile_init<{approx}u>();` | Initializes SFPU for exp (approx from param0) |
| `SFPU_OP_CHAIN_0_FUNC_0` | `exp_tile<{approx}u>(0);` | Executes exp on tile in DST[0] |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Indicates input data format |

**Compute Config**:

| Setting | Value | Description |
|---|---|---|
| math_fidelity | HiFi4 | Highest fidelity math mode |
| math_approx_mode | false | `get_op_approx_mode(EXP)` returns false |
| fp32_dest_acc_en | From args | Controls whether DEST accumulator is FP32 |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Base address of source buffer in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to read |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of destination buffer in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) for this core to write |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Unused for exp (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for exp (always 0) |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Uses `TensorAccessor` to resolve page addresses. Reads one page at a time into `c_0` with a NoC barrier after each read. Supports optional `BACKWARDS` mode (not used for exp).

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page writer. Waits for one page in `c_2`, writes via NoC, pops the page. Supports `OUT_SHARDED` and `BACKWARDS` modes (not used for exp in the interleaved factory).

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
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // Conditionally includes exp.h when SFPU_OP_EXP_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Total number of tiles this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for standard unary factory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // Initialize SFPU pipeline: sets up unpack from c_0 and pack to c_2
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // Reserve space in output CB for one tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // Acquire exclusive access to DST register file for math thread

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // Wait until reader has pushed 1 tile into c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // Unpack tile 0 from c_0 into DST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // For exp: expands to exp_tile_init<approx>(); exp_tile<approx>(0);
#endif

            tile_regs_commit();  // Signal that math is done, DST is ready for pack thread

            tile_regs_wait();  // Wait for pack thread to be ready to consume DST

            pack_tile(0, tt::CBIndex::c_2);  // Pack DST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // Free the consumed input tile slot in c_0

            tile_regs_release();  // Release DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // Publish the block of tiles to writer
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
(Wormhole version at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` is identical.)

The high-level API is in: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`

#### Annotated SFPU Kernel Source

The top-level dispatch function `calculate_exponential` is the entry point called by the `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macro:

```cpp
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,           // 8 iterations = 8 SFPU lanes, processing 1 full tile face (32 elements per lane x 8 lanes = 256 elements = half of 32x32 tile); called twice for RC mode
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Fast approximate path: uses TTI macro sequences (SFPLOADMACRO) for hardware-accelerated exp
        // This path delegates to _calculate_exponential_ in the tt_llk submodule
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximate (improved accuracy) path: iterates over SFPU lanes
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // Load current lane's value from DEST register 0
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // Optional input scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);  // Dispatch to appropriate precision path
            sfpi::dst_reg[0] = result;  // Write result back to DEST register 0
            sfpi::dst_reg++;            // Advance to next SFPU lane (next 32-element column)
        }
    }
}
```

The `_sfpu_exp_improved_` template selects between two algorithms based on DEST accumulator precision:

```cpp
// When DEST is bfloat16: use the fast exp_21f polynomial approximation
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);
}

// When DEST is float32: use the accurate Cody-Waite range reduction with Taylor series
template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);
}
```

**Primary implementation -- `_sfpu_exp_21f_` (bfloat16 DEST path)**:

```cpp
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // Algorithm: exp(x) = 2^(x/ln2) = 2^(x_i) * 2^(x_f)
    // Based on Moroz et al. 2022, Section 5 "exp_21f" algorithm

    constexpr float ONE_LN2 = 1.4426950216293334961f;  // 1/ln(2)
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);      // Scale to IEEE-754 biased exponent range: z = x/ln2 + 127

    // Clamp to [0, 255] to prevent overflow/underflow in subsequent bit manipulation
    // Values outside this range would produce invalid IEEE-754 representations
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);  // xlog2 = max(0, xlog2) -- sorts the pair so threshold_low <= xlog2
    sfpi::vec_min_max(xlog2, threshold_high);  // xlog2 = min(xlog2, 255) -- sorts the pair so xlog2 <= threshold_high

    // Convert floating-point xlog2 to integer representation
    // This function extracts exponent and mantissa and shifts to produce an integer
    // The implicit 2^23 scaling is absorbed here (saves one SFPADDI instruction)
    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    // Decompose z into integer and fractional parts using IEEE-754 field extraction
    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract exponent bits = 2^(integer part of x/ln2)
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // Extract 9-bit mantissa = fractional part in [0, 1)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // Convert mantissa integer to float for polynomial eval

    // 2nd-degree polynomial approximation of 2^(x_f) over the range [0, 2^23]
    // Coefficients are pre-scaled to account for the 2^23 factor in frac
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: set the exponent of frac to exponential_part, producing 2^(x_i) * 2^(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // Explicit round-to-nearest-even conversion to bfloat16
        // Prevents accuracy loss from SFPSTORE truncation (e.g., 80.8 -> 81 instead of 80.5)
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

**Helper function -- `_float_to_int32_for_exp21f_`**:

```cpp
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);     // Extract biased exponent field
    sfpi::vInt man = sfpi::exman8(val);     // Extract 8-bit mantissa with implicit leading 1 (value in [1, 2))
    // Shift mantissa left by exponent amount to produce integer value
    // This works because val is pre-scaled so that the implicit 2^23 multiplication is unnecessary
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
    return man;
}
```

**Accurate FP32 implementation -- `_sfpu_exp_f32_accurate_`** (used when `fp32_dest_acc_en = true`):

```cpp
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Overflow/underflow thresholds (in x/ln2 domain)
    constexpr float OVERFLOW_THRESHOLD = 128.0f;     // exp(128*ln2) ~ 3.4e38 (near float max)
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;   // exp(-127*ln2) ~ 5.9e-39 (near float min)

    constexpr float INV_LN2 = 1.4426950408889634f;   // 1/ln(2), full float32 precision
    sfpi::vFloat z = val * INV_LN2;                   // Convert to base-2: z = x / ln(2)

    sfpi::vInt exp_bits = sfpi::exexp(z);             // Extract exponent for NaN detection

    // Conditional handling of special cases using SFPU predicated execution (v_if/v_elseif/v_else)
    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();    // Saturate to +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;                             // Underflow to 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();   // NaN propagation
    }
    v_else {
        // Step 1: Round z to nearest integer k
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Step 2: Cody-Waite range reduction for extended precision
        // r = x - k*ln(2) computed as r = k*(-LN2_HI) + val; then r += k*(-LN2_LO)
        // Negated constants enable single SFPMAD instruction optimization
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;
        sfpi::vFloat r_hi = k * LN2_HI + val;     // High-precision subtraction
        sfpi::vFloat r = k * LN2_LO + r_hi;        // Low-precision correction

        // Step 3: 7th-order Taylor series for exp(r) where |r| < ln(2)/2
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,       // 1
            sfpi::vConst1,       // 1
            0.5f,                // 1/2!
            1.0f / 6.0f,        // 1/3!
            1.0f / 24.0f,       // 1/4!
            1.0f / 120.0f,      // 1/5!
            1.0f / 720.0f,      // 1/6!
            1.0f / 5040.0f      // 1/7!
        );

        // Step 4: Scale by 2^k via exponent manipulation (ldexp)
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);   // Get current exponent of polynomial result
        sfpi::vInt new_exp = p_exp + k_int;             // Add k to shift by 2^k
        result = sfpi::setexp(p, new_exp);              // Write new exponent into result
    }
    v_endif;

    return result;
}
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::exexp(val)` | Extracts the biased exponent field from a float (bits [30:23]) |
| `sfpi::exexp_nodebias(val)` | Extracts the raw exponent field without IEEE-754 bias subtraction |
| `sfpi::exman8(val)` | Extracts 8-bit mantissa with implicit leading 1 bit |
| `sfpi::exman9(val)` | Extracts 9-bit mantissa |
| `sfpi::shft(val, amount)` | Barrel-shifts a value left/right by the specified amount |
| `sfpi::setexp(frac, exp)` | Sets the exponent field of a floating-point value |
| `sfpi::addexp(val, offset)` | Adds an offset to the exponent field (multiply by 2^offset) |
| `sfpi::int32_to_float(val, mode)` | Converts a 32-bit integer to floating-point |
| `sfpi::float_to_fp16b(val, mode)` | Converts float32 to bfloat16 with rounding |
| `sfpi::s2vFloat16b(val)` | Converts a scalar uint16 to SFPU vFloat (BF16 interpretation) |
| `sfpi::vec_min_max(a, b)` | Sorts two vector values so that a <= b after the call |
| `sfpi::reinterpret<T>(val)` | Reinterprets bits of one SFPU type as another (no conversion) |
| `sfpi::dst_reg[n]` | Accesses DEST register at lane offset n |
| `sfpi::dst_reg++` | Advances the DEST register pointer to next SFPU lane |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (per-lane conditional) |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Evaluates polynomial c0 + c1*x + c2*x^2 + ... using Horner's method |
| `sfpi::vConst0` | Vector constant 0.0f |
| `sfpi::vConst1` | Vector constant 1.0f |

#### SFPU Register Usage

- **DEST registers** (`dst_reg[0]`): The tile data resides in DEST after `copy_tile` unpacks it. The SFPU reads from and writes back to DEST register 0 for each lane. After processing, `dst_reg++` advances to the next lane.
- **LREGs (Local Registers)**: SFPU local registers (LReg0-LReg3) are used implicitly by SFPI operations for intermediate values. The `_sfpu_exp_21f_` function uses multiple LREGs for `xlog2`, `z`, `frac`, etc. The compiler manages LREG allocation.
- **Condition codes**: The `v_if`/`v_elseif`/`v_else` constructs in `_sfpu_exp_f32_accurate_` use SFPU condition code registers for per-lane predication.

#### SFPU Execution Flow

1. **Tile acquisition**: `cb_wait_front(c_0, 1)` blocks until the reader has pushed a tile.
2. **Unpack to DEST**: `copy_tile(c_0, 0, 0)` unpacks the tile from CB `c_0` into DEST register 0. The unpack engine converts from the CB data format to the DEST accumulator format.
3. **SFPU init**: `exp_tile_init<approx>()` calls `sfpu::exp_init<approx, fast_approx, scale, clamp_negative>()` which invokes `_init_exponential_` from the tt_llk submodule. This configures SFPU macro sequence registers and loads constants (e.g., -88.5 threshold for clamping in approximate mode).
4. **SFPU compute**: `exp_tile<approx>(0)` dispatches `calculate_exponential` which:
   - In **non-approximate mode** (`approx=false`): iterates 8 times over SFPU lanes. Each iteration loads from `dst_reg[0]`, computes `_sfpu_exp_improved_`, writes result back, and advances to next lane. This is called twice (for VectorMode::RC) to cover all 16 faces of the tile (8 lanes x 2 passes = 16 columns, each with 32 rows).
   - In **approximate mode** (`approx=true`): uses hardware-accelerated TTI_SFPLOADMACRO sequences for faster but less accurate computation.
5. **Pack**: `pack_tile(0, c_2)` packs the result from DEST[0] back into the output CB `c_2`, converting to the output data format.
6. **Release**: `cb_pop_front(c_0, 1)` frees the input CB slot, `cb_push_back(c_2, 1)` publishes the result to the writer.

#### SFPU Configuration

- **`SFPU_OP_EXP_INCLUDE`**: Define set to `1` to include `exp.h` via `sfpu_split_includes.h`.
- **`math_approx_mode`**: Always `false` for exp (from `get_op_approx_mode`). However, the exp operation itself has a parameter (param0) that controls whether `exp_tile_init/exp_tile` are instantiated with `approx=true`. By default, `exp` is created with `UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true))`, meaning `approx=1` is the default.
- **`math_fidelity`**: Set to `MathFidelity::HiFi4` (highest fidelity).
- **`fp32_dest_acc_en`**: Passed from operation attributes. When true, uses `_sfpu_exp_f32_accurate_` (Cody-Waite + 7th-order Taylor). When false, uses `_sfpu_exp_21f_` (Moroz et al. 2nd-degree polynomial).
- **`unpack_to_dest_mode`**: `UnpackToDestFp32` when `preserve_fp32_precision` is set; `Default` otherwise.

#### Hardware Compatibility Notes

The `ckernel_sfpu_exp.h` files under `blackhole/` and `wormhole_b0/` are **identical** in their implementations of `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`, and `calculate_exponential`. The only architectural difference noted in code comments is:
- **Wormhole**: SFPMAD can only do `VD = VA * VB + VC`, so Cody-Waite constants are negated to enable single-instruction optimization.
- **Blackhole**: SFPMAD has `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC` modifiers, but the implementation still uses negated constants for consistency.

The `_calculate_exponential_` function (in the tt_llk submodule, not checked out) provides the approximate-mode implementation using `TTI_SFPLOADMACRO` hardware macro sequences, which may have architecture-specific instruction encodings.

## Implementation Notes

1. **Parameterized approx mode**: Unlike most unary ops where `math_approx_mode` comes from `get_op_approx_mode()`, exp passes its approximation setting as a template parameter via the op's `param0`. The default ttnn `exp` function sets `param0 = true` (approximate), meaning the default behavior uses the fast approximate path through `_calculate_exponential_` in the LLK, not the improved `_sfpu_exp_21f_` path.

2. **Three accuracy tiers**:
   - **Fast approximate** (`approx=true`): Hardware macro-based (`_calculate_exponential_` in tt_llk), fastest but least accurate
   - **Improved bfloat16** (`approx=false`, `fp32_dest_acc_en=false`): `_sfpu_exp_21f_` with 2nd-degree polynomial, good for bfloat16
   - **Accurate float32** (`approx=false`, `fp32_dest_acc_en=true`): `_sfpu_exp_f32_accurate_` with Cody-Waite + 7th-order Taylor, < 1 ULP

3. **Clamping**: In the `_sfpu_exp_21f_` path, input is clamped via `vec_min_max` to prevent IEEE-754 overflow in intermediate bit manipulation. The approximate path has its own clamping via `_calculate_exponential_` (clamping to -88.5).

4. **Program caching**: `override_runtime_arguments` only updates buffer addresses, allowing program reuse across calls with different tensor addresses but same shapes and configurations.

5. **Generic kernel**: `eltwise_sfpu.cpp` is shared by all standard unary SFPU ops. The operation-specific behavior is entirely injected via `SFPU_OP_CHAIN_0` defines at compile time.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary elementwise operation program factory structured? What kernels does it use?"
   **Reason**: Needed to understand the overall architecture of the unary program factory before reading source code.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), the define injection mechanism via `get_block_defines`, and how `UnaryOpType` selects the compute kernel path.

2. **Query**: "What is the SFPU exp kernel implementation? Where is exp_tile defined and what SFPU instructions does it use?"
   **Reason**: Needed to trace the full call chain from `exp_tile` to the SFPU kernel implementation.
   **Key Findings**: Identified three implementations (`_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`), the template-based dispatch via `_sfpu_exp_improved_`, and the key SFPI intrinsics used.

3. **Query**: "Where is _calculate_exponential_ defined? What does it do?"
   **Reason**: The function is called in the in-repo code but defined in the tt_llk submodule which is not checked out.
   **Key Findings**: Confirmed it is in the tt_llk submodule's `ckernel_sfpu_exp.h`. The approximate mode uses TTI_SFPLOADMACRO hardware macro sequences. The non-approximate mode iterates over SFPU lanes calling `_sfpu_exp_improved_`.

### Confluence References
Not consulted for this analysis. The in-repo source code and DeepWiki provided sufficient detail on the SFPU instructions used.

### Glean References
Not consulted for this analysis.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to determine compute kernel path for exp and understand define generation.
   **Key Information**: `exp` uses the default `eltwise_sfpu.cpp` kernel. Defines include `SFPU_OP_EXP_INCLUDE` and `SFPU_OP_CHAIN_0` with `exp_tile_init`/`exp_tile` calls. Default param0 for exp is `true` (approximate mode).

2. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand how `SFPU_TEMPLATE_PARAMS_KERNEL_FN` and `SFPU_TEMPLATE_INIT_KERNEL` macros expand.
   **Key Information**: `SFPU_TEMPLATE_PARAMS_KERNEL_FN` calls `_llk_math_eltwise_unary_sfpu_params_` with `ckernel::sfpu::calculate_exponential` as the functor, passing all template parameters.
