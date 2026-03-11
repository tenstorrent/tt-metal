## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the SQRT operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN`) and `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (`_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_sqrt.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu.h` (`_llk_math_eltwise_unary_sfpu_init_`, `_llk_math_eltwise_unary_sfpu_start_`, address mode config) and `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` (`llk_math_eltwise_unary_sfpu_init`) |

### Call Chain

1. **`sqrt_tile(idst)`** (in `sqrt.h`) expands the `MATH(...)` macro which gates execution to the math RISC-V core, then invokes `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN(calculate_sqrt, APPROX, 8, DST_ACCUM_MODE, FAST_APPROX, idst, RC)`.
2. **`SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN`** (in `llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_sqrt<APPROX, 8, DST_ACCUM_MODE, FAST_APPROX>, idst, (int)VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, stalls until the SFPU is ready, then iterates through 4 tile faces (for `VectorMode::RC`), calling the functor once per face.
4. **`ckernel::sfpu::calculate_sqrt<APPROX, 8, DST_ACCUM_MODE, FAST_APPROX>()`** (in `ckernel_sfpu_sqrt.h` at the arch-specific ckernel layer) delegates to `_calculate_sqrt_<APPROX, 8, DST_ACCUM_MODE, FAST_APPROX>(8)`.
5. **`_calculate_sqrt_`** (in tt_llk `ckernel_sfpu_sqrt.h`) dispatches to `_calculate_sqrt_internal_` (non-legacy path), which loops 8 iterations per face, reading `dst_reg[0]`, computing `_calculate_sqrt_body_`, writing the result back, and advancing `dst_reg++`.
6. **`_calculate_sqrt_body_`** performs the actual square root approximation using an integer bit-manipulation seed followed by Newton-Raphson-style refinement.

For initialization: `sqrt_tile_init()` calls `SFPU_INIT_KERNEL_CALL(sqrt, sfpu::sqrt_init, APPROX)` which expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROX>(sqrt_init<APPROX>)`. This first calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sqrt>()` (configures SFPU config register, address modes, resets counters), then calls `sqrt_init<APPROX>()` which delegates to `_init_sqrt_<APPROX>()` to load magic constants into programmable constant registers.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h
// (Wormhole B0 version is functionally identical)

// Implementation notes, see the original file for more details
// Reference: Kokosinski et al., "Fast and accurate approximation algorithms for
// computing floating point square root", Numerical Algorithms (2024).

template <bool APPROXIMATE = false, bool RECIPROCAL = false, bool FAST_APPROX = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x) // RECIPROCAL=false for sqrt (true for rsqrt)
{
    // Integer bit-shift seed: i = reinterpret(x) >> 1 gives initial rsqrt approximation
    sfpi::vInt i   = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    // y0 = reinterpret(magic_constant - i) -- initial reciprocal sqrt estimate
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    if constexpr (APPROXIMATE)
    {
        // Algorithm SQRT_10-bits, with modifications for reciprocal.
        sfpi::vFloat c           = x * y;
        sfpi::vFloat negative_y  = -y;
        sfpi::vFloat infinity    = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::reinterpret<sfpi::vInt>(infinity);
        sfpi::vFloat t           = sfpi::vConstFloatPrgm1 + negative_y * c; // t = C1 - y*c, one Newton step
        if constexpr (RECIPROCAL)
        {
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x != 0.
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)
            {
                y = y * t;
            }
            // Otherwise, if x = 0, then y = inf; if x = inf, then y = 0.
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            y = c; // c = x * y0 is the sqrt estimate (not rsqrt)
            // If x != inf.  Otherwise, y = inf, since c = inf.
            v_if (sfpi::reinterpret<sfpi::vInt>(x) != infinity_bits)
            {
                y = y * t; // refine: sqrt(x) ~= c * t
            }
            v_endif;
        }
    }
    else
    {
        // Algorithm SQRT_23-bits, with modifications for reciprocal.
        sfpi::vFloat xy            = x * y;
        sfpi::vFloat negative_y    = -y;
        sfpi::vFloat c             = negative_y * xy; // c = -y * (x*y)
        sfpi::vFloat infinity      = sfpi::s2vFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits   = sfpi::reinterpret<sfpi::vInt>(infinity);
        // Two-term polynomial refinement of rsqrt: y = y * (C1 + c*(C2 + c))
        y                          = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
        xy                         = x * y;
        negative_y                 = -y;
        sfpi::vFloat one_minus_xyy = sfpi::vConst1 + (negative_y * xy); // residual: 1 - x*y^2

        if constexpr (RECIPROCAL)
        {
            sfpi::vFloat half_y              = sfpi::addexp(y, -1); // half_y = y * 2^(-1) = y/2
            sfpi::vInt x_bits                = sfpi::reinterpret<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x != 0.
            v_if (infinity_minus_x_bits != 0 && x_bits != 0)
            {
                y = one_minus_xyy * half_y + y; // Newton correction for rsqrt
            }
            // Otherwise, if x = 0, then y = inf; if x = inf, then y = 0.
            v_else
            {
                y = sfpi::reinterpret<sfpi::vFloat>(infinity_minus_x_bits);
            }
            v_endif;
        }
        else
        {
            sfpi::vFloat half_xy = 0.5f * xy; // half_xy = x*y/2
            // If x == inf, we need to skip to avoid y = inf - inf = nan; y will already be inf.
            v_if (sfpi::reinterpret<sfpi::vInt>(x) < infinity_bits)
            {
                y = one_minus_xyy * half_xy + xy; // Newton correction for sqrt
            }
            v_endif;
        }
    }

    if constexpr (!FAST_APPROX)
    {
        v_if (x < 0.0f)
        {
            y = std::numeric_limits<float>::quiet_NaN(); // negative input -> NaN
        }
        v_endif;
    }

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = tmp; // write FP32 result directly
        }
        else
        {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0)); // convert to FP16b before writeback
        }
        sfpi::dst_reg++; // advance to next row within the face
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat = false>
inline void _calculate_sqrt_(int iterations) // legacy_compat=false in standard path
{
    if constexpr (legacy_compat)
    {
        return _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(iterations);
    }
    else
    {
        return _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>(iterations);
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
inline void _init_sqrt_() // legacy_compat=false in standard path
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            // SQRT_10-bits magic constants
            sfpi::vConstIntPrgm0   = 0x5f0b3892; // magic constant for initial rsqrt seed
            sfpi::vConstFloatPrgm1 = 1.89099014875f; // refinement coefficient C1
        }
        else
        {
            // SQRT_23-bits magic constants
            sfpi::vConstIntPrgm0   = 0x5f1110a0; // magic constant for initial rsqrt seed
            sfpi::vConstFloatPrgm1 = 2.2825186f;  // polynomial coefficient C1
            sfpi::vConstFloatPrgm2 = 2.2533049f;  // polynomial coefficient C2
        }
    }
}
```

### SFPU Instructions Used

| SFPI Construct / Instruction | SFPU Hardware Instruction | Description |
|------------------------------|---------------------------|-------------|
| `sfpi::reinterpret<vInt/vFloat/vUInt>(...)` | None (zero-cost type alias) | Bitwise reinterpretation between vector types without modifying bit patterns |
| `sfpi::dst_reg[0]` (read) | `SFPLOAD` | Loads a vector from the DEST register file at the current address |
| `sfpi::dst_reg[0] = ...` (write) | `SFPSTORE` | Stores a vector back to the DEST register file |
| `sfpi::dst_reg++` | `TTINCRWC` | Increments the DEST register write cursor by 2 rows (advances within a face) |
| `sfpi::vConstIntPrgm0 = ...` | `SFPLOADI` / CREG write | Writes a 32-bit integer constant to programmable constant register PRGM0 |
| `sfpi::vConstFloatPrgm1 = ...` | `SFPLOADI` / CREG write | Writes a float constant to programmable constant register PRGM1 |
| `sfpi::vConstFloatPrgm2 = ...` | `SFPLOADI` / CREG write | Writes a float constant to programmable constant register PRGM2 |
| `sfpi::vConstIntPrgm0` (read) | CREG file access | Reads the programmable constant register as integer |
| `sfpi::vConstFloatPrgm1` (read) | CREG file access | Reads the programmable constant register as float |
| `sfpi::vConst1` (read) | CREG file access | Reads the hardwired constant 1.0f from the constant register file |
| `sfpi::s2vFloat16b(...)` | `SFPLOADI` with `SFPLOADI_MOD0_FLOATB` | Compile-time FP32-to-FP16b conversion; loads result as immediate |
| `sfpi::addexp(y, -1)` | `SFPDIVP2` with `MOD1_ADD` | Adds -1 to the exponent of y, effectively computing y/2 (only used in RECIPROCAL path) |
| `float_to_fp16b(tmp, 0)` | `SFPSTOCHRND` with `MOD1_FP32_TO_FP16B` | Runtime FP32-to-FP16b conversion with round-to-even (mode=0) |
| `v_if (condition)` | `SFPXCMP` or `SFPXFCMP` + `SFPPUSHC` | Evaluates comparison, pushes condition code for predicated execution |
| `v_else` | `SFPCOMPC` + `SFPPUSHC` | Complements condition code for the else branch |
| `v_endif` | `SFPPOPC` | Pops condition code stack, restoring prior predication state |
| Arithmetic: `*`, `+`, `-` | `SFPMUL`, `SFPADD`, `SFPMUL`/`SFPADD` with negate | Vector multiply, add, and negate operations on SFPU ALU |
| `>> 1` (on vUInt) | `SFPSHFT` | Right-shift integer vector by 1 bit (halves integer representation of float for seed) |
| Integer subtract (`vConstIntPrgm0 - i`) | `SFPIADD` (with negate) | Integer subtraction for the magic-constant-minus-shifted-bits seed computation |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **`dst_reg[0..7]`** (per face) | Each iteration reads from `dst_reg[0]` (current row), computes sqrt, writes result back, then increments. 8 iterations per face process all 8 row-pairs (16 rows = half a 32-row face, with each SFPU row covering 2 tile rows). |
| **`vConstIntPrgm0`** (CREG PRGM0) | Holds the magic integer constant for the initial reciprocal-sqrt seed. `0x5f0b3892` in approximate mode, `0x5f1110a0` in full-precision mode. This is analogous to the "fast inverse square root" magic number. |
| **`vConstFloatPrgm1`** (CREG PRGM1) | Holds the first refinement coefficient. `1.89099014875f` (approx) or `2.2825186f` (full precision). |
| **`vConstFloatPrgm2`** (CREG PRGM2) | Holds the second polynomial coefficient. Only used in full-precision mode: `2.2533049f`. Unused in approximate mode. |
| **`vConst1`** (hardwired CREG) | The constant `1.0f`, used in the full-precision path for computing the residual `1 - x*y^2`. |
| **LREG (local registers)** | Intermediate values (`i`, `y`, `c`, `xy`, `negative_y`, `t`, `half_xy`, `one_minus_xyy`, `infinity`, `infinity_bits`) are held in SFPU local registers (LREGs). The compiler allocates these from the available LREG pool (up to 4 per thread on Wormhole, more on Blackhole). |

### Address Mode Configuration

The SQRT operation uses `SfpuType::sqrt`, which is **not** in the special-case lists in `eltwise_unary_sfpu_configure_addrmod()`. Therefore, only the default **`ADDR_MOD_7`** is configured:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All increments are zero. The SFPU kernel manages DEST register advancement explicitly via `sfpi::dst_reg++` (which emits `TTINCRWC` instructions) rather than relying on automatic address mode increments.

This configuration is **identical between Wormhole B0 and Blackhole**. The only difference is that Blackhole includes `SfpuType::reciprocal` in the `ADDR_MOD_6` special-case list while Wormhole does not, but this does not affect SQRT.

Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (advancing by 16 rows) to move from one 16x16 face to the next. On Wormhole, `_llk_math_eltwise_unary_sfpu_start_` additionally calls `math::set_addr_mod_base()` and `_llk_math_eltwise_unary_sfpu_done_` calls `math::clear_addr_mod_base()` with an extra SFPU stall; Blackhole omits these calls.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SQRT SFPU kernel work? Trace the call path from the compute kernel API through LLK dispatch down to the ckernel SFPU implementation."
   **Reason**: Needed to identify all files in the abstraction layers and understand the macro-based dispatch mechanism.
   **Key Findings**: Identified the 4-layer abstraction (API header -> LLK macros -> LLK params -> ckernel SFPU), confirmed the `SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN` macro is the dispatch entry, and the `SFPU_INIT_KERNEL_CALL` macro for init.

2. **Query**: "How is the SQRT SFPU kernel implemented in the LLK/ckernel layer?" (to `tenstorrent/tt-llk`)
   **Reason**: Needed the detailed ckernel implementation including the algorithm, constants, and iteration structure.
   **Key Findings**: Confirmed the `_calculate_sqrt_body_` function uses a magic-number initial seed (similar to fast inverse sqrt) followed by Newton-Raphson refinement. Two algorithm variants: SQRT_10-bits (approximate, ~10 bits precision) and SQRT_23-bits (full, ~23 bits precision).

3. **Query**: "Explain SFPI constructs: reinterpret, s2vFloat16b, addexp, vConstIntPrgm, dst_reg, v_if/v_else/v_endif, float_to_fp16b and their hardware instruction mappings." (to `tenstorrent/sfpi`)
   **Reason**: Needed to map every SFPI construct used in the kernel to actual SFPU hardware instructions.
   **Key Findings**: Complete mapping obtained. Key insights: `reinterpret` is zero-cost, `addexp` maps to `SFPDIVP2`, `float_to_fp16b` maps to `SFPSTOCHRND`, condition codes use `SFPPUSHC`/`SFPPOPC` stack. `dst_reg++` generates `TTINCRWC`.

### Confluence References
No Confluence queries were needed. The DeepWiki responses from `tenstorrent/sfpi` and `tenstorrent/tt-isa-documentation` provided sufficient SFPU instruction detail.

### Glean References
No Glean queries were needed. All required information was available through open-source repositories.
