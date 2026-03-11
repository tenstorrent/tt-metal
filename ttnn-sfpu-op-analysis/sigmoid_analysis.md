## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the SIGMOID operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sigmoid.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** calls `sigmoid_tile(i)` from `compute_kernel_api.h`.
2. **`sigmoid_tile<vec_mode, fast_and_approx>(idst)`** wraps `MATH((llk_math_eltwise_unary_sfpu_sigmoid<fast_and_approx, DST_ACCUM_MODE>(idst, vec_mode)))`, dispatching to the math thread.
3. **`llk_math_eltwise_unary_sfpu_sigmoid<APPROXIMATE, is_fp32_dest_acc_en>(dst_index, vector_mode)`** calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sigmoid<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode)`.
4. **`_llk_math_eltwise_unary_sfpu_params_`** (in tt_llk) sets the DST write address, stalls until MATH is ready, then iterates over faces (4 faces for RC mode), calling `calculate_sigmoid` once per face, advancing the DST pointer by 16 rows between faces.
5. **`calculate_sigmoid<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS=8>()`** either calls `_sfpu_sigmoid_` in a loop (accurate path) or delegates to `calculate_sigmoid_appx` (approximate path).
6. **`_sfpu_sigmoid_<is_fp32_acc_to_dest_mode>(x)`** computes `1 / (1 + exp(-x))` using `_sfpu_exp_improved_` (or `_sfpu_exp_21f_`) followed by `_sfpu_reciprocal_`.

Init chain: `sigmoid_tile_init<fast_and_approx>()` -> `llk_math_eltwise_unary_sfpu_sigmoid_init<APPROXIMATE>()` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>(sfpu::sigmoid_init<APPROXIMATE>)` -> first calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sigmoid>()` (configures ADDR_MOD_7, resets counters), then calls `sigmoid_init<APPROXIMATE>()` which loads reciprocal polynomial coefficients (accurate path) or LUT constants (approximate path).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h
// (Wormhole B0 and Blackhole versions are identical)

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);  // Cody-Waite range reduction + 7th order Taylor
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);  // Moroz et al. exp_21f algorithm (~1 ULP bfloat16)
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;  // 1 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);  // 2 Newton-Raphson iters for float32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator);  // 1 Newton-Raphson iter for bfloat16 precision
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() { // APPROXIMATION_MODE=false (typical), is_fp32_dest_acc_en=DST_ACCUM_MODE, ITERATIONS=8
    if constexpr (!APPROXIMATION_MODE) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // Load from current DST row

            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);

            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));  // Round to bfloat16
            }

            sfpi::dst_reg[0] = result;  // Store back to DST
            sfpi::dst_reg++;  // Advance to next row
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();  // LUT-based fast approximation path
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_reciprocal_<false, false>();  // Loads reciprocal polynomial coefficients into PrgmRegs
    } else {
        sigmoid_appx_init();  // Loads LUT constants into LReg0-2 via SFPLOADI
    }
}
```

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h
// (Wormhole B0 and Blackhole versions are identical)

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    vUInt l0 = l_reg[LRegs::LReg0];  // Save LUT coefficients from local regs
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];

        dst_reg[0] = lut(val, l0, l1, l2) + 0.5f;  // SFPLUT + SFPADDI: piecewise linear approx + offset

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;  // Restore local regs (they may be clobbered by lut)
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

inline void sigmoid_appx_init() {
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x3DFF;
    imm1 = 0x21D8;
    imm2 = 0xFF10;
    TTI_SFPLOADI(0, 2, imm0);  // Load LUT entry 0 into LReg0, mod=2 (16-bit immediate)
    TTI_SFPLOADI(1, 2, imm1);  // Load LUT entry 1 into LReg1
    TTI_SFPLOADI(2, 2, imm2);  // Load LUT entry 2 into LReg2
}
```

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h
// (Called by _sfpu_sigmoid_ for the denominator reciprocal)

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Scale input to [1.0, 2.0) range and negate: setman copies mantissa from in into -1.0
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2 (minimizes max relative error on [1,2])
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Implementation notes, see the original file for more details
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue with quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Scale factor: set mantissa to zero
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Scale factor adjustment: scale = scale*0.5
    scale *= 0.5f;

    // Continue Newton-Raphson: y = y + y*t
    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Second Newton-Raphson iteration: t = 1.0 - x*y; y = y + y*t
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling factor and restore input sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Polynomial y = k2 - k1*x + k0*x^2 minimises max relative error for 1/x over [1,2), via Sollya
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2
}
```

### SFPU Instructions Used

**Accurate path** (`APPROXIMATION_MODE=false`):

| SFPI Intrinsic / Instruction | Description |
|-----|-------------|
| `dst_reg[0]` (SFPLOAD/SFPSTORE) | Load a row of 32 elements from DST register into SFPU LREG, and store back after computation |
| `dst_reg++` (SFPINCRWC) | Increment the DST register write counter to advance to the next row |
| `operator*`, `operator+` (SFPMUL/SFPMAD) | Floating-point multiply and multiply-accumulate, used throughout exp and reciprocal computations |
| `operator-` (SFPMUL with -1, or negation via sign bit) | Negate input for `exp(-x)` |
| `sfpi::vConst1` (SFPLOAD from const reg) | Load constant 1.0 for `1 + exp(-x)` addition |
| `sfpi::setman` (SFPSETMAN) | Set the mantissa field of a float -- used in reciprocal to normalize input to [1,2) and clear mantissa of scale factor |
| `sfpi::setsgn` (SFPSETSGN) | Set the sign bit of the result to match the input sign in reciprocal |
| `sfpi::setexp` (SFPSETEXP) | Set the exponent field -- used in exp to recombine integer and fractional parts as `2^(x_i) * 2^(x_f)` |
| `~vUInt` (SFPNOT) | Bitwise NOT used in reciprocal to compute `255 - in.Exp` for the scale factor |
| `sfpi::exexp` / `sfpi::exexp_nodebias` (SFPEXEXP) | Extract exponent (with/without IEEE-754 bias removal) -- used in exp for overflow detection and integer part extraction |
| `sfpi::exman8` / `sfpi::exman9` (SFPEXMAN) | Extract mantissa with 8-bit or 9-bit padding -- used in exp for mantissa/fractional part extraction |
| `sfpi::shft` (SFPSHFT) | Logical shift used in `_float_to_int32_for_exp21f_` to convert float to fixed-point integer |
| `sfpi::int32_to_float` (SFPCAST) | Convert integer to float for the fractional part in exp_21f |
| `sfpi::addexp` (SFPDIVP2) | Add integer to exponent field, used in exp_61f to multiply by 2^-23 |
| `sfpi::vec_min_max` (SFPMIN/SFPMAX or conditional swap) | Clamp values to [0, 255] range in exp to avoid overflow |
| `sfpi::float_to_fp16b` (SFPSTOCHRND) | Convert float32 to bfloat16 with round-to-nearest-even when `is_fp32_dest_acc_en=false` |
| `sfpi::reinterpret<vFloat/vInt>` | Bitcast between float and integer vector types (no instruction emitted, just type reinterpretation) |
| `v_if`/`v_elseif`/`v_else`/`v_endif` (SFPSETCC/SFPENCC/SFPCOMPC) | Conditional execution via condition codes in the fp32 accurate exp path for overflow/underflow/NaN handling |
| PolynomialEvaluator::eval (series of SFPMAD) | Horner's method polynomial evaluation -- compiles to a chain of SFPMAD instructions |

**Approximate path** (`APPROXIMATION_MODE=true`):

| SFPI Intrinsic / Instruction | Description |
|-----|-------------|
| `dst_reg[0]` (SFPLOAD/SFPSTORE) | Load/store from DST register |
| `dst_reg++` (SFPINCRWC) | Advance DST row pointer |
| `l_reg[LRegs::LRegN]` (SFPLOAD/SFPSTORE local) | Load/store from SFPU local registers LReg0-2 |
| `lut(val, l0, l1, l2)` (SFPLUT) | Hardware 3-entry LUT instruction: selects one of l0/l1/l2 based on input exponent, computes piecewise linear `a*|x| + b` |
| `+ 0.5f` (SFPADDI) | Add immediate 0.5 to shift the LUT output to the sigmoid range [0, 1] |
| `TTI_SFPLOADI` (SFPLOADI) | Load 16-bit immediate into local register during init |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST register (dst_reg)** | Source and destination for tile data. Each face has 16 rows of 16 elements (32-wide vectors processed 2 faces at a time). The SFPU reads one row at a time via `dst_reg[0]`, processes it, writes back, then advances via `dst_reg++`. |
| **LREG0-LREG7** | SFPU local registers (32-element vectors). In accurate mode, intermediates live in compiler-allocated LREGs. In approximate mode, LReg0-2 hold LUT coefficients loaded during init; they are saved/restored around the `lut()` call since `lut` may clobber them. |
| **vConstFloatPrgm0** | Holds reciprocal polynomial coefficient k0 = 0.3232325 (accurate path) |
| **vConstFloatPrgm1** | Holds reciprocal polynomial coefficient k1 = 1.4545460 (accurate path) |
| **vConstFloatPrgm2** | Holds reciprocal polynomial coefficient k2 = 2.1212125 (accurate path) |
| **vConst0** | Constant 0.0, used for underflow result in exp |
| **vConst1** | Constant 1.0, used for `1 + exp(-x)` and Newton-Raphson iterations |
| **vConstNeg1** | Constant -1.0, used in reciprocal to normalize input via `setman` |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sigmoid>()` configures a single address mode:

**ADDR_MOD_7** (used for all SFPU load/store operations):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
.set(ADDR_MOD_7);
```

All three fields (srca, srcb, dest) have zero auto-increment. This means the DST register address does NOT auto-increment between SFPU instructions within a single row computation. Row advancement is handled explicitly by `dst_reg++` (which compiles to SFPINCRWC) and by the `TTI_SETRWC` calls in `_llk_math_eltwise_unary_sfpu_params_` that advance the DST pointer by 16 rows between faces.

This configuration is **identical on both Wormhole B0 and Blackhole**. No additional address modes (e.g., ADDR_MOD_6) are configured for `SfpuType::sigmoid` specifically. Note that on Blackhole, `SfpuType::reciprocal` gets an additional `ADDR_MOD_6` with `dest.incr=2`, but this is not used for the sigmoid operation since sigmoid uses its own `SfpuType::sigmoid` enum.

Note on architectural difference in `_llk_math_eltwise_unary_sfpu_params_`: Wormhole B0 calls `math::set_addr_mod_base()` before the SFPU stall and `math::clear_addr_mod_base()` after, whereas Blackhole omits these calls. This reflects differences in how the two architectures manage address mode base registers. The Blackhole version also uses `_llk_math_eltwise_unary_sfpu_start_`/`_llk_math_eltwise_unary_sfpu_done_` helpers instead of inlining the logic, and it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (which calls `math::inc_dst_addr<8>()` twice) to advance between faces, compared to Wormhole's `TTI_SETRWC` approach. Both ultimately achieve the same 16-row advancement per face.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the sigmoid SFPU kernel implemented? Trace from the compute kernel API through the LLK dispatch layer down to the ckernel SFPU implementation."
   **Reason**: To identify the full call chain and file locations for each abstraction layer.
   **Key Findings**: Confirmed the 3-layer architecture (API -> LLK dispatch -> ckernel SFPU), identified file paths for both Wormhole and Blackhole variants, and learned that sigmoid supports both accurate (exp+reciprocal) and approximate (LUT) paths.

2. **Query**: "How is the sigmoid SFPU kernel implemented in tt-llk? Show the call chain from llk_math_eltwise_unary_sfpu to the ckernel_sfpu_sigmoid implementation."
   **Reason**: To understand the tt-llk library's role in dispatching SFPU work and the face iteration mechanism.
   **Key Findings**: Confirmed `_llk_math_eltwise_unary_sfpu_params_` is the central dispatch function that handles face iteration, DST address management, and SFPU stalling. Also learned about `ADDR_MOD_7` configuration with zero increments.

3. **Query**: "What is the sfpi::lut() function? How does it work and what SFPU instruction does it map to? Also explain dst_reg, l_reg, vFloat, vConst1, setman, setsgn, setexp, exexp, exexp_nodebias, exman8, exman9, float_to_fp16b, shft, addexp, int32_to_float, and vec_min_max."
   **Reason**: To understand the SFPI intrinsics used in the sigmoid kernel and their mapping to hardware instructions.
   **Key Findings**: `lut()` maps to `SFPLUT` hardware instruction, performs piecewise linear approximation using exponent-based entry selection. Confirmed mappings for all major SFPI intrinsics to their underlying SFPU instructions (SFPSETMAN, SFPSETSGN, SFPSETEXP, SFPEXEXP, SFPEXMAN, SFPSTOCHRND, SFPSHFT, SFPCAST, SFPDIVP2).

### Confluence References
No Confluence page was consulted for this analysis. The DeepWiki and source code provided sufficient detail for all SFPU instructions used.

### Glean References
No Glean searches were performed for this analysis.
