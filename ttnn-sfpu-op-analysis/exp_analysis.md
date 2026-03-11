## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the EXP operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu.h` (init/start/done) and `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (params/face iteration) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h` (lower-level: `_init_exponential_`, `_calculate_exponential_`, `_sfpu_exp_`, `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, `_sfpu_exp_f32_accurate_`) and `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` (upper-level: `calculate_exponential`, `exp_init`, `_sfpu_exp_improved_`) |
| **Parameters Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_TEMPLATE_PARAMS_KERNEL_FN` and `SFPU_TEMPLATE_INIT_KERNEL`) |

### Call Chain

1. **`exp_tile(idst)`** in `exp.h` invokes the `MATH(SFPU_TEMPLATE_PARAMS_KERNEL_FN(...))` macro, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_exponential<...>, idst, vector_mode, scale)`.
2. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) calls `_llk_math_eltwise_unary_sfpu_start_` to set the DST write address and stall until SFPU is ready, then iterates over 4 faces (in RC mode), calling the `calculate_exponential` functor once per face.
3. **`calculate_exponential`** (in `ckernel_sfpu_exp.h` upper-level) branches based on `APPROXIMATION_MODE`: if true, delegates to `_calculate_exponential_` (lower-level); if false, loops over `ITERATIONS` (default 8) reading `dst_reg[0]`, calling `_sfpu_exp_improved_<is_fp32_dest_acc_en>`, writing back, and advancing `dst_reg++`.
4. **`_sfpu_exp_improved_`** dispatches to `_sfpu_exp_21f_<false>` when `is_fp32_dest_acc_en=false` (bfloat16 dest) or `_sfpu_exp_f32_accurate_` when `is_fp32_dest_acc_en=true` (fp32 dest).
5. **`_calculate_exponential_`** (lower-level, for approximation mode) has multiple code paths: `FAST_APPROX && CLAMP_NEGATIVE` uses SFPLOADMACRO sequences for sanitization then computation; `FAST_APPROX` without clamping uses replay-buffer-accelerated LOADMACRO+SFPSHFT2 pairs; non-fast-approx loops calling `_calculate_exponential_piecewise_` which uses `_calculate_exponential_approx_`.
6. After all faces complete, **`_llk_math_eltwise_unary_sfpu_done_`** clears the DST register address.

### Annotated SFPU Kernel Source

The EXP operation has two implementation files. The **upper-level** file wraps the lower-level functions with the tile iteration logic and dispatch based on `APPROXIMATION_MODE` and `is_fp32_dest_acc_en`. The **lower-level** file (from tt_llk) contains the core algorithms and initialization. Both are included below.

#### Upper-Level Wrapper (identical across Blackhole and Wormhole except for minor comment differences)

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat sfpu_exp(sfpi::vFloat val) { return _sfpu_exp_(val); }

// Implementation notes, see the original file for more details
sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);               // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);               // SFPEXMAN: extract mantissa with implicit 1 (8-bit)
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: shift mantissa left by exponent
    return man;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) { // is_fp32_dest_acc_en selects bfp16 rounding at end
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);    // SFPMAD: x/ln2 + bias

    sfpi::vFloat threshold_low = 0.f;                  // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);  // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2);            // SFPSWAP: clamp low
    sfpi::vec_min_max(xlog2, threshold_high);           // SFPSWAP: clamp high

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXEXP (no debias): integer part
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));    // SFPEXMAN: 9-bit mantissa (fractional part)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 -> float

    // 2nd degree polynomial: 2^(x_f) approx on [0, 2^23]
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
    // Expands to Horner form via SFPMAD chain: ((c2*x + c1)*x + c0)

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: combine 2^(x_i) * 2^(x_f)

    if constexpr (!is_fp32_dest_acc_en) {
        // SFPCAST: explicit fp32 -> bfp16b round-to-nearest-even before SFPSTORE truncation
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp_61f_(sfpi::vFloat val) {
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = val * ONE_LN2 + 127.f;

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
    frac = sfpi::addexp(frac, -23);  // SFPADDI: multiply by 2^-23 via exponent add

    // 6th degree polynomial: 2^x on [0, 1]
    frac = PolynomialEvaluator::eval(
        frac, sfpi::vConst1, 0.69314699f, 0.24022982f, 0.055483369f, 0.0096788315f, 0.001243946f, 0.0002170391f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_round_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);  // 2^23 + 2^22

    sfpi::vFloat tmp = z + c231;       // SFPMAD (add)
    sfpi::vFloat k = tmp - c231;       // SFPMAD (sub)
    k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231); // SFPIADD

    return k;
}

sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2;               // SFPMAD: scale to base-2

    sfpi::vInt exp_bits = sfpi::exexp(z);          // SFPEXEXP: for NaN detection

    v_if(z >= OVERFLOW_THRESHOLD) {                // SFPSETCC: condition code >= threshold
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {           // SFPSETCC
        result = sfpi::vConst0;
    }
    v_elseif(exp_bits == 255) {                    // SFPSETCC: NaN check (exponent = 0xFF)
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);

        // Cody-Waite range reduction (constants pre-negated for SFPMAD optimization)
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;     // SFPMAD: r_hi = val - k*ln2_hi
        sfpi::vFloat r = k * LN2_LO + r_hi;       // SFPMAD: r = r_hi - k*ln2_lo

        // 7th order Taylor series for exp(r) via Horner form
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

        // Scale by 2^k via exponent manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);  // SFPEXEXP (no debias)
        sfpi::vInt new_exp = p_exp + k_int;           // SFPIADD
        result = sfpi::setexp(p, new_exp);            // SFPSETEXP
    }
    v_endif;

    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_(sfpi::vFloat val);

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<false>(sfpi::vFloat val) {
    return _sfpu_exp_21f_<false>(val);
}

template <>
sfpi_inline sfpi::vFloat _sfpu_exp_improved_<true>(sfpi::vFloat val) {
    return _sfpu_exp_f32_accurate_(val);
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];     // SFPLOAD: read from DEST
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor); // SFPMAD: optional scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;                // SFPSTORE: write to DEST
            sfpi::dst_reg++;                          // TTI_SETRWC: advance DEST pointer
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### Lower-Level Core SFPU Functions (from tt_llk, Blackhole variant)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h

namespace ckernel::sfpu
{

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);           // SFPEXEXP: extract biased exponent
    v_if (exp >= 0)                        // SFPSETCC: condition on exponent
    {
        val = setexp(val, 126);            // SFPSETEXP: force exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281); // SFPMAD
    val              = val * tmp + sfpi::vConst1;                               // SFPMAD

    v_if (exp >= 0)                        // SFPSETCC
    {
        val = val * val;                   // SFPMUL: first squaring
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;                 // SFPIADD: decrement exponent counter
            v_and(exp >= 0);               // Narrow predication: only lanes with exp >= 0 continue
            val = val * val;               // SFPMUL: repeated squaring
        }
    }
    v_endif;

    return val;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS         = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;

        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;    // Pre-loaded 1/ln2
        sfpi::vFloat conv           = in * vConstLn2Recip;        // SFPMAD: multiply by 1/ln2

        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp    = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73; // SFPIADD: clear exp bits

        tmp += SP_BIAS;                                           // SFPIADD: add bias

        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS)); // SFPSHFT: shift int bits to exponent
    }
    else
    {
        out = _sfpu_exp_(sfpi::setsgn(in, 0));                    // SFPSETSGN + _sfpu_exp_

        v_if (in < 0)                                             // SFPSETCC
        {
            out = _sfpu_reciprocal_<2>(out);                      // reciprocal for negative inputs
        }
        v_endif;
    }

    return out;
}

inline sfpi::vFloat _calculate_exponential_approx_(sfpi::vFloat in)
{
    sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;         // Pre-loaded 1/ln2
    sfpi::vFloat c23_73         = sfpi::vConstFloatPrgm1;         // Pre-loaded C23_73 constant
    sfpi::vInt adj_exp          = sfpi::vConstIntPrgm2;           // Pre-loaded ADJ_EXP
    in                          = in * vConstLn2Recip + c23_73;   // SFPMAD

    sfpi::vInt in_short = adj_exp + sfpi::reinterpret<sfpi::vInt>(in); // SFPIADD: remove exponent + bias mantissa

    in_short <<= 10 - p_exp::FRAC_BITS;                          // SFPSHFT: shift to exponent position
    return sfpi::reinterpret<sfpi::vFloat>(in_short);
}

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{
    sfpi::vFloat result = 0.0f;
    if constexpr (SCALE_EN)
    {
        in = in * sfpi::s2vFloat16b(exp_base_scale_factor);       // SFPMAD: optional scaling
    }
    if constexpr (APPROXIMATION_MODE)
    {
        if constexpr (!SKIP_POSITIVE_CHECK)
        {
            v_if (in >= 89)                                       // SFPSETCC: overflow guard
            {
                sfpi::vFloat in_inf = std::numeric_limits<float>::infinity();
                result              = in_inf;
            }
            v_elseif (in < -42)                                   // SFPSETCC: underflow guard
            {
                result = 0.0f;
            }
            v_else
            {
                result = _calculate_exponential_approx_(in);
            }
            v_endif;
        }
        else
        {
            v_if (in < -42)
            {
                result = 0.0f;
            }
            v_else
            {
                result = _calculate_exponential_approx_(in);
            }
            v_endif;
        }
    }
    else
    {
        result = _sfpu_exp_(sfpi::setsgn(in, 0));                 // SFPSETSGN: force positive, then Horner exp

        v_if (in < 0)                                             // SFPSETCC
        {
            result = _sfpu_reciprocal_<2>(result);                // exp(-x) = 1/exp(x)
        }
        v_endif;
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // Implementation notes, see the original file for more details
        // Uses SFPLOADMACRO sequences: 8 sanitization passes (SWAP against -88.5) + 8 computation passes (MAD, ROUND, SHIFT, STORE)
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);   // Sanitize: LD, SWAP, STORE for each of 8 DEST sub-blocks
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 2);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 4);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 6);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 8);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 10);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 12);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 14);

        // Compute: LD, MAD, ROUND, SHIFT, STORE for each of 8 DEST sub-blocks
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 10);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 12);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14);
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8)
    {
        // 8-element version: replay buffer, ~2.5 cycles/element
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},  // Auto-increment DEST by 2 per LOADMACRO
        }
            .set(ADDR_MOD_7);

        lltt::replay(0, 16);     // Replay 16 instructions (8 LM + 8 SHFT2, first 2 SHFT2 are dummy)

        // Drain: final 2 SHFT2s
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 32)
    {
        // 32-element version: replay buffer, ~2.125 cycles/element
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        lltt::replay(0, 32);     // 2 replays of 32 instructions
        lltt::replay(0, 32);

        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else
    {
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vFloat in     = sfpi::dst_reg[0];  // SFPLOAD
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(in, exp_base_scale_factor);
            sfpi::dst_reg[0]    = result;             // SFPSTORE
            sfpi::dst_reg++;                          // TTI_SETRWC
        }
    }
}

// _init_exponential_ is 340+ lines; abbreviated here, full source in original file.
// Three paths:
// 1. FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE: loads constants into LREGs (A, B-C, threshold=-88.5),
//    programs macro instructions (SWAP, MAD, STOCHRND, SHFT) into macro registers 4-7,
//    configures macro sequence registers 0 and 1.
// 2. FAST_APPROX && APPROXIMATION_MODE && !CLAMP_NEGATIVE: loads constants (A, B-C, shift=15) into LREG[12-14],
//    programs macro instructions (MAD, STOCHRND, SETSGN) via backdoor, configures macro sequence register 0,
//    records 32 instructions into replay buffer (LM+SHFT2 pairs).
// 3. APPROXIMATION_MODE only: loads vConstFloatPrgm0=1.442695 (1/ln2), vConstFloatPrgm1=C23_73, vConstIntPrgm2=ADJ_EXP.
// 4. !APPROXIMATION_MODE: calls _init_sfpu_reciprocal_<false>() for the reciprocal used on negative inputs.

} // namespace ckernel::sfpu
```

### SFPU Instructions Used

The following SFPU instructions and SFPI intrinsics are used across all code paths of the EXP kernel:

| Instruction / Intrinsic | SFPU Instruction | Description |
|--------------------------|------------------|-------------|
| `sfpi::dst_reg[0]` (read) | **SFPLOAD** | Loads a vector from DEST register into an LREG for SFPU processing |
| `sfpi::dst_reg[0] = ...` (write) | **SFPSTORE** | Stores an LREG vector back to DEST register |
| `sfpi::exexp(val)` | **SFPEXEXP** | Extracts the biased exponent field from a float vector |
| `sfpi::exexp_nodebias(val)` | **SFPEXEXP** (no debias variant) | Extracts the raw (non-debiased) exponent field |
| `sfpi::exman8(val)` | **SFPEXMAN** | Extracts 8-bit mantissa with implicit leading 1 |
| `sfpi::exman9(val)` | **SFPEXMAN** | Extracts 9-bit mantissa |
| `sfpi::setexp(val, exp)` | **SFPSETEXP** | Sets the exponent field of a float vector, used to recombine integer and fractional parts |
| `sfpi::setsgn(val, 0)` | **SFPSETSGN** | Forces the sign bit to a given value (0 = positive) |
| `sfpi::shft(val, amt)` | **SFPSHFT** | Arithmetic/logical shift of vector elements |
| `sfpi::addexp(val, imm)` | **SFPADDI** | Adds an immediate to the exponent field (efficient multiply by power of 2) |
| `sfpi::int32_to_float(val, 0)` | **SFPCAST** | Converts integer vector to float |
| `sfpi::float_to_fp16b(val, 0)` | **SFPCAST** | Converts fp32 to bfp16b with round-to-nearest-even |
| `sfpi::vec_min_max(a, b)` | **SFPSWAP** | Swaps elements so the smaller is in `a` and larger in `b` (used for clamping) |
| `val * coeff + addend` | **SFPMAD** | Fused multiply-add, the workhorse of polynomial evaluation |
| `val * val` | **SFPMUL** (via SFPMAD with addend=0) | Multiply used in repeated squaring |
| `v_if / v_elseif / v_else / v_endif` | **SFPSETCC / SFPENCC / SFPCOMPC / SFPPUSHCC / SFPPOPCC** | Per-lane predication (condition codes) controlling which SIMD lanes execute |
| `v_and(cond)` | **SFPSETCC** with AND narrowing | Narrows the active lane mask within a predicated block |
| `reinterpret<vInt>(val) - imm` | **SFPIADD** | Integer subtract on reinterpreted float bits |
| `tmp << shift_amt` | **SFPSHFT** | Left shift to move integer bits into exponent position |
| `TTI_SFPLOADMACRO(...)` | **SFPLOADMACRO** | Executes a pre-programmed macro sequence (LD, MAD, ROUND, SHIFT, STORE) in a single instruction; central to the fast-approx path |
| `TTI_SFPSHFT2(...)` | **SFPSHFT2** | Shift instruction variant used with replay buffer for fast-approx without clamping |
| `TTI_SFPMAD(...)` | **SFPMAD** | Direct TTI macro instruction used for backdoor programming of macro registers |
| `TTI_SFP_STOCH_RND(...)` | **SFP_STOCH_RND** | Stochastic/deterministic rounding, converts FP32 to INT16 in the fast-approx path |
| `TTI_SFPSHFT(...)` | **SFPSHFT** | Direct shift instruction used for backdoor programming |
| `TTI_SFPSETSGN(...)` | **SFPSETSGN** | Direct set-sign instruction used for backdoor programming of macro register 7 |
| `TTI_SFPLOADI(...)` | **SFPLOADI** | Loads a 16-bit immediate into the low or high half of LREG[0] |
| `TTI_SFPCONFIG(...)` | **SFPCONFIG** | Configures SFPU registers: stores LREG[0] into target LREG or macro sequence register |
| `TTI_SFPNOP` | **SFPNOP** | No-operation, used for pipeline timing between dependent operations |
| `TTI_STALLWAIT(...)` | **STALLWAIT** | Stalls until SFPU pipeline is ready (used at start/end of SFPU dispatch) |
| `TTI_SETRWC(...)` | **SETRWC** | Sets/resets read-write counters for DEST register addressing |
| `lltt::replay(...)` | **Replay buffer** | Replays a recorded sequence of instructions from the hardware replay buffer |
| `lltt::record(...)` | **Replay buffer** | Records instructions into the hardware replay buffer |
| `PolynomialEvaluator::eval(...)` | Chain of **SFPMAD** | Evaluates a polynomial using Horner's method, emitting one SFPMAD per coefficient |

### SFPU Register Usage

**LREG[0-3]**: Working registers used by SFPI for intermediate computations. In the SFPI code paths (`_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, `_sfpu_exp_`), the compiler allocates these automatically. In the TTI macro paths, they are used round-robin (LREG0-3 cycle) by SFPLOADMACRO for loading values from DEST and storing intermediate results.

**LREG[4]**: In the fast-approx (no-clamp) path, SFPSHFT2 writes its result to LREG[4], which SETSGN then reads as its VC source.

**LREG[12]**: Stores constant `A = 256.0 / ln(2) = 369.33...` in the fast-approx path. This is the scaling factor for the Schraudolph algorithm.

**LREG[13]**: Stores constant `B - C = 32500.818...` in the fast-approx path. This is the bias correction term.

**LREG[14]**: In the fast-approx with clamping path, stores the threshold `-88.5` used by SFPSWAP for input sanitization. In the fast-approx without clamping path, stores the shift amount `15` for SFPSHFT2.

**LREG[16]** (staging register): Used in the fast-approx (no-clamp) path as a temporary output destination for SETSGN to avoid write-port conflicts with the STORE unit.

**vConstFloatPrgm0**: Stores `1.442695` (1/ln2) for the non-fast approximation mode path.

**vConstFloatPrgm1**: Stores the `C23_73` constant for the non-fast approximation mode.

**vConstIntPrgm2**: Stores the `ADJ_EXP` constant for the non-fast approximation mode.

**DEST registers**: The SFPU reads input tiles from and writes results to DEST. The `_llk_math_eltwise_unary_sfpu_params_` function iterates over 4 faces (each 16 rows x 16 cols), with `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` incrementing the DEST address by 16 between faces. Within each face, `dst_reg++` (SETRWC) advances by 1 row per iteration, processing 8 iterations (rows) per face call.

**Macro Instruction Registers (0-7)**: In the fast-approx path, registers 4-7 are programmed with: (4) SFPSWAP, (5) SFPMAD, (6) SFP_STOCH_RND, (7) SFPSHFT (with clamping) or SFPSETSGN (without clamping). Registers 0-3 are fixed hardware macros (NOP, reserved, NOP, SFPSTORE).

**Macro Sequence Registers**: Sequence Register 0 is configured for the computation macro (LD, MAD, ROUND, SHIFT, STORE). Sequence Register 1 is configured for the sanitization macro (LD, SWAP, STORE) in the clamping variant.

### Address Mode Configuration

The address mode for the EXP SFPU operation is configured in `_llk_math_eltwise_unary_sfpu_init_` which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::exponential>()`.

**Default ADDR_MOD_7** (set for all SFPU unary operations, including EXP):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},   // No auto-increment; SFPI code manages DEST addressing via dst_reg++
}
    .set(ADDR_MOD_7);
```

This is the same on both Wormhole and Blackhole. The `SfpuType::exponential` does not trigger any of the special-case `ADDR_MOD_6` configurations (those are reserved for `reciprocal`, `typecast`, `topk_local_sort`, and min/max operations).

**Override in fast-approx paths**: Within `_calculate_exponential_` for the `FAST_APPROX && APPROXIMATION_MODE && !CLAMP_NEGATIVE` paths (ITERATIONS==8 or ITERATIONS==32), ADDR_MOD_7 is **reconfigured** with `dest.incr = 2` to enable auto-increment of the DEST pointer by 2 for each SFPLOADMACRO instruction. This allows the replay-buffer-based loop to process consecutive DEST sub-blocks without explicit pointer management.

**Wormhole vs Blackhole differences**: The ADDR_MOD configuration is identical between architectures. The only difference is that Wormhole uses literal `3` for the ADDR_MOD parameter in SFPLOADMACRO calls (corresponding to ADDR_MOD_3 which maps to the same slot), while Blackhole uses the named constant `ADDR_MOD_7`. Additionally, Wormhole's `_llk_math_eltwise_unary_sfpu_start_` calls `math::set_addr_mod_base()` and `_llk_math_eltwise_unary_sfpu_done_` calls `math::clear_addr_mod_base()` with an additional `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)`, while Blackhole omits these (addr_mod_base management is not needed on Blackhole).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the EXP SFPU kernel work? Trace the call chain from exp_tile_init/exp_tile in compute_kernel_api through LLK to the ckernel SFPU implementation. What files are involved?"
   **Reason**: Needed to understand the full abstraction layer stack and identify all files involved in the EXP operation's SFPU path.
   **Key Findings**: Identified the 4-layer architecture (API -> LLK macros -> LLK dispatch -> ckernel SFPU), confirmed the existence of `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, and `_sfpu_exp_` as the core algorithms, and learned about the `APPROXIMATION_MODE` / `is_fp32_dest_acc_en` dispatch logic.

2. **Query**: "How is the exponential (exp) SFPU kernel implemented? Show the call chain from llk_math_eltwise_unary_sfpu_exponential through to ckernel_sfpu_exp.h. What SFPU instructions are used?" (asked to `tenstorrent/tt-llk`)
   **Reason**: Needed deeper detail on the LLK-level dispatch, the `_calculate_exponential_` function structure, and the TTI instruction sequences used in the fast-approx path.
   **Key Findings**: Confirmed the SFPLOADMACRO-based fast path, the replay buffer usage for ITERATIONS==8 and ITERATIONS==32, the ADDR_MOD_7 reconfiguration for auto-increment, and the full list of TTI instructions (SFPLOADMACRO, SFPSHFT2, SFPMAD, SFP_STOCH_RND, SFPSHFT, SFPLOADI, SFPCONFIG).

### Confluence References
No Confluence queries were needed for this analysis. The SFPU instructions used in the EXP kernel are well-documented through the source code comments and DeepWiki responses.

### Glean References
No Glean queries were needed for this analysis.
