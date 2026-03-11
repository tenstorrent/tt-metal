## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the LOG (natural logarithm) operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (defines `log_tile()` and `log_tile_init()`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel `eltwise_sfpu.cpp` invokes `SFPU_OP_CHAIN_0` which expands to `log_tile_init(); log_tile(0);` (injected via preprocessor defines from `unary_op_utils.cpp`).
2. `log_tile_init<fast_and_approx>()` in `compute_kernel_api.h` calls `MATH(llk_math_eltwise_unary_sfpu_log_init<APPROX, fast_and_approx, DST_ACCUM_MODE>())`.
3. `llk_math_eltwise_unary_sfpu_log_init()` in `llk_math_eltwise_unary_sfpu_log.h` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>()` (which configures SFPU config reg, address modes, and resets counters), then calls `sfpu::log_init<APPROXIMATE, FAST_APPROX, is_fp32_dest_acc_en>()` to load programmable constants.
4. `log_tile(idst)` calls `llk_math_eltwise_unary_sfpu_log()` which dispatches through `_llk_math_eltwise_unary_sfpu_params_()` -- this sets the DST write address, stalls until SFPU is ready, then iterates over 4 faces (in RC vector mode) calling `sfpu::calculate_log()` per face.
5. `calculate_log()` loops 8 iterations (one per row-pair within a 16-row face), reading from `dst_reg[0]`, calling either `calculate_log_body()` (non-FP32 path) or `calculate_log_f32_body()` (FP32 accumulator path), writing back to `dst_reg[0]`, then incrementing `dst_reg++`.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {
    // HAS_BASE_SCALING=false for plain LOG op

    // Normalize input to [1, 2) by setting exponent to IEEE754 bias (127)
    sfpi::vFloat x = sfpi::setexp(in, 127);  // set exp to exp bias (put in range of 1-2)

    // Minimax approximation of log(x) over [1; 2] calculated using Sollya with the following command:
    // > fpminimax(log(x), 5, [|single...|], [1+2^(-20); 2], relative);
    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,   // coeff0 = -2.0069785118103027 (loaded in log_init)
        sfpi::vConstFloatPrgm2,   // coeff1 =  3.767500400543213 (loaded in log_init)
        -2.800232410430908,
        1.3681391477584839,
        -0.3706687390804291,
        0.04224011301994324);

    // Extract debiased exponent as integer
    sfpi::vInt exp = sfpi::exexp(in);

    // Convert negative exponents from two's complement to sign-magnitude for int32_to_float
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);  // 0 = no additional shift
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;    // ln(2) = 0.693147... (loaded in log_init)
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    // ln(0) = -inf
    v_if(in == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            // If input is infinity, return infinity
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {                     // +inf or negative input -> NaN
            result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
        }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));  // convert to bf16
    }

    return result;
}

// Implementation notes, see the original file for more details
template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    // HAS_BASE_SCALING=false for plain LOG op
    sfpi::vFloat result;

    // Check for special cases
    sfpi::vInt exp = sfpi::exexp(val);  // Get debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < 0.f) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == 0.f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        // Step 1: Extract mantissa, normalize to [1, 2)
        sfpi::vFloat m = sfpi::setexp(val, 127);

        // Step 2: Range reduction to [sqrt(2)/2, sqrt(2)]
        v_if(m >= sfpi::vConstFloatPrgm1) {  // vConstFloatPrgm1 = sqrt(2) = 1.41421...
            m = m * 0.5f;
            exp = exp + 1;
        }
        v_endif;

        // Step 3: Transform to z = (m - 1) / (m + 1), mapping to [-0.172, 0.172]
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;

        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1);  // 2 Newton-Raphson iterations
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;

        sfpi::vFloat z2 = z * z;

        // Step 4: Polynomial approximation of ln(m) = 2z(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11)
        sfpi::vFloat p = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,
            0.3333333333333333f,   // 1/3
            0.2f,                   // 1/5
            0.14285714285714285f,   // 1/7
            0.1111111111111111f,    // 1/9
            .09090909090909091f);   // 1/11

        sfpi::vFloat ln_m = 2.0f * (z * p);

        // Convert exponent to sign-magnitude for int32_to_float
        v_if(exp < 0) {
            sfpi::vInt exp_abs = ~exp + 1;
            exp = sfpi::setsgn(exp_abs, 1);  // set sign bit for negative
        }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        // Step 5: ln(x) = exp * ln(2) + ln(m)
        result = expf * sfpi::vConstFloatPrgm2 + ln_m;  // vConstFloatPrgm2 = ln(2) = 0.693147...

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }
    }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
    // APPROXIMATION_MODE=true, FAST_APPROX=false, HAS_BASE_SCALING=false, is_fp32_dest_acc_en=false (typical)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.693147182464599609375;  // ln(2)
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;       // polynomial coeff c0
        sfpi::vConstFloatPrgm2 = 3.767500400543213;          // polynomial coeff c1
    } else {
        // FP32 path: reciprocal init sets vConstFloatPrgm0 = 2.0f (for Newton-Raphson)
        _init_sfpu_reciprocal_</*approximation_mode*/ false>();
        // Additional programmable constants for FP32 path:
        sfpi::vConstFloatPrgm1 = 1.4142135381698608f;   // sqrt(2) -- range reduction threshold
        sfpi::vConstFloatPrgm2 = 0.69314718246459961f;   // ln(2) -- exponent scaling
    }
}
```

**Wormhole B0 differences**: The Wormhole implementation is nearly identical to Blackhole. Key differences:
- `log_init()` FP32 path calls `_init_reciprocal_<false, false>()` instead of `_init_sfpu_reciprocal_<false>()`. On WH, `_init_reciprocal_` delegates to `_init_sfpu_reciprocal_`, which sets all three programmable constants for polynomial reciprocal approximation (`vConstFloatPrgm0`=0.323..., `vConstFloatPrgm1`=1.454..., `vConstFloatPrgm2`=2.121...), consuming all programmable constant slots. The WH FP32 `calculate_log_f32_body` therefore uses `constexpr float` literals for sqrt(2) and ln(2) instead of programmable constants.
- The `_llk_math_eltwise_unary_sfpu_start_` on WH additionally calls `math::set_addr_mod_base()`, and `_done_` additionally calls `TTI_STALLWAIT` for `WAIT_SFPU` and `math::clear_addr_mod_base()`.

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `sfpi::setexp(v, imm)` | Sets the exponent field of a floating-point value to `imm`. Used to normalize mantissa into [1, 2) range by setting exponent to 127 (IEEE754 bias). Maps to `SFPSETEXP`. |
| `sfpi::exexp(v)` | Extracts the debiased exponent of a floating-point value as a signed integer. Maps to `SFPEXEXP`. |
| `sfpi::setsgn(v, bit)` | Sets the sign bit of an integer/float value. Used to convert two's complement negative to sign-magnitude for `int32_to_float`. Maps to `SFPSETSGN`. |
| `sfpi::int32_to_float(v, shift)` | Converts a sign-magnitude integer to IEEE754 float. `shift=0` means no additional exponent shift. Maps to `SFPCAST` (INT32->FP32 variant). |
| `sfpi::float_to_fp16b(v, mode)` | Converts FP32 result to BFloat16 format when destination accumulator is not FP32. Maps to `SFPSTOCHRND` or `SFPCAST` depending on mode. |
| `sfpi::approx_recip(v)` | Hardware reciprocal approximation (used inside `_sfpu_reciprocal_` in the FP32 path). Maps to `SFPLUT` with reciprocal LUT. |
| `sfpi::dst_reg[n]` | Reads/writes the DEST register at offset `n` from the current row pointer. Maps to `SFPLOAD`/`SFPSTORE`. |
| `sfpi::dst_reg++` | Advances the DEST register row pointer by 1 row (2 elements in the SFPU's 32-wide SIMD). Maps to DEST address auto-increment. |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers shared across all SFPU rows. Written via `SFPLOADI` + `SFPCONFIG` path. Mapped to LREG[11..13]. |
| `sfpi::vConst1` | Hardware constant register holding 1.0f. Mapped to LCONST_1. |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution using condition codes. Each comparison sets per-lane condition flags; subsequent operations only execute on lanes where the condition is true. Maps to `SFPSETCC` / `SFPENCC` / `SFPCOMPC` instructions. |
| Arithmetic (`*`, `+`, `-`) on `vFloat` | SFPU multiply-add operations. Map to `SFPMAD` (fused multiply-add) or `SFPMUL` instructions. |
| `PolynomialEvaluator::eval(...)` | Horner's method polynomial evaluation -- expands to a chain of `SFPMAD` instructions at compile time. |
| `sfpi::reinterpret<T>(v)` | Bitwise reinterpretation between vFloat/vInt/vUInt types. No instruction emitted -- it is a type-system cast only. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers** (`dst_reg[0]`) | Source input and result output. Each iteration reads one row-pair from DEST, computes ln(x), and writes back. 8 iterations per face cover all 16 rows of a 16x16 face. |
| **LREG[0..3]** | General-purpose SFPU local registers. Used implicitly by SFPI compiler for intermediate values (`x`, `series_result`, `exp`, `expf`, `result`, etc.). The SFPI compiler manages register allocation across these 4 registers. |
| **vConstFloatPrgm0** (LREG[11]) | Non-FP32 path: `ln(2) = 0.693147...`. FP32 path on BH: `2.0f` (for reciprocal Newton-Raphson). FP32 path on WH: polynomial reciprocal coeff. |
| **vConstFloatPrgm1** (LREG[12]) | Non-FP32 path: polynomial coefficient `-2.0069785...`. FP32 path on BH: `sqrt(2) = 1.41421...`. FP32 path on WH: polynomial reciprocal coeff. |
| **vConstFloatPrgm2** (LREG[13]) | Non-FP32 path: polynomial coefficient `3.76750...`. FP32 path on BH: `ln(2) = 0.693147...`. FP32 path on WH: polynomial reciprocal coeff. |
| **vConst1** (LCONST_1) | Hardware constant `1.0f`. Used in FP32 path for `m - 1` and `m + 1` computations. |
| **Condition code flags** | Per-lane predication flags manipulated by `v_if`/`v_elseif`/`v_else`/`v_endif`. Used for special-case handling (zero, infinity, negative) and sign-magnitude conversion of negative exponents. |

### Address Mode Configuration

The LOG operation uses `SfpuType::log`, which does **not** match any of the special-case `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod`. Therefore, only the default `ADDR_MOD_7` is configured:

**ADDR_MOD_7** (both Blackhole and Wormhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All three address increments are zero. The SFPU kernel manages DEST register advancement explicitly via `sfpi::dst_reg++` within the `calculate_log` loop rather than relying on hardware auto-increment. This is the standard configuration for most SFPU unary operations.

**Wormhole-specific behavior**: The WH `_llk_math_eltwise_unary_sfpu_start_` additionally calls `math::set_addr_mod_base()` before the SFPU stall, and `_done_` calls `math::clear_addr_mod_base()` after. These are not present in the Blackhole version. On WH, `_done_` also includes an explicit `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` to wait for SFPU completion, which Blackhole omits.

The SFPU configuration register is initialized via `_init_sfpu_config_reg()` which executes `TTI_SFPCONFIG(0, 0xF, 1)` -- this sets the SFPU configuration register to enable all 4 rows of the SFPU for processing. Counters are reset via `math::reset_counters(p_setrwc::SET_ABD_F)`.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the log (natural logarithm) SFPU kernel implemented? Trace the call chain from the compute kernel API through LLK dispatch to the core ckernel SFPU implementation."
   **Reason**: Needed to understand the complete call chain and identify all relevant source files.
   **Key Findings**: Confirmed the 4-layer abstraction (compute API -> LLK dispatch -> ckernel SFPU -> params dispatch). Identified that BH and WH implementations are nearly identical. Learned the two code paths: `calculate_log_body` (non-FP32, Minimax polynomial) and `calculate_log_f32_body` (FP32, z-transform with Taylor series).

2. **Query**: "How is the log SFPU kernel implemented in the LLK layer? What SFPU instructions and registers does it use?"
   **Reason**: Needed LLK-layer specifics including the params dispatch mechanism and register usage.
   **Key Findings**: Confirmed the `_llk_math_eltwise_unary_sfpu_params_` dispatch with VectorMode::RC processing all 4 faces. Identified SFPU instructions: `setexp`, `exexp`, `setsgn`, `int32_to_float`.

3. **Query**: "How does the SFPI programming interface support logarithm computation? What intrinsics or instructions are used?"
   **Reason**: Needed to understand the SFPI-level intrinsics that map to hardware instructions.
   **Key Findings**: Confirmed that log is built from primitive operations (no dedicated log instruction). The algorithm uses exponent-mantissa decomposition with polynomial approximation. SFPI maps to SFPSETEXP, SFPEXEXP, SFPSETSGN, SFPMAD assembly instructions.

### Confluence References
No Confluence references were needed for this analysis. The DeepWiki and source code provided sufficient detail on the SFPU instructions used.

### Glean References
No Glean references were needed for this analysis.
