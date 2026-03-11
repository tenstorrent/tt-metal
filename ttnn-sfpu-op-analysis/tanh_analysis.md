## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the TANH operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (defines `tanh_tile_init<>()` and `tanh_tile<>()`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic variadic dispatcher) |

### Call Chain

1. **Compute kernel** calls `tanh_tile_init<fast_and_approx>()` then `tanh_tile<fast_and_approx>(idst)` from the API header (`compute_kernel_api.h`).
2. **API Header** expands these via the `MATH()` macro to `llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()` and `llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)` in the LLK dispatch header.
3. **LLK Dispatch (init)** calls `llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>()` which invokes `_llk_math_eltwise_unary_sfpu_init_<SfpuType::tanh>()` (sets SFPU config register, configures `ADDR_MOD_7`, resets counters), then calls the init callback `sfpu::tanh_init<APPROXIMATE, is_fp32_dest_acc_en>()`.
4. **LLK Dispatch (compute)** calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_tanh<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, VectorMode::RC)`.
5. **Parameters Dispatch** (`_llk_math_eltwise_unary_sfpu_params_`) sets the DEST write address via `_llk_math_eltwise_unary_sfpu_start_`, stalls until SFPU is ready, then iterates over 4 faces (in RC mode), calling `calculate_tanh<APPROXIMATE, is_fp32_dest_acc_en, 8>()` once per face, incrementing the DEST face address between calls, and finally calls `_llk_math_eltwise_unary_sfpu_done_()`.
6. **Core SFPU** (`calculate_tanh`) processes 8 rows per face (ITERATIONS=8), branching on `APPROXIMATION_MODE` and `is_fp32_dest_acc_en` to select one of three algorithm paths.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat val) { // is_fp32_dest_acc_en=true (only called when true)
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float POLYNOMIAL_THRESHOLD = 0.6f;

    sfpi::vFloat abs_val = sfpi::abs(val); // SFPABS with MOD1_FLOAT

    v_if(abs_val < POLYNOMIAL_THRESHOLD) {
        // Implementation notes, see the original file for more details
        sfpi::vFloat x2 = val * val;

        sfpi::vFloat p = PolynomialEvaluator::eval(
            x2,
            0.999999940395355224609375f,
            -0.33332359790802001953125f,
            0.13310669362545013427734375f,
            -5.21197654306888580322265625e-2f,
            1.5497927553951740264892578125e-2f);

        result = val * p;
    }
    v_else {
        // Normal region: Use tanh(x) = 2*sigmoid(2x) - 1
        sfpi::vFloat two_x = 2.f * val;
        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(two_x); // sigmoid(2x) = 1/(1+exp(-2x))

        result = 2.f * sig - sfpi::vConst1;
    }
    v_endif;

    return result;
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_continued_fraction_(sfpi::vFloat val) {
    // Implementation notes, see the original file for more details

    sfpi::vFloat x = sfpi::abs(val); // SFPABS with MOD1_FLOAT

    sfpi::vFloat x2 = x * x;
    sfpi::vFloat numerator = x * (135135.f + x2 * (17326.f + x2 * (378.f + x2)));

    sfpi::vFloat denominator = PolynomialEvaluator::eval(x2, 135135.f, 62370.f, 3150.f, 28.f);

    sfpi::vFloat result = numerator * ckernel::sfpu::_sfpu_reciprocal_<2>(denominator); // 2 Newton-Raphson iterations

    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value); // SFPSWAP with MOD1_VEC_MIN_MAX; clamps result to [-1,1]

    result = sfpi::setsgn(result, val); // SFPSETSGN; restore original sign

    return result;
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    sfpi::vFloat val = sfpi::abs(x); // SFPABS with MOD1_FLOAT

    // Sollya-optimized degree-6 polynomial in |x|
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,                  // constant term = 0
        0.999004364013671875,
        3.0897438526153564453125e-2,
        -0.4890659749507904052734375,
        sfpi::vConstFloatPrgm2,         // 0.281917631626129150390625 (loaded in tanh_init)
        sfpi::vConstFloatPrgm1,         // -6.6649019718170166015625e-2 (loaded in tanh_init)
        sfpi::vConstFloatPrgm0);        // 5.876733921468257904052734375e-3 (loaded in tanh_init)

    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value); // SFPSWAP with MOD1_VEC_MIN_MAX; clamps to [0,1]

    result = sfpi::setsgn(result, x); // SFPSETSGN; restore original sign

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() { // APPROXIMATION_MODE resolved at compile time, ITERATIONS=8
    if constexpr (APPROXIMATION_MODE) {
        // LUT-based approximation path
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0]; // save LReg0
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1]; // save LReg1
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2]; // save LReg2

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];     // SFPLOAD from DEST
            val = sfpi::lut(val, l0, l1, l2);        // SFPLUT with MOD0_SGN_RETAIN; piecewise linear approx
            sfpi::dst_reg[0] = val;                   // SFPSTORE to DEST

            sfpi::dst_reg++;                          // advance DEST pointer by 1 row
        }

        l_reg[sfpi::LRegs::LReg0] = l0; // restore LReg0
        l_reg[sfpi::LRegs::LReg1] = l1; // restore LReg1
        l_reg[sfpi::LRegs::LReg2] = l2; // restore LReg2
    } else {

        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];      // SFPLOAD from DEST

            sfpi::vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                // FP32 accurate: minimax polynomial for |x|<0.6, sigmoid-based otherwise
                result = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(val);
            } else {
                // BF16: Sollya polynomial approximation
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND FP32->FP16B
            }

            sfpi::dst_reg[0] = result;                // SFPSTORE to DEST
            sfpi::dst_reg++;                          // advance DEST pointer
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        uint imm0 = 0x1DFF;  // LUT coefficients: 0.90625*x (slope and intercept for |x| < 1.0)
        uint imm1 = 0x481A;  // LUT coefficients: 0.09375*x + 0.8125 (for 1.0 <= |x| < 2.0)
        uint imm2 = 0xFF00;  // LUT coefficients: 1.0 (saturated for |x| >= 2.0)
        _sfpu_load_imm16_(0, imm0); // SFPLOADI to LReg0, insmod=2 (unsigned int16)
        _sfpu_load_imm16_(1, imm1); // SFPLOADI to LReg1, insmod=2
        _sfpu_load_imm16_(2, imm2); // SFPLOADI to LReg2, insmod=2
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            sigmoid_init<false>(); // inits reciprocal: sets vConstFloatPrgm0 = 2.0f
        } else {
            // Load polynomial tail coefficients into programmable constant registers
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;  // SFPCONFIG path
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;      // SFPCONFIG path
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;        // SFPCONFIG path
        }
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

The tanh kernel uses different instruction sets depending on the algorithm path:

**Approximation Mode (LUT path)**:
| Instruction | SFPI Function | Description |
|-------------|---------------|-------------|
| `SFPLOADI` | `_sfpu_load_imm16_()` | Loads 16-bit immediate values into LRegs 0-2 as LUT coefficients during init. `insmod=2` writes unsigned int16, right-justified with zero-padded MSBs. |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads a vector of 32 elements from DEST register into an SFPU local register. |
| `SFPLUT` | `sfpi::lut(val, l0, l1, l2)` | Piecewise linear function evaluation. Selects one of 3 coefficient pairs from LReg[0..2] based on input magnitude ranges ([0,1), [1,2), [2,inf)), extracts 8-bit slope and intercept via `Lut8ToFp32()`, computes `slope * |x| + intercept`. Uses `MOD0_SGN_RETAIN` to preserve the input sign in the result. |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` (write) | Stores an SFPU local register value back to DEST register. |

**Non-approximation, BF16 path (polynomial)**:
| Instruction | SFPI Function | Description |
|-------------|---------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads from DEST. |
| `SFPABS` | `sfpi::abs()` | Computes absolute value of a floating-point vector. Uses `MOD1_FLOAT`. |
| `SFPMAD` | Arithmetic operators (`*`, `+`, `-`) | Multiply-add operations for polynomial (Horner's method) evaluation. Each `a + x * b` compiles to an SFPMAD. |
| `SFPSWAP` | `sfpi::vec_min_max()` | With `MOD1_VEC_MIN_MAX`, computes element-wise min/max between two registers. Used to clamp result to [0, 1] before sign restoration. |
| `SFPSETSGN` | `sfpi::setsgn(result, x)` | Copies the sign bit from the original input `x` to the result, implementing `tanh(-x) = -tanh(x)`. |
| `SFPSTOCHRND` | `sfpi::float_to_fp16b()` | Converts FP32 result to BF16 (FP16B) format. Uses `MOD1_FP32_TO_FP16B`. The parameter `0` selects stochastic rounding. |
| `SFPSTORE` | `sfpi::dst_reg[0] = result` (write) | Stores result back to DEST. |
| `SFPCONFIG` | `sfpi::vConstFloatPrgm{0,1,2} = ...` | Writes polynomial coefficients to programmable constant registers (shared across rows) during init. |

**Non-approximation, FP32 path (accurate sigmoid-based)**:
All the above plus:
| Instruction | SFPI Function | Description |
|-------------|---------------|-------------|
| `SFPLZ` | (via `v_if` condition code) | Compare operations for branching (`|x| < 0.6`). |
| `SFPARECIP` | `sfpi::approx_recip()` | Hardware approximate reciprocal instruction. Used inside `_sfpu_reciprocal_<2>()` as the initial estimate before Newton-Raphson refinement. `MOD1_RECIP` mode. |
| `SFPMAD` | (Newton-Raphson iterations) | Two refinement iterations inside `_sfpu_reciprocal_<2>()`: computes `t = x*y - 2.0`, then `y = y * -t`. |
| `SFPNOP` | (implicit) | Pipeline NOPs inserted by the compiler between dependent SFPMAD instructions. |

### SFPU Register Usage

**DEST Registers**:
- The SFPU reads input values from and writes results back to the DEST register file. Each face consists of 8 rows (ITERATIONS=8) of 32 elements. The `dst_reg` pointer auto-increments after each iteration via `dst_reg++`.
- DEST base address is set by `_llk_math_eltwise_unary_sfpu_start_` using the `dst_index` parameter, and face advancement (by 16 rows = 2 x `inc_dst_addr<8>()`) is done between the 4 face iterations in the params dispatcher.

**SFPU Local Registers (LRegs)**:
- **Approximation mode**: LReg0, LReg1, LReg2 hold the three LUT coefficient pairs (slope + intercept encoded as 16-bit values). LReg3 is implicitly used by the `SFPLUT` instruction as the input operand. The kernel saves and restores LReg0-2 around the computation loop.
- **Polynomial mode (BF16)**: Uses LRegs as temporary operands for SFPMAD chains. Programmable constant registers (`vConstFloatPrgm0/1/2`, which map to LRegs 11-13 via SFPCONFIG) hold the three highest-degree polynomial coefficients.
- **FP32 accurate mode**: Uses `vConstFloatPrgm0 = 2.0f` (set by `sigmoid_init`) for Newton-Raphson reciprocal refinement. LRegs are used as temporaries for the minimax polynomial evaluation and the sigmoid computation path.

**Hardware Constants**:
- `vConst0` (0.0f), `vConst1` (1.0f), `vConst1p4424` (~1.4424), `LCONST_neg1` (-1.0f) are fixed hardware constants available to the SFPU without loading.

### Address Mode Configuration

The SFPU address mode for tanh is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::tanh>()` by calling `eltwise_unary_sfpu_configure_addrmod<SfpuType::tanh>()`.

Since `SfpuType::tanh` does not match any of the special-cased types (topk_local_sort, reciprocal, typecast, unary_max/min), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This means DEST register addressing does NOT auto-increment between SFPU instructions within a single iteration. Instead, the SFPU kernel manages DEST advancement explicitly via `dst_reg++` (which compiles to an SFPINCRWC or equivalent pointer increment) after each row is processed.

**Wormhole vs Blackhole**: The `ADDR_MOD_7` configuration is identical across both architectures for `SfpuType::tanh`. The only difference in the overall `_llk_math_eltwise_unary_sfpu_start_` flow is that Wormhole additionally calls `math::set_addr_mod_base()` on entry and `math::clear_addr_mod_base()` plus an extra stall (`STALL_CFG, WAIT_SFPU`) on exit; Blackhole omits these steps. The Blackhole `eltwise_unary_sfpu_configure_addrmod` also includes `SfpuType::reciprocal` in its `ADDR_MOD_6` special case, which Wormhole does not -- but this does not affect tanh.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the tanh SFPU kernel implemented? What is the call chain from the compute kernel API (tanh_tile) through LLK dispatch to the core SFPU implementation?"
   **Reason**: Needed to identify the full abstraction layer chain and file paths for the tanh operation.
   **Key Findings**: Confirmed the 3-path algorithm (LUT approximation, polynomial, sigmoid-based accurate), file locations in both blackhole and wormhole_b0 directories, and that implementations are identical across architectures.

2. **Query**: "How is the tanh SFPU kernel implemented in LLK? What is the call chain from llk_math_eltwise_unary_sfpu_tanh through to the ckernel implementation?"
   **Reason**: Needed to understand the LLK dispatch layer mechanics and the `_llk_math_eltwise_unary_sfpu_params_` function's role.
   **Key Findings**: Confirmed the params dispatcher iterates over faces, calling the SFPU function once per face with the `_start_`/`_done_` bookending pattern.

3. **Query**: "How does SFPI implement tanh? What SFPU instructions and intrinsics are used?"
   **Reason**: Needed to understand what hardware instructions the SFPI abstractions compile to.
   **Key Findings**: LUT path uses `SFPLUT` with `MOD0_SGN_RETAIN`; `sfpi::lut()` compiles to `__builtin_rvtt_sfplut`. The LUT performs piecewise linear approximation over 3 input magnitude ranges.

4. **Query**: "What does the SFPLUT instruction do? What are the modes (SFPLUT_MOD0_SGN_RETAIN)?"
   **Reason**: Needed detailed SFPLUT instruction semantics for the approximation path.
   **Key Findings**: SFPLUT selects one of 3 LReg coefficient pairs based on |input| ranges [0,1), [1,2), [2,inf). Each pair contains an 8-bit slope and intercept, converted to FP32 via `Lut8ToFp32()`. `SGN_RETAIN` preserves the original sign bit in the output.

5. **Query**: "What SFPU instructions do sfpi::abs(), sfpi::setsgn(), sfpi::vec_min_max(), sfpi::float_to_fp16b(), sfpi::approx_recip(), sfpi::lut() compile to?"
   **Reason**: Needed to map high-level SFPI functions to actual hardware instructions for the instruction usage table.
   **Key Findings**: `abs()` -> `SFPABS`, `setsgn()` -> `SFPSETSGN`, `vec_min_max()` -> `SFPSWAP` with `MOD1_VEC_MIN_MAX`, `float_to_fp16b()` -> `SFPSTOCHRND` with `MOD1_FP32_TO_FP16B`, `approx_recip()` -> `SFPARECIP`, `lut()` -> `SFPLUT`.

### Confluence References
No Confluence page sections were consulted for this analysis. The DeepWiki queries provided sufficient SFPU ISA detail.

### Glean References
No confidential hardware specs were needed for this analysis.
