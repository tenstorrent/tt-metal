## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the SiLU (Sigmoid Linear Unit) operation, which computes `silu(x) = x * sigmoid(x)`.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** calls `silu_tile(idst)` from the tile-level API (`compute_kernel_api.h`), which expands via the `MATH(...)` macro to invoke `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)`.
2. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_silu`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, vector_mode)`, which sets the DST write address, stalls until the SFPU is ready, and invokes the functor once per face (4 faces for `VectorMode::RC`).
3. **Core SFPU function** `calculate_silu<is_fp32_dest_acc_en, 8>()` executes the inner loop: for each of 8 rows within a face, it loads from `dst_reg[0]`, computes `x * sigmoid(x)` by calling `_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)`, optionally rounds to bfloat16, and stores back.
4. **Sigmoid helper** `_sfpu_sigmoid_` computes `1 / (1 + exp(-x))` by calling either `_sfpu_exp_improved_` (fp32 mode) or `_sfpu_exp_21f_` (bfloat16 mode) for `exp(-x)`, then `_sfpu_reciprocal_` with 2 or 1 Newton-Raphson iterations respectively.

**Initialization chain**: `silu_tile_init()` calls `llk_math_eltwise_unary_sfpu_silu_init<APPROX>()`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::silu>()` (configures SFPU config register, address mode, resets counters), then calls `silu_init<APPROX>()` which loads the reciprocal polynomial coefficients into programmable constant registers.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h
// (Blackhole variant is identical)

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS> // ITERATIONS=8 (one 16-row face processed as 8 SFPU rows of 2 datums each)
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];          // SFPLOAD: load current row from DEST register

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x); // SFPMUL of x with sigmoid result

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND: round fp32 to bf16
        }

        sfpi::dst_reg[0] = result;                  // SFPSTORE: write result back to DEST register
        sfpi::dst_reg++;                             // Advance DEST register pointer by 1 row
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    // Wormhole variant: passes APPROXIMATION_MODE through to reciprocal init
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();              // Load reciprocal polynomial coefficients (k0, k1, k2) into vConstFloatPrgm{0,1,2}
    } else {
        _init_sfpu_reciprocal_<true>();               // Same coefficients regardless of APPROXIMATION_MODE
    }
}

}  // namespace ckernel::sfpu
```

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h
// (Blackhole variant is identical)
// Helper function called by calculate_silu

namespace ckernel::sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);   // Higher-accuracy exp for fp32 mode
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);        // ~1 ULP accuracy exp for bfloat16 mode; template param true = negate input
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x; // 1 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);   // 2 Newton-Raphson iterations for fp32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator);   // 1 Newton-Raphson iteration for bfloat16 precision
    }

    return result;
}

}  // namespace ckernel::sfpu
```

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h
// Reciprocal function called by _sfpu_sigmoid_

namespace ckernel::sfpu {

// max_iter=2: fp32 precision; max_iter=1: bfloat16 precision; max_iter=0: same as 1 currently
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: scale input to [-2,-1), preserving mantissa

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD: k1 + k0*(-x)

    // Scale factor: ~in gives 255-in.Exp for correct power-of-2 scaling
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);                     // SFPNOT: bitwise complement for exponent inversion

    y = sfpi::vConstFloatPrgm2 + y * negative_x;                                    // SFPMAD: k2 + (k1+k0*(-x))*(-x)

    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN: zero mantissa of scale factor

    // First Newton-Raphson iteration
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;                                // SFPMAD: 1 + (-x)*y = residual
    scale *= 0.5f;                                                                   // SFPMUL: adjust scale exponent by -1
    y = y + y * t;                                                                   // SFPMAD: y*(1+t) = refined estimate

    if constexpr (max_iter > 1)
    {
        // Second Newton-Raphson iteration for fp32 precision
        t = sfpi::vConst1 + negative_x * y;                                         // SFPMAD
        y = y + y * t;                                                               // SFPMAD
    }

    y = y * scale;                                                                   // SFPMUL: apply power-of-2 scaling
    y = sfpi::setsgn(y, in);                                                         // SFPSETSGN: restore original sign

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Minimax polynomial coefficients for 1/x over [1,2) via Sollya
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // SFPLOADI: k0 (x^2 coefficient)
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // SFPLOADI: k1 (x coefficient, negated in use)
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // SFPLOADI: k2 (constant term)
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (`dst_reg[0]` read) | Loads a vector of elements from the DEST register file at the current row address into an SFPU local register (vFloat). |
| `SFPSTORE` (`dst_reg[0]` write) | Stores a vector of elements from an SFPU local register back to the DEST register file at the current row address. |
| `SFPMUL` (vFloat `*` operator) | Performs element-wise floating-point multiplication between two SFPU vector registers. Used for `x * sigmoid(x)`, `scale *= 0.5f`, `y * scale`, and intermediate products. |
| `SFPMAD` (fused multiply-add via `a + b * c`) | Fused multiply-add operation. Used extensively in the reciprocal polynomial evaluation (`k1 + k0*x`, `k2 + y*x`) and Newton-Raphson iterations (`1 + (-x)*y`, `y + y*t`). |
| `SFPADD` (vFloat `+` operator) | Element-wise floating-point addition. Used for `1 + exp(-x)` in sigmoid denominator computation. |
| `SFPNEG` (unary `-` operator) | Negates a floating-point vector. Used to compute `-x` for `exp(-x)` in sigmoid. |
| `SFPNOT` (`~` on vUInt) | Bitwise NOT on integer vector. Used in reciprocal to compute `255 - exponent` for scale factor derivation. |
| `SFPSETMAN` (`sfpi::setman`) | Sets the mantissa field of a float vector from another source while preserving sign/exponent. Used to normalize input to [1,2) range and to zero out scale factor mantissa. |
| `SFPSETSGN` (`sfpi::setsgn`) | Sets the sign bit of a float vector to match another value. Used at the end of reciprocal to restore original sign. |
| `SFPLOADI` (vConstFloatPrgm assignment) | Loads an immediate constant into one of the SFPU programmable constant registers (Prgm0/1/2). Used during init to load reciprocal polynomial coefficients. |
| `SFPSTOCHRND` (`float_to_fp16b`) | Stochastic or nearest rounding from fp32 to bfloat16 format. Applied when `is_fp32_dest_acc_en=false` to truncate results before storing. |
| `SFPLUT` (inside `_sfpu_exp_21f_`) | Lookup table instruction used by the exp approximation kernel for initial range-reduced exponential estimate. |

Additionally, the exp helper functions (`_sfpu_exp_21f_` and `_sfpu_exp_improved_`) use their own set of SFPU instructions internally (SFPLUT, SFPMUL, SFPMAD, SFPEXEXP, SFPSETEXP, etc.) for exponential computation -- these are documented in the exp SFPU analysis.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Source and destination for tile data. Each `dst_reg[0]` read/write accesses the current row (set by the DST write address pointer). The pointer auto-increments via `dst_reg++` after each iteration. |
| **SFPU local registers (LRegs)** | `vFloat` variables (`x`, `result`, `exp_neg_x`, `denominator`, `negative_x`, `y`, `scale`, `t`) are allocated to SFPU local registers L0-L3 by the compiler. The SFPU has 4 local registers per lane. |
| **vConstFloatPrgm0** | Loaded with reciprocal polynomial coefficient k0 = 0.3232325... during `silu_init()`. |
| **vConstFloatPrgm1** | Loaded with reciprocal polynomial coefficient k1 = 1.4545459... during `silu_init()`. |
| **vConstFloatPrgm2** | Loaded with reciprocal polynomial coefficient k2 = 2.121212... during `silu_init()`. |
| **vConst1** | Hardware constant register holding 1.0f. Used in sigmoid for `1 + exp(-x)` and in reciprocal for Newton-Raphson residual `1 + (-x)*y`. |
| **vConstNeg1** | Hardware constant register holding -1.0f. Used in reciprocal `setman` to scale input to [-2, -1) range. |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::silu>()` via `eltwise_unary_sfpu_configure_addrmod<SfpuType::silu>()`.

For SiLU (and all standard unary SFPU operations that are not topk, typecast, or min/max), the configuration sets **ADDR_MOD_7** with all-zero increments:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

This means:
- **srca increment**: 0 (not used by SFPU operations)
- **srcb increment**: 0 (not used by SFPU operations)
- **dest increment**: 0 -- the DEST register pointer is not auto-incremented by the address mode hardware; instead, the SFPU kernel manually advances it via `dst_reg++` (which compiles to `SFPINCRWC` or equivalent) after each row iteration.

ADDR_MOD_7 is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2 which are typically used by the A2D (accumulate-to-DEST) path.

The `_llk_math_eltwise_unary_sfpu_params_` function also calls `math::set_addr_mod_base()` before the SFPU functor and `math::clear_addr_mod_base()` after completion. Between face iterations, it issues `TTI_SETRWC` instructions to advance the DEST pointer by 16 rows (2 increments of 8) to move to the next face.

This configuration is **identical for Wormhole and Blackhole** -- the `eltwise_unary_sfpu_configure_addrmod` function has the same implementation on both architectures for `SfpuType::silu`.

**Architectural note on init**: The Wormhole `silu_init` passes `APPROXIMATION_MODE` through to `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()`, while the Blackhole variant always calls `sigmoid_init<false>()` which calls `_init_reciprocal_<false, false>()` then `_init_sfpu_reciprocal_<false>()`. In practice the loaded coefficients are the same regardless of `APPROXIMATION_MODE`, so the behavior is functionally identical.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SILU (SiLU / Swish) unary SFPU kernel work? What is the call chain from the compute kernel API through LLK to the ckernel SFPU implementation? What files implement the SILU SFPU kernel?"
   **Reason**: To establish the full call chain and identify all relevant source files before reading code.
   **Key Findings**: Confirmed the 4-layer abstraction (API -> LLK -> ckernel -> sub-functions), identified that SiLU depends on sigmoid which depends on exp and reciprocal, located the ckernel_sfpu_silu.h files for both Wormhole and Blackhole.

2. **Query**: "How is the SILU SFPU kernel implemented in the LLK layer? What ckernel functions are called for silu?" (asked to `tenstorrent/tt-llk`)
   **Reason**: To understand the LLK dispatch mechanism and the params function that orchestrates face iteration.
   **Key Findings**: Confirmed `_llk_math_eltwise_unary_sfpu_params_` handles face iteration with TTI_SETRWC between faces, identified ADDR_MOD_7 configuration with zero increments, and confirmed the init/start/done lifecycle.

### Confluence References
No Confluence queries were needed for this analysis. The SFPU instructions used (SFPLOAD, SFPSTORE, SFPMUL, SFPMAD, SFPNOT, SFPSETMAN, SFPSETSGN, SFPLOADI, SFPSTOCHRND) are well-documented via DeepWiki and source code comments.

### Glean References
No Glean queries were needed for this analysis. The source code and DeepWiki provided sufficient detail on all SFPU instructions and register usage.
