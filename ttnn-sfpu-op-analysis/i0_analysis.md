## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the I0 (zeroth order modified Bessel function of the first kind) operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/i0.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro-based, no dedicated LLK file for I0) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h` |
| **Parameters Dispatch** | `llk_math_eltwise_unary_sfpu_params.h` (from tt_llk submodule, provides `_llk_math_eltwise_unary_sfpu_params_`) |

### Call Chain

1. The compute kernel calls `i0_tile(idst)` from the API header `eltwise_unary/i0.h`.
2. `i0_tile` expands the macro `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_i0, RC, APPROX, idst)`, which resolves to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_i0<APPROX>, idst, (int)VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (from the tt_llk submodule) sets the destination write address via `_llk_math_eltwise_unary_sfpu_start_`, then iterates over all 4 faces of the tile (VectorMode::RC), calling `calculate_i0<true>()` for each face, and finishes with `_llk_math_eltwise_unary_sfpu_done_`.
4. `calculate_i0<true>()` in `ckernel_sfpu_i0.h` executes 8 iterations (one per row pair within a face), reading from `dst_reg[0]`, computing the polynomial approximation, and writing back.

The init path follows: `i0_tile_init()` expands `SFPU_UNARY_KERNEL_INIT(i0, APPROX)` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::i0, true>()`, which in turn calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::i0>()` to configure ADDR_MOD_7 and prepare the SFPU pipeline.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h
// NOTE: Wormhole and Blackhole implementations are identical.

using namespace sfpi;

namespace ckernel {
namespace sfpu {

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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() { // APPROXIMATION_MODE=true (resolved via APPROX define)
#pragma GCC unroll 0         // Prevent compiler unrolling to reduce code size

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];    // SFPLOAD from dest register at current offset
        vFloat x = input * input;     // SFPMUL: compute x^2, basis for polynomial

        result = 1.0f + POLYVAL10(    // Horner's method: 10th degree polynomial in x^2
                            1.50E-22f,            // coef10 (x^20 term)
                            7.24E-20f,            // coef9  (x^18 term)
                            2.90E-17f,            // coef8  (x^16 term)
                            9.39E-15f,            // coef7  (x^14 term)
                            2.40E-12f,            // coef6  (x^12 term)
                            4.71E-10f,            // coef5  (x^10 term)
                            6.78E-08f,            // coef4  (x^8 term)
                            0.000006781684028f,   // coef3  (x^6 term)
                            0.0004340277778f,     // coef2  (x^4 term)
                            0.015625f,            // coef1  (x^2 term) = 1/64
                            0.25f,                // coef0  (x^0 term in poly, added to 1.0)
                            x);

        dst_reg[0] = result;  // SFPSTORE result back to dest register
        dst_reg++;            // TTI_INCRWC: advance dest pointer by SFP_DESTREG_STRIDE (2)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** (`dst_reg[0]` read) | Loads a 64-element vector from the destination register file at the current offset into an SFPU local register (LReg). Generated by the `vFloat input = dst_reg[0]` expression. |
| **SFPMUL** (`*` operator on vFloat) | Performs element-wise floating-point multiplication of two vectors. Used for `input * input` (computing x^2) and for each Horner step in the `POLYVAL10` macro where `... * t4` appears. |
| **SFPMAD** (multiply-add fusion) | The compiler may fuse sequences of `a * b + c` (from Horner's method steps like `(coef_n + coef_{n+1} * t4) * t4`) into SFPMAD (multiply-accumulate) instructions. Each Horner step is a candidate for MAD fusion. |
| **SFPIADD / SFPADD** (addition with float immediates) | Used for the `1.0f + ...` addition and for adding polynomial coefficients. The compiler selects between immediate-add forms depending on operand types. |
| **SFPSTORE** (`dst_reg[0] = result`) | Stores a 64-element vector from an SFPU local register back to the destination register file. Uses `SFPSTORE_ADDR_MODE_NOINC` (no auto-increment on store). |
| **TTI_INCRWC** (`dst_reg++`) | Increments the destination register write pointer by `SFP_DESTREG_STRIDE` (2), advancing to the next row pair within the tile face. Maps to `__builtin_rvtt_ttincrwc`. |
| **SFPLOADI** (float literal loads) | Loads immediate floating-point constants (the polynomial coefficients) into SFPU local registers. The compiler generates these for each coefficient in the POLYVAL10 expansion. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **dst_reg (DEST register file)** | The primary data source and sink. Each iteration reads one 64-element vector from `dst_reg[0]` (the current row pair), computes the I0 result, and writes it back to the same location. The pointer is then incremented by 2 (`dst_reg++`). Over 8 iterations, this processes all 16 rows of a tile face (8 iterations x 2 rows per DEST entry = 16 rows). |
| **LReg (Local Registers, L0-L7)** | SFPU local registers used as temporaries for the polynomial evaluation. The compiler allocates LRegs for `input`, `x` (= input^2), intermediate Horner accumulations, and the final `result`. The I0 kernel requires multiple LRegs simultaneously due to the deep Horner chain. Typically L0-L3 are used for operands/results, with the compiler managing spills if needed. |
| **vConstFloatPrgm0-3** | Not explicitly used in this kernel. The I0 implementation relies on inline float literals rather than pre-loaded programmable constants. |

### Address Mode Configuration

The I0 operation uses the default address mode configuration for standard unary SFPU operations, set during `i0_tile_init()` via `_llk_math_eltwise_unary_sfpu_init_<SfpuType::i0>()` which calls `eltwise_unary_sfpu_configure_addrmod()`.

**ADDR_MOD_7** (default for unary SFPU):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

All increments are zero because the SFPU kernel manages destination register advancement explicitly via `dst_reg++` (TTI_INCRWC) rather than relying on hardware auto-increment. This configuration is the same across Wormhole and Blackhole architectures. The `_llk_math_eltwise_unary_sfpu_params_` function handles face-to-face advancement using TTI_SETRWC instructions between face iterations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the I0 (modified Bessel function of first kind, order 0) SFPU kernel work? What is the implementation in ckernel_sfpu_i0.h?"
   **Reason**: Initial understanding of the I0 SFPU kernel structure and polynomial approximation strategy.
   **Key Findings**: I0 uses a 10th-degree polynomial (POLYVAL10 macro) evaluated via Horner's method on x^2 (the squared input). Processes 8 iterations per face with `APPROXIMATION_MODE=true`.

2. **Query**: "What does _llk_math_eltwise_unary_sfpu_params_ do? How does it set up address modes (ADDR_MOD) and call the SFPU function?"
   **Reason**: Understanding the LLK dispatch layer that bridges the API call to the core SFPU kernel.
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` sets up the dest write address, iterates over tile faces based on VectorMode (RC = all 4 faces), calls the SFPU function for each face, and cleans up. ADDR_MOD_7 with zero increments is the default configuration.

3. **Query**: "How do dst_reg reads and writes work in SFPI? When code does dst_reg[0] = result and dst_reg++, what SFPU instructions are generated?"
   **Reason**: Understanding the instruction-level mapping of SFPI C++ constructs to SFPU hardware instructions.
   **Key Findings**: `dst_reg[0]` read generates SFPLOAD, `dst_reg[0] = result` generates SFPSTORE with NOINC mode, `dst_reg++` generates TTI_INCRWC incrementing by SFP_DESTREG_STRIDE=2. vFloat multiplication maps to SFPMUL, with multiply-add patterns potentially fusing into SFPMAD.

### Confluence References

No Confluence references were needed. The I0 kernel uses only basic SFPU instructions (LOAD, STORE, MUL, MAD, LOADI, INCRWC) which are well-documented in DeepWiki.

### Glean References

No Glean references were needed for this analysis.
