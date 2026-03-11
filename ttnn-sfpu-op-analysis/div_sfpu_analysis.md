## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the DIV (legacy SFPU) binary operation.

**Important context**: The `get_defines_fp32` function in `binary_op_utils.cpp` generates the defines for the fp32 SFPU path of DIV. For non-INT32 floating-point types, the defines produced are:
- `BINOP_INIT` = `div_binary_tile_init();`
- `BINARY_SFPU_OP` = `div_binary_tile(i*2, i*2+1, i*2);`

For INT32 inputs, the defines use `div_int32_tile_init()` / `div_int32_tile` instead. This analysis covers the floating-point path (`div_binary_tile`).

The compute kernel (`eltwise_binary_sfpu_kernel.cpp`) loads both input tiles into DST at interleaved positions (input A at `i*2`, input B at `i*2+1`), then calls `BINOP_INIT` followed by `BINARY_SFPU_OP` per tile. The result overwrites DST at `i*2` (same as input A).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (metal overlay) and `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` (LLK source-of-truth) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** calls `div_binary_tile_init()` and `div_binary_tile(i*2, i*2+1, i*2)` via the `BINOP_INIT` and `BINARY_SFPU_OP` preprocessor defines.
2. **`div_binary_tile_init()`** (API header) expands to `MATH(llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>())`.
3. **`llk_math_eltwise_binary_sfpu_binop_init`** (LLK dispatch) calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` to configure SFPU registers and address modes, then calls `sfpu_binary_init<APPROX, BinaryOp::DIV>()` which delegates to `_sfpu_binary_init_<APPROX, BinaryOp::DIV>()`.
4. **`_sfpu_binary_init_`** (LLK core) detects `BINOP == BinaryOp::DIV` and calls `_init_sfpu_reciprocal_<false>()` to load polynomial coefficients into programmable constant registers.
5. **`div_binary_tile(idst0, idst1, odst)`** (API header) expands to `MATH(llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>(idst0, idst1, odst))`.
6. **`llk_math_eltwise_binary_sfpu_binop_div`** (LLK dispatch) calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary_div<APPROX, BinaryOp::DIV, 8, is_fp32_dest_acc_en>, ...)`.
7. **`_llk_math_eltwise_binary_sfpu_params_`** (LLK params) sets the DST write address, stalls until SFPU is ready, then iterates over 4 faces (RC mode), calling `calculate_sfpu_binary_div` for each face and advancing the DST address by 16 rows between faces.
8. **`calculate_sfpu_binary_div`** (metal overlay in `ckernel_sfpu_binary.h`) is the core SFPU function that performs `in0 * _sfpu_reciprocal_<2>(in1)` with special-case handling for division edge cases.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h
// (metal overlay — identical for Wormhole and Blackhole)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=APPROX (compile-time), BINOP=BinaryOp::DIV, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 rows / SFP_DESTREG_STRIDE(2) = 32 addressable rows per tile in SFPI
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load row from input A tile in DST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load row from input B tile in DST
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1); // Core division: a / b = a * (1/b) with 2 Newton-Raphson iterations

        v_if(in1 == 0) { // Handle division by zero
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); } // 0/0 = NaN
            v_else {
                result = std::numeric_limits<float>::infinity(); // x/0 = +/-inf
                result = sfpi::setsgn(result, in0); // Sign of result matches sign of numerator
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; } // x/x = 1.0 (exact result avoidance of rounding error)
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // When DST is not FP32, apply software Round-to-Nearest-Even to truncate to BF16
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result row to output tile in DST
        sfpi::dst_reg++; // Advance to next row across all tile views
    }
}
```

The `float32_to_bf16_rne` helper used for non-FP32 mode:

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in); // Reinterpret float as unsigned int
    sfpi::vUInt lsb = (bits >> 16) & 1; // Extract LSB of BF16 mantissa for tie-breaking
    bits = bits + 0x7fffU + lsb; // Add rounding bias: 0x7fff + lsb implements RNE
    bits = bits & 0xFFFF0000U; // Zero out lower 16 bits to get BF16 in upper half
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}
```

The core reciprocal function differs between architectures.

**Wormhole B0** uses a pure SFPI polynomial + Newton-Raphson approach:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) {
    // Scale input to [1.0, 2.0) range and negate
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    // Coefficients loaded during init: PrgmConst0=0.3232325, PrgmConst1=1.4545459, PrgmConst2=2.1212124
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Scale factor: 1/in = 1/x * 2^(127-in.Exp)
    // scale.Exp = 255 - in.Exp = ~in.Exp (via SFPNOT)
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Set mantissa to zero for clean power-of-two scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First Newton-Raphson iteration: t = 1.0 - x*y, y = y + y*t
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Adjust scale: scale*0.5 handles +-inf/+-0 edge cases
    scale *= 0.5f;

    y = y + y * t;

    if constexpr (max_iter > 1) {
        // Second Newton-Raphson iteration
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scale and restore original sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_() {
    // Polynomial coefficients for quadratic initial estimate of 1/x on [1,2)
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2
}
```

**Blackhole** uses hardware `approx_recip` (SFPARECIP instruction) + Newton-Raphson refinement:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x) {
    // Hardware approximate reciprocal instruction (SFPARECIP)
    sfpi::vFloat y = sfpi::approx_recip(x);

    if constexpr (max_iter > 0) {
        // t = x * y - 2.0 (negated form for NaN detection via sign check)
        // vConstFloatPrgm0 was set to 2.0 during init
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (max_iter > 1) {
            sfpi::vFloat y1 = y * -t - sfpi::vConst0; // y1 = y * (2 - x*y)
            // NaN check: if t=NaN from 0*inf, t>=0 so we skip refinement
            v_if (t < 0) {
                t = x * y1 - sfpi::vConstFloatPrgm0; // Second iteration
                y = y1 * -t - sfpi::vConst0;
            }
            v_endif;
        } else {
            v_if (t < 0) { // Skip if NaN (from 0*inf)
                y = y * -t - sfpi::vConst0;
            }
            v_endif;
        }
    }

    return y;
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_() {
    if constexpr (!APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 2.0f; // Constant used in Newton-Raphson: t = x*y - 2.0
    }
}
```

The initialization function called from `_sfpu_binary_init_`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_() {
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW) {
        // For DIV: init reciprocal with APPROXIMATION_MODE=false (always uses 2 NR iterations)
        _init_sfpu_reciprocal_<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[offset]` (load) | **SFPLOAD** -- Reads a vector row from DEST register file into an SFPU local register (LREG). The offset indexes into the tile's rows. |
| `sfpi::dst_reg[offset] = val` (store) | **SFPSTORE** -- Writes an SFPU local register value back to a DEST register row. |
| `sfpi::dst_reg++` | **SFPINCRWC** -- Increments the SFPU's internal DEST row pointer by the stride amount (typically 2). |
| `*` (vFloat multiply) | **SFPMUL** (or **SFPMAD** with addend=0) -- Floating-point multiply of two SFPU vector registers. Used for `in0 * reciprocal`. |
| `+` / `-` (vFloat add/sub) | **SFPADD** (or **SFPMAD**) -- Floating-point addition/subtraction. Used in Newton-Raphson iterations. |
| `sfpi::approx_recip(x)` | **SFPARECIP** -- Hardware approximate reciprocal instruction (Blackhole only). Provides initial ~7-bit accurate estimate. |
| `sfpi::setman(a, b)` | **SFPSETMAN** -- Replaces the mantissa of float `a` with mantissa bits from `b`. Used in Wormhole to normalize input range. |
| `sfpi::setsgn(a, b)` | **SFPSETSGN** -- Sets the sign bit of `a` to match `b`. Used for sign correction after reciprocal and for `x/0 = +/-inf`. |
| `sfpi::reinterpret<T>(v)` | **SFPMOV** (or no-op) -- Reinterprets the bit pattern of a vector register between float/int types. |
| `~` (bitwise NOT on vUInt) | **SFPNOT** -- Bitwise NOT. Used in Wormhole to compute `255 - exponent` for the scale factor. |
| `>>` (shift right on vUInt) | **SFPSHFT** -- Right shift. Used in `float32_to_bf16_rne` to extract the BF16 LSB. |
| `&` (bitwise AND on vUInt) | **SFPAND** -- Bitwise AND. Used to mask bits in BF16 rounding. |
| `v_if` / `v_elseif` / `v_endif` | **SFPSETCC** / **SFPENCC** / **SFPCOMPC** -- Condition code manipulation for per-lane predicated execution. Comparisons like `== 0`, `< 0` set the CC register, and subsequent operations only execute on lanes where the condition is true. |
| `sfpi::vConst1` | **LCONST_1** -- Hardware constant register holding `1.0f`. |
| `sfpi::vConst0` | **LCONST_0** -- Hardware constant register holding `0.0f`. |
| `sfpi::vConstNeg1` | **LCONST_neg1** -- Hardware constant register holding `-1.0f`. Used in Wormhole `setman`. |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers loaded during init. Wormhole: polynomial coefficients. Blackhole: `2.0f`. |
| `TTI_STALLWAIT` | Stalls the SFPU pipeline until the math unit signals completion, ensuring DEST data is ready. |
| `TTI_SETRWC` | Sets read/write counters for DEST addressing. Used between faces to advance the base address by 16 rows. |

### SFPU Register Usage

**DEST Register File**:
- `dst_index_in0 * 32` (= `i*2 * 32`): Input A tile rows. Read via `sfpi::dst_reg[...]`, overwritten with result.
- `dst_index_in1 * 32` (= `(i*2+1) * 32`): Input B tile rows. Read via `sfpi::dst_reg[...]`, consumed but not overwritten.
- `dst_index_out * 32` (= `i*2 * 32`): Output tile rows. Same location as input A (in-place overwrite).
- The `dst_reg++` increments the internal row pointer by `SFP_DESTREG_STRIDE` (2) after each of 8 iterations, covering all 16 rows per face. Between faces, `TTI_SETRWC` advances by 16 rows.

**SFPU Local Registers (LREGs)** -- used internally by SFPI compiler:
- LREGs are allocated by the SFPI compiler for temporary values (`in0`, `in1`, `result`, `t`, `y`, `scale`, `negative_x`, etc.).
- The SFPU has 8 local registers (LREG0-LREG7) available for computation. The reciprocal function uses several for intermediate Newton-Raphson values.

**Programmable Constant Registers**:
- **Wormhole**: `vConstFloatPrgm0` = 0.3232325 (k0), `vConstFloatPrgm1` = 1.4545459 (k1), `vConstFloatPrgm2` = 2.1212124 (k2) -- polynomial coefficients for initial reciprocal estimate.
- **Blackhole**: `vConstFloatPrgm0` = 2.0 -- constant for Newton-Raphson formula `t = x*y - 2.0`.

### Address Mode Configuration

The address mode for DIV (as for all binary SFPU operations using `SfpuType::unused`) is configured in `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`:

**ADDR_MOD_7** is set with all zero increments:
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

This is the same configuration for both **Wormhole B0** and **Blackhole**. The ADDR_MOD_7 slot is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2 which are used by the A2D (copy_tile) operations that precede the SFPU work.

The `dest.incr = 0` means the SFPU does not auto-increment the DEST base address between SFPU instruction executions -- instead, the SFPU kernel manages row advancement explicitly via `sfpi::dst_reg++` (which compiles to SFPINCRWC), and face advancement is handled by `TTI_SETRWC` in the params dispatch loop.

No `ADDR_MOD_6` is configured for DIV because the `SfpuType::unused` template parameter does not match any of the special types (mul_int32, max, min, etc.) that require `ADDR_MOD_6` with `dest.incr = 2`.

**Wormhole-specific detail**: `_llk_math_eltwise_binary_sfpu_start_` additionally calls `math::set_addr_mod_base()`, and `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()`. Blackhole does not include these calls.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary SFPU compute kernel work for operations like DIV? What defines are used and how does get_defines_fp32 generate them?"
   **Reason**: Needed to understand the compile-time define generation path from `BinaryOpType::DIV` to the actual SFPU function calls.
   **Key Findings**: DIV in `get_defines_fp32` generates `BINOP_INIT = div_binary_tile_init()` and `BINARY_SFPU_OP = div_binary_tile(i*2, i*2+1, i*2)`. The non-fp32 path uses reciprocal+multiply decomposition via `get_defines`.

2. **Query**: "What is the implementation of _sfpu_reciprocal_, _sfpu_binary_init_, _llk_math_eltwise_binary_sfpu_init_, and _llk_math_eltwise_binary_sfpu_params_? Where are they defined? What SFPU instructions do they use? Show the addr_mod configuration."
   **Reason**: Needed to trace the full call chain from API to core SFPU implementation and understand address mode setup.
   **Key Findings**: `_sfpu_binary_init_` calls `_init_sfpu_reciprocal_<false>()` for DIV. `_llk_math_eltwise_binary_sfpu_init_` configures ADDR_MOD_7 with zero increments. The params function iterates over 4 faces in RC mode, advancing DEST by 16 rows between faces.

3. **Query**: "What is _sfpu_reciprocal_ and _init_sfpu_reciprocal_? What SFPU instructions are used? How does Newton-Raphson work with different iteration counts?"
   **Reason**: Needed detailed understanding of the reciprocal algorithm that is the core of the DIV operation.
   **Key Findings**: Wormhole uses quadratic polynomial initial estimate + 2 NR iterations with software exponent manipulation. Blackhole uses hardware SFPARECIP instruction + 2 NR iterations. `max_iter=2` gives float32 precision (<=1 ulp).

### Confluence References
Not consulted for this analysis -- DeepWiki and source code provided sufficient detail on the SFPU instructions used.

### Glean References
Not consulted for this analysis.
