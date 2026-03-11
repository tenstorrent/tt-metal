## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the GELU operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (via macro in `llk_math_eltwise_unary_sfpu_macros.h`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_gelu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `gelu_tile<fast_and_approx>(idst)` defined in `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h`.
2. `gelu_tile` expands the macro `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, fast_and_approx, idst)`, which resolves to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_gelu<APPROXIMATE>, idst, (int)VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) calls `_llk_math_eltwise_unary_sfpu_start_` to set up the DEST write address and stall until SFPU is ready, then iterates over tile faces (4 faces for RC mode), calling `ckernel::sfpu::calculate_gelu<APPROXIMATE>()` for each face.
4. `calculate_gelu` (in the metal overlay `ckernel_sfpu_gelu.h`) delegates to `_calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>()` when `APPROXIMATION_MODE=true`, or to a Chebyshev polynomial loop when `APPROXIMATION_MODE=false`.
5. `_calculate_gelu_` (in the tt_llk `ckernel_sfpu_gelu.h`) dispatches to `_calculate_gelu_appx_<ITERATIONS>()` or `_calculate_gelu_accurate_<ITERATIONS>()`.

For initialization: `gelu_tile_init<fast_and_approx>()` expands `SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, fast_and_approx)`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>(gelu_init<APPROXIMATE>)`. This first calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::gelu>()` (which initializes the SFPU config register, configures ADDR_MOD_7, and resets counters), then calls `gelu_init<APPROXIMATE>()` which delegates to `_init_gelu_<APPROXIMATE>()` to load LUT coefficients.

### Annotated SFPU Kernel Source

The GELU kernel has two separate implementations depending on `APPROXIMATION_MODE`:
- **Approximate mode** (`APPROXIMATION_MODE=true`): Uses a 6-entry piecewise linear LUT via `lut2_sign` for fast approximation.
- **Accurate mode** (`APPROXIMATION_MODE=false`): Uses a CDF polynomial approximation (5th-degree Chebyshev) with conditional branching.

Additionally, the metal overlay `ckernel_sfpu_gelu.h` provides an alternate accurate path using a 15th-degree Chebyshev polynomial, which is active when `APPROXIMATION_MODE=false` and the metal overlay's `calculate_gelu` is used directly. The tt_llk `_calculate_gelu_` accurate path uses CDF polynomial approximation instead.

#### Core SFPU Implementation (tt_llk)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h
// (Wormhole B0 implementation is identical)

template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_gelu_core_(sfpi::vFloat in) // used by gelu_derivative, not directly by gelu
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        result = in;
    }
    else
    {
        // f = (0.044715*x^3 + x)
        result = (in * in) * (in * sfpi::s2vFloat16b(0.044715f)) + in;
        result *= sfpi::s2vFloat16b(0.79788f);
    }

    return result;
}

template <int ITERATIONS>
inline void _calculate_gelu_appx_() // ITERATIONS=8 (default, processes 8 rows per face)
{
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0]; // LUT slope coefficients (low range)
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1]; // LUT slope coefficients (mid range)
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2]; // LUT slope coefficients (high range)
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4]; // LUT offset coefficients (low range)
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5]; // LUT offset coefficients (mid range)
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6]; // LUT offset coefficients (high range)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in      = sfpi::dst_reg[0];       // SFPLOAD from DEST register
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;  // 0.5f, loaded during init
        sfpi::vFloat half_in = in * half;               // SFPMUL: x/2
        sfpi::vFloat result  = lut2_sign(in, l0, l1, l2, l4, l5, l6); // SFPLUTFP32 with 6-entry FP16 table and sign update
        result               = half_in + result;        // SFPADD: x/2 + lut_result

        sfpi::dst_reg[0] = result;                      // SFPSTORE to DEST register

        sfpi::dst_reg++;                                // INCRWC: advance DEST pointer by 4 (SFP_DESTREG_STRIDE)
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0; // Restore local registers (compiler hint to preserve values)
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <int ITERATIONS>
inline void _calculate_gelu_accurate_() // ITERATIONS=8
{
    constexpr bool scaled = true;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in     = sfpi::dst_reg[0];             // SFPLOAD from DEST register
        sfpi::vFloat result = _calculate_cdf_appx_(in, scaled); // CDF approximation, then multiply by input
        sfpi::dst_reg[0]    = result;                        // SFPSTORE to DEST register
        sfpi::dst_reg++;                                     // INCRWC
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_gelu_()
{
    if constexpr (APPROXIMATION_MODE)
    {
        _calculate_gelu_appx_<ITERATIONS>();
    }
    else
    {
        _calculate_gelu_accurate_<ITERATIONS>();
    }
}
```

#### Init Function (tt_llk)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h

template <bool APPROXIMATION_MODE>
inline void _init_gelu_()
{
    sfpi::vConstFloatPrgm0 = 0.5f; // SFPCONFIG: program constant register 0 = 0.5

    // Load 6-entry piecewise linear LUT coefficients into local registers.
    // Each 32-bit immediate packs two FP16 values: [hi_16:lo_16]
    // The LUT approximates erf(x/sqrt(2)) in 6 ranges:
    //   [0.0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0), [2.0, 3.0), [3.0, inf)
    // LReg0-2 hold slope (A) coefficients, LReg4-6 hold offset (B) coefficients.
    // For range i: result = A_i * |x| + B_i

    // LReg0: [hi=A(0.5..1.0)=0.4939, lo=A(0.0..0.5)=0.1928]
    _sfpu_load_imm32_(0, 0x37E7322B); // SFPLOADI x2 into LReg0
    // LReg4: [hi=B(0.5..1.0)=-0.1605, lo=B(0.0..0.5)=-0.0150]
    _sfpu_load_imm32_(4, 0xB12286D8); // SFPLOADI x2 into LReg4

    // LReg1: [hi=A(1.5..2.0)=0.6099, lo=A(1.0..1.5)=0.6189]
    _sfpu_load_imm32_(1, 0x38E138F3); // SFPLOADI x2 into LReg1
    // LReg5: [hi=B(1.5..2.0)=-0.2635, lo=B(1.0..1.5)=-0.2797]
    _sfpu_load_imm32_(5, 0xB437B479); // SFPLOADI x2 into LReg5

    // LReg2: [hi=A(>=3.0)=0.50, lo=A(2.0..3.0)=0.5402]
    _sfpu_load_imm32_(2, 0x38003852); // SFPLOADI x2 into LReg2
    // LReg6: [hi=B(>=3.0)=inf(saturate to 0.5), lo=B(2.0..3.0)=-0.1194]
    _sfpu_load_imm32_(6, 0x7c00afa4); // SFPLOADI x2 into LReg6
}
```

#### CDF Approximation (used by accurate mode)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_cdf.h
// (Wormhole B0 implementation is identical)

inline sfpi::vFloat _calculate_pos_cdf_appx_(sfpi::vFloat val)
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat result;
    v_if (val < 2.5f)
    {
        result = POLYVAL5<sfpi::vFloat>(0.0122792f, -0.05281024f, -0.03048313f, 0.41314081f, 0.49866379f, val);
    }
    v_else
    {
        result = 0.44656975f * val + 0.58216001f;
    }
    v_endif;

    v_if (result > 1.0f)
    {
        result = 1.0f;
    }
    v_endif;
    return result;
}

inline sfpi::vFloat _calculate_cdf_appx_(sfpi::vFloat val, bool scaled = false)
{
    sfpi::vFloat result = 0.0f;

    v_if (val < 0.0f)                                  // SFPSETCC: set condition on val < 0
    {
        result = 1.0f - _calculate_pos_cdf_appx_(-val); // CDF(-x) = 1 - CDF(x)
    }
    v_else                                             // SFPCOMPC: complement condition
    {
        result = _calculate_pos_cdf_appx_(val);
    }
    v_endif;                                           // SFPPOPC: restore condition code

    if (scaled)
    {
        result *= val; // GELU: CDF(x) * x
    }
    return result;
}
```

#### Metal Overlay (alternate accurate path via Chebyshev)

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
// (Wormhole B0 version is identical)

inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {                               // SFPSETCC on val >= -5.5
        result = POLYVAL15(
            -1.81205228163e-09,
            -4.59055119276e-08,
            -3.74540617693e-07,
            -2.29754133825e-07,
            1.19076782913e-05,
            4.25116466215e-05,
            -0.000138391838381,
            -0.000862052441087,
            0.000768340223025,
            0.0092074331601,
            -0.00208478037614,
            -0.0656369476513,
            0.00244542739174,
            0.398579460781,
            0.499174645395,
            2.98325768482e-05,
            val);                                      // 15th-degree Horner evaluation via SFPMUL/SFPADD chain

        result = setsgn(result, val);                  // SFPSETSGN: copy sign bit of val into result
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>(); // delegates to LUT-based appx
    } else {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = in;
        v_if(in == 0.0f) { result = 0.0f; }           // SFPSETCC: zero passthrough
        v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); } // 15th-order Chebyshev for |x| < 3
        v_endif;                                       // for |x| >= 3, result = x (identity, since GELU(x) ~ x)
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    }
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (`dst_reg[0]` read) | Loads a vector of values from the DEST register file into an SFPU local register (LReg3 by default). |
| `SFPSTORE` (`dst_reg[0]` write) | Stores a vector from an SFPU local register back to the DEST register file. |
| `SFPLOADI` (via `_sfpu_load_imm32_`, `_sfpu_load_imm16_`) | Loads an immediate value into an SFPU local register. `insmod=10` writes lower 16 bits, `insmod=8` writes upper 16 bits, `insmod=2` writes unsigned 16-bit right-justified. |
| `SFPLUTFP32` (via `lut2_sign`) | Performs a 6-entry FP16 piecewise linear lookup table evaluation: `result = A_i * |x| + B_i` for the range containing `|x|`. The `SGN_UPDATE` mode copies the sign of the input to the result. This is the core of the approximate GELU path. |
| `SFPMUL` (via `*` operator on `vFloat`) | Vector multiply. Used for `in * half`, polynomial coefficient multiplication, and CDF scaling (`result *= val`). |
| `SFPADD` (via `+` operator on `vFloat`) | Vector add. Used for `half_in + result`, Horner polynomial accumulation steps, and CDF combination. |
| `SFPMAD` (via fused multiply-add in Horner evaluation) | Fused multiply-add `a*b+c`. The compiler may fuse adjacent multiply-add patterns in the polynomial evaluations. |
| `SFPSETCC` (via `v_if` conditions) | Sets the per-lane condition code based on a comparison (e.g., `val < 0.0f`, `val >= -5.5f`, `in == 0.0f`). Only lanes satisfying the condition execute subsequent instructions. |
| `SFPPUSHC` (via `v_if`) | Pushes the current condition code onto the condition code stack before setting a new condition. |
| `SFPCOMPC` (via `v_else`, `v_elseif`) | Complements the current condition code, toggling which lanes are active. |
| `SFPPOPC` (via `v_endif`) | Pops the condition code stack, restoring the previous condition state. |
| `SFPSETSGN` (via `setsgn(result, val)`) | Copies the sign bit from `val` into `result`. Used in the Chebyshev accurate path to ensure correct sign. |
| `SFPCONFIG` (via `vConstFloatPrgm0 = 0.5f`) | Programs the constant register PrgmConst0 with 0.5f, making it available as `vConstFloatPrgm0`. |
| `SFPCONFIG` (via `_init_sfpu_config_reg`) | Initializes the SFPU configuration register (dest register format, etc.) during `_llk_math_eltwise_unary_sfpu_init_`. |
| `INCRWC` (via `dst_reg++`) | Increments the DEST register write counter by `SFP_DESTREG_STRIDE` (4 rows), advancing to the next row group within a face. |
| `SFPNOP` / `STALLWAIT` | Implicit pipeline synchronization. `TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)` ensures the SFPU is ready before starting computation. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Input/output for the GELU operation. Each tile face has 8 row-groups of 4 rows. The SFPU reads from `dst_reg[0]` (current row group), computes, writes back to `dst_reg[0]`, then increments via `dst_reg++`. |
| **LReg0** | Approximate path: packed FP16 LUT slopes for ranges [0.0, 0.5) and [0.5, 1.0). Value: `0x37E7322B` (hi=0.4939, lo=0.1928). |
| **LReg1** | Approximate path: packed FP16 LUT slopes for ranges [1.0, 1.5) and [1.5, 2.0). Value: `0x38E138F3` (hi=0.6099, lo=0.6189). |
| **LReg2** | Approximate path: packed FP16 LUT slopes for ranges [2.0, 3.0) and [3.0, inf). Value: `0x38003852` (hi=0.50, lo=0.5402). |
| **LReg3** | Implicitly used by SFPLOAD/SFPSTORE as the default data transfer register between DEST and SFPU. Also used as temporary for `vFloat` variables allocated by the compiler. |
| **LReg4** | Approximate path: packed FP16 LUT offsets for ranges [0.0, 0.5) and [0.5, 1.0). Value: `0xB12286D8` (hi=-0.1605, lo=-0.0150). |
| **LReg5** | Approximate path: packed FP16 LUT offsets for ranges [1.0, 1.5) and [1.5, 2.0). Value: `0xB437B479` (hi=-0.2635, lo=-0.2797). |
| **LReg6** | Approximate path: packed FP16 LUT offsets for ranges [2.0, 3.0) and [3.0, inf). Value: `0x7c00afa4` (hi=inf/saturate, lo=-0.1194). |
| **LReg7** | Compiler-allocated temporary register for intermediate `vFloat`/`vUInt` values during computation. |
| **vConstFloatPrgm0** | Programmable constant register, set to `0.5f` during `_init_gelu_`. Used as the `half` multiplier in the approximate path. |
| **Condition Code Stack** | Used by `v_if`/`v_elseif`/`v_else`/`v_endif` for per-lane predication. The accurate paths use nested conditionals (sign check, range check, clamp check). |

### Address Mode Configuration

The GELU operation uses `ADDR_MOD_7` configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::gelu>()`. The address mode is set identically for both Wormhole B0 and Blackhole:

```
ADDR_MOD_7:
  .srca = { .incr = 0 }   // No SRC_A increment (not used by SFPU)
  .srcb = { .incr = 0 }   // No SRC_B increment (not used by SFPU)
  .dest = { .incr = 0 }   // No automatic DEST increment (manual increment via dst_reg++)
```

GELU does not fall into any of the special-case operations (reciprocal, typecast, unary_max/min) that would configure `ADDR_MOD_6` with `.dest = {.incr = 2}`. The DEST pointer advancement is handled explicitly by `dst_reg++` (which emits `INCRWC`) within the SFPU kernel loop, incrementing by `SFP_DESTREG_STRIDE` (4 rows) per iteration.

**Wormhole B0 difference**: The Wormhole B0 variant of `_llk_math_eltwise_unary_sfpu_init_` additionally calls `math::set_addr_mod_base()` before the operation and `math::clear_addr_mod_base()` after (with a `STALLWAIT` for SFPU completion). Blackhole does not have these calls. The actual `ADDR_MOD_7` field values are identical between the two architectures.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the GELU SFPU kernel implemented? Trace from the compute kernel API (gelu_tile) through LLK dispatch to the ckernel SFPU implementation."
   **Reason**: To identify all files in the GELU call chain and understand the two-path structure (approximate vs accurate).
   **Key Findings**: Confirmed the API header at `compute/eltwise_unary/gelu.h`, the macro-based dispatch through `SFPU_UNARY_NO_PARAM_KERNEL_FN`, and the two implementation paths in `ckernel_sfpu_gelu.h`. The metal overlay adds a Chebyshev 15th-order polynomial path for accurate mode.

2. **Query**: "How is the GELU SFPU kernel implemented in tt-llk? What is the call chain from llk_math to ckernel_sfpu_gelu? What SFPU instructions and registers does it use?"
   **Reason**: To understand the tt_llk-level implementation details, especially the LUT-based approximate path and register usage.
   **Key Findings**: The approximate path uses `lut2_sign` with 6 LRegs (L0-L2, L4-L6) for a 6-entry piecewise linear table. The accurate path calls `_calculate_cdf_appx_` for CDF-based computation. The `_llk_math_eltwise_unary_sfpu_params_` function handles face iteration.

3. **Query**: "How are SFPI instructions like lreg_dest, vFloat, vConst, dst_reg used in SFPU kernels?"
   **Reason**: To understand the SFPI programming model for register access and conditional execution.
   **Key Findings**: `dst_reg` provides indexed access to the 512-entry DEST register file. `l_reg[LRegs::LRegN]` provides access to 8 local registers. `vConstFloatPrgm0` is a programmable constant. `v_if`/`v_endif` map to `SFPPUSHC`/`SFPSETCC`/`SFPPOPC` for per-lane conditional execution.

4. **Query**: "What is the lut2_sign function in SFPI? How does it work? What SFPU instruction does it map to?"
   **Reason**: To understand the core instruction in the approximate GELU path.
   **Key Findings**: `lut2_sign` maps to `SFPLUTFP32` with the `SGN_UPDATE` flag. It performs a 6-entry piecewise linear lookup: selects `A_i, B_i` coefficients based on `|x|` range, computes `A_i * |x| + B_i`, then copies the sign of the input to the result. Uses `SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1` mode with split points at 0.5, 1.0, 1.5, 2.0, and 3.0.

5. **Query**: "What is the SFPLUTFP32 instruction? How does it work? What are the LUT modes?"
   **Reason**: To get ISA-level detail on the LUT instruction.
   **Key Findings**: `SFPLUTFP32` supports 4 modes: FP32 3-entry, FP16 3-entry, FP16 6-entry table 1 (splits at 0.5/1.0/1.5/2.0/3.0), and FP16 6-entry table 2 (splits at 0.5/1.0/1.5/2.0/4.0). Coefficients are packed as FP16 pairs in LRegs 0-2 (slopes) and LRegs 4-6 (offsets).

### Confluence References

Not consulted for this analysis. The DeepWiki queries provided sufficient detail on SFPU instructions used in the GELU kernel.

### Glean References

Not consulted for this analysis. The implementation details were fully available through DeepWiki and source code inspection.
