## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the RECIP (reciprocal) operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu.h` and `llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_recip.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `recip_tile_init()` which expands the macro `SFPU_THREE_TEMPLATE_PARAM_INIT(reciprocal, sfpu::recip_init, APPROX, DST_ACCUM_MODE, legacy_compat)`, invoking `llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROXIMATE>()` followed by `sfpu::recip_init<APPROX, DST_ACCUM_MODE, legacy_compat>()`.
2. `llk_math_eltwise_unary_sfpu_init` calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::reciprocal>()` which initializes the SFPU config register, configures address modes (ADDR_MOD_7 with zero increments, and on Blackhole ADDR_MOD_6 with dest incr=2), and resets counters.
3. `recip_init` delegates to `_init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, legacy_compat>()` which selects one of three init paths on Blackhole (fast_7b, fast_8b_3c, fast_24b_5c based on precision mode) or sets polynomial constants on Wormhole.
4. The compute kernel calls `recip_tile(idst)` which expands the macro `SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN(calculate_reciprocal, APPROX, DST_ACCUM_MODE, 8, legacy_compat, idst, vector_mode)`, invoking `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_reciprocal<APPROX, DST_ACCUM_MODE, 8, legacy_compat>, idst, vector_mode)`.
5. `_llk_math_eltwise_unary_sfpu_params_` sets the DST write address, stalls until SFPU is ready, then calls `calculate_reciprocal` once per face (4 times for VectorMode::RC), incrementing the face address between calls.
6. `calculate_reciprocal` calls `_calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS=8, is_fp32_dest_acc_en, legacy_compat>(8)`, which dispatches to the architecture-specific internal implementation.

### Annotated SFPU Kernel Source

**Blackhole Implementation** -- uses SFPLOADMACRO-based instruction scheduling with SFPARECIP hardware instruction:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x) // Used by other ops (sigmoid, div, log, etc.) not by the tile-level RECIP directly
{
    // approx_recip emits SFPARECIP instruction: hardware LUT-based reciprocal
    // Returns +/-0 for x = +/-inf or x >= +/-2^126, and +/-inf for x = +/-0
    sfpi::vFloat y = sfpi::approx_recip(x);

    if constexpr (max_iter > 0)
    {
        // Negated Newton-Raphson: t = x*y - 2.0 (vConstFloatPrgm0 = 2.0f)
        // Negation makes NaN detection easier via sign check
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (max_iter > 1)
        {
            sfpi::vFloat y1 = y * -t - sfpi::vConst0; // y1 = y * (2 - x*y)
            // If t=NaN then t>=0; this check hides in the SFPNOP slot of preceding SFPMAD
            v_if (t < 0) // condition code: only update if no NaN
            {
                t = x * y1 - sfpi::vConstFloatPrgm0;
                y = y1 * -t - sfpi::vConst0;
            }
            v_endif;
        }
        else
        {
            v_if (t < 0)
            {
                y = y * -t - sfpi::vConst0;
            }
            v_endif;
        }
    }

    return y;
}

// Approximate reciprocal, ~7b precision, throughput of 1 cycle per 32 elements.
// Uses SFPLOADMACRO to schedule: load -> SFPARECIP -> store.
inline void _calculate_reciprocal_fast_7b_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_6, 0); // macro 0: load, arecip, store — ADDR_MOD_6 has dest.incr=2
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// BF16 reciprocal, ~8b precision, throughput of 3 cycles per 32 elements.
// Implementation notes, see the original file for more details
inline void _calculate_reciprocal_fast_8b_3c_(const int iterations)
{
    constexpr int x           = p_sfpu::LREG1;
    constexpr int t           = p_sfpu::LREG1;
    constexpr int offset      = 0;
    constexpr int prev_offset = -4 & 0x3ff;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000); // L0 = 0x80000000 (sign bit mask)
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);      // L7 = x (indirect VD register index)

    // Prologue: first two iterations with SFPNOP for pipeline fill
    const int fill_end = iterations < 2 ? iterations : 2;
#pragma GCC unroll 2
    for (int d = 0; d < fill_end; d++)
    {
        int y = 3 + (d % 3);
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));  // macro 0: MOD0_FMT_SRCB
        TTI_SFPNOP;
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2)); // macro 1: MOD0_FMT_LO16_ONLY
    }

    // Main loop: all three SFPLOADMACROs active
#pragma GCC unroll 6
    for (int d = 2; d < iterations; d++)
    {
        int y = 3 + (d % 3);
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));      // macro 0: MOD0_FMT_SRCB
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_7, prev_offset | (t >> 2)); // macro 2: MOD0_FMT_LO16
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2));     // macro 1: MOD0_FMT_LO16_ONLY
    }

    // Fill gap with SFPNOPs when iterations < 2
#pragma GCC unroll 2
    for (int d = iterations; d < 2; d++)
    {
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    // Epilogue: drain final two iterations
    const int drain_start = iterations < 2 ? 2 : iterations;
#pragma GCC unroll 2
    for (int d = drain_start; d < iterations + 2; d++)
    {
        TTI_SFPNOP;
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_6, prev_offset | (t >> 2)); // macro 2: MOD0_FMT_LO16
        TTI_SFPNOP;
    }

    TTI_SFPNOP;
}

// FP32 reciprocal, ~24b precision, throughput of 5 cycles per 32 elements.
// Implementation notes, see the original file for more details
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations)
{
    lltt::replay(0, 4);                            // replay instructions 0-3 from replay buffer
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);             // load from DEST into LREG7

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++)
    {
        lltt::replay(0, 5);                        // replay instructions 0-4 from replay buffer
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);                            // replay instruction 1
    TTI_SFPNOP;
    lltt::replay(3, 2);                            // replay instructions 3-4

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations) // Dispatches based on precision mode
{
    if constexpr (APPROXIMATION_MODE)
    {
        _calculate_reciprocal_fast_7b_(iterations);   // ~7-bit, 1 cycle/32 elems
    }
    else if constexpr (is_fp32_dest_acc_en)
    {
        _calculate_reciprocal_fast_24b_5c_(iterations); // ~24-bit, 5 cycles/32 elems
    }
    else
    {
        _calculate_reciprocal_fast_8b_3c_(iterations);  // ~8-bit BF16, 3 cycles/32 elems
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations)
{
    if constexpr (legacy_compat)
    {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
    else
    {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

// ~7b precision init: configures SFPLOADMACRO scheduling for arecip+store
inline void _init_reciprocal_fast_7b_()
{
    // Implementation notes, see the original file for more details
    TTI_SFPARECIP(0, 0, 12, 0); // InstructionTemplate[0]: approximate reciprocal

    constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0); // L16=1, arecip template 0
    constexpr std::uint32_t mad_bits    = 0;
    constexpr std::uint32_t round_bits  = 0;
    constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;       // L16=1, store offset=1

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

    TTI_SFPCONFIG(0, 4, 0); // Write macro 0 config

    // Misc: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1 for macro 0
    TTI_SFPCONFIG(0x110, 8, 1);
}

// ~8b BF16 precision init: configures multi-step macro pipeline
inline void _init_reciprocal_fast_8b_3c_()
{
    constexpr int x = p_sfpu::LREG1;
    constexpr int t = p_sfpu::LREG1;

    TTI_SFPARECIP(0, 0, 12, 0);                                              // InstructionTemplate[0]
    TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, 0, 13, 8);              // InstructionTemplate[1]: x = 0*0 + y, SFPMAD_MOD1_INDIRECT_VD
    TTI_SFPMAD(x, 0, p_sfpu::LCONST_neg1, 14, 0);                          // InstructionTemplate[2]: y = x * y - 1
    TTI_SFPIADD(0, t, 15, sfpi::SFPIADD_MOD1_CC_NONE);                     // InstructionTemplate[3]: integer add for LSB correction

    // Macro 0 config
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x80 | 0x00 | (0 << 3) | 3;     // StoreMod0=MOD0_FMT_SRCB, no UsesLoadMod0ForStore
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1 config
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (5 << 3) | (4 + 3); // IADD template 3
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (0 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2 config
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: StoreMod0=MOD0_FMT_SRCB, WaitForElapsedInstructions=1 for all macros
    TTI_SFPCONFIG(0x700, 8, 1);
}

// ~24b FP32 precision init: configures 5-step macro pipeline with replay buffer
inline void _init_reciprocal_fast_24b_5c_()
{
    constexpr int e  = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z  = p_sfpu::LREG2;
    constexpr int y  = p_sfpu::LREG3;

    TTI_SFPARECIP(0, 0, 12, 0);                                              // InstructionTemplate[0]: arecip
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);                    // InstructionTemplate[1]: e = -e*y + 1.0
    TTI_SFPMAD(t2, p_sfpu::LREG0, z, 14, 0);                               // InstructionTemplate[2]: t2 = t2*e + e or z
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // InstructionTemplate[3]: t2 = min(t2, 1.0)

    // Macro 0: [y] — load, arecip, store
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (6 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0);
    }
    // Macro 1: [e] — arecip + MAD e = -e*y + 1.0
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2: [t2] — t2 = t2*e + e
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (2 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (0 << 3) | (4 + 2);
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }
    // Macro 3: [z] — z = t2*y + y
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (1 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1 for all macros
    TTI_SFPCONFIG(0xff0, 8, 1);

    constexpr std::uint32_t prev_offset = -2 & 0x3ff;
    constexpr std::uint32_t offset      = 0;

    // Load replay buffer with the 6-instruction sequence
    load_replay_buf(
        0,
        6,
        [e, t2, z, y, offset, prev_offset]
        {
            TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
            TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, ADDR_MOD_7, prev_offset | (t2 >> 2));
            TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, ADDR_MOD_7, offset | (e >> 2));
            TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1); // SFPMAD_MOD1_NEGATE_VA
            TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_6, prev_offset | (z >> 2));
            TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_7, prev_offset | (z >> 2));
        });
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_() // Called only for the _sfpu_reciprocal_ (non-LOADMACRO) path
{
    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 2.0f; // Used as Newton-Raphson constant: t = x*y - 2.0
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_()
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            _init_reciprocal_fast_7b_();     // Configure SFPLOADMACRO for ~7b path
        }
        else if constexpr (is_fp32_dest_acc_en)
        {
            _init_reciprocal_fast_24b_5c_(); // Configure SFPLOADMACRO + replay buffer for ~24b path
        }
        else
        {
            _init_reciprocal_fast_8b_3c_();  // Configure SFPLOADMACRO for ~8b BF16 path
        }
    }
}

} // namespace sfpu
} // namespace ckernel
```

**Wormhole Implementation** -- uses software Newton-Raphson with SFPI intrinsics (no SFPARECIP or SFPLOADMACRO):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
// max_iter = 2: sufficient for float32 precision (<=1 ulps).
// max_iter = 1: sufficient for bfloat16/float16 precision (<=0.5 ulps).
// max_iter = 0: same effect as max_iter=1 currently.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) // max_iter=0 for APPROX, 1 for BF16, 2 for FP32
{
    // Combine sign and exponent of -1.0 with the mantissa of `in`.
    // Scales input to [-2.0, -1.0) range. If in=+/-0 or in=+/-inf, then x=+/-1.
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    // Coefficients minimize max relative error for 1/x over [1,2) via Sollya
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Compute scale factor: scale.Exp = 255-in.Exp = ~in.Exp via SFPNOT
    // This handles: in.Exp==0 -> +/-inf, in.Exp==255 -> +/-0
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Clear mantissa from scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // Newton-Raphson iteration 1: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // scale *= 0.5 to correct for the 255 vs 254 exponent offset
    scale *= 0.5f;

    // y = y + y*t (Newton-Raphson refinement)
    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Newton-Raphson iteration 2
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling factor and restore original sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            sfpi::dst_reg[0] = _sfpu_reciprocal_<0>(in);      // 0 NR iterations (same as 1 currently)
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                sfpi::dst_reg[0] = _sfpu_reciprocal_<2>(in);  // 2 NR iterations for FP32
            }
            else
            {
                sfpi::vFloat out = _sfpu_reciprocal_<1>(in);   // 1 NR iteration for BF16
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0)); // truncate to BF16
            }
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations)
{
    if constexpr (legacy_compat)
    {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
    else
    {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    // Polynomial coefficients for quadratic initial estimate y = k2 - k1*x + k0*x^2
    // Minimise max relative error for 1/x over [1,2) via Sollya
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_()
{
    if constexpr (!legacy_compat)
    {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>(); // Always sets polynomial constants on Wormhole
    }
}

} // namespace sfpu
} // namespace ckernel
```

### SFPU Instructions Used

**Blackhole (SFPLOADMACRO-based paths):**

| Instruction | Description |
|---|---|
| `SFPARECIP` | Hardware approximate reciprocal instruction. Uses a 128-entry LUT indexed by mantissa bits [16:22] to produce an initial ~7-bit reciprocal estimate. Exponent is computed as `253 - input_exponent`. Returns +/-inf for zero inputs and +/-0 for infinity inputs. |
| `SFPLOADMACRO` | Composite instruction that schedules load, simple (SFPARECIP), MAD, and store sub-operations across pipeline stages. Each SFPLOADMACRO invocation selects a macro configuration (0-3) and a destination register (VD). The macro config determines which instruction templates and store modes are used. |
| `SFPMAD` | Fused multiply-add: `result = VA * VB + VC`. Used in the 8b and 24b paths for Newton-Raphson refinement (e.g., `y = x * y - 1`). Mod1 variants include `SFPMAD_MOD1_INDIRECT_VD` (indirect destination via L7) and `SFPMAD_MOD1_NEGATE_VA` (negate first operand). |
| `SFPIADD` | Integer add on SFPU registers. Used in the 8b path for LSB correction of the BF16 result (`y += t` where t is a shifted version of y). |
| `SFPSWAP` | Swap/min/max operation. Used in the 24b path with `SFPSWAP_MOD1_VEC_MIN_MAX` to clamp the error term: `t2 = min(t2, 1.0)`, replacing NaN with 1.0. |
| `SFPLOADI` | Load immediate value into an SFPU register. Used to load constants (e.g., 0x8000 for sign bit mask) and to configure macro scheduling bits (simple_bits, mad_bits, store_bits). |
| `SFPCONFIG` | Configure SFPU state: write macro definitions (instruction template indices, store modes) and misc config (UsesLoadMod0ForStore, WaitForElapsedInstructions). |
| `SFPLOAD` | Load data from DEST register into an SFPU local register. Used in the 24b path to load initial values. |
| `SFPNOP` | No-operation, used for pipeline filling/draining and to satisfy instruction latency requirements. |
| `lltt::replay` / `load_replay_buf` | Replay buffer mechanism (Blackhole-specific): stores a sequence of instructions that can be replayed multiple times, reducing instruction fetch overhead in the 24b path. |

**Wormhole (SFPI-based path):**

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::setman(dst, src)` | Set the mantissa field of `dst` to the mantissa of `src`, preserving sign and exponent. Used to normalize the input to [-2, -1) range and to clear mantissa from the scale factor. |
| `sfpi::reinterpret<T>(v)` | Reinterpret the bit pattern of a value as a different type (e.g., vFloat to vInt). No bits change, only the type interpretation. |
| `sfpi::setsgn(val, sign_src)` | Set the sign bit of `val` to match the sign of `sign_src`. Used at the end to restore the original input sign. |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers. Loaded with polynomial coefficients k0, k1, k2 during init. |
| `sfpi::vConstNeg1` | Hardware constant -1.0f. Used as the base for `setman` to create the normalized negative input. |
| `sfpi::vConst1` | Hardware constant 1.0f. Used in Newton-Raphson: `t = 1.0 + negative_x * y`. |
| `sfpi::dst_reg[0]` / `sfpi::dst_reg++` | Read/write DEST register at current offset, then advance to next row (2 elements). |
| `~` (bitwise NOT via SFPNOT) | Inverts all bits to compute `255 - exponent` efficiently for the scale factor. |
| `float_to_fp16b(val, mode)` | Convert FP32 to BF16 format (truncate lower 16 mantissa bits). Used in non-FP32 non-approximate mode on Wormhole. |
| `v_if` / `v_endif` | Conditional execution based on SFPU condition codes. Used in the legacy compat path for sign handling. |

### SFPU Register Usage

**Blackhole (SFPLOADMACRO paths):**

| Register | Usage |
|---|---|
| `LREG0` (L0) | 7b path: not explicitly used. 8b path: holds 0x80000000 (sign bit mask for BF16 LSB correction). 24b path: holds error term `e`. |
| `LREG1` (L1) | 8b path: holds `x` (copy of input) and `t` (shifted result for LSB correction). 24b path: holds `t2` (accumulated error polynomial). |
| `LREG2` (L2) | 24b path: holds `z` (final scaled result). |
| `LREG3-LREG5` (L3-L5) | 8b path: rotating destination registers `y` = `3 + (d % 3)` cycles through L3, L4, L5 to hide pipeline latency. 24b path: L3 = `y` (arecip result). |
| `LREG7` (L7) | 8b path: holds the indirect VD index (value of `x` = LREG1 = 1) for `SFPMAD_MOD1_INDIRECT_VD`. |
| `DEST registers` | Source and destination for tile data. Each iteration reads from and writes back to DEST at the current offset, advancing by the address mode increment. |

**Wormhole (SFPI path):**

| Register | Usage |
|---|---|
| `vConstFloatPrgm0` | Polynomial coefficient k0 = 0.3232325... |
| `vConstFloatPrgm1` | Polynomial coefficient k1 = 1.4545459... |
| `vConstFloatPrgm2` | Polynomial coefficient k2 = 2.121212... |
| `vConstNeg1` | Hardware constant -1.0f, used as the base for input normalization. |
| `vConst1` | Hardware constant 1.0f, used in Newton-Raphson iterations. |
| `dst_reg[0]` | Current DEST register row; read for input, written with output. Incremented via `dst_reg++` (which advances by the configured step). |
| Local vFloat variables (`negative_x`, `y`, `t`, `scale`) | Mapped to SFPU local registers (LREG0-LREG3) by the compiler. |

### Address Mode Configuration

**Blackhole:**

The RECIP operation on Blackhole configures two address modes during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::reciprocal>()`:

- **ADDR_MOD_7**: `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- No auto-increment. Used by SFPLOADMACRO calls that should not advance the DEST pointer (e.g., intermediate loads within a multi-step pipeline).
- **ADDR_MOD_6**: `{srca.incr=0, srcb.incr=0, dest.incr=2}` -- DEST increments by 2 rows per SFPLOADMACRO invocation. Used by the final SFPLOADMACRO in each pipeline step to advance to the next pair of rows in the tile face. The `dest.incr=2` value corresponds to advancing through 2 rows of 32 elements (64 elements) in the DEST register.

Note: `SfpuType::reciprocal` is explicitly listed in the Blackhole `eltwise_unary_sfpu_configure_addrmod` function as requiring ADDR_MOD_6 with `dest.incr=2`, alongside typecast and min/max operations.

**Wormhole:**

On Wormhole, the RECIP operation configures only one address mode:

- **ADDR_MOD_7**: `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- No auto-increment.
- **ADDR_MOD_6**: NOT configured for reciprocal on Wormhole (the `SfpuType::reciprocal` is absent from the Wormhole `if constexpr` condition for ADDR_MOD_6). This is because Wormhole uses the SFPI `dst_reg++` operator for DEST advancement rather than hardware address mode increments.

The key difference is that Blackhole's SFPLOADMACRO-based paths rely on hardware address mode increments (ADDR_MOD_6 with dest.incr=2) to advance through tile rows, while Wormhole's SFPI-based paths use explicit `dst_reg++` calls in the iteration loop.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the RECIP (reciprocal) SFPU kernel work? Trace the call chain from compute_kernel_api through LLK to the ckernel SFPU implementation."
   **Reason**: Needed to identify file paths and understand the overall call chain structure before reading source files.
   **Key Findings**: Identified the three-layer architecture (compute API -> LLK macros -> ckernel SFPU), confirmed file paths for both Wormhole and Blackhole implementations, and learned about the template parameter system (APPROXIMATION_MODE, is_fp32_dest_acc_en, legacy_compat).

2. **Query**: "How is the reciprocal (recip) SFPU operation implemented in the LLK layer? What is the call chain from llk_math_eltwise_unary_sfpu_recip down to _calculate_reciprocal_? What SFPU instructions and registers does it use?"
   **Reason**: Needed detailed LLK-level implementation details including SFPU instructions used in each precision path.
   **Key Findings**: Confirmed the three precision paths (7b approximate, 8b BF16, 24b FP32) on Blackhole, identified key instructions (SFPARECIP, SFPLOADMACRO, SFPMAD, SFPIADD, SFPSWAP), and learned about the Newton-Raphson approach on Wormhole with quadratic initial estimate.

3. **Query**: "What does approx_recip do in SFPI?" (to tenstorrent/sfpi)
   **Reason**: Needed to understand what hardware instruction the `sfpi::approx_recip` intrinsic maps to and how it works.
   **Key Findings**: `approx_recip` emits the `__builtin_rvtt_sfparecip` intrinsic, which is Blackhole-specific. Wormhole does not have this instruction and uses software-based polynomial + Newton-Raphson instead.

4. **Query**: "What is the SFPARECIP instruction?" (to tenstorrent/tt-isa-documentation)
   **Reason**: Needed ISA-level details on how the approximate reciprocal hardware instruction works.
   **Key Findings**: SFPARECIP uses a 128-entry LUT indexed by mantissa bits [16:22], computes exponent as `253 - input_exponent`, handles special cases (zero -> inf, inf -> 0, denormals). Supports reciprocal, conditional reciprocal, and exponential modes via Mod1 field.

### Confluence References
Not consulted for this analysis. DeepWiki and source code provided sufficient detail on all SFPU instructions used.

### Glean References
Not consulted for this analysis. The open-source ISA documentation via DeepWiki was sufficient for understanding SFPARECIP and related instructions.
