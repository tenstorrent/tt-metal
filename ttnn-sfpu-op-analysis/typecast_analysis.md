## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. Typecast is a unique SFPU operation because it does not have a single SFPU kernel -- instead, it dispatches to one of 20+ conversion-specific SFPU kernels depending on the input/output data format pair. Some format pairs bypass the SFPU entirely (handled by unpacker/packer). Many conversion paths leverage the Blackhole `SFPLOADMACRO` instruction for high-throughput pipelined execution.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_typecast.h` (underscore-prefixed core functions) and `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` (wrapper + inline implementations for fp32_to_uint8, uint_to_uint8) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel (`eltwise_typecast.cpp`) calls `TYPECAST_LLK(0)`, which is macro-defined at compile time to `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)` with the numeric data format IDs baked in.
2. `typecast_tile<IN_DTYPE, OUT_DTYPE>(idst)` (in `typecast.h`) calls `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)` via the `MATH()` macro, which restricts execution to the math RISC-V core.
3. `llk_math_eltwise_unary_sfpu_typecast` (in `llk_math_eltwise_unary_sfpu_typecast.h`) is a large `if constexpr` dispatch that selects the appropriate conversion-specific SFPU function (e.g., `calculate_typecast_fp32_to_uint16<false, 8>`) and invokes it via `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until the SFPU is available, then calls the SFPU function once per face (4 times for RC vector mode, which processes all four 16x16 faces of a 32x32 tile).
5. The SFPU function itself (e.g., `_calculate_typecast_fp32_to_uint16_<false, 8>()` in `ckernel_sfpu_typecast.h`) executes SFPU instructions that operate on 8 rows of a 16x16 face per invocation (ITERATIONS=8).

Similarly, `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` which calls `llk_math_eltwise_unary_sfpu_typecast_init`, which in turn calls `llk_math_eltwise_unary_sfpu_init<SfpuType::typecast, APPROXIMATE>` with a format-specific init function. The init configures ADDR_MOD registers and, for SFPLOADMACRO-based conversions, programs the instruction templates and macros via `SFPCONFIG`.

### Annotated SFPU Kernel Source

The typecast operation has many conversion-specific kernels. Below is the complete Blackhole core implementation file, which contains all conversion kernels and their init functions. The Wormhole version is structurally identical but differs in ADDR_MOD indices (Wormhole uses ADDR_MOD_2/ADDR_MOD_3 where Blackhole uses ADDR_MOD_6/ADDR_MOD_7) because Wormhole calls `math::set_addr_mod_base()` during `_llk_math_eltwise_unary_sfpu_start_` while Blackhole does not.

Two conversion kernels (`calculate_typecast_fp32_to_uint8` and `calculate_typecast_uint_to_uint8`) are defined directly in the arch-specific wrapper file (`ckernel_sfpu_typecast.h` in `llk_api/llk_sfpu/`) rather than in the tt_llk submodule.

#### Blackhole Core SFPU Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 2-cycle-per-row throughput: load -> max(v,0) -> rnd_to_uint16 -> store

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, v >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 1-cycle-per-row throughput: load LO16 -> cast -> rnd_fp16b -> store

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 4 cycles per row. L0=0.0, L1=-2^31. Handles sign via abs+setsgn+indirect MAD.

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG: L7 = t >> 31
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // exp = in.Exp; LaneEnabled = (exp >= 0) via combined sign+exponent CC
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = INT_MIN (0x80000000 as bf16)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31; CC_LT0 enables lanes where exp < 31 (not overflow)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8 (adjust to shift mantissa from bit23 position)
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) -- mantissa with implicit 1 at bit 23
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // result = mantissa << shift_amount
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);

        // LaneEnabled = in < 0 (sign bit check)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = ~result + 1 (two's complement negation, only for negative inputs)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // LaneEnabled = in >= 0 (skip negative values -- result stays 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // exp = in.Exp; CC further filters by exponent sign
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xffffffff (UINT32_MAX for overflow case)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32; CC_LT0 enables lanes where exp < 32 (not overflow)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // result <<= shift_amount
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 3 cycles per row. Implements banker's rounding for fp32->bf16 via SFPLOADMACRO.
    // Uses instruction templates to right-shift by 16, extract LSB, add 0x7fff rounding bias.

    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_6, b >> 2);
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 1 cycle per row. SFPLOADMACRO handles load, cast, and store in pipelined fashion.

    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 4 cycles per row. L0=0.0, L1=-2^31. Uses abs, shift sign into L7, cast, setsgn, indirect MAD.

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // L7 = t >> 31
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 3 cycles per row. L0=0.0, L1=2^31. Shifts sign into L7, clears sign, casts, indirect MAD adds 2^31 if needed.

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5); // L7 = v >> 31
        TT_SFPSETSGN(0, v, v, 1); // clear sign bit (set sign to 0)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 3 cycles per row using 3 SFPLOADMACRO calls. L0=0.0, L1=2^31.

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_7, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 1 cycle per row. Load LO16 -> store INT32 via SFPLOADMACRO (zero-extend).

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 2 cycles per row. Load high 16, negate to get saturation mask, OR with full value, store LO16.

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_6, b >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // 3 cycles per row. Load INT32, cast to fp32, clamp to [0, 65535] via SFPSWAP+SFPSTOCHRND.

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, a >> 2);
        TT_SFPCAST(a, a, 0);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

#### Arch-Specific Wrapper (fp32_to_uint8, uint_to_uint8 -- Blackhole)

These two kernels are defined directly in the arch-specific wrapper, not in the tt_llk submodule. They use raw TTI instructions rather than SFPLOADMACRO.

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() // APPROXIMATION_MODE=false, ITERATIONS=8
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // exponent = exexp(in) -- extract debiased exponent
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // mantissa = exman8(in) -- extract mantissa with implicit 1 at bit 23
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // exponent -= 23; shift amount = exponent - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa >>= (23 - exponent) via variable shift
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0 (check sign for two's complement)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // mantissa = ~mantissa + 1 (two's complement negation, conditional on sign)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa += 256 (bias to handle wrap-around before masking)
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true (re-enable all lanes)
        TTI_SFPENCC(0, 0, 0, 0);
        // mantissa &= 0xFF (LREG12 = vConstIntPrgm0 = 0xFF from init)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() // APPROXIMATION_MODE=false, ITERATIONS=8
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0); // load as uint16 (zero-extended)
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0); // load as int32/uint32
        }
        // value += 256 (bias before masking to handle underflow)
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // value &= 0xFF (LREG12 = vConstIntPrgm0 = 0xFF from init)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}
```

#### Init Functions (Blackhole)

Init functions that are not SFPLOADMACRO-based are simple (e.g., `init_typecast_fp32_to_uint8` just sets `vConstIntPrgm0 = 0xFF`). The SFPLOADMACRO-based inits are more complex, programming instruction templates and macro configurations via `SFPCONFIG`. Below is one representative example:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_uint16_()
{
    // InstructionTemplate[0]: SFPSWAP -- clamps to max(0, value)
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf);

    // InstructionTemplate[1]: SFPSTOCHRND -- converts fp32 to uint16 with saturation
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6); // SFPSTOCHRND_MOD1_FP32_TO_UINT16

    // Macro 0: programs the SFPLOADMACRO pipeline stages
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0); // enable, template[0]
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (2 << 3) | (4 + 1); // VD=16, template[1]
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;       // VD=16, delay=3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0); // Program macro slot 0
    }

    // Misc config: StoreMod0=LO16, UnitDelayKind=WaitForElapsedInstructions
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

// Simple init for uint8 conversions:
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF; // mask constant loaded into LREG12
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF; // mask constant loaded into LREG12
}
```

### SFPU Instructions Used

The typecast operation uses a wide variety of SFPU instructions depending on the conversion path:

| Instruction | Description |
|---|---|
| `SFPLOADMACRO` / `TT_SFPLOADMACRO` | Macro-scheduled load from DEST into LREG with automatic pipeline dispatch of Simple/MAD/Round/Store sub-operations. Achieves high throughput (1-4 cycles per row) by overlapping operations across pipeline stages. Used by most conversion paths on Blackhole. |
| `SFPLOAD` / `TTI_SFPLOAD` | Load a single row from DEST register into an SFPU local register (LREG). Load mode determines format interpretation (DEFAULT=float, INT32=integer, LO16=lower 16 bits). |
| `SFPSTORE` / `TTI_SFPSTORE` | Store a row from LREG back to DEST register. Store mode determines output format (INT32, FP32, FP16B, LO16). |
| `SFPLOADI` / `TTI_SFPLOADI` | Load an immediate value into an LREG. MOD0 selects format: USHORT (unsigned 16-bit), SHORT (signed 16-bit), FLOATB (bfloat16), LOWER/UPPER (for programming SFPLOADMACRO configs). |
| `SFPEXEXP` / `TTI_SFPEXEXP` | Extract the debiased exponent from a float in LREG. MOD1 flags can set condition codes based on exponent sign and magnitude. |
| `SFPEXMAN` / `TTI_SFPEXMAN` | Extract the mantissa from a float with the implicit leading 1 bit restored at bit position 23. |
| `SFPSHFT` / `TTI_SFPSHFT` | Variable shift: shift LREG_dst by the amount in LREG_src (positive=left, negative=right). Used to convert mantissa+exponent to integer. |
| `SFPSHFT2` / `TTI_SFPSHFT2` | Shift with two modes: MOD1=5 shifts by LREG amount (used to extract sign bit), MOD1=6 shifts by immediate. |
| `SFPIADD` / `TTI_SFPIADD` | Integer add with multiple modes: ARG_IMM adds 12-bit signed immediate, ARG_2SCOMP_LREG_DST negates dst. CC_LT0 sets condition code if result < 0, CC_NONE skips CC update. |
| `SFPSETCC` / `TTI_SFPSETCC` | Set condition code (lane enable mask) based on LREG value. MOD1_LREG_LT0 enables lanes where value < 0, MOD1_LREG_GTE0 enables lanes where value >= 0. |
| `SFPENCC` / `TTI_SFPENCC` | Enable all lanes (clear condition code mask). Restores full SIMD execution after conditional operations. |
| `SFPAND` / `TTI_SFPAND` | Bitwise AND between two LREGs. Used to mask values (e.g., `value & 0xFF` for uint8 truncation). |
| `SFPOR` / `TTI_SFPOR` | Bitwise OR between two LREGs. Used in uint32_to_uint16 to combine saturation mask with value. |
| `SFPABS` / `TT_SFPABS` | Compute absolute value of LREG. Used in int32-to-float conversions to handle sign separately. |
| `SFPCAST` / `TTI_SFPCAST` | Cast unsigned integer to float (uint32 to fp32). Used in int-to-float conversion paths. |
| `SFPSETSGN` / `TT_SFPSETSGN` | Set or clear the sign bit of a float. MOD1=1 sets sign to immediate (0=positive), MOD1=0 copies sign from another LREG. |
| `SFPMAD` / `TTI_SFPMAD` | Multiply-add: `VA * VB + VC`. With MOD1_INDIRECT_VA, VA is selected indirectly via LREG7, enabling conditional addition of constants (e.g., adding -2^31 for signed int range). |
| `SFPSWAP` / `TTI_SFPSWAP` | Min/max swap. With MOD1=0xf, computes `max(LCONST_0, LREG)` to clamp negative values to 0. |
| `SFP_STOCH_RND` / `TTI_SFP_STOCH_RND` | Stochastic/deterministic rounding for format conversion. MOD1=1 converts fp32 to fp16b, MOD1=6 converts fp32 to uint16 with saturation to [0, 65535]. |
| `SFPCONFIG` / `TTI_SFPCONFIG` | Configure SFPLOADMACRO macro slots and miscellaneous settings (store format, delay kind). Programs the macro instruction pipeline. |
| `SFPNOP` / `TTI_SFPNOP` | No operation. Required for pipeline draining after SFPLOADMACRO sequences and for timing-sensitive instruction scheduling. |
| `SFPGT` / `TTI_SFPGT` | Greater-than comparison. Used in uint32_to_uint16 init to generate saturation mask (`value > 0 ? -1 : 0`). |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG0** | Primary input/working register. In many SFPLOADMACRO paths, alternates with LREG1 for double-buffering. In int-to-float paths, holds constant 0.0. |
| **LREG1** | Secondary working register. Alternates with LREG0 for double-buffering. In int-to-float paths, holds constant -2^31 or 2^31. |
| **LREG2** | Working register. Used as scratch (e.g., holds exponent in fp32-to-int paths). In SFPLOADMACRO int conversions, used alongside LREG3 for input alternation. |
| **LREG3** | Alternates with LREG2 for double-buffered input in int32-to-float conversions. |
| **LREG4** | Temporary register `t` in int32-to-float paths. Holds abs(value) during sign extraction and casting. |
| **LREG7** | Indirect addressing register. Stores the sign bit (0 or 1) extracted via `SFPSHFT2(v, LREG12, LREG7, 5)`. Used by `SFPMAD` with `MOD1_INDIRECT_VA` to select between LREG0 (0.0) and LREG1 (+-2^31). |
| **LREG12** (`vConstIntPrgm0`) | Programmable constant register. Set to `0xFF` for uint8 masking, `1` for fp32_to_fp16b LSB extraction, or `-31` for shift-based operations. Also used as shift source in `SFPSHFT2`. |
| **LREG13** (`vConstIntPrgm1`) | Programmable constant register. Set to `0x7fff` for fp32_to_fp16b rounding bias. |
| **LCONST_0** | Hardware constant 0.0 (float) / 0 (int). Used in SFPSWAP for clamping and in SFPIADD for two's complement. |
| **LCONST_1** | Hardware constant 1.0 (float). Used as VB in SFPMAD for `L[L7] * 1.0 + v` pattern. |
| **DEST registers** | Source and destination for tile data. SFPLOAD reads from DEST, SFPSTORE writes back. Auto-increment controlled by ADDR_MOD. |

### Address Mode Configuration

The typecast operation configures two ADDR_MOD entries during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>()`:

**Blackhole:**
- **ADDR_MOD_7**: `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- Used by `SFPLOAD` and some `SFPLOADMACRO` calls. No auto-increment; the SFPLOADMACRO pipeline or explicit iteration handles address progression.
- **ADDR_MOD_6**: `{srca.incr=0, srcb.incr=0, dest.incr=2}` -- Used by `SFPSTORE` and the final `SFPLOADMACRO` call in a sequence. Auto-increments DEST address by 2 rows after each operation, advancing through the 8 rows of a face half.

**Wormhole:**
- **ADDR_MOD_7**: `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- Same as Blackhole. No auto-increment.
- **ADDR_MOD_6**: `{srca.incr=0, srcb.incr=0, dest.incr=2}` -- Same as Blackhole. Auto-increments by 2.

However, the Wormhole SFPU kernels reference `ADDR_MOD_2` and `ADDR_MOD_3` (instead of `ADDR_MOD_6` and `ADDR_MOD_7`) in their TTI instructions. This is because Wormhole's `_llk_math_eltwise_unary_sfpu_start_` calls `math::set_addr_mod_base()`, which remaps the base address mode register bank. The logical behavior is equivalent: one mode with dest.incr=0 for loads and one with dest.incr=2 for stores.

The `dest.incr=2` value means that after each SFPSTORE (or the store stage of SFPLOADMACRO), the DEST pointer advances by 2 rows. Since the SFPU processes one row per iteration and the loop runs for ITERATIONS=8, this covers all 8 rows of a face half (with the pipeline consuming 2 rows per logical iteration due to double-buffering in many kernels). The `_llk_math_eltwise_unary_sfpu_params_` function calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces to advance to the next 16x16 face.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the typecast_tile compute API work? What is the SFPU kernel implementation for typecast? Trace the call chain from typecast_tile through LLK to the ckernel SFPU implementation."
   **Reason**: Needed to understand the full abstraction layer stack and identify which files contain the core SFPU implementation.
   **Key Findings**: Confirmed the call chain: `typecast_tile` -> `llk_math_eltwise_unary_sfpu_typecast` -> format-specific SFPU functions. Identified that typecast has 20+ conversion paths, some handled entirely by unpacker/packer without SFPU involvement. The SFPU kernels use TTI macros mapping to hardware instructions.

### Confluence References
Not consulted for this analysis. The SFPU instructions used are well-documented in the codebase comments and DeepWiki.

### Glean References
Not consulted for this analysis.
