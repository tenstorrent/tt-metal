## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to for the TYPECAST (rm_chunked) operation. Unlike most SFPU operations that have a single kernel path, typecast is a multi-kernel dispatcher -- the specific SFPU function invoked depends on the compile-time input/output data format pair. Some format pairs require no SFPU work at all (handled entirely by unpacker/packer).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_typecast.h` (underscore-prefixed functions) and `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` (wrappers + arch-local functions like `calculate_typecast_fp32_to_uint8`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel (`eltwise_typecast.cpp`) calls `TYPECAST_LLK(0)`, which expands (via program factory defines) to `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)`.
2. `typecast_tile<IN_DTYPE, OUT_DTYPE>(idst)` (in `api/compute/eltwise_unary/typecast.h`) gates on `TRISC_MATH` and calls `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)`.
3. `llk_math_eltwise_unary_sfpu_typecast` (in `llk_math_eltwise_unary_sfpu_typecast.h`) casts the integer format IDs to `DataFormat` enums and uses `if constexpr` to select the appropriate `ckernel::sfpu::calculate_typecast_*` function, passing it to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) calls `_llk_math_eltwise_unary_sfpu_start_` to set the DEST write address, then iterates over 4 faces (for `VectorMode::RC`), calling the selected SFPU function once per face (each call processes 8 rows = one 16x16 face).
5. The selected `calculate_typecast_*` wrapper delegates to the `_calculate_typecast_*_` underscore-prefixed implementation which issues the actual SFPU instructions.

The init path follows a parallel chain: `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` which calls `llk_math_eltwise_unary_sfpu_typecast_init`, which selects the appropriate `init_typecast_*` function and passes it to `llk_math_eltwise_unary_sfpu_init<SfpuType::typecast, APPROXIMATE>(init_func)`. This calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>()` (which configures `ADDR_MOD_6` and `ADDR_MOD_7`), then invokes the format-specific init function.

### Annotated SFPU Kernel Source

The typecast operation is unique in that it comprises **16+ distinct SFPU kernel functions** selected at compile time based on input/output data format. Several format pairs bypass the SFPU entirely (e.g., Float16_b <-> Float32, Float16_b <-> Bfp8_b, etc.) because the conversion is handled by the unpacker or packer hardware. Below is the complete source from the Blackhole architecture variant. The Wormhole variant is structurally identical but uses different `ADDR_MOD` indices (ADDR_MOD_2/ADDR_MOD_3 instead of ADDR_MOD_6/ADDR_MOD_7 in some functions) and has a different `_calculate_typecast_fp32_to_int32_` implementation (SFPI-based instead of TTI-based).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 2-cycle-per-row throughput.
    // Pipeline: Load -> max(v, 0.0) -> rnd(v) to uint16 -> Store L16

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
    // Uses SFPLOADMACRO for 1-cycle-per-row throughput.
    // Pipeline: Load LO16 -> cast(v) -> rnd(v) to fp16b -> Store L16

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
    // 4 cycles per row. Uses abs() + cast + setsgn + SFPMAD with indirect VA
    // L0=0.0, L1=-2**31. L7 stores sign bit for indirect VA selection.

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

        // exp = in.Exp; LaneEnabled = (exp >= 0) -- combined sign+exponent CC
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = INT_MIN (0x80000000 as float bits)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31; LaneEnabled = (exp < 31) -- overflow guard
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) -- mantissa with implicit 1 at bit 23
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // result = mantissa << shift_amount
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result (two's complement negation)
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

        // LaneEnabled = in >= 0 -- clamp negative to 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // exp = in.Exp; LaneEnabled further refined by exp >= 0
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xffffffff (max uint32)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32; LaneEnabled = (exp < 32) -- overflow guard
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman8(in) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
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
    // 3 cycles per row. Uses SFPLOADMACRO with manual rounding (shift right 16,
    // extract LSB, add 0x7fff, store upper 16 bits as BF16).

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
    // 1 cycle per row. Load LO16 -> cast -> Store FP32.

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
    // 4 cycles per row. Uses abs + SFPCAST + setsgn + SFPMAD with indirect VA.

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // L7 = t >> 31 (sign bit)
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
    // 3 cycles per row. Uses SFPCAST + SFPMAD with indirect VA + SFPSTOCHRND.

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5); // L7 = v >> 31
        TT_SFPSETSGN(0, v, v, 1); // v = setsgn(v, 0) -- clear sign bit
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
    // 3 cycles per row. Uses 3 SFPLOADMACRO per iteration with setsgn + cast + SFPMAD.

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
    // 1 cycle per row. Load LO16 -> Store INT32 (zero-extend).

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
    // 2 cycles per row. Uses SFPGT + SFPOR for saturation clamping.

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
    // 3 cycles per row. Uses SFPCAST + SFPSWAP(max(0,v)) + SFPSTOCHRND.

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

The following are arch-local functions defined in the wrapper layer (identical for both Blackhole and Wormhole, differing only in ADDR_MOD indices):

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // exponent = exexp(in)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // mantissa = exman8(in)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // exponent -= 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa = mantissa >> (23 - exponent)
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // mantissa = ~mantissa + 1 (two's complement)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa += 256
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // mantissa &= 0xFF (LREG12 = vConstIntPrgm0 = 0xFF set in init)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        }
        // value += 256 (ensures low byte is preserved after AND)
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // value &= 0xFF
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}
```

### SFPU Instructions Used

The typecast operation uses a wide variety of SFPU instructions across its many conversion paths:

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Loads a datum from DEST register into an SFPU local register (LREG). Mode controls interpretation: DEFAULT (FP32), INT32, LO16 (lower 16 bits). |
| `SFPSTORE` | Stores a datum from an SFPU local register back to DEST. Mode controls output format (INT32, FP32, FP16B, LO16). |
| `SFPLOADI` | Loads an immediate constant into an SFPU local register. MOD0 selects format: USHORT (unsigned 16-bit), SHORT (signed 16-bit sign-extended), FLOATB (BFloat16). |
| `SFPLOADMACRO` / `TT_SFPLOADMACRO` | Triggers a pre-configured macro sequence that pipelines Load/Simple/MAD/Round/Store sub-units for high-throughput conversion. The macro template and scheduling bits are configured during init via `SFPCONFIG`. |
| `SFPCONFIG` | Configures SFPLOADMACRO templates: instruction templates, macro scheduling bits (simple/mad/round/store sub-unit assignments), and miscellaneous settings (store format, unit delay). |
| `SFPEXEXP` | Extracts the debiased exponent from a floating-point value. MOD1 flags control condition code side-effects: `SET_CC_SGN_EXP` enables lanes with positive sign+exponent, `SET_CC_COMP_EXP` composes with existing CC. |
| `SFPEXMAN` | Extracts the mantissa with implicit leading 1 at bit 23 (exman8 mode). |
| `SFPIADD` | Integer add with immediate or register operand. MOD1 controls: `ARG_IMM` (immediate mode), `ARG_2SCOMP_LREG_DST` (two's complement negation of destination), `CC_NONE` (no CC update), `CC_LT0` (enable lanes where result < 0). |
| `SFPSHFT` | Variable bit-shift: shifts LREG by signed amount in another LREG. Positive = left shift, negative = right shift. |
| `SFPSHFT2` | Shift with secondary modes: `MOD1_SHFT_IMM` (immediate shift amount), `MOD1_SHFT_LREG` (shift by LREG12/vConstIntPrgm0). Used for sign-bit extraction (>>31) and rounding (>>16). |
| `SFPSETCC` | Sets the condition code (lane enable mask) based on register comparison: `LREG_LT0` (lane enabled if value < 0), `LREG_GTE0` (lane enabled if value >= 0). |
| `SFPENCC` | Unconditionally enables all lanes (clears condition code). |
| `SFPCAST` | Converts between integer and floating-point representations within SFPU local registers. |
| `SFPABS` | Computes absolute value of a local register. |
| `SFPSETSGN` | Sets or clears the sign bit. `MOD1_ARG_IMM` clears sign (sets to 0); standard mode copies sign from another register. |
| `SFPMAD` | Multiply-add: computes VA * VB + VC. `MOD1_INDIRECT_VA` selects VA from L[L7] indirection (used for sign-dependent correction with 2^31 constants). |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding for format conversion. Modes: `FP32_TO_FP16B` (round FP32 to BF16), `FP32_TO_UINT16` (convert FP32 to uint16 with saturation clamping). |
| `SFPSWAP` | Swap/min-max operation. With `MOD1=0xf` and `LCONST_0`, performs `max(0, v)` to clamp negative values to zero. |
| `SFPAND` | Bitwise AND between local registers. Used for masking (e.g., `& 0xFF` for uint8, `& 1` for rounding LSB extraction). |
| `SFPOR` | Bitwise OR between local registers. Used in uint32-to-uint16 conversion for combining high and low halves. |
| `SFPGT` | Greater-than comparison. With `MOD1_SET_VD`, sets destination to -1 (all ones) or 0 based on comparison, used for uint32-to-uint16 saturation. |
| `SFPNOP` | No-operation. Required for pipeline hazard avoidance between SFPLOADMACRO-scheduled sub-operations. |

### SFPU Register Usage

**Local Registers (LREGs)**:
- **LREG0-LREG3**: Primary working registers. Most conversion functions alternate between LREG0/LREG1 (or LREG2/LREG3 for int32 conversions) across iterations to enable pipelining.
- **LREG4 (alias `t`)**: Temporary register used in int32-to-float conversions for holding absolute value during sign processing.
- **LREG7**: Stores the sign bit (0 or 1) extracted via `SFPSHFT2(v, LREG12, LREG7, 5)` (shift right by 31). Used as an indirect index for `SFPMAD` with `INDIRECT_VA` mode, selecting between L0 (0.0) and L1 (+-2^31).
- **LREG12 (vConstIntPrgm0)**: Programmable constant register. Set to different values by init functions: `0xFF` for uint8 masking, `1` for LSB extraction (fp32-to-fp16b rounding), `-31` for shift constants.
- **LREG13 (vConstIntPrgm1)**: Programmable constant register. Set to `0x7fff` for fp32-to-fp16b round-to-nearest-even bias.
- **LCONST_0**: Hardware constant register containing 0.0 (float) / 0 (int). Used as zero operand for two's complement negation and max(0, v) operations.
- **LCONST_1**: Hardware constant register containing 1.0. Used as the VB operand in `SFPMAD` for `VA * 1.0 + VC` patterns.

**DEST Registers**: The SFPU operates on DEST register rows. `SFPLOAD` reads from the current DEST address; `SFPSTORE` writes back. The DEST address auto-increments by 2 per operation (controlled by `ADDR_MOD_6`), so each iteration processes one row of 16 elements within a face. The `_llk_math_eltwise_unary_sfpu_params_` framework advances between the 4 faces (each 16x16) of a 32x32 tile by incrementing the DEST address by 16 between face invocations.

### Address Mode Configuration

The typecast operation configures two address modes during initialization (`_llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>()`):

**ADDR_MOD_7**: Used for SFPLOAD instructions (reading from DEST without advancing).
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},   // No DEST auto-increment -- manual control
}
```

**ADDR_MOD_6**: Used for SFPSTORE instructions and many SFPLOADMACRO calls (reading/writing with DEST advance).
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 2},   // Advance DEST by 2 rows per store (each SFPU row = 16 elements, 2 rows = 32 elements)
}
```

This configuration is **identical between Wormhole and Blackhole** architectures. However, the Wormhole SFPU implementations use `ADDR_MOD_2` and `ADDR_MOD_3` instead of `ADDR_MOD_6` and `ADDR_MOD_7` respectively in the actual calculate/store instructions. This is because Wormhole's `eltwise_unary_sfpu_configure_addrmod` sets the same logical configuration on ADDR_MOD_6/ADDR_MOD_7, and the individual SFPU kernel implementations reference ADDR_MOD_2/ADDR_MOD_3 which are configured by the A2D (unpack-to-DEST) machinery. The Blackhole variant unifies on ADDR_MOD_6/ADDR_MOD_7 for both the configuration and the instruction references.

Note: Blackhole also includes `SfpuType::reciprocal` in the same `ADDR_MOD_6` configuration block as typecast, while Wormhole does not (reciprocal uses a different ADDR_MOD on Wormhole).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "Where is the typecast_tile and typecast_tile_init compute API implemented? Trace through abstraction layers from compute_kernel_api/typecast.h down to the LLK dispatch and core SFPU ckernel implementation."
   **Reason**: Needed to locate the full file hierarchy for the typecast SFPU kernel across abstraction layers.
   **Key Findings**: Confirmed the four-layer architecture: API header -> LLK dispatch (llk_math_eltwise_unary_sfpu_typecast.h) -> arch-specific wrappers (ckernel_sfpu_typecast.h in ckernels/) -> core implementations (sfpu/ckernel_sfpu_typecast.h in tt_llk). Some conversions are handled entirely by packer/unpacker without SFPU involvement.

2. **Query**: "How is the typecast SFPU kernel implemented in the LLK layer? What SFPU instructions does typecast use?"
   **Reason**: Needed to understand the full set of SFPU instructions used across all typecast conversion paths and the dispatch mechanism.
   **Key Findings**: Identified the comprehensive instruction set including SFPCAST, SFPLOADMACRO, SFPEXEXP, SFPEXMAN, SFPSHFT, SFPIADD, SFPSETCC, SFPENCC, SFP_STOCH_RND, SFPMAD, SFPSETSGN, SFPSWAP, SFPAND, SFPOR, SFPGT, and SFPABS. Confirmed that SFPLOADMACRO is the primary throughput optimization mechanism for most conversion paths.

### Confluence References
No Confluence references were consulted for this analysis. The instruction behavior was sufficiently documented in the source code comments and DeepWiki.

### Glean References
No Glean references were consulted for this analysis.
