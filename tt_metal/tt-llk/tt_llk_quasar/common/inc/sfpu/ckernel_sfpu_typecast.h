// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// SFPCAST instr_mod1 selectors (assembly.yaml SFPCAST): pick the 32-bit cast direction.
constexpr std::uint32_t SFPCAST_INT32_TO_FP32_RNE = 0x0; // int32 sign+mag -> fp32, round-nearest-even
constexpr std::uint32_t SFPCAST_FP32_TO_INT32_RNE = 0x4; // fp32 -> int32, round-nearest-even

// SFP_STOCH_RND instr_mod1 bit 3: use the immediate descale field in place of srcb (int narrowing).
constexpr std::uint32_t STOCHRND_BIT_IMM_DESCALE = 1 << 3;

// Compile-time format classification for the typecast datapath. In the SFPU register file a
// float loads to fp32 and an int loads to sign-magnitude int32, so the conversion sequence is
// picked from these classes, not from individual formats.
template <DataFormat FMT>
inline constexpr bool _typecast_is_float_()
{
    return FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 || FMT == DataFormat::Tf32;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_fp16_()
{
    return FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_int32_wide_()
{
    // Quasar's DataFormat enum has no UInt32; Int32 is the only 32-bit integer format.
    return FMT == DataFormat::Int32;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_int8_()
{
    return FMT == DataFormat::UInt8 || FMT == DataFormat::Int8;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_unsigned_int_()
{
    return FMT == DataFormat::UInt8 || FMT == DataFormat::UInt16;
}

// Emit the round-and-narrow instruction (fp32 -> narrow int, or int32 -> narrow int). UInt16 has
// no _sfpu_stochround_conversion_ entry, so it uses the raw FP32_TO_UINT16 selector directly
// (matches the legacy ckernel_sfpu_typecast_fp16b_uint16.h). The lreg indices are template
// parameters because TTI_SFP_STOCH_RND encodes them as instruction immediates.
template <DataFormat CAST_SRC_FMT, DataFormat DST_FMT, std::uint32_t LREG_IN, std::uint32_t LREG_OUT>
inline void _typecast_stochrnd_narrow_()
{
    constexpr std::uint32_t stochrnd_conv = []
    {
        if constexpr (DST_FMT == DataFormat::UInt16)
        {
            return p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT16;
        }
        else
        {
            return ::_sfpu_stochround_conversion_<CAST_SRC_FMT, DST_FMT>();
        }
    }();
    // The immediate-descale path is an int-narrowing feature; float -> fp16 narrowing does not
    // use it (and must not, to keep the validated fp32 -> fp16 encoding unchanged).
    constexpr std::uint32_t descale = _typecast_is_float_<DST_FMT>() ? 0u : STOCHRND_BIT_IMM_DESCALE;
    TTI_SFP_STOCH_RND(
        p_sfpu::sfp_stochrnd_rnd_mod::NearEven, 0 /* imm8_math */, 0 /* lreg_b: unused, descale via imm */, LREG_IN, LREG_OUT, descale | stochrnd_conv);
}

// SFPU arithmetic cast: conversions the datapath cannot do, performed in the SFPU register file.
// The (src,dst) classes select the load/store sfpmem modes and the bridge instruction(s):
//   float->float : load; narrow via stochrnd (-> fp16) or widen with a plain store (-> fp32).
//   float->int32 : load; SFPCAST fp32 -> int32.
//   float->narrowint : load; clamp negatives (unsigned); stochrnd fp32 -> narrow int.
//   int->float   : load; SFPCAST int32 -> fp32; narrow via stochrnd if the dst is fp16.
//   int->int     : load; stochrnd to 8-bit, else width handled by the store sfpmem mode.
// Note: MX / block-float typecasts are NOT handled here. Those are a pure unpack/pack gasket
// format conversion (dest holds Float16_b after the unpack reconfig, and the pack reconfig
// converts Float16_b -> MXFP8), i.e. a datacopy — they never route through this SFPU op.
template <DataFormat SRC_FMT, DataFormat DST_FMT>
inline void _calculate_typecast_arith_sfp_rows_()
{
    constexpr std::uint32_t sfpmem_src = ::_sfpu_sfpmem_type_<SRC_FMT>();
    constexpr std::uint32_t sfpmem_dst = ::_sfpu_sfpmem_type_<DST_FMT>();

    constexpr bool src_float = _typecast_is_float_<SRC_FMT>();
    constexpr bool dst_float = _typecast_is_float_<DST_FMT>();

    TTI_SFPLOAD(p_sfpu::LREG0, sfpmem_src, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);

    if constexpr (src_float && dst_float)
    {
        // float -> float. Narrowing to fp16 rounds; widening to fp32 is just the store (the
        // value already sits in LREG0 as fp32 after the load).
        if constexpr (_typecast_is_fp16_<DST_FMT>())
        {
            _typecast_stochrnd_narrow_<DataFormat::Float32, DST_FMT, p_sfpu::LREG0, p_sfpu::LREG1>();
        }
    }
    else if constexpr (src_float && !dst_float)
    {
        // float -> int. The direct fp32 -> narrow-int SFP_STOCH_RND modes (fp32->uint8/int8/
        // uint16/int16) do not work on Quasar (they round to all-zeros), so compose two proven
        // steps: SFPCAST fp32 -> int32, then narrow from int32 (int32->int8 SFP_STOCH_RND for an
        // 8-bit dst, or the store sfpmem mode for a 16-bit dst).
        //
        // Clamp negatives to 0 first for unsigned targets, so the int32 the narrow sees is already
        // non-negative. Mirror the relu pattern: gate on (x < 0) and zero those lanes with a
        // CC-predicated SFPMAD (x*0 + 0). SFPLOADI is NOT CC-predicated, so it would zero EVERY lane
        // (the all-zeros bug this replaces).
        if constexpr (_typecast_is_unsigned_int_<DST_FMT>())
        {
            TTI_SFPSETCC(0 /* imm12 */, p_sfpu::LREG0, 0 /* mod1: CC_res <= LREG0 < 0 */);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // negatives -> 0
            TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC, re-enable all lanes */);
        }
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, SFPCAST_FP32_TO_INT32_RNE);
        if constexpr (_typecast_is_int8_<DST_FMT>())
        {
            // 8-bit dst: int32 -> int8 round-and-narrow (the proven INT32_TO_(U)INT8 path).
            _typecast_stochrnd_narrow_<DataFormat::Int32, DST_FMT, p_sfpu::LREG1, p_sfpu::LREG2>();
        }
        // Int32 and 16-bit dst: the value is already int32 in LREG1; for a 16-bit dst the store
        // sfpmem mode performs the narrowing (matches the int->int path, e.g. Int32->UInt16).
    }
    else if constexpr (!src_float && dst_float)
    {
        // int -> float. SFPCAST yields fp32; a narrow fp16 dst then rounds fp32 -> fp16.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, SFPCAST_INT32_TO_FP32_RNE);
        if constexpr (_typecast_is_fp16_<DST_FMT>())
        {
            _typecast_stochrnd_narrow_<DataFormat::Float32, DST_FMT, p_sfpu::LREG1, p_sfpu::LREG2>();
        }
    }
    else
    {
        // int -> int. 8-bit dst rounds-and-narrows; wider/equal dst (incl. 16-bit narrowing) is
        // handled by the store sfpmem mode, with the value already in LREG0 after the load.
        if constexpr (_typecast_is_int8_<DST_FMT>())
        {
            _typecast_stochrnd_narrow_<DataFormat::Int32, DST_FMT, p_sfpu::LREG0, p_sfpu::LREG1>();
        }
    }

    // Result register selected at compile time per branch (no instruction is emitted for the
    // pick): LREG0 for the no-cast stores, LREG2 for the int->fp16 two-step, else LREG1.
    constexpr std::uint32_t result_lreg = []
    {
        if constexpr (src_float && dst_float)
        {
            return _typecast_is_fp16_<DST_FMT>() ? p_sfpu::LREG1 : p_sfpu::LREG0;
        }
        else if constexpr (src_float && !dst_float)
        {
            // 8-bit dst ends in LREG2 (SFPCAST -> LREG1, then int32->int8 narrow -> LREG2);
            // Int32 and 16-bit dst end in LREG1 (the SFPCAST result, 16-bit narrowed by the store).
            return _typecast_is_int8_<DST_FMT>() ? p_sfpu::LREG2 : p_sfpu::LREG1;
        }
        else if constexpr (!src_float && dst_float)
        {
            return _typecast_is_fp16_<DST_FMT>() ? p_sfpu::LREG2 : p_sfpu::LREG1;
        }
        else
        {
            return _typecast_is_int8_<DST_FMT>() ? p_sfpu::LREG1 : p_sfpu::LREG0;
        }
    }();

    TTI_SFPSTORE(result_lreg, sfpmem_dst, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);
}

// Parameterized SFPU typecast: performs the (src,dst) arithmetic cast in the SFPU register file.
// MX / block-float typecasts are a pure unpack/pack gasket format conversion (a datacopy), so
// they never reach this op; instantiating it with an MX endpoint is unsupported by design.
template <DataFormat SRC_FMT, DataFormat DST_FMT, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_typecast_arith_sfp_rows_<SRC_FMT, DST_FMT>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
