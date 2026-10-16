// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    const sfpi::vFloat slope_v = Converter::as_float(slope);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < 0.0f)
        {
            v = v * slope_v;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

// Float relu_max. Reflects the current Wormhole float behavior (the Prgm1 scalar-threshold
// form lives in _relu_max_float_impl_ on Blackhole; that optimization is a separate item).
// The integer instantiation is retired: the plain compares below are not safe over the full
// int32 range, and int thresholds route to _relu_max_int_impl_ instead (item 13).
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold)
        {
            result = threshold;
        }
        v_endif;
        v_if (result < 0)
        {
            result = 0;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Dst holds int32 in whatever encoding the caller put there, and ttnn's is two's complement,
// which is what the compares below assume. Pinned rather than left to sfpi's default, because
// that default is per-arch: SM32 (converting) on Wormhole, so an unpinned load/store would
// not read the two's-complement word the compares need.
inline constexpr sfpi::DataLayout INT_DEST_LAYOUT = sfpi::DataLayout::I32;

// Integer relu_max: max(0, min(a, threshold)) on int32 in Dst. Mirrors the metal relu_clamp_int.
// A threshold <= 0 gives 0 on every lane, and a positive threshold zeroes the negative lanes
// first -- which is their whole answer -- so the `a > threshold` compare only ever sees
// non-negative operands. That keeps the compare exact under either arch's integer compare
// lowering, where a bare ordered compare over the full int32 range is not (the difference of
// operands 2^31 or more apart overflows the subtract-based forms).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_int_impl_(const int iterations, const int threshold)
{
    if (threshold > 0)
    {
        const sfpi::vInt v_threshold = threshold;
        for (int d = 0; d < iterations; d++)
        {
            sfpi::vInt a = sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>();
            v_if (a < 0)
            {
                a = 0;
            }
            v_endif;
            v_if (a > v_threshold)
            {
                a = v_threshold;
            }
            v_endif;
            sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>() = a;
            sfpi::dst_reg++;
        }
    }
    else
    {
        const sfpi::vInt zero = 0;
        for (int d = 0; d < iterations; d++)
        {
            sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>() = zero;
            sfpi::dst_reg++;
        }
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: sfpi::vInt has no float constructor, so the assignment below would otherwise fail as an "
            "ambiguous conversion");
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            // Raw two's-complement int bits, the encoding _relu_min_'s integer branch takes.
            // The old decode -- int(Converter::as_float(threshold)) -- read a float bit
            // pattern: a raw int threshold like 5 arrived as a denormal and truncated to 0,
            // and a float-encoding threshold fed an integer compare that subtracts and
            // inverts its answer once the operands are 2^31 or more apart.
            // _relu_max_int_impl_ takes the threshold directly and keeps its compare exact.
            _relu_max_int_impl_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, static_cast<int>(threshold));
            return;
        }
        v_threshold = Converter::as_float(threshold);
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// Contract: the threshold is an implicit input in LREG2, not a parameter -- every caller must
// load it with _sfpu_load_imm32_ first, which the raw-TTI body cannot express in its signature.
// SFPLOAD_INSTR_MOD has to stay a template parameter because it feeds an immediate asm operand:
// as a runtime argument it compiles only while the optimiser folds it, and fails at -O0.
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore SFPLOAD_INSTR_MOD>
inline void _relu_min_impl_(const int iterations)
{
    static_assert(
        SFPLOAD_INSTR_MOD == InstrModLoadStore::DEFAULT || SFPLOAD_INSTR_MOD == InstrModLoadStore::INT32_2S_COMP,
        "SFPLOAD_INSTR_MOD must be DEFAULT (fp32 datapath) or INT32_2S_COMP (integer datapath)");

    for (int d = 0; d < iterations; d++)
    {
        // Load input tensor to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, SFPLOAD_INSTR_MOD, ADDR_MOD_3, 0);
        // Copy value param from lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        // Store the result
        TTI_SFPSTORE(p_sfpu::LREG1, SFPLOAD_INSTR_MOD, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold)
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    // The load/store mode the branches below encode the threshold for. Only the integer
    // datapath needs a non-default mode, and which branch runs is fixed by <T, VectorType>,
    // so this is a compile-time constant rather than a variable the branches assign -- see
    // the note on _relu_min_impl_'s SFPLOAD_INSTR_MOD parameter.
    constexpr InstrModLoadStore SFPLOAD_INSTR_MOD =
        (std::is_same_v<T, std::uint32_t> && std::is_same_v<VectorType, sfpi::vInt>) ? InstrModLoadStore::INT32_2S_COMP : InstrModLoadStore::DEFAULT;

    // Invariant every branch below must uphold: leave the threshold in LREG2, in the encoding
    // SFPLOAD_INSTR_MOD selects. A branch that sets only a local vector compiles clean and then
    // reads whatever the previously executed SFPU kernel left in LREG2, so the load is not
    // optional on any path.
    if constexpr (std::is_same_v<T, float>)
    {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: the LREG2 load below bit-casts the float, which is meaningless for an integer datapath");
        _sfpu_load_imm32_(p_sfpu::LREG2, __builtin_bit_cast(std::uint32_t, threshold));
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            // SFPSWAP orders its operands as sign+magnitude, and INT32_2S_COMP is the load
            // mode that converts a two's-complement DEST word into that form. The threshold
            // does not arrive through a load, so it is re-encoded by hand here to match --
            // scoped to this branch, since on a float the same bits mean something else.
            constexpr std::uint32_t SIGN_MAG_SIGN_BIT      = 0x80000000u;
            constexpr std::uint32_t SIGN_MAG_MAX_MAGNITUDE = 0x7FFFFFFFu;

            const int scalar       = static_cast<int>(threshold);
            std::uint32_t sign_mag = static_cast<std::uint32_t>(scalar);
            if (scalar < 0)
            {
                // Negate in unsigned: -INT_MIN is signed overflow. INT_MIN has no
                // sign+magnitude form, so it saturates to -(2^31 - 1); the clamp is what
                // saturates it, where a mask would encode negative zero and clamp at 0.
                const std::uint32_t magnitude = -static_cast<std::uint32_t>(scalar);
                sign_mag                      = SIGN_MAG_SIGN_BIT | (magnitude > SIGN_MAG_MAX_MAGNITUDE ? SIGN_MAG_MAX_MAGNITUDE : magnitude);
            }
            _sfpu_load_imm32_(p_sfpu::LREG2, sign_mag);
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<APPROXIMATION_MODE, ITERATIONS, SFPLOAD_INSTR_MOD>(ITERATIONS);
}

} // namespace sfpu
} // namespace ckernel
