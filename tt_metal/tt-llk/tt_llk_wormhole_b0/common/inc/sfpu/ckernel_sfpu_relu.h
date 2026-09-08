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
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

// Contract: the threshold is an *implicit input in LREG2*, not a parameter. Every caller must
// load it with _sfpu_load_imm32_(p_sfpu::LREG2, ...) before calling. The body is raw TTI, so
// this dependency cannot be expressed in the signature -- keep it out of the parameter list
// rather than carrying an argument the body never reads.
//
// SFPLOAD_INSTR_MOD is a template parameter rather than a function argument because it feeds
// the "n" (immediate) operand of TTI_SFPLOAD/TTI_SFPSTORE. Passed by value it compiles only
// for as long as the optimiser folds the constant into the asm, which makes a hard build
// requirement out of an optimisation. Measured on this toolchain (sfpi 7.74.0, gcc 15.1.0),
// both instantiations in one TU: as a runtime argument -O1/-O2/-O3 fold it and -O0 fails with
// "impossible constraint in 'asm'"; as a template parameter all four levels compile.
// ComputeConfig::opt_level is user-settable, so that is worth not relying on. Every sibling
// integer kernel declares the mode constexpr for the same reason (see ckernel_sfpu_add_int.h,
// ckernel_sfpu_sub_int.h, ckernel_sfpu_topk.h).
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore SFPLOAD_INSTR_MOD>
inline void _relu_min_impl_(const int iterations)
{
    static_assert(
        SFPLOAD_INSTR_MOD == InstrModLoadStore::DEFAULT || SFPLOAD_INSTR_MOD == InstrModLoadStore::INT32,
        "SFPLOAD_INSTR_MOD must be DEFAULT (fp32 datapath) or INT32 (integer datapath)");

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
        (std::is_same_v<T, std::uint32_t> && std::is_same_v<VectorType, sfpi::vInt>) ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

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
            // SFPSWAP orders its operands as sign+magnitude, so both of them have to be in
            // that form before the compare. They get there by different routes, and the two
            // have to agree:
            //
            //   the input      arrives through SFPLOAD, so the *instruction mode* converts
            //                  it. InstrModLoadStore::INT32 (SFP format I32) is the mode
            //                  that translates DEST's two's complement into sign+magnitude
            //                  on the way in, and back again on the SFPSTORE.
            //   the threshold  is written straight into LREG2 by _sfpu_load_imm32_, which
            //                  bypasses any load conversion, so it is re-encoded by hand
            //                  here to match what the input will look like.
            //
            // Selecting INT32_2S_COMP (SFP format SM32) above instead is what tt-metal #55643
            // was: that mode loads raw, so the input stayed two's complement while the
            // threshold was sign+magnitude, and SFPSWAP compared two different encodings.
            // It is the right mode only where DEST already holds sign+magnitude.
            //
            // Scoped to this branch because the re-encoding is only meaningful for an
            // integer threshold -- applying it to a float would reinterpret, not convert.
            const int scalar       = static_cast<int>(threshold);
            std::uint32_t sign_mag = static_cast<std::uint32_t>(scalar);
            if (scalar < 0)
            {
                // Negate in unsigned: -INT_MIN is signed overflow. INT_MIN has no
                // sign+magnitude form either -- the magnitude field is 31 bits -- so its
                // magnitude saturates and the threshold lands on -(2^31 - 1) rather than
                // round-tripping. The clamp is what makes that happen: masking instead
                // would take INT_MIN's magnitude of 0x80000000 down to 0, i.e. encode
                // sign+magnitude negative zero and clamp at 0 instead of at the low end.
                const std::uint32_t magnitude = -static_cast<std::uint32_t>(scalar);
                sign_mag                      = 0x80000000u | (magnitude > 0x7FFFFFFFu ? 0x7FFFFFFFu : magnitude);
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
