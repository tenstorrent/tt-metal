// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// One SFPU sweep for the hot/cold gated reduce: gate = act(scale * DST[gate]), up = up_act(scale * DST[up]),
// DST[gate] = gate * up * out_scale. The two reduced sums sit in adjacent DST tile slots with the SFPU
// base pointed at the gate slot, so a batch of experts only moves the base. The chain stays in fp32
// and rounds once, at the store; with a bf16 DST, separate scale, activation and multiply passes
// would each round in between.

#pragma once

#include <cstdint>

namespace ckernel
{
namespace sfpu
{

// Outside the thread gate because kernels name these modes in code that also compiles for UNPACK;
// only the SFPU sweep below is thread-gated.
enum class GatedReduceGate : std::uint32_t
{
    Silu,
    ClampedSilu
};
enum class GatedReduceUp : std::uint32_t
{
    Identity,
    Clamp
};

} // namespace sfpu
} // namespace ckernel

#if defined(TRISC_PACK) || defined(TRISC_MATH) || defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK)
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel
{
namespace sfpu
{

/**
 * @brief Fuse scaling, gate/up activation and multiplication of adjacent DEST tiles.
 *
 * @note Initialize the shared unary SFPU state and call sigmoid_init<false>() first.
 *       Point the SFPU base at the gate slot; the following Tile32x32 slot contains up.
 *       Only the gate slot is written. Keep both slots in the acquired DEST section.
 *       Scalar arguments are FP32 bit patterns; ClampedSilu reads limit and alpha,
 *       Clamp reads limit. Disabled scale arguments are ignored.
 */
template <GatedReduceGate GATE, GatedReduceUp UP, bool GATE_SCALE, bool UP_SCALE, bool OUT_SCALE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_gated_reduce(std::uint32_t scale_bits, std::uint32_t out_scale_bits, std::uint32_t limit_bits, std::uint32_t alpha_bits)
{
    // A 32x32 DST slot is 32 SFPU vectors; the up sum lives in the slot after the gate sum.
    constexpr int up_offset = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat g = sfpi::dst_reg[0];
        if constexpr (GATE_SCALE)
        {
            g = g * sfpi::as<sfpi::vFloat>(sfpi::vInt(scale_bits));
        }
        if constexpr (GATE == GatedReduceGate::Silu)
        {
            g = g * _sfpu_sigmoid_<is_fp32_dest_acc_en>(g);
        }
        else if constexpr (GATE == GatedReduceGate::ClampedSilu)
        {
            sfpi::vFloat limit = sfpi::as<sfpi::vFloat>(sfpi::vInt(limit_bits));
            sfpi::vFloat alpha = sfpi::as<sfpi::vFloat>(sfpi::vInt(alpha_bits));
            g                  = sfpi::min(g, limit);
            g                  = g * _sfpu_sigmoid_<is_fp32_dest_acc_en>(alpha * g);
        }

        sfpi::vFloat u = sfpi::dst_reg[up_offset];
        if constexpr (UP_SCALE)
        {
            u = u * sfpi::as<sfpi::vFloat>(sfpi::vInt(scale_bits));
        }
        if constexpr (UP == GatedReduceUp::Clamp)
        {
            sfpi::vFloat limit = sfpi::as<sfpi::vFloat>(sfpi::vInt(limit_bits));
            u                  = sfpi::clamp(u, sfpi::setsgn(limit, 1), limit);
        }

        sfpi::vFloat result = g * u;
        if constexpr (OUT_SCALE)
        {
            result = result * sfpi::as<sfpi::vFloat>(sfpi::vInt(out_scale_bits));
        }
        // A bf16 DST store truncates, one ulp low on about half the outputs; round to nearest instead.
        if constexpr (!is_fp32_dest_acc_en)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
#endif // SFPU math/pack threads
