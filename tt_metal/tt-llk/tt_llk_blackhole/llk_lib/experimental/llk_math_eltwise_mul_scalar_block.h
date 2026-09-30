// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace ckernel
{

/**
 * @brief Program the address modifiers of the scalar block multiply and reset the counters.
 *
 * ADDR_MOD_7 steps SrcA and DEST by one 8-row group per multiply. Above LoFi the block is multiplied once per
 * fidelity phase and ADDR_MOD_6 carries the phase step: the last multiply of a phase advances the fidelity counter
 * (its SrcA and DEST steps do not matter, the SETRWC that opens the next phase resets both counters). At LoFi only
 * ADDR_MOD_7 is programmed, so a LoFi kernel is what it was before the fidelity template existed. ADDR_MOD_6 is also
 * programmed by the matmul and the SFPU inits, as ADDR_MOD_7 is by the SFPU init: a kernel that runs one of those
 * between this init and the block multiply re-runs this init.
 *
 * @tparam math_fidelity: Fidelity phases of the multiply, values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_mul_scalar_block_init_()
{
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);

    if constexpr (is_high_fidelity(math_fidelity))
    {
        addr_mod_t {
            .srca     = {.incr = 0},
            .srcb     = {.incr = 0},
            .dest     = {.incr = 0},
            .fidelity = {.incr = 1},
        }
            .set(ADDR_MOD_6);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Multiply block_size whole 32x32 tiles in SrcA by the scalar in SrcB into DEST tiles dst_index onward.
 *
 * Per tile, 8 ELWMUL with the scalar broadcast cover the 64 DEST rows. Above LoFi the 8 multiplies run once per
 * fidelity phase (2, 3 or 4 phases), so the products carry the mantissa bits the phases add, as the standard
 * binary multiply does at the same math_fidelity: each phase opens with the SrcA and DEST counters back at the tile
 * start, the eighth multiply of a phase steps the fidelity counter through ADDR_MOD_6, and the SETRWC that closes
 * the tile resets it for the next tile.
 *
 * @tparam math_fidelity: Fidelity phases of the multiply, must match the init.
 * @param dst_index: DEST tile of the first product.
 * @param block_size: Number of tiles in the block.
 */
template <MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_mul_scalar_block_(const std::uint32_t dst_index, const std::uint32_t block_size)
{
    constexpr std::uint32_t kOpsPerTile = 8;
    constexpr std::uint32_t kPhases     = is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1;

    for (std::uint32_t i = 0; i < block_size; ++i)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + i);
        if constexpr (kPhases == 1)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
            for (std::uint32_t op = 0; op < kOpsPerTile; ++op)
            {
                TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else
        {
            for (std::uint32_t phase = 0; phase < kPhases; ++phase)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
                for (std::uint32_t op = 0; op < kOpsPerTile - 1; ++op)
                {
                    TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
                }
                TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_6, 0);
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB_F);
        }
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    math::clear_dst_reg_addr();
}

} // namespace ckernel
