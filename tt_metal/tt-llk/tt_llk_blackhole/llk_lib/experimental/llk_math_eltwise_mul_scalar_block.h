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

// A 32-bit DEST in half sync keeps a SETRWC per phase: there the packer reads one half while the multiply fills the
// other, and the denser phase boundary of the counter clearing ADDR_MOD_6 slows the pack.
template <DstSync Dst, bool is_fp32_dest_acc_en>
constexpr bool mul_scalar_block_phase_clears_counters = !(is_fp32_dest_acc_en && Dst == DstSync::SyncHalf);

/**
 * @brief Program the address modifiers of the scalar block multiply and reset the counters.
 *
 * ADDR_MOD_7 steps SrcA and DEST by one 8-row group per multiply; above LoFi ADDR_MOD_6 ends a phase on its last
 * multiply, stepping the fidelity and, outside a 32-bit DEST in half sync, going back to the tile start. Other inits
 * may reprogram either slot, so a kernel that runs one in between re-runs this.
 *
 * @tparam Dst: DEST sync mode of the kernel.
 * @tparam is_fp32_dest_acc_en: Whether DEST holds 32-bit values.
 * @tparam math_fidelity: Fidelity phases of the multiply, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_scalar_block_init_.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity = MathFidelity::LoFi>
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
        constexpr std::uint8_t clr = mul_scalar_block_phase_clears_counters<Dst, is_fp32_dest_acc_en> ? 1 : 0;
        addr_mod_t {
            .srca     = {.incr = 0, .clr = clr},
            .srcb     = {.incr = 0, .clr = clr},
            .dest     = {.incr = 0, .clr = clr},
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
 * Above LoFi the 8 multiplies per tile run once per fidelity phase, the last one starting the next phase through
 * ADDR_MOD_6, so the products are those of the standard binary multiply at the same math_fidelity.
 *
 * @tparam Dst: DEST sync mode of the kernel, must match the init.
 * @tparam is_fp32_dest_acc_en: Whether DEST holds 32-bit values, must match the init.
 * @tparam math_fidelity: Fidelity phases of the multiply, must match the init.
 * @param dst_index: DEST tile of the first product.
 * @param block_size: Number of tiles in the block.
 * @note Call @ref _llk_math_eltwise_mul_scalar_block_init_ with the same template arguments before this function;
 *       @ref _llk_unpack_AB_scalar_block_ feeds it on the unpack thread.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity = MathFidelity::LoFi>
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
        else if constexpr (mul_scalar_block_phase_clears_counters<Dst, is_fp32_dest_acc_en>)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
            for (std::uint32_t phase = 0; phase < kPhases; ++phase)
            {
                for (std::uint32_t op = 0; op < kOpsPerTile - 1; ++op)
                {
                    TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
                }
                TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_6, 0);
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB_F);
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
