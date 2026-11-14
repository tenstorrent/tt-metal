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

// A 32-bit DEST in half sync keeps a SETRWC per phase: the counter clearing ADDR_MOD_6 would slow the concurrent pack.
template <DstSync Dst, bool is_fp32_dest_acc_en>
constexpr bool mul_scalar_block_phase_clears_counters = !(is_fp32_dest_acc_en && Dst == DstSync::SyncHalf);

/**
 * @brief Program the address modifiers of the scalar block multiply.
 *
 * @note Other inits may reprogram ADDR_MOD_5 to 7, so a kernel that runs one in between re-runs this.
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

    if constexpr (is_high_fidelity(math_fidelity) && mul_scalar_block_phase_clears_counters<Dst, is_fp32_dest_acc_en>)
    {
        // A phase ends back at the tile start; a tile ends at the next tile, on fidelity phase 0.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1},
            .srcb     = {.incr = 0, .clr = 1},
            .dest     = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = 1},
        }
            .set(ADDR_MOD_6);
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1},
            .srcb     = {.incr = 0, .clr = 1},
            .dest     = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_5);
    }
    else if constexpr (is_high_fidelity(math_fidelity))
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
 * @brief Multiply block_size whole 32x32 tiles in SrcA by the SrcB scalar into DEST tiles from dst_index.
 *
 * @note Call the init with the same template arguments first. Above LoFi this overwrites the FPU half of the replay
 *       buffer, so an op that recorded there re-runs its init after this.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_mul_scalar_block_(const std::uint32_t dst_index, const std::uint32_t block_size)
{
    constexpr std::uint32_t kOpsPerTile = 8;
    constexpr std::uint32_t kPhases     = is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1;

    if constexpr (kPhases > 1 && mul_scalar_block_phase_clears_counters<Dst, is_fp32_dest_acc_en>)
    {
        // TRISC1 cannot issue the multiplies as fast as the FPU runs them: the first tile records a fidelity phase and
        // the last phase, the others replay them. TTI_REPLAY, since lltt makes the compiler reload the caller's values.
        constexpr std::uint32_t kBase = math::replay_buf_offset;
        const auto phase              = []
        {
            for (std::uint32_t op = 0; op < kOpsPerTile - 1; ++op)
            {
                TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
            }
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_6, 0);
        };
        const auto last_phase = []
        {
            for (std::uint32_t op = 0; op < kOpsPerTile - 1; ++op)
            {
                TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
            }
            TTI_ELWMUL(p_setrwc::CLR_A, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_5, 0);
        };
        const auto record_tile = [&]
        {
            TTI_REPLAY(kBase, kOpsPerTile, 1, 1);
            phase();
            for (std::uint32_t p = 1; p < kPhases - 1; ++p)
            {
                TTI_REPLAY(kBase, kOpsPerTile, 0, 0);
            }
            TTI_REPLAY(kBase + kOpsPerTile, kOpsPerTile, 1, 1);
            last_phase();
        };
        const auto replay_tile = []
        {
            for (std::uint32_t p = 1; p < kPhases - 1; ++p)
            {
                TTI_REPLAY(kBase, kOpsPerTile, 0, 0);
            }
            TTI_REPLAY(kBase, 2 * kOpsPerTile, 0, 0);
        };
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
        if (block_size > 0)
        {
#if defined(ENABLE_GATHERING)
            disable_gathering();
#endif
            record_tile();
#if defined(ENABLE_GATHERING)
            enable_gathering();
#endif
            for (std::uint32_t i = 1; i < block_size; ++i)
            {
                replay_tile();
            }
        }
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        math::clear_dst_reg_addr();
        return;
    }

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
