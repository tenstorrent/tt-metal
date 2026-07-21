// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TopK-XL copy: math datacopy MOP for A2D with a single outer loop; inner count matches total
// MOVA2D/ELWADD steps across all faces (same instruction count as stock 4×inner, different nesting).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"

namespace ckernel
{

inline void _llk_math_topk_xl_copy_init_([[maybe_unused]] const std::uint32_t dst_format)
{
    // Address mod for unpacking from srcA to dest.
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    // MOP for unpacking from srcA to dest.
    ckernel_template tmp(1, 8, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Minimal slice of _llk_math_eltwise_unary_datacopy_: A2D, no SrcB broadcast, TopK-XL MOP already programmed in init.
inline void _llk_math_topk_xl_copy_(const std::uint32_t dst_index, const std::uint32_t dst_format, const std::uint32_t elements_this_tile)
{
    if (is_32bit_input(dst_format, dst_format))
    {
        math_unpack_to_dest_math_ready();
        // clear to -inf first for padding
        if (elements_this_tile < 1024)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
            ckernel_template::run();
            math::clear_dst_reg_addr();
        }
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        math::math_unpack_to_dest_tile_ready();

        if (elements_this_tile == 1024)
        {
            const std::uint32_t dst_format_masked = masked_data_format(dst_format);
            const int clear_fp32                  = static_cast<int>(
                dst_format_masked == (std::uint32_t)DataFormat::Float32 || dst_format_masked == (std::uint32_t)DataFormat::Int32 ||
                dst_format_masked == (std::uint32_t)DataFormat::UInt32);
            const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
            const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < 4; i++)
            {
                TT_ZEROACC(p_zeroacc::CLR_16, clear_fp32, 1 /*clear zero flags*/, ADDR_MOD_3, get_dest_index_in_faces(local_tile, i));
            }
        }
    }
    else
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        ckernel_template::run();
        math::clear_dst_reg_addr();
    }
}

} // namespace ckernel
