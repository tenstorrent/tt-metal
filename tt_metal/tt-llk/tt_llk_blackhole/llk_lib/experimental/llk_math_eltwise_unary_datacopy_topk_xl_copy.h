// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TopK-XL copy: math datacopy MOP for A2D with a single outer loop; inner count matches total
// MOVA2D/ELWADD steps across all faces (same instruction count as stock 4×inner, different nesting).
//
// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h. Same Tensix
// instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

namespace hal_port
{
namespace topk_xl_copy
{
// TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0)
inline constexpr std::uint32_t ELWADD_COPY = hal::fpu::get_operation<hal::fpu::Elementwise {
    .operation        = hal::fpu::ElementwiseOperation::Add,
    .broadcast        = hal::fpu::SrcBBroadcast::None,
    .dest_write       = hal::fpu::DestWriteMode::Overwrite,
    .address_modifier = ADDR_MOD_0,
    .dest_row_offset  = 0,
    .release          = hal::fpu::SourceRelease::None}>();

// TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB)
inline constexpr std::uint32_t SETRWC_CLR_AB_SET_AB =
    hal::counters::rwc::MathCounter {
        .src_a = hal::counters::rwc::Set(0), .src_b = hal::counters::rwc::Set(0), .flip = hal::counters::rwc::Src::A | hal::counters::rwc::Src::B}
        .operation();
} // namespace topk_xl_copy
} // namespace hal_port

namespace ckernel
{

inline void _llk_math_topk_xl_copy_init_([[maybe_unused]] const std::uint32_t dst_format)
{
    namespace cfg = hal::cfg;
    namespace mop = hal::mop;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;

    // Address mod for unpacking from srcA to dest.
    set_math_addr_mod<cfg::Sec::S0, MathAddrMod {.srca_incr = 8, .dest_incr = 8}>();
    set_math_addr_mod<cfg::Sec::S3, MathAddrMod {}>();

    // MOP for unpacking from srcA to dest: ckernel_template(1, 8, ELWADD) + set_end_op(SETRWC(CLR_AB, ..., SET_AB))
    mop::program(mop::loop(1, mop::loop(8, hal_port::topk_xl_copy::ELWADD_COPY), hal_port::topk_xl_copy::SETRWC_CLR_AB_SET_AB));

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f();
}

// Minimal slice of _llk_math_eltwise_unary_datacopy_: A2D, no SrcB broadcast, TopK-XL MOP already programmed in init.
inline void _llk_math_topk_xl_copy_(const std::uint32_t dst_index, const std::uint32_t dst_format, const std::uint32_t elements_this_tile)
{
    namespace dst = hal::dst;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    if (elements_this_tile != 0 && is_32bit_input(dst_format, dst_format))
    {
        hal_port::math_unpack_to_dest_math_ready();
        // clear to -inf first for padding
        if (elements_this_tile < 1024)
        {
            hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
            Mop::run();
            hal_port::clear_dst_reg_addr();
        }
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        hal_port::math_unpack_to_dest_tile_ready();

        if (elements_this_tile == 1024)
        {
            const std::uint32_t dst_format_masked = masked_data_format(dst_format);
            const int clear_fp32                  = static_cast<int>(
                dst_format_masked == static_cast<std::uint32_t>(DataFormat::Float32) || dst_format_masked == static_cast<std::uint32_t>(DataFormat::Int32) ||
                dst_format_masked == static_cast<std::uint32_t>(DataFormat::UInt32));
            const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
            const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < 4; i++)
            {
                // TT_ZEROACC(p_zeroacc::CLR_16, clear_fp32, 1 /*clear zero flags*/, ADDR_MOD_3, face)
                dst::run(dst::Zero {
                    .scope        = dst::ZeroScope::Face,
                    .index        = get_dest_index_in_faces(local_tile, i),
                    .flags        = dst::ZeroFlagAction::ClearFlags,
                    .width        = clear_fp32 ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16,
                    .address_mode = ADDR_MOD_3});
            }
        }
    }
    else
    {
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        Mop::run();
        hal_port::clear_dst_reg_addr();
    }
}

} // namespace ckernel
