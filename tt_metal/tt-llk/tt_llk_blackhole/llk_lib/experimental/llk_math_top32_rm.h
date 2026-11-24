// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_top32_rm.h. Same Tensix instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

namespace hal_port
{
namespace top32_rm
{
// TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0)
inline constexpr std::uint32_t ELWADD_COPY = hal::fpu::get_operation<hal::fpu::Elementwise {
    .operation        = hal::fpu::ElementwiseOperation::Add,
    .broadcast        = hal::fpu::SrcBBroadcast::None,
    .dest_write       = hal::fpu::DestWriteMode::Overwrite,
    .address_modifier = ADDR_MOD_2,
    .dest_row_offset  = 0,
    .release          = hal::fpu::SourceRelease::None}>();

// TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0)
inline constexpr std::uint32_t MOVA2D_8_ROWS =
    hal::move::Transfer<hal::move::SrcA, hal::move::Dst> {
        .source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_2, .number_of_rows = hal::move::rows::Eight}
        .get_operation();
} // namespace top32_rm
} // namespace hal_port

inline void llk_math_top32_rm_configure_addrmod()
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    set_math_addr_mod<Sec::S3, MathAddrMod {}>();
    set_math_addr_mod<Sec::S0, MathAddrMod {.srca_incr = 1, .dest_incr = 1}>();
    set_math_addr_mod<Sec::S2, MathAddrMod {.srca_incr = 8, .dest_incr = 8}>();
}

template <bool is_fp32_dest_acc_en, bool is_int_fpu_en = false>
inline void llk_math_top32_rm_configure_mop([[maybe_unused]] const std::uint32_t total_rows, const std::uint32_t dst_format)
{
    namespace mop = hal::mop;

    const std::uint32_t innerloop = 2;
    const std::uint32_t outerloop = 1;

    if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) || (dst_format == to_underlying(DataFormat::UInt8)))
    {
        // ckernel_template(outerloop, innerloop, ELWADD)
        mop::program(mop::loop(outerloop, mop::loop(innerloop, hal_port::top32_rm::ELWADD_COPY)));
    }
    else
    {
        // ckernel_template(outerloop, innerloop, MOVA2D)
        mop::program(mop::loop(outerloop, mop::loop(innerloop, hal_port::top32_rm::MOVA2D_8_ROWS)));
    }
}

template <bool is_fp32_dest_acc_en, bool is_int_fpu_en = false>
inline void _llk_math_top32_rm_init_([[maybe_unused]] const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255)
{
    namespace cfg = hal::cfg;

    llk_math_top32_rm_configure_addrmod();

    const std::uint32_t num_rows = 8;
    llk_math_top32_rm_configure_mop<is_fp32_dest_acc_en, is_int_fpu_en>(num_rows, dst_format);

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f();
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool unpack_to_dest = false>
inline void _llk_math_top32_rm_(const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t num_faces)
{
    namespace rwc = hal::counters::rwc;
    namespace dst = hal::dst;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    if (unpack_to_dest && is_32bit_input(src_format, dst_format))
    {
        hal_port::math_unpack_to_dest_math_ready();
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        hal_port::math_unpack_to_dest_tile_ready();

        // Due to bug in Blackhole Tensix (more details in budabackend/#2730) when an event with side effect of clearing
        // DEST zero flags (such as Unpack-to-dest or RISC-to-dest) and a ZEROACC instruction from packer occur in the
        // same cycle, zero flags clearing is dropped. To mitigate that, we issue additional zero flag clear instruction
        // immediately after unpack tile to dest is done. RISC-to-dest event is not currently used.

        const std::uint32_t dst_format_masked = masked_data_format(dst_format);
        const int clear_fp32                  = static_cast<int>(
            dst_format_masked == static_cast<std::uint32_t>(DataFormat::Float32) || dst_format_masked == static_cast<std::uint32_t>(DataFormat::Int32) ||
            dst_format_masked == static_cast<std::uint32_t>(DataFormat::UInt32));
        const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < num_faces; i++)
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
    else
    {
        hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        // always do all 4 faces because the lower faces are set to -inf for sorting
        for (std::uint32_t i = 0; i < 4; i++)
        {
            Mop::run();
        }
        // TTI_SETRWC(CLR_AB, 0, 0, 0, 0, SET_AB)
        rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();

        hal_port::clear_dst_reg_addr();
    }
}
