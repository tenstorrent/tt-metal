// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_sdpa_bcast_col_srca_srcb_reuse.h. Same instruction sequence
// and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

// One ADDR_MOD slot and its addr_mod_t::set(n) equivalent, shared by the four SDPA math headers. The block is identical
// in each of them; the guard keeps a translation unit that includes several to one definition.
#ifndef HAL_PORT_SDPA_ADDR_MOD_DEFINED_
#define HAL_PORT_SDPA_ADDR_MOD_DEFINED_

namespace hal_port
{

// One ADDR_MOD slot, in the units of ckernel::addr_mod_t (Dest increment already in its 10-bit encoding).
struct SdpaAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t dest_c2cr = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// addr_mod_t::set(n): three SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <std::uint8_t Slot, SdpaAddrMod M>
inline void set_sdpa_addr_mod()
{
    namespace cfg        = hal::cfg;
    constexpr cfg::Sec S = static_cast<cfg::Sec>(Slot);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c2cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU): STALLWAIT(STALL_SYNC, MATH) then SEMPOST.
inline void sdpa_post_fpu_sfpu()
{
    namespace sync = hal::sync;
    static_assert(ckernel::semaphore::FPU_SFPU == 0, "FPU_SFPU is semaphore 0");
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::post<sync::Semaphore::S0>();
}

// TTI_ZEROACC(CLR_ALL / CLR_HALF, IS_FP32_MATH_DEST_EN, 0, ADDR_MOD_1, ...): the SDPA preambles' optional dest clear.
template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
inline void sdpa_clear_dest()
{
    namespace dst                  = hal::dst;
    constexpr dst::DestWidth width = IS_FP32_MATH_DEST_EN ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;
    if constexpr (DST == DstSync::SyncFull)
    {
        dst::run<dst::Zero {.scope = dst::ZeroScope::All, .index = 0, .width = width, .address_mode = ADDR_MOD_1}>();
    }
    else
    {
        static_assert(DST == DstSync::SyncHalf);
        dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = ckernel::dest_offset_id, .width = width, .address_mode = ADDR_MOD_1});
    }
}

} // namespace hal_port
#endif // HAL_PORT_SDPA_ADDR_MOD_DEFINED_

namespace hal_port
{
// TT_OP_ELWADD / ELWSUB / ELWMUL(clr, accumulate, SRCB_BCAST_COL, addr_mod, 0)
template <EltwiseBinaryType eltwise_binary_type>
constexpr std::uint32_t sdpa_bcast_col_srca_elw(const hal::fpu::SourceRelease release, const hal::fpu::DestWriteMode dest_write, const std::uint8_t addr_mod)
{
    constexpr hal::fpu::ElementwiseOperation operation = (eltwise_binary_type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
                                                         : (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                                                                              : hal::fpu::ElementwiseOperation::Multiply;
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = operation,
        .broadcast        = hal::fpu::SrcBBroadcast::Column,
        .dest_write       = dest_write,
        .address_modifier = addr_mod,
        .dest_row_offset  = 0,
        .release          = release});
}

inline constexpr hal::replay::BufferRange sdpa_bcast_col_srca_d2a_replay {ckernel::math::replay_buf_offset, 2};

// TTI_MOVD2A(0, MATH_HALO_ROWS + Row, ADDR_MOD_1, MOV_4_ROWS, Row): Dest rows Row..Row+3 -> SrcA rows Row..Row+3.
template <std::uint32_t Row>
inline constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcA> sdpa_bcast_col_srca_d2a {
    .source_row = Row, .destination_row = p_mova2d::MATH_HALO_ROWS + Row, .address_mode = ADDR_MOD_1, .number_of_rows = hal::move::rows::Four};
} // namespace hal_port

template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void sdpa_bcast_col_srca_srcb_reuse_configure_mop(const std::uint32_t num_faces = 4, const std::uint32_t acc_to_dest = 0)
{
    using hal::fpu::DestWriteMode;
    using hal::fpu::SourceRelease;
    using hal_port::sdpa_bcast_col_srca_elw;
    using namespace hal::mop;
    namespace move = hal::move;

    LLK_ASSERT(num_faces == 2, "num_faces must be 2");
    LLK_ASSERT(acc_to_dest <= 1, "acc_to_dest is the one-bit dest-accumulate field");
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // Scalar broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    // load_replay_buf<NoExec>(replay_buf_offset, 2, ...): record only.
    hal::replay::record<hal_port::sdpa_bcast_col_srca_d2a_replay>(
        []
        {
            move::run<hal_port::sdpa_bcast_col_srca_d2a<0>>();
            move::run<hal_port::sdpa_bcast_col_srca_d2a<4>>();
        });

    constexpr std::uint32_t d2a_instr = hal::replay::get_operation<hal_port::sdpa_bcast_col_srca_d2a_replay>();
    const DestWriteMode accumulate    = acc_to_dest ? DestWriteMode::Accumulate : DestWriteMode::Overwrite;
    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        // ckernel_template(1, num_faces, d2a_instr, ELW(CLR_NONE, acc_to_dest, ..., ADDR_MOD_0)) with last_outer on ADDR_MOD_2
        const std::uint32_t op       = sdpa_bcast_col_srca_elw<eltwise_binary_type>(SourceRelease::None, accumulate, ADDR_MOD_0);
        const std::uint32_t op_outer = sdpa_bcast_col_srca_elw<eltwise_binary_type>(SourceRelease::None, accumulate, ADDR_MOD_2);
        program(loop(1, loop(num_faces, d2a_instr, op)).patch(last_outer, op_outer));
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        const std::uint32_t op       = sdpa_bcast_col_srca_elw<eltwise_binary_type>(SourceRelease::None, DestWriteMode::Overwrite, ADDR_MOD_0);
        const std::uint32_t op_inner = sdpa_bcast_col_srca_elw<eltwise_binary_type>(SourceRelease::None, DestWriteMode::Overwrite, ADDR_MOD_2);
        if constexpr (high_fidelity)
        {
            // ckernel_template(num_faces, fidelity, ELWMUL(ADDR_MOD_0)) with start op d2a_instr, last_inner on ADDR_MOD_2
            // and last_outer on ADDR_MOD_3 (num_faces == 2, so last_inner is issued).
            const std::uint32_t op_outer = sdpa_bcast_col_srca_elw<eltwise_binary_type>(SourceRelease::None, DestWriteMode::Overwrite, ADDR_MOD_3);
            program(loop(num_faces, d2a_instr, loop(to_underlying(math_fidelity), op)).patch(last_outer, op_outer).patch(last_inner, op_inner));
        }
        else
        {
            // ckernel_template(1, num_faces, d2a_instr, ELWMUL(ADDR_MOD_0)) with last_outer on ADDR_MOD_2
            program(loop(1, loop(num_faces, d2a_instr, op)).patch(last_outer, op_inner));
        }
    }
}

template <DstSync DST, bool IS_FP32_MATH_DEST_EN, bool clear_dest = false>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_(std::uint32_t isrc)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace move = hal::move;
    // MOVD2B for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    sync::wait::stall<
        sync::StallTarget::Math,
        sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcAValid | sync::StallCondition::SrcBValid>();
    std::uint32_t src_index = isrc + get_dest_buffer_base();
    // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(src_index);
    // TTI_MOVD2B(0, SRC_ZERO_OFFSET + row, ADDR_MOD_1, MOV_4_ROWS, row)
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 0, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 0, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 4, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 4, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    if constexpr (clear_dest)
    {
        hal_port::sdpa_clear_dest<DST, IS_FP32_MATH_DEST_EN>();
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    std::uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest      = false,
    bool skip_signalling = false,
    std::uint32_t output_granularity>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_(std::uint32_t dst_index)
{
    namespace cfg = hal::cfg;
    namespace rwc = hal::counters::rwc;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    static_assert(output_granularity >= 1, "output_granularity must be >= 1");
    static_assert(num_tiles % output_granularity == 0, "num_tiles must be divisible by output_granularity");

    // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + base)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dst_index + get_dest_buffer_base());
    // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0)}>();

    // output_granularity tiles are computed per FPU->SFPU signal. output_granularity == 1
    // recovers the per-tile signalling cadence.
    for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num += output_granularity)
    {
        for (std::uint32_t g = 0; g < output_granularity; g++)
        {
            Mop::run();
        }
        if constexpr (!skip_signalling)
        {
            hal_port::sdpa_post_fpu_sfpu();
        }
    }
    // TTI_SETRWC(CLR_AB, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();
}

template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity, std::uint32_t FIDELITY_INCREMENT>
inline void sdpa_bcast_col_srca_srcb_reuse_configure_addrmod([[maybe_unused]] const std::uint32_t num_faces)
{
    using hal_port::SdpaAddrMod;
    using hal_port::set_sdpa_addr_mod;
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        set_sdpa_addr_mod<ADDR_MOD_1, SdpaAddrMod {}>();

        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity)
        {
            set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.dest_cr = 1, .fid_incr = FIDELITY_INCREMENT}>();
            set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {.dest_incr = 8, .dest_c2cr = 1, .fid_clr = 1}>();
            set_sdpa_addr_mod<ADDR_MOD_3, SdpaAddrMod {.dest_incr = 8, .dest_c2cr = 1, .fid_clr = 1}>();
        }
        else
        {
            set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.dest_incr = 8}>();
            set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {.dest_incr = 8}>();
        }
    }
}

template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity, bool skip_addrmod = false>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_init_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr std::uint32_t math_fidelity_increment = 1;

    if constexpr (!skip_addrmod)
    {
        sdpa_bcast_col_srca_srcb_reuse_configure_addrmod<eltwise_binary_type, math_fidelity, math_fidelity_increment>(num_faces);
    }

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        sdpa_bcast_col_srca_srcb_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>(num_faces, acc_to_dest);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0)
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    // math::reset_counters(p_setrwc::SET_ABD_F)
    hal_port::reset_counters_abd_f();
}
