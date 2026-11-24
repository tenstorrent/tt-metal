// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_sdpa_custom_mm.h. Same instruction sequence and access
// kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
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
using namespace ckernel::math;

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

template <bool transpose>
inline void sdpa_custom_mm_configure_addrmod()
{
    using hal_port::SdpaAddrMod;
    using hal_port::set_sdpa_addr_mod;
    constexpr std::uint32_t face_r_dim           = 8;
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR  = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR  = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = (1024 - face_r_dim);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR  = 2 * face_r_dim;

    set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.srca_incr = ADDR_MOD_0_SRCA_INCR, .dest_incr = face_r_dim}>();
    set_sdpa_addr_mod<ADDR_MOD_1, SdpaAddrMod {.srca_incr = ADDR_MOD_1_SRCA_INCR, .srcb_incr = 16, .dest_incr = ADDR_MOD_1_DEST_INCR}>();
    set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = ADDR_MOD_2_DEST_INCR, .dest_cr = 1}>();
    set_sdpa_addr_mod<ADDR_MOD_3, SdpaAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_clr = 1}>();
    set_sdpa_addr_mod<ADDR_MOD_4, SdpaAddrMod {.srcb_incr = 1, .dest_incr = 8}>();
    set_sdpa_addr_mod<ADDR_MOD_7, SdpaAddrMod {}>();
}

namespace hal_port
{
// TTI_MVMUL(clr, 0, addr_mod, 0)
template <hal::fpu::SourceRelease Release, std::uint8_t AddrMod>
inline constexpr hal::fpu::MatrixMultiply sdpa_custom_mm_mvmul {
    .broadcast = hal::fpu::SrcBRowBroadcast::None, .address_modifier = AddrMod, .dest_row_offset = 0, .release = Release};

inline constexpr hal::replay::BufferRange sdpa_custom_mm_replay {ckernel::math::replay_buf_offset, 3};
} // namespace hal_port

inline void sdpa_custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    namespace fpu = hal::fpu;
    using fpu::SourceRelease;
    using hal_port::sdpa_custom_mm_mvmul;

    // load_replay_buf(replay_buf_offset, 3, ...): record only.
    hal::replay::record<hal_port::sdpa_custom_mm_replay>(
        []
        {
            fpu::run<sdpa_custom_mm_mvmul<SourceRelease::None, ADDR_MOD_0>>(); // 0
            fpu::run<sdpa_custom_mm_mvmul<SourceRelease::None, ADDR_MOD_1>>(); // 16
            fpu::run<sdpa_custom_mm_mvmul<SourceRelease::None, ADDR_MOD_0>>(); // 0
        });

    constexpr std::uint32_t mvmul_base      = hal::replay::get_operation<hal_port::sdpa_custom_mm_replay>();
    constexpr std::uint32_t mvmul_end_tile  = fpu::get_operation<sdpa_custom_mm_mvmul<SourceRelease::SrcA, ADDR_MOD_2>>();
    constexpr std::uint32_t mvmul_end_block = fpu::get_operation<sdpa_custom_mm_mvmul<SourceRelease::Both, ADDR_MOD_3>>();

    // ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile) with both last-iteration ops mvmul_end_block. With an outer
    // count of 1 the expander never issues the last_inner op, which the HAL therefore does not patch (it keeps
    // mvmul_end_tile in that unread word).
    using namespace hal::mop;
    program(loop(1, loop(ct_dim, mvmul_base, mvmul_end_tile)).patch(last_outer, mvmul_end_block));
}

/**
 * @brief Configure the math thread for sdpa_custom_mm: programs address mods and the MVMUL MOP.
 *
 * @tparam transpose: Transpose in1 faces during the multiply.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}; currently unused (the
 *        address mods are fixed), kept for symmetry with @ref _llk_unpack_AB_custom_mm_init_.
 * @param ct_dim: Number of output column tiles.
 * @note Establishes the operand-driven default Src zero-substitution state, which @ref _llk_math_sdpa_custom_mm_
 *       asserts under LLK asserts. Re-run this init after any math op that leaves the flag at keep (a copy init does).
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_custom_mm_init_.
 */
template <bool transpose = false>
inline void _llk_math_sdpa_custom_mm_init_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1)
{
    sdpa_custom_mm_configure_addrmod<transpose>();
    sdpa_custom_mm_configure_mop(operandB_face_r_dim, ct_dim);

    // math::reset_counters(p_setrwc::SET_ABD_F)
    hal_port::reset_counters_abd_f();

    // A preceding datacopy leaves the Src zero flag at keep; MVMUL needs the operand-driven value.
    hal_port::_configure_default_zero_flag_state_();
}

inline void _llk_math_sdpa_custom_mm_mask_dest_(const std::uint32_t dst_index, const std::uint32_t ct_dim, bool mask_chunk = false)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace move = hal::move;
    namespace rwc  = hal::counters::rwc;
    namespace dst  = hal::dst;

    // Zero Dest
    std::uint32_t dst_offset = dst_index + get_dest_buffer_base();
    // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_offset)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dst_offset);
    if (mask_chunk)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcBValid>();
        // Zero Dest
        for (std::uint32_t i = 0; i < ct_dim * 2; i++)
        {
            // TTI_MOVB2D(0, SRC_ZERO_OFFSET, ADDR_MOD_4, MOV_8_ROW_BRCST, 0)
            move::run<move::Transfer<move::SrcB, move::Dst> {
                .source_row      = p_movb2d::SRC_ZERO_OFFSET,
                .destination_row = 0,
                .address_mode    = ADDR_MOD_4,
                .number_of_rows  = move::rows::Eight,
                .broadcast       = move::Broadcast::Row}>();
        }
        // TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD)
        rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B}>();
    }
    else
    {
        // Zero Dest
        // ZEROACC auto-detects the DEST bank, so its tile index must be bank-relative (dst_index, not
        // dst_offset which already includes the bank base -> would double-count the bank on bank 1).
        std::uint32_t dst_face = dst_index / 16;
        // A tile is 2 faces of 8x16, so clearing 16 rows per tile is equivalent to clearing the tile
        for (std::uint32_t i = 0; i < ct_dim; i++)
        {
            // TT_ZEROACC(CLR_16, 0, 0, ADDR_MOD_7, dst_face)
            dst::run(dst::Zero {.scope = dst::ZeroScope::Face, .index = dst_face, .address_mode = ADDR_MOD_7});
            dst_face++;
        }
    }
}

// Call _llk_math_sdpa_custom_mm_init_ first; re-run it after any op that leaves the Src zero flag at keep.
template <std::uint32_t signal_granularity = 1>
inline void _llk_math_sdpa_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const bool mask_chunk      = false)
{
    namespace fpu = hal::fpu;
    using fpu::SourceRelease;
    using hal_port::sdpa_custom_mm_mvmul;
    using Mop = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    static_assert(signal_granularity >= 1, "signal_granularity must be >= 1");
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "sdpa_custom_mm: Src zero-substitution flag does not hold the operand-driven value; an op between the init and "
        "this call left it at keep, so denormal Src operands would not be flushed");
    LLK_ASSERT(ct_dim % signal_granularity == 0, "ct_dim must be divisible by signal_granularity for FPU->SFPU signal counts to balance");
    // dst offset initialized by _llk_math_sdpa_custom_mm_mask_dest_
    _llk_math_sdpa_custom_mm_mask_dest_(dst_index, ct_dim, mask_chunk);

    for (std::uint32_t i = 0; i < kt_dim - 1; i++)
    {
        Mop::run();
    }
    if (signal_granularity == ct_dim)
    {
        Mop::run();
        hal_port::sdpa_post_fpu_sfpu();
    }
    else
    {
        for (std::uint32_t i = 0; i < ct_dim - 1; i++)
        {
            hal::replay::run<hal_port::sdpa_custom_mm_replay>();
            fpu::run<sdpa_custom_mm_mvmul<SourceRelease::SrcA, ADDR_MOD_2>>();
            if (i % signal_granularity == signal_granularity - 1)
            {
                hal_port::sdpa_post_fpu_sfpu();
            }
        }
        hal::replay::run<hal_port::sdpa_custom_mm_replay>();
        fpu::run<sdpa_custom_mm_mvmul<SourceRelease::Both, ADDR_MOD_3>>();
        hal_port::sdpa_post_fpu_sfpu();
    }
}
