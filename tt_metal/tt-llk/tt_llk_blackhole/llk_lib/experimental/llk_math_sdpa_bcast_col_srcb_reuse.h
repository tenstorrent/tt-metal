// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_sdpa_bcast_col_srcb_reuse.h. Same instruction sequence and
// access kinds as production, expressed through the HAL.

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
constexpr std::uint32_t sdpa_bcast_col_elw(const hal::fpu::SourceRelease release, const hal::fpu::DestWriteMode dest_write, const std::uint8_t addr_mod)
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
} // namespace hal_port

template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void sdpa_bcast_col_srcb_reuse_configure_mop(const std::uint32_t num_faces = 4)
{
    using hal::fpu::DestWriteMode;
    using hal::fpu::SourceRelease;
    using hal_port::sdpa_bcast_col_elw;
    using namespace hal::mop;

    LLK_ASSERT(num_faces == 2, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // Scalar broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    //
    // ADD/SUB: the execute runs this MOP twice over the same dest rows (pass 1 ends on ADDR_MOD_3, which returns dest
    // to row 0 and advances SrcB to the second column source), so the dest-accumulate bit is always set; without it
    // pass 2 would overwrite pass 1. The output tiles must therefore be zero on entry, as for MUL, whose ELWMUL
    // accumulates into dest regardless of that bit (see the init doc).
    //
    // ckernel_template(outer, num_faces, op) with last_inner = op on ADDR_MOD_2 and last_outer = op on ADDR_MOD_3. The
    // expander never issues the last_inner op with an outer count of 1, so the HAL does not patch it then (num_tiles == 1).
    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        const std::uint32_t op       = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Accumulate, ADDR_MOD_0);
        const std::uint32_t op_inner = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Accumulate, ADDR_MOD_2);
        const std::uint32_t op_outer = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Accumulate, ADDR_MOD_3);
        if constexpr (num_tiles > 1)
        {
            program(loop(num_tiles, loop(num_faces, op)).patch(last_outer, op_outer).patch(last_inner, op_inner));
        }
        else
        {
            program(loop(num_tiles, loop(num_faces, op)).patch(last_outer, op_outer));
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        const std::uint32_t op_inner = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Overwrite, ADDR_MOD_2);
        const std::uint32_t op_outer = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Overwrite, ADDR_MOD_3);
        if constexpr (high_fidelity)
        {
            // ckernel_template(num_faces, fidelity, ELWMUL(0, 0, ...)): num_faces == 2, so last_inner is issued.
            const std::uint32_t op = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::None, DestWriteMode::Overwrite, ADDR_MOD_0);
            program(loop(num_faces, loop(to_underlying(math_fidelity), op)).patch(last_outer, op_outer).patch(last_inner, op_inner));
        }
        else
        {
            const std::uint32_t op = sdpa_bcast_col_elw<eltwise_binary_type>(SourceRelease::SrcA, DestWriteMode::Overwrite, ADDR_MOD_0);
            if constexpr (num_tiles > 1)
            {
                program(loop(num_tiles, loop(num_faces, op)).patch(last_outer, op_outer).patch(last_inner, op_inner));
            }
            else
            {
                program(loop(num_tiles, loop(num_faces, op)).patch(last_outer, op_outer));
            }
        }
    }
}

template <DstSync DST, bool IS_FP32_MATH_DEST_EN, bool clear_dest = false>
inline void _llk_math_sdpa_bcast_col_srcb_reuse_preamble_()
{
    namespace sync = hal::sync;
    namespace move = hal::move;
    // MOVD2B for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    // TTI_MOVD2B(0, SRC_ZERO_OFFSET + srcb_row, ADDR_MOD_1, MOV_4_ROWS, dest_row)
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 0, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 0, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 4, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 4, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 64, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 8, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 68, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 12, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::Four}>();
    if constexpr (clear_dest)
    {
        hal_port::sdpa_clear_dest<DST, IS_FP32_MATH_DEST_EN>();
    }
}

inline void _llk_math_sdpa_bcast_col_srcb_reuse_postamble_()
{
    namespace rwc = hal::counters::rwc;
    // TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_BD)
    rwc::set<rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B}>();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    std::uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest = false>
inline void _llk_math_sdpa_bcast_col_srcb_reuse_(std::uint32_t dst_index)
{
    namespace rwc = hal::counters::rwc;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_BD)
    rwc::set<rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::Set(0)}>();

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        Mop::run();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        // Row and no broadcasted behaves similarly
        if constexpr (high_fidelity)
        {
            for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++)
            {
                Mop::run();
            }
            // TTI_SETRWC(CLR_NONE, 0 /*cr*/, 0 /*d*/, 8 /*b*/, 0 /*a*/, SET_BD)
            rwc::set<rwc::MathCounter {.src_b = rwc::Set(8), .dest = rwc::Set(0)}>();
        }
        else
        {
            Mop::run();
        }
    }

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        Mop::run();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        // Row and no broadcasted behaves similarly
        if constexpr (high_fidelity)
        {
            for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++)
            {
                Mop::run();
            }
        }
        else
        {
            Mop::run();
        }
    }
}

namespace hal_port
{
// sdpa_bcast_col_srcb_reuse_configure_addrmod for one compile-time num_faces (the ADDR_MOD_3 Dest increment depends on it).
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity, bool dense, std::uint32_t num_faces>
inline void sdpa_bcast_col_srcb_reuse_addrmod()
{
    constexpr std::uint32_t FIDELITY_INCREMENT = 1;
    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t num_dest_faces     = dense ? 2 : 4;
    constexpr std::uint32_t face_step =
        static_cast<std::uint32_t>(16 + (static_cast<std::int32_t>(num_dest_faces) - static_cast<std::int32_t>(num_faces)) * 16);

    set_sdpa_addr_mod<ADDR_MOD_1, SdpaAddrMod {}>();

    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity)
    {
        set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.dest_cr = 1, .fid_incr = FIDELITY_INCREMENT}>();
        set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {.dest_incr = 16, .dest_c2cr = 1, .fid_clr = 1}>();
        set_sdpa_addr_mod<ADDR_MOD_3, SdpaAddrMod {.dest_incr = face_step & 0x3FF, .dest_c2cr = 1, .fid_clr = 1}>();
    }
    else
    {
        set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.dest_incr = 16}>();
        set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {.dest_incr = face_step & 0x3FF}>();
        set_sdpa_addr_mod<ADDR_MOD_3, SdpaAddrMod {.srcb_incr = 8, .dest_clr = 1}>();
    }
}
} // namespace hal_port

template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity, bool dense = false>
inline void sdpa_bcast_col_srcb_reuse_configure_addrmod(const std::uint32_t num_faces)
{
    if constexpr (dense)
    {
        LLK_ASSERT(num_faces == 2, "num_faces must be 2 for dense packing");
    }
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        // One arm per num_faces: every addrmod field must be a compile-time constant for the immediate SETC16, as
        // production's TTI_SETC16 requires (it compiles only where num_faces folds to a constant).
        switch (num_faces)
        {
            case 1:
                hal_port::sdpa_bcast_col_srcb_reuse_addrmod<eltwise_binary_type, math_fidelity, dense, 1>();
                break;
            case 2:
                hal_port::sdpa_bcast_col_srcb_reuse_addrmod<eltwise_binary_type, math_fidelity, dense, 2>();
                break;
            default:
                hal_port::sdpa_bcast_col_srcb_reuse_addrmod<eltwise_binary_type, math_fidelity, dense, 4>();
                break;
        }
    }
}

/**
 * @brief Configure the math thread for the SrcB-reuse column-broadcast op: Dest = A0 op bcast_col(P1) + A1 op bcast_col(P2).
 *
 * Programs the address modifiers and the MOP that the execute runs twice over the same dest rows, once per column
 * source held in SrcB.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam num_tiles: Number of dest tiles processed per execute call.
 * @tparam math_fidelity: Math fidelity of the ELWMUL path; a high fidelity selects the fidelity-phase MOP.
 * @tparam dense: Place two-face tiles back to back in dest (requires num_faces == 2), values = <true/false>
 * @param num_faces: Faces per operand tile; the MOP currently requires 2.
 * @note Call this before @ref _llk_math_sdpa_bcast_col_srcb_reuse_preamble_ and
 *       @ref _llk_math_sdpa_bcast_col_srcb_reuse_, and run @ref _llk_math_sdpa_bcast_col_srcb_reuse_postamble_ after.
 * @note On the unpack thread, pair with @ref _llk_unpack_A_sdpa_init_ and issue
 *       @ref _llk_unpack_A_sdpa_set_srcb_dummy_valid_ before the operand unpacks.
 * @note The MOP accumulates into DEST for every op, so the output tiles must be zero on entry: a freshly released
 *       dest section, or clear_dest = true on @ref _llk_math_sdpa_bcast_col_srcb_reuse_preamble_ (which zeroes the
 *       whole dest half under SyncHalf, or all of dest under SyncFull, not only the output tiles).
 */
template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity, bool dense = false>
inline void _llk_math_sdpa_bcast_col_srcb_reuse_init_(const std::uint32_t num_faces)
{
    namespace cfg = hal::cfg;
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    sdpa_bcast_col_srcb_reuse_configure_addrmod<eltwise_binary_type, math_fidelity, dense>(num_faces);

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        sdpa_bcast_col_srcb_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>(num_faces);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0)
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    // math::reset_counters(p_setrwc::SET_ABD_F)
    hal_port::reset_counters_abd_f();
}
