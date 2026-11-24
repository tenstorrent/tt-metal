// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_sdpa.h. Same instruction sequence and access kinds as
// production, expressed through the HAL, except the HAL-GAPs marked below.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <
    std::uint32_t num_tiles,
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_sdpa_mop_config_(
    const bool transpose_of_faces, const std::uint32_t num_faces, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        (((BType == BroadcastType::NONE) && (!acc_to_dest) && (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE))),
        "Not supported configuration when unpacking to dest!");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    static constexpr std::uint32_t unpack_srca = hal::unpack::get_operation<hal::unpack::DataTransfer {
        .engine     = hal::unpack::Engine::Unpacker0,
        .increments = {.channel0 = {.z = 1}},
        .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    }>();
    static_assert(unpack_srca == TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), "SrcA unpack encoding");

    using namespace hal::mop;
    if (num_faces == 1)
    {
        // ckernel_template(1, num_tiles, unpack_srca)
        program(loop(1, loop(num_tiles, unpack_srca)));
    }
    else
    {
        // ckernel_template(1, num_tiles * num_faces / 2, unpack_srca, unpack_srca)
        const std::uint32_t innerloop = num_tiles * num_faces / 2;
        program(loop(1, loop(innerloop, unpack_srca, unpack_srca)));
    }
}

template <
    std::uint32_t num_tiles,
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_sdpa_init_(
    const std::uint32_t transpose_of_faces          = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    namespace cfg = hal::cfg;
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // Set transpose register to prevent state pollution: cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face_16x16_transpose);

    // TODO NC: Find out why we need to disable src zero flags for uint16 dst format #960
    // bool disable_src_zero_flag_val = disable_src_zero_flag || (static_cast<uint>(unpack_dst_format) ==
    // static_cast<uint>(DataFormat::UInt16));
    // cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable_src_zero_flag_val ? 1 : 0);

    constexpr std::uint32_t UNP_SEL = (BType == BroadcastType::NONE) ? p_setadc::UNP_A : p_setadc::UNP_B;
    hal_port::config_unpacker_x_end<UNP_SEL>(face_r_dim);

    _llk_unpack_A_sdpa_mop_config_<num_tiles, BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}

inline void _llk_unpack_A_sdpa_set_srcb_dummy_valid_()
{
    namespace sync = hal::sync;
    // TTI_STALLWAIT(STALL_UNPACK, UNPACK)
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    // Stall_Clr_Cntrl=1 so the publication waits on the bank it clears.
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
    // (UNP_ZEROSRC, SET_DVALID, Stall_Clr_Cntrl) has no HAL descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

inline void _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_()
{
    namespace sync = hal::sync;
    // TTI_STALLWAIT(STALL_UNPACK, UNPACK)
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    // Stall_Clr_Cntrl=1 on each: per-instruction, so the SrcB publication flipping the
    // unpacker's bank pointer cannot leave the SrcA publication relying on a stale gate.
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
    // (UNP_ZEROSRC, SET_DVALID, Stall_Clr_Cntrl) has no HAL descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}
