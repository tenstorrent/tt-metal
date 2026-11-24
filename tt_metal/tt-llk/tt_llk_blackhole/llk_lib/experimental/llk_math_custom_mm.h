// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_custom_mm.h. Same instruction sequence and access kinds
// as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "experimental/llk_custom_mm_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;
using namespace ckernel::math;

namespace hal_port
{

// TT_OP_MVMUL(clear_dvalid, 0, addr_mode, 0)
template <std::uint8_t AddrMode, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline constexpr std::uint32_t custom_mm_mvmul_op = hal::fpu::get_operation<hal::fpu::MatrixMultiply {.address_modifier = AddrMode, .release = Release}>();

// TTI_MVMUL(clear_dvalid, 0, addr_mode, 0)
template <std::uint8_t AddrMode, hal::fpu::SourceRelease Release = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void custom_mm_mvmul()
{
    hal::fpu::run<hal::fpu::MatrixMultiply {.address_modifier = AddrMode, .release = Release}>();
}

static_assert(custom_mm_mvmul_op<ADDR_MOD_0> == TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0));
static_assert(custom_mm_mvmul_op<ADDR_MOD_2, hal::fpu::SourceRelease::SrcA> == TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0));
static_assert(custom_mm_mvmul_op<ADDR_MOD_3, hal::fpu::SourceRelease::Both> == TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0));

// TTI_ELWADD(clear_dvalid, 1 /*dest_accum_en*/, p_elwise::SRCB_NO_BCAST, addr_mode, 0)
template <std::uint8_t AddrMode, hal::fpu::SourceRelease Release>
inline __attribute__((always_inline)) void custom_mm_elwadd_acc()
{
    constexpr hal::fpu::Elementwise op {
        .operation        = hal::fpu::ElementwiseOperation::Add,
        .broadcast        = hal::fpu::SrcBBroadcast::None,
        .dest_write       = hal::fpu::DestWriteMode::Accumulate,
        .address_modifier = AddrMode,
        .release          = Release};
    static_assert(
        hal::fpu::get_operation<op>() == TT_OP_ELWADD(static_cast<std::uint32_t>(Release), 1, p_elwise::SRCB_NO_BCAST, AddrMode, 0), "ELWADD encoding");
    hal::fpu::run<op>();
}

// TTI_MOVD2B(0, srcb_row, addr_mode, p_movd2a::MOV_4_ROWS or MOV_1_ROW, dest_row): Dest rows -> SrcB rows.
template <std::uint32_t SrcBRow, std::uint32_t DestRow, std::uint8_t AddrMode, std::uint16_t Rows = 4>
inline __attribute__((always_inline)) void custom_mm_movd2b()
{
    namespace move = hal::move;
    constexpr move::Transfer<move::Dst, move::SrcB> t {
        .source_row = DestRow, .destination_row = SrcBRow, .address_mode = AddrMode, .number_of_rows = move::RowCount {Rows}};
    static_assert(
        t.get_operation() == TT_OP_MOVD2B(0, SrcBRow, AddrMode, (Rows == 4 ? p_movd2a::MOV_4_ROWS : p_movd2a::MOV_1_ROW), DestRow), "MOVD2B encoding");
    TTI_INSN(t.get_operation());
}

// TTI_ZEROSRC(0, 1, 0, 1): zero the math-side current SrcA bank.
inline __attribute__((always_inline)) void custom_mm_zero_srca()
{
    constexpr std::uint32_t op = hal::src<hal::SourceMask::SrcA>.fill_operation<hal::src_ops::BankScope::Current, hal::src_ops::BankOwner::Math>();
    static_assert(op == TT_OP_ZEROSRC(0, 1, 0, 1), "ZEROSRC encoding");
    TTI_INSN(op);
}

} // namespace hal_port

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32] or [32, 16] ([16, 32] with transpose), giving [{1, 2, 4, 8}, 32] or [{1, 2, 4, 8}, 16] output tiles
// rt_dim: 1
// ct_dim: any integer from 1 to 34, as long as the ct_dim output tiles fit in the DEST sync region
//         (custom_mm_tile_rows of the CustomMmDestLayout each)
// kt_dim: any integer of 1 or more
// fidelity: LoFi only
// throttle: not supported

/**
 * @brief Whether ct_dim output tiles from dst_index fit in the DEST sync region.
 *
 * @tparam Dst: DEST sync mode the kernel runs with, values = <SyncHalf/SyncFull>
 * @tparam layout: DEST layout of the output tiles; dst_index counts tiles of its custom_mm_tile_rows.
 * @param dst_index: First DEST tile index, as passed to @ref _llk_math_custom_mm_.
 * @param ct_dim: Number of output column tiles.
 * @note For LLK asserts: reads the DEST accumulation mode from config.
 */
template <DstSync Dst, CustomMmDestLayout layout>
inline bool custom_mm_fits_dest(const std::uint32_t dst_index, const std::uint32_t ct_dim)
{
    constexpr std::uint32_t unit_rows = 16; // DstTileShape::Tile16x16
    return (dst_index + ct_dim) * custom_mm_tile_rows(layout) <= get_dest_max_tiles_rt<Dst, DstTileShape::Tile16x16>() * unit_rows;
}

template <bool transpose, CustomMmDestLayout layout>
inline void custom_mm_configure_addrmod()
{
    constexpr bool split_acc = custom_mm_split_acc(layout);
    constexpr bool compact   = layout == CustomMmDestLayout::Compact;

    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR  = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_0_DEST_INCR  = compact ? 8 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR  = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = (split_acc || compact) ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR  = custom_mm_tile_rows(layout);

    using hal::cfg::Sec;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;

    // Move to the next face in width dimension, if transpose that is face 2 instead of face 1,
    // if Compact output face 1 starts at row 8 instead of 16
    set_math_addr_mod<Sec::S0, MathAddrMod {.srca_incr = ADDR_MOD_0_SRCA_INCR, .dest_incr = ADDR_MOD_0_DEST_INCR}>();
    // Move to the next face in inner dimension, if transpose that is face 1 instead of face 2,
    // if split_acc next inner dim is stored at rows 8 and 24 instead of 0 and 16
    set_math_addr_mod<Sec::S1, MathAddrMod {.srca_incr = ADDR_MOD_1_SRCA_INCR, .srcb_incr = 16, .dest_incr = ADDR_MOD_1_DEST_INCR}>();
    // Move to the next tile in width dimension, custom_mm_tile_rows down
    set_math_addr_mod<Sec::S2, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = ADDR_MOD_2_DEST_INCR, .dest_cr = 1}>();
    // Move to the next inner dimension, resets everything
    set_math_addr_mod<Sec::S3, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_clr = 1}>();
    // Last MVMUL if finalization enabled, resets SrcA and SrcB, moves Dest to the beginning of the just multiplied tile
    set_math_addr_mod<Sec::S4, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_cr = 1}>();
    // Move SrcB to 32 in preparation for finalization ELWADD
    set_math_addr_mod<Sec::S5, MathAddrMod {.srcb_incr = 32}>();
    // Move to next face in all three regs, used for final accumulation
    set_math_addr_mod<Sec::S6, MathAddrMod {.srca_incr = 16, .srcb_incr = 16, .dest_incr = 16}>();
    // NOP, has to be ADDR_MOD_7 to match with address mods configured in silu for fusing matmul and activation
    set_math_addr_mod<Sec::S7, MathAddrMod {}>();
}

// Narrow in1: each MVMUL steps to the next k-face in SrcA and SrcB, accumulating onto the same output rows (+8 if
// split_acc). Run after custom_mm_configure_addrmod; overrides its ADDR_MOD_0 and leaves ADDR_MOD_1 and 6 unused.
template <CustomMmDestLayout layout>
inline void custom_mm_configure_narrow_addrmod()
{
    constexpr std::uint8_t ADDR_MOD_0_DEST_INCR = custom_mm_split_acc(layout) ? 8 : 0;

    // Move to the next face in inner dimension, F1 of in1 with or without transpose
    hal_port::set_math_addr_mod<hal::cfg::Sec::S0, hal_port::MathAddrMod {.srca_incr = 16, .srcb_incr = 16, .dest_incr = ADDR_MOD_0_DEST_INCR}>();
}

inline void custom_mm_configure_narrow_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    using hal::fpu::SourceRelease;
    using namespace hal_port;

    hal::replay::record(
        hal::replay::BufferRange {ckernel::math::replay_buf_offset, custom_mm_replay_buf_len(operandB_face_r_dim, CUSTOM_MM_NARROW_IN1_NUM_FACES)},
        [operandB_face_r_dim]
        {
            custom_mm_mvmul<ADDR_MOD_0>(); // F0 @ F0 => 0

            // Finalization phase, as in custom_mm_configure_mop but with a single output face
            custom_mm_mvmul<ADDR_MOD_4>(); // F1 @ F1 => 0 (8 if split_acc)

            // Move the 4 rows of the split partial to rows 32+ of SrcB for finalization
            custom_mm_movd2b<32 /*SrcB row*/, 0 + 8 /*Dest row*/, ADDR_MOD_5>();

            // Move lower 4 rows if they exist (SrcB is now at row 32)
            if (operandB_face_r_dim == 8)
            {
                custom_mm_movd2b<0 + 4 /*SrcB row*/, 0 + 8 + 4 /*Dest row*/, ADDR_MOD_7>();
            }

            // Zero SrcA as it is not reused and we need to add zero to the split partial in SrcB
            custom_mm_zero_srca();

            // Partial in dest + split partial in SrcB (row 32) => final output face in dest (clear only SrcA as B is reused)
            custom_mm_elwadd_acc<ADDR_MOD_2, SourceRelease::SrcA>();
        });

    // The single top MVMUL from the replay buffer, then the end of tile or block MVMULs as in custom_mm_configure_mop
    constexpr std::uint32_t mvmul_base      = hal::replay::get_operation<hal::replay::BufferRange {ckernel::math::replay_buf_offset + 0, 1 /*count*/}>();
    constexpr std::uint32_t mvmul_end_tile  = custom_mm_mvmul_op<ADDR_MOD_2, SourceRelease::SrcA>;
    constexpr std::uint32_t mvmul_end_block = custom_mm_mvmul_op<ADDR_MOD_3, SourceRelease::Both>;

    // ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile) with last_inner = last_outer = mvmul_end_block. The
    // last_inner op is never issued with an outer count of 1 (the HAL refuses that patch), so only last_outer is set.
    hal::mop::program(hal::mop::loop(1 /*outer*/, hal::mop::loop(ct_dim, mvmul_base, mvmul_end_tile)).patch(hal::mop::last_outer, mvmul_end_block));
}

template <bool split_acc>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim)
{
    using hal::fpu::SourceRelease;
    using namespace hal_port;

    hal::replay::record(
        hal::replay::BufferRange {ckernel::math::replay_buf_offset, custom_mm_replay_buf_len(operandB_face_r_dim, TILE_NUM_FACES)},
        [operandB_face_r_dim]
        {
            custom_mm_mvmul<ADDR_MOD_0>(); // F0 @ F0 => 0
            custom_mm_mvmul<ADDR_MOD_1>(); // F0 @ F1 (F2 if transpose) => 16
            custom_mm_mvmul<ADDR_MOD_0>(); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

            // Finalization phase
            custom_mm_mvmul<ADDR_MOD_4>(); // F1 @ F3 => 16 (24 if split_acc) (clear none,
                                           // as both SrcA and SrcB banks are required for finalization)

            // We will exploit AddDst for finalization to calculate:
            // Dst = partial (in dest) + (split partial (moved to SrcB) + 0 (zeroed SrcA))
            // This way we avoid additional set of moves and move only one set of partials to SrcB
            // We have to move them to SrcB as we have to zero out the other Src register
            // and that can only be SrcA as it is not reused
            // Since we are going to keep reusing SrcB,
            // we have to move partials in lower half of SrcB as upper half contains inputs that are being reused

            // Move the 4 rows of the F0 split partial to rows 32+ of SrcB for finalization
            custom_mm_movd2b<32 /*SrcB row*/, 0 + 8 /*Dest row*/, ADDR_MOD_5>();
            // Move the 4 rows of the F1 split partial to rows 48+ of SrcB for finalization
            custom_mm_movd2b<16 /*SrcB row*/, 16 + 8 /*Dest row*/, ADDR_MOD_7>();

            // Move lower 4 rows if they exist
            if (operandB_face_r_dim == 8)
            {
                custom_mm_movd2b<0 + 4 /*SrcB row*/, 0 + 8 + 4 /*Dest row*/, ADDR_MOD_7>();
                custom_mm_movd2b<16 + 4 /*SrcB row*/, 16 + 8 + 4 /*Dest row*/, ADDR_MOD_7>();
            }

            // Zero SrcA as it is not reused and we need to add zero to split partials in SrcB
            custom_mm_zero_srca();

            // F0 partial in dest + F0 split partial in SrcB (row 32) => final F0 in dest
            custom_mm_elwadd_acc<ADDR_MOD_6, SourceRelease::None>();
            // F1 partial in dest + F1 split partial in SrcB (row 48) => final F1 in dest (clear only SrcA as B is reused)
            custom_mm_elwadd_acc<ADDR_MOD_2, SourceRelease::SrcA>();
        });

    // Top 3 MVMUL instructions from the replay buffer
    constexpr std::uint32_t mvmul_base = hal::replay::get_operation<hal::replay::BufferRange {ckernel::math::replay_buf_offset + 0, 3 /*count*/}>();
    // F1 @ F3 => 16 (24 if split_acc) (clear only SrcA as SrcB is reused)
    constexpr std::uint32_t mvmul_end_tile = custom_mm_mvmul_op<ADDR_MOD_2, SourceRelease::SrcA>;
    // F1 @ F3 => 16 (24 if split_acc) (clear both SrcA and SrcB as this is the end if the block)
    constexpr std::uint32_t mvmul_end_block = custom_mm_mvmul_op<ADDR_MOD_3, SourceRelease::Both>;

    // Mop runs a single inner dim, perform ct_dim tile multiplications while reusing a single SrcB bank
    // For all ct_dim iterations except the last we run 4 MVMULs from mvmul_base and mvmul_end_tile
    // (clearing only SrcA and moving to next output tile in Dest)
    // For the last ct_dim iteration we run 4 MVMULs from mvmul_base and mvmul_end_block
    // (clearing both SrcA and SrcB and moving back to the beginning of Dest for next inner dim)

    // ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile) with last_inner = last_outer = mvmul_end_block. The
    // last_inner op is never issued with an outer count of 1 (the HAL refuses that patch), so only last_outer is set.
    hal::mop::program(hal::mop::loop(1 /*outer*/, hal::mop::loop(ct_dim, mvmul_base, mvmul_end_tile)).patch(hal::mop::last_outer, mvmul_end_block));
}

/**
 * @brief Configure the math thread for custom_mm: programs address mods and the MVMUL MOP.
 *
 * @tparam transpose: Transpose in1 faces during the multiply.
 * @tparam layout: DEST layout of the output tiles, see CustomMmDestLayout. Pack with the same layout through
 *                 @ref _llk_pack_custom_mm_init_.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @param ct_dim: Number of output column tiles, 1 to 34.
 * @param in1_num_faces: Faces in an in1 (SrcA) tile: 4 for [32, 32], 2 for [32, 16] ([16, 32] with transpose),
 *                       which gives [M, 16] output tiles.
 * @note Establishes the operand-driven default Src zero-substitution state, which @ref _llk_math_custom_mm_ asserts
 *       under LLK asserts. Re-run this init after any math op that leaves the flag at keep (a copy init does).
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_custom_mm_init_, giving it the same face count.
 */
template <bool transpose = false, CustomMmDestLayout layout = CustomMmDestLayout::Default>
inline void _llk_math_custom_mm_init_(
    const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1, const std::uint32_t in1_num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "custom_mm: ct_dim must be 1 to 34");
    LLK_ASSERT(custom_mm_legal_in1_num_faces(in1_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");
    custom_mm_configure_addrmod<transpose, layout>();
    if (custom_mm_narrow(in1_num_faces))
    {
        custom_mm_configure_narrow_addrmod<layout>();
        custom_mm_configure_narrow_mop(operandB_face_r_dim, ct_dim);
    }
    else
    {
        custom_mm_configure_mop<custom_mm_split_acc(layout)>(operandB_face_r_dim, ct_dim);
    }

    // math::reset_counters(p_setrwc::SET_ABD_F)
    hal_port::reset_counters_abd_f();

    // A preceding datacopy leaves the Src zero flag at keep; MVMUL needs the operand-driven value.
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Run the custom_mm MVMUL MOP over kt_dim k-tiles into DEST.
 *
 * @tparam layout: Must match the init.
 * @tparam finalize: Merge the split accumulators at the end of the call; defaults to true for the SplitAcc layouts,
 *                  which alone allow it. Pass false on all but the last of several calls accumulating into the
 *                  same tiles.
 * @param operandB_face_r_dim: Face row count of in0 (SrcB), one of {1, 2, 4, 8}.
 * @param dst_index: First DEST output tile, counted in custom_mm_tile_rows of the layout.
 * @param kt_dim: Number of k-tiles to accumulate, 1 or more.
 * @param ct_dim: Number of output column tiles; must match the init, whose MOP bakes it in.
 * @param in1_num_faces: Faces in an in1 tile, 4 or 2; must match the init.
 * @note Call @ref _llk_math_custom_mm_init_ first, and again after any other math op (a copy init leaves the Src
 *       zero flag at keep).
 * @note Callers that know the DEST sync mode should also assert @ref custom_mm_fits_dest.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_custom_mm_.
 */
template <CustomMmDestLayout layout = CustomMmDestLayout::Default, bool finalize = custom_mm_split_acc(layout)>
inline void _llk_math_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim        = 1,
    const std::uint32_t in1_num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "custom_mm: Src zero-substitution flag does not hold the operand-driven value; an op between the init and "
        "this call left it at keep, so denormal Src operands would not be flushed");
    LLK_ASSERT(kt_dim >= 1, "custom_mm: kt_dim must be at least 1");
    LLK_ASSERT(custom_mm_legal_in1_num_faces(in1_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");

    const std::uint32_t replay_buf_len = custom_mm_replay_buf_len(operandB_face_r_dim, in1_num_faces);
    static_assert(!finalize || custom_mm_split_acc(layout), "custom_mm: finalize merges split partials, so it needs a SplitAcc layout");
    constexpr std::uint32_t tile_rows     = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16 ? DstTileShape::Tile16x16 : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);
    hal_port::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);

    using Mop = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    // Run mop kt_dim - 1 times, the last inner dim is handled separately to choose whether to run finalization or not
    for (std::uint32_t i = 0; i < kt_dim - 1; i++)
    {
        Mop::run();
    }

    if constexpr (finalize)
    {
        // Run the full replay ct_dim - 1 times as it has the full finalization sequence for cases where SrcB is reused
        for (std::uint32_t i = 0; i < ct_dim - 1; i++)
        {
            hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len});
        }
        // Run the full replay - the last ELWADD as it differs on the last ct_dim iteration and is issued directly after
        hal::replay::run(hal::replay::BufferRange {ckernel::math::replay_buf_offset, replay_buf_len - 1});
        hal_port::custom_mm_elwadd_acc<ADDR_MOD_3, hal::fpu::SourceRelease::Both>();
    }
    else
    {
        // Just run another mop iteration if not finalizing
        Mop::run();
    }
}
