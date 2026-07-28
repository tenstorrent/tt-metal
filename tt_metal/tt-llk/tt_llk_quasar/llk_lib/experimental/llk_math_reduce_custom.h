// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// ============================================================================================
// Experimental: block reduce_max_row for Quasar
//
// Computes the row-wise MAX of a *block* of `block_ct_dim` tiles laid out contiguously in the
// width (ct) dimension, producing a single reduced column tile in DEST. Used by SDPA/softmax
// (llama) to fold the block-max into one call instead of `block_ct_dim` native reduce calls.
//
// Algorithm (translated from the Wormhole/Blackhole experimental kernel to Quasar primitives):
//   1. POOL PHASE (MOP, `block_ct_dim` iterations): for every operand tile, GMPOOL-max its faces
//      into two fixed DEST scratch slots (one per input face-row): faces F0&F1 -> slot0 (DEST row
//      0), faces F2&F3 -> slot1 (DEST row 32). The DEST counter is *parked* (ADDR_MOD_0 has
//      dest.incr = 0) and the slot is selected by the GMPOOL immediate `dst` field, so successive
//      tiles accumulate a running row-max into the same slots.
//   2. TRANSPOSE PHASE (issued once by the execute call): transpose each 1x16 row partial into a
//      16x1 column via MOVD2B(transpose=1) -> ELWADDDI, exactly as Quasar's native reduce-row does
//      (see _llk_math_reduce_row_mop_config_ in llk_math_reduce.h). slot0 -> output F0 (rows 0-15),
//      slot1 -> output F2 (rows 32-47).
//
// Assumptions (same contract as the WH/BH kernel):
//   - Scaler is 1.0 resident in F0 of the scaler tile and constant for the whole block.
//   - Operand/scaler format is bfloat16_b.
//   - Tile is 32x32 (num_faces=4) or a 16x32 tiny tile (num_faces=2, a single input face-row).
//   - MAX pool on the ROW dimension only.
//
// NOTE (device validation): GMPOOL's DEST accumulate-vs-replace behaviour and the first-tile seed
// are HW details to confirm on-device. The pool below mirrors the native reduce-row GMPOOL pair
// idiom; the running-max across tiles relies on GMPOOL maxing into the existing slot contents.
// ============================================================================================

// DEST scratch slot for the second input face-row (F2&F3) partial. Row TILE_R_DIM (= 2*FACE_R_DIM =
// 32) == output face F2, so the transpose reads and writes the same base (mirrors the native
// reduce-row per-face-row DEST advance).
static constexpr std::uint32_t REDUCE_BLOCK_SLOT1_DST = TILE_R_DIM;

/**
 * @brief Program the address modifiers used by the block reduce_max_row pool + transpose.
 *
 * ADDR_MOD_0: no counter movement (pool parks DEST; MOVD2B reads DEST at the counter). fidelity clr.
 * ADDR_MOD_1: srcb += ELTWISE_MATH_ROWS, dest += ELTWISE_MATH_ROWS -- the ELWADDDI column write-back.
 * ADDR_MOD_2: no movement (MAX is LoFi, no fidelity phases).
 */
inline void reduce_block_max_row_configure_addrmod()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = ELTWISE_MATH_ROWS},
        .dest = {.incr = ELTWISE_MATH_ROWS},
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_2: srca += one face (FACE_R_DIM rows), no CLR. Present for experiments; NOT used by
    // the checkpoint pool (which advances face-rows with ZEROSRC) or the transpose (ADDR_MOD_0/1).
    addr_mod_t {.srca = {.incr = FACE_R_DIM}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0}}.set(ADDR_MOD_2);
}

/**
 * @brief Transpose one pooled 1x16 row partial (at the current DEST counter) into a 16x1 column.
 *
 * Mirrors the inline transpose in Quasar's native _llk_math_reduce_row_mop_config_.
 */
inline void reduce_block_max_row_transpose_face_row(const bool wide_face)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);
    // Move the row partial into SrcB rows [16-31], transposed into rows [32-47].
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);
    // Write the transposed column back to DEST via plain MOVB2D (SrcB rows 32-47 -> DEST), instead of
    // ZEROSRC + ELWADDDI. Mirrors native _reduce_row_transpose_fpu_ step 4 (MOVB2D writeback). First
    // move covers DEST rows 0-7.
    TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0);
    if (wide_face)
    {
        // face_r_dim > ELTWISE_MATH_ROWS (full 16-row face): second move covers DEST rows 8-15.
        TTI_MOVB2D(p_mov::DEST_NORM, p_mov_src_to_dest::SRC_ROW32_OFFSET + 8, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8);
    }
}

/**
 * @brief Records the per-tile pool block in the replay buffer and programs the pool MOP.
 *
 * The replay buffer holds the pool ops (2 GMPOOLs per input face-row) plus a per-tile CLR_A. The MOP
 * replays this block `block_ct_dim` times, accumulating a running row-max into the two DEST slots.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const TensorShape& tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    // 32-bit DEST accumulation for block reduce_max_row is not yet implemented on Quasar: the row
    // transpose would need the hi16/lo16 split path (cf. _reduce_row_transpose_fpu_). The primary
    // bf16 16-bit-DEST path (llama SDPA) is supported first.
    LLK_ASSERT(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");

    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);
    // CHECKPOINT / KNOWN-INCOMPLETE. This "pool both face-rows into DEST, transpose once" structure
    // (ported from WH/BH) is structurally wrong for Quasar: advancing SrcA to face-row 1 BEFORE
    // transposing face-row 0 either wipes DEST (any CLR_SRCA_VLD / SETRWC(CLR_A) / CLEARDVALID) or
    // clobbers F2,F3 (ZEROSRC). ZEROSRC is the least-wrong (slot0 exact, F2,F3 a few cols short).
    // Correct fix needs a redesign to native reduce_row's transpose-BEFORE-advance rhythm; see the
    // project notes. Left at the ZEROSRC checkpoint pending that.
    const std::uint32_t pool_len = (two_face_rows ? 6u : 3u);

    load_replay_buf(
        0,
        pool_len,
        false,
        0,
        0,
        [two_face_rows]
        {
            // Face-row 0 (F0 & F1) -> slot0 (DEST row 0).
            TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            if (two_face_rows)
            {
                TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
                // Face-row 1 (F2 & F3) -> slot1 (DEST row 32).
                TTI_GMPOOL(p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_BLOCK_SLOT1_DST);
                TTI_GMPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_BLOCK_SLOT1_DST);
            }
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB);
        });

    const std::uint32_t pool_replay = TT_OP_REPLAY(0, pool_len, 0, 0, 0, 0);
    ckernel_template temp(1 /*outer*/, block_ct_dim /*inner*/, pool_replay);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the block reduce_max_row math thread.
 *
 * @tparam block_ct_dim  Number of operand tiles processed as one block.
 * @tparam is_fp32_dest_acc_en  32-bit DEST accumulation (asserted unsupported for now, see mop_config).
 * @param tensor_shape   Operand tile shape.
 * @note Pair with @ref _llk_unpack_AB_reduce_block_max_row_init_ (T0) and
 *       @ref _llk_pack_reduce_mask_config_ (T2). @ref _llk_math_reduce_block_max_row_ is the execute call.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const TensorShape& tensor_shape)
{
    reduce_block_max_row_configure_addrmod();

    // Dense/sparse DEST layout selector (matches native reduce init).
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));

    _reset_counters_<p_setrwc::SET_ABD_F>();

    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
}

/**
 * @brief Executes the block reduce_max_row: accumulate the pool across the block, then transpose once.
 *
 * @param dst_index    DEST tile index for the reduced column result.
 * @param tensor_shape Operand tile shape.
 * @note Call @ref _llk_math_reduce_block_max_row_init_ with matching template args first.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!is_fp32_dest_acc_en, "32-bit DEST block reduce_max_row not supported on Quasar yet");

    _set_dst_write_addr_by_rows_(dst_index);

    // POOL PHASE: run the MOP, accumulating the running row-max across all block_ct_dim tiles into
    // the two DEST scratch slots (0 and 32).
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // TRANSPOSE PHASE: transpose each pooled row partial into a column, once.
    const bool two_face_rows = (tensor_shape.num_faces_r_dim > 1);
    const bool wide_face     = (tensor_shape.face_r_dim > ELTWISE_MATH_ROWS);

    // slot0 (input face-row 0) -> output F0 (DEST rows 0-15). DEST counter is at the tile base.
    reduce_block_max_row_transpose_face_row(wide_face);

    if (two_face_rows)
    {
        // Move DEST to slot1 (row TILE_R_DIM == output F2) and transpose that partial in place.
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, TILE_R_DIM, p_setrwc::SET_D);
        reduce_block_max_row_transpose_face_row(wide_face);
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
