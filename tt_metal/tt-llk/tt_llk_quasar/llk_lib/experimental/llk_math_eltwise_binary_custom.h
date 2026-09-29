// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_binary_broadcast.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/*************************************************************************
 * LLK MATH ELTWISE BINARY CUSTOM - SDPA blocked bcast-col SUB with SrcB reuse (Quasar)
 *************************************************************************/

// SrcB is unpacked ONCE as a whole tile (one dvalid) and held for the whole row block (ct_dim tiles);
// the COL face pattern F0,F0,F2,F2 is produced here by SrcB read-counter arithmetic instead of by
// consuming a per-face dvalid. This mirrors the Blackhole implementation and is what makes the reuse
// possible: p_elwise::CLR_SRCB_VLD both releases and advances, so any dvalid-driven face walk would
// destroy the operand we are trying to keep.
//
// SrcB register rows for a full 32x32 tile (16 rows per face):
//     rows  0-15 = face 0 (used by COL)     rows 16-31 = face 1 (unused by COL)
//     rows 32-47 = face 2 (used by COL)     rows 48-63 = face 3 (unused by COL)
// A 16x32 tile is the first face-row of that layout: only rows 0-31 are populated and only face 0 is read.

/**
 * @brief Init the math (FPU) thread for the SDPA blocked bcast-col SUB path (Quasar).
 *
 * SUB is LoFi-only on Quasar (fidelity phases are MUL-only), so no fidelity handling is needed, and
 * the SrcB stepping addr-mods are programmed by @ref _llk_math_sub_bcast_cols_reuse_custom_ rather
 * than here, so this call only validates the shape and rewinds the counters.
 *
 * @tparam eltwise_binary_type: FPU op (ELWSUB for the SDPA SUB path).
 * @tparam broadcast_type: Broadcast type (COL for this op).
 * @param tensor_shape: Operand tile shape. 32x32 (2x2 faces) or 16x32 (1x2 faces).
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_sub_bcast_col_init_custom_ (T0); on pack, with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_sub_bcast_cols_reuse_custom_ runs the configured op on this thread.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType broadcast_type>
inline void _llk_math_eltwise_binary_init_custom_([[maybe_unused]] const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    static_assert(broadcast_type == BroadcastType::COL, "custom sub bcast-col path supports COL broadcast only");
    static_assert(eltwise_binary_type == EltwiseBinaryType::ELWSUB, "custom sub bcast-col path supports ELWSUB only");

    LLK_ASSERT(validate_tensor_shape_sub_bcast_col_custom_(tensor_shape), "custom sub bcast-col path supports 32x32 and 16x32 tiles only");

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief SDPA blocked bcast-col SUB over ct_dim column tiles reusing one held SrcB tile (Quasar).
 *
 * Each column tile subtracts the same (col-broadcast) SrcB from its SrcA and lands in dest slot
 * dst_index + i. Per face-row, four ops cover the two dest faces: the SrcB counter walks
 * +8, -8, +8, +24 so both dest faces of the row read the same SrcB face before moving on.
 * SrcA dvalid is cleared per tile (CLR_A) to flip the SrcA bank for the next unpack while SrcB is
 * held; SrcB dvalid is cleared only once, after the last tile of the block row.
 *
 * @param ct_dim: Number of column tiles written, into dest range [dst_index, dst_index + ct_dim).
 * @param tensor_shape: Operand tile shape (drives the face-row count and the dest slot stride).
 * @param dst_index: First destination tile index.
 * @note Call @ref _llk_math_eltwise_binary_init_custom_ first.
 * @note Programs ADDR_MOD_5/6/7 itself (mirroring Blackhole) so no other init can leave them stale,
 *       and restores ADDR_MOD_7 to all-zeroes on the way out because the SFPU addresses Dest through
 *       that slot; see @ref _sfpu_configure_addrmod_. ADDR_MOD_5/6 are left holding this op's values.
 */
inline void _llk_math_sub_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    LLK_ASSERT(validate_tensor_shape_sub_bcast_col_custom_(tensor_shape), "custom sub bcast-col path supports 32x32 and 16x32 tiles only");

    // Two faces make up one face-row; a full 32x32 tile has two of them, a 16x32 tile one.
    const std::uint32_t num_face_rows = tensor_shape.num_faces_r_dim;

    static_assert(
        ELTWISE_MATH_ROWS == 8, "custom sub bcast-col path hardcodes a 4-op face-row walk and a +24 face-row jump, both valid only for MATH_ROWS == 8");

    constexpr std::uint8_t SRCB_STEP      = ELTWISE_MATH_ROWS; // +8: second half of the current face
    constexpr std::uint8_t SRCB_REWIND    = static_cast<std::uint8_t>(0x3F & -static_cast<std::int32_t>(ELTWISE_MATH_ROWS)); // -8 in 6-bit two's complement
    constexpr std::uint8_t SRCB_NEXT_FROW = 3 * ELTWISE_MATH_ROWS;                                                           // +24: skip the unused odd face

    // Programmed here rather than in the init so the ELWSUBs below cannot pick up a slot some other
    // math-thread init reprogrammed in between: ADDR_MOD_7 in particular is zeroed by
    // @ref _sfpu_configure_addrmod_ and read by every SFPU Dest load/store, and ADDR_MOD_6 is claimed
    // by several metal SFPU inits. Legacy programs the same three slots in its execute path.

    // Step within a face: advance every counter by one FPU row group.
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_STEP}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_7);

    // End of an even dest face: rewind SrcB so the paired odd face rereads the same SrcB face.
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_REWIND}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_5);

    // End of an odd dest face: jump SrcB to the next face-row (face 0 -> face 2).
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_NEXT_FROW}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_6);

    // Dest slots are as tall as the tile, which is how the packer reads them back: @ref _llk_pack_
    // scales its dest tile index by the face count and its unit is one face (16 dest rows), so a 4-face
    // tile owns 64 rows and a 2-face tile 32. Deriving the stride from the shape (rather than switching
    // on a full-tile bool) single-sources that rule with the packer instead of restating it here.
    // find_max mirrors @ref _llk_math_reduce_init_: dest never packs tighter than one face.
    // Programmed here rather than in the init for the same reason as the addr-mods above: the GPR is
    // shared with the datacopy and reduce paths, whose inits may have run in between.
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));

    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        _set_dst_write_addr_by_rows_(dst_index + i);
        _reset_counters_<p_setrwc::SET_D>();

        for (std::uint32_t face_row = 0; face_row < num_face_rows; face_row++)
        {
            // Even dest face: consume this SrcB face, then rewind so the odd face rereads it.
            TTI_ELWSUB(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // SrcB 0 -> 8
            TTI_ELWSUB(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_5, 0); // SrcB 8 -> 0
            // Odd dest face: same SrcB face again, then jump to the next face-row.
            TTI_ELWSUB(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_7, 0); // SrcB 0 -> 8
            TTI_ELWSUB(p_elwise::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_6, 0); // SrcB 8 -> 32
        }

        // Release this column's SrcA tile and rewind both read counters; KEEP the held SrcB.
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB);
    }

    // Release the held SrcB once the whole block is done.
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB);

    // Hand ADDR_MOD_7 back in the state the SFPU expects: every Quasar SFPLOAD/SFPSTORE addresses Dest
    // through this slot and relies on it being all-zeroes, but @ref _sfpu_configure_addrmod_ only runs
    // from the SFPU init, which may already have happened before this op. Same literal as that function.
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
