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

/**
 * @brief Init the math (FPU) thread for the SDPA blocked bcast-col SUB path (Quasar).
 *
 * SUB is LoFi-only on Quasar (fidelity phases are MUL-only), so no fidelity handling is needed.
 * Programs the three SrcB-stepping addr-mods used by @ref _llk_math_sub_bcast_cols_reuse_custom_;
 * nothing may reprogram slots 5/6/7 between this call and the execute call.
 *
 * @tparam eltwise_binary_type: FPU op (ELWSUB for the SDPA SUB path).
 * @tparam broadcast_type: Broadcast type (COL for this op).
 * @param tensor_shape: Operand tile shape.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_sub_bcast_col_init_custom_ (T0); on pack, with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_sub_bcast_cols_reuse_custom_ runs the configured op on this thread.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType broadcast_type>
inline void _llk_math_eltwise_binary_init_custom_([[maybe_unused]] const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    static_assert(broadcast_type == BroadcastType::COL, "custom sub bcast-col path supports COL broadcast only");
    static_assert(eltwise_binary_type == EltwiseBinaryType::ELWSUB, "custom sub bcast-col path supports ELWSUB only");

    constexpr std::uint8_t SRCB_STEP      = ELTWISE_MATH_ROWS; // +8: second half of the current face
    constexpr std::uint8_t SRCB_REWIND    = static_cast<std::uint8_t>(0x3F & -static_cast<std::int32_t>(ELTWISE_MATH_ROWS)); // -8 in 6-bit two's complement
    constexpr std::uint8_t SRCB_NEXT_FROW = 3 * ELTWISE_MATH_ROWS;                                                           // +24: skip the unused odd face

    // Step within a face: advance every counter by one FPU row group.
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_STEP}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_7);

    // End of an even dest face: rewind SrcB so the paired odd face rereads the same SrcB face.
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_REWIND}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_5);

    // End of an odd dest face: jump SrcB to the next face-row (face 0 -> face 2).
    addr_mod_t {.srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.incr = SRCB_NEXT_FROW}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_6);

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
 * @param tensor_shape: Operand tile shape (drives the face-row count).
 * @param dst_index: First destination tile index.
 * @note Call @ref _llk_math_eltwise_binary_init_custom_ first. SyncHalf budget: ct_dim <= 8 (16-bit dest) / <= 4 (fp32).
 */
inline void _llk_math_sub_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Two faces make up one face-row; a full 32x32 tile has two of them.
    const std::uint32_t num_face_rows = tensor_shape.num_faces_r_dim;

    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index + i);
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

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
