// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;

template <BroadcastType bcast_type>
inline void eltwise_binary_configure_addrmod_custom()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param num_faces: Number of faces to process (1, 2, or 4)
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type>
inline void _llk_math_eltwise_binary_init_custom_(const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod_custom<src_b_bcast_type>();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_uninit_custom_()
{
    // No state to restore - all states are transient or default
}

/**
 * @brief Emit one bcast-col FPU op (ELWMUL or ELWSUB) advancing srcA/srcB/dest per addr_mod.
 *
 * The op is fixed at compile time, so the constexpr branch resolves to a single emitted instruction.
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWSUB/ELWMUL>
 * @tparam addr_mod: addr-mod slot selecting the srcA/srcB/dest strides for this op.
 */
template <EltwiseBinaryType eltwise_binary_type, std::uint8_t addr_mod>
inline void _bcast_cols_op_()
{
    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, addr_mod, 0);
    }
    else
    {
        TTI_ELWSUB(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_COL, addr_mod, 0);
    }
}

/**
 * @brief Blocked bcast-col FPU scaffold shared by the SDPA SUB and indexer_score MUL paths.
 *
 * Single-sources the addr-mod setup, srcB face-reuse addressing, and per-tile counter resets; the two
 * paths differ only in the FPU op (selected at compile time). The name tracks the shared mechanism
 * (srcB face reuse), matching WH's _llk_math_sub_bcast_cols_reuse_custom_.
 *
 * The MUL path is a dest-MAC head reduction: ELWMUL's dest_accum_en field is 0 (MAC, not overwrite) and
 * dest is cleared only by tile_regs_acquire's ZEROACC, so the caller does ONE acquire per ct_dim-column
 * batch and calls once per head into the SAME dest -> dest[col] = sum_h qk[col,h]*w[h], one pack per
 * column instead of per (column, head). cb_qk must be head-major.
 *
 * Each 16x16 face pair (one face-row) is processed by four ops (two per face; each covers an aligned
 * 8x16 block). A full 32x32 tile has two face-rows (F0/F1 and F2/F3); a 16x32 tiny tile has a single
 * face-row (F0/F1).
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWSUB/ELWMUL>
 * @param ct_dim: Number of srcA column tiles processed in the block.
 * @param tensor_shape: Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
 * @param dst_index: Absolute dest tile slot where this block-row's ct_dim tiles begin. Multi-tile-row
 *                   callers (e.g. the fuser LoopBlockRow driver) advance this per block-row so each row
 *                   lands on its own dest slots; single-tile-row callers leave it at 0.
 * @note Canonical description of the shared blocked bcast-col mechanism; other files reference this one.
 */
template <EltwiseBinaryType eltwise_binary_type>
inline void _llk_math_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWMUL || eltwise_binary_type == EltwiseBinaryType::ELWSUB,
        "blocked bcast-col reuse scaffold supports ELWMUL and ELWSUB only");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Two faces make up one face-row; a full tile has two of them, a tiny tile one.
    const std::uint32_t num_face_rows = tensor_shape.num_faces_r_dim;

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 24},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_6);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0x3F & -8}, // decrement srcB by 8
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_5);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB); // reset both src counters to 0

    // Per tile: walk srcA/dest across 4 faces while srcB reuses one face per pair (F0/F1 -> srcB face
    // 0, F2/F3 -> srcB face 2). Comments are dest/srcA counter values. ADDR_MOD_5 rewinds srcB -8 so
    // the pair's 2nd op rereads the same srcB face; ADDR_MOD_6 then jumps srcB +24 to the next pair.
    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        // Position the dest write pointer at the start of this tile's 32x32 dest
        // slot (stride 64 rows). The packer indexes tiles at the full 32x32 tile
        // stride regardless of num_faces, so tiles must be written one-per-slot.
        // The ADDR_MOD dest increments below only advance 32 rows for a 16x32 tiny
        // tile (num_faces=2) and 64 rows for a full 32x32 tile (num_faces=4); the
        // per-slot base + counter reset makes both land on the right slot.
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + i);
        math::clear_dst_reg_addr();

        for (std::uint32_t face_row = 0; face_row < num_face_rows; face_row++)
        {
            // Even face (F0 / F2): op then rewind srcB by one face (ADDR_MOD_5) so the odd face reads it.
            _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_7>(); // srcB: 0 -> 8
            _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_5>(); // srcB: 8 -> 0

            // Odd face (F1 / F3): op then advance srcB +24 (ADDR_MOD_6) to the next face-row / tile.
            _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_7>(); // srcB: 0 -> 8
            _bcast_cols_op_<eltwise_binary_type, ADDR_MOD_6>(); // srcB: 8 -> 32 (next face-row)
        }

        // Rewind srcB to 0 for the next srcA tile (CLR_A clears srcA's dvalid); srcA keeps advancing.
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
    }

    // Clear srcB dvalid on the way out.
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB);

    // Restore the dest base offset so the next op starts from tile slot 0.
    math::clear_dst_reg_addr();
}

/**
 * @brief SDPA's SUB entry point into the shared blocked bcast-col scaffold.
 *
 * Kept as an op-named wrapper because external callers reference this name directly (WH arch + the
 * tt-llk eltwise_bcast_col tests), so it can't be folded away. The MUL path has no such external caller
 * and calls @ref _llk_math_bcast_cols_reuse_custom_ <ELWMUL> directly.
 *
 * @param ct_dim: Number of srcA column tiles processed in the block.
 * @param tensor_shape: Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
 * @param dst_index: Absolute dest tile slot where this block-row's ct_dim tiles begin.
 */
inline void _llk_math_sub_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    _llk_math_bcast_cols_reuse_custom_<EltwiseBinaryType::ELWSUB>(ct_dim, tensor_shape, dst_index);
}
