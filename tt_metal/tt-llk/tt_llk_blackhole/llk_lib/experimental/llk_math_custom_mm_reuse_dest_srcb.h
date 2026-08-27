// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

// Second matmul of a fused chain.  in0 is *not* unpacked from L1 — it is moved
// straight out of DEST into SrcB with MOVD2B.  Only SrcA (the weights) is unpacked.
//
// Source (in0) tile layout in DEST — what `custom_mm<split_acc, dense_packing>`
// produces after its finalization step, for an in0 tile of shape [r, 32]:
//     cols  0..15  ->  rows  base + 0  .. base + (r-1)
//     cols 16..31  ->  rows  base + 16 .. base + 16 + (r-1)
//     next tile at base + src_tile_stride   (32 with dense_packing, else 64)
//
// Output (accumulator) tile layout in DEST — the dense 2x8-row layout shared
// with the SDPA custom matmuls:
//     cols  0..15  ->  rows base + 0 .. base + 7
//     cols 16..31  ->  rows base + 8 .. base + 15
//     next tile at base + 16
// Half the DEST footprint of a `custom_mm` output tile, which is what lets the
// two matmuls' live ranges coexist in half DEST.  Pack it with Zstride = 8 rows
// and Wstride = 16 rows (see `custom_mm_reuse_dest_srcb_pack_init`).
//
// Constraints:
// - in0 (SrcB) tile shape: [{1, 2, 4, 8}, 32]; in1 (SrcA) tile shape: [32, 32]
// - rt_dim: 1
// - nt_dim: 1 to 16
// - kt_dim: even number from 2 to 256 (inclusive)
// - fidelity: LoFi only

inline void custom_mm_reuse_dest_srcb_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Next face in the output width dim (cols 16..31)

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 8, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 0, .cr = 1}, // Return to the tile base
    }
        .set(ADDR_MOD_1); // Next face in the inner (K) dim: SrcB rows 8..15

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2); // NOP — used by the DEST -> SrcB moves

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1}, // Next output tile
    }
        .set(ADDR_MOD_3);
}

// FPU half of the math replay buffer: [replay_buf_offset, replay_buf_offset + 16).
constexpr std::uint32_t CUSTOM_MM_REUSE_FPU_REPLAY_LEN = 16;
constexpr std::uint32_t CUSTOM_MM_REUSE_REPLAY_LEN     = 4;
// Top-anchored so custom_mm's program, which sits at the window base, cannot overwrite
// it.  That is what lets this one load before the first matmul instead of between them.
constexpr std::uint32_t CUSTOM_MM_REUSE_REPLAY_OFFSET = ckernel::math::replay_buf_offset + CUSTOM_MM_REUSE_FPU_REPLAY_LEN - CUSTOM_MM_REUSE_REPLAY_LEN;
static_assert(
    CUSTOM_MM_REUSE_REPLAY_OFFSET >= ckernel::math::replay_buf_offset + 11,
    "replay program overlaps custom_mm's, which uses up to 11 entries at the window base");
// The 11 above mirrors custom_mm's function-local replay_buf_len (operandB_face_r_dim == 8
// ? 11 : 9), which is not exported; deriving both from one named constant is a follow-up
// that touches the already-merged custom_mm.h.
static_assert(
    CUSTOM_MM_REUSE_REPLAY_OFFSET + CUSTOM_MM_REUSE_REPLAY_LEN <= ckernel::math::replay_buf_offset + CUSTOM_MM_REUSE_FPU_REPLAY_LEN,
    "replay program runs past the FPU half of the math replay window");

inline void custom_mm_reuse_dest_srcb_configure_mop()
{
    // One output tile = 4 MVMULs (2 SrcA faces along K x 2 along N), all
    // accumulating into the 16 DEST rows of that tile.
    load_replay_buf(
        CUSTOM_MM_REUSE_REPLAY_OFFSET,
        CUSTOM_MM_REUSE_REPLAY_LEN,
        []
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // K 0..15  x N 0..15
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // K 0..15  x N 16..31
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // K 16..31 x N 0..15
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0);    // K 16..31 x N 16..31
        });
}

// Loads the replay program.  May run before the first matmul: custom_mm reloads only
// the base of the window, which this program sits clear of.
inline void _llk_math_custom_mm_reuse_dest_srcb_replay_init_()
{
    custom_mm_reuse_dest_srcb_configure_mop();
}

// custom_mm rewrites all eight ADDR_MODs each invocation, so this must follow its last
// math instruction.  Set load_replay = false when _replay_init_ has already run.
template <bool load_replay = true>
inline void _llk_math_custom_mm_reuse_dest_srcb_init_()
{
    custom_mm_reuse_dest_srcb_configure_addrmod();
    if constexpr (load_replay)
    {
        custom_mm_reuse_dest_srcb_configure_mop();
    }
}

// src_index / dst_index are bank-relative DEST *row* offsets (not tile slots).
//
// Preconditions:
// - the unpacker has issued a zero-and-set-dvalid on SrcB
// - DEST at dst_index is zero
// - for in0_tile_r_dim < 4, rows of the producing custom_mm's output above the tile
//   height should be zero (its clear_src=false power optimization can leave stale
//   products there; MOV_4_ROWS drags two such rows into SrcB for in0_tile_r_dim == 2 —
//   benign today because the packer reads only in0_tile_r_dim rows per face, but the
//   cross-op coupling is real)
template <std::uint32_t in0_tile_r_dim>
inline void _llk_math_custom_mm_reuse_dest_srcb_(
    const std::uint32_t src_index, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t nt_dim, const std::uint32_t src_tile_stride)
{
    static_assert(
        in0_tile_r_dim == 1 || in0_tile_r_dim == 2 || in0_tile_r_dim == 4 || in0_tile_r_dim == 8,
        "custom_mm_reuse_dest_srcb: in0 tile height must be 1, 2, 4 or 8");

    const std::uint32_t dest_buffer_base = get_dest_buffer_base();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD | p_stall::WAIT_SFPU);

    for (std::uint32_t i = 0; i < kt_dim; i++)
    {
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index + i * src_tile_stride + dest_buffer_base);
        math::reset_counters(p_setrwc::SET_ABD_F);

        // DEST -> SrcB.  MVMULs 1/2 read SrcB rows 0..7 (K cols 0..15) and
        // MVMULs 3/4 read rows 8..15 (K cols 16..31), so source face0 lands at
        // SrcB row 0 and source face1 (DEST row +16) at SrcB row 8.
        if constexpr (in0_tile_r_dim == 8)
        {
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 4);
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 16);
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 20);
        }
        else if constexpr (in0_tile_r_dim == 1)
        {
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_2, p_movd2b::MOV_1_ROW, 0);
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_2, p_movd2b::MOV_1_ROW, 16);
        }
        else
        {
            // r_dim 2 or 4: one 4-row move per face.  Rows r..3 of a custom_mm
            // output face are zero (its in0 unpack clears the SrcB rows above
            // the tile height), so over-copying them is harmless.
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 0);
            TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 16);
        }

        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + dest_buffer_base);
        for (std::uint32_t j = 0; j < nt_dim; j++)
        {
            lltt::replay(CUSTOM_MM_REUSE_REPLAY_OFFSET, CUSTOM_MM_REUSE_REPLAY_LEN);
        }
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
