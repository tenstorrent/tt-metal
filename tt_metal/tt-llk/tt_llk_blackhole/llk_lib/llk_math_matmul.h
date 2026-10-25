// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

#ifndef HF
#define HF 0
#endif

using namespace ckernel;

/**
 * @brief Program the matmul address-mod slots for the given tile shapes, transpose, and fidelity.
 *
 * Sets up the SrcA/SrcB/dest increments and the fidelity-phase reset mods (ADDR_MOD_5, plus ADDR_MOD_6 when
 * throttling or running the row MOP). The increment pattern branches on the in0/in1 face geometry (16x32, 32x16,
 * full 32x32) and transpose.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Non-zero adds the extra fidelity-clear address mod used by the throttled MOP.
 * @tparam ROW_MOP: True to program the address mods of the row MOP (see @ref matmul_runs_row_mop): ADDR_MOD_5
 *         returns DEST to the current tile's base row, held in the DEST CR, instead of to row 0, ADDR_MOD_6
 *         steps DEST to the next tile of the reuse row, and ADDR_MOD_3 (the row's last MVMUL) resets every
 *         counter for the next run.
 * @param transpose: True to transpose in1 faces during the multiply.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 * @param ct_dim: Number of column tiles in the output block; sets the row MOP's DEST step.
 * @param rt_dim: Number of row tiles in the output block; sets the row MOP's DEST step.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL, bool ROW_MOP = false>
inline void matmul_configure_addrmod(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    static_assert(!(ROW_MOP && THROTTLE_LEVEL), "The row MOP and the throttled MOP both own ADDR_MOD_6");
    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    // 16x16 inputs not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
    LLK_ASSERT(
        !((in0_tile_r_dim == FACE_R_DIM) && (in0_tile_c_dim == FACE_C_DIM) && (in1_tile_r_dim == FACE_R_DIM) && (in1_tile_c_dim == FACE_C_DIM)),
        "16x16 by 16x16 matmul is not supported");

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    if constexpr (ROW_MOP)
    {
        // reset src, return dest to the current tile's base row (dest cr), increment fidelity if we have more
        // fidelity phases
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = fidelity_increment, .clr = 0},
        }
            .set(ADDR_MOD_5);

        // reset src and fidelity, step dest cr (and the counter with it) to the base row of the next tile of the
        // reuse row: the next column when SrcB is reused along a row, the next row when SrcA is reused down a column
        const bool reuse_a                   = ct_dim >= rt_dim;
        const std::uint32_t dest_tile_stride = (reuse_a ? 1 : ct_dim) << DstTileSizeLog2[DstTileShape::Tile32x32];
        LLK_ASSERT(dest_tile_stride <= DEST_INCR_MASK, "matmul row MOP: the DEST step to the next tile does not fit the dest increment");
        const addr_mod_t next_tile {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = static_cast<std::int16_t>(dest_tile_stride), .clr = 0, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        };
        // As next_tile.set(ADDR_MOD_6), but the dest word goes through the instruction buffer: the DEST step depends
        // on ct_dim / rt_dim, which may be run-time values (a kernel whose subblock shape is a runtime arg), and
        // set() encodes every SETC16 as an immediate.
        TTI_SETC16(addr_mod_t::addr_mod_src_reg_addr[ADDR_MOD_6], next_tile.srca.val() | (next_tile.srcb.val() << 8));
        TT_SETC16(addr_mod_t::addr_mod_dest_reg_addr[ADDR_MOD_6], next_tile.dest.val() | (next_tile.fidelity.val() << 13));
        TTI_SETC16(addr_mod_t::addr_mod_bias_reg_addr[ADDR_MOD_6], next_tile.bias.val());

        // The row's last MVMUL: reset srca/srcb/fidelity and return DEST (counter and CR) to the block base, as the
        // SETRWC(SET_ABD_F) after the per-tile MOP does, so the next run starts from the same state.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_3);
    }
    else
    {
        // reset all, increment fidelity if we have more fidelity phases
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = fidelity_increment, .clr = 0},
        }
            .set(ADDR_MOD_5);
    }

    if constexpr (THROTTLE_LEVEL)
    {
        // reset all, including fidelity
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_6);
    }

    if ((is_in0_16x32 && (!is_in1_32x16)) || is_in0_32x16)
    {
        if (transpose)
        {
            addr_mod_t {
                .srca = {.incr = 32, .clr = 0, .cr = 0},
                .srcb = {.incr = 0, .clr = 0, .cr = 1}, // cr=16 before
                .dest = {.incr = 8, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 16, .clr = 0, .cr = 0},
                .srcb = {.incr = 0, .clr = 0, .cr = 1}, // cr=16 before
                .dest = {.incr = 8, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_1);
        }
    }
    else
    {
        if (is_in1_32x16)
        {
            addr_mod_t {
                .srca = {.incr = 16, .clr = 0, .cr = 0},
                .srcb = {.incr = 8, .clr = 0, .cr = 0},
                .dest = {.incr = 0, .clr = 0, .cr = 1},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 32, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_1);
            }
            else
            {
                addr_mod_t {
                    //.srca = {.incr = srca_increment, .clr = 0, .cr = 0},
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_1);
            }
        }
    }

    if (is_in1_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 8, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 0, .cr = 1}, // cr=16
        }
            .set(ADDR_MOD_2);
    }
    else if (is_in0_16x32 || is_in0_32x16)
    {
        if (partial_face)
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 32, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 0},
                    .dest = {.incr = 16, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 0},
                    .dest = {.incr = 16, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
        }
        else
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 32, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
        }
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 32, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_2);
    }

    if (is_in0_16x32)
    {
        if (partial_face)
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 1}, // srca=16
                    .srcb = {.incr = 16, .clr = 0, .cr = 0},
                    .dest = {.incr = 0, .clr = 1, .cr = 0},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 16, .clr = 0, .cr = 0},
                    .dest = {.incr = 0, .clr = 1, .cr = 0},
                }
                    .set(ADDR_MOD_4);
            }
        }
        else
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 1}, // srca=16
                    .srcb = {.incr = 16, .clr = 0, .cr = 1},
                    .dest = {.incr = 0, .clr = 0, .cr = 1},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 16, .clr = 0, .cr = 1},
                    .dest = {.incr = 0, .clr = 0, .cr = 1},
                }
                    .set(ADDR_MOD_4);
            }
        }
    }
    else if (is_in0_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 16, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_4);
    }
    else if (is_in1_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 16, .clr = 0, .cr = 1},
        }
            .set(ADDR_MOD_4);
    }
    else
    {
        if (transpose)
        {
            addr_mod_t {
                .srca = {.incr = 16, .clr = 0, .cr = 1},
                .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
                .dest = {.incr = 0, .clr = 0, .cr = 1},
            }
                .set(ADDR_MOD_4);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 32, .clr = 0, .cr = 1},
                //.srca = {.incr = srca_set, .clr = 0, .cr = 1},
                .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
                .dest = {.incr = 0, .clr = 0, .cr = 1},
            }
                .set(ADDR_MOD_4);
        }
    }
}

/**
 * @brief Build the matmul MOP: records the per-tile MVMUL sequence into the replay buffer and wraps it in a ckernel_template.
 *
 * The recorded MVMUL order depends on the in0/in1 face geometry; the inner loop count is the number of fidelity
 * phases. For high fidelity the end op clears the reused source register (SrcA or SrcB).
 *
 * With ROW_MOP the MOP covers a whole reuse row instead of one tile: the outer loop walks the row's tiles and the
 * inner loop their fidelity phases. The replay holds the tile's MVMULs but the last; that one is loop_op1 with
 * ADDR_MOD_5 between phases, and on a tile's last phase it clears the per-tile source with ADDR_MOD_6, which also
 * steps DEST to the next tile. The row's very last MVMUL clears both sources (the per-tile one and the one reused
 * along the row) and resets every counter with ADDR_MOD_3, so no SETRWC follows the run. Each tile runs the same
 * MVMULs into the same DEST rows as the per-tile MOP.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ROW_MOP: True to build the row MOP (see @ref matmul_runs_row_mop); full 32x32 tiles only.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 */
template <MathFidelity math_fidelity, bool ROW_MOP = false>
inline void matmul_configure_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    // The row MOP issues the tile's last MVMUL itself (loop_op1 or a last-loop instruction), so it is not replayed
    const std::uint32_t replay_buf_len =
        ((is_in0_16x32 && is_in1_32x16) ? 4 : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : 16)) -
        (ROW_MOP ? 1 : 0);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [high_fidelity, reuse_a, partial_face, is_in1_32x16, is_in0_16x32, is_in0_32x16, is_in1_16x32, t_dim]
        {
            if (is_in1_32x16)
            {
                if (is_in0_16x32)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8, bias=1
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B1A1 // srca=0,    srcb+=8,  dest=16 (addr_mod_4), bias=0

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B2A0 // srca+=16,  srcb+=8,  dest=16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A1 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else if (is_in0_16x32 || is_in0_32x16)
            {
                if (partial_face)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest=+16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16,  srcb+=16,  dest=0 (addr_mod_4), bias=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A2 // srca+=16,  srcb=0,  dest=+16
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(
                        p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16/=0,  srcb=16,  dest=0/+=8 (addr_mod_4), bias=0 // srca=0 dest+=8 if in0_32x16

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8/24 // dest+=24 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                if (!is_in1_16x32)
                {
                    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8
                }
            }

            if constexpr (ROW_MOP)
            {
                // B3A3 is issued by the MOP itself (see below)
            }
            else if constexpr (high_fidelity)
            {
                // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
            }
            else
            {
                if (reuse_a)
                {
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0); // B3A3 or B2A1 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
            }
        });

    // TODO: can we commonize this?
    constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;

    if constexpr (ROW_MOP)
    {
        // Outer loop: the tiles of the reuse row. Inner loop: their fidelity phases, each ending in B3A3.
        const std::uint32_t row_tiles = reuse_a ? ct_dim : rt_dim;
        ckernel_template tmp(
            row_tiles,
            inner_loops,
            lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len),
            TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0)); // B3A3 // reset srca/srcb, dest to the tile base, increment phase (addr_mod_5)

        // B3A3 of a tile's last phase: clear the per-tile source, reset srca/srcb/fidelity, dest to the next tile (addr_mod_6).
        const std::uint32_t last_phase_op = TT_OP_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_6, 0);
        tmp.set_last_inner_loop_instr(last_phase_op);
        // B3A3 of the row's last tile: also release the source reused along the row, and reset srca/srcb/fidelity
        // and dest (counter and CR) to the block base (addr_mod_3), which the per-tile path does with an end-of-row
        // SETRWC(CLR_A/CLR_B, SET_ABD_F). Folding it into the MVMUL saves that instruction's bubble at every run.
        tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0));
        tmp.program();
        return;
    }

    ckernel_template tmp(1 /* outer loop */, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));

    if constexpr (high_fidelity)
    {
        if (reuse_a)
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
        else
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
    }
    tmp.program();
}

/**
 * @brief Emit one throttled MVMUL sequence for a full 32x32 tile, interleaving NOPs to cap matmul throughput.
 *
 * Each Level specialization (1..5) uses a different NOP-to-MVMUL ratio, yielding progressively lower throughput
 * (Level 1 ~73% of max down to Level 5 ~33%).
 *
 * @tparam Level: Throttle level, values = <1/2/3/4/5>
 */
template <int Level>
void run_throttled_sequence();

template <>
void run_throttled_sequence<1>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
}

template <>
void run_throttled_sequence<2>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
}

template <>
void run_throttled_sequence<3>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
}

template <>
void run_throttled_sequence<4>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
}

template <>
void run_throttled_sequence<5>()
{
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
}

/**
 * @brief Whether the throttled matmul MOP supports this operand geometry.
 *
 * Its replay sequences are hand-written for a full 32x32 tile; on any other geometry the recorded
 * window and the emitted sequence disagree. Single source of truth for both the assert inside
 * matmul_configure_mop_throttled and the dispatch guard in _llk_math_matmul_init_.
 *
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 * @return True when the throttled MOP may be used for this geometry.
 */
inline constexpr bool matmul_throttle_supports_tile(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face)
{
    return (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM) && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == TILE_C_DIM) &&
           !partial_face;
}

/**
 * @brief Whether the execute path issues one MOP run per fidelity phase rather than one per tile.
 *
 * Above throttle level 3 at high fidelity, @ref _llk_math_matmul_ drives the fidelity phases
 * itself, while the unthrottled MOP walks them in its own inner loop. Single source of truth for
 * that branch and for the init-side guard that must not fall back to the unthrottled MOP wherever
 * it holds -- the two have to agree or the phases execute math_fidelity times over.
 *
 * @tparam math_fidelity: Math fidelity the matmul is configured for.
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level.
 * @return True when the execute path runs the MOP once per fidelity phase.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline constexpr bool matmul_throttle_drives_phases()
{
    return (THROTTLE_LEVEL > 3) && is_high_fidelity(math_fidelity);
}

/**
 * @brief Whether the matmul runs one MOP per reuse row rather than one per tile.
 *
 * The row MOP walks all tiles of a reuse row (the ct_dim tiles of an output row when ct_dim >= rt_dim, else the
 * rt_dim tiles of an output column) in a single run, so the per-tile DEST offset write and MOP start are paid once
 * per row, and the row's last MVMUL does the end-of-row source clear and counter reset (no SETRWC). With a single
 * reuse row, consecutive K steps (k_steps) share one DEST offset write. It executes the same MVMULs into the same DEST rows as the per-tile MOP, so the
 * results are bit-identical. Throttling is a throughput cap that owns the MOP and ADDR_MOD_6, so a non-zero
 * THROTTLE_LEVEL keeps the per-tile path. Single source of truth for the init and execute sides.
 *
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level.
 * @tparam ROW_MOP: Whether the caller asked for the row MOP.
 * @return True when init builds, and execute runs, the row MOP.
 */
template <int THROTTLE_LEVEL, bool ROW_MOP>
inline constexpr bool matmul_runs_row_mop()
{
    return ROW_MOP && (THROTTLE_LEVEL == 0);
}

/**
 * @brief Whether the row MOP supports this operand geometry.
 *
 * The row MOP holds the tile's base row in the DEST CR, so every MVMUL of a tile must address DEST relative to it.
 * The full 32x32 sequence does (its only DEST reset is ADDR_MOD_4's return to the CR); the partial-face sequence
 * clears DEST mid-tile, and the other tiny-tile sequences are not covered.
 *
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 * @return True when the row MOP may be used for this geometry.
 */
inline constexpr bool matmul_row_mop_supports_tile(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face)
{
    return (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM) && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == TILE_C_DIM) &&
           !partial_face;
}

/**
 * @brief Build the throttled matmul MOP, inserting NOPs between MVMULs to cap compute throughput.
 *
 * Records a per-level @ref run_throttled_sequence into the replay buffer and wraps it in a ckernel_template.
 * Each THROTTLE_LEVEL caps throughput at a fixed fraction of max: 1 -> 73%, 2 -> 67%, 3 -> 50%, 4 -> 40%, 5 -> 33%.
 * Only supported for full 32x32 tiles (asserts on partial faces or smaller tiles).
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Throttle level, values = <1/2/3/4/5>
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_mop_throttled(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    static_assert((THROTTLE_LEVEL > 0) && (THROTTLE_LEVEL <= 5), "MM throttling only enabled for THROTTLE_LEVEL={1,2,3,4,5}");
    LLK_ASSERT(
        matmul_throttle_supports_tile(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face),
        "MM throttling only enabled for full 32x32 tile size");

    const bool reuse_a = ct_dim >= rt_dim;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    constexpr std::uint32_t replay_buff_len_throttle = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);
    const std::uint32_t replay_buf_len =
        (is_in0_16x32 && is_in1_32x16) ? 4
                                       : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : replay_buff_len_throttle);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [is_in1_32x16, is_in1_16x32, is_in0_32x16, is_in0_16x32]
        {
            if (!is_in1_32x16 && !is_in1_16x32 && !is_in0_32x16 && !is_in0_16x32)
            {
                run_throttled_sequence<THROTTLE_LEVEL>();
            }
        });

    constexpr std::uint32_t outer_loops        = (THROTTLE_LEVEL > 3) ? 2 : (high_fidelity ? to_underlying(math_fidelity) : 1);
    const std::uint32_t inner_loops            = (!is_in1_16x32) ? 2 : 1;
    constexpr std::uint8_t addr_mod_inner_loop = (THROTTLE_LEVEL > 3) ? ADDR_MOD_2 : ADDR_MOD_4;
    ckernel_template tmp(
        outer_loops,
        inner_loops,
        lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len),
        TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, addr_mod_inner_loop, 0));

    if (!is_in1_32x16 && !is_in1_16x32 && !is_in0_32x16 && !is_in0_16x32)
    {
        if constexpr (high_fidelity && THROTTLE_LEVEL > 3)
        {
            tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
        }
        else if constexpr (high_fidelity)
        {
            tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_6, 0));
        }
        else
        {
            if constexpr (THROTTLE_LEVEL > 3)
            {
                tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
            }
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_5, 0));
        }
    }

    tmp.program();
}

/**
 * @brief Configure the math (FPU/matrix engine) thread for a matmul: programs address mods and the MVMUL MOP.
 *
 * Computes D = in0 * in1, where in0 is loaded to SrcB and in1 to SrcA. When THROTTLE_LEVEL > 0, builds the
 * throttled MOP variant that inserts NOPs between MVMULs to cap matmul throughput -- except where
 * @ref matmul_throttle_supports_tile rejects the geometry and the fallback below applies.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level; 0 disables throttling, valid throttled range is {1,2,3,4,5}.
 *         Throttling is a throughput cap, not a correctness input, so a requested level is silently
 *         dropped where the throttled MOP cannot serve the geometry: levels 1-3 fall back to the
 *         unthrottled MOP on any tile @ref matmul_throttle_supports_tile rejects. Levels 4-5 at high
 *         fidelity do NOT -- @ref matmul_throttle_drives_phases holds there, so the fallback would
 *         execute every fidelity phase math_fidelity times over, and those levels keep the throttled
 *         MOP. Pairing them with a tile the throttled MOP does not support wedges the math thread.
 * @tparam ROW_MOP: True to build the row MOP, which runs once per reuse row instead of once per tile (see
 *         @ref matmul_runs_row_mop); ignored when THROTTLE_LEVEL > 0. Full 32x32 tiles only. Also programs
 *         ADDR_MOD_3 and ADDR_MOD_6.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 * @param transpose: Non-zero to transpose in1 faces during the multiply.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_matmul_init_ which feeds SrcA/SrcB.
 * @note @ref _llk_math_matmul_ runs the configured matmul with matching template args. With the row MOP, every
 *       @ref _llk_math_matmul_ call must also pass this init's ct_dim and rt_dim: the MOP walks that many tiles.
 * @note Establishes the operand-driven DEFAULT Src zero-substitution state; @ref
 *       _llk_math_matmul_ asserts this precondition under LLK asserts, so a prior op that left PRESERVE
 *       must be followed by this init (or a format-changing reconfig) before the matmul.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, bool ROW_MOP = false>
inline void _llk_math_matmul_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    // in1=32x16 NOT supported with transpose (no addr_mod handling)
    LLK_ASSERT(!(transpose && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == FACE_C_DIM)), "Transpose with input 1 dimensions 32x16 not supported");

    constexpr bool row_mop = matmul_runs_row_mop<THROTTLE_LEVEL, ROW_MOP>();
    if constexpr (row_mop)
    {
        LLK_ASSERT(
            matmul_row_mop_supports_tile(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face),
            "The matmul row MOP only supports full 32x32 tiles");
    }

    matmul_configure_addrmod<math_fidelity, THROTTLE_LEVEL, row_mop>(
        transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, ct_dim, rt_dim);

    if constexpr (THROTTLE_LEVEL > 0)
    {
        // On a geometry the throttled MOP does not support, fall back to the unthrottled one:
        // throttling is only a throughput cap, and the throttled sequence wedges there.
        //
        // Sound only while the execute path is a single ckernel_template::run(). Where
        // matmul_throttle_drives_phases holds, _llk_math_matmul_ runs the MOP once per fidelity
        // phase, whereas the unthrottled MOP already walks the phases in its own inner loop -- so
        // falling back there would execute every phase math_fidelity times over. Those levels keep
        // the throttled path, which the geometry cannot serve: measured on Blackhole, throttle 5 at
        // HiFi4 on a tiny tile wedges the math thread with or without LLK asserts. Closing it needs
        // the execute side told which MOP is live, which is a wider change than this guard.
        constexpr bool fallback_matches_execute = !matmul_throttle_drives_phases<math_fidelity, THROTTLE_LEVEL>();

        if (fallback_matches_execute && !matmul_throttle_supports_tile(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face))
        {
            matmul_configure_mop<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
        }
        else
        {
            matmul_configure_mop_throttled<math_fidelity, THROTTLE_LEVEL>(
                ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
        }
    }
    else
    {
        matmul_configure_mop<math_fidelity, row_mop>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Matmul never sets the Src zero-substitution flag itself, so
    // re-establish the operand-driven DEFAULT here. A preceding copy_init/datacopy op may have left the flag in
    // PRESERVE. Relying on a reconfig_data_format to clear it is not enough: the metal wrapper
    // llk_math_reconfig_data_format_srca(old, new) skips the LLK reconfig entirely when old == new format, so on a
    // same-format reconfig the reset never runs. Without this line MVMUL would inherit a stale keep-denormals flag
    // (denormal Src results differ). Mirrors what reduce/transpose/datacopy inits already do.
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Uninitialize/cleanup after matmul operations, restoring any modified state to defaults.
 *
 * @note Reverses @ref _llk_math_matmul_init_; currently a no-op since all state is transient.
 */
inline void _llk_math_matmul_uninit_()
{
    // No state to restore - all states are transient or default
}

/**
 * @brief Perform a matmul block, accumulating in0 * in1 into the destination register.
 *
 * Iterates over the output block reusing SrcA or SrcB (whichever dimension is larger) to minimize reloads,
 * clearing the reused source register at the end of each reuse row.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level; must match the value used at init.
 * @tparam ROW_MOP: Whether init built the row MOP; must match the value used at init.
 * @param dst_index: Base tile index into the destination register for the output block.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param k_steps: Number of consecutive inner-dimension (K) steps to accumulate into the same output block, one
 *        unpacked in0/in1 set (one @ref _llk_unpack_AB_matmul_ call) per step. Equivalent to k_steps calls with
 *        k_steps = 1; with the row MOP and a single reuse row the DEST offset is then written once for all steps.
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this
 *       function, and @ref _llk_math_matmul_uninit_ after it to restore modified state. With the row MOP,
 *       ct_dim and rt_dim must also match the init.
 * @note On the unpack thread, @ref _llk_unpack_AB_matmul_ must feed the operand tiles into SrcA/SrcB.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, bool ROW_MOP = false>
inline void _llk_math_matmul_(std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1, const std::uint32_t k_steps = 1)
{
    // Zero-flag leak guard. MVMUL reads ALU_ACC_CTRL_Zero_Flag_disabled_src, which for
    // denormal Src operands changes the result (flush vs keep — measured on WH n150 / BH p150b). Matmul is
    // a "runs-in-DEFAULT" op: it never sets the flag itself and relies on hw_configure / a format-changing
    // reconfig having established the format-driven DEFAULT. If a prior copy_init/datacopy op left the flag in
    // PRESERVE and no format-changing reconfig ran before this matmul, MVMUL silently
    // keeps denormals. This assert catches exactly that leak (fires only under LLK asserts).
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "matmul: Src zero-substitution flag does not hold the operand-driven value — a prior op (copy_init/datacopy) left "
        "a keep flag before MVMUL without a format-changing reconfig; denormal Src results will differ");

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // reuse-dim

    if constexpr (matmul_runs_row_mop<THROTTLE_LEVEL, ROW_MOP>())
    {
        // One MOP run per reuse row: it steps DEST from tile to tile, clears both sources and leaves every counter
        // reset at the block base, so consecutive runs need only the row's DEST offset
        if (t_dim == 1)
        {
            // A single reuse row: every K step accumulates into the same DEST tiles, so the offset is written once
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
            for (std::uint32_t k = 0; k < k_steps; k++)
            {
                ckernel_template::run();
            }
            return;
        }
        for (std::uint32_t k = 0; k < k_steps; k++)
        {
            for (std::uint32_t t = 0; t < t_dim; t++)
            {
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t : t));
                ckernel_template::run();
            }
        }
        return;
    }

    for (std::uint32_t k = 0; k < k_steps; k++)
    {
        for (std::uint32_t t = 0; t < t_dim; t++)
        {
            for (std::uint32_t rut = 0; rut < rut_dim; rut++)
            {
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

                if constexpr (matmul_throttle_drives_phases<math_fidelity, THROTTLE_LEVEL>())
                {
                    for (std::uint32_t phase = 0; phase < to_underlying(math_fidelity); phase++)
                    {
                        ckernel_template::run();
                    }
                    if (reuse_a)
                    {
                        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                    else
                    {
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                }
                else
                {
                    ckernel_template::run();
                }

                // Clear srcB or srcA at end of reuse (once per u block row)
                if (rut == (rut_dim - 1))
                {
                    if (reuse_a)
                    {
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                    else
                    {
                        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                }
            }
        }
    }
}
