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
#include "sanitizer/api.h"

#ifndef HF
#define HF 0
#endif

using namespace ckernel;

/**
 * @brief Program the matmul address-mod slots for the given tile shapes, transpose, and fidelity.
 *
 * Sets up the SrcA/SrcB/dest increments and the fidelity-phase reset mods (ADDR_MOD_5, plus ADDR_MOD_6 when
 * throttling). The increment pattern branches on the in0/in1 face geometry (16x32, 32x16, full 32x32) and transpose.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Non-zero adds the extra fidelity-clear address mod used by the throttled MOP.
 * @param transpose: True to transpose in1 faces during the multiply.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_addrmod(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
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

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 1},
        .srcb     = {.incr = 0, .clr = 1, .cr = 1},
        .dest     = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = fidelity_increment, .clr = 0},
    }
        .set(ADDR_MOD_5);

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
                    .srca = {.incr = 32, .clr = 0, .cr = 0}, .srcb = {.incr = 0, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 0},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 0, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 0},
                    // .bias = {.incr = 1},
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
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 16, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 1, .cr = 0},
                    // .bias = {.incr = 1},
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
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 16, .clr = 0, .cr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
        }
    }
    else if (is_in0_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 16, .clr = 0, .cr = 1}, .dest = {.incr = 8, .clr = 0, .cr = 0},
            // .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
    else if (is_in1_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 8, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 1},
            // .bias = {.incr = 1},
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
                // .bias = {.incr = 1},
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
                // .bias = {.incr = 1},
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
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 */
template <MathFidelity math_fidelity>
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

    const std::uint32_t replay_buf_len =
        (is_in0_16x32 && is_in1_32x16) ? 4 : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : 16);

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

            if constexpr (high_fidelity)
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
        (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM) && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == TILE_C_DIM) && !partial_face,
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
 * throttled MOP variant that inserts NOPs between MVMULs to cap matmul throughput.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam THROTTLE_LEVEL: Compute-throughput throttle level; 0 disables throttling, valid throttled range is {1,2,3,4,5}.
 * @param in0_tile_r_dim: Row dimension of an in0 tile.
 * @param in0_tile_c_dim: Column dimension of an in0 tile.
 * @param in1_tile_r_dim: Row dimension of an in1 tile.
 * @param in1_tile_c_dim: Column dimension of an in1 tile.
 * @param partial_face: True when the tile has fewer than the full set of faces.
 * @param transpose: Non-zero to transpose in1 faces during the multiply.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_matmul_init_ which feeds SrcA/SrcB.
 * @note @ref _llk_math_matmul_ runs the configured matmul with matching template args.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
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
    llk::san::operation_init<llk::san::Operation::Matmul>(math_fidelity, THROTTLE_LEVEL, ct_dim, rt_dim);

    matmul_configure_addrmod<math_fidelity, THROTTLE_LEVEL>(transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    if constexpr (THROTTLE_LEVEL > 0)
    {
        matmul_configure_mop_throttled<math_fidelity, THROTTLE_LEVEL>(
            ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    else
    {
        matmul_configure_mop<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
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
 * @param dst_index: Base tile index into the destination register for the output block.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this
 *       function, and @ref _llk_math_matmul_uninit_ after it to restore modified state.
 * @note On the unpack thread, @ref _llk_unpack_AB_matmul_ must feed the operand tiles into SrcA/SrcB.
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_(std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1)
{
    llk::san::operation_check<llk::san::Operation::Matmul>(math_fidelity, THROTTLE_LEVEL, ct_dim, rt_dim);

    const bool reuse_a           = ct_dim >= rt_dim;
    const std::uint32_t t_dim    = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim  = reuse_a ? ct_dim : rt_dim; // reuse-dim
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            if constexpr (THROTTLE_LEVEL > 3 && high_fidelity)
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
