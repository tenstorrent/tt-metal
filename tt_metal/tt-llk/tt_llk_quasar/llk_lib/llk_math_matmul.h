// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Initializes addrmod for matrix multiply operation.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, programs addr_mods for the MXFP4_2x non-DI MOP variant (8 MVMULs covering only A0/A1 and B0/B1; SrcA in MxFp4_2x_A/B
 * drives the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;
    const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;

    if constexpr (ENABLE_2X_FORMAT)
    {
        // Non-DI MXFP4_2x traversal (mirrors the DI X2 (srca,srcb,dest) sequence):
        //   #0 (0, 0, 0)     #1 (0, 8, 8)
        //   #2 (16, 0,16)    #3 (16, 8,24)
        //   #4 (0,16,32)     #5 (0,24,40)
        //   #6 (16,16,48)    #7 (16,24,56)
        // SrcB needs two distinct "wrap" targets (0 then 16). We exploit RWC_SrcB_Cr:
        // at #1->#2 it is still 0 so srcb cr=1 wraps to 0; at #3->#4 we pump it up to
        // 16 via {cr=1, incr=16}; at #5->#6 srcb cr=1 then wraps to 16.

        // Common in-replay step (used between #0->#1, #2->#3, #4->#5, #6->#7).
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // #1 -> #2: srca steps to A1, srcb wraps back to 0 (RWC_SrcB_Cr is 0 here).
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        // #3 -> #4: srca wraps back to 0, srcb advances RWC_SrcB_Cr from 0 to 16 in
        // the same step ({cr=1, incr=16} -> srcb = 0+16 = 16, RWC_SrcB_Cr := 16).
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 16, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_2);

        // #5 -> #6: srca steps to A1, srcb wraps to RWC_SrcB_Cr (= 16 now).
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_3);

        // matmul_op (intermediate fidelity phase): reset src registers, snap dest
        // back to start of this tile, advance fidelity counter.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 0},
            .srcb     = {.incr = 0, .clr = 1, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
        }
            .set(ADDR_MOD_4);

        // matmul_op_last: end-of-tile, advance dest to next tile, clear fidelity.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 0},
            .srcb     = {.incr = 0, .clr = 1, .cr = 0},
            .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_5);
        return;
    }

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

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    // reset all, increment dest carriage return
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_3);

    addr_mod_t {
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4);

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_5);
}

/**
 * @brief Initializes addrmod for matrix multiply operation using the direct-indexing instruction variant.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_matmul_di_addrmod_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;
    const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;

    // Direct indexing supplies absolute srcb/srca/dest indices in each MVMULDI, so the
    // replayed instructions (which all select ADDR_MOD_0) must apply no auto-increment.
    // Program it explicitly: otherwise ADDR_MOD_0 is inherited from a prior matmul kernel
    // (e.g. a regular MVMUL matmul leaves dest/srcb +=8), perturbing the dest addressing.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // only increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Initializes mop config for matrix multiply operation.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, emits the non-DI MXFP4_2x variant (7-MVMUL replay traversing only A0/A1 and B0/B1; relies on SrcA being unpacked as
 * MxFp4_2x_A/B for the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_mop_config_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    const bool reuse_a = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = ENABLE_2X_FORMAT ? (8 - 1) : (16 - 1);

    if constexpr (ENABLE_2X_FORMAT)
    {
        // Non-DI MXFP4_2x: 7-MVMUL replay + matmul_op = 8 MVMULs per tile (vs 16 in plain non-DI).
        // (srca,srcb,dest) sequence mirrors the DI X2 path:
        //   #0 (0,  0,  0)  B0[0:7]*A0
        //   #1 (0,  8,  8)  B0[8:15]*A0
        //   #2 (16, 0, 16)  B0[0:7]*A1     <- ADDR_MOD_1 (srca+=16, srcb cr->0)
        //   #3 (16, 8, 24)  B0[8:15]*A1
        //   #4 (0, 16, 32)  B1[0:7]*A0     <- ADDR_MOD_2 (srca cr->0, srcb cr+=16 lifts RWC_SrcB_Cr to 16)
        //   #5 (0, 24, 40)  B1[8:15]*A0
        //   #6 (16,16, 48)  B1[0:7]*A1     <- ADDR_MOD_3 (srca+=16, srcb cr->16)
        //   #7 (16,24, 56)  B1[8:15]*A1    <- matmul_op (ADDR_MOD_4) / matmul_op_last (ADDR_MOD_5)
        load_replay_buf<0, replay_buf_len>(
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #0 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #1 -> srca+=16, srcb cr->0, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #2 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // #3 -> srca cr->0, srcb cr+=16 (=16), dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #4 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // #5 -> srca+=16, srcb cr->16, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #6 -> srcb+=8, dest+=8
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0  // A1 -> A2 && srca=16 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8
            });
    }

    constexpr std::uint32_t matmul_op_addr_mod      = ENABLE_2X_FORMAT ? ADDR_MOD_4 : ADDR_MOD_5;
    constexpr std::uint32_t matmul_op_last_addr_mod = ENABLE_2X_FORMAT ? ADDR_MOD_5 : ADDR_MOD_3;
    constexpr static std::uint32_t matmul_op        = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, matmul_op_addr_mod, 0);
    const std::uint32_t matmul_op_last =
        reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, matmul_op_last_addr_mod, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, matmul_op_last_addr_mod, 0);

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes mop config for matrix multiply operation using the direct-indexing instruction variant.
 *
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT>
inline void _llk_math_matmul_di_mop_config_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);
    const bool reuse_a                      = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len =
        ENABLE_2X_FORMAT ? 8 - 1 : 16 - 1; // -1 since the last instruction for the Tile * Tile operation will come out of the MOP
    if constexpr (ENABLE_2X_FORMAT)
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                // [B0] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0); // B0[0:7]*A0  srcb=0x0<<2='d0, srca=0x0<<2='d0, dest=0x0<<2='d0
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2); // B0[8:15]*A0 srcb=0x2<<2='d8, srca=0x0<<2='d0, dest=0x2<<2='d8
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4); // B0[0:7]*A1  srcb=0x0<<2='d0, srca=0x4<<2='d16, dest=0x4<<2='d16
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6); // B0[8:15]*A1 srcb=0x2<<2='d8, srca=0x4<<2='d16, dest=0x6<<2='d24
                // [B1] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x0, 0x0, 0x8); // B1[0:7]*A0  srcb=0x4<<2='d16, srca=0x0<<2='d0, dest=0x8<<2='d32
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x0, 0x0, 0xA); // B1[8:15]*A0 srcb=0x6<<2='d24, srca=0x0<<2='d0, dest=0xA<<2='d40
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x4, 0x0, 0xC); // B1[0:7]*A1  srcb=0x4<<2='d16, srca=0x4<<2='d16, dest=0xC<<2='d48
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                // [B0] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0); // B0[0:7]*A0  srcb=0x0<<2='d0, srca=0x0<<2='d0, dest=0x0<<2='d0
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2); // B0[8:15]*A0 srcb=0x2<<2='d8, srca=0x0<<2='d0, dest=0x2<<2='d8
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4); // B0[0:7]*A1  srcb=0x0<<2='d0, srca=0x4<<2='d16, dest=0x4<<2='d16 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6); // B0[8:15]*A1 srcb=0x2<<2='d8, srca=0x4<<2='d16, dest=0x6<<2='d24 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.

                // [B2] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x0, 0x0, 0x8); // B2[0:7]*A0  srcb=0x8<<2='d32, srca=0x0<<2='d0, dest=0x8<<2='d32
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x0, 0x0, 0xA); // B2[8:15]*A0 srcb=0xA<<2='d40, srca=0x0<<2='d0, dest=0xA<<2='d40
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x4, 0x0, 0xC); // B2[0:7]*A1  srcb=0x8<<2='d32, srca=0x4<<2='d16, dest=0xC<<2='d48 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x4, 0x0, 0xE); // B2[8:15]*A1 srcb=0xA<<2='d40, srca=0x4<<2='d16, dest=0xE<<2='d56 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.

                // [B1] x [A2 A3] (Accumulates to the result of [B0] x [A0 A1] )
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x8, 0x0, 0x0); // B1[0:7]*A2  srcb=0x4<<2='d16, srca=0x8<<2='d32, dest=0x0<<2='d0 // A2 -> A1 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x8, 0x0, 0x2); // B1[8:15]*A2 srcb=0x6<<2='d24, srca=0x8<<2='d32, dest=0x2<<2='d8 // A2 -> A1 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0xC, 0x0, 0x4); // B1[0:7]*A3  srcb=0x4<<2='d16, srca=0xC<<2='d48, dest=0x4<<2='d16
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0xC, 0x0, 0x6); // B1[8:15]*A3 srcb=0x6<<2='d24, srca=0xC<<2='d48, dest=0x6<<2='d24

                // [B3] x [A2 A3] (Accumulates to the result of [B2] x [A0 A1]  )
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0x8, 0x0, 0x8); // B3[0:7]*A2  srcb=0xC<<2='d48, srca=0x8<<2='d32, dest=0x8<<2='d32 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0x8, 0x0, 0xA); // B3[8:15]*A2 srcb=0xE<<2='d56, srca=0x8<<2='d32, dest=0xA<<2='d40 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0xC, 0x0, 0xC); // B3[0:7]*A3  srcb=0xC<<2='d48, srca=0xC<<2='d48, dest=0xC<<2='d48
            });
    }

    /* Just choose what is more readable*/
    constexpr static std::uint32_t matmul_op =
        ENABLE_2X_FORMAT ? TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x4, ADDR_MOD_1, 0xE)
                         :                                                     // B1[8:15]*A1 srcb=0x6<<2='d24, srca=0x4<<2='d16, dest=0xE<<2='d56
            TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0xC, ADDR_MOD_1, 0xE); // B3[8:15]*A3 srcb=0xE<<2='d56, srca=0xC<<2='d48, dest=0xE<<2='d56
    std::uint32_t matmul_op_last;
    if constexpr (ENABLE_2X_FORMAT)
    {
        matmul_op_last =
            reuse_a ? TT_OP_MVMULDI(p_setrwc::CLR_A, 0x0, 0x6, 0x4, ADDR_MOD_2, 0xE) : TT_OP_MVMULDI(p_setrwc::CLR_B, 0x0, 0x6, 0x4, ADDR_MOD_2, 0xE);
    }
    else
    {
        matmul_op_last =
            reuse_a ? TT_OP_MVMULDI(p_setrwc::CLR_A, 0x0, 0xE, 0xC, ADDR_MOD_2, 0xE) : TT_OP_MVMULDI(p_setrwc::CLR_B, 0x0, 0xE, 0xC, ADDR_MOD_2, 0xE);
    }

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

// Math-thread replay buffer depth. REPLAY_BUF_IDX_WIDTHS = '{6, 5, 6, 5} in tt_instruction_thread.sv,
// so TRISC1 (math) and TRISC3 hold 2^6 entries while TRISC0/TRISC2 hold 2^5.
constexpr std::uint32_t MATH_REPLAY_BUF_ENTRIES = 64;
// 8-row FPU: 16 MVMULDIs cover one 32x32 x 32x32 tile product per fidelity phase.
constexpr std::uint32_t DI_OPS_PER_TILE = 16;
// MVMULDI index unit is 4 rows and a tile is 64 rows, so one dest tile is 16 index units.
constexpr std::uint32_t DI_DEST_TILE_STRIDE = 16;
// Dest rows per 32x32 tile: 4 faces of 16 rows. SETRWC addresses dest in rows, not index units.
constexpr std::uint32_t DEST_TILE_ROWS = 64;

/**
 * @brief (srcb, srca, dest) index triples for one tile product, in 4-row units.
 */
struct di_tile_op_t
{
    std::uint8_t srcb;
    std::uint8_t srca;
    std::uint8_t dest;
};

// Ops 0-7 form the k0 partial products, ops 8-15 add the k1 faces into the same dest rows.
// Dest indices are even because the 8-row FPU writes 8 rows (2 index units) per op.
inline constexpr di_tile_op_t DI_TILE_OPS[DI_OPS_PER_TILE] = {
    {0x0, 0x0, 0x0},
    {0x2, 0x0, 0x2},
    {0x0, 0x4, 0x4},
    {0x2, 0x4, 0x6},
    {0x8, 0x0, 0x8},
    {0xA, 0x0, 0xA},
    {0x8, 0x4, 0xC},
    {0xA, 0x4, 0xE},
    {0x4, 0x8, 0x0},
    {0x6, 0x8, 0x2},
    {0x4, 0xC, 0x4},
    {0x6, 0xC, 0x6},
    {0xC, 0x8, 0x8},
    {0xE, 0x8, 0xA},
    {0xC, 0xC, 0xC},
    {0xE, 0xC, 0xE},
};

inline constexpr std::uint32_t _llk_matmul_fidelity_phases_(const ckernel::MathFidelity fidelity)
{
    return fidelity == ckernel::MathFidelity::LoFi ? 1 : to_underlying(fidelity);
}

/**
 * @brief How the dest-addressed DI stream covers one output block.
 *
 * A replayed instruction stream is fixed, so a tile's dest index can only differ from its
 * neighbour's by being written out as its own instruction: the tiles are unrolled, not looped.
 * The replay buffer therefore caps how many tiles one replay can cover, and blocks wider than
 * that are walked in several replays with the dest base moved in between.
 */
struct dest_di_plan_t
{
    std::uint32_t group_tiles; // output tiles covered by one replay
    std::uint32_t stream_len;  // MVMULDIs held in the replay buffer
    std::uint32_t t_dim;       // outer loop: the stationary dimension
    std::uint32_t rut_dim;     // inner loop: the reused dimension
    std::uint32_t dest_stride; // dest tiles between consecutive tiles of a group
    bool reuse_a;
    bool single_replay; // whole block in one replay: no dest base moves, held src released from the stream
};

/**
 * @brief Plans the dest-addressed DI stream for one block shape.
 *
 * Both halves of the op derive their loop structure from this, so @ref _llk_math_matmul_dest_di_mop_config_
 * and @ref _llk_math_matmul_block_dest_di_ cannot disagree about group size or dest layout.
 * A group never straddles the reuse loop, so its dest offsets repeat for every group and one
 * stream serves them all; group_tiles < 2 means the stream would cost more instructions than the
 * counter-driven MOP it replaces, and the caller should use @ref _llk_math_matmul_block_ instead.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls the number of FPU fidelity phases, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline dest_di_plan_t _llk_math_matmul_dest_di_plan_(const std::uint8_t ct_dim, const std::uint8_t rt_dim)
{
    constexpr std::uint32_t FIDELITY_PHASES = _llk_matmul_fidelity_phases_(MATH_FIDELITY_TYPE);
    // Tiles one replay can hold: 4 at LoFi, 2 at HiFi2, 1 at HiFi3/HiFi4.
    constexpr std::uint32_t BUDGET_TILES = MATH_REPLAY_BUF_ENTRIES / (FIDELITY_PHASES * DI_OPS_PER_TILE);

    dest_di_plan_t plan;
    plan.reuse_a     = ct_dim >= rt_dim;
    plan.t_dim       = plan.reuse_a ? rt_dim : ct_dim;
    plan.rut_dim     = plan.reuse_a ? ct_dim : rt_dim;
    plan.dest_stride = plan.reuse_a ? 1 : ct_dim;

    std::uint32_t group = (BUDGET_TILES < plan.rut_dim) ? BUDGET_TILES : plan.rut_dim;
    while (group > 1 && (plan.rut_dim % group) != 0)
    {
        group--;
    }
    plan.group_tiles   = group;
    plan.stream_len    = group * FIDELITY_PHASES * DI_OPS_PER_TILE;
    plan.single_replay = (plan.t_dim == 1) && (group == plan.rut_dim);
    return plan;
}

/**
 * @brief Reports whether the dest-addressed stream is worth using over the counter-driven MOP.
 *
 * Only when one replay covers the whole block. Measurement (docs/llk/l3/matmul_direct_indexing_quasar.md
 * sections 7.4-7.5) shows cost tracks the number of replay invocations rather than the number of
 * instructions: a single replay replacing several MOP replays wins 1.00 cyc/tile at HiFi2-HiFi4,
 * while the multi-group form loses 0.13-0.51 cyc/tile because each group's replay costs more than
 * the MOP trigger it removes, despite issuing fewer instructions.
 *
 * The grouped path stays implemented and covered by test_matmul_dest_di_grouped_quasar so the
 * decision can be revisited on silicon; it is disabled here by measurement, not by defect.
 *
 * @param plan: Stream plan from @ref _llk_math_matmul_dest_di_plan_
 */
inline bool _llk_math_matmul_dest_di_beneficial_(const dest_di_plan_t& plan)
{
    return plan.single_replay;
}

/**
 * @brief Initializes addrmod for the dest-addressed direct-indexing matrix multiply.
 *
 * Every dest index is encoded in its own MVMULDI, so no addr_mod moves a counter; the only
 * addr_mod work left is advancing and clearing the fidelity counter.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls the number of FPU fidelity phases, values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_matmul_dest_di_addrmod_()
{
    constexpr bool high_fidelity     = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT = high_fidelity ? 1 : 0;

    // Every replayed op selects this one: srca/srcb/dest all come from the instruction.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // Last op of a fidelity phase that is not the last phase of its tile.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_1);

    // Last op of a tile: clear the fidelity counter. Dest does NOT advance -- the next tile
    // carries its own dest index, which is the point of this path.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Loads one replay group of the dest-addressed direct-indexing instruction stream.
 *
 * Emits group_tiles * fidelity_phases * 16 MVMULDIs, tile major with the fidelity phases inside
 * each tile so a tile's src operands are released only once, after its last phase. Each op carries
 * its dest index in the 8-bit dst field, relative to the group's first tile, so the same stream
 * serves every group of the block and the execute half only moves the dest base between them.
 *
 * When one replay covers the whole block the held operand is released from the stream too
 * (CLR_AB on the final tile); otherwise the execute half releases it once per reuse loop, because
 * a shared stream cannot know which group ends one.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls the number of FPU fidelity phases, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note @ref _llk_math_matmul_block_dest_di_ is the matching execute call; both derive their
 * structure from @ref _llk_math_matmul_dest_di_plan_.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_matmul_dest_di_mop_config_(const std::uint8_t ct_dim, const std::uint8_t rt_dim)
{
    constexpr std::uint32_t FIDELITY_PHASES = _llk_matmul_fidelity_phases_(MATH_FIDELITY_TYPE);

    const dest_di_plan_t plan = _llk_math_matmul_dest_di_plan_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
    // Streamed operand is released per tile; the held one only at the end of a reuse loop.
    const std::uint32_t stream_clear = plan.reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B;

    load_replay_buf(
        0 /*start*/,
        plan.stream_len,
        false /*exec_while_loading*/,
        0 /*set_mutex*/,
        0 /*last*/,
        [&]
        {
            for (std::uint32_t k = 0; k < plan.group_tiles; k++)
            {
                // Dest offset within the group: contiguous when input A is reused,
                // ct_dim-strided (column walk) otherwise. The output is row major either way.
                const std::uint32_t dest_tile = k * plan.dest_stride;
                const bool ends_reuse_loop    = plan.single_replay && (k == (plan.group_tiles - 1));
                const std::uint32_t clear_src = ends_reuse_loop ? p_setrwc::CLR_AB : stream_clear;

                for (std::uint32_t phase = 0; phase < FIDELITY_PHASES; phase++)
                {
                    const bool last_phase = (phase == (FIDELITY_PHASES - 1));

                    for (std::uint32_t op = 0; op < DI_OPS_PER_TILE; op++)
                    {
                        const bool last_op_of_phase  = (op == (DI_OPS_PER_TILE - 1));
                        const std::uint32_t addr_mod = last_op_of_phase ? (last_phase ? ADDR_MOD_2 : ADDR_MOD_1) : ADDR_MOD_0;

                        TT_MVMULDI(
                            (last_op_of_phase && last_phase) ? clear_src : p_setrwc::CLR_NONE,
                            0x0 /*ins_mod*/,
                            DI_TILE_OPS[op].srcb,
                            DI_TILE_OPS[op].srca,
                            addr_mod,
                            dest_tile * DI_DEST_TILE_STRIDE + DI_TILE_OPS[op].dest);
                    }
                }
            }
        });
}

/**
 * @brief Initializes addrmod and config for matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_DIRECT_INDEXING: Enable direct indexing matrix multiplication
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @tparam ENABLE_DEST_DIRECT_ADDRESSING: Encode the dest tile index in every MVMULDI and unroll the
 * whole block into the replay buffer, so no instruction is spent on dest addressing. Requires
 * ENABLE_DIRECT_INDEXING; falls back to the plain DI path for blocks that do not fit (see
 * @ref _llk_math_matmul_dest_di_plan_).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note On the unpack thread, pair with @ref _llk_unpack_matmul_init_ (T0); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_matmul_tile_ or @ref _llk_math_matmul_block_ runs the configured matmul with matching template args.
 * @note With ENABLE_DEST_DIRECT_ADDRESSING, run @ref _llk_math_matmul_block_dest_di_ instead of @ref _llk_math_matmul_block_.
 */

template <
    ckernel::MathFidelity MATH_FIDELITY_TYPE,
    bool ENABLE_DIRECT_INDEXING        = false,
    bool ENABLE_2X_FORMAT              = false,
    bool ENABLE_DEST_DIRECT_ADDRESSING = false>
inline void _llk_math_matmul_init_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    if constexpr (ENABLE_DEST_DIRECT_ADDRESSING)
    {
        static_assert(ENABLE_DIRECT_INDEXING, "Dest direct addressing is a direct-indexing (MVMULDI) path");
        static_assert(!ENABLE_2X_FORMAT, "Dest direct addressing is implemented for the non-2x DI stream only");

        if (_llk_math_matmul_dest_di_beneficial_(_llk_math_matmul_dest_di_plan_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim)))
        {
            _llk_math_matmul_dest_di_addrmod_<MATH_FIDELITY_TYPE>();
            _llk_math_matmul_dest_di_mop_config_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
            _reset_counters_<p_setrwc::SET_ABD_F>();
            return;
        }
    }

    if constexpr (ENABLE_DIRECT_INDEXING)
    {
        // Direct-indexing path. Supports plain DI and DI+X2 (DI+X2 is the original
        // MXFP4_2x matmul implementation).
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
        _llk_math_matmul_di_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
    }
    else
    {
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
        _llk_math_matmul_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
    }

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA for a single tile.
 *
 * Input 0 = 1 tile -> SrcB reg, Input 1 = 1 tile -> SrcA reg, output = 1 tile -> Dst reg at specified dst_index.
 *
 * @param dst_index: Tile index in destination register. For DstSync::SyncHalf: values = [0-7] for 16-bit formats, values = [0-3] for 32-bit formats. For
 * DstSync::SyncFull: values = [0-15] for 16-bit formats, values = [0-7] for 32-bit formats
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this function.
 */
inline void _llk_math_matmul_tile_(const std::uint32_t dst_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA over a block of tiles.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * IMPORTANT NOTES:
 * 1. Dest index always assumed to start at 0 for this operation.
 * 2. If matrix multiplication includes kt_dim > 1 such that matrix multiplication is:
 *    Input 0 [rt_dim, kt_dim] x Input 1 [kt_dim, ct_dim] = Output [rt_dim, ct_dim],
 *    be aware that this function does not iterate over kt_dim; iterate over kt_dim externally to this function.
 *
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this function.
 */
inline void _llk_math_matmul_block_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // Matmul Block, reset the dest addr to 0 for fused kernels
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // reuse-dim

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            ckernel_template::run_bank0_sw_cntl(instrn_buffer);

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
            }
        }

        //  When rt_dim > ct_dim, the matmul block dest tile indices are not equal to 0,1,2,3..7
        //  Instead they have a ct_dim stride, for instance:
        //  If rt_dim = 4, ct_dim = 2, dest tile indices = 0,2,4,6,  1,3,5,7
        //  If rt_dim = 4, ct_dim = 3, dest tile indices = 0,3,6,9,  1,4,7,10,  2,5,8,11
        //  Below offsets by 1 tile * (t+1), for every subsequence above to start from the next dest_idx
        if (!reuse_a && ct_dim >= 2)
        {
            TT_SETRWC(p_setrwc::CLR_NONE, 0, 64 * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Does the block matrix multiply as a single dest-addressed direct-indexing replay.
 *
 * Every dest index lives in its MVMULDI, so the whole [rt_dim, ct_dim] block issues as one REPLAY:
 * no per-tile MOP restart, no dest carriage-return fixups for the ct_dim-strided column walk, and
 * no separate SETRWC to release the sources (the stream's last op per tile carries clear_dvalid).
 * Blocks whose stream exceeds the replay buffer fall back to @ref _llk_math_matmul_block_.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls the number of FPU fidelity phases, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note Call @ref _llk_math_matmul_init_ with ENABLE_DEST_DIRECT_ADDRESSING and matching template args first.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_matmul_block_dest_di_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    const dest_di_plan_t plan = _llk_math_matmul_dest_di_plan_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);

    if (!_llk_math_matmul_dest_di_beneficial_(plan))
    {
        _llk_math_matmul_block_(ct_dim, rt_dim);
        return;
    }

    // Dest base still comes from config: it carries the DstSync bank offset.
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);

    if (plan.single_replay)
    {
        // Whole block in one replay: dest indices are absolute, sources released from the stream.
        TT_REPLAY(0 /*start*/, plan.stream_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load_mode: replay*/);
        _reset_counters_<p_setrwc::SET_ABD_F>();
        return;
    }

    const std::uint32_t held_clear = plan.reuse_a ? p_setrwc::CLR_B : p_setrwc::CLR_A;

    for (std::uint32_t t = 0; t < plan.t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < plan.rut_dim; rut += plan.group_tiles)
        {
            // The stream's dest indices are relative to the group, so point the dest counter at
            // the group's first tile and replay. Dest never moves inside a group.
            const std::uint32_t base_tile = plan.reuse_a ? (t * ct_dim + rut) : (rut * ct_dim + t);
            TT_SETRWC(p_setrwc::CLR_NONE, 0, base_tile * DEST_TILE_ROWS, p_setrwc::SET_D);
            TT_REPLAY(0 /*start*/, plan.stream_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load_mode: replay*/);
        }

        // Release the reused operand once per reuse loop; the streamed one is released per tile
        // by the stream itself.
        TT_SETRWC(held_clear, 0, 0, p_setrwc::SET_AB_F);
    }

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
