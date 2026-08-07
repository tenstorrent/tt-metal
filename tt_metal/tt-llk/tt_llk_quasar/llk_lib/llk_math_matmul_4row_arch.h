// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Initializes addrmod for matrix multiply operation
 * @tparam MATH_FIDELITY: 0 = LoFi, 1 = Hifi2, 2 = HiFi3, 3 = HiFi4 - controls precision of multiplication when math is Float32 format
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, std::uint8_t CT_DIM, std::uint8_t RT_DIM>
inline void _llk_math_matmul_addrmod_()
{
    constexpr bool high_fidelity          = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT      = high_fidelity ? 1 : 0;
    constexpr std::uint16_t num_tile_incr = (CT_DIM >= RT_DIM) ? 64 : CT_DIM * 64;

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 4, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 1},
        .dest = {.incr = 4, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 4, .clr = 0, .cr = 0},
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

// Direct Indexing Method
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, std::uint8_t CT_DIM, std::uint8_t RT_DIM>
inline void _llk_math_matmul_di_addrmod_()
{
    constexpr bool high_fidelity          = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT      = high_fidelity ? 1 : 0;
    constexpr std::uint16_t num_tile_incr = (CT_DIM >= RT_DIM) ? 64 : CT_DIM * 64;

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
 * @brief Initializes mop config for matrix multiply operation
 * Input 0 dim = [rt_dim, 1]
 * Input 1 dim = [1, ct_dim]
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @tparam MATH_FIDELITY: 0 = LoFi, 1 = Hifi2, 2 = HiFi3, 3 = HiFi4 - controls precision of multiplication when math is Float32 format
 * @tparam CT_DIM: number of tiles in the column dimension for a matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, std::uint8_t CT_DIM, std::uint8_t RT_DIM>
inline void _llk_math_matmul_mop_config_()
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr int FIDELITY_PHASES = static_cast<std::uint32_t>(MATH_FIDELITY_TYPE) + 1;

    constexpr bool reuse_a = CT_DIM >= RT_DIM;

    constexpr std::uint32_t replay_buf_len = 32 - 1;

    load_replay_buf<0, replay_buf_len>(
        // Lambda function to load reply buffer
        []
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        });

    constexpr static std::uint32_t matmul_op      = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
    constexpr static std::uint32_t matmul_op_last = reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_3, 0);

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes mop config for matrix multiply operation with direct indexing matmul
 * @tparam MATH_FIDELITY: 0 = LoFi, 1 = Hifi2, 2 = HiFi3, 3 = HiFi4 - controls precision of multiplication when math is Float32 format
 * @tparam CT_DIM: number of tiles in the column dimension for a matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for a matrix multiply
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, std::uint8_t CT_DIM, std::uint8_t RT_DIM, bool EN_X2>
inline void _llk_math_matmul_di_mop_config_()
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr int FIDELITY_PHASES = static_cast<std::uint32_t>(MATH_FIDELITY_TYPE) + 1;
    constexpr bool reuse_a        = CT_DIM >= RT_DIM;

    constexpr std::uint32_t replay_buf_len = EN_X2 ? 16 - 1 : 32 - 1; // -1 since the last instruction for the Tile * Tile operation will come out of the MOP
    if constexpr (EN_X2)
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x1, 0x0, 0x0, 0x1);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x3, 0x0, 0x0, 0x3);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x1, 0x4, 0x0, 0x5);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x3, 0x4, 0x0, 0x7);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x0, 0x0, 0x8);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x5, 0x0, 0x0, 0x9);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x0, 0x0, 0xA);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x7, 0x0, 0x0, 0xB);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x4, 0x0, 0xC);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x5, 0x4, 0x0, 0xD);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x4, 0x0, 0xE);
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x1, 0x0, 0x0, 0x1);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x3, 0x0, 0x0, 0x3);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x1, 0x4, 0x0, 0x5);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x3, 0x4, 0x0, 0x7);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x0, 0x0, 0x8);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x9, 0x0, 0x0, 0x9);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x0, 0x0, 0xA);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xB, 0x0, 0x0, 0xB);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x4, 0x0, 0xC);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x9, 0x4, 0x0, 0xD);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x4, 0x0, 0xE);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xB, 0x4, 0x0, 0xF);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x8, 0x0, 0x0);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x5, 0x8, 0x0, 0x1);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x8, 0x0, 0x2);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x7, 0x8, 0x0, 0x3);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0xC, 0x0, 0x4);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x5, 0xC, 0x0, 0x5);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0xC, 0x0, 0x6);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x7, 0xC, 0x0, 0x7);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0x8, 0x0, 0x8);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xD, 0x8, 0x0, 0x9);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0x8, 0x0, 0xA);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xF, 0x8, 0x0, 0xB);

                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0xC, 0x0, 0xC);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xD, 0xC, 0x0, 0xD);
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0xC, 0x0, 0xE);
            });
    }

    constexpr static std::uint32_t matmul_op =
        EN_X2 ? TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x7, 0x4, ADDR_MOD_1, 0xF) : TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xF, 0xC, ADDR_MOD_1, 0xF);
    constexpr static std::uint32_t CLEAR_SRC = reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B;
    constexpr static std::uint32_t matmul_op_last =
        EN_X2 ? TT_OP_MVMULDI(CLEAR_SRC, 0x0, 0x7, 0x4, ADDR_MOD_2, 0xF) : TT_OP_MVMULDI(CLEAR_SRC, 0x0, 0xF, 0xC, ADDR_MOD_2, 0xF);

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes addrmod and config for matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA
 * Input 0 dim = [rt_dim, 1]
 * Input 1 dim = [1, ct_dim]
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @tparam MATH_FIDELITY: 0 = LoFi, 1 = Hifi2, 2 = HiFi3, 3 = HiFi4 - controls precision of multiplication when math is Float32 format
 * @tparam CT_DIM: number of tiles in the column dimension for a matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for a matrix multiply
 * @tparam EN_DI: Enable direct indexing matrix multiplication
 * @tparam EN_X2: Enable matrix multiplication with MXFP_2X mode, double the performance
 */

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, std::uint8_t CT_DIM, std::uint8_t RT_DIM, bool EN_DI, bool EN_X2>
inline void _llk_math_matmul_init_()
{
    if constexpr (EN_DI || EN_X2)
    {
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE, CT_DIM, RT_DIM>();
        _llk_math_matmul_di_mop_config_<MATH_FIDELITY_TYPE, CT_DIM, RT_DIM, EN_X2>();
    }
    else
    {
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, CT_DIM, RT_DIM>();
        _llk_math_matmul_mop_config_<MATH_FIDELITY_TYPE, CT_DIM, RT_DIM>();
    }

    // Matmul Block, reset the dest addr to 0 for fused kernels
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA
 * Input 0 = 1 tile -> SrcB reg
 * Input 1 = 1 tile -> SrcA reg
 * Output = 1 tile -> Dst reg at specified dst_index
 * @param dst_index: tile index in destination register, values = [0-8] for Float16b, values = [0-4] for Float32
 */
inline void _llk_math_matmul_tile_(const std::uint32_t dst_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA
 * Input 0 dim = [rt_dim, 1]
 * Input 1 dim = [1, ct_dim]
 * Output is a matrix block of dimension [rt_dim, ct_dim]
 * IMPORTANT NOTES:
 * 1. Dest index always assumed to start at 0 for this operation
 * 2. If matrix multiplication includes kt_dim > 1 such that matrix multiplication is:
 * Input 0 [rt_dim, kt_dim] x Input 1 [kt_dim, ct_dim] = Output [rt_dim, ct_dim].
 * Be Aware: this function does not iterate over kt_dim, must iterate over kt_dim externally to this function
 * @tparam CT_DIM: number of tiles in the column dimension for a matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for a matrix multiply
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 */
template <std::uint8_t CT_DIM, std::uint8_t RT_DIM>
inline void _llk_math_matmul_block_()
{
    constexpr bool reuse_a          = CT_DIM >= RT_DIM;
    constexpr std::uint32_t t_dim   = reuse_a ? RT_DIM : CT_DIM;
    constexpr std::uint32_t rut_dim = reuse_a ? CT_DIM : RT_DIM; // reuse-dim

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            ckernel_template::run_bank0_sw_cntl(instrn_buffer);

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if constexpr (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
            }
        }

        // There are only 2 scenarios when rt_dim > ct_dim, and ct_dim = 2:
        //  rt_dim = 4, ct_dim = 2
        //  rt_dim = 3, ct_dim = 2
        //  These are the only scenarios where the matmul block dest tile indices are not equal to 0,1,2,3..7
        //  The above scenarios have dest tile indices = 0,2,4,1,3,5 or 0,2,4,6,1,3,5,7
        //  Below offsets by 1 tile, for the sequence above to start from 1
        if constexpr (!reuse_a && CT_DIM == 2)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 64, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
