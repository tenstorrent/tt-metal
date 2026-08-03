// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Program the unpacker MOP for a reduce-with-scaler operation.
 *
 * Sets up the UNPACR sequence that zeroes SrcA, unpacks the operand into SrcA, and loads the
 * scaler row into SrcB across the four-unpack halo template.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_mop_config_()
{
    static constexpr std::uint32_t unpack_srca     = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_zerosrca = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);
    static constexpr std::uint32_t unpack_srcb     = TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true, // src B
        true, // halo - just used for 4 unpacks
        unpack_zerosrca,
        unpack_srca,
        TT_OP_NOP,
        TT_OP_NOP,
        0,
        unpack_srcb,
        0);
    tmp.program();
}

/**
 * @brief Initialize the unpacker for a reduce-with-scaler operation.
 *
 * Configures the SrcB format registers and scaler L1 base address, sets haloize (transpose) mode
 * according to the reduce dimension and the within-face transpose flag, programs the datum count,
 * and programs the reduce MOP.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param unpB_src_format: Source data format of the scaler operand (SrcB) in L1.
 * @param unpB_dst_format: Destination data format the scaler operand is converted to.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose.
 * @ref _llk_unpack_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (single-operand unpack pairing).
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t unpB_src_format, const std::uint32_t unpB_dst_format, const std::uint32_t within_face_16x16_transpose = 0)
{
    // Configure SrcB format registers
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xf>(unpB_src_format);

    // Set FP8 E4M3 mode for SrcB; selects the e4m3 (vs e5m2) exponent layout and clears any stale 4b-exp setting.
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(((unpB_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpB_dst_format);

    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_address_ADDR32);
    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
    TTI_NOP;

    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(ReduceDim::REDUCE_ROW == dim ? !within_face_16x16_transpose : within_face_16x16_transpose);

    TTI_SETADCXX(0b11, FACE_R_DIM * FACE_C_DIM - 1, 0x0);

    _llk_unpack_reduce_mop_config_<type, dim>();
}

/**
 * @brief Unpack a tile for a reduce-with-scaler operation.
 *
 * Programs the operand base address, temporarily narrows SrcB to a single 16-datum row for the
 * scaler, runs the reduce MOP, restores the SrcB datum count, and switches config context.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address: L1 address of the source tile.
 * @note Call @ref _llk_unpack_reduce_init_ with matching template args before this function.
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_(const std::uint32_t address)
{
    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure address
    _llk_unpack_configure_single_address_(address, cfg);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Load only 16 datums into srcB
    TTI_SETADCXX(p_setadc::UNP1, FACE_C_DIM - 1, 0x0);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    mop_run(0, 4);

    // Restore face height
    TTI_SETADCXX(p_setadc::UNP1, FACE_R_DIM * FACE_C_DIM - 1, 0x0);

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
