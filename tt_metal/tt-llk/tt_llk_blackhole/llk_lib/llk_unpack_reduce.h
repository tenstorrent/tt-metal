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
    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_zerosrca =
        TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

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
 * Configures the SrcB format registers, sets haloize (transpose) mode
 * according to the reduce dimension and the within-face transpose flag, programs the datum count,
 * and programs the reduce MOP.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param unpB_src_format: Source data format of the scaler operand (SrcB) in L1.
 * @param unpB_dst_format: Destination data format the scaler operand is converted to.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose.
 * @note REDUCE_ROW is supported for MAX only; REDUCE_ROW SUM/AVG must use the AB reduce path (static_assert).
 * @note The scaler is a single row: the first FACE_C_DIM datums of face 0 of the scaler tile, reused for every face.
 * @ref _llk_unpack_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (single-operand unpack pairing).
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t unpB_src_format, const std::uint32_t unpB_dst_format, const std::uint32_t within_face_16x16_transpose = 0)
{
    // This path's MOP clears SrcA to zero for every pool type, which a min reduce would lose to.
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    // This path always unpacks the data tile into SrcA and the scaler into SrcB; the REDUCE_ROW SUM/AVG math
    // expects the reverse layout (data in SrcB, scaler column in SrcA), which only the AB reduce unpack provides.
    static_assert(
        !(dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX),
        "REDUCE_ROW SUM/AVG needs the data tile in SrcB; use _llk_unpack_AB_reduce_init_ / _llk_unpack_AB_reduce_.");

    // Configure SrcB format registers
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xf>(unpB_src_format);

    // Set FP8 E4M3 mode for SrcB; selects the e4m3 (vs e5m2) exponent layout and clears any stale 4b-exp setting.
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(((unpB_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpB_dst_format);

    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(ReduceDim::REDUCE_ROW == dim ? !within_face_16x16_transpose : within_face_16x16_transpose);

    TTI_SETADCXX(0b11, FACE_R_DIM * FACE_C_DIM - 1, 0x0);

    _llk_unpack_reduce_mop_config_<type, dim>();
}

/**
 * @brief Unpack a tile for a reduce-with-scaler operation.
 *
 * Programs the operand (SrcA) and scaler (SrcB) base addresses, narrows SrcB to a single 16-datum
 * row for the scaler, runs the reduce MOP, restores the SrcB datum count, and switches config context.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address: L1 address of the source tile.
 * @param scaler_address: L1 address of the scaler tile. Only its first FACE_C_DIM datums (face 0, row 0) are
 *        unpacked into SrcB, and that one row is reused for every face of the data tile.
 * @note Call @ref _llk_unpack_reduce_init_ with matching template args before this function.
 * @note The scaler is an ordinary L1 tile (e.g. a circular-buffer tile) owned by the caller.
 * @note Per-face or partial scaler tiles (different values in different faces) are not supported on this path;
 *       use the AB reduce (_llk_unpack_AB_reduce_) for those.
 */
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_(const std::uint32_t address, const std::uint32_t scaler_address)
{
    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure the operand (SrcA) and scaler (SrcB) addresses
    _llk_unpack_configure_addresses_(address, scaler_address, cfg);

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
