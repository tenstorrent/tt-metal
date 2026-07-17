// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file llk_unpack_AB_reduce.h
 * @brief Unpacker kernel for reduction operations
 *
 * This file provides template functions for unpacking and configuring reduction operations
 * on TT-Wormhole B0. It supports various pooling types and reduction dimensions,
 * handling both standard and tiny tile configurations.
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "lltt.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Configures the unpacker MOP for reduction operations. Handles both tiny tiles (face_r_dim < 16) and standard tiles.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the tensor, including face_r_dim and num_faces.
 *
 * @note For tiny tiles (face_r_dim < 16), padding is applied to prevent incorrect outputs.
 * @note For REDUCE_SCALAR operations, SrcA is cleared before unpacking because SrcA is clobbered in the Math kernel.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const ckernel::TensorShape &tensor_shape)
{
    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape);

    // Data valid for clear instructions is set to 0 since the MATH kernel should not process this data.
    // pool_type == PoolType::MAX sets the clear value to neginf if the pool-type is MAX and 0 if the pool-type is AVG/SUM
    static constexpr std::uint32_t clear_pool_dep_srca =
        TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, (pool_type == PoolType::MAX) ? p_unpacr_nop::UNP_NEGINFSRC : p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t clear_pool_dep_srcb =
        TT_OP_UNPACR_NOP(p_unpacr_nop::UNP1, (pool_type == PoolType::MAX) ? p_unpacr_nop::UNP_NEGINFSRC : p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t clear_zero_srca = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);

    constexpr std::uint32_t REPLAY_BUF_LEN = 2;

    // Configure unpacker instruction for Src{A,B}. These instructions always increment L1 by 1 face.
    lltt::record<lltt::NoExec>(0, REPLAY_BUF_LEN);
    TTI_UNPACR(Srcs::SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Unpack SrcA
    TTI_UNPACR(Srcs::SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Unpack SrcB

    // MOP constants
    constexpr std::uint32_t outerloop = 1;
    const std::uint32_t innerloop     = tensor_shape.total_num_faces();

    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);

    // Padding should only be done when using tiny tiles otherwise the entire face overwrites the data read in Math
    if (tensor_shape.face_r_dim < FACE_R_DIM) // Using tiny faces
    {
        // Swapped REDUCE_ROW: data is in SrcB, pad SrcB; otherwise data is in SrcA, pad SrcA
        if constexpr (swap_operands)
        {
            ckernel_template tmp(outerloop, innerloop, clear_pool_dep_srcb, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, clear_pool_dep_srca, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
    }
    else // Using standard faces (face_r_dim = FACE_R_DIM)
    {
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR)
        {
            // For scalar reduction, clear SrcA to zero before unpacking. SrcA is clobbered in Math kernel.
            ckernel_template tmp(outerloop, innerloop, clear_zero_srca, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
        else
        {
            // For row/column reduction, no clearing needed
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize the unpacker for reduce operations
 *
 * Configures the unpacker hardware registers and MOP settings
 * for reduction operations. This includes:
 * - Setting up haloize mode for row reductions (transpose)
 * - Configuring unpacker X dimension endpoints
 * - Calling the MOP configuration routine
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the tensor, including face_r_dim and num_faces.
 *
 * @note For SUM/AVG REDUCE_ROW, operands are swapped: scaler→SrcA, data→SrcB.
 * @note For MAX REDUCE_ROW, original layout is kept: data→SrcA (transposed via haloize), scaler→SrcB.
 * @note For REDUCE_COL/REDUCE_SCALAR: Unpacker 0 (SrcA) reads face_r_dim*FACE_R_DIM datums,
 *       Unpacker 1 (SrcB) reads one row (FACE_R_DIM datums).
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (this is the scaler operand unpack pairing).
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape &tensor_shape)
{
    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape);

    // SUM/AVG REDUCE_ROW swaps operands (scaler→SrcA, data→SrcB), no transpose needed.
    // MAX REDUCE_ROW keeps original layout (data→SrcA transposed, scaler→SrcB) — GMPOOL only reads SrcA.
    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>((reduce_dim == ReduceDim::REDUCE_ROW));

    config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape.face_r_dim);

    // UNP_B reads data faces (face_r_dim rows) in swapped mode, or a single
    // scaler row in the non-swapped (MAX / COL / SCALAR) mode.
    if constexpr (swap_operands)
    {
        config_unpacker_x_end<p_setadc::UNP_B>(tensor_shape.face_r_dim);
    }
    else
    {
        config_unpacker_x_end<p_setadc::UNP_B>(1);
    }

    // Configure unpack MOP
    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(tensor_shape);
}

/**
 * @brief Execute the unpacker for reduction operations
 *
 * Performs the actual unpacking of data for reduction operations by:
 * 1. Resetting address counters
 * 2. Programming source A and B base addresses in hardware registers
 * 3. Synchronizing with Trisc using semaphores
 * 4. Running the configured MOP
 * 5. Switching unpacker configuration context
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address_a: Base address for source A data in L1 memory.
 * @param address_b: Base address for source B data in L1 memory.
 *
 * @note Call @ref _llk_unpack_AB_reduce_init_ with matching template args before this function.
 * @note This function manages dual-context switching for pipelined execution.
 * @note Semaphores ensure proper synchronization between Trisc and unpacker.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    // Reset address counters for both unpackers
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    // Get pointer to configuration registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // Wait for free context
    wait_for_next_context(2);

    // SUM/AVG REDUCE_ROW: swap operands so scaler→SrcA, data→SrcB
    // MAX REDUCE_ROW: keep original order (data→SrcA, scaler→SrcB)
    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
    if constexpr (swap_operands)
    {
        _llk_unpack_configure_addresses_(address_b, address_a, cfg);
    }
    else
    {
        _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Execute the configured MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
