// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/_stdint.h>

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Configures the unpacker MOP for reduction operations. Handles both tiny tiles (face_r_dim < 16) and standard tiles.
 *
 * @note For full 4-face SUM/AVG REDUCE_ROW, a single UNPACR path unpacks all faces in a single instruction.
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

    constexpr bool is_max                  = pool_type == PoolType::MAX;
    constexpr bool swap_operands           = (reduce_dim == ReduceDim::REDUCE_ROW) && !is_max;
    constexpr bool is_scalar               = reduce_dim == ReduceDim::REDUCE_SCALAR;
    constexpr std::uint32_t REPLAY_BUF_LEN = 2;
    constexpr std::uint32_t clear_src      = swap_operands ? Srcs::SrcB : Srcs::SrcA;

    const bool full_tile          = swap_operands && (tensor_shape.total_num_faces() == 4);
    const bool is_tiny            = tensor_shape.face_r_dim < FACE_R_DIM;
    const std::uint32_t innerloop = full_tile ? 1 : tensor_shape.total_num_faces();
    const std::uint32_t clear_val = is_max ? p_unpacr_nop::CLR_SRC_NEGINF : p_unpacr_nop::CLR_SRC_0;

    load_replay_buf(
        0,
        REPLAY_BUF_LEN,
        [full_tile]
        {
            if (full_tile)
            {
                TTI_UNPACR(Srcs::SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR(Srcs::SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            }
            else
            {
                TTI_UNPACR(Srcs::SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR(Srcs::SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            }
        });

    const std::uint32_t replay = lltt::replay_insn(0, REPLAY_BUF_LEN);

    if (is_tiny || is_scalar)
    {
        ckernel_template tmp(1, innerloop, TT_OP_UNPACR_NOP(clear_src, 0, 0, 0, 0, 0, 0, clear_val, p_unpacr_nop::CLR_SRC), replay);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, innerloop, replay);
        tmp.program();
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

    constexpr bool is_max        = pool_type == PoolType::MAX;
    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && !is_max;

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>((reduce_dim == ReduceDim::REDUCE_ROW));

    const bool full_tile = swap_operands && (tensor_shape.total_num_faces() == 4);

    if (full_tile)
    {
        const std::uint32_t x_end = tensor_shape.total_num_faces() * FACE_R_DIM * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_A, x_end, 0x0);
        TT_SETADCXX(p_setadc::UNP_B, x_end, 0x0);
    }
    else
    {
        config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape.face_r_dim);
        config_unpacker_x_end<p_setadc::UNP_B>(swap_operands ? tensor_shape.face_r_dim : 1);
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

    constexpr bool is_max        = pool_type == PoolType::MAX;
    constexpr bool swap_operands = (reduce_dim == ReduceDim::REDUCE_ROW) && !is_max;

    // SUM/AVG REDUCE_ROW swaps operands (scaler→SrcA, data→SrcB), no transpose needed.
    // MAX REDUCE_ROW keeps original layout (data→SrcA transposed, scaler→SrcB) — GMPOOL only reads SrcA.
    const std::uint32_t addr_unp_a = swap_operands ? address_b : address_a;
    const std::uint32_t addr_unp_b = swap_operands ? address_a : address_b;
    _llk_unpack_configure_addresses_(addr_unp_a, addr_unp_b, cfg);

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
