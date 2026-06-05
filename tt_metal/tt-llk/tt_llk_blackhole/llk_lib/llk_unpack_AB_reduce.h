// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file llk_unpack_AB_reduce.h
 * @brief Unpacker kernel for reduction operations
 *
 * This file provides template functions for unpacking and configuring reduction operations
 * on TT-Blackhole. It supports various pooling types and reduction dimensions,
 * handling both standard and tiny tile configurations.
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Initialize the unpacker for reduce operations
 *
 * Configures the unpacker hardware registers for reduction operations:
 * - Setting up haloize mode for row reductions (transpose)
 * - Configuring unpacker X dimension endpoints
 *
 * @tparam pool_type The type of pooling operation (MAX, SUM, AVG)
 * @tparam reduce_dim The dimension along which to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 *
 * @param tensor_shape The shape of the tensor, including face_r_dim and num_faces
 *
 * @note For REDUCE_ROW operations, the face is transposed using haloize mode
 * @note Unpacker 0 (SrcA) reads face_r_dim*FACE_R_DIM datums
 * @note Unpacker 1 (SrcB) reads one row (FACE_R_DIM datums)
 */
template <PoolType pool_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape &tensor_shape)
{
    // Validate tensor shape for tile-dependent operations
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    if constexpr (enforce_fp32_accumulation)
    {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }

    // Enable transpose (haloize mode) if reducing along rows
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(reduce_dim == ReduceDim::REDUCE_ROW);

    // Sets up Unpacker 0 to read face_r_dim*16 datums into SrcA register
    config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape.face_r_dim);

    // Sets up Unpacker 1 to read one row (16 datums) into SrcB register
    config_unpacker_x_end<p_setadc::UNP_B>(1);
}

/**
 * @brief Execute the unpacker for reduction operations using raw instructions (no MOP/replay)
 *
 * Performs the actual unpacking of data for reduction operations by:
 * 1. Resetting address counters
 * 2. Programming source A and B base addresses in hardware registers
 * 3. Synchronizing with Trisc using semaphores
 * 4. Issuing raw unpack instructions in a per-face loop
 * 5. Switching unpacker configuration context
 *
 * @tparam pool_type The type of pooling operation (MAX, SUM, AVG)
 * @tparam reduce_dim The dimension along which to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 *
 * @param address_a Base address for source A data in L1 memory
 * @param address_b Base address for source B data in L1 memory
 * @param tensor_shape The shape of the tensor (needed for face count and tiny-tile detection)
 *
 * @note This function manages dual-context switching for pipelined execution
 * @note Semaphores ensure proper synchronization between Trisc and unpacker
 * @note For tiny tiles (face_r_dim < 16), SrcA is cleared before each face to pad with pool-dependent values
 * @note For REDUCE_SCALAR with standard tiles, SrcA is cleared to zero before each face
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_(const std::uint32_t address_a, const std::uint32_t address_b, const ckernel::TensorShape &tensor_shape)
{
    // Reset address counters for both unpackers
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Get pointer to configuration registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        if (tensor_shape.face_r_dim < FACE_R_DIM)
        {
            // Tiny tiles: clear SrcA with pool-type dependent value (0 or neginf) to pad unused rows
            TTI_UNPACR_NOP(Srcs::SrcA, 0, 0, 0 /* dvalid */, 0, 0, 0, pool_type == PoolType::MAX, p_unpacr_nop::CLR_SRC);
        }
        else if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR)
        {
            // Scalar reduction: clear SrcA to zero (SrcA is clobbered in Math kernel)
            TTI_UNPACR_NOP(Srcs::SrcA, 0, 0, 0 /* dvalid */, 0, 0, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);
        }

        // Unpack SrcA — increments L1 address by 1 face
        TTI_UNPACR(Srcs::SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        // Unpack SrcB — increments L1 address by 1 face
        TTI_UNPACR(Srcs::SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
