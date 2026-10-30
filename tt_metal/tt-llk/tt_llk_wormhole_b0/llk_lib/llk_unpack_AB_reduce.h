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
 * @brief Whether a reduce unpacks its operands swapped: scaler into SrcA and data into SrcB.
 *
 * SUM/AVG REDUCE_ROW multiplies each data row by a scaler column with MVMUL, which takes the scaler from SrcA.
 * Every other reduce keeps data in SrcA and the scaler in SrcB.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 */
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool _llk_unpack_AB_reduce_swaps_operands_()
{
    return (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
}

/**
 * @brief Configures the unpacker MOP for reduction operations.
 *
 * The data operand is unpacked face by face, except for a full 4-face SUM/AVG REDUCE_ROW, which unpacks the whole
 * data tile in one UNPACR. The scaler contributes only the first row of each of its faces, the row the math thread
 * reads: one row per data face, cycling through the scaler's own faces when it has fewer than the data (e.g. a
 * [1,32] scaler with a 32x32 tile).
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data operand, including face_r_dim and num_faces.
 * @param tensor_shape_scaler: Shape of the scaler operand.
 *
 * @note For tiny tiles (face_r_dim < 16), the data bank is cleared before each face so the rows past it pad correctly.
 * @note For REDUCE_SCALAR operations, SrcA is cleared before unpacking because SrcA is clobbered in the Math kernel.
 * @note For SUM/AVG REDUCE_ROW, SrcA is zeroed before the scaler rows land in it, so MVMUL sees only the scaler
 *       column and writes zeros to the other dest columns.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape);

    constexpr bool is_max        = pool_type == PoolType::MAX;
    constexpr bool is_scalar     = reduce_dim == ReduceDim::REDUCE_SCALAR;
    constexpr bool swap_operands = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();

    constexpr std::uint32_t data_unpacker  = swap_operands ? p_unpacr_nop::UNP1 : p_unpacr_nop::UNP0;
    constexpr std::uint32_t scaler_adc     = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;
    constexpr std::uint32_t data_clear_val = is_max ? p_unpacr_nop::UNP_NEGINFSRC : p_unpacr_nop::UNP_ZEROSRC;

    const std::uint32_t num_faces        = tensor_shape.total_num_faces();
    const std::uint32_t num_scaler_faces = tensor_shape_scaler.total_num_faces();

    if (swap_operands && num_faces == MAX_NUM_FACES)
    {
        // MVMUL multiplies the whole data tile by SrcA faces 0 and 1 in one handoff (see reduce_configure_mop): unpack
        // the data tile with one UNPACR, and one scaler row into each of those two SrcA faces.
        LLK_ASSERT(tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim, "The scaler needs a face for every column of data faces");
        constexpr std::uint32_t replay_len = 4;
        lltt::record<lltt::NoExec>(0, replay_len);
        TTI_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR(SrcA, 0b00010001 /*ch0_z+=1, ch1_z+=1*/, 0, 0, 0, 1, 0 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        ckernel_template tmp(1, 1, lltt::replay_insn(0, replay_len));
        tmp.program();
        return;
    }

    // One SrcA/SrcB handoff per data face; the scaler wraps back to its first face when it has fewer faces than the data.
    const bool wrap_scaler = num_scaler_faces < num_faces;
    LLK_ASSERT(
        !wrap_scaler || tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim,
        "A scaler with fewer faces than the data needs a face for every column of data faces");
    const std::uint32_t outerloop = wrap_scaler ? num_faces / num_scaler_faces : 1;
    const std::uint32_t innerloop = wrap_scaler ? num_scaler_faces : num_faces;
    const bool clear_data         = is_scalar || tensor_shape.face_r_dim < FACE_R_DIM;

    const std::uint32_t replay_len = (clear_data ? 1 : 0) + (swap_operands ? 1 : 0) + 2;
    lltt::record<lltt::NoExec>(0, replay_len);
    if (clear_data)
    {
        TTI_UNPACR_NOP(data_unpacker, data_clear_val);
    }
    if constexpr (swap_operands)
    {
        TTI_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
    }
    TTI_UNPACR(SrcA, 0b01 /*ch0_z+=1*/, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b01 /*ch0_z+=1*/, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, replay_len));
    if (wrap_scaler)
    {
        tmp.set_end_op(TT_OP_SETADCZW(scaler_adc, 0, 0, 0, 0, 0b0001));
    }
    tmp.program();
}

/**
 * @brief Initialize the unpacker for reduce operations
 *
 * Configures haloize (transpose) for row reductions and each unpacker's datum count, then programs the MOP.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data operand (operand A), including face_r_dim and num_faces.
 * @param tensor_shape_scaler: Shape of the scaler operand (operand B), e.g. a [1,32] tile.
 *
 * @note For SUM/AVG REDUCE_ROW, operands are swapped: scaler→SrcA, data→SrcB. Configure each unpacker for the operand
 *       it reads before this init, formats and face geometry: unpacker 0 for the scaler, unpacker 1 for the data. A
 *       scaler shaped unlike the data needs the face geometry swapped too, not just the formats.
 * @note For MAX REDUCE_ROW, original layout is kept: data→SrcA (transposed via haloize), scaler→SrcB.
 * @note The scaler's unpacker reads one row (FACE_C_DIM datums) per face; the data's reads full faces.
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (this is the scaler operand unpack pairing).
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    // The partial-face clear picks UNP_NEGINFSRC only for MAX, so MIN would pad SrcA with zero where it needs +inf.
    static_assert(
        pool_type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape);

    constexpr bool swap_operands       = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();
    constexpr std::uint32_t data_adc   = swap_operands ? p_setadc::UNP_B : p_setadc::UNP_A;
    constexpr std::uint32_t scaler_adc = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>((reduce_dim == ReduceDim::REDUCE_ROW));

    if (swap_operands && tensor_shape.total_num_faces() == MAX_NUM_FACES)
    {
        TT_SETADCXX(data_adc, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    }
    else
    {
        config_unpacker_x_end<data_adc>(tensor_shape.face_r_dim);
    }
    config_unpacker_x_end<scaler_adc>(1);

    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(tensor_shape, tensor_shape_scaler);
}

/**
 * @brief Initialize the unpacker for reduce operations with a scaler shaped like the data.
 *
 * Forwards to the two-shape init with the scaler's shape equal to the data's.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data and scaler tensors, including face_r_dim and num_faces.
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape)
{
    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim>(tensor_shape, tensor_shape);
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

    constexpr bool swap_operands = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();

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
