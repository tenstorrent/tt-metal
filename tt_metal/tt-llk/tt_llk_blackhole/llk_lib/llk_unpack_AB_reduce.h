// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Whether the reduce unpacks the scaler into SrcA and the data into SrcB.
 *
 * True for SUM/AVG REDUCE_ROW, whose MVMUL takes the scaler from SrcA.
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
 * @brief Whether the swapped path reads one row per scaler face, into a zeroed SrcA.
 *
 * True for a scaler shorter than a face, e.g. [1,32]; the zeros keep the result's other columns zero. A full-height
 * scaler face is read whole, so its rows past the first must be zero, as in a standard reduce scaler tile.
 *
 * @param tensor_shape_scaler: Shape of the scaler operand.
 */
inline bool _llk_unpack_AB_reduce_scaler_row_only_(const ckernel::TensorShape tensor_shape_scaler)
{
    return tensor_shape_scaler.face_r_dim < FACE_R_DIM;
}

/**
 * @brief Whether the swapped path reads a 32x32 scaler in one UNPACR, like the data.
 *
 * @param tensor_shape: Shape of the data operand.
 * @param tensor_shape_scaler: Shape of the scaler operand.
 */
inline bool _llk_unpack_AB_reduce_scaler_whole_tile_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    return tensor_shape.total_num_faces() == MAX_NUM_FACES && tensor_shape_scaler.total_num_faces() == MAX_NUM_FACES &&
           !_llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler);
}

/**
 * @brief Configures the unpacker MOP for reduction operations.
 *
 * The data is unpacked face by face, or in one UNPACR for a 4-face SUM/AVG REDUCE_ROW. The scaler gives the first
 * row of one of its faces per data face, cycling through its faces when it has fewer than the data. The swapped
 * path can read scaler faces whole (see @ref _llk_unpack_AB_reduce_scaler_row_only_).
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data operand.
 * @param tensor_shape_scaler: Shape of the scaler operand.
 *
 * @note For faces shorter than 16 rows, the data bank is cleared before each face.
 * @note For REDUCE_SCALAR operations, SrcA is cleared before unpacking because SrcA is clobbered in the Math kernel.
 * @note With a short scaler on the swapped path, both SrcA banks are zeroed once here. This math path never writes
 *       SrcA, so they stay zero; re-run the init after any op that does (e.g. MOVD2A, MOVB2A).
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape_scaler);

    constexpr bool is_max        = pool_type == PoolType::MAX;
    constexpr bool is_scalar     = reduce_dim == ReduceDim::REDUCE_SCALAR;
    constexpr bool swap_operands = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();

    constexpr std::uint32_t data_unpacker  = swap_operands ? Srcs::SrcB : Srcs::SrcA;
    constexpr std::uint32_t scaler_adc     = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;
    constexpr std::uint32_t data_clear_val = is_max ? p_unpacr_nop::CLR_SRC_NEGINF : p_unpacr_nop::CLR_SRC_0;

    const std::uint32_t num_faces        = tensor_shape.total_num_faces();
    const std::uint32_t num_scaler_faces = tensor_shape_scaler.total_num_faces();

    if (swap_operands && _llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler))
    {
        _llk_unpack_clear_src_banks_<SrcA>();
    }

    if (swap_operands && num_faces == MAX_NUM_FACES)
    {
        // One handoff per tile: the whole data tile, and scaler faces 0 and 1 into the SrcA faces MVMUL reads.
        LLK_ASSERT(tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim, "The scaler needs a face for every column of data faces");
        if (_llk_unpack_AB_reduce_scaler_whole_tile_(tensor_shape, tensor_shape_scaler))
        {
            constexpr std::uint32_t whole_tile_replay_len = 2;
            load_replay_buf(
                0,
                whole_tile_replay_len,
                []
                {
                    TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                    TTI_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                });
            ckernel_template tmp(1, 1, lltt::replay_insn(0, whole_tile_replay_len));
            tmp.program();
            return;
        }
        constexpr std::uint32_t replay_len = 3;
        load_replay_buf(
            0,
            replay_len,
            []
            {
                TTI_UNPACR(SrcA, 0b00010001 /*ch0_z+=1, ch1_z+=1*/, 0, 0, 0, 1, 0 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            });

        ckernel_template tmp(1, 1, lltt::replay_insn(0, replay_len));
        tmp.program();
        return;
    }

    // One handoff per data face; the scaler wraps when it has fewer faces.
    const bool wrap_scaler = num_scaler_faces < num_faces;
    LLK_ASSERT(
        !wrap_scaler || tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim,
        "A scaler with fewer faces than the data needs a face for every column of data faces");
    const std::uint32_t outerloop = wrap_scaler ? num_faces / num_scaler_faces : 1;
    const std::uint32_t innerloop = wrap_scaler ? num_scaler_faces : num_faces;
    const bool clear_data         = is_scalar || tensor_shape.face_r_dim < FACE_R_DIM;

    const std::uint32_t replay_len = (clear_data ? 1 : 0) + 2;
    load_replay_buf(
        0,
        replay_len,
        [clear_data]
        {
            if (clear_data)
            {
                TTI_UNPACR_NOP(data_unpacker, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, data_clear_val, p_unpacr_nop::CLR_SRC);
            }
            TTI_UNPACR(SrcA, 0b01 /*ch0_z+=1*/, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcB, 0b01 /*ch0_z+=1*/, 0, 0, 0, 1, 1 /*SetDatValid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        });

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
 * @param tensor_shape: Shape of the data operand (operand A).
 * @param tensor_shape_scaler: Shape of the scaler operand (operand B), e.g. [1,32].
 *
 * @note For SUM/AVG REDUCE_ROW, operands are swapped: scaler→SrcA, data→SrcB. Configure unpacker 0 for the scaler
 *       and unpacker 1 for the data before this init, face geometry as well as formats.
 * @note For MAX REDUCE_ROW, original layout is kept: data→SrcA (transposed via haloize), scaler→SrcB.
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (this is the scaler operand unpack pairing).
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    // The partial-face clear picks CLR_SRC_NEGINF only for MAX, so MIN would pad SrcA with zero where it needs +inf.
    static_assert(
        pool_type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape_scaler);

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
    // The scaler reads one row per face, unless the swapped path reads it whole.
    if (swap_operands && _llk_unpack_AB_reduce_scaler_whole_tile_(tensor_shape, tensor_shape_scaler))
    {
        TT_SETADCXX(scaler_adc, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    }
    else
    {
        config_unpacker_x_end<scaler_adc>(swap_operands && !_llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler) ? FACE_R_DIM : 1);
    }

    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(tensor_shape, tensor_shape_scaler);
}

/**
 * @brief Initialize the unpacker for reduce operations with a scaler shaped like the data.
 *
 * Forwards to the two-shape init with the scaler's shape equal to the data's.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of both the data and the scaler.
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
