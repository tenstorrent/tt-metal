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
#include "llk_reduce_block.h"
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
 * @note The REDUCE_SCALAR math kernel writes SrcA rows 0 to 15 (MOVB2A); a full face rewrites them on the next unpack,
 *       so only tiny tiles need the source clear.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const ckernel::TensorShape tensor_shape)
{
    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape);

    constexpr bool is_max                  = pool_type == PoolType::MAX;
    constexpr bool swap_operands           = (reduce_dim == ReduceDim::REDUCE_ROW) && !is_max;
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

    if (is_tiny)
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
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape)
{
    // The partial-face clear picks CLR_SRC_NEGINF only for MAX, so MIN would pad SrcA with zero where it needs +inf.
    static_assert(
        pool_type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

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

/**
 * @brief Unpack a block of consecutive data tiles against one scaler tile with one context acquire per chunk of
 *        @ref REDUCE_BLOCK_MAX_TILES tiles.
 *
 * Each tile runs the MOP of @ref _llk_unpack_AB_reduce_init_, so the source registers receive what the per tile call
 * gives them. The data address advances in the instruction stream: through the unpacker's Z counter when the tiles are
 * face-contiguous in L1, otherwise by adding the tile stride to the context's base address (CFGSHIFTMASK). When
 * @ref reduce_block_holds_scaler holds, the scaler tile is unpacked into SrcA once per chunk and only the data tiles
 * follow.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address_a: L1 address of the first data tile (16 B units).
 * @param address_b: L1 address of the scaler tile (16 B units).
 * @param num_tiles: Number of consecutive data tiles.
 * @param tile_stride_16B: Distance between the starts of consecutive data tiles in L1 (16 B units).
 * @param unpack_src_format: L1 data format of the data tiles.
 * @param tensor_shape: Shape of the data tile, the one the init received.
 * @note Call @ref _llk_unpack_AB_reduce_init_ with matching template args before this function.
 * @note Pair with @ref _llk_math_reduce_block_ on the math thread with the same tile count.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_block_(
    std::uint32_t address_a,
    const std::uint32_t address_b,
    std::uint32_t num_tiles,
    const std::uint32_t tile_stride_16B,
    const std::uint32_t unpack_src_format,
    const ckernel::TensorShape tensor_shape)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_block_", tensor_shape);

    if (num_tiles < 2)
    {
        if (num_tiles == 1)
        {
            _llk_unpack_AB_reduce_<pool_type, reduce_dim>(address_a, address_b);
        }
        return;
    }
    LLK_ASSERT(is_valid_L1_address(address_a + (num_tiles - 1) * tile_stride_16B), "L1 address of the last tile must be in valid L1 memory region");

    constexpr bool swap_operands       = (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
    constexpr std::uint32_t data_unp   = swap_operands ? p_setadc::UNP_B : p_setadc::UNP_A;
    constexpr std::uint32_t scaler_unp = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;

    const bool holds_scaler   = reduce_block_holds_scaler<pool_type, reduce_dim>(tensor_shape);
    const std::uint32_t faces = tensor_shape.total_num_faces();
    const bool contiguous     = tile_stride_16B * 16 == SCALE_DATUM_SIZE(unpack_src_format, faces * tensor_shape.face_r_dim * FACE_C_DIM);

    if (!contiguous)
    {
        // The tile stride for CFGSHIFTMASK, written in the instruction stream behind the CFGSHIFTMASKs of an earlier block
        // SCRATCH_SEC0 is shared: the tilize, tilizeA_B and compressed custom_mm inits set it and run again after this
        TT_SETDMAREG(0, LOWER_HALFWORD(tile_stride_16B), 0, LO_16(p_gpr_unpack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(tile_stride_16B), 0, HI_16(p_gpr_unpack::TMP0));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
    }

    while (num_tiles > 0)
    {
        const std::uint32_t chunk = num_tiles < REDUCE_BLOCK_MAX_TILES ? num_tiles : REDUCE_BLOCK_MAX_TILES;

        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

        volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
        const std::uint32_t context            = unp_cfg_context;
        const std::uint32_t data_reg   = swap_operands ? ((context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32)
                                                       : ((context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        const std::uint32_t scaler_reg = swap_operands ? ((context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32)
                                                       : ((context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
        const std::uint32_t advance    = TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, data_reg);

        wait_for_next_context(2);

        // The CFGSHIFTMASK form adds the stride before every tile, the first one included
        cfg[data_reg]   = contiguous ? address_a : address_a - tile_stride_16B;
        cfg[scaler_reg] = address_b;

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Hold the UNPACRs and the CFGSHIFTMASKs until the base address stores from the RISC have landed
        TTI_STALLWAIT(p_stall::STALL_UNPACK | p_stall::STALL_CFG, p_stall::TRISC_CFG);

        if (holds_scaler)
        {
            TTI_UNPACR(Srcs::SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            for (std::uint32_t tile = 0; tile < chunk; tile++)
            {
                if (contiguous)
                {
                    TTI_UNPACR(Srcs::SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                    // Z runs past the tile's Z dim into the next tile, linear in L1 (as in the block row max)
                    TTI_INCADCZW(data_unp, 0, 0, 0, 4);
                }
                else
                {
                    TT_INSN(advance);
                    TTI_SETADCZW(data_unp, 0, 0, 0, 0, 0b1111);
                    TTI_UNPACR(Srcs::SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                }
            }
        }
        else
        {
            for (std::uint32_t tile = 0; tile < chunk; tile++)
            {
                if (contiguous)
                {
                    // The data Z counter runs past the tile's Z dim into the next tile, linear in L1 (as in the block
                    // row max); the scaler restarts at its first face
                    TTI_SETADCZW(scaler_unp, 0, 0, 0, 0, 0b1111);
                }
                else
                {
                    TT_INSN(advance);
                    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
                }
                ckernel::ckernel_template::run();
            }
        }

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);

        address_a += chunk * tile_stride_16B;
        num_tiles -= chunk;
    }
}
