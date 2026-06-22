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
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Program the unpacker MOP for a tilize operation.
 *
 * Selects between unpacking to SrcA (with a SrcB dvalid NOP) or straight to dest.
 *
 * @param narrow_tile: Whether the tile is narrow (single column of faces).
 * @param unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 */
inline void _llk_unpack_tilize_mop_config_(const bool narrow_tile = false, const bool unpack_to_dest = false)
{
    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*CH0/CH1 Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_zerosrc    = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID); // WA for tenstorrent/budabackend#1230

    const std::uint32_t outerloop     = narrow_tile ? 1 : 2;
    constexpr std::uint32_t innerloop = 1;

    if (unpack_to_dest)
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
        tmp.program();
    }
    else
    {
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_zerosrc, unpack_srcb_set_dvalid);
        tmp.set_start_op(unpack_srca);
        tmp.program();
    }
}

/**
 * @brief Initialize the unpacker for a tilize operation.
 *
 * Disables face transpose, configures the unpacker into tileize mode (throttle, shift amount,
 * per-tile X/Z dims) for the given block column dimension, decides whether 32-bit datums must be
 * unpacked to dest, and programs the tilize MOP.
 *
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param ct_dim: Number of column tiles in the block, used to size the column dimension.
 * @param face_r_dim: Rows per face.
 * @param narrow_tile: Whether the tile is narrow (single column of faces).
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @note Call @ref _llk_unpack_tilize_uninit_ after this function to restore the modified tile-descriptor state.
 * @ref _llk_unpack_tilize_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ (DataCopyType::A2D) is the matching init on the math thread.
 */
inline void _llk_unpack_tilize_init_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    const bool unpack_to_dest = (unpack_src_format == to_underlying(DataFormat::UInt32)) || (unpack_src_format == to_underlying(DataFormat::Int32)) ||
                                (unpack_dst_format == to_underlying(DataFormat::Float32));
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);
    const bool narrow_layout        = narrow_tile || (num_faces == 1);

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); // GPR preloaded with  16 | (16 << 16)

    _llk_unpack_tilize_mop_config_(narrow_layout, unpack_to_dest);
}

/**
 * @brief Internal helper that unpacks (tilizes) faces into the SrcA register.
 *
 * Loops over the face groups, computing each iteration's L1 address from the top/bottom face
 * offsets, and runs the tilize MOP while synchronizing through the unpack semaphore and switching
 * config context per iteration.
 *
 * @param base_address: L1 base address of the source tile buffer.
 * @param num_loops: Number of face-group iterations to unpack.
 * @param top_face_offset_address: 16B-word offset to the top faces within the tile.
 * @param bot_face_offset_address: 16B-word offset added on the second iteration for the bottom faces.
 */
inline void unpack_tilize_impl(
    const std::uint32_t base_address, std::uint32_t num_loops, std::uint32_t top_face_offset_address, std::uint32_t bot_face_offset_address)
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    for (std::uint32_t n = 0; n < num_loops; n++)
    {
        std::uint32_t address = base_address + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

        // Wait for free context
        wait_for_next_context(2);

        // Validate and configure address
        _llk_unpack_configure_single_address_(address, cfg);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Run MOP
        ckernel::ckernel_template::run();

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

/**
 * @brief Internal helper that unpacks (tilizes) faces directly into the dest register.
 *
 * Sets the dest write address, unpacks the top faces, then (if more than one face group) reprograms
 * the L1 base address and unpacks the bottom faces, finishing with the dest completion handshake.
 *
 * @param base_address: L1 base address of the source tile buffer.
 * @param unpack_src_format: Source data format of the operand, used to compute the dest write address.
 * @param num_loops: Number of face-group iterations (>1 also unpacks the bottom faces).
 * @param top_face_offset_address: 16B-word offset to the top faces within the tile.
 * @param bot_face_offset_address: 16B-word offset added for the bottom faces.
 */
inline void unpack_tilize_to_dest_impl(
    const std::uint32_t base_address,
    std::uint32_t unpack_src_format,
    std::uint32_t num_loops,
    std::uint32_t top_face_offset_address,
    std::uint32_t bot_face_offset_address)
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Unpack to dest register
    set_dst_write_addr(unp_cfg_context, unpack_src_format);
    wait_for_dest_available();

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);
    std::uint32_t address = base_address + top_face_offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    // Get tile address
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Unpack top faces
    ckernel::ckernel_template::run();

    // Unpack bottom faces if needed
    if (num_loops > 1)
    {
        // Needed to stall counter reconfiguration until unpacker finishes previous instruction
        TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::UNPACK);

        // Don't clear the CH1 W counter - needed for multiple tiles
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1011);

        // Increment address to point to bottom faces in L1
        address += bot_face_offset_address;
        LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

        // Get tile address
        TT_SETDMAREG(0, LOWER_HALFWORD(address), 0, LO_16(p_gpr_unpack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(address), 0, HI_16(p_gpr_unpack::TMP0));
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::THCON);

        // Unpack bottom faces
        ckernel::ckernel_template::run();
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    unpack_to_dest_tile_done(unp_cfg_context);
}

/**
 * @brief Unpack and tilize a tile from L1 into SrcA or the dest register.
 *
 * Computes the per-face L1 offsets for the selected column tile, then dispatches to the source-
 * register helper (@ref unpack_tilize_impl) or, for 32-bit datums, the dest helper
 * (@ref unpack_tilize_to_dest_impl).
 *
 * @param base_address: L1 base address of the source tile buffer.
 * @param tile_index: Column tile index selecting which tile to unpack.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param block_ct_dim: Number of column tiles in the block, used to compute the bottom-face offset.
 * @param face_r_dim: Rows per face.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces).
 * @note Call @ref _llk_unpack_tilize_init_ before this function, and
 *       @ref _llk_unpack_tilize_uninit_ after it to restore modified state.
 */
inline void _llk_unpack_tilize_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    std::uint32_t unpack_src_format = 0,
    std::uint32_t unpack_dst_format = 0,
    std::uint32_t block_ct_dim      = 0,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool narrow_tile          = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    const bool unpack_to_dest = (unpack_src_format == to_underlying(DataFormat::UInt32)) || (unpack_src_format == to_underlying(DataFormat::Int32)) ||
                                (unpack_dst_format == to_underlying(DataFormat::Float32));

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);
    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
    // For narrow tile we unpack 1 face in each iteration
    // For num_faces == 1 we unpack a single face per tile
    // Offset address is in 16B words
    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B   = block_ct_dim * (narrow_tile ? FACE_C_DIM / 16 : TILE_C_DIM / 16);
    std::uint32_t bot_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, face_r_dim * block_c_dim_16B); //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    std::uint32_t num_loops = (num_faces == 1) ? 1 : (narrow_tile ? 2 : num_faces >> 1);

    if (!unpack_to_dest)
    {
        unpack_tilize_impl(base_address, num_loops, top_face_offset_address, bot_face_offset_address);
    }
    else
    {
        // Unpack tilize to DEST works with only one config context, hence it needs to be reset before calling the function.
        reset_config_context();
        unpack_tilize_to_dest_impl(base_address, unpack_src_format, num_loops, top_face_offset_address, bot_face_offset_address);
    }
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

/**
 * @brief Program the unpacker MOP/replay buffer for tilize-A-with-unpack-B.
 *
 * Builds a replay buffer that unpacks one 1x16 row of SrcA at a time and advances the SrcA L1
 * base address (per config context) by the programmed column stride.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, (zero_srcA ? 0b010001 : 0b1), 0, 0, 0, 1, (zero_srcA ? 0 : 1), p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(
        SrcB,
        (zero_srcA ? 0b010001 : (reload_srcB ? 0b0 : 0b1)),
        0,
        0,
        0,
        1,
        (zero_srcA ? 0 : 1),
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1); // Skip face ptr inc if same face is reloaded into srcB
    static constexpr std::uint32_t unpack_neginf_srca = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_NEGINFSRC); // Needed for max pool
    static constexpr std::uint32_t unpack_zero_srca   = TT_OP_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);   // Needed for dot product
    static constexpr std::uint32_t unpack_srcb_2_face =
        TT_OP_UNPACR(SrcB, 0b100010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product
    static constexpr std::uint32_t unpack_srca_dat_valid =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot product
    static constexpr std::uint32_t unpack_srcb_dat_valid =
        TT_OP_UNPACR(SrcB, (reload_srcB ? 0b0 : 0b1), 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Needed for dot
                                                                                                                // product

    const std::uint32_t innerloop = zero_srcA ? (num_faces > 2 ? 2 : (num_faces - 1)) : 1;
    const std::uint32_t outerloop = zero_srcA ? 1 : (num_faces > 2) ? num_faces / 2 : num_faces;
    ckernel_template tmp(outerloop, innerloop, unpack_srca, ((zero_srcA && num_faces == 2) ? unpack_srcb_2_face : unpack_srcb));
    if constexpr (neginf_srcA)
    {
        tmp.set_start_op(unpack_neginf_srca);
    }
    else if constexpr (zero_srcA_reduce)
    {
        tmp.set_start_op(unpack_zero_srca);
    }
    else if constexpr (zero_srcA)
    {
        if (num_faces < 4)
        {
            tmp.set_start_op(unpack_zero_srca);
            tmp.set_end_ops(unpack_srca_dat_valid, unpack_srcb_dat_valid);
        }
    }
    tmp.program();
}

/**
 * @brief Initialize the unpacker to tilize operand A while unpacking operand B.
 *
 * Programs the column stride used to advance SrcA's L1 address (via the CFGSHIFTMASK scratch
 * register), sets per-unpacker datum counts (one row for SrcA, full face for SrcB) and SrcA's Y
 * stride, disables face transpose, and programs the tilize-A-B MOP.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param unpack_src_format: Source data format of operand A in L1.
 * @param unpack_dst_format: Destination data format operand A is converted to.
 * @param narrow_tile: Whether the tile is narrow (single column of faces).
 * @param ct_dim: Number of column tiles in the block, used to size the column stride.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @note Call @ref _llk_unpack_tilizeA_B_uninit_ after this function to restore the modified stride/datum-count state.
 * @ref _llk_unpack_tilizeA_B_ is the matching execute call.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const bool narrow_tile,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces       = 4,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * ((narrow_tile || (num_faces == 1)) ? FACE_C_DIM : TILE_C_DIM);

    // Set face dim
    TT_SETADCXX(p_setadc::UNP_A, unpA_face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_face_r_dim * FACE_C_DIM - 1, 0x0);

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32,
                 p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1,
        0,
        0,
        0,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::FACE_DIM_1x16); // GPR preloaded with  16 | (16 << 16)

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(num_faces);
}

/**
 * @brief Tilize operand A and unpack operand B, face by face, into SrcA and SrcB.
 *
 * Loops over the faces, computing each face's SrcA L1 address, optionally clearing SrcA to
 * zero, unpacking the SrcB face, then unpacking the face's rows into SrcA (row by row via the MOP)
 * and setting data-valid, synchronizing through the unpack semaphore each iteration.
 *
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param face_r_dim: Rows per face.
 * @param narrow_tile: Whether the tile is narrow (single column of faces).
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param address_b: L1 address of operand B's face data.
 * @param tile_index_a: Column tile index into operand A.
 * @param block_ct_dim: Number of column tiles in the block, used to compute face strides.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ with matching template args before this function, and
 *       @ref _llk_unpack_tilizeA_B_uninit_ after it to restore modified state.
 */
template <bool zero_srcA = false>
inline void _llk_unpack_tilizeA_B_(
    std::uint32_t unpA_src_format,
    std::uint32_t face_r_dim,
    std::uint32_t narrow_tile,
    std::uint32_t base_address_a,
    std::uint32_t address_b,
    std::uint32_t tile_index_a,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpA_src_format, tile_index_a) << (narrow_tile ? 0 : 1);

    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
    // For narrow tile we unpack 1 face in each iteration
    // Offset address is in 16B words
    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B   = block_ct_dim * ((narrow_tile || (num_faces == 1)) ? FACE_C_DIM / 16 : TILE_C_DIM / 16);
    std::uint32_t bot_face_offset_address = SCALE_DATUM_SIZE(unpA_src_format, face_r_dim * block_c_dim_16B); //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    std::uint32_t num_loops = narrow_tile ? 2 : ((num_faces > 1) ? num_faces / 2 : 1);

    // Clear z/w start counters for SrcB
    TTI_SETADCZW(UNP1, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    for (std::uint32_t n = 0; n < num_loops; n++)
    {
        std::uint32_t address_a = base_address_a + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        if constexpr (zero_srcA)
        {
            if (num_faces == 4 && n == 1)
            {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1011);
            }
            else
            {
                TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
            }
        }
        else
        {
            TTI_SETADCZW(UNP0, 0, 0, 0, 0, 0b1111);
        }

        // Wait for free context
        wait_for_next_context(2);

        // Validate and configure addresses
        _llk_unpack_configure_addresses_(address_a, address_b, cfg);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Run MOP
        if constexpr (zero_srcA)
        {
            if (num_faces == 4 && n == 0)
            {
                TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
            }

            ckernel::ckernel_template::run();

            if (num_faces == 4 && n != 0)
            {
                TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
            }
        }
        else
        {
            ckernel::ckernel_template::run();
        }

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

/**
 * @brief Restore unpacker state after a tilize operation.
 *
 * Reverts the tile descriptor Y and Z dimensions to defaults and rewrites the unpack config
 * (clearing tilize mode) so subsequent ops see a normal tile layout. x-start/x-end is transient
 * and reprogrammed by each operation's init (see tt-llk#1036), so it is not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @note Call @ref _llk_unpack_tilize_init_ before this function.
 */
inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format)
{
    // Stalling SETDMAREG done by THCON until UNPACK finishes
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);

    // Revert Z and Y dim value back to default:
    // THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1 - word 1 of the same-named register
    // y-dim sits in lower 16 bits and is set to 1 by default
    // z-dim sits in upper 16 bits and is set to unpA_num_faces which is 4 by default
    // TODO NC: Make this configurable and restored to a default operand state under tt-llk#1161
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xffff0000>(4);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0x0000ffff>(1);

    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32,
                 p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1,
        0,
        0,
        0,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::FACE_DIM_16x16); // GPR preloaded with  16 | (16 << 16)}
}

/**
 * @brief Restore unpacker state after a tilize-A-with-unpack-B operation.
 *
 * Resets the SrcA/SrcB Z/W counters and rewrites the unpack config and tile X-dim back to the
 * default 16x16 face layout. x-start/x-end is transient and reprogrammed by each operation's init
 * (see tt-llk#1036), so it is not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ before this function.
 */
inline void _llk_unpack_tilizeA_B_uninit_(const std::uint32_t unpack_dst_format)
{
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);

    // reset z/w counters
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));

    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32,
                 p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(
        1,
        0,
        0,
        0,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::FACE_DIM_16x16); // GPR preloaded with  16 | (16 << 16)}
}

/*************************************************************************
 * LLK UNPACK FAST TILIZE (Tilize single input using both unpackers and packer)
 * full_dim is the tensor width in number of tiles
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim is 1 (only if full_dim is 1) or 2 and 3 (for any other full_dim)
 * each call can process unit_dim * num_units tiles but when unit_dim is 2 or 3 all tiles must be in a single row
 * changing between unit_dim 1 and 2/3 requires reconfiguration while changing between 2 and 3 does not
 * base_address is the 16B base address of the start of the tile row
 * tile_index is the index of the tile inside that row
 * currently supports only 4 16x16 faces per tile
 * supported input formats are: FP32 (via FP16 or TF32) or FP16_B
 *************************************************************************/

/**
 * @brief Program the unpacker MOP for fast tilize (both unpackers feeding the packer).
 *
 * Sets up the UNPACR template that reads tile rows into SrcA and SrcB for the unit_dim 2/3 fast
 * tilize paths (UNPACR instructions for unit_dim 2, SKIP instructions for unit_dim 3).
 */
inline void _llk_unpack_fast_tilize_mop_config_()
{
    // Y moves to the next tile, Z moves to the next row (both ch0 and ch1)
    // constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1 = 0b00'10'00'01;
    // constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0 = 0b00'00'10'00;
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1 = 0b00'11'00'01;
    // constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0 = 0b00'00'11'00;

    // UNPACR instructions are used with unit_dim 2 and SKIP instructions are used with unit_dim 3
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,
        false,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0),
        TT_OP_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0),
        TT_OP_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0));

    tmp.program();
}

/**
 * @brief Initialize the unpacker for fast tilize (single input via both unpackers and the packer).
 *
 * Saves the unpacker tile-descriptor and stride state to GPRs, reprograms the per-unpacker tile
 * X/Y/Z dims and CH1 Z strides for row-major fast tilize over a row of full_dim tiles, and programs
 * the fast-tilize MOP.
 *
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param full_dim: Tensor width in number of tiles (drives the Y dimension / row stride).
 * @note Call @ref _llk_unpack_fast_tilize_uninit_ after this function to restore the saved unpacker state.
 * @note On the math thread, pair with @ref _llk_math_fast_tilize_init_ and on the pack thread with @ref _llk_pack_fast_tilize_init_ (same unit_dim).
 * @ref _llk_unpack_fast_tilize_block_ is the matching execute call.
 */
inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, std::uint32_t full_dim)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // save the following state that is going to be modified:
    // tile x, y, and z dims for both unpackers
    // CH1 Z stride for both unpackers
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // set x dim to single tile width, moving across y counter moves to the next tile in row major
    // set y dim to full dim, moving across z counter moves to the next row in row major
    // set z dim to single face height, moving across w counter moves to the next face row in row major
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, full_dim, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, FACE_R_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, TILE_C_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, full_dim, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, FACE_R_DIM, p_setdmareg::MODE_IMMEDIATE, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // for unit_dim 2 or 3 unpacker read sizes are multiples of 32 datums (64 or 96) so CH1 Z stride is set to 32 datums
    // for unit_dim 1 unpacker reads whole tile per iteration so CH1 counter is not used
    // why are CH1 strides in bytes?
    // SCALE_DATUM_SIZE wouldn't work here since it doesn't have a case for TF32
    std::uint32_t ch1_x_stride = (unpack_dst_format & 0x3) == to_underlying(DataFormat::Float32) ? 4 : 2;
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(TILE_C_DIM * ch1_x_stride);
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}

/**
 * @brief Restore unpacker state after a fast tilize operation.
 *
 * Rewrites the unpacker tile-descriptor and CH1 Z-stride registers saved by
 * @ref _llk_unpack_fast_tilize_init_ and resets all address counters.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @note Call @ref _llk_unpack_fast_tilize_init_ before this function.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_uninit_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    // restore saved state
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, p_cfg::WRCFG_32b, UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1);

    // reset all counters
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
}

/**
 * @brief Fast-tilize a block of tiles using both unpackers into SrcA and SrcB.
 *
 * Computes the SrcA/SrcB L1 addresses for the block, programs the per-unit datum counts, and unpacks
 * num_units units of unit_dim tiles each via per-unit ADDRMOD sequences (one tile for unit_dim 1, a
 * row of 2 or 3 tiles for unit_dim 2/3), synchronizing through the unpack semaphore.
 *
 * @param base_address: 16B base address of the start of the tile row.
 * @param tile_index: Index of the tile within that row.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unit_dim: Tiles processed per iteration (1 only if full_dim is 1, otherwise 2 or 3).
 * @param num_units: Number of units processed in this call.
 * @param full_dim: Tensor width in number of tiles.
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @note Call @ref _llk_unpack_fast_tilize_init_ before this function, and
 *       @ref _llk_unpack_fast_tilize_uninit_ after it to restore modified state.
 * @note On the math thread, @ref _llk_math_fast_tilize_block_ consumes SrcA/SrcB; on the pack thread, @ref _llk_pack_fast_tilize_block_ writes the tilized L1
 * output.
 */
inline void _llk_unpack_fast_tilize_block_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units,
    const std::uint32_t full_dim,
    const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4");
    LLK_ASSERT(
        (unit_dim == 2 && num_faces == 2) || num_faces == 4, "16x32 tiny tiles are only supported for tensors with even-sized tile widths for fast_tilize");
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    std::uint32_t address = base_address + (SCALE_DATUM_SIZE(unpack_src_format, tile_index * TILE_C_DIM) >> 4); // move by tile width in 16B words
    // for unit_dim 2 UNPA reads top faces and UNPB reads bottom faces
    // for unit_dim 3 UNPA reads top 8 rows of top then bottom faces, UNPB reads bottom 8 rows of top then bottom faces
    // tiny tiles will use same UPKB scheme as unit_dim 3, but only does top faces
    std::uint32_t unpB_row_offset = unit_dim == 2 && num_faces == 4 ? FACE_R_DIM : (FACE_R_DIM / 2);
    std::uint32_t unpB_address    = address + (SCALE_DATUM_SIZE(unpack_src_format, full_dim * TILE_C_DIM * unpB_row_offset) >> 4);

    // reset all counters since X start and end are set after this
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));

    // unit_dim 1 reads the whole tile while unit_dim 2 and 3 read one row from 2 or 3 tiles
    if (unit_dim == 1)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, TILE_R_DIM * TILE_C_DIM - 1, 0x0);
    }
    else if (unit_dim == 2)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, 2 * TILE_C_DIM - 1, 0x0);
    }
    else if (unit_dim == 3)
    {
        TTI_SETADCXX(p_setadc::UNP_AB, 3 * TILE_C_DIM - 1, 0x0);
    }
    else
    {
        // replace this with a proper assert once it's available
        // FWASSERT("Unsupported unit_dim", false);
    }

    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address, unpB_address, cfg);

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Y moves to the next tile, Z moves to the next row (both ch0 and ch1)
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    // constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1 = 0b00'10'00'01;
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0 = 0b00'00'10'00;
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1 = 0b00'11'00'01;
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0 = 0b00'00'11'00;

    for (std::uint32_t i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // read whole tile contiguously then move two face rows down to the next tile
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 1);
            TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 2, 0);
        }
        else if (unit_dim == 2 && num_faces == 4)
        {
            // read top(A)/bottom(B) faces of two tiles in a row (4 faces each), switch bank,
            // then move to the next two tiles (CH0Y += 2) and back to the top of a tile (CH01Z = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < FACE_R_DIM - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        }
        else if (unit_dim == 2 && num_faces == 2)
        {
            // read top 8(A)/bottom 8(B) rows of top faces of two tiles in a row (4 halves of a face each),
            // then move to the next two tiles (CH0Y += 2) and back to the top of a tile (CH01Z = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < FACE_R_DIM / 2 - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_2_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM / 2 - 1) - 1, 0x0);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_2_CH0Z_0, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        }
        else if (unit_dim == 3)
        {
            // read top 8(A)/bottom 8(B) rows of top faces of three tiles in a row (6 halves of a face each), switch bank,
            // then move to the bottom faces (CH0W = 1) and back to the top of a face (CH01Z = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < (FACE_R_DIM / 2) - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, ((FACE_R_DIM / 2) - 1) - 1, 0xFFFF);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 1, 0, SETADC_CH01(p_setadc::ZW));

            // read top 8(A)/bottom 8(B) rows of bottom faces of three tiles in a row (6 halves of a face each), switch bank,
            // then move to the top faces of the next three tiles (CH0Y += 3) and back to top of a tile (CH01Z = 0, CH0W = 0)
            // inside mop:
            // for (std::uint32_t j = 0; j < (FACE_R_DIM / 2) - 1; j++)
            // {
            //     TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            //     TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_3_CH0Y_0_CH0Z_1, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, ((FACE_R_DIM / 2) - 1) - 1, 0xFFFF);
            TTI_UNPACR_COMMON(SrcA, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0, 1);
            TTI_UNPACR_COMMON(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_3_CH0Z_0, 1);
            TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        }
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    switch_config_context(unp_cfg_context);
}
