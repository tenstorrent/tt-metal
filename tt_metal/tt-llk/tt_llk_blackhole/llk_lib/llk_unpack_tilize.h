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

inline void _llk_unpack_tilize_mop_config_(
    [[maybe_unused]] const bool narrow_tile = false, const bool unpack_to_dest = false, const bool skip_bh_workaround = false)
{
    LLK_ASSERT(!narrow_tile, "narrow_tile: this parameter is unused");

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(0, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    const std::uint32_t outerloop = (!skip_bh_workaround || (skip_bh_workaround && narrow_tile)) ? 1 : 2;
    const std::uint32_t innerloop = 1;

    ckernel_template tmp(outerloop, innerloop, unpack_to_dest ? unpack_srca_to_dest : unpack_srcb_set_dvalid);

    if (skip_bh_workaround)
    {
        static constexpr std::uint32_t unpack_srcb_zerosrc = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::UNP_NOP, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_zerosrc, unpack_srcb_set_dvalid);
    }

    if (!unpack_to_dest)
    {
        tmp.set_start_op(unpack_srca);
    }

    tmp.program();
}

inline void _llk_unpack_tilize_init_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    LLK_ASSERT(face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be 2, 4, 8, or 16 for tilize");
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t block_c_dim = ct_dim * (narrow_tile ? FACE_C_DIM : TILE_C_DIM);

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_dest(static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), unpack_to_dest),
        "Unsupported unpacker format conversion.");

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
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);

    const bool is_8bit_format = IS_8BIT_FORMAT(unpack_src_format);
    // 8bit datums do not need the blackhole workaround therefore we fallback to regular tilize operation like for wormhole.
    if (is_8bit_format)
    {
        TTI_WRCFG(p_gpr_unpack::FACE_DIM_1x16, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    }
    else
    {
        // below is the configuration for unpack for srca
        const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
        const std::uint32_t Tile_z_dim = 1;
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));
        // Set x-dim to cover entire tile (face_r_dim * num_faces * FACE_C_DIM)
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0xffff0000>(0 | (Tile_x_dim << 16));
        // Set z-dim to 1 as X dim is set to cover the entire tile, so no need to iterate over faces.
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(0 | (Tile_z_dim << 16));

        // Set x-end for Unpackers to (face_r_dim * num_faces * FACE_C_DIM - 1)
        TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0);
    }

    _llk_unpack_tilize_mop_config_(narrow_tile, unpack_to_dest, is_8bit_format);
}

inline void _llk_unpack_tilize_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    std::uint32_t unpack_src_format                 = 0,
    std::uint32_t unpack_dst_format                 = 0,
    [[maybe_unused]] std::uint32_t block_ct_dim     = 0,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t num_faces  = 4,
    const bool narrow_tile                          = false)
{
    LLK_ASSERT(block_ct_dim == 0, "block_ct_dim: this parameter is unused");
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);

    // 8bit datums do not need the blackhole workaround therefore we fallback to regular tilize operation like for wormhole.
    if (IS_8BIT_FORMAT(unpack_src_format))
    {
        // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
        // For narrow tile we unpack 1 face in each iteration
        // Offset address is in 16B words
        // Datum count = tile_index*face_r_dim (/16 to get word count)

        const auto config_vec                 = read_unpack_config();
        const std::uint32_t shift_amount      = config_vec[0].shift_amount;
        std::uint32_t bot_face_offset_address = shift_amount * face_r_dim; // bytes for bottom faces

        // Program srcA and srcB base addresses
        std::uint32_t num_loops = narrow_tile ? 2 : num_faces / 2;

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
    else
    {
        // Program srcA and srcB base addresses
        // FIXME MT: This should be revisited for narrow tiles
        // std::uint32_t num_loops = narrow_tile ? 2 : num_faces/2;

        std::uint32_t address = base_address + top_face_offset_address;
        LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

        // Clear z/w start counters
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

        // Wait for free context
        wait_for_next_context(2);

        // Get tile address
        if (0 == unp_cfg_context)
        {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        }
        else
        {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        }

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        if (unpack_to_dest)
        {
            // Unpack to dest
            set_dst_write_addr(unp_cfg_context, unpack_src_format);
            wait_for_dest_available();
        }

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Run MOP
        ckernel::ckernel_template::run();

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        if (unpack_to_dest)
        {
            unpack_to_dest_tile_done(unp_cfg_context);
        }

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const std::uint32_t num_faces, const std::uint32_t face_r_dim)
{
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // Revert X dim value to default.
    // TODO NC: Issue tt-llk#1036 will make this transient
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Revert Z dim value back to default.
    const std::uint32_t Tile_z_dim = num_faces;
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xffff0000>(Tile_z_dim);

    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // GPR preloaded with  16 | (16 << 16)}
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

// TODO: add support for all the template parameters
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const bool narrow_tile = false, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t replay_buf_run_len  = 6;
    const std::uint32_t replay_buf_half_len = replay_buf_run_len >> 1;

    // Lambda function to set up replay buffer
    load_replay_buf(
        0,
        replay_buf_run_len,
        []
        {
            // Unpacks 1x16 row of datums to SrcA
            TTI_UNPACR(SrcA, 0b01000000 /*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

            // THCON_SEC0_REG3_Base_address_ADDR32 =  THCON_SEC0_REG3_Base_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;

            // Unpacks 1x16 row of datums to SrcA
            TTI_UNPACR(SrcA, 0b01000000 /*CH1_Y+=1*/, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

            // THCON_SEC0_REG3_Base_cntx1_address_ADDR32 =  THCON_SEC0_REG3_Base_cntx1_address_ADDR32 +  SCRATCH_SEC0_val_ADDR32
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        });

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                     // src B
        false,                                     // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_half_len), // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_half_len, replay_buf_half_len), // runs when context is 1
        0,
        0);

    tmp.program();
}

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
    // Sets the block_c_dim for unpack to use to increment the L1 address
    const std::uint32_t c_dim_size = SCALE_DATUM_SIZE(unpack_src_format, ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM)) >> 4;

    // This sets the scratch register that CFGSHIFTMASK instruction uses to increment the L1 address
    TT_SETDMAREG(0, LOWER_HALFWORD(c_dim_size), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(c_dim_size), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, SCRATCH_SEC0_val_ADDR32);
    TTI_NOP;

    // Unpack 1 row of 1x16 at a time for SrcA
    config_unpacker_x_end<p_setadc::UNP_A>(1);
    config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);

    // Set Y stride for SrcA to be one 1x16 row of datums
    std::uint32_t unpA_ch1_y_stride = SCALE_DATUM_SIZE(unpack_dst_format, FACE_C_DIM);
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(narrow_tile, num_faces);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_(
    std::uint32_t unpA_src_format,
    std::uint32_t face_r_dim,
    std::uint32_t narrow_tile,
    std::uint32_t base_address_a,
    std::uint32_t address_b,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t offset_address_a = SCALE_DATUM_SIZE(unpA_src_format, tile_index_a) << 1;
    const std::uint32_t address_a        = base_address_a + offset_address_a;

    const std::uint32_t block_c_dim = block_ct_dim * ((num_faces == 1) ? FACE_C_DIM : TILE_C_DIM) * face_r_dim;
    const bool run_r_dim_loop       = (face_r_dim > 1);

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Clear z/w start counters for SrcA/B
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, 0b1111);

    for (std::uint32_t n = 0; n < num_faces; n++)
    {
        /*
        Face 0: address = base_address
        Face 1: address = base_address + 1x16 row of datums
        Face 2: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim (address for the bottom 2 faces of tiles)
        Face 3: address = base_address + block_ct_dim * TILE_C_DIM * face_r_dim + 1x16 row of datums
        */
        std::uint32_t address_face_a = (n % 2 == 0) ? address_a : (address_a + (SCALE_DATUM_SIZE(unpA_src_format, FACE_C_DIM) >> 4));
        address_face_a += (n >= 2) ? ((SCALE_DATUM_SIZE(unpA_src_format, block_c_dim)) >> 4) : 0;

        // Wait for free context
        wait_for_next_context(2);

        if constexpr (neginf_srcA)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr::UNP_CLRSRC_NEGINF, p_unpacr::UNP_CLRSRC);
        }
        else if constexpr (zero_srcA_reduce)
        {
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
        }

        // Validate and configure addresses
        _llk_unpack_configure_addresses_(address_face_a, address_b, cfg);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // Reset Y counters for SrcA
        TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010);
        // Unpack SrcB 16x16 face & Set Data Valid

        // If reload_srcB, only first face needs to be loaded, otherwise CH0_Z+=1
        TTI_UNPACR(SrcB, reload_srcB ? 0b0 : 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // Unpacks face_r_dim-1 rows of 1x16 datums to SrcA
        if (run_r_dim_loop)
        {
            ckernel_unpack_template::run(face_r_dim - 1, unp_cfg_context == 0 ? 0 : 0xffff);
        }

        // Unpack last SrcA row of a 16x16 face and SetDvalid
        TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

inline void _llk_unpack_tilizeA_B_uninit_(const std::uint32_t unpack_dst_format, const std::uint32_t face_r_dim)
{
    // Revert X dim value to default.
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    // TODO NC: Issue tt-llk#1036 will make this transient
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);

    // _llk_unpack_tilizeA_B uses y-stride and updates y counter
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);

    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // GPR preloaded with  16 | (16 << 16)}
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;
}

// ============================================================================
// BH Fast-Tilize Unpack
//
// Software tilization via UNPACR address modes. Processes 4 tiles at a time
// (unit_dim=4). Each UNPACR reads 128 datums (4 tile widths) into 8 SrcA rows.
// CH1_Z stride = 256 bytes (8 contiguous SrcA rows per read).
// MASK_LOOP MOP with zmask=0x80808080 fires dvalid every 8th read
// (32 reads total, 4 dvalids per unit).
// ============================================================================

inline void _llk_unpack_fast_tilize_mop_config_()
{
    // addr_mode: CH0_Z+=1 (next L1 row), CH1_Z+=1 (next SrcA dest with gap)
    constexpr std::uint8_t ADDRMOD = 0b00'01'00'01;

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                 // unpackB
        false,                                 // unpackHalo
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD, 0), // A0: read, no dvalid
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_UNPACR_COMMON(SrcA, ADDRMOD, 1), // skipA: read WITH dvalid
        TT_OP_NOP,
        TT_OP_NOP);
    tmp.program();
}

// BH fast-tilize: block height is always 1 (one row of tiles per call).
// Multiple rows are handled by the caller, looping over rows and calling
// the block function once per chunk per row.
inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t ct_dim)
{
    // Context-safe writes only: Tile_x_dim (WRCFG below writes the full 32-bit word,
    // covering cntx0 low-16 and cntx1 high-16), TileDescriptor (shared across contexts),
    // Zstride (RMW on shared reg), and SETADCXX (thread-scoped counter, not per
    // cfg context — see ISA SETADCXX). Per-call context switching happens in
    // _llk_unpack_fast_tilize_block_.

    // Save state
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    // Save the unpacker Out_data_format word so we can restore it in uninit if the
    // fp32/tf32 → bf16 downgrade below modifies it.
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, THCON_SEC0_REG2_Out_data_format_ADDR32);

    // BH fast-tilize forces a 16-bit DEST view in the math thread (MOVA2D cannot
    // safely write Dst32b in this flow — see _llk_math_fast_tilize_init_). That makes
    // TF32/Float32 SrcA incompatible: MOVA2D(TF32) + Dst16b is ISA UB, and MOVA2D(bf16)
    // + Dst32b is also UB. The only consistent combination is bf16 SrcA + Dst16b, so
    // downgrade SrcA output to Float16_b here when the caller requested fp32/tf32.
    // The unpacker performs the fp32 → bf16 conversion on the L1 → SrcA path. This
    // matches the precision Metal previously consumed from the fp32 fast-tilize path.
    const std::uint32_t effective_dst_format = (unpack_dst_format == (std::uint32_t)DataFormat::Float32 || unpack_dst_format == (std::uint32_t)DataFormat::Tf32)
                                                   ? (std::uint32_t)DataFormat::Float16_b
                                                   : unpack_dst_format;
    if (effective_dst_format != unpack_dst_format)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(effective_dst_format);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    // Tile_x_dim = 32, Tile_y_dim = ct_dim, Tile_z_dim = 16
    TT_SETDMAREG(0, TILE_C_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, TILE_C_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);

    TT_SETDMAREG(0, ct_dim, 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    // BH HW bug (AutoTTSync.md †): TileDescriptor words 1-3 (YDim, ZDim) are not
    // tracked as unpacker resources. Explicit stall ensures the write completes.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    // X counter end = 4 tile widths (maximum unit_dim). reinit_xdim adjusts this
    // before any chunk that uses a smaller unit_dim (2 or 3).
    // CH1_Z stride stays at 4-wide (8 SrcA rows) regardless of unit_dim.
    // This creates natural gaps in SrcA for unit_dim < 4, preserving the DEST layout.
    TT_SETADCXX(p_setadc::UNP_A, 4 * TILE_C_DIM - 1, 0x0);

    // CH1 Z stride: controls SrcA dest address gap between reads.
    // Uses effective_dst_format because Float32/Tf32 are downgraded to bf16 above.
    const std::uint32_t ch1_x_stride = (effective_dst_format == (std::uint32_t)DataFormat::Float32 ||
                                        effective_dst_format == (std::uint32_t)DataFormat::Int32 || effective_dst_format == (std::uint32_t)DataFormat::Tf32)
                                           ? 4
                                           : 2;
    // stride = 4 * 32 * 2 = 256 bytes = 8 contiguous SrcA rows per read
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(4 * TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}

// Reconfigure X counter for a different unit_dim without full reinit.
// CH1_Z stride and MOP stay unchanged — only the read width changes.
inline void _llk_unpack_fast_tilize_reinit_xdim_(const std::uint32_t unit_dim)
{
    TT_SETADCXX(p_setadc::UNP_A, unit_dim * TILE_C_DIM - 1, 0x0);
}

// One call = one row-chunk (one unit_dim, one MOP run).
// Block height is always 1; multiple rows and chunks are loops in the caller.
inline void _llk_unpack_fast_tilize_block_(
    const std::uint32_t base_address,
    [[maybe_unused]] const std::uint32_t tile_index,
    [[maybe_unused]] const std::uint32_t unpack_src_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4,
    const std::uint32_t col_start                  = 0)
{
    // Standard BH unpacker context dance (see _llk_unpack_untilize_pass_).
    // Programs REG3 Base_address for the current cfg context via cfg[] write,
    // then synchronises Trisc↔T6 so the MOP runs with that base in place.
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base_address, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // L1 addressing via CH0 counters:
    //   Y selects tile within row (Y+=col_start to reach this chunk's start column)
    //   Z selects tensor row (0..31)
    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b0011); // reset CH0_X=0, CH0_Y=0
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111); // reset all Z,W = 0

    // Position at col_start (for chunks after the first in a row).
    // INCADCXY Ch0_Y field is 3 bits (max 7), so loop for larger offsets.
    {
        std::uint32_t remaining = col_start;
        while (remaining > 0)
        {
            std::uint32_t inc = (remaining > 7) ? 7 : remaining;
            TT_INCADCXY(p_setadc::UNP_A, 0, 0, inc, 0);
            remaining -= inc;
        }
    }

    // Hoist zmask high 16 bits — they persist in mop_zmask_hi16 until changed.
    constexpr std::uint32_t ZMASK = 0x80808080;
    TT_MOP_CFG(ZMASK >> 16);
    TT_MOP(0, 32 - 1, ZMASK & 0xFFFF);

    // Release the unpacker context acquired above and advance the software
    // tracker so the next call targets the other cfg context slot.
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_uninit_()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, 0, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, 0, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    // Restore Out_data_format (init may have downgraded it from fp32/tf32 to bf16).
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);

    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, 0b1010);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111);
}
