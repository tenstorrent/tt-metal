// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../../common/tensor_shape.h"
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
 * @brief Program the unpacker MOP for the non-8-bit tilize path (Blackhole whole-tile workaround).
 *
 * Selects between unpacking to SrcA (with a SrcB dvalid NOP) or straight to dest. 8-bit formats
 * use the inline 2-context path in @ref _llk_unpack_tilize_ instead (no MOP).
 *
 * @param unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 */
inline void _llk_unpack_tilize_mop_config_(const bool unpack_to_dest = false)
{
    // SrcB SET_DVALID + UNP_ZEROSRC: math's ELWADD branch (FP32 dest-accumulate) reads SrcB,
    // so it must be both dvalid and zeroed. MOVA2D branch ignores it (harmless handshake).
    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(0, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    ckernel_template tmp(1 /*outerloop*/, 1 /*innerloop*/, unpack_to_dest ? unpack_srca_to_dest : unpack_srcb_set_dvalid);

    if (!unpack_to_dest)
    {
        tmp.set_start_op(unpack_srca);
    }

    tmp.program();
}

/**
 * @brief Initialize the unpacker for a tilize operation.
 *
 * Disables face transpose, configures the unpacker into tileize mode (throttle, shift amount,
 * per-tile X/Z dims) for the given block column dimension, decides whether 32-bit datums must be
 * unpacked to dest, and:
 *   - 8-bit formats: programs the 1x16 face x_dim per context and (for num_faces==4) the
 *     SCRATCH_SEC0_val preload used by the inline CFGSHIFTMASK. No MOP is programmed —
 *     the UNPACR sequence is issued inline by @ref _llk_unpack_tilize_.
 *   - non-8-bit formats: programs the whole-tile x_dim/z_dim descriptor state and the
 *     MOP template via @ref _llk_unpack_tilize_mop_config_ (Blackhole workaround).
 *
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param ct_dim: Number of column tiles in the block, used to size the column dimension.
 * @param face_r_dim: Rows per face, valid values = <2, 4, 8, 16>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces). Not supported on the 8-bit path (asserted false).
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @note Call @ref _llk_unpack_tilize_uninit_ to restore the modified tile-descriptor state.
 * @ref _llk_unpack_tilize_ is the matching execute call.
 * @ref _llk_math_eltwise_unary_datacopy_init_ (A2D, PackMode::Tilize) is the matching init on the math thread.
 */
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

    const bool is_8bit_format        = IS_8BIT_FORMAT(unpack_src_format);
    const std::uint32_t shift_amount = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    // Override default settings to enable tilize mode
    unpack_config_u config   = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    config.f.tileize_mode    = 1;
    config.f.shift_amount    = shift_amount;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
    if (is_8bit_format)
    {
        LLK_ASSERT(!narrow_tile, "8-bit tilize narrow_tile not supported");
        LLK_ASSERT(!unpack_to_dest, "8-bit tilize unpack_to_dest not supported");

        // x_dim per context = FACE_C_DIM (1x16 face read). FACE_DIM_1x16 holds
        // FACE_C_DIM packed in both halfwords, so a 32b WRCFG programs both cntx0 and cntx1.
        TTI_WRCFG(p_gpr_unpack::FACE_DIM_1x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);

        // SCRATCH_SEC0_val is the increment CFGSHIFTMASK adds to Base_address between
        // top and bottom face pairs. Only used when num_faces==4 (top pair to bottom pair).
        // For num_faces==2 (top pair only) no CFGSHIFTMASK fires, so the preload is skipped.
        //   bot_face_offset = shift_amount * face_r_dim       (in 16B words, jump from top to bottom faces)
        //   -2 compensates for Z=2 L1 offset (Z * XDim * DatumSize = 2 * 16 * 1 = 32 bytes = 2 16B-words)
        if (num_faces == 4)
        {
            const std::uint32_t base_address_increment = shift_amount * face_r_dim - 2;
            TT_SETDMAREG(0, LOWER_HALFWORD(base_address_increment), 0, LO_16(p_gpr_unpack::TMP0));
            TT_SETDMAREG(0, UPPER_HALFWORD(base_address_increment), 0, HI_16(p_gpr_unpack::TMP0));
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        }
    }
    else
    {
        // Whole-tile unpacking (non-8-bit BH workaround): x_dim covers the entire tile, z_dim=1
        const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
        const std::uint32_t Tile_z_dim = 1;
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));
        // Set x-dim to cover entire tile (face_r_dim * num_faces * FACE_C_DIM)
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (Tile_x_dim << 16));
        // Set z-dim to 1 as X dim is set to cover the entire tile, so no need to iterate over faces.
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (Tile_z_dim << 16));

        // Set x-end for Unpackers to (face_r_dim * num_faces * FACE_C_DIM - 1)
        TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0);

        _llk_unpack_tilize_mop_config_(unpack_to_dest);
    }
}

/**
 * @brief Unpack and tilize a tile from L1 into SrcA or the dest register.
 *
 * Computes the L1 base address for the selected column tile and unpacks it into SrcA.
 *   - 8-bit formats: issues an inline 2-context UNPACR sequence (2 UNPACRs for num_faces=2, or
 *     4 UNPACRs + CFGSHIFTMASK for num_faces=4) plus a SrcB SET_DVALID + UNP_ZEROSRC handshake.
 *     Emits 1 DVALID per tile. No MOP; `face_r_dim` is not consumed by the sequence itself
 *     (already programmed via init's SCRATCH_SEC0_val preload) but drives the init-time preload.
 *   - non-8-bit formats: runs the whole-tile MOP programmed at init, and — when unpacking
 *     32-bit datums to dest — manages the dest write address and completion handshake.
 *
 * @param base_address: L1 base address of the source tile buffer.
 * @param tile_index: Column tile index selecting which tile to unpack.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param unpack_dst_format: Destination data format the operand is converted to.
 * @param face_r_dim: Rows per face.
 * @param num_faces: Number of faces in the tile, valid values = <2, 4>.
 * @param narrow_tile: Whether the tile is narrow (single column of faces). Not supported on the 8-bit path.
 * @note Call @ref _llk_unpack_tilize_init_ before this function, and
 *       @ref _llk_unpack_tilize_uninit_ after it to restore modified state.
 */
inline void _llk_unpack_tilize_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    std::uint32_t unpack_src_format                 = 0,
    std::uint32_t unpack_dst_format                 = 0,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool narrow_tile                          = false)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4 for tilize");

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // In case of 32-bit numbers, we have to unpack into dest register
    // For integers, always unpack to dest. For Float32, only if unpack_dst_format is Float32 (lossless tilize mode)
    auto unpack_source_format = static_cast<DataFormat>(unpack_src_format);
    auto unpack_dest_format   = static_cast<DataFormat>(unpack_dst_format);

    const bool unpack_to_dest =
        (unpack_source_format == DataFormat::UInt32) || (unpack_source_format == DataFormat::Int32) || (unpack_dest_format == DataFormat::Float32);

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << (narrow_tile ? 0 : 1);

    // 8-bit path: 2-context inline UNPACR sequence. AddrMode=0b00010001 advances BOTH
    // Ch0.Z (L1 input read) and Ch1.Z (SrcA output write) so face N lands in SrcA rows
    // 16*N..16*N+15 via the Zstride configured by configure_unpack_AB. This relies on
    // SRCA_SET_SetOvrdWithAddr = 1 (letting OutAddr drive the SrcA row directly, without
    // SrcRow being added). Contract: the bit is set to 1 by configure_unpack_AB at unpack
    // init (see cunpack_common.h:946 -- TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4)) and is the
    // resting state. It is transiently cleared to 0 only by set_dst_write_addr (cunpack
    // _common.h:1015) with a paired restore in unpack_to_dest_tile_done (line 1009).
    // Because this path asserts !unpack_to_dest at init, we can never be inside that
    // transient window on entry -- the bit is guaranteed to be 1. NOTE: future FP8
    // unpack_to_dest support would enter with the bit cleared and must either restore
    // it explicitly or use a different addressing scheme.
    // The active context selects between REG3_Base_address (cntx0) and REG3_Base_cntx1
    // _address for both the base-address cfg write and the CFGSHIFTMASK target.
    //   num_faces==4: 4 UNPACRs (face 3 with SetDvalid) + CFGSHIFTMASK between top and
    //                 bottom face pairs (Ch1.Z keeps advancing across the CFGSHIFTMASK
    //                 since it touches REG3 only).
    //   num_faces==2: 2 UNPACRs (top pair only, face 1 with SetDvalid). No CFGSHIFTMASK.
    if (IS_8BIT_FORMAT(unpack_src_format))
    {
        const std::uint32_t address = base_address + top_face_offset_address;
        LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

        // Clear Z/W counters on UNP_A for both channels (Ch0 for L1 read, Ch1 for SrcA write)
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

        // Wait for a free context (up to 2 tiles in flight)
        wait_for_next_context(2);

        // Write top-face base address to the active context's REG3
        if (0 == unp_cfg_context)
        {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        }
        else
        {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        }
        semaphore_post(semaphore::UNPACK_SYNC);

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        if (num_faces == 4)
        {
            // Face 0: Ch0.Z=0, Ch1.Z=0 -> SrcA rows 0..15
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            // Face 1: Ch0.Z=1, Ch1.Z=1 -> SrcA rows 16..31
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

            // Shift the active context's Base_address by SCRATCH_SEC0_val (= bot_face_offset - 2
            // in 16B words) so face 2/3 read correctly at Z=2/3 on the L1 side. CFGSHIFTMASK's
            // target is an instruction immediate, so we pick the variant at compile time per context.
            if (0 == unp_cfg_context)
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            }
            else
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            }
            TTI_NOP;

            // Face 2: Ch0.Z=2, Ch1.Z=2 -> SrcA rows 32..47
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            // Face 3: Ch0.Z=3, Ch1.Z=3 -> SrcA rows 48..63. SetDvalid flips the SrcA bank,
            // handing the full 64-row tile to math.
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        else // num_faces == 2: top face pair only, no CFGSHIFTMASK
        {
            // Face 0: Ch0.Z=0, Ch1.Z=0 -> SrcA rows 0..15
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 0 /*no Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            // Face 1: Ch0.Z=1, Ch1.Z=1 -> SrcA rows 16..31. SetDvalid flips the SrcA bank.
            TTI_UNPACR(SrcA, 0b00010001 /*Ch0.Z+=1, Ch1.Z+=1*/, 0, 0, 0, 1, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }

        // SrcB DVALID + UNP_ZEROSRC. Required so math's ELWADD branch (dest_acc=Yes,
        // is_int_fpu_en, or dst==UInt8) sees a zero SrcB; for the MOVA2D branch it's
        // a no-op handshake but kept uniform across formats.
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Flip to the other context for the next tile
        switch_config_context(unp_cfg_context);
    }
    else
    {
        // Non-8-bit path: whole-tile unpack via BH workaround configuration (x_dim covers full tile).
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
            // Pair with set_dst_write_addr above (both keyed on unpack_src_format), so the
            // canonical Z-stride restore matches the value programmed on entry.
            unpack_to_dest_tile_done(unp_cfg_context, unpack_src_format);
        }

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}

/**
 * @brief Restore unpacker state after a tilize operation.
 *
 * Reverts the tile descriptor Z-dimension to default and rewrites the unpack config (clearing
 * tilize mode) so subsequent ops see a normal tile layout. x-start/x-end is transient and
 * reprogrammed by the next operation's init (see tt-llk#1036), so it is not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @param tensor_shape: Tile geometry; total_num_faces() restores the Z dimension (valid values
 *                      = <1, 2, 4>) and face_r_dim restores the canonical Tile_x_dim.
 * @note Call @ref _llk_unpack_tilize_init_ before this function.
 */
inline void _llk_unpack_tilize_uninit_(const std::uint32_t unpack_dst_format, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    const std::uint32_t num_faces  = tensor_shape.total_num_faces();
    const std::uint32_t face_r_dim = tensor_shape.face_r_dim;

    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // Restore tile-descriptor Z and X dim to the canonical baseline programmed by
    // configure_unpack_AB. Z-dim equals the operand's num_faces; X-dim is 0 because the
    // per-context override in Tile_x_dim_cntx0 (set below) is what the unpacker actually
    // consumes for srcA. The non-8-bit init path mutates X-dim (to face_r_dim*num_faces*FACE_C_DIM)
    // so it must be reverted here too to keep the operand operation-restorable.
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, TILE_DESC_UPPER_HALFWORD_MASK>(num_faces);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 16, TILE_DESC_UPPER_HALFWORD_MASK>(CANONICAL_UNPA_TILE_X_DIM);

    // The unpack-config[0] write below also clears tileize_mode, haloize_mode, and the
    // other word-0 fields back to 0, mirroring what the zero-initialised config struct
    // produces in configure_unpack_AB.
    unpack_config_u config = {0};

    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode   = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32);
    // Restore Tile_x_dim_cntx0 to the canonical face_dim-derived value. The previous
    // FACE_DIM_16x16 GPR was correct only for face_r_dim=16; tiny tiles need a
    // face_r_dim-aware value to match the baseline programmed by configure_unpack_AB.
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(canonical_unpA_tile_x_dim_cntx(face_r_dim));
    TTI_NOP;
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
// TODO: add support for all the template parameters
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_mop_config_(const std::uint32_t num_faces = 4)
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
 * @param ct_dim: Number of column tiles in the block, used to size the column stride.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @note Call @ref _llk_unpack_tilizeA_B_uninit_ to restore the modified stride/datum-count state.
 * @ref _llk_unpack_tilizeA_B_ is the matching execute call.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_init_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces       = 4,
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

    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(num_faces);
}

/**
 * @brief Tilize operand A and unpack operand B, face by face, into SrcA and SrcB.
 *
 * Loops over the faces, computing each face's SrcA L1 address, optionally clearing SrcA to
 * neginf/zero, unpacking the SrcB face, then unpacking the face's rows into SrcA (row by row via
 * the MOP) and setting data-valid, synchronizing through the unpack semaphore each iteration.
 *
 * @tparam neginf_srcA: Clear SrcA to negative infinity before unpacking (e.g. for max-reduce).
 * @tparam reload_srcB: Reload SrcB once rather than incrementing its face each step.
 * @tparam zero_srcA: Clear SrcA to zero before unpacking.
 * @tparam zero_srcA_reduce: Clear SrcA to zero before unpacking for a reduce fused with tilize.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param face_r_dim: Rows per face.
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param address_b: L1 address of operand B's face data.
 * @param tile_index_a: Column tile index into operand A.
 * @param block_ct_dim: Number of column tiles in the block, used to compute face strides.
 * @param num_faces: Number of faces in the tile, valid values = <1, 2, 4>.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ with matching template args before this function, and
 *       @ref _llk_unpack_tilizeA_B_uninit_ after it to restore modified state.
 */
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void _llk_unpack_tilizeA_B_(
    std::uint32_t unpA_src_format,
    std::uint32_t face_r_dim,
    std::uint32_t base_address_a,
    std::uint32_t address_b,
    std::uint32_t tile_index_a,
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

/**
 * @brief Restore unpacker state after a tilize-A-with-unpack-B operation.
 *
 * Reverts the SrcA Y counter and rewrites the unpack config and tile X-dim back to the default
 * 16x16 face layout. x-start/x-end is transient and reprogrammed by the next operation's init
 * (see tt-llk#1036), so it is not restored here.
 *
 * @param unpack_dst_format: Destination data format to restore in the unpack config.
 * @note Call @ref _llk_unpack_tilizeA_B_init_ before this function.
 */
inline void _llk_unpack_tilizeA_B_uninit_(const std::uint32_t unpack_dst_format)
{
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);

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
    // Restore canonical srcA Y-stride. _llk_unpack_tilizeA_B_init_ mutates it to a per-op
    // value (SCALE_DATUM_SIZE(unpack_dst_format, FACE_C_DIM)); restoring the baseline
    // programmed by configure_unpack_AB keeps this op from leaking Y-stride to the next op.
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(
        canonical_unpA_y_stride(unpack_dst_format));
    TTI_NOP;
}
