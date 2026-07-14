// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_pack.h"

using namespace ckernel;
using namespace ckernel::packer;

/*************************************************************************
 * LLK PACK FAST TILIZE (Tilize single input using both unpackers and packer)
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim and num_units must match the ones given to the unpacker and math (all unit_dim usage notes from the unpacker also apply here)
 * tile_index is the index of the tile inside the destination register to read from
 * address is the 16B address of where to start packing to (usually the start of the tile row)
 * currently supports only 4 16x16 faces per tile
 * supported output formats are: FP32, FP16_B, BFP8_B, BFP4_B
 * both dest modes are supported (same usage notes from math apply here)
 * only DstSync::SyncHalf is supported
 * tiles are expected to be split into top and bottom faces in separate halves of the active dest bank
 *************************************************************************/

/**
 * @brief Configure the ADDR_MOD slots used by the fast-tilize pack MOP.
 *
 * Programs the src Y increment/clear patterns that step the packer to the next row, to the same face
 * in the next tile, and back to the start of a unit. The per-row stride in ADDR_MOD_0 depends on the
 * number of contiguous faces loaded by a single unpacker instruction, which is set by unit_dim.
 *
 * @param unit_dim: Number of tiles processed per iteration (1, 2, or 3); selects the row stride.
 */
inline void _llk_pack_fast_tilize_addrmod_config_(const std::uint32_t unit_dim)
{
    // first two address mods move to the next row, the stride depends on the number of contiguous faces loaded in the single unpacker instruction
    // for unit_dim 1, that is 2 so the stride is 2, and analogously for unit_dims 2 and 3 its 4 and 6
    if (unit_dim == 1)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 2},
        }
            .set(ADDR_MOD_0);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 4},
        }
            .set(ADDR_MOD_0);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 6},
    }
        .set(ADDR_MOD_2);

    // this address mod moves to the same face in the next tile
    // go back to the first row using cr and then move by the number of contiguous faces in the tile (always 2 irrespective of unit_dim)
    addr_mod_pack_t {
        .y_src = {.incr = 2, .cr = 1},
    }
        .set(ADDR_MOD_1);

    // this address mod moves back to the beginning of the unit and separate instruction will increment the z counter to move to the next unit
    // unit here refers to the interleaved set of 4 * unit_dim faces (half in the top half of the active dest bank and half in the bottom half)
    addr_mod_pack_t {
        .y_src = {.clr = 1},
    }
        .set(ADDR_MOD_3);
}

/**
 * @brief Build and program the packer MOP template for fast tilize.
 *
 * Programs an unpack-style MOP whose loop body issues the common-packer PACR instructions (using
 * ADDR_MOD_0 and ADDR_MOD_2) that pack rows of the interleaved face layout into tilized L1 output.
 *
 * @note @ref _llk_pack_fast_tilize_addrmod_config_ must have programmed the ADDR_MOD slots.
 */
inline void _llk_pack_fast_tilize_mop_config_()
{
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,
        false,
        TT_OP_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0),
        TT_OP_NOP,
        TT_OP_NOP);

    tmp.program();
}

/**
 * @brief Initialize the packer for a fast-tilize pack op.
 *
 * Sets the 32-bit dest read mode, programs the per-tile L1 address offset, programs the per-packer
 * dest offsets for the interleaved top/bottom-half face layout, selects the dest registers, and
 * configures the ADDR_MODs and MOP. Only DstSync::SyncHalf is supported; supported output formats are
 * FP32, FP16_B, BFP8_B, and BFP4_B.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull> (only SyncHalf supported)
 * @param use_32bit_dest: True to read dest as 32-bit data (set when the unpack source format is TF32).
 * @param pack_dst_format: Destination (L1) data format.
 * @param unit_dim: Number of tiles processed per iteration, valid values = <1, 2, 3>
 * @param num_faces: Faces per tile, valid values = <2, 4>
 * @param l1_tile_elements: Number of datums per output tile, used to size the per-tile L1 offset.
 * @note On the unpack thread, pair with @ref _llk_unpack_fast_tilize_init_ and on the math thread with @ref _llk_math_fast_tilize_init_ (same unit_dim).
 * @note Pair with @ref _llk_pack_fast_tilize_uninit_ after the matching @ref _llk_pack_fast_tilize_block_ execute calls.
 */
template <DstSync Dst>
inline void _llk_pack_fast_tilize_init_(
    const std::uint32_t use_32bit_dest,
    const std::uint32_t pack_dst_format,
    const std::uint32_t unit_dim,
    const std::uint32_t num_faces        = 4,
    const std::uint32_t l1_tile_elements = TILE_C_DIM * TILE_R_DIM)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4");
    LLK_ASSERT(
        pack_dst_format == to_underlying(DataFormat::Float16_b) || pack_dst_format == to_underlying(DataFormat::Float32) || num_faces == 4,
        "16x32 tiny tiles are only supported with float16_b or float32 output formats");
    // instead of using the actual is_fp32_dest_acc_en flag dest 32 bit mode is enabled if unpack_dst_format is TF32
    // this is due to a hw quirk with MOVA2D and MOVB2D
    // so clear PCK_DEST_RD_CTRL_Read_32b_data unless unpack_src_format is TF32
    // unpack src format is not easy to determine here so use an argument that is going to be computed at the higher level
    if (!use_32bit_dest)
    {
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
    }

    // set the address offset to the size of the tile in 16B words
    std::uint32_t tile_size = _llk_pack_output_size_bytes_(pack_dst_format, l1_tile_elements) >> 4;
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, tile_size, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));

    // since faces are interleaved and the top and bottom faces are in the separate halves of the active dest bank, each packer needs a special offset
    // difference between 16 bit dest and 32 bit dest is where the half of the active bank is (256 rows vs 128 rows)
    // stallwait and select_packer_dest_registers just replicate what _llk_init_packer_dest_offset_registers_ does
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);

    // TTI_SETDMAREG requires compile-time immediate operands; branch on num_faces and use literals.
    if (!use_32bit_dest)
    {
        if (num_faces == 2)
        {
            // tiny tiles will use a similar packer scheme as unit_dim 2, but treats second tile as bottom 2 faces of the first tile
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x002, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x003, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x002, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x003, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
        }
        else
        {
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x100, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x101, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x100, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x101, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
        }
    }
    else
    {
        if (num_faces == 2)
        {
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x002, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x003, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x002, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x003, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
        }
        else
        {
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x080, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x081, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x080, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
            TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x081, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
        }
    }
    select_packer_dest_registers<Dst>();

    // each packer packs a single row per call and in total each packer will pack a single face
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    _llk_pack_fast_tilize_addrmod_config_(unit_dim);

    // UNPACR instructions are used with unit_dim 1 and 2 and SKIP instructions are used with unit_dim 3
    LLK_ASSERT(unit_dim == 1 || unit_dim == 2 || unit_dim == 3, "unit_dim must be 1, 2, or 3");
    _llk_pack_fast_tilize_mop_config_();
}

/**
 * @brief Tear down the packer after a fast-tilize pack op and restore default pack state.
 *
 * Restores the 32-bit dest read mode to the original is_fp32_dest_acc_en value, restores the default
 * packer dest offsets and X/counter state, and re-runs the standard packer init with default settings
 * so a subsequent (non fast-tilize) pack starts from a clean configuration.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @param pack_dst_format: Destination (L1) data format used to re-run the default packer init.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 * @note Call @ref _llk_pack_fast_tilize_init_ before this function.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_tilize_uninit_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // restore PCK_DEST_RD_CTRL_Read_32b_data to the original value
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(is_fp32_dest_acc_en);

    // restore default packer dest offsets
    _llk_init_packer_dest_offset_registers_<Dst>();

    // reset counters
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));

    // for some reason short inits avoid the packer init (probably since it is usually the same)
    // but that means calling it here with reasonable defaults is needed
    // it just initializes the address mods and mop
    _llk_pack_init_<PackMode::Default, false /* zero_output */>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile);
}

/**
 * @brief Fast-tilize-pack a block of units from the destination register to L1.
 *
 * Programs the L1 destination address and the packer counters (using the Z counter, whose stride is a
 * single face, to seek to the start tile), then packs num_units units, advancing the dest and L1
 * destination per unit. Each unit covers unit_dim tiles; the per-unit PACR sequence is selected by
 * unit_dim (1, 2, or 3).
 *
 * @param tile_index: Index of the first source tile in the destination register.
 * @param address: L1 destination base address (16B-aligned, typically the start of the tile row).
 * @param unit_dim: Number of tiles processed per unit, valid values = <1, 2, 3>
 * @param num_units: Number of units to pack in this call.
 * @param num_faces: Faces per tile, valid values = <2, 4>
 * @note Call @ref _llk_pack_fast_tilize_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_fast_tilize_uninit_ once all fast-tilize-pack calls are complete.
 * @note On the math thread, @ref _llk_math_fast_tilize_block_ must have written the split top/bottom-half faces into dest.
 */
inline void _llk_pack_fast_tilize_block_(
    const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t unit_dim, const std::uint32_t num_units, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 2 || num_faces == 4, "num_faces must be 2 or 4");
    // use false here so that the 31st bit of the address remains set as the offset addresses for the other packers continue to be used
    // while the address for the first packer is manipulated using ADDDMAREG and REG2FLOP
    program_packer_destination(address, false);

    // reset counters and set the W counter
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
    // move to the start tile index, instead of using the standard W counter whose stride is a single tile
    // use the Z counter whose stride is a single face as tiles are split into halves of the active dest bank
    // so only move 2 faces per tile_index
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index << 1);

    for (std::uint32_t i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in dest (same counter rationale as for tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 2); // CH0Z += 2
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // this pack should behave as a no op aside from the address mod side effect (which is resetting the Y counter to the beginning of the tile)
            // but it actually provides some kind of a stall required when modifying the L1 base address while the packer is running
            // and has less performance impact than a PACK PACK STALLWAIT
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
        else if (unit_dim == 2)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            // don't do this for tiny tiles, since we pack 2 tiny tiles as 1 full sized tile
            if (num_faces == 4)
            {
                TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
                TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
                // same notes for the flush bit as above
                // address mod here moves to the next tile in the same unit
                TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
                // pack a single tile
                // inside mop:
                // for (uint j = 0; j < 15; j++)
                // {
                //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
                // }
                TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
                TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            }
            // move to the next unit in dest (2 * 2 faces, same thing as tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 4); // CH0Z += 4
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here resets to the beginning of the unit
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
        else if (unit_dim == 3)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here moves to the next tile in the same unit
            TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here moves to the next tile in the same unit
            TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next unit in dest (3 * 2 faces, same thing as tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 6); // CH0Z += 6
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here resets to the beginning of the unit
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
    }
}
