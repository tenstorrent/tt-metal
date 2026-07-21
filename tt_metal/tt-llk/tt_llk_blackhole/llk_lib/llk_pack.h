// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"
#include "sanitizer/api.h"

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 *
 * Programs ADDR_MOD_0/1/2 with the src/dest Y and Z increment/clear patterns the pack MOP relies
 * on to traverse the destination register and step through faces for the given layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    if constexpr (pack_mode == PackMode::Untilize)
    {
        /*  Y src & Y dest inc by 1 to give strided increments:
            Rows: 0, 16, 1, 17, 2, 18, ........ 15, 31
        */
        addr_mod_pack_t {.y_src = {.incr = 1}, .y_dst = {.incr = 1}, .z_src = {.incr = 0}, .z_dst = {.incr = 0}}.set(ADDR_MOD_0);

        /* Increment Faces by 2 to give next 2 faces:
            Rows: 32, 48, 33, 49, 34, 50........47, 63
        */
        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}, .z_dst = {.incr = 0}}.set(ADDR_MOD_1);

        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}, .z_dst = {.incr = 0, .clr = 1}}.set(
            ADDR_MOD_2);
    }
    else if constexpr (pack_mode == PackMode::Tilize)
    {
        addr_mod_pack_t {.y_src = {.incr = 4}, .y_dst = {.incr = 2}, .z_src = {.incr = 0}, .z_dst = {.incr = 0}}.set(ADDR_MOD_0);

        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 0, .clr = 1}, .z_src = {.incr = 0}, .z_dst = {.incr = 0}}.set(ADDR_MOD_1);

        // Increment faces by 2 (jump 2 dest address 32)
        addr_mod_pack_t {.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}, .z_dst = {.incr = 0}}.set(ADDR_MOD_2);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 4},
            .y_dst = {.incr = 4},
        }
            .set(ADDR_MOD_0);

        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 1},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_1);

        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 4, .clr = 0, .cr = 0},
            .z_src = {.incr = 1, .clr = 0},
        }
            .set(ADDR_MOD_2);
    }
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 *
 * Programs the ckernel MOP (and, for tilize, a replay buffer) with the PACR instruction sequence
 * that packs one tile worth of data, selecting packer interfaces and ADDR_MODs per the layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note @ref _llk_pack_configure_addrmod_ must have programmed the ADDR_MOD slots for the same pack_mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    constexpr std::uint32_t MEGAROW          = 1;
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    if constexpr (pack_mode == PackMode::Untilize)
    {
        const std::uint32_t PACK_INTF_SEL  = (tile_c_dim < TILE_C_DIM) ? p_pacr::SINGLE_INTF_ACTIVE : p_pacr::TWO_INTFS_ACTIVE;
        const std::uint32_t MOP_INNER_LOOP = face_r_dim;
        const std::uint32_t MOP_OUTER_LOOP = (tile_c_dim < TILE_C_DIM) ? num_faces : (num_faces >> 1);

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL,
                0,
                MEGAROW,
                p_pacr::NO_CTXT_CTRL,
                0,
                0));

        tmp.set_last_inner_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            MEGAROW,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1));
        tmp.program();
    }
    else if constexpr (pack_mode == PackMode::Tilize)
    {
        const std::uint32_t PACK_INTF_SEL_0 = 0b0101;
        const std::uint32_t PACK_INTF_SEL_1 = 0b1010;
        const std::uint32_t MOP_INNER_LOOP  = 1;
        const std::uint32_t MOP_OUTER_LOOP  = (num_faces > 1) ? (num_faces >> 1) : 1;

        // Last row of half-tile (face_r_dim rows) is different between halves, so can't be replayed.
        LLK_ASSERT(face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be 2, 4, 8, or 16 for tilize");
        const std::uint32_t replay_buf_len = face_r_dim - 1;

        // This replay buffer finishes 2 faces
        load_replay_buf(
            0,
            replay_buf_len,
            // Lambda function to set up replay buffer
            [PACK_INTF_SEL_0, PACK_INTF_SEL_1, ZERO_OUTPUT_FLAG, MEGAROW, face_r_dim]
            {
                // Number of instructions per face (minus 1 for the last special instruction)
                const std::uint32_t num_instrs_per_face = (face_r_dim >> 1) - 1;

                // Face 0 -> mask rows 1010
                // First num_instrs_per_face instructions use ADDR_MOD_0
                for (std::uint32_t i = 0; i < num_instrs_per_face; i++)
                {
                    TTI_PACR(
                        p_pacr::CFG_CTXT_0,
                        p_pacr::NO_ROW_PAD_ZERO,
                        p_pacr::DST_ACCESS_NORMAL_MODE,
                        ADDR_MOD_0,
                        p_pacr::ADDR_CNT_CTXT_0,
                        ZERO_OUTPUT_FLAG,
                        PACK_INTF_SEL_0,
                        0,
                        MEGAROW,
                        p_pacr::NO_CTXT_CTRL,
                        0,
                        0);
                }
                // Last instruction for Face 0 uses ADDR_MOD_1
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_1,
                    p_pacr::ADDR_CNT_CTXT_0,
                    ZERO_OUTPUT_FLAG,
                    PACK_INTF_SEL_0,
                    0,
                    MEGAROW,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);

                // Face 1 -> mask rows 0101
                // num_instrs_per_face instructions use ADDR_MOD_0
                // The last instruction is handled separately outside the replay buffer
                for (std::uint32_t i = 0; i < num_instrs_per_face; i++)
                {
                    TTI_PACR(
                        p_pacr::CFG_CTXT_0,
                        p_pacr::NO_ROW_PAD_ZERO,
                        p_pacr::DST_ACCESS_NORMAL_MODE,
                        ADDR_MOD_0,
                        p_pacr::ADDR_CNT_CTXT_0,
                        ZERO_OUTPUT_FLAG,
                        PACK_INTF_SEL_1,
                        0,
                        MEGAROW,
                        p_pacr::NO_CTXT_CTRL,
                        0,
                        0);
                }
                // Last PACR instruction of the half-tile must go separately in the MOP. This is to be able to override it, to ensure that for the second half
                // the tile is closed correctly.
            });

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            lltt::replay_insn(0, replay_buf_len),
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_2,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL_1,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                0) // don't close tile
        );

        // Close the tile only when it is actually done.
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL_1,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1));

        tmp.set_end_op(TT_OP_SETADCZW(p_setadc::PAC, 0, num_faces >> 1, 0, 0, 0b0100)); // ch0_z = 0, ch1_z = num_faces >> 1;

        tmp.program();
    }
    else
    {
        const std::uint32_t PACK_INTF_SEL =
            face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

        const std::uint32_t MOP_INNER_LOOP = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
        const std::uint32_t MOP_OUTER_LOOP = num_faces * num_tiles;

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                ZERO_OUTPUT_FLAG,
                PACK_INTF_SEL,
                0,
                0,
                0,
                0,
                0));
        tmp.set_last_inner_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            ZERO_OUTPUT_FLAG,
            PACK_INTF_SEL,
            0,
            0,
            0,
            0,
            1));

        // if (partial_face) {
        //     tmp.set_start_op(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0,
        //     ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Don't close the tile, point to the next face
        //     tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0)); // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
        //     tmp.set_loop_op1(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0,
        //     ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Close the tile
        // }

        tmp.program();
    }
}

namespace llk_pack_internal_bh
{
/**
 * @brief Shared packer-init body behind the @ref _llk_pack_init_ overloads.
 *
 * Runs the common init sequence with each stage individually skippable: program the ADDR_MOD slots,
 * build the MOP template, set the packer strides, and program the packer X (datum) counter. The
 * skip_* template flags let a caller reuse state already established by a prior init or hw-configure.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, the packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave the ADDR_MOD slots untouched.
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @param pack_src_format: Source (dest register) data format; only used when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note Init owns the packer X (datum) counter (SETADCXX): every init programs its own value, mirroring
 *       the Wormhole contract. On Blackhole the value is always a single row (FACE_C_DIM - 1).
 */
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides>
inline void pack_init_apply(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);
    }

    // Program the packer X (datum) counter. Per the "inits own SETADCXX" contract, every init sets its
    // own value; on Blackhole x_start/x_end must stay within a single row (0..FACE_C_DIM-1).
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}
} // namespace llk_pack_internal_bh

/**
 * @brief Reconfigure the packer source/destination data formats and tile geometry at runtime.
 *
 * Used to switch the packer to a new data format without a full HW re-configure.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    llk::san::pack_operand_configure<true>(
        is_fp32_dest_acc_en, pack_src_format, pack_dst_format, llk::san::IGNORE, tile_c_dim, num_faces, partial_face, llk::san::IGNORE);

    reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
}

/**
 * @brief Enable or disable reading the destination register as 32-bit data for the packer.
 *
 * @param enable: True to read dest as 32-bit (FP32) data, false otherwise.
 * @note Stalls on the pack pipe before modifying the PCK_DEST_RD_CTRL config register.
 */
inline void _llk_pack_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
}

/**
 * @brief One-time hardware configuration of the packer for a given data format and tile geometry.
 *
 * Programs the packer config registers (formats, strides, relu) for the chosen pack mode. Call once
 * before the init/execute sequence.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param relu_config: Packed relu mode and threshold configuration (0 disables relu).
 * @note For 8-bit unpack-source datums, do not use PackMode::Tilize: the Blackhole row-unswizzling workaround is skipped (and is unnecessary, as 8-bit formats
 * are unaffected by the issue).
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t tile_c_dim  = TILE_C_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const std::uint32_t relu_config = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // sstanisic todo: partial face, narrow tile are weird (see #47440)
    llk::san::pack_operand_configure(is_fp32_dest_acc_en, pack_src_format, pack_dst_format, face_r_dim, tile_c_dim, num_faces, partial_face, llk::san::IGNORE);

    configure_pack<is_fp32_dest_acc_en, pack_mode>(pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, relu_config);
}

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op given a source format.
 *
 * The single Blackhole pack-init entry point: programs ADDR_MODs, the MOP template, (unless skipped)
 * the packer strides, and the packer X (datum) counter. Per the "inits own SETADCXX" contract, every
 * init sets its own X-counter value (FACE_C_DIM - 1 on Blackhole). When packing tilized 8-bit datums
 * the Blackhole row-unswizzle workaround can be skipped (the issue does not affect 8-bit datums).
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: When true, do not re-program the packer strides (e.g. when a prior
 *         hw-configure / reconfig already established them, or the caller programs them itself).
 * @param pack_src_format: Source (dest register) data format. Only consulted when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @param skip_bh_tilize_workaround: When true (8-bit src datums), skip the Blackhole tilize row-unswizzle workaround.
 * @note Pair with @ref _llk_pack_uninit_ after the matching @ref _llk_pack_ execute calls.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool skip_bh_tilize_workaround)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const DataFormat src_format = static_cast<DataFormat>(pack_src_format);
    if (src_format == DataFormat::Float32)
    {
        LLK_ASSERT(num_tiles <= 4, "Max supported num_tiles for FLOAT32 is 4.");
    }
    else if ((src_format == DataFormat::Float16) || (src_format == DataFormat::Float16_b))
    {
        LLK_ASSERT(num_tiles <= 8, "Max supported num_tiles for FLOAT16 or FLOAT16_B is 8.");
    }

    llk::san::pack_operand_check(llk::san::IGNORE, pack_src_format, llk::san::IGNORE, face_r_dim, tile_c_dim, num_faces, llk::san::IGNORE, llk::san::IGNORE);
    llk::san::operation_init<llk::san::Operation::Pack>();

    // 8bit datums in the unpack src format are not affected by the blackhole issue,
    // so we can skip the workaround which involves unswizzling rows in the tile.
    if (skip_bh_tilize_workaround && pack_mode == PackMode::Tilize)
    {
        llk_pack_internal_bh::pack_init_apply<PackMode::Default, zero_output, skip_addrmod_config, skip_packer_strides>(
            pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
    }
    else
    {
        llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
            pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
    }
}

/**
 * @brief Tear down the packer after a pack op (no-op on Blackhole).
 *
 * On Blackhole @ref _llk_pack_init_ always sets the PAC X counter to FACE_C_DIM - 1 (a single row),
 * which is also its default value, so there is no per-pack state for this teardown to restore.
 *
 * @note Pairs with @ref _llk_pack_init_.
 */
inline void _llk_pack_uninit_()
{
    // sstanisic todo: contract cannot be enforced if Pack has an uninit, without killing performance
    // llk::san::operation_uninit<llk::san::Operation::Pack>();

    // No state to restore - Blackhole pack_init sets PAC X counter to FACE_C_DIM - 1 which is the default.
}

/**
 * @brief Pack one tile from the destination register to an L1 address.
 *
 * Selects the source dest tile, programs the L1 destination address, runs the packer MOP, and resets
 * the Z counters afterward.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (Tilize not supported here)
 * @param tile_index: Index of the source tile in the destination register.
 * @param address: L1 destination address for the packed tile.
 * @note Call @ref _llk_pack_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_uninit_ once all pack calls are complete.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    llk::san::operation_check<llk::san::Operation::Pack>();

    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Blackhole: _llk_pack_ supports PackMode::Default and PackMode::Untilize only");
    set_dst_write_addr(tile_index);

    program_packer_destination(address);

    ckernel::ckernel_template::run();

    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
}
