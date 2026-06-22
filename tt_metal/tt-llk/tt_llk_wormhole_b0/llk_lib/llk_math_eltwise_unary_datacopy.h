// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "sanitizer/api.h"

using namespace ckernel;

// local function declarations
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format);

/**
 * @brief Copy a tile into the destination register, optionally broadcasting source B.
 *
 * For the unpack-to-dest path with 32-bit data, applies the hi16/lo16 broadcast workarounds (Issue #449);
 * otherwise runs the preconfigured datacopy MOP.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param dst_index: Tile index into the destination register.
 * @param src_format: Source data format (DataFormat enum underlying value).
 * @param dst_format: Destination data format (DataFormat enum underlying value).
 * @note Call @ref _llk_math_eltwise_unary_datacopy_init_ with matching template args before this function, and
 *       @ref _llk_math_eltwise_unary_datacopy_uninit_ after it to restore modified state.
 * @note On the unpack thread, @ref _llk_unpack_A_ must feed the tile into SrcA/SrcB (or dest for unpack-to-dest).
 */
template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_(const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format)
{
    if constexpr (type == DataCopyType::A2D)
    {
        llk::san::math_operand_check(dst_format, llk::san::IGNORE);
    }
    else
    {
        llk::san::math_operand_check(llk::san::IGNORE, dst_format);
    }
    llk::san::operation_check<llk::san::Operation::EltwiseUnaryDatacopy>(type, src_b_bcast_type, dst_format);

    if (unpack_to_dest && is_32bit_input(src_format, dst_format))
    {
        math_unpack_to_dest_math_ready();
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        math::math_unpack_to_dest_tile_ready();

        // Tile base row in Dst32b space: each 32x32 tile is 4 faces × 16 rows = 64 rows.
        const std::uint32_t tile_base = dst_index * 64;

        // Switch from the default SrcA format bank to the override bank so manual SrcA_val writes control MOVB2D behavior.
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);

        // The 32b hi16/lo16 MOVB2D below must not flush datums with a zero low byte; own the Src
        // zero-substitution flag via the math state tracker.
        math::_configure_mov_ops_zero_flag_state_();

        if constexpr (src_b_bcast_type == BroadcastType::ROW)
        {
            TTI_SETDVALID(0b10);

            // Broadcast 32-bit data in 2 parts (hi16 then lo16).
            // MOVB2D(DEST_NORM) with SrcAFmt=TF32 writes hi16 to Dst32b (Adj32 addressing).
            // MOVB2D(DEST_32B_LOW) with SrcAFmt!=TF32 writes lo16 to Dst32b.
            // Process one source row at a time so B data is consumed before overwrite.

            // Source row 0 → faces 0 (rows 0-15) and 2 (rows 32-47)
            TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 0);
            TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 0);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));

            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 0);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 8);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 32);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 40);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Float32));

            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 0);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 8);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 32);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 40);

            // Source row 16 → faces 1 (rows 16-31) and 3 (rows 48-63)
            TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 16);
            TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 16);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));

            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 16);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 24);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 48);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 56);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Float32));

            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 16);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 24);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 48);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, tile_base + 56);

            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
        {
            TTI_SETDVALID(0b10);

            TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 0);
            TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, tile_base + 0);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));

            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 0);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 8);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 16);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 24);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 32);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 40);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 48);
            TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 56);

            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Float32));

            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 0);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 8);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 16);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 24);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 32);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 40);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 48);
            TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, tile_base + 56);

            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            TTI_SETDVALID(0b10);

#pragma GCC unroll 2
            for (int offset = 0; offset < 2; ++offset)
            {
                // Face base row in Dst32b: offset 0 = faces 0+1 (rows 0-31), offset 1 = faces 2+3 (rows 32-63).
                const std::uint32_t face_base = tile_base + offset * 32;

                TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 0);
                TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 4);
                TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 8);
                TT_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 12);
                TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 0);
                TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 4);
                TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 8);
                TT_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, face_base + 12);

                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));

                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 0);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 4);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 8);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 12);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 16);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 20);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 24);
                TT_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 28);

                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Float32));

                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 0);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 4);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 8);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 12);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 16);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 20);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 24);
                TT_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, face_base + 28);
            }

            TTI_CLEARDVALID(0b10, 0);
        }
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
    }
    else
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (is_fp32_dest_acc_en && src_b_bcast_type != BroadcastType::NONE)
        {
            // UInt16 case needs to use format switching for 32bit dest
            // without the debug bit 11 hack to write into high bits
            // avoiding BroadcastType::NONE mode as that path is used by SFPU
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);
                cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));
            }
        }

        if constexpr (type == DataCopyType::A2D)
        {
            ckernel_template::run();
        }
        else if constexpr (type == DataCopyType::B2D)
        {
            if constexpr (src_b_bcast_type == BroadcastType::COL)
            {
                // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0);
            }
            else
            {
                ckernel_template::run();
            }
        }

        if constexpr (is_fp32_dest_acc_en && src_b_bcast_type != BroadcastType::NONE)
        {
            // Undo format switching option: clear the override and zero-flag-disable bits set above so the
            // implied SrcA format and default zero-flag behavior are restored (matches the Blackhole path).
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
                cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            }
        }

        math::clear_dst_reg_addr();
    }
}

/**
 * @brief Program the address-mod slots for a datacopy: single-row and 8-row (or 4-row for UInt16) dest/source steps.
 *
 * The increment pattern depends on the datacopy direction (A2D walks SrcA, B2D walks SrcB) and broadcast type;
 * UInt16 B2D uses 4-row steps because it relies on MOVB2D.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param dst_format: Destination data format (DataFormat enum underlying value); selects the UInt16 4-row step.
 */
template <DataCopyType type, BroadcastType bcast_type = BroadcastType::NONE>
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format)
{
    // Use srcA for data movement
    if constexpr (type == DataCopyType::A2D)
    {
        addr_mod_t {
            .srca = {.incr = 1},
            .srcb = {.incr = 0},
            .dest = {.incr = 1},
        }
            .set(ADDR_MOD_0);

        // Just unpack into A and move to Dest
        addr_mod_t {
            .srca = {.incr = 8},
            .srcb = {.incr = 0},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_2);
    }
    else
    {
        if constexpr (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR)
        {
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 1},
            }
                .set(ADDR_MOD_0);

            // Just unpack into B and move to Dest
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_2);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 1},
                .dest = {.incr = 1},
            }
                .set(ADDR_MOD_0);

            // Just unpack into B and move to Dest
            if (dst_format == to_underlying(DataFormat::UInt16)) // UInt16 case needs to use MOVB2D, which is 4 rows per op
            {
                addr_mod_t {
                    .srca = {.incr = 0},
                    .srcb = {.incr = 4},
                    .dest = {.incr = 4},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 0},
                    .srcb = {.incr = 8},
                    .dest = {.incr = 8},
                }
                    .set(ADDR_MOD_2);
            }
        }
    }

    if constexpr (bcast_type != BroadcastType::NONE)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_3);
    }
}

/**
 * @brief Program the datacopy MOP, selecting the move instruction (MOVA2D/MOVB2D/ELWADD) per direction, format, and broadcast.
 *
 * A2D normally uses MOVA2D but falls back to ELWADD when dest is FP32/INT (except UInt16, which stays on MOVA2D) or the datum is UInt8; B2D uses
 * MOVB2D (or ELWADD for non-UInt16 column broadcast) with loop counts derived from the broadcast type.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam is_int_fpu_en: Enable integer FPU datapath (forces the ELWADD move path like FP32 dest).
 * @param rows_per_inst: Rows moved per instruction (selects single-row vs. multi-row addr mode and loop count).
 * @param total_rows: Total rows to move across the inner loop.
 * @param num_faces: Number of faces in the tile.
 * @param dst_format: Destination data format (DataFormat enum underlying value); selects UInt16 special-casing.
 */
template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType bcast_type = BroadcastType::NONE, bool is_int_fpu_en = false>
inline void eltwise_unary_configure_mop(std::uint32_t rows_per_inst, std::uint32_t total_rows, const std::uint32_t num_faces, const std::uint32_t dst_format)
{
    // always move 32x32 tile, packed as 16x16x4

    if constexpr (type == DataCopyType::A2D)
    {
        std::uint32_t innerloop = (rows_per_inst == p_mova2d::MOV_1_ROW) ? total_rows : (total_rows >> 3);
        std::uint32_t outerloop = num_faces;

        if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) || (dst_format == to_underlying(DataFormat::UInt8)))
        {
            // use elwadd to handle unpacking data into src A as fp16, but dest is in fp32 mode OR to handle uint8 datums
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else if (is_fp32_dest_acc_en && (dst_format == to_underlying(DataFormat::UInt16)))
        {
            // Typecasting uint16 to 32bit data, need data to be written to lower 16 bits without modification
            // to be consumed by SFPU easily.
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
    else if constexpr (type == DataCopyType::B2D)
    {
        std::uint32_t addr_mod  = (rows_per_inst == p_movb2d::MOV_1_ROW) ? ADDR_MOD_0 : ADDR_MOD_2;
        std::uint32_t innerloop = (rows_per_inst == p_movb2d::MOV_1_ROW) ? total_rows : (total_rows >> 2);
        std::uint32_t outerloop = 4;
        auto broadcast_type     = p_movb2d::MOV_1_ROW; // No broadcast;

        if constexpr (bcast_type == BroadcastType::COL)
        {
            innerloop = 16 >> 3; // elwadd produces 8 rows per op
            // The mop only runs for 2 outer loops and mop is called twice for col broadcast
            outerloop = 2;
            // ELWADD with zeros will be used for non UInt16 case, since it moves 8 rows per cycle
            broadcast_type = p_elwise::SRCB_BCAST_COL;
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                innerloop      = 16 >> 2; // movb2d produces 4 rows per op
                broadcast_type = p_movb2d::MOV_4_ROWS_D0_BRCST;
            }
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            innerloop      = (total_rows >> 3);
            broadcast_type = p_movb2d::MOV_8_ROW_BRCST;
        }
        else if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            outerloop      = 1;
            innerloop      = num_faces * (total_rows >> 3);
            broadcast_type = p_elwise::SRCB_BCAST_ALL;
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                broadcast_type = p_movb2d::MOV_8_ROW_BRCST_D0_BRCST;
            }
        }

        if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0));
                tmp.program();
            }
            else
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0));
                tmp.program();
            }
        }
        else if constexpr (bcast_type == BroadcastType::COL)
        {
            if (dst_format ==
                to_underlying(DataFormat::UInt16)) // UInt16 case needs to use MOVB2D because for ELWADD FPU interprets some numbers as a float with exp 0
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
                tmp.set_end_op(TT_OP_SETRWC(0, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
                tmp.program();
            }
            else // ELWADD is used for non UInt16 case, since it moves 8 rows per cycle
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
                tmp.set_end_op(TT_OP_SETRWC(0, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
                tmp.program();
            }
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, rows_per_inst, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize the math thread (address mods and MOP) for an elementwise unary datacopy.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @param num_faces: Number of faces in the tile (must be 1, 2, or 4).
 * @param dst_format: Destination data format (DataFormat enum underlying value); 255 means unset.
 * @note On the unpack thread, pair with @ref _llk_unpack_A_init_ which feeds the tile.
 * @note @ref _llk_math_eltwise_unary_datacopy_ runs the configured op with matching template args.
 */
template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool is_int_fpu_en = false>
inline void _llk_math_eltwise_unary_datacopy_init_(const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    if constexpr (type == DataCopyType::A2D)
    {
        llk::san::math_operand_check(dst_format, llk::san::IGNORE);
    }
    else
    {
        llk::san::math_operand_check(llk::san::IGNORE, dst_format);
    }
    llk::san::operation_init<llk::san::Operation::EltwiseUnaryDatacopy>(type, src_b_bcast_type, dst_format);

    eltwise_unary_configure_addrmod<type, src_b_bcast_type>(dst_format);

    if constexpr (type == DataCopyType::A2D && src_b_bcast_type == BroadcastType::NONE)
    {
        eltwise_unary_configure_mop<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(p_mova2d::MOV_8_ROWS, 16, num_faces, dst_format);
    }
    else if constexpr (type == DataCopyType::B2D)
    {
        eltwise_unary_configure_mop<type, false, src_b_bcast_type>(p_movb2d::MOV_4_ROWS, 16, num_faces, dst_format);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Uninitialize after an elementwise unary datacopy.
 *
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Whether unpack wrote directly to dest.
 * @note Reverses @ref _llk_math_eltwise_unary_datacopy_init_; currently a no-op since all state is transient.
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_uninit_()
{
}
