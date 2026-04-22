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

using namespace ckernel;

// local function declarations
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format);

template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_(
    const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // For 32bit data, each half of DEST can take 16 tiles. Since dest offset is returned as if 16bit data are used, we need to
    // adjust it to offset in faces for 32bit data.
    if (unpack_to_dest && is_32bit_input(src_format, dst_format))
    {
        math_unpack_to_dest_math_ready();
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        math::math_unpack_to_dest_tile_ready();

        // Due to bug in Blackhole Tensix (more details in budabackend/#2730) when an event with side effect of clearing DEST zero flags
        // (such as Unpack-to-dest or RISC-to-dest) and a ZEROACC instruction from packer occur in the same cycle,
        // zero flags clearing is dropped.
        // To mitigate that, we issue additional zero flag clear instruction immediately after unpack tile to dest is done.
        // RISC-to-dest event is not currently used.

        const std::uint32_t dst_format_masked = masked_data_format(dst_format);
        const int clear_fp32                  = static_cast<int>(
            dst_format_masked == (std::uint32_t)DataFormat::Float32 || dst_format_masked == (std::uint32_t)DataFormat::Int32 ||
            dst_format_masked == (std::uint32_t)DataFormat::UInt32);
        const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < num_faces; i++)
        {
            // Clears zero flags in DEST for one face.
            TT_ZEROACC(p_zeroacc::CLR_16, clear_fp32, 1 /*clear zero flags*/, ADDR_MOD_3, get_dest_index_in_faces(local_tile, i));
        }

        if constexpr (src_b_bcast_type == BroadcastType::ROW)
        {
            // workarounds for hi/lo D2B/B2D (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);           // Set Fp32 ALU mode to 0 because of a bug
            TTI_SETDVALID(0b10);

            // Broadcast 32-bit data in 2 parts (hi16 then lo16).
            // MOVB2D(dest_32b_lo=0) zeroes the lo16 physical slot in DEST as a side effect.
            // To preserve lo16: read hi16 into B[SRC_ROW16_OFFSET] and lo16 into B[SRC_ZERO_OFFSET]
            // (separate B addresses), then broadcast both. Process one source row at a time
            // so B data is consumed before being overwritten for the next source row.
            // Source row 0 → faces 0 (rows 0-15) and 2 (rows 32-47)
            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);

            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 0);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 8);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 32);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 40);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 8);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 32);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 40);

            // Source row 16 → faces 1 (rows 16-31) and 3 (rows 48-63)
            // Row 16 is untouched by the above broadcasts so hi16/lo16 are intact.
            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 16);
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 16);

            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 16);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 24);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 48);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 56);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 16);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 24);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 48);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 56);

            // restore fp32 mode
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
        {
            // workarounds for hi/lo D2B/B2D (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);           // Set Fp32 ALU mode to 0 because of a bug
            TTI_SETDVALID(0b10);

            // move back to B and broadcast in 2 parts, first hi16 bits then lo16 bits

            // Read hi16 into B[SRC_ROW16_OFFSET] and lo16 into B[SRC_ZERO_OFFSET] before any writes.
            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);

            // broadcast hi bits B2D
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 0);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 8);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 16);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 24);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 32);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 40);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 48);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 56);

            // broadcast lo bits B2D
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 8);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 16);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 24);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 32);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 40);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 48);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 56);

            // restore fp32 mode
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // workarounds for hi/lo D2B/B2D (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);           // Set Fp32 ALU mode to 0 because of a bug
            TTI_SETDVALID(0b10);

            // Use SRC_ROW16_OFFSET for hi16 and SRC_ZERO_OFFSET for lo16 so both can
            // coexist in B. Per offset: read both hi16+lo16, then broadcast both.
#pragma GCC unroll 2
            for (int offset = 0; offset < 2; ++offset)
            {
                // read hi16 into B[SRC_ROW16_OFFSET+{0,4,8,12}]
                TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 0);
                TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 4);
                TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 8);
                TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 12);
                // read lo16 into B[SRC_ZERO_OFFSET+{0,4,8,12}] (separate B addresses)
                TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 0);
                TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 4);
                TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 8);
                TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 12);

                // broadcast hi16 from B[SRC_ROW16_OFFSET+...]
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 0);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 4);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 8);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 12);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 16);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 20);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 24);
                TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 28);
                // broadcast lo16 from B[SRC_ZERO_OFFSET+...]
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 0);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 4);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 8);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 12);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 16);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 20);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 24);
                TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ZERO_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 28);
            }

            // restore fp32 mode
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
            TTI_CLEARDVALID(0b10, 0);
        }
    }
    else
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (type == A2D)
        {
            ckernel_template::run();
        }
        else if constexpr (type == B2D)
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

        math::clear_dst_reg_addr();
    }
}

template <DataCopyType type, BroadcastType bcast_type = BroadcastType::NONE>
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format)
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    // Use srcA for data movement
    if constexpr (type == A2D) // Overrides BCAST_TYPE when type is A2D
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
}

template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType bcast_type = BroadcastType::NONE, bool tilize = false, bool is_int_fpu_en = false>
inline void eltwise_unary_configure_mop(std::uint32_t rows_per_inst, std::uint32_t total_rows, const std::uint32_t num_faces, const std::uint32_t dst_format)
{
    // always move 32x32 tile, packed as 16x16x4

    if constexpr (type == A2D)
    {
        std::uint32_t innerloop = (rows_per_inst == p_mova2d::MOV_1_ROW) ? total_rows : (total_rows >> 3);
        std::uint32_t outerloop = tilize ? 1 : num_faces;

        if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) || (dst_format == to_underlying(DataFormat::UInt8)))
        {
            // use elwadd to handle unpacking data into src A as fp16, but dest is in fp32 mode OR to handle uint8 datums
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
    else if constexpr (type == B2D)
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
            broadcast_type = p_movb2d::MOV_8_ROW_BRCST_D0_BRCST;
        }

        if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0));
            tmp.program();
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

// If using 8bit datums for unpack src, skip_bh_tilize_workaround should be set to true because blackhole tilize workaround is not done for 8 bit datum formats.
template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool tilize = false, bool is_int_fpu_en = false>
inline void _llk_math_eltwise_unary_datacopy_init_(
    const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255, const bool skip_bh_tilize_workaround = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    eltwise_unary_configure_addrmod<type, src_b_bcast_type>(dst_format);

    if constexpr (type == A2D && src_b_bcast_type == BroadcastType::NONE)
    {
        const std::uint32_t num_rows = (tilize && !skip_bh_tilize_workaround) ? 64 : 16;

        if (skip_bh_tilize_workaround)
        {
            eltwise_unary_configure_mop<type, is_fp32_dest_acc_en, src_b_bcast_type, false /* tilize */, is_int_fpu_en>(
                p_mova2d::MOV_8_ROWS, 16, num_faces, dst_format);
        }
        else
        {
            eltwise_unary_configure_mop<type, is_fp32_dest_acc_en, src_b_bcast_type, tilize, is_int_fpu_en>(
                p_mova2d::MOV_8_ROWS, num_rows, num_faces, dst_format);
        }
    }
    else if constexpr (type == B2D)
    {
        eltwise_unary_configure_mop<type, false, src_b_bcast_type>(p_movb2d::MOV_4_ROWS, 16, num_faces, dst_format);
    }

    // Workaround for HW bug (budabackend#1948): tilize with UInt32/Int32 needs debug feature bit 11 disabled
    if constexpr (tilize)
    {
        if ((dst_format == static_cast<std::uint32_t>(DataFormat::UInt32)) || (dst_format == static_cast<std::uint32_t>(DataFormat::Int32)))
        {
            _llk_math_dbg_feature_disable_();
        }
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_uninit_()
{
    // clear debug feature disable
    if constexpr (src_b_bcast_type != BroadcastType::NONE && unpack_to_dest)
    {
        _llk_math_dbg_feature_enable_();
    }
}

// ============================================================================
// BH Fast-Tilize Math
//
// Modified A2D datacopy for fast-tilize's SrcA layout (8 data rows + 8 gap rows).
// MOVA2D MOV_8_ROWS with srca_incr=16, dest_incr=16 copies data rows while
// preserving the gap layout. 4 MOVA2D per dvalid = 32 data rows.
//
// Accumulates 8 dvalids into one DEST half-bank (512 rows for bf16):
//   dvalid 0: DEST rows 0-7, 16-23, 32-39, 48-55
//   dvalid 1: DEST rows 64-71, 80-87, 96-103, 112-119
//   ...
//   dvalid 7: DEST rows 448-455, 464-471, 480-487, 496-503
// Total: 32 data groups × 8 rows × 16 cols = 4096 datums = 4 tiles.
// Pack then reads 4 tiles from the DEST half using DST_ACCESS_STRIDED_MODE.
// ============================================================================

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_fast_tilize_init_([[maybe_unused]] const std::uint32_t unpack_dst_format, [[maybe_unused]] const std::uint32_t unit_dim)
{
    // DEST remap (remap_addrs + swizzle_32b) is enabled here to mirror
    // pack_untilize_dest_init's BH workaround (see tt-metal#17132 / tt-llk#989).
    // Kernels that call fast_tilize_init from inside a loop pay the tensix_sync
    // cost per iteration; systematic hoist is tracked in the same issues.
    // No uninit — leaving the bits set is benign for subsequent ops.
    _llk_math_reconfig_remap_(true);

    // Compat fp32-dest: MOVA2D does not correctly handle 32-bit DEST rows (BH HW quirk,
    // same as WH). Temporarily clear Fp32_enabled so MOVA2D treats DEST as 16-bit.
    // Restored in uninit. This is safe for bf16/fp16 outputs where 32-bit DEST is only
    // used for accumulation precision, not for the data path through fast-tilize.
    if constexpr (is_fp32_dest_acc_en)
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }

    // Enable SrcA bank return in SETRWC(CLR_A). Without this, SETRWC flips the
    // bank pointer but does NOT return the bank to Unpackers, causing a livelock
    // when the Unpacker runs out of free banks.
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    // srca incr=8 (contiguous SrcA), dest incr=16 (8 data + 8 gap in DEST for pack stride-16)
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }
        .set(ADDR_MOD_2);

    // outerloop=4 (4 dvalids per unit), innerloop=8 (8 MOVA2D per dvalid).
    // SETRWC in end_op clears/flips SrcA between dvalids; DEST RWC keeps
    // advancing across outer loops. Single run() per unit.
    ckernel_template tmp(4, 8, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_A));
    tmp.program();
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t unpack_dst_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    const std::uint32_t num_units,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    for (std::uint32_t u = 0; u < num_units; u++)
    {
        // Use SrcRegs (not DestReg) — DestReg sends a hardware mailbox to the unpack
        // thread, but fast-tilize unpack uses counter-based addressing and never reads
        // the mailbox. After ~4 unread writes the mailbox FIFO fills and math RISC-V
        // deadlocks. SrcRegs uses TT_SETC16 to set DEST offset directly.
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        // 1 MOP = 4 dvalids × 8 MOVA2D + 4 SETRWC(CLR_A)
        ckernel_template::run();

        math::clear_dst_reg_addr();
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_tilize_uninit_([[maybe_unused]] const std::uint32_t unpack_dst_format)
{
    // Restore fp32 dest mode if it was cleared in init
    if constexpr (is_fp32_dest_acc_en)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
    }

    // DEST remap is cleared by pack uninit.

    // Restore standard addr_mod for A2D
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}
