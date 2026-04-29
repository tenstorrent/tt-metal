// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_sfpu_load_config.h"

namespace ckernel
{
namespace sfpu
{

// Index-tracking is not needed since values and indices are fused into a single 32-bit container
inline void _topk_xl_init_()
{
}

inline void set_dst_write_addr_offset(std::uint32_t addr)
{
    std::uint32_t dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

template <int group_2_offset = 16>
inline void load16_rows_x2()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, 12);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 12);
}

template <int group_2_offset = 16, bool inc_dst_addr_32 = false>
inline void store16_rows_x2()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, 12);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 4);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 8);
    if constexpr (inc_dst_addr_32)
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP32, ADDR_MOD_6, group_2_offset + 12);
    }
    else
    {
        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP32, ADDR_MOD_7, group_2_offset + 12);
    }
}

inline void bitonic_sort_len_2()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    // TTI_SFPTRANSP(0, 0, 0, 0); // skip since fused with len_4
}

inline void bitonic_sort_len_4()
{
    // TTI_SFPTRANSP(0, 0, 0, 0); // skip since fused with len_2

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_sort_len_8()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_sort_len_16()
{
    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

inline void bitonic_sort_len_32(bool ascending)
{
    if (ascending)
    {
        // Step 5
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    else
    {
        // Step 5
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        // Step 3
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

inline void bitonic_sort_len_k(bool ascending)
{
    if (ascending)
    {
        // Step 6
        TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        // Step 6
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    }
}

// Inline helper: transpose one 16x16 face of 32-bit data in Dst using SrcA format switching.
// Performs two passes (hi16 then lo16) through SrcB[16:31] with TRNSPSRCB.
// On entry Fp32_enabled must be 1 (for Dst32b reads). On exit Fp32_enabled is 1.
template <int dst_offset>
inline void transpose_dest_face_32b()
{
    // Transpose low 16 bits and backup it in SrcA
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);

    // --- Pass 1: hi16 ---
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 12);

    // lo16 writeback: Fp32_enabled=0 + SrcA=Float32 -> writes lo16, preserves hi16
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 0);
    TTI_MOVA2D(1, 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 8);

    // Restore Fp32_enabled=1 for subsequent 32-bit Dst access
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

inline void transpose_8_faces()
{
    // Disable implied SrcA format inference so manual SrcA format switches take effect.
    // On Blackhole the ALU format is normally inferred; we must disable this for
    // the SrcA format switching approach to control MOVD2B/MOVB2D behavior.
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    // Disable zero flag to prevent mantissa flushing when exponent bits are 0.
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);

    transpose_dest_face_32b<0>();
    transpose_dest_face_32b<16>();
    transpose_dest_face_32b<32>();
    transpose_dest_face_32b<48>();
    transpose_dest_face_32b<64>();
    transpose_dest_face_32b<80>();
    transpose_dest_face_32b<96>();
    transpose_dest_face_32b<112>();

    // Restore
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
}

// clang-format off
/**
 * local sort for two consecutive DST tiles, K=2048 (packed u32: f16 key | u16 index)
 * builds bitonic sequences in columns of DST
 */
// clang-format on

template <bool APPROXIMATION_MODE>
inline void _topk_xl_local_sort_(const std::uint32_t dst_index, const bool ascending)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bool dir                            = ascending;
    const std::uint32_t tile_offset     = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    constexpr int consecutive_32_offset = 16;

    for (int col = 0; col < 2; col++)
    {
        // ------------------------------------------------------------
        // build bitonic sequences of len=32
        // ------------------------------------------------------------
        // each loop processes 32 / 128 rows
        for (int i = 0; i < 4; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_2();
            bitonic_sort_len_4();
            bitonic_sort_len_8();
            bitonic_sort_len_16();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            dir = !dir;
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ------------------------------------------------------------
        // build bitonic sequences of len=64
        // ------------------------------------------------------------
        // each loop processes 64 / 128 rows
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, true>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            dir = !dir;
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            dir = !dir;
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // ------------------------------------------------------------
        // build bitonic sequences of len=128
        // ------------------------------------------------------------
        // each loop processes 32 / 128 rows
        for (int i = 0; i < 4; i++)
        {
            load16_rows_x2<64>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<64, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        // each loop processes 64 / 128 rows
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, true>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // set dst addr to work on odd columns, then back to even columns
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        // alternate directions for even/odd columns
        dir = !dir;
    }

    // ------------------------------------------------------------
    // build bitonic sequences of len=256
    // ------------------------------------------------------------
    // set every SFPU instance to alternate SWAP direction
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x4444, 0xF, 8);
    // each loop processes 16 / 128 rows
    for (int i = 0; i < 8; i++)
    {
        load16_rows_x2<2>();
        bitonic_sort_len_k(dir);
        store16_rows_x2<2, false>();
        TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // even/odd columns
    for (int col = 0; col < 2; col++)
    {
        // each loop processes 32 / 128 rows
        for (int i = 0; i < 4; i++)
        {
            load16_rows_x2<64>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<64, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        // each loop processes 64 / 128 rows
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, true>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // set dst addr to work on odd columns, then back to even columns
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // clear sfpu config

    // ------------------------------------------------------------
    // build bitonic sequences of len=512
    // ------------------------------------------------------------
    // transpose DST faces in order to do bitonic sort for 512+
    // TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    transpose_8_faces();
    // each loop processes 16 / 128 rows
    for (int i = 0; i < 8; i++)
    {
        load16_rows_x2<2>();
        TTI_SFPTRANSP(0, 0, 0, 0);
        bitonic_sort_len_4();
        store16_rows_x2<2, false>();
        TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // transpose DST faces back
    transpose_8_faces();
    // set every 2 SFPU instances to alternate SWAP direction
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5050, 0xF, 8);
    for (int col = 0; col < 2; col++)
    {
        // each loop processes 32 / 128 rows
        for (int i = 0; i < 4; i++)
        {
            load16_rows_x2<64>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<64, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        // each loop processes 64 / 128 rows
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, false>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, true>();
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
            TTI_INCRWC(0, 8, 0, 0); // increment dst address by 8
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (int i = 0; i < 2; i++)
        {
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
            load16_rows_x2<consecutive_32_offset>();
            bitonic_sort_len_32(dir);
            store16_rows_x2<consecutive_32_offset, true>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        // set dst addr to work on odd columns, then back to even columns
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
    TTI_SFPCONFIG(0x0000, 0xF, 1); // clear sfpu config

    // clear srcA and srcB valids - needed for DST transpose
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <bool APPROXIMATION_MODE>
inline void _topk_xl_merge_(const std::uint32_t dst_index)
{
    // TODO: merge stage for two length-2048 subsequences at idst
}

template <bool APPROXIMATION_MODE>
inline void _topk_xl_rebuild_(const std::uint32_t dst_index, const bool ascending)
{
    // TODO: rebuild stage for length-2048 subsequence at idst
}

} // namespace sfpu
} // namespace ckernel
