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
