// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Temporary public LLK in Blaze pending upstream integration.
//
// A 1024-element list sorted as two 512 quarters on two cores, then merged into the layout topk_xl_half_prep leaves,
// ready for topk_xl_combine_halves.
//
// local_sort<512> keeps its list in faces 0 and 1 of a tile, rank bits q8..q0 = column bits 3..0, face column, row bits
// 3..0. With quarter A in tile `in_tile` and quarter B in `in_tile` + 1, sorted in opposite directions, the 1024 list's
// rank bit b9 is the tile: one compare-exchange across the two tiles and a rebuild<512> of each tile sort it, as the
// last phase of a bitonic sort would. The stock 1024 order puts b9 in a column bit (an SFPU lane), so the quarters
// cannot take the stock merge. After a face transpose of the four faces the lanes hold b3..b1 as half_prep leaves them;
// one SFPTRANSP pass moves b9 and b4 into the row-in-LREG bits and the stores place the rest.
#pragma once

#include <cstdint>

#include "sfpu/experimental/ckernel_sfpu_topk_xl_halves.h"

namespace ckernel::sfpu
{

// Compare-exchange of faces 0 and 1 of the tile at the math base with the same faces 64 rows on; `ascending` leaves
// the smaller word in the first tile. Iteration (face column, parity) covers that face's 16 rows in both tiles.
inline void topk_xl_quarter_exchange(const std::uint32_t row_base, const bool ascending)
{
    for (std::uint32_t it = 0; it < 4; ++it)
    {
        set_dst_write_addr_offset(row_base + 16 * (it >> 1) + 2 * (it & 1));
        if (it == 0)
        {
            load_replay_buf<Exec>(
                0,
                16,
                [ascending]
                {
                    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
                    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
                    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
                    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 0);
                    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 4);
                    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 8);
                    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 12);
                    topk_xl_swap4<p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG7>(
                        ascending);
                    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
                    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
                    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
                });
            load_replay_buf<Exec>(
                16,
                4,
                []
                {
                    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 0);
                    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 4);
                    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 8);
                    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 64 + 12);
                });
        }
        else
        {
            lltt::replay(0, 16);
            lltt::replay(16, 4);
        }
    }
}

// Quarter-pair loads for iteration b8 (row bit 3 of the transposed quarters), relative to the iteration base: LREG
// (g, l1, l0) <- rows +4 g (b7), tile +64 l1 (b9), face column +16 l0 (b4); the LREG's rows hold (b6, b5).
template <int in_rows>
inline void topk_xl_quarter_loads()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 16);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 64);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 80);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 4);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 20);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 68);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 84);
}

// After the SFPTRANSP LREG (g, l1, l0) holds (b7, b6, b5) with (b9, b4) in its rows: rows +8 g, face column +16 l1,
// face row +32 l0 of the output tile.
template <int out_rows>
inline void topk_xl_quarter_stores()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 32);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 16);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 48);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 8);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 40);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 24);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, out_rows + 56);
}

// Faces 0 and 1 of tiles `in_tile` and `in_tile` + 1, faces transposed -> the half_prep layout at tile `out_tile`,
// which overlaps neither. The input's row bit 3 (b8) lands in the output's row bit 2: the iteration base advances 4
// rows for b8 and the b8 = 1 loads add the other 4.
template <std::uint32_t in_tile, std::uint32_t out_tile>
inline void topk_xl_quarter_pass()
{
    static_assert(out_tile < in_tile || out_tile > in_tile + 1, "the pass reads and writes different tiles");
    constexpr std::uint32_t lower = in_tile < out_tile ? in_tile : out_tile;
    constexpr int in_rows         = 64 * static_cast<int>(in_tile - lower);
    constexpr int out_rows        = 64 * static_cast<int>(out_tile - lower);
    // lltt::replay takes an immediate start, which `it` picks, so the loop unrolls whatever GCC's size heuristics say.
#pragma GCC unroll 4
    for (std::uint32_t it = 0; it < 4; ++it)
    {
        // Iteration (b8, parity).
        set_dst_write_addr_offset(64 * lower + 4 * (it >> 1) + 2 * (it & 1));
        if (it == 0)
        {
            load_replay_buf<Exec>(0, 8, [] { topk_xl_quarter_loads<in_rows>(); });
            TTI_SFPTRANSP(0, 0, 0, 0);
            load_replay_buf<Exec>(8, 8, [] { topk_xl_quarter_stores<out_rows>(); });
        }
        else if (it == 2)
        {
            load_replay_buf<Exec>(16, 8, [] { topk_xl_quarter_loads<in_rows + 4>(); });
            TTI_SFPTRANSP(0, 0, 0, 0);
            lltt::replay(8, 8);
        }
        else if (it == 1)
        {
            lltt::replay(0, 8);
            TTI_SFPTRANSP(0, 0, 0, 0);
            lltt::replay(8, 8);
        }
        else
        {
            // A replay's start must be a constant; with gathering on the loop is not unrolled to fold it.
            lltt::replay(16, 8);
            TTI_SFPTRANSP(0, 0, 0, 0);
            lltt::replay(8, 8);
        }
    }
}

template <std::uint32_t in_tile, std::uint32_t out_tile>
inline void _topk_xl_combine_quarters_(const bool ascending)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_xl_quarter_exchange(64 * in_tile, ascending);
    topk_rebuild_build2048_mop_config();
    topk_xl_rebuild_at<512>(64 * in_tile, ascending);
    topk_xl_rebuild_at<512>(64 * (in_tile + 1), ascending);
    topk_mop_config<true>();
    enter_transpose_cfg_block();
    set_dst_write_addr_offset(64 * in_tile);
    topk_xl_transpose_faces(2);
    set_dst_write_addr_offset(64 * (in_tile + 1));
    topk_xl_transpose_faces(2);
    leave_transpose_cfg_block();
    topk_xl_quarter_pass<in_tile, out_tile>();
    set_dst_write_addr_offset(64 * out_tile);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

} // namespace ckernel::sfpu
