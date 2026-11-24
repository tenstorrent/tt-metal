// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Temporary public LLK in Blaze pending upstream integration.
//
// A 2048-element list sorted as two 1024 halves on two cores, then combined into the stock topk_xl 2048 order.
//
// A 2048 list's sort position is b10..b0. In the stock 2048 order b10..b7 are a 32x32 tile's column bits 3..0, b6 the
// tile, b5 the face row, b4 the face column and b3..b0 the row bits 3..0. A local_sort<1024> half holds b9..b0 in the
// stock 1024 order (b9..b6 in the column bits); which half gives b10. A 1024 list keeps three rank bits in SFPU lanes
// but a 2048 list only two (next to b10), so no merge network over the lanes combines the halves. After a face
// transpose the lanes match and only (tile, row bits 3..0) differ, by a rotation of b10..b6. Moving a bit between the
// two row-in-LREG positions takes two SFPTRANSP passes with a store and load between: half_prep runs one per half on
// the core that sorted it, the combine body runs the other.
#pragma once

#include <cstdint>

#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

namespace ckernel::sfpu
{

// Out-of-line copies of stock helpers, which inlined at every call site overflow the split cores' kernel config buffer,
// and the stock local sort and rebuild built on them, which the half and quarter splits share.

// transpose_dest_face_32b on the 32-bit Dst face at the Dst counter, except the last MOVA2D advances the counter to the
// next face by ADDR_MOD_5 (+16 rows, from topk_xl_init) instead of ADDR_MOD_7.
inline void topk_xl_transpose_face_and_advance()
{
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 12);
    TTI_TRNSPSRCB;
    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, 12);
    TTI_TRNSPSRCB;
    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, 12);
    // lo16 written back with Fp32_enabled = 0 keeps the hi16 just stored.
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));
    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, 0);
    TTI_MOVA2D(1, 8, ADDR_MOD_5, p_mova2d::MOV_8_ROWS, 8);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// transpose_N_faces<num_faces, true, 256, false> from the math base, inside an open transpose cfg block. Unlike the
// stock helper, the moves wait for earlier SFPU stores and later SFPLOADs wait for the moves.
[[gnu::noinline]] inline void topk_xl_transpose_faces(const std::uint32_t num_faces)
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#pragma GCC unroll 1
    for (std::uint32_t f = 0; f < num_faces; ++f)
    {
        topk_xl_transpose_face_and_advance();
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

// With TOPK_XL_FUSED_MACRO the canonical block's SFPLOADMACRO (row_scale_factor >= 2) needs templates, sequences and
// Misc that only the stock entry points program; Blaze entry points to it run this first. Clobbers LREG0.
template <int row_scale_factor>
inline void topk_xl_canonical_macro_configure()
{
#if TOPK_XL_FUSED_MACRO
    if constexpr (row_scale_factor >= 2)
    {
        topk_xl_fused_macro::configure();
    }
#endif
}

template <int row_scale_factor>
[[gnu::noinline]] inline void topk_xl_canonical_block(const bool dir)
{
    canonical_big_block_with_replay<row_scale_factor>(dir);
}

// The canonical block on both column parities of the tile at tile_offset; leaves the Dst write offset there.
template <int row_scale_factor>
inline void topk_xl_canonical_columns(const std::uint32_t tile_offset, const bool dir)
{
    for (int col = 0; col < 2; col++)
    {
        topk_xl_canonical_block<row_scale_factor>(dir);
        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

// One stride-2 build phase: `n_iters` passes of load, `sort` and store. The first pass records the load and the store
// in replay slots [0, 16) as it runs, so `sort` must leave them alone.
template <int n_iters, typename Sort>
inline void topk_xl_stride2_phase(Sort sort)
{
    load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<2>(); });
    sort();
    load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<2, 16>(); });
    for (int i = 1; i < n_iters; i++)
    {
        lltt::replay(0, 8);
        sort();
        lltt::replay(8, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// _topk_xl_local_sort_generic_<K> for K = 512 or 1024 as built under TOPK_XL_BLAZE_COMPAT, on the out-of-line helpers
// above: the same phases, so the same list.
template <std::uint32_t K>
inline void _topk_xl_local_sort_ool_(const std::uint32_t dst_index, const bool ascending)
{
    static_assert(K == 512 || K == 1024, "the out-of-line sort covers K = 512 and 1024");
    constexpr int row_scale_factor    = K == 512 ? 1 : 2;
    constexpr int n_iters_stride2     = 2 * row_scale_factor;
    constexpr std::uint32_t num_faces = 2 * row_scale_factor;
    topk_xl_canonical_macro_configure<row_scale_factor>();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    bool dir                        = ascending;
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    for (int col = 0; col < 2; col++)
    {
        // Length 32.
        for (int i = 0; i < row_scale_factor; i++)
        {
            load16_rows_x2<16>();
            bitonic_sort_len_2();
            bitonic_sort_len_4(ascending);
            if (i == 0)
            {
                load_replay_buf<Exec>(
                    0,
                    32,
                    [ascending]
                    {
                        bitonic_sort_len_8(ascending);
                        bitonic_sort_len_16();
                    });
            }
            else
            {
                lltt::replay(0, 32);
            }
            bitonic_sort_len_32(dir);
            store16_rows_x2<16, 32>();
            if constexpr (row_scale_factor > 1)
            {
                dir = !dir;
            }
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

        if constexpr (row_scale_factor > 1)
        {
            // Length 64.
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 16>();
            load16_rows_x2<32>();
            bitonic_sort_len_k(dir);
            store16_rows_x2<32, 48>();
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

            // Stride-16 sort_32, both iterations in one direction.
            load_replay_buf<Exec>(0, 8, [] { load16_rows_x2<16>(); });
            bitonic_sort_len_32(dir);
            load_replay_buf<Exec>(8, 8, [] { store16_rows_x2<16, 32>(); });
            lltt::replay(0, 8);
            bitonic_sort_len_32(dir);
            lltt::replay(8, 8);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
        dir = !dir;
    }

    // Length K/8: SFPCONFIG flips the swap direction of every other lane.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x4444, 0xF, 8);
    topk_xl_stride2_phase<n_iters_stride2>([dir] { bitonic_sort_len_k(dir); });
    topk_xl_canonical_columns<row_scale_factor>(tile_offset, dir);
    TTI_SFPCONFIG(0x0000, 0xF, 1);

    enter_transpose_cfg_block();

    // Length K/4.
    topk_xl_transpose_faces(num_faces);
    topk_xl_stride2_phase<n_iters_stride2>(
        [dir]
        {
            TTI_SFPTRANSP(0, 0, 0, 0);
            bitonic_sort_len_4(dir);
        });
    topk_xl_transpose_faces(num_faces);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5050, 0xF, 8);
    topk_xl_canonical_columns<row_scale_factor>(tile_offset, dir);
    TTI_SFPCONFIG(0x0000, 0xF, 1);

    // Length K/2.
    topk_xl_transpose_faces(num_faces);
    topk_xl_stride2_phase<n_iters_stride2>([dir] { bitonic_sort_len_8(dir); });
    topk_xl_transpose_faces(num_faces);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x5500, 0xF, 8);
    topk_xl_canonical_columns<row_scale_factor>(tile_offset, dir);
    TTI_SFPCONFIG(0x0000, 0xF, 1);

    // Length K.
    topk_xl_transpose_faces(num_faces);
    topk_xl_stride2_phase<n_iters_stride2>([dir] { bitonic_sort_len_16_alt(dir); });
    topk_xl_transpose_faces(num_faces);
    leave_transpose_cfg_block();
    topk_xl_canonical_columns<row_scale_factor>(tile_offset, dir);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// _topk_xl_rebuild_generic_<K, true> for K = 512 or 1024 on the out-of-line helpers, from the math base `row_base`. The
// caller programs the build MOP (topk_rebuild_build2048_mop_config) before its rebuilds and restores the merge MOP
// (topk_mop_config<true>) and releases SrcA and SrcB (CLR_AB) after them, so several rebuilds share one of each; the
// release has to wait for the caller's last transpose.
template <std::uint32_t K>
[[gnu::noinline]] inline void topk_xl_rebuild_at(const std::uint32_t row_base, const bool dir)
{
    static_assert(K == 512 || K == 1024, "the out-of-line rebuild covers K = 512 and 1024");
    constexpr int row_scale_factor    = K == 512 ? 1 : 2;
    constexpr std::uint32_t num_faces = 2 * row_scale_factor;
    topk_xl_canonical_macro_configure<row_scale_factor>();
    set_dst_write_addr_offset(row_base);
    enter_transpose_cfg_block();
    topk_xl_transpose_faces(num_faces);
    load_replay_buf<Exec>(
        0,
        16,
        [dir]
        {
            load16_rows_x2<2>();
            bitonic_sort_len_16_alt_swaps<true>(dir);
        });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(16, 8, [dir] { bitonic_sort_len_16_alt_swaps<true>(dir); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(24, 8, [] { store16_rows_x2<2, 16>(); });
    ckernel_unpack_template::run(2 * row_scale_factor - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_xl_transpose_faces(num_faces);
    leave_transpose_cfg_block();
    topk_xl_canonical_columns<row_scale_factor>(row_base, dir);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <std::uint32_t K>
inline void _topk_xl_add_lsb_indices_base_rt_(const std::uint32_t base)
{
    static_assert(K == 512 || K == 1024, "stamp supports half/full tiles");
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);
    if (__builtin_constant_p(base))
    {
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, base);
    }
    else
    {
        TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, base);
    }
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(257, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);

    load_replay_buf<Exec>(
        0,
        16,
        []
        {
            TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
            TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
            TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 2);
            TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
            TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);
            TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
            TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);
            TTI_SFPIADD(64, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(64, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(64, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPIADD(64, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
            TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 16 + 0);
            TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_6, 16 + 2);
        });
    for (int i = 1; i < 4; i++)
    {
        lltt::replay(0, 16);
    }
    if constexpr (K == 1024)
    {
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Skips the face pair the first four bodies did not cover (+16 rows).
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_4, 0);
        for (int i = 0; i < 4; i++)
        {
            lltt::replay(0, 16);
        }
    }
}

template <std::uint32_t core_id, std::uint32_t base>
inline void _topk_xl_add_lsb_indices_base_()
{
    static_assert(((core_id << 11) | base) < (1u << 16) && (base & 1023u) == 0, "stamp must fit the u16 index");
    _topk_xl_add_lsb_indices_base_rt_<1024>((core_id << 11) | base);
}

// Pass one of the two: LREG (g, l1, l0) <-> face rows (+4 g, +8 l1, +16 l0),
// in place. SFPTRANSP swaps (l1, l0) with the row-in-LREG pair.
inline void topk_xl_half_pass_loads()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 24);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 20);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 28);
}

inline void topk_xl_half_pass_stores()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 24);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 20);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 28);
}

// local_sort<1024> tile -> faces transposed, with rows R3 = b7, R2 = b8, R1 = b9, R0 = b4 and the face column bit = b6.
inline void _topk_xl_half_prep_(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];
    enter_transpose_cfg_block();
    topk_xl_transpose_faces(4);
    leave_transpose_cfg_block();
    for (std::uint32_t it = 0; it < 4; ++it)
    {
        // Face row (it >> 1) and column parity (it & 1); the face column and row groups are LREG bits.
        set_dst_write_addr_offset(tile_offset + 32 * (it >> 1) + 2 * (it & 1));
        if (it == 0)
        {
            load_replay_buf<Exec>(0, 8, [] { topk_xl_half_pass_loads(); });
            TTI_SFPTRANSP(0, 0, 0, 0);
            load_replay_buf<Exec>(8, 8, [] { topk_xl_half_pass_stores(); });
        }
        else
        {
            lltt::replay(0, 8);
            TTI_SFPTRANSP(0, 0, 0, 0);
            lltt::replay(8, 8);
        }
    }
    set_dst_write_addr_offset(tile_offset);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// Combine body loads: LREG (g, l1, l0) <- half g, rows +4 l1 (b8) and +8 l0 (b7).
// `in_rows` is the first input row relative to the output base of the iteration.
template <int in_rows>
inline void topk_xl_combine_loads()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 8);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 4);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 12);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 64);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 72);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 68);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, in_rows + 76);
}

// Compare-exchange of four LREG pairs (a, b); `ascending` puts the smaller word in a. ALL_ROWS_MAX puts the max in VC.
template <int a0, int b0, int a1, int b1, int a2, int b2, int a3, int b3>
inline void topk_xl_swap4(const bool ascending)
{
    if (ascending)
    {
        TTI_SFPSWAP(0, b0, a0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, b1, a1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, b2, a2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, b3, a3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        TTI_SFPSWAP(0, a0, b0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, a1, b1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, a2, b2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, a3, b3, p_sfpswap::ALL_ROWS_MAX);
    }
}

// half_prep'd halves sorted in opposite directions at dst_out + 2 (b10 = 0) and dst_out + 3 (b10 = 1) -> one
// `ascending` list in the stock 2048 order at dst_out and dst_out + 1. Each iteration reads both halves and writes one
// output tile, so output and input cannot overlap.
inline void _topk_xl_combine_halves_(const std::uint32_t dst_out, const bool ascending)
{
    const std::uint32_t tile_out = dst_out << DstTileSizeLog2[DstTileShape::Tile32x32];
    topk_xl_canonical_macro_configure<4>();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (std::uint32_t z = 0; z < 2; ++z)
    {
        for (std::uint32_t it = 0; it < 4; ++it)
        {
            // Iteration (z = b6, face row, parity) reads face (row, z) of both halves
            // and writes output tile z, both face columns.
            set_dst_write_addr_offset(tile_out + 64 * z + 32 * (it >> 1) + 2 * (it & 1));
            if (it == 0)
            {
                // Input face (row, z) sits 128 + 16 z rows past output tile 0's face (row);
                // relative to output tile z that is 128 - 48 z.
                if (z == 0)
                {
                    load_replay_buf<Exec>(0, 8, [] { topk_xl_combine_loads<128>(); });
                }
                else
                {
                    load_replay_buf<Exec>(0, 8, [] { topk_xl_combine_loads<80>(); });
                }
            }
            else
            {
                lltt::replay(0, 8);
            }
            if (z == 0 && it == 0)
            {
                // b10: LREG g = 0 against g = 1.
                load_replay_buf<Exec>(
                    8,
                    4,
                    [ascending]
                    {
                        topk_xl_swap4<p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG7>(
                            ascending);
                    });
                TTI_SFPTRANSP(0, 0, 0, 0);
                // b9, moved into l1 by the transpose.
                load_replay_buf<Exec>(
                    12,
                    4,
                    [ascending]
                    {
                        topk_xl_swap4<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG7>(
                            ascending);
                    });
                TTI_SFPTRANSP(0, 0, 0, 0);
                // b8 in l1, then b7 in l0.
                load_replay_buf<Exec>(16, 8, [ascending] { bitonic_sort_len_16_alt_swaps<true>(ascending); });
                TTI_SFPTRANSP(0, 0, 0, 0);
                // LREG (g, l1, l0) = (b10, b9, b4) -> rows +8 g, +4 l1, face column +16 l0;
                // the rows inside an LREG now hold (b8, b7).
                load_replay_buf<Exec>(
                    24,
                    8,
                    []
                    {
                        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 16);
                        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
                        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 20);
                        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
                        TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 24);
                        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
                        TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, 28);
                    });
            }
            else
            {
                lltt::replay(8, 4);
                TTI_SFPTRANSP(0, 0, 0, 0);
                lltt::replay(12, 4);
                TTI_SFPTRANSP(0, 0, 0, 0);
                lltt::replay(16, 8);
                TTI_SFPTRANSP(0, 0, 0, 0);
                lltt::replay(24, 8);
            }
        }
    }
    set_dst_write_addr_offset(tile_out);
    // Back to the stock layout, then the tail of rebuild<2048>: b6 (tile) down to b0.
    enter_transpose_cfg_block();
    topk_xl_transpose_faces(8);
    leave_transpose_cfg_block();
    topk_xl_canonical_columns<4>(tile_out, ascending);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

} // namespace ckernel::sfpu
