// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tensix_transpose experiment, compute. Unit = kwu w-tiles of cb_in, w-tile wt = 32 slots
// (rows w) of Ct * 32 values (columns c). For each c-tile ct, tile (wt, ct) of the RM block is
// tilized, transposed to [c][w] and lands in DEST slot wt; pack_untilize_dest<kwu> then writes the
// kwu DEST tiles as 32 rows c of kwu * 32 values w: one cb_rows page per c-tile.
//
// Variants (one define):
//   TTX_TWO_PASS  tilize_block -> cb_tiles (L1), then transpose_tile (unpacker transpose) into DEST.
//   TTX_FUSED     one pass: unpack_tilize -> datacopy (tilize MOP) -> transpose_dest in DEST, the
//                 math MOP re-programmed twice per DEST batch (datacopy, then transpose_dest).
//   TTX_FUSED_RAW one pass: unpack_tilize -> 8 raw MOVA2D (no MOP) -> transpose_dest MOP, programmed
//                 once per kernel; transpose_dest reuses the tilize unpack's SrcA/SrcB valids, so no
//                 dummy-valid unpack and no per-batch init.
//   TTX_FUSED_D2B one pass: unpack_tilize -> 8 MOVA2D into a scratch DEST tile (row-linear), then per
//                 16x16 block 16 single-row MOVD2B gather -> TRNSPSRCB -> 4 MOVB2D into its transposed
//                 face of the output DEST tile. Needs one spare DEST tile: kw <= 7.
// TTX_FUSED and TTX_FUSED_RAW are WRONG on Blackhole (kept as measured evidence): BH's 16-bit
// unpack_tilize lands the RM block in SrcA row-linearly (SrcA row j = block row j / 2, half j % 2)
// and the face reorder happens in the packer (PackMode::Tilize), so DEST holds untilized rows and the
// in-DEST transpose interleaves the two c-halves. (A gather by 64 single-row MOVA2D, DEST face F row
// R <- SrcA row 32 * (F / 2) + 2 * R + F % 2, did not move single rows on BH: measured wrong, dropped.)
// TTX_FUSED_D2B does the reorder on the DEST -> SrcB move instead (single-row MOVD2B, as LLK uses).
// Raw LLK: kernel_lib has no transpose helper; the untilize helper's unpack init has no transpose, and
// the tilize helper packs to L1 (it cannot leave the tile in DEST for a transpose). The fused
// variants bypass compute_kernel_lib::tilize to keep the tile in DEST between tilize and untilize.

#include <cstdint>
#include <utility>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_in = get_compile_time_arg_val(0);
constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
constexpr uint32_t cb_tiles = get_compile_time_arg_val(2);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(3);
constexpr uint32_t kw = get_compile_time_arg_val(4);
constexpr uint32_t kw_tail = get_compile_time_arg_val(5);
constexpr uint32_t units_per_row = get_compile_time_arg_val(6);
// w-tiles per cb_in page: 1 with the experiment's reader, kw with the op's shared reader (one page per
// unit). A unit is always unit_pages pages.
constexpr uint32_t in_page_tiles = get_compile_time_arg_val(7);
constexpr uint32_t unit_pages = kw / in_page_tiles;
static_assert(kw % in_page_tiles == 0);

constexpr uint32_t TILE_ROWS = 32;
static_assert(kw >= 1 && kw <= 8 && kw_tail >= 1 && kw_tail <= kw, "a unit's w-tiles fit one half of 16-bit DEST");
#if defined(TTX_FUSED_D2B)
static_assert(kw <= 7, "TTX_FUSED_D2B needs a spare DEST tile after the unit's last one");
#endif

#ifdef TTX_ZONES
#define TTX_ZONE(name) DeviceZoneScopedN(name)
#else
#define TTX_ZONE(name)
#endif

// Tile (wt, ct) of the unit's RM block, as a tilize tile index from cb_in's read pointer: a block of
// 32 rows spans tensor_ct * 32 tile-column steps.
ALWI uint32_t tilize_index(uint32_t wt, uint32_t ct) { return wt * tensor_ct * TILE_ROWS + ct; }

#if defined(TTX_FUSED_D2B)
// One 16x16 block (wh, ch) of the row-linear scratch tile (DEST offset 64): its 16 rows (scratch rows
// 32 * wh + 2 * r + ch) are gathered row by row into SrcB rows 16..31, transposed there, and written
// as face 2 * ch + wh of the output tile (DEST offset 0) -- the block lands transposed and in the
// transposed face position, so tilize, transpose and face swap are one pass.
template <uint32_t WH, uint32_t CH, uint32_t... R>
ALWI void transpose_block_d2b(std::integer_sequence<uint32_t, R...>) {
    (TTI_MOVD2B(p_mov::DEST_NORM, 16 + R, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 64 + 32 * WH + 2 * R + CH), ...);
    TTI_TRNSPSRCB;
    constexpr uint32_t out = 16 * (2 * CH + WH);
    TTI_MOVB2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, out + 0);
    TTI_MOVB2D(p_mov::DEST_NORM, 20, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, out + 4);
    TTI_MOVB2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, out + 8);
    TTI_MOVB2D(p_mov::DEST_NORM, 28, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, out + 12);
}

// SrcA (row-linear tile from BH's 16-bit unpack_tilize) -> DEST tile dst_index + 1 (scratch) by 8
// MOVA2D, then the four gathered block transposes into DEST tile dst_index. ADDR_MOD_1 (programmed
// by transpose_dest_init) has zero increments: every move addresses by immediate offsets.
ALWI void copy_transpose_raw(uint32_t dst_index) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    math::_configure_preserve_zero_flag_state_();
    TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 0);
    TTI_MOVA2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 8);
    TTI_MOVA2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 16);
    TTI_MOVA2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 24);
    TTI_MOVA2D(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 32);
    TTI_MOVA2D(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 40);
    TTI_MOVA2D(p_mov::DEST_NORM, 48, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 48);
    TTI_MOVA2D(p_mov::DEST_NORM, 56, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 64 + 56);
    transpose_block_d2b<0, 0>(std::make_integer_sequence<uint32_t, 16>{});
    transpose_block_d2b<0, 1>(std::make_integer_sequence<uint32_t, 16>{});
    transpose_block_d2b<1, 0>(std::make_integer_sequence<uint32_t, 16>{});
    transpose_block_d2b<1, 1>(std::make_integer_sequence<uint32_t, 16>{});
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
#endif
}
#endif

#if defined(TTX_FUSED_RAW)
// srcA (tilized tile, 64 rows) -> DEST tile dst_index, then the in-place 32x32 transpose. The
// transpose MOP's ADDR_MOD_1 has zero increments, so the eight explicit MOVA2D address both
// register files by immediate offsets. SrcA/SrcB stay valid until the transpose's CLR_AB.
ALWI void copy_transpose_raw(uint32_t dst_index) {
#ifdef TRISC_MATH
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
    TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);
    TTI_MOVA2D(p_mov::DEST_NORM, 8, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 8);
    TTI_MOVA2D(p_mov::DEST_NORM, 16, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 16);
    TTI_MOVA2D(p_mov::DEST_NORM, 24, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 24);
    TTI_MOVA2D(p_mov::DEST_NORM, 32, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 32);
    TTI_MOVA2D(p_mov::DEST_NORM, 40, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 40);
    TTI_MOVA2D(p_mov::DEST_NORM, 48, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 48);
    TTI_MOVA2D(p_mov::DEST_NORM, 56, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 56);
    _llk_math_transpose_dest_<true, false>(dst_index);
#endif
}
#endif

template <uint32_t kwu>
ALWI void pack_rows() {
    cb_reserve_back(cb_rows, 1);
    tile_regs_wait();
    pack_untilize_dest<kwu, kwu>(cb_rows);
    tile_regs_release();
    cb_push_back(cb_rows, 1);
}

#if defined(TTX_TWO_PASS)
template <uint32_t kwu>
ALWI void process_unit() {
    {
        TTX_ZONE("CP-TILIZE");
        tilize_init(cb_in, tensor_ct, cb_tiles);
        cb_reserve_back(cb_tiles, kwu * tensor_ct);
        for (uint32_t wt = 0; wt < kwu; ++wt) {
            cb_wait_front(cb_in, wt / in_page_tiles + 1);
            tilize_block(cb_in, tensor_ct, cb_tiles, wt * tensor_ct, wt * tensor_ct);
        }
        cb_push_back(cb_tiles, kwu * tensor_ct);
        cb_wait_front(cb_in, unit_pages);
        cb_pop_front(cb_in, unit_pages);
        tilize_uninit(cb_in, cb_tiles);
    }
    {
        TTX_ZONE("CP-TRANSPOSE");
        transpose_init(cb_tiles);
        pack_untilize_dest_init<kwu, kwu>(cb_rows);
        cb_wait_front(cb_tiles, kwu * tensor_ct);
        for (uint32_t ct = 0; ct < tensor_ct; ++ct) {
            tile_regs_acquire();
            for (uint32_t wt = 0; wt < kwu; ++wt) {
                transpose_tile(cb_tiles, wt * tensor_ct + ct, wt);
            }
            tile_regs_commit();
            pack_rows<kwu>();
        }
        cb_pop_front(cb_tiles, kwu * tensor_ct);
        pack_untilize_uninit(cb_rows);
    }
}
#else
template <uint32_t kwu>
ALWI void process_unit() {
    for (uint32_t ct = 0; ct < tensor_ct; ++ct) {
        TTX_ZONE("CP-BATCH");
        tile_regs_acquire();
#if defined(TTX_FUSED_RAW) || defined(TTX_FUSED_D2B)
        for (uint32_t wt = 0; wt < kwu; ++wt) {
            if (ct == 0) {
                cb_wait_front(cb_in, wt / in_page_tiles + 1);
            }
            UNPACK((llk_unpack_tilize(cb_in, tilize_index(wt, ct))));
            copy_transpose_raw(wt);
        }
#else
        MATH((llk_math_eltwise_unary_datacopy_init<
              DataCopyType::A2D,
              false /*fp32 dest*/,
              BroadcastType::NONE,
              false /*is_int_en*/,
              PackMode::Tilize>(cb_in)));
        for (uint32_t wt = 0; wt < kwu; ++wt) {
            if (ct == 0) {
                cb_wait_front(cb_in, wt / in_page_tiles + 1);
            }
            UNPACK((llk_unpack_tilize(cb_in, tilize_index(wt, ct))));
            MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, false, BroadcastType::NONE, false>(wt, cb_in)));
        }
        MATH((llk_math_transpose_dest_init<true, false>()));
        for (uint32_t wt = 0; wt < kwu; ++wt) {
            UNPACK((llk_unpack_set_srcb_dummy_valid()));
            MATH((llk_math_transpose_dest<true, false>(wt)));
        }
#endif
        tile_regs_commit();
        pack_rows<kwu>();
    }
    cb_wait_front(cb_in, unit_pages);
    cb_pop_front(cb_in, unit_pages);
}
#endif

void kernel_main() {
    uint32_t remaining = get_arg_val<uint32_t>(0);
    uint32_t row_pos = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_in, cb_rows);
#if !defined(TTX_TWO_PASS)
    UNPACK((llk_unpack_tilize_init(cb_in, tensor_ct)));
#if defined(TTX_FUSED_RAW) || defined(TTX_FUSED_D2B)
    MATH((llk_math_transpose_dest_init<true, false>()));
#endif
#endif

    // pack_untilize's block width is a template argument: re-init when a row's tail unit is narrower.
    bool tail_mode = false;
#if !defined(TTX_TWO_PASS)
    pack_untilize_dest_init<kw, kw>(cb_rows);
#endif
    while (remaining > 0) {
        const bool tail = (kw_tail != kw) && (row_pos == units_per_row - 1);
        if constexpr (kw_tail != kw) {
#if !defined(TTX_TWO_PASS)
            if (tail != tail_mode) {
                pack_untilize_uninit(cb_rows);
                if (tail) {
                    pack_untilize_dest_init<kw_tail, kw_tail>(cb_rows);
                } else {
                    pack_untilize_dest_init<kw, kw>(cb_rows);
                }
            }
#endif
        }
        tail_mode = tail;
        if (tail) {
            process_unit<kw_tail>();
        } else {
            process_unit<kw>();
        }
        if (++row_pos == units_per_row) {
            row_pos = 0;
        }
        --remaining;
    }
#if !defined(TTX_TWO_PASS)
    pack_untilize_uninit(cb_rows);
#endif
}
