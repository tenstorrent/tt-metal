// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt>
ALWI void transpose_with_untilize(uint32_t cb_tilize, uint32_t cb_untilize, uint32_t cb_out) {
    uint32_t tile_idx = 0;

    for (uint32_t w = 0; w < Wt; ++w) {
        transpose_wh_init_short(cb_tilize);
        cb_reserve_back(cb_untilize, Ht);
        for (uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            transpose_wh_tile(cb_tilize, tile_idx, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_untilize);
            tile_regs_release();
            tile_idx += Wt;
        }
        tile_idx = tile_idx - HtWt + 1;
        cb_push_back(cb_untilize, Ht);

        // tilize
        // need to add this hw config here, otherwise pcc is bad
        UNPACK((llk_unpack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE>(cb_untilize)));
        MATH((llk_math_hw_configure_disaggregated(cb_untilize, cb_untilize)));
        untilize_init_short(cb_untilize);
        cb_wait_front(cb_untilize, Ht);
        cb_reserve_back(cb_out, Ht);
        untilize_block(cb_untilize, Ht, cb_out);
        cb_push_back(cb_out, Ht);
        cb_pop_front(cb_untilize, Ht);
        untilize_uninit(cb_untilize);
    }
}

template <
    uint32_t Wt,
    uint32_t Ht,
    uint32_t HtWt,
    bool use_narrow_row,
    uint32_t row_size,
    uint32_t pack_num_pages_last_col,
    uint32_t pack_num_pages_last_row_col>
ALWI void transpose_with_pack_untilize_narrow_row(uint32_t cb_tilize, uint32_t cb_out) {
    uint32_t tile_idx = 0;

    transpose_wh_init_short(cb_tilize);
    pack_untilize_dst_init_short<Ht, Ht, false, use_narrow_row, row_size>(cb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_wh_tile(cb_tilize, tile_idx, h);
            tile_idx += Wt;
        }

        tile_regs_commit();

        if (w == Wt - 1) {  // last row
            cb_reserve_back(cb_out, pack_num_pages_last_row_col);
            tile_regs_wait();
            pack_untilize_dst<Ht, Ht, false, use_narrow_row, row_size>(cb_out);
            tile_regs_release();
            cb_push_back(cb_out, pack_num_pages_last_row_col);
        } else {
            cb_reserve_back(cb_out, pack_num_pages_last_col);
            tile_regs_wait();
            pack_untilize_dst<Ht, Ht, false, use_narrow_row, row_size>(cb_out);
            tile_regs_release();
            cb_push_back(cb_out, pack_num_pages_last_col);
        }
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
}

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt>
ALWI void transpose_with_pack_untilize(uint32_t cb_tilize, uint32_t cb_out) {
    uint32_t tile_idx = 0;

    transpose_wh_init_short(cb_tilize);
    pack_untilize_dst_init_short<Ht>(cb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_wh_tile(cb_tilize, tile_idx, h);
            tile_idx += Wt;
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, Ht);
        tile_regs_wait();
        pack_untilize_dst<Ht>(cb_out);

        tile_regs_release();
        cb_push_back(cb_out, Ht);

        cb_wait_front(cb_out, Ht);
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
#ifdef SHARDED
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t last_output_row_num_datums = get_compile_time_arg_val(4);
    constexpr uint32_t pack_num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t pack_num_pages_last_col = get_compile_time_arg_val(6);
    constexpr uint32_t pack_num_pages_last_row = get_compile_time_arg_val(7);
    constexpr uint32_t pack_num_pages_last_row_col = get_compile_time_arg_val(8);

    constexpr bool use_narrow_row = last_output_row_num_datums < TILE_WIDTH ? true : false;
    constexpr uint32_t row_size = last_output_row_num_datums < TILE_WIDTH ? last_output_row_num_datums : TILE_WIDTH;
#else
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(0);
#endif

#ifdef SHARDED
    constexpr auto cb_in = tt::CBIndex::c_24;
    constexpr auto cb_tilize = tt::CBIndex::c_25;
    constexpr auto cb_untilize = tt::CBIndex::c_26;
    constexpr auto cb_out =
        (Ht > 8) ? tt::CBIndex::c_27 : tt::CBIndex::c_16;  // temporary fix until pack_untilze is fully fixed
#else
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_24;
    constexpr auto cb_untilize = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
#endif

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // tilize input
        tilize_init_short(cb_in, Wt, cb_tilize);
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_wait_front(cb_in, Wt);
            cb_reserve_back(cb_tilize, Wt);
            tilize_block(cb_in, Wt, cb_tilize);
            cb_push_back(cb_tilize, Wt);
            cb_pop_front(cb_in, Wt);
        }
        tilize_uninit(cb_in, cb_tilize);

        // transpose
        cb_wait_front(cb_tilize, HtWt);
        uint32_t tile_idx = 0;
        if constexpr (Ht > 8) {  // temporary fix until pack_untilze is fully fixed
            transpose_with_untilize<Wt, Ht, HtWt>(cb_tilize, cb_untilize, cb_out);
        } else {
#ifdef SHARDED
            if constexpr (use_narrow_row) {
                transpose_with_pack_untilize_narrow_row<
                    Wt,
                    Ht,
                    HtWt,
                    use_narrow_row,
                    row_size,
                    pack_num_pages_last_col,
                    pack_num_pages_last_row_col>(cb_tilize, cb_out);
            } else {
                transpose_with_pack_untilize<Wt, Ht, HtWt>(cb_tilize, cb_out);
            }
#else
            transpose_with_pack_untilize<Wt, Ht, HtWt>(cb_tilize, cb_out);
#endif
        }
        cb_pop_front(cb_tilize, HtWt);
    }
}
}  // namespace NAMESPACE
