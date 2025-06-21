// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;       // input
    const auto cb_one = input_id++;     // one
    const auto cb_mask_w = input_id++;  // mask_w

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;  // output

    std::uint8_t intermed_id{tt::CB::c_intermed0};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;

    const auto cb_val = cb_tmp0;     // f(x)
    const auto cb_cal = cb_tmp1;     // calculate f(x) over dimension
    const auto cb_reduce = cb_tmp2;  // reduce f(x)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_wait_front(cb_one, onetile);  // comes from the reader

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);  // comes from the reader
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            // f(x)
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_val, onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_w && (col_idx == Wt - 1)) {
                copy_tile_init_with_dt(cb_mask_w);
                copy_tile(cb_mask_w, 0, dst1);
                mask_tile_init();
#ifdef MINUS_INF
                mask_posinf_tile(dst0, dst1);
#else
                mask_tile(dst0, dst1);
#endif
            }
#ifdef IS_ZERO
            unary_ne_tile_init();
            unary_ne_tile(dst0, 0);
#else
            abs_tile_init();
            abs_tile(dst0);
#endif

#ifdef MINUS_INF
            negative_tile_init();
            negative_tile(dst0);
#endif
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_val);
            tile_regs_release();

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_val, onetile);

            // calculate f(x) over dimension
            if (col_idx == 0) {
                tile_regs_acquire();
                cb_wait_front(cb_val, onetile);
                cb_reserve_back(cb_cal, onetile);

                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_pop_front(cb_val, onetile);
                cb_push_back(cb_cal, onetile);
            } else {
                tile_regs_acquire();
                cb_wait_front(cb_val, onetile);
                cb_wait_front(cb_cal, onetile);
                cb_reserve_back(cb_cal, onetile);
#ifdef IS_ZERO
                add_tiles_init_with_dt(cb_val, cb_cal);
                add_tiles(cb_val, cb_cal, 0, 0, dst0);
#else
                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);

                copy_tile_init_with_dt(cb_cal);
                copy_tile(cb_cal, 0, dst1);

                max_tile_init();
                max_tile(dst0, dst1);
#endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_pop_front(cb_val, onetile);
                cb_pop_front(cb_cal, onetile);
                cb_push_back(cb_cal, onetile);
            }
        }
        // reduce f(x)
        tile_regs_acquire();
        cb_wait_front(cb_cal, onetile);
        cb_reserve_back(cb_reduce, onetile);

        reduce_init_delta_with_dt<REDUCE_OP, REDUCE_DIM>(cb_reduce, cb_cal, cb_one);
        reduce_tile(cb_cal, cb_one, 0, 0, dst0);
        reduce_uninit();
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_reduce);
        tile_regs_release();

        cb_pop_front(cb_cal, onetile);
        cb_push_back(cb_reduce, onetile);

        tile_regs_acquire();

        cb_wait_front(cb_reduce, onetile);
        cb_reserve_back(cb_y, onetile);

        copy_tile_init_with_dt(cb_reduce);
        copy_tile(cb_reduce, 0, dst0);
#ifdef MINUS_INF
        negative_tile_init();
        negative_tile(dst0);
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_y);
        tile_regs_release();

        cb_pop_front(cb_reduce, onetile);
        cb_push_back(cb_y, onetile);
    }

    cb_pop_front(cb_one, onetile);
    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }

}  // void MAIN
}  // namespace NAMESPACE
