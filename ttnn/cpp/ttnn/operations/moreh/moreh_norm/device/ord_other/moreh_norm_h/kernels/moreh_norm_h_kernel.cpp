// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);

    constexpr std::uint8_t input_id = tt::CB::c_in0;
    constexpr auto cb_x = input_id + 0;
    DataflowBuffer dfb_x_obj(cb_x);  // input
    constexpr auto cb_one = input_id + 1;
    DataflowBuffer dfb_one_obj(cb_one);  // one
    constexpr auto cb_mask_h = input_id + 2;
    DataflowBuffer dfb_mask_h_obj(cb_mask_h);  // mask_h

    constexpr auto cb_y = tt::CB::c_out0;
    DataflowBuffer dfb_y_obj(cb_y);  // output

    constexpr std::uint8_t intermed_id = tt::CB::c_intermed0;
    constexpr auto cb_tmp0 = intermed_id + 0;
    constexpr auto cb_tmp1 = intermed_id + 1;
    constexpr auto cb_tmp2 = intermed_id + 2;

    constexpr auto cb_val = cb_tmp0;
    DataflowBuffer dfb_val_obj(cb_val);  // f(x)
    constexpr auto cb_cal = cb_tmp1;
    DataflowBuffer dfb_cal_obj(cb_cal);  // calculate f(x) over dimension
    constexpr auto cb_reduce = cb_tmp2;
    DataflowBuffer dfb_reduce_obj(cb_reduce);  // reduce f(x)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    compute_kernel_hw_startup(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    dfb_one_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);  // comes from the reader
    }
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            // f(x)
            tile_regs_acquire();
            dfb_x_obj.wait_front(onetile);  // comes from the reader
            dfb_val_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_x_obj);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_h && (row_idx == Ht - 1)) {
                copy_tile_init_with_dt(dfb_mask_h_obj);
                copy_tile(cb_mask_h, 0, dst1);

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
            pack_tile_with_dt(dst0, dfb_val_obj);
            tile_regs_release();

            dfb_x_obj.pop_front(onetile);
            dfb_val_obj.push_back(onetile);

            // calculate f(x) over dimension
            if (row_idx == 0) {
                tile_regs_acquire();
                dfb_val_obj.wait_front(onetile);
                dfb_cal_obj.reserve_back(onetile);

                copy_tile_init_with_dt(dfb_val_obj);
                copy_tile(cb_val, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_cal_obj);
                tile_regs_release();

                dfb_val_obj.pop_front(onetile);
                dfb_cal_obj.push_back(onetile);

            } else {
                tile_regs_acquire();
                dfb_val_obj.wait_front(onetile);
                dfb_cal_obj.wait_front(onetile);
                dfb_cal_obj.reserve_back(onetile);
#ifdef IS_ZERO
                add_tiles_init_with_dt(dfb_val_obj, dfb_cal_obj);
                add_tiles(cb_val, cb_cal, 0, 0, dst0);
#else
                copy_tile_init_with_dt(dfb_val_obj);
                copy_tile(cb_val, 0, dst0);

                copy_tile_init_with_dt(dfb_cal_obj);
                copy_tile(cb_cal, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
#endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_cal_obj);
                tile_regs_release();

                dfb_val_obj.pop_front(onetile);
                dfb_cal_obj.pop_front(onetile);
                dfb_cal_obj.push_back(onetile);
            }
        }
        // reduce f(x)
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_cal, cb_one, cb_reduce>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        tile_regs_acquire();

        dfb_reduce_obj.wait_front(onetile);
        dfb_y_obj.reserve_back(onetile);

        copy_tile_init_with_dt(dfb_reduce_obj);
        copy_tile(cb_reduce, 0, dst0);
#ifdef MINUS_INF
        negative_tile_init();
        negative_tile(dst0);
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_y_obj);
        tile_regs_release();

        dfb_reduce_obj.pop_front(onetile);
        dfb_y_obj.push_back(onetile);
    }

    dfb_one_obj.pop_front(onetile);
    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
}
