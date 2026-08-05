// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_cols_per_core = get_arg(args::num_cols_per_core);
    const auto Ht = get_arg(args::Ht);
    const auto origin_h = get_arg(args::origin_h);

    DataflowBuffer dfb_x_obj(dfb::x);            // input
    DataflowBuffer dfb_one_obj(dfb::one);        // one
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);  // mask_h

    DataflowBuffer dfb_y_obj(dfb::y);  // output

    // Compute-private intermediates: this kernel is their only toucher, so each is self-looped on the
    // host (bound PRODUCER and CONSUMER under one accessor name) and one object drives both directions.
    DataflowBuffer dfb_val_obj(dfb::val);        // f(x)
    DataflowBuffer dfb_cal_obj(dfb::cal);        // calculate f(x) over dimension
    DataflowBuffer dfb_reduce_obj(dfb::reduce);  // reduce f(x)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(dfb::x, dfb::x, dfb::y);

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
            copy_tile(dfb::x, 0, dst0);

            if (do_mask_h && (row_idx == Ht - 1)) {
                copy_tile_init_with_dt(dfb_mask_h_obj);
                copy_tile(dfb::mask_h, 0, dst1);

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
                copy_tile(dfb::val, 0, dst0);
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
                add_tiles(dfb::val, dfb::cal, 0, 0, dst0);
#else
                copy_tile_init_with_dt(dfb_val_obj);
                copy_tile(dfb::val, 0, dst0);

                copy_tile_init_with_dt(dfb_cal_obj);
                copy_tile(dfb::cal, 0, dst1);

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
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::cal, dfb::one, dfb::reduce>(
            compute_kernel_lib::ReduceInputBlockShape::single());

        tile_regs_acquire();

        dfb_reduce_obj.wait_front(onetile);
        dfb_y_obj.reserve_back(onetile);

        copy_tile_init_with_dt(dfb_reduce_obj);
        copy_tile(dfb::reduce, 0, dst0);
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
