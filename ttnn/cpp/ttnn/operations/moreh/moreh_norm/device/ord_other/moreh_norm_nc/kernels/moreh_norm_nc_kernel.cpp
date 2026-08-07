// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_output_tiles_per_core = get_arg(args::num_output_tiles_per_core);
    const auto num_reduced_tiles_along_dim = get_arg(args::num_reduced_tiles_along_dim);

    DataflowBuffer dfb_x_obj(dfb::x);      // input
    DataflowBuffer dfb_one_obj(dfb::one);  // one

    DataflowBuffer dfb_y_obj(dfb::y);  // output

    // Compute-private intermediates: this kernel is their only toucher, so each is self-looped on the
    // host (bound PRODUCER and CONSUMER under one accessor name) and one object drives both directions.
    DataflowBuffer dfb_val_obj(dfb::val);  // f(x)
    DataflowBuffer dfb_cal_obj(dfb::cal);  // calculate f(x) over dimensions

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // x != 0
            tile_regs_acquire();
            dfb_x_obj.wait_front(onetile);  // comes from the reader
            dfb_val_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_x_obj);
            copy_tile(dfb::x, 0, dst0);
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

            // Add(x != 0)
            if (inner_idx == 0) {
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

        // Compute cb_y
        tile_regs_acquire();

        dfb_cal_obj.wait_front(onetile);
        dfb_y_obj.reserve_back(onetile);

        copy_tile_init_with_dt(dfb_cal_obj);
        copy_tile(dfb::cal, 0, dst0);
#ifdef MINUS_INF
        negative_tile_init();
        negative_tile(dst0);
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_y_obj);
        tile_regs_release();

        dfb_cal_obj.pop_front(onetile);
        dfb_y_obj.push_back(onetile);
    }
    dfb_one_obj.pop_front(onetile);
}
