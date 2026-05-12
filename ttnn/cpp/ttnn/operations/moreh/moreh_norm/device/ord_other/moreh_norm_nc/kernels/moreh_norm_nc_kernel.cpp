// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;
    experimental::CircularBuffer cb_x_obj(cb_x);  // input
    const auto cb_one = input_id++;
    experimental::CircularBuffer cb_one_obj(cb_one);  // one

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;
    experimental::CircularBuffer cb_y_obj(cb_y);  // output

    std::uint8_t intermed_id{tt::CB::c_intermed0};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;

    const auto cb_val = cb_tmp0;
    experimental::CircularBuffer cb_val_obj(cb_val);  // f(x)
    const auto cb_cal = cb_tmp1;
    experimental::CircularBuffer cb_cal_obj(cb_cal);  // calculate f(x) over dimensions

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_one_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // x != 0
            tile_regs_acquire();
            cb_x_obj.wait_front(onetile);  // comes from the reader
            cb_val_obj.reserve_back(onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);
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

            cb_x_obj.pop_front(onetile);
            cb_val_obj.push_back(onetile);

            // Add(x != 0)
            if (inner_idx == 0) {
                tile_regs_acquire();
                cb_val_obj.wait_front(onetile);
                cb_cal_obj.reserve_back(onetile);

                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_val_obj.pop_front(onetile);
                cb_cal_obj.push_back(onetile);

            } else {
                tile_regs_acquire();
                cb_val_obj.wait_front(onetile);
                cb_cal_obj.wait_front(onetile);
                cb_cal_obj.reserve_back(onetile);
#ifdef IS_ZERO
                add_tiles_init_with_dt(cb_val, cb_cal);
                add_tiles(cb_val, cb_cal, 0, 0, dst0);
#else
                copy_tile_init_with_dt(cb_val);
                copy_tile(cb_val, 0, dst0);

                copy_tile_init_with_dt(cb_cal);
                copy_tile(cb_cal, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
#endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_cal);
                tile_regs_release();

                cb_val_obj.pop_front(onetile);
                cb_cal_obj.pop_front(onetile);
                cb_cal_obj.push_back(onetile);
            }
        }

        // Compute cb_y
        tile_regs_acquire();

        cb_cal_obj.wait_front(onetile);
        cb_y_obj.reserve_back(onetile);

        copy_tile_init_with_dt(cb_cal);
        copy_tile(cb_cal, 0, dst0);
#ifdef MINUS_INF
        negative_tile_init();
        negative_tile(dst0);
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_y);
        tile_regs_release();

        cb_cal_obj.pop_front(onetile);
        cb_y_obj.push_back(onetile);
    }
    cb_one_obj.pop_front(onetile);
}
