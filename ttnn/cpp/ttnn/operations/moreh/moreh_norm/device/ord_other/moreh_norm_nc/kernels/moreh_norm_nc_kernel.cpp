// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;    // input
    const auto cb_one = input_id++;  // one

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;  // output

    std::uint8_t intermed_id{tt::CB::c_intermed0};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;

    const auto cb_val = cb_tmp0;  // f(x)
    const auto cb_cal = cb_tmp1;  // calculate f(x) over dimensions

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_wait_front(cb_one, onetile);  // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // x != 0
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_val, onetile);

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

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_val, onetile);

            // Add(x != 0)
            if (inner_idx == 0) {
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

        // Compute cb_y
        tile_regs_acquire();

        cb_wait_front(cb_cal, onetile);
        cb_reserve_back(cb_y, onetile);

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

        cb_pop_front(cb_cal, onetile);
        cb_push_back(cb_y, onetile);
    }
    cb_pop_front(cb_one, onetile);

}  // void MAIN
}  // namespace NAMESPACE
