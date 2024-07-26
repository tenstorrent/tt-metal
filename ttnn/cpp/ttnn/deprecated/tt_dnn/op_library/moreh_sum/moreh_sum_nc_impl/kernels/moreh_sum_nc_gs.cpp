// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_wait_front(cb_in1, onetile);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            bool last_out = (j == num_input_tiles - 1);
            uint32_t cb_add  = (enable_reload) ? (cb_intermed0) : (cb_in1);

            cb_wait_front(cb_in0, onetile);
            if (enable_reload) {
                cb_wait_front(cb_intermed0, onetile);
            }

            tile_regs_acquire();
            #if defined FP32_DEST_ACC_EN
                unpack_reconfig_data_format(cb_in0, cb_add);
            #endif
            add_tiles_init(cb_in0, cb_add);
            add_tiles(cb_in0, cb_add, first_tile, first_tile, dst0);
            tile_regs_commit();

            cb_pop_front(cb_in0, onetile);
            if (enable_reload) {
                cb_pop_front(cb_intermed0, onetile);
            }

            uint32_t cb_out = (last_out) ? (cb_out0) : (cb_intermed0);
            cb_reserve_back(cb_out, onetile);
            tile_regs_wait();
            #if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_out);
            #endif
            pack_tile(dst0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, onetile);
            enable_reload = true;
        }
    }
}
}  // namespace NAMESPACE
