// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t input_granularity = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    constexpr uint32_t num_input_tiles_iter = num_input_tiles / input_granularity;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_wait_front(cb_in1, onetile);

    // For each assigned output tile, process the input tiles in a doubly nested
    // loop. The inner loop processes the number of tiles specified by
    // input_granularity. The outer loop executes num_input_tiles / input_granularity
    // times.
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        add_tiles_init(cb_in0, cb_in1, true);
        reconfig_data_format(cb_in0, cb_in1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_wait_front(cb_in0, input_granularity);
            for (uint32_t k = 0; k < input_granularity; k++) {
                add_tiles(cb_in0, cb_in1, k, first_tile, dst0);
            }
            cb_pop_front(cb_in0, input_granularity);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out0, onetile);
        pack_reconfig_data_format(cb_out0);
        tile_regs_wait();
        pack_tile(dst0, cb_out0);
        tile_regs_release();
        cb_push_back(cb_out0, onetile);
    }
}
}  // namespace NAMESPACE
