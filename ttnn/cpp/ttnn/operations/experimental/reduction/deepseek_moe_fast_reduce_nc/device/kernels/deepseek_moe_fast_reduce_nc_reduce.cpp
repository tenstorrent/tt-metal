// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

constexpr uint32_t num_output_tiles_to_process = get_compile_time_arg_val(0);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(1);
constexpr uint32_t input_granularity = get_compile_time_arg_val(2);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(3);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(4);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(5);

void kernel_main() {
    // hardcoded constants
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(compute_input_cb_id_0, compute_input_cb_id_1, compute_output_cb_id);

    // For each assigned output tile, process the input tiles in a doubly nested loop.
    // The inner loop processes the number of tiles specified by input_granularity.
    // The outer loop executes reduction_dim_size / input_granularity times.
    cb_wait_front(compute_input_cb_id_1, one_tile);
    for (uint32_t i = 0; i < num_output_tiles_to_process; i++) {
        add_tiles_init(compute_input_cb_id_0, compute_input_cb_id_1, true);
        reconfig_data_format(compute_input_cb_id_0, compute_input_cb_id_1);
        tile_regs_acquire();

        constexpr uint32_t num_input_tiles_iter = reduction_dim_size / input_granularity;
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_wait_front(compute_input_cb_id_0, input_granularity);
            for (uint32_t k = 0; k < input_granularity; ++k) {
                add_tiles(compute_input_cb_id_0, compute_input_cb_id_1, k, first_tile, dst0);
            }
            cb_pop_front(compute_input_cb_id_0, input_granularity);
        }

        tile_regs_commit();
        cb_reserve_back(compute_output_cb_id, one_tile);
        pack_reconfig_data_format(compute_output_cb_id);
        tile_regs_wait();
        pack_tile(dst0, compute_output_cb_id);
        tile_regs_release();
        cb_push_back(compute_output_cb_id, one_tile);
    }
}
