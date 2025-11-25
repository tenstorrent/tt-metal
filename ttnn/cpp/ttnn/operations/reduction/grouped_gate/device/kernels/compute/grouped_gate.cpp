// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    // Dummy compute kernel
    constexpr uint32_t sigmoid_input_cb_index = get_named_compile_time_arg_val("sigmoid_input_cb_index");
    constexpr uint32_t add_bias_cb_index = get_named_compile_time_arg_val("add_bias_cb_index");
    constexpr uint32_t weights_cb_index = get_named_compile_time_arg_val("weights_cb_index");
    constexpr uint32_t indices_cb_index = get_named_compile_time_arg_val("indices_cb_index");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);

    binary_op_init_common(sigmoid_input_cb_index, add_bias_cb_index, weights_cb_index);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        base_page = height_tile * width_tiles;
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            cb_wait_front(sigmoid_input_cb_index, 1);
            cb_wait_front(add_bias_cb_index, 1);

            // do stuff

            cb_pop_front(sigmoid_input_cb_index, 1);
            cb_pop_front(add_bias_cb_index, 1);
        }
    }
}
}  // namespace NAMESPACE
