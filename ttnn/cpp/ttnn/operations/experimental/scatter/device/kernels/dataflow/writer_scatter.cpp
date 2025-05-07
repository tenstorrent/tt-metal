// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

void kernel_main() {
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src_tensor_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(3);
    constexpr uint32_t index_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t src_tensor_addr = get_compile_time_arg_val(5);
    constexpr uint32_t input_tensor_cb = get_compile_time_arg_val(6);
    constexpr uint32_t index_tensor_cb = get_compile_time_arg_val(7);
    constexpr uint32_t src_tensor_cb = get_compile_time_arg_val(8);
    constexpr uint32_t output_tensor_cb = get_compile_time_arg_val(9);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(10);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(11);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(12);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(13);
}
