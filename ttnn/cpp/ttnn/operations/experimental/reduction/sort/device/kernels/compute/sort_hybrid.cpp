// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

// #include "sort_common.hpp"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    // constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    // constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    // constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint32_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

}  // MAIN
}  // namespace NAMESPACE
