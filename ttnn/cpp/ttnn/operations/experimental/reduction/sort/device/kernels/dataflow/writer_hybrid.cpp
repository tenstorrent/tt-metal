// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    // const uint32_t start_core_physical_coord_x = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    // constexpr bool input_tensor_is_dram = get_compile_time_arg_val(7) == 1;
    // constexpr bool output_tensor_is_dram = get_compile_time_arg_val(8) == 1;
    // constexpr bool output_index_tensor_is_dram = get_compile_time_arg_val(9) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint32_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
}
