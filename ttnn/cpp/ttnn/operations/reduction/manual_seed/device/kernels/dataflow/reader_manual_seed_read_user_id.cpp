// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/constants.hpp>

void kernel_main() {
    // Runtime args
    const uint32_t user_ids_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t core_id = get_compile_time_arg_val(0);
    constexpr uint32_t user_ids_cb_index = get_compile_time_arg_val(1);
    constexpr auto user_ids_tensor_accessor_args = TensorAccessorArgs<2>();

    // Constants
    constexpr uint32_t one_tile = 1;

    // Index tensor config
    constexpr uint32_t user_ids_tensor_tile_size_bytes = get_tile_size(user_ids_cb_index);
    constexpr DataFormat user_ids_tensor_data_format = get_dataformat(user_ids_cb_index);
    const auto user_ids_tensor_dram =
        TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr, user_ids_tensor_tile_size_bytes);

    // Read user_id from circular buffer
    cb_reserve_back(user_ids_cb_index, one_tile);
    const uint32_t l1_write_addr_index = get_write_ptr(user_ids_cb_index);
    noc_async_read_tile(0, user_ids_tensor_dram, l1_write_addr_index);
    noc_async_read_barrier();
    cb_push_back(user_ids_cb_index, one_tile);
}
