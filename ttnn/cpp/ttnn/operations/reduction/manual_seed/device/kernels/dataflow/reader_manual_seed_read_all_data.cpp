// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"
#include <tt-metalium/constants.hpp>
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    // Runtime args
    const uint32_t user_ids_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t seeds_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t user_ids_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t seeds_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_communication_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t number_of_ids = get_compile_time_arg_val(3);
    constexpr auto user_ids_tensor_accessor_args = TensorAccessorArgs<4>();
    constexpr auto seeds_tensor_accessor_args =
        TensorAccessorArgs<user_ids_tensor_accessor_args.next_compile_time_args_offset()>();

    // Constants
    constexpr uint32_t one_tile = 1;

    // Index tensor config
    constexpr DataFormat user_ids_tensor_data_format = get_dataformat(user_ids_cb_index);
    const auto user_ids_tensor_dram = TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr);

    constexpr DataFormat seeds_tensor_data_format = get_dataformat(seeds_cb_index);
    const auto seeds_tensor_dram = TensorAccessor(seeds_tensor_accessor_args, seeds_tensor_buffer_addr);

    experimental::Noc noc;
    experimental::CircularBuffer user_ids_cb(user_ids_cb_index);
    experimental::CircularBuffer seeds_cb(seeds_cb_index);
    experimental::CircularBuffer kernel_communication_cb(kernel_communication_cb_index);

    // Read user_id from circular buffer
    user_ids_cb.reserve_back(one_tile);
    const uint32_t l1_write_addr_index = user_ids_cb.get_write_ptr();
    noc.async_read(
        user_ids_tensor_dram, user_ids_cb, get_tile_size(user_ids_cb_index), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    // Read seeds from circular buffer
    seeds_cb.reserve_back(one_tile);
    const uint32_t seeds_l1_write_addr_index = seeds_cb.get_write_ptr();
    noc.async_read(seeds_tensor_dram, seeds_cb, get_tile_size(seeds_cb_index), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    // Process user_ids
    uint32_t seed = 0;
    bool is_user_id = false;
    experimental::CoreLocalMem<volatile uint32_t> user_id(l1_write_addr_index);
    experimental::CoreLocalMem<volatile uint32_t> seeds(seeds_l1_write_addr_index);
    for (uint32_t id = 0; id < number_of_ids; ++id) {
        if (core_id == user_id[id]) {
            is_user_id = true;  // Indicate match
            seed = seeds[id];
            break;
        }
    }

    // Prepare message for compute kernel
    kernel_communication_cb.reserve_back(one_tile);
    experimental::CoreLocalMem<volatile uint32_t> communication_ptr(kernel_communication_cb.get_write_ptr());
    communication_ptr[0] = is_user_id ? 1 : 0;
    communication_ptr[1] = is_user_id ? seed : 0;

    // Send to compute kernel
    kernel_communication_cb.push_back(one_tile);
}
