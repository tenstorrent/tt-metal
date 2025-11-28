// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
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
    constexpr uint32_t number_of_ids = get_compile_time_arg_val(2);
    constexpr auto user_ids_tensor_accessor_args = TensorAccessorArgs<3>();
    constexpr auto seeds_tensor_accessor_args =
        TensorAccessorArgs<user_ids_tensor_accessor_args.next_compile_time_args_offset()>();

    // Constants
    constexpr uint32_t one_tile = 1;

    // Index tensor config
    constexpr uint32_t user_ids_tensor_tile_size_bytes = get_tile_size(user_ids_cb_index);
    constexpr DataFormat user_ids_tensor_data_format = get_dataformat(user_ids_cb_index);
    const auto user_ids_tensor_dram =
        TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr, user_ids_tensor_tile_size_bytes);

    constexpr uint32_t seeds_tensor_tile_size_bytes = get_tile_size(seeds_cb_index);
    constexpr DataFormat seeds_tensor_data_format = get_dataformat(seeds_cb_index);
    const auto seeds_tensor_dram =
        TensorAccessor(seeds_tensor_accessor_args, seeds_tensor_buffer_addr, seeds_tensor_tile_size_bytes);

    // Read user_id from circular buffer
    cb_reserve_back(user_ids_cb_index, one_tile);
    const uint32_t l1_write_addr_index = get_write_ptr(user_ids_cb_index);
    noc_async_read_tile(0, user_ids_tensor_dram, l1_write_addr_index);
    noc_async_read_barrier();

    // Read seeds from circular buffer
    cb_reserve_back(seeds_cb_index, one_tile);
    const uint32_t seeds_l1_write_addr_index = get_write_ptr(seeds_cb_index);
    noc_async_read_tile(0, seeds_tensor_dram, seeds_l1_write_addr_index);
    noc_async_read_barrier();

    // Process user_ids
    uint32_t seed = 0;
    bool is_user_id = false;
    volatile tt_l1_ptr uint32_t* user_id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr_index);
    volatile tt_l1_ptr uint32_t* seeds = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seeds_l1_write_addr_index);
    for (uint32_t id = 0; id < number_of_ids; ++id) {
        if (core_id == user_id[id]) {
            is_user_id = true;  // Indicate match
            seed = seeds[id];
            break;
        }
    }

    // Send result to compute kernel via mailbox
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, is_user_id);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, is_user_id);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, is_user_id);

    // Send seed to compute kernel via mailbox
    if (is_user_id) {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, seed);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, seed);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, seed);
    }
}
