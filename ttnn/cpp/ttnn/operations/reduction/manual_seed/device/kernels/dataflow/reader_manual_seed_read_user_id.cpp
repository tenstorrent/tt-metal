// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include <tt-metalium/constants.hpp>
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    // Runtime args
    const uint32_t user_ids_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t user_ids_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_communication_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t number_of_ids = get_compile_time_arg_val(2);
    constexpr auto user_ids_tensor_accessor_args = TensorAccessorArgs<3>();

    // Constants
    constexpr uint32_t one_tile = 1;

    // Index tensor config
    constexpr DataFormat user_ids_tensor_data_format = get_dataformat(user_ids_cb_index);
    const auto user_ids_tensor_dram = TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr);

    Noc noc;
    DataflowBuffer user_ids_cb(user_ids_cb_index);
    DataflowBuffer kernel_communication_cb(kernel_communication_cb_index);

    // Read user_id from circular buffer
    user_ids_cb.reserve_back(one_tile);
    const uint32_t l1_write_addr_index = user_ids_cb.get_write_ptr();
    noc.async_read(
        user_ids_tensor_dram, user_ids_cb, get_tile_size(user_ids_cb_index), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    // Process user_ids
    bool is_user_id = false;
    CoreLocalMem<volatile uint32_t> ptr(l1_write_addr_index);
    for (uint32_t id = 0; id < number_of_ids; ++id) {
        if (core_id == ptr[id]) {
            is_user_id = true;  // Indicate match
            break;
        }
    }

    // Prepare message for compute kernel
    kernel_communication_cb.reserve_back(one_tile);
    CoreLocalMem<volatile uint32_t> communication_ptr(kernel_communication_cb.get_write_ptr());
    communication_ptr[0] = is_user_id ? 1 : 0;

    // Send to compute kernel
    kernel_communication_cb.push_back(one_tile);
}
