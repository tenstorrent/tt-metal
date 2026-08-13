// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include <tt-metalium/constants.hpp>
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    // Runtime args
    const auto core_id = get_arg(args::core_id);

    // Compile time args
    constexpr auto number_of_ids = get_arg(args::number_of_ids);

    // Constants
    constexpr uint32_t one_tile = 1;

    Noc noc;
    DataflowBuffer user_ids_dfb(dfb::user_ids);
    DataflowBuffer kernel_communication_dfb(dfb::kernel_communication);

    // Index tensor config
    const DataFormat user_ids_tensor_data_format = user_ids_dfb.get_dataformat();
    const auto user_ids_tensor_dram = TensorAccessor(tensor::user_ids);

    // Read user_id from dataflow buffer
    user_ids_dfb.reserve_back(one_tile);
    const uint32_t l1_write_addr_index = user_ids_dfb.get_write_ptr();
    noc.async_read(
        user_ids_tensor_dram, user_ids_dfb, user_ids_dfb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
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
    kernel_communication_dfb.reserve_back(one_tile);
    CoreLocalMem<volatile uint32_t> communication_ptr(kernel_communication_dfb.get_write_ptr());
    communication_ptr[0] = is_user_id ? 1 : 0;

    // Send to compute kernel
    kernel_communication_dfb.push_back(one_tile);
}
