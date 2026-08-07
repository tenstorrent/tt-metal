// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include <cstdint>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(2);
    constexpr auto tensor0_args = TensorAccessorArgs<3>();

    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;

    static_assert(cb_page_size % input_page_size == 0);

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);

    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0);

    Noc noc_obj;
    CircularBuffer cb0(cb0_id);

    for (uint32_t page_id = input_page_id_start; page_id < input_page_id_end;) {
        cb0.reserve_back(1);
        uint32_t l1_write_addr = cb0.get_write_ptr();

        // fill CB page
        for (uint32_t input = 0; input < inputs_per_cb_page; input++) {
            if (page_id + input >= input_page_id_end) [[unlikely]] {
                break;
            }
            auto cb_input_page = l1_write_addr + input * input_page_size;
            noc_obj.async_read(
                tensor0_addrgen,
                CoreLocalMem<uint8_t>(cb_input_page),
                input_page_size,
                {.page_id = page_id + input},
                {});
        }
        page_id += inputs_per_cb_page;
        noc_obj.async_read_barrier();
        cb0.push_back(1);
    }
}
