// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

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

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);

    TensorAccessor tensor0_addrgen(tensor0_args, tensor_address0, input_page_size);

    for (uint32_t page_id = input_page_id_start; page_id < input_page_id_end;) {
        cb_reserve_back(cb0_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb0_id);
        // fill CB page
        uint32_t input = 0;
        for (; input < inputs_per_cb_page && page_id + input < input_page_id_end; input++) {
            auto noc_src_addr = tensor0_addrgen.get_noc_addr(page_id + input, 0);
            noc_async_read(noc_src_addr, l1_write_addr + input * input_page_size, input_page_size);
        }
        page_id += input;
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
    }
}
