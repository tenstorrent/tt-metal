// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/debug/dprint.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t num_tensors = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t num_pages_per_core = get_arg_val<uint32_t>(2);
    const uint32_t num_tensors_times_rows_per_shard = get_arg_val<uint32_t>(3);
    const uint32_t num_pages_per_tensor = get_arg_val<uint32_t>(4);

    uint32_t arg_index = 5;

    const auto s = TensorAccessor(dst_args, dst_addr);
    experimental::Noc noc;

    for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
        const uint32_t input_shard_cb_id = get_arg_val<uint32_t>(arg_index++);
        experimental::CircularBuffer input_shard_cb(input_shard_cb_id);
        input_shard_cb.wait_front(num_pages_per_tensor);
        uint32_t page_id = 0;
        for (uint32_t page_id_input = 0; page_id_input < num_pages_per_tensor; page_id_input++) {
            uint32_t input_page_id = page_id + num_pages_per_core * core_id * num_tensors + tensor_id;
            noc.async_write(
                input_shard_cb,
                s,
                stick_size,
                {.offset_bytes = page_id_input * stick_size},
                {.page_id = input_page_id});
            noc.async_write_barrier();
            page_id += num_tensors;
        }
        input_shard_cb.pop_front(num_pages_per_tensor);
    }
}
