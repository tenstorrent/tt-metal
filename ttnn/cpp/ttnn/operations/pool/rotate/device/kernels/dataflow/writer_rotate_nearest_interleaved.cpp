// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Compile-time arguments
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_cb_pages = get_compile_time_arg_val(2);
    constexpr uint32_t batch_size = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_addr, output_stick_nbytes);

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks;) {
        uint32_t sticks_this_batch =
            (num_sticks - local_stick_idx) < batch_size ? (num_sticks - local_stick_idx) : batch_size;
        cb_wait_front(output_cb_index, sticks_this_batch);
        uint32_t l1_read_addr = get_read_ptr(output_cb_index);

        for (uint32_t i = 0; i < sticks_this_batch; i++, local_stick_idx++) {
            const uint32_t global_stick_idx = start_stick_id + local_stick_idx;
            const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(global_stick_idx);
            noc_async_write(l1_read_addr, output_noc_addr, output_stick_nbytes);
            l1_read_addr += output_stick_nbytes;
        }
        noc_async_write_barrier();
        cb_pop_front(output_cb_index, sticks_this_batch);
    }
}
