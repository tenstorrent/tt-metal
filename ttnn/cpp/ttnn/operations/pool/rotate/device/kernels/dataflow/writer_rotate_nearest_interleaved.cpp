// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Compile-time arguments
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_nbytes = get_compile_time_arg_val(1);

    // Tensor accessor for output tensor (starts at compile-time arg index 2)
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_addr, output_stick_nbytes);

    // Write each output stick from CB to DRAM
    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks; local_stick_idx++) {
        const uint32_t global_stick_idx = start_stick_id + local_stick_idx;

        // Wait for data in CB from reader
        cb_wait_front(output_cb_index, 1);
        uint32_t l1_read_addr = get_read_ptr(output_cb_index);

        // Write to output tensor
        const uint32_t output_stick_index = global_stick_idx;
        const uint64_t output_noc_addr = output_tensor_accessor.get_noc_addr(output_stick_index);
        noc_async_write(l1_read_addr, output_noc_addr, output_stick_nbytes);
        noc_async_write_barrier();

        // Free CB space
        cb_pop_front(output_cb_index, 1);
    }
}
