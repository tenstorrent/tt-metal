// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);  // Output tensor DRAM address
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);          // Number of output sticks for this core
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);      // Starting output stick ID

    // Compile-time arguments
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);             // Output CB ID
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(1);  // Aligned stick size in bytes

    // Tensor accessor compile-time args start at index 3
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_buffer_addr, aligned_stick_nbytes);

    // Process sticks assigned to this core
    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        // Wait for data in CB
        cb_wait_front(cb_id_out, 1);

        // Get L1 read address
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write to output DRAM
        uint64_t dst_noc_addr = output_tensor_accessor.get_noc_addr(stick_id);
        noc_async_write(l1_read_addr, dst_noc_addr, aligned_stick_nbytes);

        // Wait for write to complete
        noc_async_write_barrier();

        // Pop from CB
        cb_pop_front(cb_id_out, 1);

        stick_id++;
    }
}
