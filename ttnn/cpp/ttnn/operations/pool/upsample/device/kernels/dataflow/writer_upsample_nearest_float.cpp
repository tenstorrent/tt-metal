// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

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

    experimental::CB out_cb(cb_id_out);
    experimental::Noc noc;

    // Process sticks assigned to this core
    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        // Wait for data in CB
        out_cb.wait_front(1);

        // Write to output DRAM
        noc.async_write(out_cb, output_tensor_accessor, aligned_stick_nbytes, {}, {.page_id = stick_id});

        // Wait for write to complete
        noc.async_write_barrier();

        // Pop from CB
        out_cb.pop_front(1);

        stick_id++;
    }
}
