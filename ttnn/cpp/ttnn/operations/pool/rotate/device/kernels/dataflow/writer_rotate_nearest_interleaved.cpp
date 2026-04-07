// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Compile-time arguments
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_cb_pages = get_compile_time_arg_val(2);
    constexpr uint32_t burst_size = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_addr, output_stick_nbytes);

    experimental::CB output_cb(output_cb_id);
    experimental::Noc noc;

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks;) {
        uint32_t sticks_this_burst =
            (num_sticks - local_stick_idx) < burst_size ? (num_sticks - local_stick_idx) : burst_size;
        output_cb.wait_front(sticks_this_burst);
        uint32_t read_offset = 0;

        for (uint32_t i = 0; i < sticks_this_burst; i++, local_stick_idx++) {
            const uint32_t global_stick_idx = start_stick_id + local_stick_idx;
            noc.async_write(
                output_cb,
                output_tensor_accessor,
                output_stick_nbytes,
                {.offset_bytes = read_offset},
                {.page_id = global_stick_idx});
            read_offset += output_stick_nbytes;
        }
        noc.async_write_barrier();
        output_cb.pop_front(sticks_this_burst);
    }
}
