// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_to_write = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t ntiles_c = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);

    experimental::CB out_cb(cb_id_out0);
    experimental::Noc noc;

    uint32_t end_stick_id = start_stick_id + num_sticks_to_write;

    // For grid sample: output is row major, each stick is written directly
    // We wait for ntiles_c pages to accumulate one full output stick
    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
        {
            // Wait for ntiles_c pages in output CB (one full stick)
            out_cb.wait_front(ntiles_c);

            // Write the complete stick
            noc.async_write(out_cb, s0, output_stick_size, {}, {.page_id = stick_id});

            noc.async_write_barrier();

            // Pop the ntiles_c pages we just consumed
            out_cb.pop_front(ntiles_c);
        }
    }
}
