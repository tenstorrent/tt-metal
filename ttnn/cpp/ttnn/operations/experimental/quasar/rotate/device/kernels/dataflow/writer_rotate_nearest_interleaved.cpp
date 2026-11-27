// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime arguments
    uint32_t num_sticks = get_arg(args::num_sticks);
    uint32_t start_stick_id = get_arg(args::start_stick_id);

    // Compile-time arguments
    constexpr uint32_t output_stick_nbytes = get_arg(args::output_stick_nbytes);
    constexpr uint32_t num_dfb_pages = get_arg(args::num_dfb_pages);
    constexpr uint32_t burst_size = get_arg(args::burst_size);
    constexpr uint32_t dfb_page_size = get_arg(args::dfb_page_size);  // stride between sticks in the DFB
    const auto output_tensor_accessor = TensorAccessor(tensor::output);

    DataflowBuffer output_dfb(dfb::out);
    Noc noc;

    for (uint32_t local_stick_idx = 0; local_stick_idx < num_sticks;) {
        uint32_t sticks_this_burst =
            (num_sticks - local_stick_idx) < burst_size ? (num_sticks - local_stick_idx) : burst_size;
        output_dfb.wait_front(sticks_this_burst);
        uint32_t read_offset = 0;

        for (uint32_t i = 0; i < sticks_this_burst; i++, local_stick_idx++) {
            const uint32_t global_stick_idx = start_stick_id + local_stick_idx;
            noc.async_write(
                output_dfb,
                output_tensor_accessor,
                output_stick_nbytes,
                {.offset_bytes = read_offset},
                {.page_id = global_stick_idx});
            read_offset += dfb_page_size;
        }
        noc.async_write_barrier();
        output_dfb.pop_front(sticks_this_burst);
    }
}
