// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Runtime arguments
    const auto num_sticks = get_arg(args::num_sticks);          // Number of output sticks for this core
    const auto start_stick_id = get_arg(args::start_stick_id);  // Starting output stick ID

    // Compile-time arguments
    constexpr auto aligned_stick_nbytes = get_arg(args::aligned_stick_nbytes);  // Aligned stick size in bytes

    const auto output_tensor_accessor = TensorAccessor(tensor::output);

    DataflowBuffer out_dfb(dfb::out);
    Noc noc;

    // Process sticks assigned to this core
    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        // Wait for data in CB
        out_dfb.wait_front(1);

        // Write to output DRAM
        noc.async_write(out_dfb, output_tensor_accessor, aligned_stick_nbytes, {}, {.page_id = stick_id});

        // Wait for write to complete
        noc.async_write_barrier();

        // Pop from CB
        out_dfb.pop_front(1);

        stick_id++;
    }
}
