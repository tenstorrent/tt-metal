// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_sticks_to_write = get_arg(args::num_sticks_to_write);
    uint32_t start_stick_id = get_arg(args::start_stick_id);

    constexpr uint32_t output_stick_size = get_arg(args::output_stick_size);
    constexpr uint32_t ntiles_c = get_arg(args::ntiles_c);

    const auto s0 = TensorAccessor(tensor::output);

    DataflowBuffer out_dfb(dfb::out);
    Noc noc;

    uint32_t end_stick_id = start_stick_id + num_sticks_to_write;

    // For grid sample: output is row major, each stick is written directly
    // We wait for ntiles_c pages to accumulate one full output stick
    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
        {
            // Wait for ntiles_c pages in output DFB (one full stick)
            out_dfb.wait_front(ntiles_c);

            // Write the complete stick
            noc.async_write(out_dfb, s0, output_stick_size, {}, {.page_id = stick_id});

            noc.async_write_barrier();

            // Pop the ntiles_c pages we just consumed
            out_dfb.pop_front(ntiles_c);
        }
    }
}
