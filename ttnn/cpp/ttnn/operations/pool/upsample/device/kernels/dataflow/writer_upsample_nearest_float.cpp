// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <uint32_t aligned_stick_nbytes>
TT_KERNEL void writer(uint32_t num_sticks, uint32_t start_stick_id) {
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
