// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <uint32_t output_stick_size, uint32_t ntiles_c, typename TensorAccessorType>
inline __attribute__((always_inline)) void write_pool_sticks(
    const TensorAccessorType& s0, uint32_t cb_id_out0, uint32_t num_sticks_to_write, uint32_t start_stick_id) {
    DataflowBuffer out_dfb(cb_id_out0);
    Noc noc;

    uint32_t end_stick_id = start_stick_id + num_sticks_to_write;

    // For grid sample: output is row major, each stick is written directly
    // We wait for ntiles_c pages to accumulate one full output stick
    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
        {
            // Wait for ntiles_c pages in output CB (one full stick)
            out_dfb.wait_front(ntiles_c);

            // Write the complete stick
            noc.async_write(out_dfb, s0, output_stick_size, {}, {.page_id = stick_id});

            noc.async_write_barrier();

            // Pop the ntiles_c pages we just consumed
            out_dfb.pop_front(ntiles_c);
        }
    }
}
