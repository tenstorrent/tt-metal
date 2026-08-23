// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <uint32_t output_stick_size, uint32_t ntiles_c, typename TensorAccessorType>
FORCE_INLINE void write_pool_sticks(
    const TensorAccessorType& output, uint32_t cb_id_out0, uint32_t num_sticks_to_write, uint32_t start_stick_id) {
    DataflowBuffer out_dfb(cb_id_out0);
    Noc noc;

    uint32_t end_stick_id = start_stick_id + num_sticks_to_write;

    // Pool-family outputs are row major, so each complete stick is written directly.
    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
        // Wait for ntiles_c pages to accumulate one full output stick.
        out_dfb.wait_front(ntiles_c);

        noc.async_write(out_dfb, output, output_stick_size, {}, {.page_id = stick_id});
        noc.async_write_barrier();

        out_dfb.pop_front(ntiles_c);
    }
}
