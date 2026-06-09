// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads num_tiles tiles from 1 input tensor and pushes them to CB c_0.
// Page bytes come from the CB's configured page size so dtype is data-driven.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb0 = 0;
    constexpr auto src0_args = TensorAccessorArgs<0>();

    const uint32_t bytes0 = get_local_cb_interface(cb0).fifo_page_size;
    const auto s0 = TensorAccessor(src0_args, src0_addr);

    Noc noc;
    CircularBuffer c0(cb0);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        c0.reserve_back(onetile);
        noc.async_read(s0, c0, bytes0, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        c0.push_back(onetile);
    }
}
