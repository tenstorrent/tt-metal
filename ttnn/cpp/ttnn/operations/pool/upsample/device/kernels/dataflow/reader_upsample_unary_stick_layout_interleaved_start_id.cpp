// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <uint32_t aligned_input_unit_size>
TT_KERNEL void reader(uint32_t num_pages, uint32_t start_page_id) {
    const auto s0 = TensorAccessor(tensor::input);

    DataflowBuffer in_dfb(dfb::in0);
    Noc noc;

    const uint32_t end_id = start_page_id + num_pages;

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        in_dfb.reserve_back(1);

        noc.async_read(s0, in_dfb, aligned_input_unit_size, {.page_id = i}, {});

        noc.async_read_barrier();

        in_dfb.push_back(1);
    }
}
