// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);
    // page_size (NoC read size) is shape-derived, passed as a runtime arg to keep the binary
    // resolution-invariant.
    uint32_t page_size = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(src_args, src_addr);

    DataflowBuffer in_dfb(cb_id_in0);
    Noc noc;

    const uint32_t end_id = start_page_id + num_pages;

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        in_dfb.reserve_back(1);

        noc.async_read(s0, in_dfb, page_size, {.page_id = i}, {});

        noc.async_read_barrier();

        in_dfb.push_back(1);
    }
}
