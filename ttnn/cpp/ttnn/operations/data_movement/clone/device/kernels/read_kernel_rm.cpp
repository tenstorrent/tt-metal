// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer src_dfb(dfb::src);
    Noc noc;
    const auto s = TensorAccessor(tensor::input);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        src_dfb.reserve_back(1);
        noc.async_read(s, src_dfb, stick_size, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        src_dfb.push_back(1);
    }
}
