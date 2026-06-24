// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + num_tiles;

    constexpr auto input_args = TensorAccessorArgs<1>();
    const auto input_addrg = TensorAccessor(input_args, input_addr);

    const uint32_t page_bytes = get_local_cb_interface(in_cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb_in(in_cb_id);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_in.reserve_back(1);
        noc.async_read(input_addrg, cb_in, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
