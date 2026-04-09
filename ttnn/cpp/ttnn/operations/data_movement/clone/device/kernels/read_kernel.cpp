// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    experimental::CircularBuffer src_cb(src_cb_id);
    experimental::Noc noc;
    const auto s = TensorAccessor(src_args, input_buffer_address, get_tile_size(src_cb_id));
    const uint32_t tile_bytes = get_tile_size(src_cb_id);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        src_cb.reserve_back(1);
        noc.async_read(s, src_cb, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        src_cb.push_back(1);
    }
}
