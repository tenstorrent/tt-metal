// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    experimental::CircularBuffer dst_cb(dst_cb_id);
    experimental::Noc noc;
    const auto s = TensorAccessor(dst_args, output_buffer_address);
    const uint32_t tile_bytes = get_tile_size(dst_cb_id);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dst_cb.wait_front(1);
        noc.async_write(dst_cb, s, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dst_cb.pop_front(1);
    }
}
