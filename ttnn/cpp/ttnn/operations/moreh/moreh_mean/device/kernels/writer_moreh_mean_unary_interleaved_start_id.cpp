// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    constexpr uint32_t onetile = 1;

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onetile);
        noc.async_write(cb_out, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_out.pop_front(onetile);
    }
}
