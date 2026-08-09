// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr auto tensor_args = TensorAccessorArgs<0>();

    // runtime args
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer cb_out_obj(cb_id_out);

    const uint32_t output_tile_bytes = get_tile_size(cb_id_out);
    auto tensor_accessor = TensorAccessor(tensor_args, output_addr);

    // Output tile index is the loop counter itself: the output has the shape of
    // the full-width operands, so the page order matches the order compute
    // produces.
    for (uint32_t i = start_id; i < start_id + num_output_tiles; ++i) {
        cb_out_obj.wait_front(onetile);
        noc.async_write(cb_out_obj, tensor_accessor, output_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_out_obj.pop_front(onetile);
    }
}
