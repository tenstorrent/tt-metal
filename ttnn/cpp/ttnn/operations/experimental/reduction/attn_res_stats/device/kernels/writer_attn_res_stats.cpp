// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t dots_page_offset = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    // runtime args
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_rows = get_arg_val<uint32_t>(1);
    const auto start_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_id_out);

    Noc noc;
    DataflowBuffer out_buf(cb_id_out);

    auto tensor_accessor = TensorAccessor(tensor_args, output_addr);

    // Compute packs the sum of squares first, then the dot. The two land a whole
    // candidate axis apart so that the pair arrives at the collective stacked.
    for (uint32_t r = start_row; r < start_row + num_rows; ++r) {
        out_buf.wait_front(onetile);
        noc.async_write(out_buf, tensor_accessor, output_tile_bytes, {.offset_bytes = 0}, {.page_id = r});
        noc.async_write_barrier();
        out_buf.pop_front(onetile);

        out_buf.wait_front(onetile);
        noc.async_write(
            out_buf, tensor_accessor, output_tile_bytes, {.offset_bytes = 0}, {.page_id = r + dots_page_offset});
        noc.async_write_barrier();
        out_buf.pop_front(onetile);
    }
}
