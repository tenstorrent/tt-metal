// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height = get_arg_val<uint32_t>(1);
    const uint32_t block_width_bytes = get_arg_val<uint32_t>(2);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    const uint32_t output_width_in_pages = get_arg_val<uint32_t>(5);

    // compile-time args
    constexpr uint32_t dfb_id_out0 = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out(dfb_id_out0);

    uint32_t stick_id = start_id;
    dfb_out.wait_front(block_height);
    uint32_t cb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            dfb_out, s, block_width_bytes, {.offset_bytes = cb_read_offset}, {.page_id = stick_id, .offset_bytes = 0});
        stick_id += output_width_in_pages;
        cb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    dfb_out.pop_front(block_height);
}
