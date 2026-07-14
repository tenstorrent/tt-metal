// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t block_width_bytes = get_arg(args::block_width_bytes);
    const uint32_t padded_block_width_bytes = get_arg(args::padded_block_width_bytes);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t output_width_in_pages = get_arg(args::output_width_in_pages);

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t stick_id = start_id;
    cb_out.wait_front(block_height);
    uint32_t cb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            cb_out, s, block_width_bytes, {.offset_bytes = cb_read_offset}, {.page_id = stick_id, .offset_bytes = 0});
        stick_id += output_width_in_pages;
        cb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_height);
}
