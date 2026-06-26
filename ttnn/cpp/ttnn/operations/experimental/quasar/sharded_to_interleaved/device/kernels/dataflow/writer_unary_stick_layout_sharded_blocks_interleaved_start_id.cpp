// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t block_width_bytes = get_arg(args::block_width_bytes);
    const uint32_t padded_block_width_bytes = get_arg(args::padded_block_width_bytes);
    const uint32_t input_width_offset_bytes = get_arg(args::input_width_offset_bytes);
    const uint32_t start_id = get_arg(args::start_id);

    // The destination-buffer base address is bound via the tensor parameter (tensor::dst),
    // replacing the legacy buffer-address RTA slot 0. The legacy writer pre-shifted the base
    // by `input_width_offset_bytes`; in the typed model that per-core byte shift becomes the
    // destination-side `offset_bytes` on each write.
    const auto s0 = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t stick_id = start_id;
    cb_out.wait_front(block_height);
    uint32_t cb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            cb_out,
            s0,
            block_width_bytes,
            {.offset_bytes = cb_read_offset},
            {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});
        stick_id++;
        cb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_height);
}
