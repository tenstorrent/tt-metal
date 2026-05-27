// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp.
//
// Consumes a block of sharded sticks from a DFB and writes them stick-by-stick to
// the interleaved row-major output via a TensorAccessor with input-width offset.
//
// Bindings (named, from host KernelSpec):
//   dfb::out                       — DFB endpoint (CONSUMER) — backed by SRC_DFB or OUT_DFB
//   ta::out                        — TensorAccessor (output, interleaved row-major)
//   args::block_height
//   args::block_width_bytes
//   args::padded_block_width_bytes
//   args::input_width_offset_bytes
//   args::start_id

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto block_height = get_arg(args::block_height);
    auto block_width_bytes = get_arg(args::block_width_bytes);
    auto padded_block_width_bytes = get_arg(args::padded_block_width_bytes);
    auto input_width_offset_bytes = get_arg(args::input_width_offset_bytes);
    auto start_id = get_arg(args::start_id);

    // The legacy plumbing added input_width_offset_bytes to the tensor's base
    // address. With named TensorBinding we can no longer adjust the base address
    // host-side, so we apply the offset to each page's offset_bytes instead.
    const auto s0 = TensorAccessor(ta::out);

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
