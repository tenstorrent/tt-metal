// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded_stick_layout_start_id.cpp. Drains a row-major shard out of a
// DFB and writes it stick-by-stick into an interleaved output tensor. Only the plumbing changes: the
// buffer-index compile-time arg becomes dfb::out, the accessor-args / base-address pair becomes the
// tensor::dst binding, and the positional runtime args become named ones. The transfer loop is
// untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.
//
// The binding names below (dfb::out, tensor::dst) and the named argument set are this fork's interface:
// every later consumer inherits them, so they are taken from the kernel's own vocabulary rather than
// any one op's locals, and are not renamed once a consumer exists.

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
    DataflowBuffer dfb_out(dfb::out);

    uint32_t stick_id = start_id;
    dfb_out.wait_front(block_height);
    uint32_t dfb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            dfb_out, s, block_width_bytes, {.offset_bytes = dfb_read_offset}, {.page_id = stick_id, .offset_bytes = 0});
        stick_id += output_width_in_pages;
        dfb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    dfb_out.pop_front(block_height);
}
