// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp. Drains a
// row-major shard out of a DFB and writes it stick-by-stick into an interleaved output tensor. Only the
// plumbing changes: the buffer-index compile-time arg becomes dfb::out, the accessor-args /
// base-address pair becomes the tensor::dst binding, and the positional runtime args become named ones.
// The transfer loop — including the per-write destination offset — is untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.

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

    // The accessor base must stay the unshifted buffer base: Metal 2.0 supplies it from the tensor
    // binding and offers no seam for a pre-offset base. The per-core column shift rides each write
    // as a destination `offset_bytes` instead, which resolves to the same NoC address.
    const auto s0 = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb::out — this core's row-major shard, already resident in L1 (the reader hands over the
    // pages of the borrowed input buffer; no format conversion happens on the row-major path)
    DataflowBuffer dfb_out(dfb::out);

    uint32_t stick_id = start_id;
    dfb_out.wait_front(block_height);
    uint32_t dfb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            dfb_out,
            s0,
            block_width_bytes,
            {.offset_bytes = dfb_read_offset},
            {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});
        stick_id++;
        dfb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    dfb_out.pop_front(block_height);
}
