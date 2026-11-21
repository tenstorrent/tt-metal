// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// upsample writer: block operation `replicate_store_block` (op_design.md, Block schedule).
//
// No kernel-library helper writes one L1 stick to several interleaved row-major pages (the design's "Helpers
// considered and rejected" table): raw Device-2.0 NoC + DFB API, as in the Quasar example writer (explicit
// sync).
//
// Writer thread t of W takes blocks t, t+W, ... of this cluster's range. Input stick s (global row q = s / in_w,
// column w = s % in_w) goes to the output pages
//     out_page(s, a, b) = ((q * scale_h + a) * in_w + w) * scale_w + b,   0 <= a < scale_h, 0 <= b < scale_w
// (batch folds into q because n*H*scale_h + h*scale_h = q*scale_h). All block_extent*scale_h*scale_w writes
// of a block come from the one resident entry; one barrier per block before the entry is popped.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_stick = get_arg(args::start_stick);
    const uint32_t core_sticks = get_arg(args::core_sticks);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t block_sticks = get_arg(args::block_sticks);
    constexpr uint32_t stick_bytes = get_arg(args::stick_bytes);
    constexpr uint32_t stick_stride = get_arg(args::stick_stride);
    constexpr uint32_t scale_h = get_arg(args::scale_h);
    constexpr uint32_t scale_w = get_arg(args::scale_w);
    constexpr uint32_t in_w = get_arg(args::in_w);
    constexpr uint32_t thread_mask = get_arg(args::writer_thread_mask);  // test probe; all ones by default
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer ring(dfb::cb_input_sticks);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);

    const uint32_t thread_id = get_my_thread_id();
    const bool moves_bytes = (thread_mask >> thread_id) & 1u;
    const uint32_t num_blocks = (core_sticks + block_sticks - 1) / block_sticks;

    for (uint32_t block = thread_id; block < num_blocks; block += W) {
        const uint32_t first = block * block_sticks;
        const uint32_t remaining = core_sticks - first;
        const uint32_t extent = remaining < block_sticks ? remaining : block_sticks;  // ragged last block
        const uint32_t s0 = start_stick + first;
        const uint32_t q0 = s0 / in_w;  // one division per block; stepped per stick below
        const uint32_t w0 = s0 - q0 * in_w;

        ring.wait_front(1);
        if (moves_bytes) {
            for (uint32_t a = 0; a < scale_h; ++a) {
                uint32_t q = q0;
                uint32_t w = w0;
                for (uint32_t j = 0; j < extent; ++j) {
                    const uint32_t page = ((q * scale_h + a) * in_w + w) * scale_w;
                    const uint32_t offset = j * stick_stride;
                    for (uint32_t b = 0; b < scale_w; ++b) {
                        noc.async_write<NocOptions::TXN_ID>(
                            ring, acc_out, stick_bytes, {.offset_bytes = offset}, {.page_id = page + b}, user_txn);
                    }
                    if (++w == in_w) {
                        w = 0;
                        ++q;
                    }
                }
            }
            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
        }
        ring.pop_front(1);
    }
    noc_async_write_set_trid(0, noc.get_noc_id());
}
