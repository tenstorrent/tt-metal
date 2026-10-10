// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_start_id.cpp. Identical dataflow logic; the CB and input tensor
// are sourced from Metal 2.0 named bindings (dfb::in / tensor::input) and named args instead of a
// CB-index CTA, a buffer-address RTA, and TensorAccessorArgs plumbing. The legacy
// reader_unary_start_id.cpp is retained for the not-yet-ported multi_core factory; delete this
// fork's twin once both factories are on Metal 2.0.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_page_id = get_arg(args::start_page_id);
    constexpr uint32_t sub_block_tiles = get_arg(args::sub_block_tiles);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);

    const auto s = TensorAccessor(tensor::input);

    uint32_t end_page_id = start_page_id + num_tiles;
    // Thread t of N reads the sub-blocks t, t + N, ... of sub_block_tiles consecutive tiles, the unit
    // compute waits for: with several threads the DFB uses the ALL pattern, which gives each thread
    // its own contiguous region and lets compute take sub-blocks from the threads in turn.
    // Implicit sync: each TXN_ID read claims a DFB entry and posts its credit when it lands.
    const uint32_t sub_block_step = get_num_threads() * sub_block_tiles;
    for (uint32_t first = start_page_id + get_my_thread_id() * sub_block_tiles; first < end_page_id;
         first += sub_block_step) {
        for (uint32_t page_id = first; page_id < first + sub_block_tiles && page_id < end_page_id; ++page_id) {
            noc.async_read<NocOptions::TXN_ID>(s, cb_in, {.page_id = page_id, .offset_bytes = 0}, {});
        }
    }
}
