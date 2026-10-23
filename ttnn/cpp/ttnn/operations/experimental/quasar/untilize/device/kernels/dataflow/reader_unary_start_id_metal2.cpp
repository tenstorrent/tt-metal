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
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);

    const uint32_t tile_bytes = cb_in.get_tile_size();

    const auto s = TensorAccessor(tensor::input);

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_in.reserve_back(1);
        noc.async_read(s, cb_in, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
