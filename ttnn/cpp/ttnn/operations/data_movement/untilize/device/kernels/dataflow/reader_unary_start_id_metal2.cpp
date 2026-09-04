// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of reader_unary_start_id.cpp, which lives beside it. Ops ported to
// Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until the last
// of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in, tensor::src) and the named argument set are this fork's interface:
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
    const auto num_tiles = get_arg(args::num_tiles);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    const auto s = TensorAccessor(tensor::src);

    uint32_t end_page_id = start_id + num_tiles;
    for (uint32_t page_id = start_id; page_id < end_page_id; ++page_id) {
        dfb_in.reserve_back(1);
        noc.async_read(s, dfb_in, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in.push_back(1);
    }
}
