// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of reader_unary_nd_sharded_blocks.cpp, which lives beside it. Ops
// ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until
// the last of them migrates and the original is retired, changes here likely belong there too.
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
    const auto start_shard_id = get_arg(args::start_shard_id);

    // compile-time args
    constexpr auto num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr auto num_shards = get_arg(args::num_shards);
    constexpr auto num_cores = get_arg(args::num_cores);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);
    const uint32_t tile_size_bytes = dfb_in.get_tile_size();

    const auto accessor_src = TensorAccessor(tensor::src);
    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_src.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end();
             page_iter += num_tiles_per_input_block) {
            dfb_in.reserve_back(num_tiles_per_input_block);
            noc.async_read(
                accessor_src,
                dfb_in,
                tile_size_bytes * num_tiles_per_input_block,
                {.page_id = page_iter->page_id(), .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in.push_back(num_tiles_per_input_block);
        }
    }
}
