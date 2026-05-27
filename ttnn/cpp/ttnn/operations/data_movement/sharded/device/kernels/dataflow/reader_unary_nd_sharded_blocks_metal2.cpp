// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_nd_sharded_blocks.cpp.
//
// Reads ND-sharded input blocks via TensorAccessor.shard_pages() and pushes
// them tile-by-tile into the producer DFB.
//
// Bindings:
//   dfb::input                       — DFB endpoint (PRODUCER)
//   ta::input                        — TensorAccessor (ND-sharded input)
//   args::num_tiles_per_input_block  — CTA
//   args::num_shards                 — CTA
//   args::num_cores                  — CTA
//   args::start_shard_id             — RTA

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr auto num_shards = get_arg(args::num_shards);
    constexpr auto num_cores = get_arg(args::num_cores);

    auto start_shard_id = get_arg(args::start_shard_id);

    const uint32_t tile_size_bytes = get_tile_size(dfb::input);

    Noc noc;
    DataflowBuffer cb_in(dfb::input);

    const auto accessor_src = TensorAccessor(ta::input);
    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_src.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end();
             page_iter += num_tiles_per_input_block) {
            cb_in.reserve_back(num_tiles_per_input_block);
            noc.async_read(
                accessor_src,
                cb_in,
                tile_size_bytes * num_tiles_per_input_block,
                {.page_id = page_iter->page_id(), .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in.push_back(num_tiles_per_input_block);
        }
    }
}
