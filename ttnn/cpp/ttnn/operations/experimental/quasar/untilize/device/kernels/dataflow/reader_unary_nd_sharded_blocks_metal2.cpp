// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_nd_sharded_blocks.cpp. Block-by-block reader for nd-sharded input.
// Identical dataflow; the CB index becomes a dfb:: binding, the source TensorAccessor a tensor::
// binding (src_addr runtime arg gone), and the remaining args are named. The legacy copy is retained
// for the not-yet-ported nd-shard factories.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_shard_id = get_arg(args::start_shard_id);

    constexpr uint32_t num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr uint32_t num_shards = get_arg(args::num_shards);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const auto accessor_src = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t tile_size_bytes = cb_in.get_entry_size();

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
