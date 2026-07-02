// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_sharded_blocks.cpp. Block-by-block reader for sharded inputs (L1 or
// DRAM, used for uneven/DRAM sharding). Identical dataflow logic; the CB index becomes a dfb:: binding,
// tiles_per_block a named compile-time arg, the source TensorAccessor a tensor:: binding (the src_addr
// runtime arg is gone), and start_shard_id / num_blocks become named runtime args. The legacy copy is
// retained for the not-yet-ported sharded untilize factories.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_shard_id = get_arg(args::start_shard_id);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    const auto accessor_src = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t block_size_bytes = tiles_per_block * cb_in.get_entry_size();

    auto shard_pages = accessor_src.shard_pages(start_shard_id);
    auto page_iter = shard_pages.begin();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_in.reserve_back(tiles_per_block);
        // *page_iter is a tensor_accessor::Page; its noc_traits_t specialization resolves the
        // source NoC address, so a single block (tiles_per_block contiguous pages) is read at once.
        noc.async_read(*page_iter, cb_in, block_size_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
        page_iter += tiles_per_block;
        noc.async_read_barrier();
        cb_in.push_back(tiles_per_block);
    }
}
