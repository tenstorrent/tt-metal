// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of
// ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp
// Forked (not modified in place) because the legacy source is shared by ops that have not yet
// migrated to Metal 2.0 named bindings (untilize ND-shard factory). See METAL2_PORT_REPORT.md.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
// Stale cross-op include carried over from the legacy kernel; no symbol from it is used (sharded
// access goes through TensorAccessor.shard_pages). Left in place per METAL2_PORT_REPORT.md.
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const uint32_t start_shard_id = get_arg(args::start_shard_id);

    // compile-time args
    constexpr uint32_t num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr uint32_t num_shards = get_arg(args::num_shards);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t tile_size_bytes = cb_in.get_tile_size();

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
