// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the tilized-output embedding program with the interleaved output: drains the tiles of
// this core's blocks, whose range follows the reader's split from the common row count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/embedding/device/kernels/embeddings_fused_split.hpp"

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t blocks_per_batch = get_arg(args::blocks_per_batch);

    constexpr uint32_t num_tiles_per_block = get_arg(args::num_tiles_per_block);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const auto [start_block, end_block] = embeddings_fused_block_range(core_idx, blocks_per_batch, batch, num_cores);
    const uint32_t start_id = start_block * num_tiles_per_block;
    const uint32_t end_id = end_block * num_tiles_per_block;

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const uint32_t page_bytes = dfb.get_entry_size();
    const auto s = TensorAccessor(tensor::dst);

    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb.wait_front(1);
        noc.async_write(dfb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
