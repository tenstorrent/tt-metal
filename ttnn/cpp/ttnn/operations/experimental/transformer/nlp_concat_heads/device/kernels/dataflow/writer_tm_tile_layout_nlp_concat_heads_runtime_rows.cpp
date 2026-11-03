// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drains the reader's (row, head) blocks to the interleaved output; the page range is derived from
// the same split as the reader's block range, so a hit only rewrites the common row count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/nlp_concat_heads_split.hpp"

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t h_tiles = get_arg(args::h_tiles);

    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr bool head_parallel_ok = get_arg(args::head_parallel_ok) != 0;

    const auto [start_block, end_block] =
        nlp_concat_heads_block_range(core_idx, h_tiles, batch, in0_c, num_cores, head_parallel_ok);
    const uint32_t start_id = start_block * in0_w_tiles;
    const uint32_t end_id = end_block * in0_w_tiles;

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
