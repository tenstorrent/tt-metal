// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
namespace generic = norm::kernel_util::generic;

void kernel_main() {
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);
    const uint32_t tile_offset = get_arg(args::writer_start);
    const uint32_t row_skip = get_arg(args::row_skip);  // output tiles past this op's column window, per row

    constexpr auto blk = get_arg(args::block_size);  // needed for correctness of softmax/LN kernels
    constexpr uint32_t onetile = 1;

    const Noc noc;
    DataflowBuffer dfb_out0(dfb::out);

    const uint32_t tile_bytes = dfb_out0.get_tile_size();

    const auto s = TensorAccessor(tensor::dst);
#ifdef EXTRA_DSTS
    // more destinations with dst's layout: every tile goes to each of them too
    const auto s1 = TensorAccessor(tensor::dst_x1);
#if EXTRA_DSTS > 1
    const auto s2 = TensorAccessor(tensor::dst_x2);
#endif
#endif

    uint32_t tile_id = tile_offset;
    for (uint32_t h = 0; h < num_tile_rows; h++) {
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_out0.wait_front(static_cast<uint16_t>(block.full_block_size()));
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(dfb_out0, s, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
#ifdef EXTRA_DSTS
                noc.async_write(dfb_out0, s1, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
#if EXTRA_DSTS > 1
                noc.async_write(dfb_out0, s2, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
#endif
#endif
                tile_id++;
                idx++;
            }
            noc.async_write_barrier();
            dfb_out0.pop_front(static_cast<uint16_t>(block.full_block_size()));
        }
        tile_id += row_skip;
    }
}
