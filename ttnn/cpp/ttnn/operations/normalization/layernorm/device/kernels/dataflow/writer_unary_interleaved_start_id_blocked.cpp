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
    uint32_t Wt = get_arg(args::Wt);
    uint32_t num_tile_rows = get_arg(args::num_tile_rows);
    uint32_t tile_offset = get_arg(args::start_tile_row);

    constexpr uint32_t blk = get_arg(args::block_size);  // needed for correctness of softmax/LN kernels

    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(dfb::cb_out);

    Noc noc;
    DataflowBuffer cb_out0(dfb::cb_out);

    const auto s = TensorAccessor(ta::output);

    uint32_t tile_id = tile_offset;
    for (uint32_t h = 0; h < num_tile_rows; h++) {
        for (auto block : generic::blocks(Wt, blk)) {
            cb_out0.wait_front(block.full_block_size());
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(cb_out0, s, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
                tile_id++;
                idx++;
            }
            noc.async_write_barrier();
            cb_out0.pop_front(block.full_block_size());
        }
    }
}
