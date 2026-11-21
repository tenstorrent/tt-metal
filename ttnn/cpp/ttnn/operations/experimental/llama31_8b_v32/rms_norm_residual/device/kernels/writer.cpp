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

    constexpr auto blk = get_arg(args::block_size);
    constexpr uint32_t onetile = 1;

    const Noc noc;
    DataflowBuffer dfb_out0(dfb::out);

    const uint32_t tile_bytes = dfb_out0.get_tile_size();

    const auto s = TensorAccessor(tensor::dst);

    uint32_t tile_id = tile_offset;
    DataflowBuffer dfb_hout(dfb::hout);
    const auto s_h = TensorAccessor(tensor::srch);
    const uint32_t h_bytes = dfb_hout.get_tile_size();
    for (uint32_t h = 0; h < num_tile_rows; h++) {
        {
            uint32_t ht = tile_id;
            for (auto block : generic::blocks(Wt, blk)) {
                dfb_hout.wait_front(static_cast<uint16_t>(block.full_block_size()));
                uint32_t hi = 0;
                for (auto i : block.local()) {
                    (void)i;
                    noc.async_write(dfb_hout, s_h, h_bytes, {.offset_bytes = hi * h_bytes}, {.page_id = ht});
                    ht++;
                    hi++;
                }
                noc.async_writes_flushed();
                dfb_hout.pop_front(static_cast<uint16_t>(block.full_block_size()));
            }
        }
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_out0.wait_front(static_cast<uint16_t>(block.full_block_size()));
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(dfb_out0, s, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
                tile_id++;
                idx++;
            }
            noc.async_writes_flushed();
            dfb_out0.pop_front(static_cast<uint16_t>(block.full_block_size()));
        }
    }
    noc.async_write_barrier();
}
