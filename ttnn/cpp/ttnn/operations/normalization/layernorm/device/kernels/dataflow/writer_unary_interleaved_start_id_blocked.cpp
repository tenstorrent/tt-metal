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

    constexpr auto blk = get_arg(args::block_size);  // needed for correctness of softmax/LN kernels
    constexpr uint32_t onetile = 1;

    const Noc noc;
    DataflowBuffer dfb_out0(dfb::out);

    const uint32_t tile_bytes = dfb_out0.get_tile_size();

    const auto s = TensorAccessor(tensor::dst);

#ifdef RESIDUAL_OUT
    // Fused residual add with two outputs: h = a + b arrives in dfb_h_out. Compute packs a whole row
    // of h before any normalized tile of that row, so each row drains h first, then out.
    DataflowBuffer dfb_h_out(dfb::h_out);
    const uint32_t h_tile_bytes = dfb_h_out.get_tile_size();
    const auto s_h = TensorAccessor(tensor::dst_h);
#endif

    uint32_t tile_id = tile_offset;
    for (uint32_t h = 0; h < num_tile_rows; h++) {
#ifdef RESIDUAL_OUT
        uint32_t h_tile_id = tile_id;
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_h_out.wait_front(static_cast<uint16_t>(block.full_block_size()));
            uint32_t h_idx = 0;
            for (auto i : block.local()) {
                noc.async_write(
                    dfb_h_out, s_h, h_tile_bytes, {.offset_bytes = h_idx * h_tile_bytes}, {.page_id = h_tile_id});
                h_tile_id++;
                h_idx++;
            }
            noc.async_write_barrier();
            dfb_h_out.pop_front(static_cast<uint16_t>(block.full_block_size()));
        }
#endif
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_out0.wait_front(static_cast<uint16_t>(block.full_block_size()));
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(dfb_out0, s, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
                tile_id++;
                idx++;
            }
            noc.async_write_barrier();
            dfb_out0.pop_front(static_cast<uint16_t>(block.full_block_size()));
        }
    }
}
