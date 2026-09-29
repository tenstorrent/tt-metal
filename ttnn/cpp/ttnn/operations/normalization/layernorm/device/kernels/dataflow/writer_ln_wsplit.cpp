// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width-split interleaved RMSNorm writer (LayerNormDefaultProgramConfig.width_split > 1; this core's Wt tiles of one
// tile row start at tile writer_start): reads the residual b (FUSE_PRE_ADD) while the reader fetches a on the other
// NoC, then drains h = a + b (RESIDUAL_OUT) and the second half of the output blocks (the reader drains the first
// half).
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "wsplit_dataflow.h"
#include <tt-metalium/constants.hpp>
namespace generic = norm::kernel_util::generic;

void kernel_main() {
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t tile_offset = get_arg(args::writer_start);
    constexpr auto blk = get_arg(args::block_size);

    const Noc noc;

#ifdef FUSE_PRE_ADD
    {
        DataflowBuffer dfb_inb(dfb::inb);
        const uint32_t b_tile_bytes = dfb_inb.get_tile_size();
        const auto src_b = TensorAccessor(tensor::src_b);
        uint32_t b_tile_id = tile_offset;
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_inb.reserve_back(static_cast<uint16_t>(block.full_block_size()));
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_read(
                    src_b, dfb_inb, b_tile_bytes, {.page_id = b_tile_id}, {.offset_bytes = idx * b_tile_bytes});
                b_tile_id++;
                idx++;
            }
            noc.async_read_barrier();
            dfb_inb.push_back(static_cast<uint16_t>(block.full_block_size()));
        }
    }
#endif

#ifdef RESIDUAL_OUT
    {
        DataflowBuffer dfb_h_out(dfb::h_out);
        norm::layernorm::wsplit::drain_blocks(
            noc, dfb_h_out, TensorAccessor(tensor::dst_h), tile_offset, Wt, blk, 0, Wt);
    }
#endif
    {
        DataflowBuffer dfb_out0(dfb::out);
        norm::layernorm::wsplit::drain_blocks(
            noc,
            dfb_out0,
            TensorAccessor(tensor::dst),
            tile_offset,
            Wt,
            blk,
            norm::layernorm::wsplit::reader_out_blocks(Wt, blk),
            Wt);
    }
    noc.async_write_barrier();
}
