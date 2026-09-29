// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    std::uint32_t offset = get_arg(args::offset);
    auto batch_offset = get_arg(args::batch_offset);
    auto w_blk = get_arg(args::w_blk);
    auto batch_b = get_arg(args::batch_b);

    constexpr std::uint32_t onetile = 1;

    // src1 (input_b) base address + layout arrive via the tensor binding (tensor::src1).
    const auto s1 = TensorAccessor(tensor::src1);

    Noc noc;
    // dfb::in0 borrows the resident input_a shard; dfb::in1 is the input_b FIFO.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const std::uint32_t tile_bytes = dfb_in1.get_tile_size();

    dfb_in0.push_back(Ht * Wt);
    for (std::uint32_t b = 0; b < batch_b; b++) {
        for (std::uint32_t wt = 0; wt < Wt; wt += w_blk) {
            dfb_in1.reserve_back(w_blk);
            std::uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr();
            for (std::uint32_t r = 0; r < w_blk; r++) {
                CoreLocalMem<std::uint32_t> dst(l1_write_addr_in1);
                noc.async_read(
                    s1, dst, tile_bytes, {.page_id = offset + wt + r, .offset_bytes = 0}, {.offset_bytes = 0});
                l1_write_addr_in1 += tile_bytes;
            }
            noc.async_read_barrier();
            dfb_in1.push_back(w_blk);
        }
        offset += batch_offset;
    }
}
