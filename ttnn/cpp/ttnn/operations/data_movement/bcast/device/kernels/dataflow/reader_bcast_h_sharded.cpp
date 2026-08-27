// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    std::uint32_t offset = get_arg(args::offset);
    auto NC = get_arg(args::NC);
    auto batch_offset = get_arg(args::batch_offset);

    constexpr std::uint32_t onetile = 1;

    // src1 (input_b) base address + layout arrive via the tensor binding (tensor::src1).
    const auto s1 = TensorAccessor(tensor::src1);

    Noc noc;
    // dfb::in0 borrows the resident input_a shard; dfb::in1 is the input_b FIFO.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const std::uint32_t tile_bytes_1 = dfb_in1.get_tile_size();

    dfb_in0.push_back(Ht * Wt);
    for (std::uint32_t ht = 0; ht < Ht; ht++) {
        for (std::uint32_t wt = 0; wt < Wt; wt++) {
            dfb_in1.reserve_back(onetile);
            noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = offset, .offset_bytes = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in1.push_back(onetile);
            offset++;
        }

        offset -= Wt;
        if (ht % NC == (NC - 1)) {
            offset += batch_offset;
        }
    }
}
