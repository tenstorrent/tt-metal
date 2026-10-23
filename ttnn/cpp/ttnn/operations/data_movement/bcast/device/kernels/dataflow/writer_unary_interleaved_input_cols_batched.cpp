// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // (legacy positional layout put Ht at index 3, to match the regular writer_unary arg order)
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    auto Wt_read = get_arg(args::Wt_read);
    auto Wt_skip = get_arg(args::Wt_skip);
    auto NC = get_arg(args::NC);
    auto HtWt = get_arg(args::HtWt);

    constexpr std::uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const std::uint32_t tile_bytes = dfb_out.get_tile_size();

    std::uint32_t tile_id = 0;
    std::uint32_t i_nc = 0;
    for (std::uint32_t nc = 0; nc < NC; nc++) {
        tile_id = i_nc + Wt_read;
        for (std::uint32_t i = 0; i < Ht; i++) {
            for (std::uint32_t j = 0; j < Wt; j++) {
                dfb_out.wait_front(onetile);
                noc.async_write(dfb_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id, .offset_bytes = 0});
                noc.async_write_barrier();
                dfb_out.pop_front(onetile);

                tile_id++;
            }
            tile_id += Wt_skip;
        }
        i_nc += HtWt;
    }
}
