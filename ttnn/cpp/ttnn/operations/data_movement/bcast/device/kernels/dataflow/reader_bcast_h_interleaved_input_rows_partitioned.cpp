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
    auto src0_num_tiles = get_arg(args::src0_num_tiles);
    auto NCHtWt = get_arg(args::NCHtWt);
    auto NC = get_arg(args::NC);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    auto nc1 = get_arg(args::nc1);
    auto start_id = get_arg(args::start_id);
    auto HtWt = get_arg(args::HtWt);

    constexpr std::uint32_t onetile = 1;

    // src0 / src1 base addresses + layout arrive via the tensor bindings (tensor::src0 / tensor::src1).
    const auto s0 = TensorAccessor(tensor::src0);
    const auto s1 = TensorAccessor(tensor::src1);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const std::uint32_t tile_bytes_0 = dfb_in0.get_tile_size();
    const std::uint32_t tile_bytes_1 = dfb_in1.get_tile_size();

    std::uint32_t num_tiles = src0_num_tiles;
    std::uint32_t i = 0;
    std::uint32_t i1 = 0;
    std::uint32_t i_nc = 0;
    for (std::uint32_t nc = 0; nc < NC; nc++) {
        i = i_nc + start_id;
        for (std::uint32_t ht = 0; ht < Ht; ht++) {
            for (std::uint32_t wt = 0; wt < Wt; wt++) {
                dfb_in0.reserve_back(onetile);
                noc.async_read(s0, dfb_in0, tile_bytes_0, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in0.push_back(onetile);

                dfb_in1.reserve_back(onetile);
                noc.async_read(s1, dfb_in1, tile_bytes_1, {.page_id = i1, .offset_bytes = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in1.push_back(onetile);
                i1++;
                i++;
            }

            i1 -= Wt;
        }
        if (nc1 == 0) {
            i1 += Wt;
        }
        i_nc += HtWt;
    }
}
