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
    std::uint32_t N = get_arg(args::num_rows);
    std::uint32_t tile_offset = get_arg(args::tile_offset);
    std::uint32_t Wt = get_arg(args::Wt);

    constexpr auto dfb_id_out = dfb::out;
    constexpr std::uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb_id_out);
    std::uint32_t tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t tile_id = tile_offset;
    for (std::uint32_t i = 0; i < N; i++) {
        dfb_out_obj.wait_front(Wt);
        for (std::uint32_t w = 0; w < Wt; w++) {
            noc.async_write(dfb_out_obj, s, tile_bytes, {.offset_bytes = w * tile_bytes}, {.page_id = tile_id});
            tile_id++;
        }
        noc.async_write_barrier();
        dfb_out_obj.pop_front(Wt);
    }
}
