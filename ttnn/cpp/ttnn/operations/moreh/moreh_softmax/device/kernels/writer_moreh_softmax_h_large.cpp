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
    std::uint32_t Ht = get_arg(args::Ht);
    std::uint32_t Wt = get_arg(args::Wt);

    constexpr auto dfb_id_out = dfb::out;
    constexpr std::uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb_id_out);
    const auto out_tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t blk = 1;

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t w_idx = curr_tile % Wt;
        std::uint32_t nc_idx = curr_tile / Wt;
        std::uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (std::uint32_t h = 0; h < Ht; h++) {
            dfb_out_obj.wait_front(blk);
            noc.async_write(dfb_out_obj, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
            noc.async_write_barrier();
            dfb_out_obj.pop_front(blk);
            tile_idx += Wt;
        }
        curr_tile += 1;
    }
}
