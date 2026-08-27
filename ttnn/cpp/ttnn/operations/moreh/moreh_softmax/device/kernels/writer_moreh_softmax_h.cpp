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
    // Runtime args
    std::uint32_t N = get_arg(args::num_rows);
    std::uint32_t tile_offset = get_arg(args::tile_offset);
    std::uint32_t Ht = get_arg(args::Ht);
    std::uint32_t Wt = get_arg(args::Wt);

    // Constants
    constexpr auto dfb_id_out = dfb::out;
    constexpr std::uint32_t onetile = 1;

    // Output tensor
    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb_id_out);
    const std::uint32_t tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t w_idx = curr_tile % Wt;
        std::uint32_t nc_idx = curr_tile / Wt;
        std::uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;

        dfb_out_obj.wait_front(Ht);
        for (std::uint32_t h = 0; h < Ht; h++) {
            noc.async_write(dfb_out_obj, s, tile_bytes, {.offset_bytes = h * tile_bytes}, {.page_id = tile_idx});
            tile_idx += Wt;
        }
        noc.async_write_barrier();
        dfb_out_obj.pop_front(Ht);
        curr_tile += 1;
    }
}
