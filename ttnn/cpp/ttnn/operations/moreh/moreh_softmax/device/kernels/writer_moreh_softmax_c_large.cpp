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
    std::uint32_t num_tiles = get_arg(args::num_tiles);
    std::uint32_t tile_offset = get_arg(args::tile_offset);
    std::uint32_t outer_stride = get_arg(args::outer_stride);
    std::uint32_t inner_size = get_arg(args::inner_size);
    std::uint32_t dim_size = get_arg(args::dim_size);

    constexpr auto dfb_out = dfb::out;

    constexpr std::uint32_t onetile = 1;

    const auto dst_out = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb_out);
    const auto out_tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < num_tiles; i += onetile) {
        std::uint32_t outer_idx = curr_tile / (inner_size);
        std::uint32_t inner_idx = curr_tile % inner_size;
        std::uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        std::uint32_t dim_stride = inner_size;
        for (std::uint32_t d = 0; d < dim_size; d++) {
            dfb_out_obj.wait_front(onetile);
            noc.async_write(dfb_out_obj, dst_out, out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
            noc.async_write_barrier();
            dfb_out_obj.pop_front(onetile);
            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
