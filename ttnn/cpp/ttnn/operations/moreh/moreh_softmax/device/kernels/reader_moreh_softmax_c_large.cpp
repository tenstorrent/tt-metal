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

    constexpr auto dfb_in = dfb::in;

    constexpr std::uint32_t onetile = 1;

    const auto src_in = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in_obj(dfb_in);
    const auto in_tile_bytes = dfb_in_obj.get_entry_size();

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < num_tiles; i += onetile) {
        std::uint32_t outer_idx = curr_tile / (inner_size);
        std::uint32_t inner_idx = curr_tile % inner_size;
        std::uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        std::uint32_t dim_stride = inner_size;
        for (std::uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (std::uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (std::uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
