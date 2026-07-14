// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t outer_stride = get_arg_val<uint32_t>(3);
    uint32_t inner_size = get_arg_val<uint32_t>(4);
    uint32_t dim_size = get_arg_val<uint32_t>(5);

    constexpr auto cb_in = tt::CBIndex::c_0;

    constexpr uint32_t onetile = 1;

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto src_in = TensorAccessor(in_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in_obj(cb_in);
    const auto in_tile_bytes = get_tile_size(cb_in);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        uint32_t outer_idx = curr_tile / (inner_size);
        uint32_t inner_idx = curr_tile % inner_size;
        uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        uint32_t dim_stride = inner_size;
        for (uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
