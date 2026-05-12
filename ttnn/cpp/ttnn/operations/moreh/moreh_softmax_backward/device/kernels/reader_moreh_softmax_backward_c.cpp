// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t dy_addr = get_arg_val<uint32_t>(1);

    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t outer_stride = get_arg_val<uint32_t>(4);
    uint32_t inner_size = get_arg_val<uint32_t>(5);
    uint32_t dim_size = get_arg_val<uint32_t>(6);

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;

    constexpr uint32_t onetile = 1;

    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto dy_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto y_in = TensorAccessor(y_args, y_addr);
    const auto dy_in = TensorAccessor(dy_args, dy_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_y_obj(cb_y);
    experimental::CircularBuffer cb_dy_obj(cb_dy);
    const auto y_tile_bytes = get_tile_size(cb_y);
    const auto dy_tile_bytes = get_tile_size(cb_dy);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        uint32_t outer_idx = curr_tile / (inner_size);
        uint32_t inner_idx = curr_tile % inner_size;
        uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        uint32_t dim_stride = inner_size;
        for (uint32_t d = 0; d < dim_size; d++) {
#ifndef LOG
            cb_y_obj.reserve_back(onetile);
            noc.async_read(y_in, cb_y_obj, y_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_y_obj.push_back(onetile);
#endif

            cb_dy_obj.reserve_back(onetile);
            noc.async_read(dy_in, cb_dy_obj, dy_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_dy_obj.push_back(onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_dy_obj.reserve_back(onetile);
            noc.async_read(dy_in, cb_dy_obj, dy_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_dy_obj.push_back(onetile);

            cb_y_obj.reserve_back(onetile);
            noc.async_read(y_in, cb_y_obj, y_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_y_obj.push_back(onetile);

            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
