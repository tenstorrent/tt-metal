// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

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

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    uint32_t y_tile_bytes = get_tile_size(cb_y);
    uint32_t dy_tile_bytes = get_tile_size(cb_dy);

    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto dy_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto y_in = TensorAccessor(y_args, y_addr, y_tile_bytes);
    const auto dy_in = TensorAccessor(dy_args, dy_addr, dy_tile_bytes);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        uint32_t outer_idx = curr_tile / (inner_size);
        uint32_t inner_idx = curr_tile % inner_size;
        uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        uint32_t dim_stride = inner_size;
        for (uint32_t d = 0; d < dim_size; d++) {
#ifndef LOG
            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(tile_idx, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);
#endif

            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(tile_idx, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(tile_idx, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);

            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(tile_idx, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);

            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
