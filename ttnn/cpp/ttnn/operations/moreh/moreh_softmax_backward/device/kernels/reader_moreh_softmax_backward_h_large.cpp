// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t dy_addr = get_arg_val<uint32_t>(1);

    uint32_t N = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t Ht = get_arg_val<uint32_t>(4);
    uint32_t Wt = get_arg_val<uint32_t>(5);

    uint32_t scaler = get_arg_val<uint32_t>(6);
    uint32_t mask_h = get_arg_val<uint32_t>(7);

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto dy_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto y_in = TensorAccessor(y_args, y_addr);
    const auto dy_in = TensorAccessor(dy_args, dy_addr);

    generate_bcast_scaler(cb_scaler, scaler);
    generate_mask_h(cb_mask, mask_h);

    experimental::Noc noc;
    experimental::CircularBuffer cb_y_obj(cb_y);
    experimental::CircularBuffer cb_dy_obj(cb_dy);
    const auto y_tile_bytes = get_tile_size(cb_y);
    const auto dy_tile_bytes = get_tile_size(cb_dy);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
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

            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
            cb_y_obj.reserve_back(onetile);
            noc.async_read(y_in, cb_y_obj, y_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_y_obj.push_back(onetile);

            cb_dy_obj.reserve_back(onetile);
            noc.async_read(dy_in, cb_dy_obj, dy_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_dy_obj.push_back(onetile);

            tile_idx += Wt;
        }
        curr_tile += 1;
    }
}
