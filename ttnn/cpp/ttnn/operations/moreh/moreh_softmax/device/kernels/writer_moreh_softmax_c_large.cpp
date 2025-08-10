// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t outer_stride = get_arg_val<uint32_t>(3);
    uint32_t inner_size = get_arg_val<uint32_t>(4);
    uint32_t dim_size = get_arg_val<uint32_t>(5);

    constexpr auto cb_out = tt::CBIndex::c_16;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t dst_out_tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto dst_out = TensorAccessor(out_args, dst_addr, dst_out_tile_bytes);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        uint32_t outer_idx = curr_tile / (inner_size);
        uint32_t inner_idx = curr_tile % inner_size;
        uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        uint32_t dim_stride = inner_size;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_wait_front(cb_out, onetile);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            noc_async_write_tile(tile_idx, dst_out, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out, onetile);
            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
