// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t outer_stride = get_arg_val<uint32_t>(3);
    uint32_t inner_size = get_arg_val<uint32_t>(4);
    uint32_t dim_size = get_arg_val<uint32_t>(5);

    constexpr auto cb_in = tt::CBIndex::c_0;

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src_in_tile_bytes = get_tile_size(cb_in);
    const DataFormat src_in_data_format = get_dataformat(cb_in);

    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<in_is_dram> src_in = {
        .bank_base_address = src_addr, .page_size = src_in_tile_bytes, .data_format = src_in_data_format};

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        uint32_t outer_idx = curr_tile / (inner_size);
        uint32_t inner_idx = curr_tile % inner_size;
        uint32_t tile_idx = outer_idx * outer_stride + inner_idx;

        uint32_t dim_stride = inner_size;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += dim_stride;
        }

        tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < dim_size; d++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += dim_stride;
        }
        curr_tile += 1;
    }
}
