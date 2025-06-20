// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t dy_addr = get_arg_val<uint32_t>(1);

    uint32_t N = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);

    uint32_t scaler = get_arg_val<uint32_t>(5);
    uint32_t mask_w = get_arg_val<uint32_t>(6);

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t y_tile_bytes = get_tile_size(cb_y);
    const DataFormat y_data_format = get_dataformat(cb_y);

    uint32_t dy_tile_bytes = get_tile_size(cb_dy);
    const DataFormat dy_data_format = get_dataformat(cb_dy);

    constexpr bool y_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dy_is_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGenFast<y_is_dram> y_in = {
        .bank_base_address = y_addr, .page_size = y_tile_bytes, .data_format = y_data_format};

    const InterleavedAddrGenFast<dy_is_dram> dy_in = {
        .bank_base_address = dy_addr, .page_size = dy_tile_bytes, .data_format = dy_data_format};

    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler(cb_scaler, scaler);
    generate_mask_w(cb_mask, mask_w);

    // read ublocks from src0 to CB0, then push ublocks to compute (unpacker)
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t curr_offset_i = curr_tile;
        for (uint32_t w = 0; w < Wt; w++) {
#ifndef LOG
            // read y
            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(curr_tile, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);
#endif

            // read dy
            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(curr_tile, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);

            curr_tile++;
        }

        curr_tile = curr_offset_i;
        for (uint32_t w = 0; w < Wt; w++) {
            // read y
            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(curr_tile, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);

            // read dy
            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(curr_tile, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);

            curr_tile++;
        }
    }
}
