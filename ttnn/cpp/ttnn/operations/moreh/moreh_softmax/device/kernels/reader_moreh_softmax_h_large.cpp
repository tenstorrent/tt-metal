// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t Ht = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t scaler = get_arg_val<uint32_t>(5);
    const uint32_t mask_h = get_arg_val<uint32_t>(6);

    // Constants
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;

    // Ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src_in_tile_bytes = get_tile_size(cb_in);

    // Input tensor
    constexpr bool is_fp32 = get_compile_time_arg_val(0) == 1;
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto src_in = TensorAccessor(in_args, src_addr, src_in_tile_bytes);

    // Generate scaler and mask tiles
    if (is_fp32) {
        generate_bcast_scaler<uint32_t>(cb_scaler, scaler);
        generate_mask_h<uint32_t>(cb_mask, mask_h);
    } else {
        generate_bcast_scaler<uint16_t>(cb_scaler, scaler);
        generate_mask_h<uint16_t>(cb_mask, mask_h);
    }

    // read ublocks from src0 to CB0, then push ublocks to compute kernel
    uint32_t l1_write_addr_in = 0;
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += Wt;
        }

        curr_tile += 1;
    }
}
