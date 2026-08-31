// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pass2_reader.cpp — BRISC0 / reader for device-side Stockham Pass 2.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "pass2_common.h"

void kernel_main() {
    const uint32_t in_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t first_tile = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr auto tile_args = TensorAccessorArgs<0>();

    const auto in_r_gen = TensorAccessor(tile_args, in_r_addr);
    const auto in_i_gen = TensorAccessor(tile_args, in_i_addr);
    const auto tw_r_gen = TensorAccessor(tile_args, tw_r_addr);
    const auto tw_i_gen = TensorAccessor(tile_args, tw_i_addr);

    for (uint32_t k = 0; k < num_tiles; ++k) {
        const uint32_t tile_idx = first_tile + k;

        cb_reserve_back(CB_A_R, 1);
        cb_reserve_back(CB_A_I, 1);
        cb_reserve_back(CB_T_R, 1);
        cb_reserve_back(CB_T_I, 1);

        noc_async_read_page(tile_idx, in_r_gen, get_write_ptr(CB_A_R));
        noc_async_read_page(tile_idx, in_i_gen, get_write_ptr(CB_A_I));
        noc_async_read_page(tile_idx, tw_r_gen, get_write_ptr(CB_T_R));
        noc_async_read_page(tile_idx, tw_i_gen, get_write_ptr(CB_T_I));
        noc_async_read_barrier();

        cb_push_back(CB_A_R, 1);
        cb_push_back(CB_A_I, 1);
        cb_push_back(CB_T_R, 1);
        cb_push_back(CB_T_I, 1);
    }
}
