// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const auto param_addr = get_arg_val<uint32_t>(0);
    const auto exp_avg_addr = get_arg_val<uint32_t>(1);
    const auto exp_avg_sq_addr = get_arg_val<uint32_t>(2);

    const auto num_tiles_per_core = get_arg_val<uint32_t>(4);
    const auto start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_param = tt::CB::c_out0;
    constexpr uint32_t cb_id_exp_avg = tt::CB::c_out1;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CB::c_out2;

    const uint32_t param_tile_bytes = get_tile_size(cb_id_param);
    const auto param_data_format = get_dataformat(cb_id_param);

    const uint32_t exp_avg_tile_bytes = get_tile_size(cb_id_exp_avg);
    const auto exp_avg_data_format = get_dataformat(cb_id_exp_avg);

    const uint32_t exp_avg_sq_tile_bytes = get_tile_size(cb_id_exp_avg_sq);
    const auto exp_avg_sq_data_format = get_dataformat(cb_id_exp_avg_sq);

    constexpr bool param_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool exp_avg_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool exp_avg_sq_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool max_exp_avg_sq_is_dram = get_compile_time_arg_val(3) == 1;

    const InterleavedAddrGenFast<param_is_dram> param_addrg = {
        .bank_base_address = param_addr,
        .page_size = param_tile_bytes,
        .data_format = param_data_format};

    const InterleavedAddrGenFast<exp_avg_is_dram> exp_avg_addrg = {
        .bank_base_address = exp_avg_addr, .page_size = exp_avg_tile_bytes, .data_format = exp_avg_data_format};

    const InterleavedAddrGenFast<exp_avg_sq_is_dram> exp_avg_sq_addrg = {
        .bank_base_address = exp_avg_sq_addr, .page_size = exp_avg_sq_tile_bytes, .data_format = exp_avg_sq_data_format};

#ifdef AMSGRAD
    constexpr uint32_t cb_id_max_exp_avg_sq = tt::CB::c_out3;
    const auto max_exp_avg_sq_addr = get_arg_val<uint32_t>(3);
    const uint32_t max_exp_avg_sq_tile_bytes = get_tile_size(cb_id_max_exp_avg_sq);
    const auto max_exp_avg_sq_data_format = get_dataformat(cb_id_max_exp_avg_sq);
    const InterleavedAddrGenFast<max_exp_avg_sq_is_dram> max_exp_avg_sq_addrg = {
        .bank_base_address = max_exp_avg_sq_addr, .page_size = max_exp_avg_sq_tile_bytes, .data_format = max_exp_avg_sq_data_format};
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_wait_front(cb_id_param, onetile);
        uint32_t param_l1_write_addr = get_read_ptr(cb_id_param);
        noc_async_write_tile(i, param_addrg, param_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_param, onetile);

        cb_wait_front(cb_id_exp_avg, onetile);
        uint32_t exp_avg_l1_write_addr = get_read_ptr(cb_id_exp_avg);
        noc_async_write_tile(i, exp_avg_addrg, exp_avg_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_exp_avg, onetile);

        cb_wait_front(cb_id_exp_avg_sq, onetile);
        uint32_t exp_avg_sq_l1_write_addr = get_read_ptr(cb_id_exp_avg_sq);
        noc_async_write_tile(i, exp_avg_sq_addrg, exp_avg_sq_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_exp_avg_sq, onetile);

#ifdef AMSGRAD
        cb_wait_front(cb_id_max_exp_avg_sq, onetile);
        uint32_t max_exp_avg_sq_l1_write_addr = get_read_ptr(cb_id_max_exp_avg_sq);
        noc_async_write_tile(i, max_exp_avg_sq_addrg, max_exp_avg_sq_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_max_exp_avg_sq, onetile);
#endif
    }
}
