// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const auto param_addr = get_arg_val<uint32_t>(i++);
    const auto exp_avg_addr = get_arg_val<uint32_t>(i++);
    const auto exp_avg_sq_addr = get_arg_val<uint32_t>(i++);
    const auto max_exp_avg_sq_addr = get_arg_val<uint32_t>(i++);

    const auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto start_id = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_param = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_exp_avg = tt::CBIndex::c_17;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CBIndex::c_18;

    const uint32_t param_tile_bytes = get_tile_size(cb_id_param);
    const uint32_t exp_avg_tile_bytes = get_tile_size(cb_id_exp_avg);
    const uint32_t exp_avg_sq_tile_bytes = get_tile_size(cb_id_exp_avg_sq);

    constexpr auto param_args = TensorAccessorArgs<0>();
    constexpr auto exp_avg_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();

    const auto param_addrg = TensorAccessor(param_args, param_addr, param_tile_bytes);
    const auto exp_avg_addrg = TensorAccessor(exp_avg_args, exp_avg_addr, exp_avg_tile_bytes);
    const auto exp_avg_sq_addrg = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr, exp_avg_sq_tile_bytes);

#ifdef AMSGRAD
    constexpr uint32_t cb_id_max_exp_avg_sq = tt::CBIndex::c_19;
    const uint32_t max_exp_avg_sq_tile_bytes = get_tile_size(cb_id_max_exp_avg_sq);
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();
    const auto max_exp_avg_sq_addrg =
        TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr, max_exp_avg_sq_tile_bytes);
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
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
