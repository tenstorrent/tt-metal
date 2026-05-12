// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const auto param_addr = get_arg_val<uint32_t>(0);
    const auto exp_avg_addr = get_arg_val<uint32_t>(1);
    const auto exp_avg_sq_addr = get_arg_val<uint32_t>(2);

    const auto num_tiles_per_core = get_arg_val<uint32_t>(4);
    const auto start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_param = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_exp_avg = tt::CBIndex::c_17;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CBIndex::c_18;

    constexpr auto param_args = TensorAccessorArgs<0>();
    constexpr auto exp_avg_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();

    const auto param_addrg = TensorAccessor(param_args, param_addr);
    const auto exp_avg_addrg = TensorAccessor(exp_avg_args, exp_avg_addr);
    const auto exp_avg_sq_addrg = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr);

#ifdef AMSGRAD
    constexpr uint32_t cb_id_max_exp_avg_sq = tt::CBIndex::c_19;
    const auto max_exp_avg_sq_addr = get_arg_val<uint32_t>(3);
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();
    const auto max_exp_avg_sq_addrg = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr);
#endif

    experimental::Noc noc;
    experimental::CircularBuffer cb_param(cb_id_param);
    experimental::CircularBuffer cb_exp_avg(cb_id_exp_avg);
    experimental::CircularBuffer cb_exp_avg_sq(cb_id_exp_avg_sq);
#ifdef AMSGRAD
    experimental::CircularBuffer cb_max_exp_avg_sq(cb_id_max_exp_avg_sq);
#endif

    const auto param_tile_bytes = get_tile_size(cb_id_param);
    const auto exp_avg_tile_bytes = get_tile_size(cb_id_exp_avg);
    const auto exp_avg_sq_tile_bytes = get_tile_size(cb_id_exp_avg_sq);
#ifdef AMSGRAD
    const auto max_exp_avg_sq_tile_bytes = get_tile_size(cb_id_max_exp_avg_sq);
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_param.wait_front(onetile);
        noc.async_write(cb_param, param_addrg, param_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_param.pop_front(onetile);

        cb_exp_avg.wait_front(onetile);
        noc.async_write(cb_exp_avg, exp_avg_addrg, exp_avg_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_exp_avg.pop_front(onetile);

        cb_exp_avg_sq.wait_front(onetile);
        noc.async_write(cb_exp_avg_sq, exp_avg_sq_addrg, exp_avg_sq_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_exp_avg_sq.pop_front(onetile);

#ifdef AMSGRAD
        cb_max_exp_avg_sq.wait_front(onetile);
        noc.async_write(
            cb_max_exp_avg_sq, max_exp_avg_sq_addrg, max_exp_avg_sq_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_max_exp_avg_sq.pop_front(onetile);
#endif
    }
}
