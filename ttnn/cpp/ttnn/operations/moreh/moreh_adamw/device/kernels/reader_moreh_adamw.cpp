// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const auto param_addr = get_arg_val<uint32_t>(i++);
    const auto grad_addr = get_arg_val<uint32_t>(i++);
    const auto exp_avg_addr = get_arg_val<uint32_t>(i++);
    const auto exp_avg_sq_addr = get_arg_val<uint32_t>(i++);
    const auto max_exp_avg_sq_addr = get_arg_val<uint32_t>(i++);

    const auto lr = get_arg_val<uint32_t>(i++);
    const auto beta1 = get_arg_val<uint32_t>(i++);
    const auto beta2 = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);
    const auto weight_decay = get_arg_val<uint32_t>(i++);

    const auto step = get_arg_val<uint32_t>(i++);
    const auto amsgrad = get_arg_val<uint32_t>(i++) == 1;
    const auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto start_id = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_param = tt::CB::c_in0;
    constexpr uint32_t cb_id_grad = tt::CB::c_in1;
    constexpr uint32_t cb_id_exp_avg = tt::CB::c_in2;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CB::c_in3;

    // lr, beta1, beta2, eps, weight_decay
    constexpr uint32_t cb_scalar_args = tt::CB::c_in5;
    constexpr uint32_t cb_id_one = tt::CB::c_in6;

    const uint32_t param_tile_bytes = get_tile_size(cb_id_param);
    const auto param_data_format = get_dataformat(cb_id_param);

    const uint32_t grad_tile_bytes = get_tile_size(cb_id_grad);
    const auto grad_data_format = get_dataformat(cb_id_grad);

    const uint32_t exp_avg_tile_bytes = get_tile_size(cb_id_exp_avg);
    const auto exp_avg_data_format = get_dataformat(cb_id_exp_avg);

    const uint32_t exp_avg_sq_tile_bytes = get_tile_size(cb_id_exp_avg_sq);
    const auto exp_avg_sq_data_format = get_dataformat(cb_id_exp_avg_sq);

    constexpr bool param_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool grad_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool exp_avg_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool exp_avg_sq_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool max_exp_avg_sq_is_dram = get_compile_time_arg_val(4) == 1;

    const InterleavedAddrGenFast<param_is_dram> param_addrg = {
        .bank_base_address = param_addr,
        .page_size = param_tile_bytes,
        .data_format = param_data_format};

    const InterleavedAddrGenFast<grad_is_dram> grad_addrg = {
        .bank_base_address = grad_addr, .page_size = grad_tile_bytes, .data_format = grad_data_format};

    const InterleavedAddrGenFast<exp_avg_is_dram> exp_avg_addrg = {
        .bank_base_address = exp_avg_addr, .page_size = exp_avg_tile_bytes, .data_format = exp_avg_data_format};

    const InterleavedAddrGenFast<exp_avg_sq_is_dram> exp_avg_sq_addrg = {
        .bank_base_address = exp_avg_sq_addr, .page_size = exp_avg_sq_tile_bytes, .data_format = exp_avg_sq_data_format};

#ifdef AMSGRAD
    constexpr uint32_t cb_id_max_exp_avg_sq = tt::CB::c_in4;
    const uint32_t max_exp_avg_sq_tile_bytes = get_tile_size(cb_id_max_exp_avg_sq);
    const auto max_exp_avg_sq_data_format = get_dataformat(cb_id_max_exp_avg_sq);
    const InterleavedAddrGenFast<max_exp_avg_sq_is_dram> max_exp_avg_sq_addrg = {
        .bank_base_address = max_exp_avg_sq_addr, .page_size = max_exp_avg_sq_tile_bytes, .data_format = max_exp_avg_sq_data_format};
#endif

    fill_cb_with_value(cb_scalar_args, lr);
    fill_cb_with_value(cb_scalar_args, beta1);
    fill_cb_with_value(cb_scalar_args, beta2);
    fill_cb_with_value(cb_scalar_args, eps);
    fill_cb_with_value(cb_scalar_args, weight_decay);
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_one, scaler.u);

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_reserve_back(cb_id_param, onetile);
        uint32_t param_l1_write_addr = get_write_ptr(cb_id_param);
        noc_async_read_tile(i, param_addrg, param_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_param, onetile);

        cb_reserve_back(cb_id_grad, onetile);
        uint32_t grad_l1_write_addr = get_write_ptr(cb_id_grad);
        noc_async_read_tile(i, grad_addrg, grad_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_grad, onetile);

        cb_reserve_back(cb_id_exp_avg, onetile);
        uint32_t exp_avg_l1_write_addr = get_write_ptr(cb_id_exp_avg);
        noc_async_read_tile(i, exp_avg_addrg, exp_avg_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_exp_avg, onetile);

        cb_reserve_back(cb_id_exp_avg_sq, onetile);
        uint32_t exp_avg_sq_l1_write_addr = get_write_ptr(cb_id_exp_avg_sq);
        noc_async_read_tile(i, exp_avg_sq_addrg, exp_avg_sq_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_exp_avg_sq, onetile);

#ifdef AMSGRAD
        cb_reserve_back(cb_id_max_exp_avg_sq, onetile);
        uint32_t max_exp_avg_sq_l1_write_addr = get_write_ptr(cb_id_max_exp_avg_sq);
        noc_async_read_tile(i, max_exp_avg_sq_addrg, max_exp_avg_sq_l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_max_exp_avg_sq, onetile);
#endif
    }
}
