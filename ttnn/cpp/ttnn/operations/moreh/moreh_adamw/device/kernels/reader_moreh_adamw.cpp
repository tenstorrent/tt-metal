// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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
    const auto beta1_exponent = get_arg_val<uint32_t>(i++);
    const auto beta2_exponent = get_arg_val<uint32_t>(i++);

    const auto step = get_arg_val<uint32_t>(i++);
    const auto amsgrad = get_arg_val<uint32_t>(i++) == 1;
    const auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto start_id = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_param = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_grad = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_exp_avg = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CBIndex::c_3;

    // lr, beta1, beta2, eps, weight_decay
    constexpr uint32_t cb_scalar_args = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_one = tt::CBIndex::c_6;

    constexpr uint32_t cb_beta1_exponent = tt::CBIndex::c_28;
    constexpr uint32_t cb_beta2_exponent = tt::CBIndex::c_29;

    constexpr auto param_args = TensorAccessorArgs<0>();
    constexpr auto grad_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();

    const auto param_addrg = TensorAccessor(param_args, param_addr);
    const auto grad_addrg = TensorAccessor(grad_args, grad_addr);
    const auto exp_avg_addrg = TensorAccessor(exp_avg_args, exp_avg_addr);
    const auto exp_avg_sq_addrg = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr);

#ifdef AMSGRAD
    constexpr uint32_t cb_id_max_exp_avg_sq = tt::CBIndex::c_4;
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();
    const auto max_exp_avg_sq_addrg = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr);
#endif

    fill_cb_with_value(cb_scalar_args, lr);
    fill_cb_with_value(cb_scalar_args, beta1);
    fill_cb_with_value(cb_scalar_args, beta2);
    fill_cb_with_value(cb_scalar_args, eps);
    fill_cb_with_value(cb_scalar_args, weight_decay);
    fill_cb_with_value(cb_beta1_exponent, beta1_exponent);
    fill_cb_with_value(cb_beta2_exponent, beta2_exponent);
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_one, scaler.u);

    experimental::Noc noc;
    experimental::CircularBuffer cb_param(cb_id_param);
    experimental::CircularBuffer cb_grad(cb_id_grad);
    experimental::CircularBuffer cb_exp_avg(cb_id_exp_avg);
    experimental::CircularBuffer cb_exp_avg_sq(cb_id_exp_avg_sq);
#ifdef AMSGRAD
    experimental::CircularBuffer cb_max_exp_avg_sq(cb_id_max_exp_avg_sq);
#endif

    const auto param_tile_bytes = get_tile_size(cb_id_param);
    const auto grad_tile_bytes = get_tile_size(cb_id_grad);
    const auto exp_avg_tile_bytes = get_tile_size(cb_id_exp_avg);
    const auto exp_avg_sq_tile_bytes = get_tile_size(cb_id_exp_avg_sq);
#ifdef AMSGRAD
    const auto max_exp_avg_sq_tile_bytes = get_tile_size(cb_id_max_exp_avg_sq);
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_param.reserve_back(onetile);
        noc.async_read(param_addrg, cb_param, param_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_param.push_back(onetile);

        cb_grad.reserve_back(onetile);
        noc.async_read(grad_addrg, cb_grad, grad_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_grad.push_back(onetile);

        cb_exp_avg.reserve_back(onetile);
        noc.async_read(exp_avg_addrg, cb_exp_avg, exp_avg_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_exp_avg.push_back(onetile);

        cb_exp_avg_sq.reserve_back(onetile);
        noc.async_read(exp_avg_sq_addrg, cb_exp_avg_sq, exp_avg_sq_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_exp_avg_sq.push_back(onetile);

#ifdef AMSGRAD
        cb_max_exp_avg_sq.reserve_back(onetile);
        noc.async_read(
            max_exp_avg_sq_addrg, cb_max_exp_avg_sq, max_exp_avg_sq_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_max_exp_avg_sq.push_back(onetile);
#endif
    }
}
