// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

void fill_cb_with_value(uint32_t cb_id, uint32_t value) {
    experimental::CircularBuffer cb(cb_id);
    cb.reserve_back(1);

#if defined FP32_DEST_ACC_EN
    experimental::CoreLocalMem<uint32_t> ptr(cb.get_write_ptr());
    for (int j = 0; j < 1024; j++) {
        ptr[j] = value;
    }
#else
    experimental::CoreLocalMem<uint16_t> ptr(cb.get_write_ptr());
    for (int j = 0; j < 1024; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
#endif

    cb.push_back(1);
}

void kernel_main() {
    const auto param_addr = get_arg_val<uint32_t>(0);
    const auto grad_addr = get_arg_val<uint32_t>(1);
    const auto exp_avg_addr = get_arg_val<uint32_t>(2);
    const auto exp_avg_sq_addr = get_arg_val<uint32_t>(3);

    const auto lr = get_arg_val<uint32_t>(5);
    const auto beta1 = get_arg_val<uint32_t>(6);
    const auto beta2 = get_arg_val<uint32_t>(7);
    const auto eps = get_arg_val<uint32_t>(8);
    const auto weight_decay = get_arg_val<uint32_t>(9);

    const auto step = get_arg_val<uint32_t>(10);
    const auto amsgrad = get_arg_val<uint32_t>(11) == 1;
    const auto num_tiles_per_core = get_arg_val<uint32_t>(12);
    const auto start_id = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_id_param = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_grad = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_exp_avg = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_exp_avg_sq = tt::CBIndex::c_3;

    // lr, beta1, beta2, eps, weight_decay
    constexpr uint32_t cb_scalar_args = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_one = tt::CBIndex::c_6;

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
    const auto max_exp_avg_sq_addr = get_arg_val<uint32_t>(4);
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();
    const auto max_exp_avg_sq_addrg = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr);
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
