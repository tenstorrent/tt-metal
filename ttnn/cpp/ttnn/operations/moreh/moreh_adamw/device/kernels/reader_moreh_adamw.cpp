// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto lr = get_arg(args::lr);
    const auto beta1 = get_arg(args::beta1);
    const auto beta2 = get_arg(args::beta2);
    const auto eps = get_arg(args::eps);
    const auto weight_decay = get_arg(args::weight_decay);
    const auto beta1_exponent = get_arg(args::beta1_exponent);
    const auto beta2_exponent = get_arg(args::beta2_exponent);

    const auto step = get_arg(args::step);
    const auto amsgrad = get_arg(args::amsgrad) == 1;
    const auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    const auto start_id = get_arg(args::start_id);

    const auto param_addrg = TensorAccessor(tensor::param_in);
    const auto grad_addrg = TensorAccessor(tensor::grad);
    const auto exp_avg_addrg = TensorAccessor(tensor::exp_avg_in);
    const auto exp_avg_sq_addrg = TensorAccessor(tensor::exp_avg_sq_in);

#ifdef AMSGRAD
    const auto max_exp_avg_sq_addrg = TensorAccessor(tensor::max_exp_avg_sq_in);
#endif

    // scalar_args holds lr, beta1, beta2, eps, weight_decay — one entry each, in that order.
    DataflowBuffer dfb_scalar(dfb::scalar_args);
    DataflowBuffer dfb_beta1_exp(dfb::beta1_exponent);
    DataflowBuffer dfb_beta2_exp(dfb::beta2_exponent);
    DataflowBuffer dfb_one(dfb::one);
    fill_cb_with_value(dfb_scalar, lr);
    fill_cb_with_value(dfb_scalar, beta1);
    fill_cb_with_value(dfb_scalar, beta2);
    fill_cb_with_value(dfb_scalar, eps);
    fill_cb_with_value(dfb_scalar, weight_decay);
    fill_cb_with_value(dfb_beta1_exp, beta1_exponent);
    fill_cb_with_value(dfb_beta2_exp, beta2_exponent);
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(dfb_one, scaler.u);

    Noc noc;
    DataflowBuffer dfb_param(dfb::param_in);
    DataflowBuffer dfb_grad(dfb::grad);
    DataflowBuffer dfb_exp_avg(dfb::exp_avg_in);
    DataflowBuffer dfb_exp_avg_sq(dfb::exp_avg_sq_in);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq(dfb::max_exp_avg_sq_in);
#endif

    const auto param_tile_bytes = dfb_param.get_tile_size();
    const auto grad_tile_bytes = dfb_grad.get_tile_size();
    const auto exp_avg_tile_bytes = dfb_exp_avg.get_tile_size();
    const auto exp_avg_sq_tile_bytes = dfb_exp_avg_sq.get_tile_size();
#ifdef AMSGRAD
    const auto max_exp_avg_sq_tile_bytes = dfb_max_exp_avg_sq.get_tile_size();
#endif

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_param.reserve_back(onetile);
        noc.async_read(param_addrg, dfb_param, param_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_param.push_back(onetile);

        dfb_grad.reserve_back(onetile);
        noc.async_read(grad_addrg, dfb_grad, grad_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_grad.push_back(onetile);

        dfb_exp_avg.reserve_back(onetile);
        noc.async_read(exp_avg_addrg, dfb_exp_avg, exp_avg_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_exp_avg.push_back(onetile);

        dfb_exp_avg_sq.reserve_back(onetile);
        noc.async_read(exp_avg_sq_addrg, dfb_exp_avg_sq, exp_avg_sq_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_exp_avg_sq.push_back(onetile);

#ifdef AMSGRAD
        dfb_max_exp_avg_sq.reserve_back(onetile);
        noc.async_read(
            max_exp_avg_sq_addrg, dfb_max_exp_avg_sq, max_exp_avg_sq_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_max_exp_avg_sq.push_back(onetile);
#endif
    }
}
