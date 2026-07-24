// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t param_in_addr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t grad_addr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t momentum_in_addr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t num_tiles = get_arg_val<uint32_t>(i);
    i++;
    uint32_t tile_offset = get_arg_val<uint32_t>(i);
    i++;
    uint32_t lr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t momentum = get_arg_val<uint32_t>(i);
    i++;
    uint32_t dampening = get_arg_val<uint32_t>(i);
    i++;
    uint32_t weight_decay = get_arg_val<uint32_t>(i);
    i++;
    uint32_t one = get_arg_val<uint32_t>(i);
    i++;

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad = tt::CBIndex::c_1;
    constexpr auto cb_momentum_in = tt::CBIndex::c_2;

    constexpr auto cb_scalar_args = tt::CBIndex::c_24;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    constexpr auto param_in_args = TensorAccessorArgs<0>();
    auto param_in = TensorAccessor(param_in_args, param_in_addr);
    constexpr auto grad_args = TensorAccessorArgs<param_in_args.next_compile_time_args_offset()>();
    auto grad = TensorAccessor(grad_args, grad_addr);

#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    constexpr auto momentum_in_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    auto momentum_in = TensorAccessor(momentum_in_args, momentum_in_addr);
#endif

    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    fill_cb_with_value(dfb_scalar_args_obj, lr);
    fill_cb_with_value(dfb_scalar_args_obj, momentum);
    fill_cb_with_value(dfb_scalar_args_obj, dampening);
    fill_cb_with_value(dfb_scalar_args_obj, weight_decay);
    fill_cb_with_value(dfb_scalar_args_obj, one);

    Noc noc;
    DataflowBuffer dfb_param_in_obj(cb_param_in);
    DataflowBuffer dfb_grad_obj(cb_grad);
    const auto param_in_tile_bytes = get_tile_size(cb_param_in);
    const auto grad_tile_bytes = get_tile_size(cb_grad);
#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    DataflowBuffer dfb_momentum_in_obj(cb_momentum_in);
    const auto momentum_in_tile_bytes = get_tile_size(cb_momentum_in);
#endif

    uint32_t curr_tile = tile_offset;

    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        dfb_param_in_obj.reserve_back(onetile);
        noc.async_read(param_in, dfb_param_in_obj, param_in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_param_in_obj.push_back(onetile);

        dfb_grad_obj.reserve_back(onetile);
        noc.async_read(grad, dfb_grad_obj, grad_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_grad_obj.push_back(onetile);

#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
        dfb_momentum_in_obj.reserve_back(onetile);
        noc.async_read(
            momentum_in, dfb_momentum_in_obj, momentum_in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_momentum_in_obj.push_back(onetile);
#endif
        curr_tile++;
    }
}
