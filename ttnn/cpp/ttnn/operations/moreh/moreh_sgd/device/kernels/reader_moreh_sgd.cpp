// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

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
    auto param_in = TensorAccessor(param_in_args, param_in_addr, get_tile_size(cb_param_in));
    constexpr auto grad_args = TensorAccessorArgs<param_in_args.next_compile_time_args_offset()>();
    auto grad = TensorAccessor(grad_args, grad_addr, get_tile_size(cb_grad));

#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    constexpr auto momentum_in_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    auto momentum_in = TensorAccessor(momentum_in_args, momentum_in_addr, get_tile_size(cb_momentum_in));
#endif

    fill_cb_with_value(cb_scalar_args, lr);
    fill_cb_with_value(cb_scalar_args, momentum);
    fill_cb_with_value(cb_scalar_args, dampening);
    fill_cb_with_value(cb_scalar_args, weight_decay);
    fill_cb_with_value(cb_scalar_args, one);

    uint32_t l1_write_addr;

    uint32_t curr_tile = tile_offset;

    for (uint32_t i = 0; i < num_tiles; i += onetile) {
        // param_in
        noc_async_read_tile_helper(cb_param_in, onetile, curr_tile, param_in);

        // grad
        noc_async_read_tile_helper(cb_grad, onetile, curr_tile, grad);

// momentum
#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
        noc_async_read_tile_helper(cb_momentum_in, onetile, curr_tile, momentum_in);
#endif
        curr_tile++;
    }
}
