// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t tile_offset = get_arg(args::tile_offset);
    const uint32_t lr = get_arg(args::lr);
    const uint32_t momentum = get_arg(args::momentum);
    const uint32_t dampening = get_arg(args::dampening);
    const uint32_t weight_decay = get_arg(args::weight_decay);
    const uint32_t one = get_arg(args::one);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    // Input tensor accessors (base address + layout supplied by the TensorBindings).
    const auto param_in = TensorAccessor(tensor::param_in);
    const auto grad = TensorAccessor(tensor::grad);

#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    const auto momentum_in = TensorAccessor(tensor::momentum_in);
#endif

    DataflowBuffer dfb_scalar_args_obj(dfb::scalar_args);
    fill_cb_with_value(dfb_scalar_args_obj, lr);
    fill_cb_with_value(dfb_scalar_args_obj, momentum);
    fill_cb_with_value(dfb_scalar_args_obj, dampening);
    fill_cb_with_value(dfb_scalar_args_obj, weight_decay);
    fill_cb_with_value(dfb_scalar_args_obj, one);

    Noc noc;
    DataflowBuffer dfb_param_in_obj(dfb::param_in);
    DataflowBuffer dfb_grad_obj(dfb::grad);
    const auto param_in_tile_bytes = dfb_param_in_obj.get_tile_size();
    const auto grad_tile_bytes = dfb_grad_obj.get_tile_size();
#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    DataflowBuffer dfb_momentum_in_obj(dfb::momentum_in);
    const auto momentum_in_tile_bytes = dfb_momentum_in_obj.get_tile_size();
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
