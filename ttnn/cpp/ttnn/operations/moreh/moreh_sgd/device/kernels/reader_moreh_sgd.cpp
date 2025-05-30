// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "accessor/sharded_accessor.h"

#include "dprint.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t param_in_addr = get_arg_val<uint32_t>(i++);
    uint32_t grad_addr = get_arg_val<uint32_t>(i++);
    uint32_t momentum_in_addr = get_arg_val<uint32_t>(i++);
    uint32_t num_tiles = get_arg_val<uint32_t>(i++);
    uint32_t tile_offset = get_arg_val<uint32_t>(i++);
    uint32_t lr = get_arg_val<uint32_t>(i++);
    uint32_t momentum = get_arg_val<uint32_t>(i++);
    uint32_t dampening = get_arg_val<uint32_t>(i++);
    uint32_t weight_decay = get_arg_val<uint32_t>(i++);
    uint32_t one = get_arg_val<uint32_t>(i++);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad = tt::CBIndex::c_1;
    constexpr auto cb_momentum_in = tt::CBIndex::c_2;

    constexpr auto cb_scalar_args = tt::CBIndex::c_24;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    constexpr auto param_in_page_size = get_tile_size(cb_param_in);
    constexpr auto grad_page_size = get_tile_size(cb_grad);

#ifdef PARAM_IN_SHARDED
    constexpr uint32_t param_in_base_idx_cta = 3;
    constexpr uint32_t param_in_base_idx_crta = 0;
    using param_in_dspec = nd_sharding::distribution_spec_t<param_in_base_idx_cta, param_in_base_idx_crta>;

    auto param_in_sharded_accessor =
        nd_sharding::ShardedAccessor<param_in_dspec>{param_in_addr, param_in_page_size};

    constexpr uint32_t param_in_cta_skip = nd_sharding::compile_time_args_skip<param_in_dspec>();
    constexpr auto param_in_crta_skip = nd_sharding::runtime_args_skip<param_in_dspec>();
#else
    constexpr uint32_t param_in_base_idx_cta = 3;
    constexpr uint32_t param_in_base_idx_crta = 0;

    constexpr uint32_t param_in_cta_skip = 0;
    constexpr uint32_t param_in_crta_skip = 0;
    auto param_in = InterleavedAddrGenFastHelper(param_in_addr, cb_param_in, 0);
#endif

#ifdef GRAD_SHARDED
    // constexpr uint32_t grad_base_idx_cta = param_in_base_idx_cta + param_in_cta_skip;
    // constexpr uint32_t grad_base_idx_crta = param_in_base_idx_crta + param_in_crta_skip;
    // DPRINT << "grad_base_idx_cta: " << grad_base_idx_cta
    //        << ", grad_base_idx_crta: " << grad_base_idx_crta << "\n";;
    // using grad_dspec = nd_sharding::distribution_spec_t<grad_base_idx_cta, grad_base_idx_crta>;
    
    // auto grad_sharded_accessor =
    //     nd_sharding::ShardedAccessor<grad_dspec>{grad_addr, grad_page_size};
#else
    auto grad = InterleavedAddrGenFastHelper(grad_addr, cb_grad, 1);
#endif

#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
    auto momentum_in = InterleavedAddrGenFastHelper(momentum_in_addr, cb_momentum_in, 2);
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
#ifdef PARAM_IN_SHARDED
        cb_reserve_back(cb_param_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_param_in);
        param_in_sharded_accessor.noc_async_read_page(curr_tile, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_param_in, 1);
#else
        noc_async_read_tile_helper(cb_param_in, onetile, curr_tile, param_in);
#endif

        // grad
#ifdef GRAD_SHARDED
        cb_reserve_back(cb_grad, 1);
        // l1_write_addr = get_write_ptr(cb_grad);
        // grad_sharded_accessor.noc_async_read_page(curr_tile, l1_write_addr);
        // noc_async_read_barrier();
        cb_push_back(cb_grad, 1);
#else
        noc_async_read_tile_helper(cb_grad, onetile, curr_tile, grad);
#endif

// momentum
#if defined(MOMENTUM) && defined(MOMENTUM_INITIALIZED)
        noc_async_read_tile_helper(cb_momentum_in, onetile, curr_tile, momentum_in);
#endif
        curr_tile++;
    }
}
