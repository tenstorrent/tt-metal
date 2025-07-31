// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    uint32_t param_out_addr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t momentum_out_addr = get_arg_val<uint32_t>(i);
    i++;
    uint32_t num_tiles = get_arg_val<uint32_t>(i);
    i++;
    uint32_t tile_offset = get_arg_val<uint32_t>(i);
    i++;

    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_momentum_out = tt::CBIndex::c_17;

    constexpr uint32_t onetile = 1;

    // param_out
    constexpr auto param_out_args = TensorAccessorArgs<0>();
    auto param_out = TensorAccessor(param_out_args, param_out_addr, get_tile_size(cb_param_out));

// param_out
#if defined(MOMENTUM)
    constexpr auto momentum_out_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();
    auto momentum_out = TensorAccessor(momentum_out_args, momentum_out_addr, get_tile_size(cb_momentum_out));
#endif

    uint32_t l1_read_addr;

    uint32_t tile_idx = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i++) {
#if defined(MOMENTUM)
        // momentum_out
        noc_async_write_tile_helper(cb_momentum_out, onetile, tile_idx, momentum_out);
#endif

        // param_out
        noc_async_write_tile_helper(cb_param_out, onetile, tile_idx, param_out);

        tile_idx++;
    }
}
