// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

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

    constexpr auto cb_param_out = tt::CB::c_out0;
    constexpr auto cb_momentum_out = tt::CB::c_out1;

    constexpr uint32_t onetile = 1;

    // param_out
    auto param_out = InterleavedAddrGenFastHelper(param_out_addr, cb_param_out, 0);

// param_out
#if defined(MOMENTUM)
    auto momentum_out = InterleavedAddrGenFastHelper(momentum_out_addr, cb_momentum_out, 1);
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
