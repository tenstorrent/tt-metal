// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

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
    auto param_out = TensorAccessor(param_out_args, param_out_addr);

// param_out
#if defined(MOMENTUM)
    constexpr auto momentum_out_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();
    auto momentum_out = TensorAccessor(momentum_out_args, momentum_out_addr);
#endif

    Noc noc;
    DataflowBuffer dfb_param_out_obj(cb_param_out);
    const auto param_out_tile_bytes = get_tile_size(cb_param_out);
#if defined(MOMENTUM)
    DataflowBuffer dfb_momentum_out_obj(cb_momentum_out);
    const auto momentum_out_tile_bytes = get_tile_size(cb_momentum_out);
#endif

    uint32_t tile_idx = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i++) {
#if defined(MOMENTUM)
        dfb_momentum_out_obj.wait_front(onetile);
        noc.async_write(
            dfb_momentum_out_obj, momentum_out, momentum_out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
        noc.async_write_barrier();
        dfb_momentum_out_obj.pop_front(onetile);
#endif

        dfb_param_out_obj.wait_front(onetile);
        noc.async_write(dfb_param_out_obj, param_out, param_out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
        noc.async_write_barrier();
        dfb_param_out_obj.pop_front(onetile);

        tile_idx++;
    }
}
