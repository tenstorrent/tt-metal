// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t tile_offset = get_arg(args::tile_offset);

    constexpr uint32_t onetile = 1;

    // param_out (base address + layout supplied by the TensorBinding).
    const auto param_out = TensorAccessor(tensor::param_out);

// momentum_out
#if defined(MOMENTUM)
    const auto momentum_out = TensorAccessor(tensor::momentum_out);
#endif

    Noc noc;
    DataflowBuffer dfb_param_out_obj(dfb::param_out);
    const auto param_out_tile_bytes = dfb_param_out_obj.get_tile_size();
#if defined(MOMENTUM)
    DataflowBuffer dfb_momentum_out_obj(dfb::momentum_out);
    const auto momentum_out_tile_bytes = dfb_momentum_out_obj.get_tile_size();
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
