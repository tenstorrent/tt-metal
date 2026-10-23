// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);
    constexpr uint32_t in0_HtWt = get_arg(args::in0_HtWt);

    DataflowBuffer dfb_in0(dfb::in0);
    uint32_t single_tile_size_bytes = dfb_in0.get_tile_size();
    const auto s0 = TensorAccessor(tensor::input);

    uint32_t l1_write_addr_in0 = dfb_in0.get_write_ptr();
    uint32_t in0_tensor_current_tile_id = in0_tensor_tile_id;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        dfb_in0.reserve_back(in0_w_tiles);

        in0_tensor_current_tile_id = in0_tensor_tile_id;
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(l1_write_addr_in0),
                single_tile_size_bytes,
                {.page_id = in0_tensor_current_tile_id},
                {});
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_current_tile_id++;
        }
        in0_tensor_tile_id += in0_HtWt;
        noc.async_read_barrier();
        dfb_in0.push_back(in0_w_tiles);
    }
}
