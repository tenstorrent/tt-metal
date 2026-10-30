// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tensix_types.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t out_tensor_tile_id = get_arg(args::out_tensor_tile_id);

    // COMPILE TIME ARGS
    // WRITER COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);

    DataflowBuffer dfb_out0(dfb::out0);  // same buffer as the reader's dfb_in0
    uint32_t single_tile_size_bytes = dfb_out0.get_tile_size();
    const auto s = TensorAccessor(tensor::output);

    uint32_t l1_read_addr = dfb_out0.get_read_ptr();
    uint32_t out_num_tiles_read = in0_w_tiles;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        dfb_out0.wait_front(out_num_tiles_read);

        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr), s, single_tile_size_bytes, {}, {.page_id = out_tensor_tile_id});
            l1_read_addr += single_tile_size_bytes;
            out_tensor_tile_id++;
        }
        out_num_tiles_read += in0_w_tiles;
    }

    noc.async_write_barrier();
    dfb_out0.pop_front(out_num_tiles_read);
}
