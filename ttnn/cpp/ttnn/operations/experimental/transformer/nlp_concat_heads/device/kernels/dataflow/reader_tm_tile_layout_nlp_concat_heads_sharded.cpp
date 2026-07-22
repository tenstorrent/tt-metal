// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t nheads = get_arg(args::nheads);                                      // This is per core per risc
    uint32_t start_read_offset_bytes = get_arg(args::start_read_offset_bytes);    // offset by nheads * in0_HtWt
    uint32_t start_write_offset_bytes = get_arg(args::start_write_offset_bytes);  // offset by nheads * in0_Wt

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr auto in0_h_tiles = get_arg(args::in0_h_tiles);
    constexpr auto head_dim_size_bytes = get_arg(args::head_dim_size_bytes);
    constexpr auto out_row_size_bytes =
        get_arg(args::out_row_size_bytes);  // total nheads per core * in0_w_tiles * single_tile_size_bytes
    constexpr auto block_size = get_arg(args::block_size);  // total nheads per core * in0_HtWt

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_out0(dfb::out0);

    const uint32_t single_tile_size_bytes = dfb_in0.get_tile_size();

    dfb_in0.reserve_back(block_size);  // Redundant
    dfb_out0.reserve_back(block_size);

    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint src_ep;

    uint32_t src_l1_read_addr = dfb_in0.get_read_ptr() + start_read_offset_bytes;
    uint32_t l1_write_addr = dfb_out0.get_write_ptr() + start_write_offset_bytes;

    for (uint32_t i = 0; i < nheads; ++i) {
        uint32_t curr_l1_write_addr = l1_write_addr;
        for (uint32_t j = 0; j < in0_h_tiles; ++j) {
            noc.async_read(
                src_ep,
                CoreLocalMem<uint32_t>(curr_l1_write_addr),
                head_dim_size_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_l1_read_addr},
                {});
            src_l1_read_addr += head_dim_size_bytes;
            curr_l1_write_addr += out_row_size_bytes;
        }
        l1_write_addr += head_dim_size_bytes;
    }

    noc.async_read_barrier();
    // dfb_out0.push_back(block_size);
}
