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
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t in0_h_dim = get_arg(args::in0_h_dim);
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);

    // COMPILE TIME ARGS
    constexpr auto in0_h_tiles = get_arg(args::in0_h_tiles);
    constexpr auto in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr auto in0_c = get_arg(args::in0_c);
    constexpr auto in0_HtWt = get_arg(args::in0_HtWt);

    const auto s0 = TensorAccessor(tensor::in0);

    DataflowBuffer dfb_in0(dfb::src0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_tile_size();

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    uint32_t l1_write_addr;
    uint32_t in0_tensor_current_tile_id;
    uint32_t in0_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        l1_write_addr = dfb_in0.get_write_ptr();

        in0_tensor_current_tile_id_along_c = in0_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
            in0_tensor_current_tile_id = in0_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
                dfb_in0.reserve_back(block_size);

                noc.async_read(
                    s0,
                    CoreLocalMem<uint32_t>(l1_write_addr),
                    single_tile_size_bytes,
                    {.page_id = in0_tensor_current_tile_id},
                    {});
                l1_write_addr += single_tile_size_bytes;
                in0_tensor_current_tile_id++;

                noc.async_read_barrier();
                dfb_in0.push_back(block_size);
            }
            in0_tensor_current_tile_id_along_c += in0_HtWt;
        }

        // Update in0_tensor_tile_id for next h_dim or batch if we finish one CHtWt
        in0_h_dim++;
        if (in0_h_dim < in0_h_tiles) {
            in0_tensor_tile_id += in0_w_tiles;
        } else {
            in0_tensor_tile_id = in0_tensor_current_tile_id;
            in0_h_dim = 0;
        }
    }
}
