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
    uint32_t q_out_h_dim = get_arg(args::q_out_h_dim);
    uint32_t q_out_tensor_tile_id = get_arg(args::q_out_tensor_tile_id);
    uint32_t kv_out_tensor_tile_id = get_arg(args::kv_out_tensor_tile_id);

    // COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);
    constexpr uint32_t q_out_h_tiles = get_arg(args::q_out_h_tiles);
    constexpr uint32_t q_out_w_tiles = get_arg(args::q_out_w_tiles);
    constexpr uint32_t q_out_c = get_arg(args::q_out_c);
    constexpr uint32_t q_out_HtWt = get_arg(args::q_out_HtWt);

    DataflowBuffer dfb_out0(dfb::out);  // same as the reader's dfb::in
    const uint32_t single_tile_size_bytes = dfb_out0.get_tile_size();
    const auto sq = TensorAccessor(tensor::q);
    const auto sk = TensorAccessor(tensor::k);
    const auto sv = TensorAccessor(tensor::v);

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    uint32_t l1_read_addr;
    uint32_t out_num_tiles_read;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id;
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        l1_read_addr = dfb_out0.get_read_ptr();
        out_num_tiles_read = 0;

        out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
            q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                out_num_tiles_read += block_size;
                dfb_out0.wait_front(out_num_tiles_read);

                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sq,
                    single_tile_size_bytes,
                    {},
                    {.page_id = q_out_tensor_current_tile_id});
                l1_read_addr += single_tile_size_bytes;
                q_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            dfb_out0.wait_front(out_num_tiles_read);

            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                sk,
                single_tile_size_bytes,
                {},
                {.page_id = out_tensor_current_tile_id});
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            dfb_out0.wait_front(out_num_tiles_read);

            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                sv,
                single_tile_size_bytes,
                {},
                {.page_id = out_tensor_current_tile_id});
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tensor_tile_id += q_out_w_tiles;
        } else {
            q_out_tensor_tile_id = q_out_tensor_current_tile_id;
            q_out_h_dim = 0;
        }

        kv_out_tensor_tile_id += kv_num_tiles;

        noc.async_write_barrier();
        dfb_out0.pop_front(out_num_tiles_read);
    }
}
