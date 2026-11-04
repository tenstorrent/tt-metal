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
    uint32_t out_tensor_tile_id = get_arg(args::out_tensor_tile_id);
    uint32_t out_tensor_tile_id_with_transpose = get_arg(args::out_tensor_tile_id_with_transpose);

    // COMPILE TIME ARGS
    // WRITER COMPILE TIME ARGS
    constexpr bool block_size_is_one = get_arg(args::block_size_is_one) == 1;
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t out_num_blocks_per_tensor = get_arg(args::out_num_blocks_per_tensor);
    constexpr uint32_t out_num_c_per_block = get_arg(args::out_num_c_per_block);
    constexpr uint32_t out_w_tiles = get_arg(args::out_w_tiles);
    constexpr uint32_t out_h_tiles = get_arg(args::out_h_tiles);
    constexpr uint32_t out_HtWt = get_arg(args::out_HtWt);

    DataflowBuffer dfb_out0(dfb::k);   // K heads, after the compute transpose
    DataflowBuffer dfb_out1(dfb::qv);  // same buffer the reader fills with Q and V heads
    const uint32_t single_tile_size_bytes = dfb_out0.get_tile_size();
    const auto sq = TensorAccessor(tensor::q);
    const auto sk = TensorAccessor(tensor::k);
    const auto sv = TensorAccessor(tensor::v);

    uint32_t l1_read_addr_out0 = dfb_out0.get_read_ptr();
    uint32_t l1_read_addr_out1 = dfb_out1.get_read_ptr();
    uint32_t out_num_tiles_read_out0 = block_size;
    uint32_t out_num_tiles_read_out1 = block_size;
    uint32_t out_tensor_current_tile_id_along_c;
    uint32_t out_tensor_current_tile_id;

    // Create q head
    out_tensor_current_tile_id_along_c = out_tensor_tile_id;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            dfb_out1.wait_front(out_num_tiles_read_out1);
            out_num_tiles_read_out1 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    dfb_out1.wait_front(out_num_tiles_read_out1);
                    out_num_tiles_read_out1++;
                }

                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr_out1),
                    sq,
                    single_tile_size_bytes,
                    {},
                    {.page_id = out_tensor_current_tile_id});
                l1_read_addr_out1 += single_tile_size_bytes;
                out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += out_HtWt;
        }
    }

    // Create k head
    out_tensor_current_tile_id = out_tensor_tile_id_with_transpose;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            dfb_out0.wait_front(out_num_tiles_read_out0);
            out_num_tiles_read_out0 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    dfb_out0.wait_front(out_num_tiles_read_out0);
                    out_num_tiles_read_out0++;
                }

                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr_out0),
                    sk,
                    single_tile_size_bytes,
                    {},
                    {.page_id = out_tensor_current_tile_id});
                l1_read_addr_out0 += single_tile_size_bytes;
                out_tensor_current_tile_id += out_h_tiles;
            }
        }
    }

    // Create v head
    out_tensor_current_tile_id_along_c = out_tensor_tile_id;
    for (uint32_t block_idx = 0; block_idx < out_num_blocks_per_tensor; block_idx++) {
        if constexpr (!block_size_is_one) {
            dfb_out1.wait_front(out_num_tiles_read_out1);
            out_num_tiles_read_out1 += block_size;
        }

        for (uint32_t c_dim_idx = 0; c_dim_idx < out_num_c_per_block; c_dim_idx++) {
            out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                if constexpr (block_size_is_one) {
                    dfb_out1.wait_front(out_num_tiles_read_out1);
                    out_num_tiles_read_out1++;
                }

                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr_out1),
                    sv,
                    single_tile_size_bytes,
                    {},
                    {.page_id = out_tensor_current_tile_id});
                l1_read_addr_out1 += single_tile_size_bytes;
                out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += out_HtWt;
        }
    }

    noc.async_write_barrier();
    dfb_out0.pop_front(out_num_tiles_read_out0);
    dfb_out1.pop_front(out_num_tiles_read_out1);
}
