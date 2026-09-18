// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    // READER
    // in1 tensor args
    auto in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    // batch args
    const auto batch = get_arg(args::batch);

    // WRITER
    // out tensor args
    auto out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);

#ifdef FUSE_BIAS
    // bias tensor args
    const auto in3_tensor_start_tile_id = get_arg(args::in3_tensor_start_tile_id);
#endif

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr auto in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    constexpr auto in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    constexpr auto in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);
    // in1 block args
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto in1_block_h = get_arg(args::in1_block_h);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr auto num_blocks = get_arg(args::num_blocks);
    // batch args
    constexpr auto bcast_B = get_arg(args::bcast_B);
    constexpr auto KtNt = get_arg(args::KtNt);

    // WRITER
    // out tensor args
    constexpr auto out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr auto out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr auto out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr auto out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    constexpr auto out_num_subblocks_w = get_arg(args::out_num_subblocks_w);
    constexpr auto out_num_subblocks_h = get_arg(args::out_num_subblocks_h);
    // batch args
    constexpr auto MtNt = get_arg(args::MtNt);

    const Noc noc;
    // in1 block staging (this kernel fills it, compute drains it) and the output block
    // (compute fills it, this kernel drains it to the output tensor).
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

#ifdef FUSE_BIAS
    // Load the whole per-batch [M, N] bias block once.
    // It's reused across all of this core's batch iterations (broadcast over batch).
    constexpr uint32_t bias_block_ntiles = out_subblock_h * out_num_subblocks_h * in1_block_w;  // M*N tiles
    DataflowBuffer dfb_in3(dfb::bias);
    const uint32_t bias_single_tile_size_bytes = dfb_in3.get_tile_size();
    const auto s3 = TensorAccessor(tensor::bias);
    dfb_in3.reserve_back(bias_block_ntiles);
    uint32_t in3_write_offset = 0;
    uint32_t in3_tensor_tile_id = in3_tensor_start_tile_id;
    for (uint32_t t = 0; t < bias_block_ntiles; ++t) {
        noc.async_read(
            s3,
            dfb_in3,
            bias_single_tile_size_bytes,
            {.page_id = in3_tensor_tile_id},
            {.offset_bytes = in3_write_offset});
        in3_write_offset += bias_single_tile_size_bytes;
        in3_tensor_tile_id += 1;  // [M, N] bias is row-major contiguous (stride 1)
    }
    noc.async_read_barrier();
    dfb_in3.push_back(bias_block_ntiles);
#endif

#ifdef IN1_SHARDED
    const uint32_t in1_num_tiles = batch * num_blocks * in1_block_h * in1_block_w;
    dfb_in1.reserve_back(in1_num_tiles);
    dfb_in1.push_back(in1_num_tiles);
#else
    const uint32_t in1_single_tile_size_bytes = dfb_in1.get_tile_size();
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the in1
    // buffer's entries are sized to match (see the program factory), so tiles are laid out in L1 at
    // the padded stride while the NOC reads the unpadded tile of data into each padded slot. No-op
    // when the tile size is already aligned.
    const uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
    const auto s1 = TensorAccessor(tensor::in1);
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
    const uint32_t output_single_tile_size_bytes = dfb_out.get_tile_size();
    const auto s = TensorAccessor(tensor::out);
#endif  // OUT_SHARDED

#if not defined IN1_SHARDED or not defined OUT_SHARDED
    for (uint32_t b = 0; b < batch; ++b) {
#ifndef IN1_SHARDED
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in1.reserve_back(in1_block_num_tiles);

            uint32_t in1_write_offset = 0;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
                    noc.async_read(
                        s1,
                        dfb_in1,
                        in1_single_tile_size_bytes,
                        {.page_id = in1_tensor_tile_id},
                        {.offset_bytes = in1_write_offset});
                    in1_write_offset += in1_aligned_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc.async_read_barrier();

            dfb_in1.push_back(in1_block_num_tiles);
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
        // WRITER
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                dfb_out.wait_front(out_subblock_tile_count);
                uint32_t out_read_offset = 0;

                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        noc.async_write(
                            dfb_out,
                            s,
                            output_single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});

                        out_read_offset += output_single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc.async_write_barrier();
                dfb_out.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
#endif  // OUT_SHARDED
    }
#endif  // not defined IN1_SHARDED or not defined OUT_SHARDED

#ifdef OUT_SHARDED
    dfb_out.wait_front(batch * out_num_subblocks_h * out_num_subblocks_w * out_subblock_w * out_subblock_h);
#endif  // OUT_SHARDED
}
