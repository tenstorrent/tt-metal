// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Metal 2.0 port of reader_writer_bmm_tile_layout_in1.cpp.
// Reads the in1 (weight) operand tiles into the in1 dataflow buffer and writes output subblocks from
// the out dataflow buffer. The legacy in1/out tensor address RTAs and TensorAccessorArgs<19>/
// TensorAccessorArgs<next> plumbing are replaced by the tensor::in1 / tensor::out typed bindings;
// the legacy named CB index CTAs ("cb_in1", "cb_out") are replaced by the dfb::in1 / dfb::out tokens.
void kernel_main() {
    // RUNTIME ARGS
    // READER
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    // batch args
    const uint32_t batch = get_arg(args::batch);
    // WRITER
    uint32_t out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    constexpr uint32_t in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    constexpr uint32_t in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);
    // in1 block args
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    constexpr uint32_t in1_block_h = get_arg(args::in1_block_h);
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    // batch args
    constexpr uint32_t bcast_B = get_arg(args::bcast_B);
    constexpr uint32_t KtNt = get_arg(args::KtNt);
    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr uint32_t out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_tile_count = get_arg(args::out_subblock_tile_count);
    constexpr uint32_t out_num_subblocks_w = get_arg(args::out_num_subblocks_w);
    constexpr uint32_t out_num_subblocks_h = get_arg(args::out_num_subblocks_h);
    // batch args
    constexpr uint32_t MtNt = get_arg(args::MtNt);

    constexpr uint32_t cb_id_in1 = dfb::in1;
    // WRITER
    constexpr uint32_t cb_id_out0 = dfb::out;

    Noc noc;
    DataflowBuffer cb_in1(dfb::in1);
    DataflowBuffer cb_out(dfb::out);

#ifdef IN1_SHARDED
    const uint32_t in1_num_tiles = batch * num_blocks * in1_block_h * in1_block_w;
    cb_in1.reserve_back(in1_num_tiles);
    cb_in1.push_back(in1_num_tiles);
#else
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM and the in1
    // CB pages are sized to match (see the program factory), so tiles are laid out in L1 at the
    // padded stride while the NOC reads the unpadded tile of data into each padded slot. No-op when
    // the tile size is already aligned.
    const uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
    const auto s1 = TensorAccessor(tensor::in1);
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    const auto s = TensorAccessor(tensor::out);
#endif  // OUT_SHARDED

#if not defined IN1_SHARDED or not defined OUT_SHARDED
    for (uint32_t b = 0; b < batch; ++b) {
#ifndef IN1_SHARDED
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in1.reserve_back(in1_block_num_tiles);

            uint32_t in1_write_offset = 0;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
                    noc.async_read(
                        s1,
                        cb_in1,
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

            cb_in1.push_back(in1_block_num_tiles);
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

                cb_out.wait_front(out_subblock_tile_count);
                uint32_t out_read_offset = 0;

                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        // A DataflowBuffer used as a NoC write source resolves to get_read_ptr() +
                        // offset_bytes (see noc_traits_t<DataflowBuffer>::src_addr), i.e. the legacy
                        // use<CircularBuffer::AddrSelector::READ_PTR>(cb_out) semantics.
                        noc.async_write(
                            cb_out,
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
                cb_out.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
#endif  // OUT_SHARDED
    }
#endif  // not defined IN1_SHARDED or not defined OUT_SHARDED

#ifdef OUT_SHARDED
    cb_out.wait_front(batch * out_num_subblocks_h * out_num_subblocks_w * out_subblock_w * out_subblock_h);
#endif  // OUT_SHARDED
}
