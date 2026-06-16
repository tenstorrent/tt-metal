// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the MatmulMultiCoreReuseOptimized factory, so ported in
// place. Logic, #ifdefs, and loop bounds are unchanged from the legacy reader/writer;
// only the access mechanism moves to named bindings: the in1 / out tensor addresses ->
// ta::b / ta::out, CB ids -> dfb::cb_in1 / dfb::cb_out / dfb::cb_in1_intermediate,
// positional CT/RT args -> get_arg(args::...). The in1 CB (IN1_SHARDED) and out CB
// (OUT_SHARDED) become borrowed-memory DFBs backed by tensors `b` / `out`; on those
// paths the matching ta:: construction lives inside the existing #ifndef block, so the
// factory binds the tensor to this kernel only on the corresponding NoC path.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    // READER
    // in1 tensor args (addr now arrives via the ta::b binding)
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    // batch args
    const uint32_t batch = get_arg(args::batch);

    // WRITER
    // out tensor args (addr now arrives via the ta::out binding)
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

    constexpr uint32_t cb_id_in1 = dfb::cb_in1;
    constexpr uint32_t one_tile = 1;
    // WRITER
    constexpr uint32_t cb_id_out0 = dfb::cb_out;

    Noc noc;
    CircularBuffer cb_in1(cb_id_in1);
    CircularBuffer cb_out(cb_id_out0);

#ifdef IN1_SHARDED
    const uint32_t in1_num_tiles = batch * num_blocks * in1_block_h * in1_block_w;
    cb_in1.reserve_back(in1_num_tiles);
    cb_in1.push_back(in1_num_tiles);
#else
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(ta::b);
#endif  // IN1_SHARDED

#ifndef OUT_SHARDED
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    const auto s = TensorAccessor(ta::out);
#endif  // OUT_SHARDED

#if not defined IN1_SHARDED or not defined OUT_SHARDED
    for (uint32_t b = 0; b < batch; ++b) {
#ifndef IN1_SHARDED
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in1.reserve_back(in1_block_num_tiles);

#ifdef INTERMEDIATE_CB_READ
            constexpr uint32_t in1_intermediate_cb_index = dfb::cb_in1_intermediate;
            CircularBuffer cb_helper(in1_intermediate_cb_index);
            cb_helper.reserve_back(one_tile);
#endif  // INTERMEDIATE_CB_READ

            uint32_t in1_write_offset = 0;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
#ifndef INTERMEDIATE_CB_READ
                    noc.async_read(
                        s1,
                        cb_in1,
                        in1_single_tile_size_bytes,
                        {.page_id = in1_tensor_tile_id},
                        {.offset_bytes = in1_write_offset});
#else
                    noc.async_read(
                        s1,
                        cb_helper,
                        in1_single_tile_size_bytes,
                        {.page_id = in1_tensor_tile_id},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    memcpy(
                        /*dst=*/reinterpret_cast<void*>(cb_in1.get_write_ptr() + in1_write_offset),
                        /*src=*/reinterpret_cast<const void*>(cb_helper.get_write_ptr()),
                        /*size=*/in1_single_tile_size_bytes);
#endif  // INTERMEDIATE_CB_READ
                    in1_write_offset += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc.async_read_barrier();

            cb_in1.push_back(in1_block_num_tiles);
#ifdef INTERMEDIATE_CB_READ
            // Clean up helper CB
            cb_helper.push_back(one_tile);
            cb_helper.wait_front(one_tile);
            cb_helper.pop_front(one_tile);
#endif  // INTERMEDIATE_CB_READ
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
                        // A bare DFB used as a NoC source is already read-pointer-sourced, so the
                        // legacy use<CircularBuffer::AddrSelector::READ_PTR>(cb_out) wrapper drops.
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
