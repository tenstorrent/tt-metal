// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in1_receiver_writer_padding.cpp, which lives beside it.
// Factories ported to Metal 2.0 bind this fork; the original serves the consumers still on the
// legacy ProgramDescriptor API. Until the last of them migrates and the original is retired, changes
// to either copy likely belong in the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// The reduce-scatter op signaler is gated behind FUSE_OP_REDUCE_SCATTER rather than a compile-time
// arg: OpSignaler consumes *positional* runtime args through an index it advances by reference, and
// it lives outside this op's directory (ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp), so
// it cannot be fed from named arguments without changing a file this port may not touch.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#ifdef FUSE_OP_REDUCE_SCATTER
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // READER
#ifdef FUSE_OP_REDUCE_SCATTER
    uint32_t rt_args_idx = 0;
#endif
    // in1 mcast args
    const uint32_t in1_mcast_sender_noc_x = get_arg(args::in1_mcast_sender_noc_x);
    const uint32_t in1_mcast_sender_noc_y = get_arg(args::in1_mcast_sender_noc_y);

    // WRITER
    // out tensor args
    uint32_t out_tensor_start_tile_id = get_arg(args::out_tensor_start_tile_id);

    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg(args::out_num_nonzero_subblocks_h);
    const uint32_t out_last_num_nonzero_subblocks_h = get_arg(args::out_last_num_nonzero_subblocks_h);
    const uint32_t out_last_subblock_h = get_arg(args::out_last_subblock_h);
    const uint32_t padded_block_tiles_h_skip = get_arg(args::padded_block_tiles_h_skip);

    const uint32_t out_num_nonzero_subblocks_w = get_arg(args::out_num_nonzero_subblocks_w);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg(args::out_last_num_nonzero_subblocks_w);
    const uint32_t out_last_subblock_w = get_arg(args::out_last_subblock_w);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg(args::padded_subblock_tiles_addr_skip);
    const uint32_t padded_block_tiles_w_skip = get_arg(args::padded_block_tiles_w_skip);

#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_h_dim = get_arg(args::last_num_blocks_h_dim);
    const uint32_t last_num_blocks_w_dim = get_arg(args::last_num_blocks_w_dim);
#endif

    // COMPILE TIME ARGS
    // READER
    // in1 block args
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // batch args
    constexpr auto batch = get_arg(args::batch);

    // WRITER
    // out tensor args
    constexpr auto out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr auto out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr auto out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr auto out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr auto out_tensor_next_w_dim_block_stride = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr auto out_tensor_next_h_dim_block_stride = get_arg(args::out_tensor_next_h_dim_block_stride);

    // out subblock args
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_tile_count = get_arg(args::out_subblock_tile_count);

    // batch args
    constexpr auto MtNt = get_arg(args::MtNt);  // if 0
    // Don't need batch; same as batch from READER args

#ifdef FUSE_BIAS
    // in3 block args
    constexpr auto in3_block_w = get_arg(args::in3_block_w);
#endif

#ifdef FUSE_OP_REDUCE_SCATTER
    OpSignaler op_signaler = OpSignaler(rt_args_idx);
#endif
    // WRITER

    const Noc noc;
    // in1 is filled here from the sender's multicast and drained by the compute kernel; out is filled
    // by the compute kernel's packer and drained here.
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    Semaphore sender_sem(sem::in1_mcast_sender);
    Semaphore receiver_sem(sem::in1_mcast_receiver);
#ifdef FUSE_BIAS
    // bias is filled here from the sender's multicast and consumed by the compute kernel's bias add.
    DataflowBuffer dfb_in3(dfb::bias);
#endif

    // WRITER
    // single-tile
    const uint32_t output_single_tile_size_bytes = dfb_out.get_tile_size();

    // WRITER
    const auto s = TensorAccessor(tensor::out);
    // `s` is only consumed inside the `#ifndef OUT_SHARDED` write path below; mark it used so
    // sharded builds don't warn (-Wunused-but-set-variable).
    (void)s;

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    // Operand 1
                    dfb_in1.reserve_back(in1_block_num_tiles);

                    // Set in1 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    dfb_in1.push_back(in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
                // Only read bias on first batch, or we have multiple output blocks
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    // Operand 2
                    dfb_in3.reserve_back(in3_block_w);

                    // Set in1 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    dfb_in3.push_back(in3_block_w);
                }
#endif

#ifndef OUT_SHARDED
                // WRITER
                const uint32_t num_blocks_h_dim_ =
                    bh >= last_num_blocks_h_dim - 1 ? last_num_blocks_h_dim : num_blocks_h_dim;
                const uint32_t num_blocks_w_dim_ =
                    bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
                uint32_t out_num_nonzero_subblocks_w_ = out_num_nonzero_subblocks_w;
                if (bh == num_blocks_h_dim_ - 1) {
                    out_num_nonzero_subblocks_h_ = out_last_num_nonzero_subblocks_h;
                }
                if (bw == num_blocks_w_dim_ - 1) {
                    out_num_nonzero_subblocks_w_ = out_last_num_nonzero_subblocks_w;
                }
                uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
                for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
                    uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                    for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
                        uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                        uint32_t out_subblock_h_ = out_subblock_h;
                        uint32_t out_subblock_w_ = out_subblock_w;
                        uint32_t subblock_tiles_addr_skip = 0;
                        if (bh == num_blocks_h_dim_ - 1 && sbh == out_num_nonzero_subblocks_h_ - 1) {
                            out_subblock_h_ = out_last_subblock_h;
                        }
                        if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                            out_subblock_w_ = out_last_subblock_w;
                            subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                        }

                        dfb_out.wait_front(out_subblock_tile_count);
                        uint32_t out_read_offset = 0;

                        for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                            uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                            for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                if (bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_) {
                                    noc.async_write(
                                        dfb_out,
                                        s,
                                        output_single_tile_size_bytes,
                                        {.offset_bytes = out_read_offset},
                                        {.page_id = out_tensor_tile_id});
                                }

                                out_read_offset += output_single_tile_size_bytes;

                                out_tensor_tile_id += out_tensor_stride_w;
                            }
                            // Skip padded tiles in subblock along row
                            out_read_offset += subblock_tiles_addr_skip;
                            out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                        }

                        noc.async_write_barrier();

                        dfb_out.pop_front(out_subblock_tile_count);
                        out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                    }
                    // Pop fully padded subblocks along the row
                    if (bw == num_blocks_w_dim_ - 1) {
                        dfb_out.wait_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                        dfb_out.pop_front(static_cast<uint16_t>(padded_block_tiles_w_skip));
                    }
                    out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                }
                // Pop row(s) of fully padded subblocks
                if (bh == num_blocks_h_dim_ - 1) {
                    dfb_out.wait_front(static_cast<uint16_t>(padded_block_tiles_h_skip));
                    dfb_out.pop_front(static_cast<uint16_t>(padded_block_tiles_h_skip));
                }
#endif
                out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
            }
            out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
        }
        out_tensor_start_tile_id += MtNt;

#ifdef FUSE_OP_REDUCE_SCATTER
        // Signal reduce_scatter to go
        op_signaler.synchronize_workers_and_signal_op(0);
#endif
    }

#ifdef OUT_SHARDED
    dfb_out.wait_front(static_cast<uint16_t>(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h));
#endif
}
