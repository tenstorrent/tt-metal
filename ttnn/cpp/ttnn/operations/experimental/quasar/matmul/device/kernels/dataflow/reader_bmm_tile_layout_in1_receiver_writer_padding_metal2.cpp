// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in1_receiver_writer_padding.cpp.
//
// Algorithm body matches the legacy kernel; only the host-binding surface is converted to Metal 2.0:
//   - positional get_compile_time_arg_val(N)  -> named get_arg(args::name)
//   - positional get_arg_val<uint32_t>(i)     -> named get_arg(args::name)
//   - out tensor address RTA + TensorAccessorArgs -> tensor::out typed binding
//   - named CB-index CTAs ("cb_in1","cb_out","cb_bias") -> dfb:: tokens
//   - Semaphore<>(get_compile_time_arg_val(id)) -> Semaphore(sem::name)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // DEBUG: matmul mcast hang diagnosis (remove after)

void kernel_main() {
    DPRINT("WRM enter\n");  // DEBUG: matmul pre-kernel_main confirmation (remove after)
    // READER
    uint32_t rt_args_idx = 0;
    // in1 mcast args
    const uint32_t in1_mcast_sender_noc_x = get_arg(args::in1_mcast_sender_noc_x);
    const uint32_t in1_mcast_sender_noc_y = get_arg(args::in1_mcast_sender_noc_y);

    // WRITER
    // out tensor args (out_tensor_addr is now the tensor::out binding)
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
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // batch args
    constexpr uint32_t batch = get_arg(args::batch);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_arg(args::out_tensor_stride_w);
    constexpr uint32_t out_tensor_stride_h = get_arg(args::out_tensor_stride_h);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_arg(args::out_tensor_next_h_dim_block_stride);

    // out subblock args
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_tile_count = get_arg(args::out_subblock_tile_count);

    // batch args
    constexpr uint32_t MtNt = get_arg(args::MtNt);  // if 0
    // Don't need batch; same as batch from READER args

#ifdef FUSE_BIAS
    // in3 block args
    constexpr uint32_t in3_block_w = get_arg(args::in3_block_w);

    constexpr uint32_t cb_id_in3 = dfb::cb_bias;
#endif
    constexpr bool fuse_op_reduce_scatter = (bool)get_arg(args::fuse_op_reduce_scatter);

    OpSignaler op_signaler;
    if constexpr (fuse_op_reduce_scatter) {
        op_signaler = OpSignaler(rt_args_idx);
    }
    // WRITER

    constexpr uint32_t cb_id_in1 = dfb::cb_in1;

    // WRITER
    constexpr uint32_t cb_id_out0 = dfb::cb_out;

    // WRITER
    // single-tile
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);

    Noc noc;
    DataflowBuffer cb_in1(dfb::cb_in1);
    DataflowBuffer cb_out(dfb::cb_out);
    Semaphore sender_sem(sem::in1_sender);
    Semaphore receiver_sem(sem::in1_receiver);
#ifdef FUSE_BIAS
    DataflowBuffer cb_in3(dfb::cb_bias);
#endif

    // WRITER
    // Constructed to materialize the tensor::out binding; not read directly -> maybe_unused (matches the
    // in1_sender_writer sibling) to avoid -Wunused-but-set-variable.
    [[maybe_unused]] const auto s = TensorAccessor(tensor::out);

    // DEBUG: matmul mcast hang — each receiver reports its own NoC coords and which core it
    // increments (in1_mcast_sender_noc_x/y). Compare against the sender's mcast rectangle.
    DPRINT(
        "RECV core x={} y={} -> bumps sender x={} y={}\n",
        (uint32_t)my_x[noc.get_noc_id()],
        (uint32_t)my_y[noc.get_noc_id()],
        in1_mcast_sender_noc_x,
        in1_mcast_sender_noc_y);

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    // Operand 1
                    cb_in1.reserve_back(in1_block_num_tiles);

                    // Set in1 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    // DEBUG: matmul mcast hang — print once on the first block. A receiver that hangs
                    // at the wait above (sender's VALID mcast didn't reach it) will NOT print this.
                    if (b == 0 && bh == 0 && bw == 0 && block == 0) {
                        DPRINT(
                            "RECV in1 VALID x={} y={}\n",
                            (uint32_t)my_x[noc.get_noc_id()],
                            (uint32_t)my_y[noc.get_noc_id()]);
                    }

                    cb_in1.push_back(in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
                // Only read bias on first batch, or we have multiple output blocks
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    // Operand 2
                    cb_in3.reserve_back(in3_block_w);

                    // Set in1 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

                    // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    cb_in3.push_back(in3_block_w);
                }
#endif

#ifndef OUT_SHARDED
                // WRITER
                uint32_t num_blocks_h_dim_ = bh >= last_num_blocks_h_dim - 1 ? last_num_blocks_h_dim : num_blocks_h_dim;
                uint32_t num_blocks_w_dim_ = bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
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

                        cb_out.wait_front(out_subblock_tile_count);
                        uint32_t out_read_offset = 0;

                        for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                            uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                            for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                if (bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_) {
                                    // A DataflowBuffer used as a NoC write source resolves to
                                    // get_read_ptr() + offset_bytes, matching the legacy
                                    // use<CircularBuffer::AddrSelector::READ_PTR>(cb_out) semantics.
                                    noc.async_write(
                                        cb_out,
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

                        cb_out.pop_front(out_subblock_tile_count);
                        out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                    }
                    // Pop fully padded subblocks along the row
                    if (bw == num_blocks_w_dim_ - 1) {
                        cb_out.wait_front(padded_block_tiles_w_skip);
                        cb_out.pop_front(padded_block_tiles_w_skip);
                    }
                    out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                }
                // Pop row(s) of fully padded subblocks
                if (bh == num_blocks_h_dim_ - 1) {
                    cb_out.wait_front(padded_block_tiles_h_skip);
                    cb_out.pop_front(padded_block_tiles_h_skip);
                }
#endif
                out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
            }
            out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
        }
        out_tensor_start_tile_id += MtNt;

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    cb_out.wait_front(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
    // Drain outstanding NOC writes AND atomics (sender_sem.up) before returning. Under Metal 2.0 the
    // FW kernel epilogue does not drain the kernel's outstanding NOC transactions the way the legacy
    // runtime did, so a kernel that returns with an un-acked atomic/write leaves the core "running"
    // and it never signals program completion -> dispatch process_wait hangs.
    noc.async_full_barrier();
    DPRINT("WRM end\n");  // DEBUG: matmul layer3 hang
}
