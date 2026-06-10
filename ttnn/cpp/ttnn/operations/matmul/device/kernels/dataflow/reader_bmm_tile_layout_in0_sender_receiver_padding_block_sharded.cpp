// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    constexpr bool core_has_output_block_work = (bool)get_compile_time_arg_val(0);
    constexpr bool core_in_in0_receiver_mcast_grid = (bool)get_compile_time_arg_val(1);

    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_last_ktile_h = get_compile_time_arg_val(5);

    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(8);
    // in0 mcast args
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(11);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(12);
    constexpr uint32_t num_x = get_compile_time_arg_val(13);
    constexpr uint32_t num_y = get_compile_time_arg_val(14);
    constexpr bool transpose_mcast = (bool)get_compile_time_arg_val(15);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(18);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(19);

    constexpr uint32_t batch = get_compile_time_arg_val(20);
    constexpr bool fuse_op = (bool)get_compile_time_arg_val(21);

    uint32_t rt_args_idx = 0;
    const uint32_t sender_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_x)));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_y)));

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_id_in2 = get_named_compile_time_arg_val("cb_in0_sharded");  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    // In case we need to send multiple blocks per shard, and shard height in tiles is greater than 1
    // Than we first need to extract the sub-blocks from the shard, and then send them to the destinations
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t out_block_h = shard_height_in_tiles / num_blocks_h_dim;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = shard_read_stride * in0_block_h;

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in2(cb_id_in2);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    Semaphore<> sender_sem(get_compile_time_arg_val(9));
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    Semaphore<> receiver_sem(get_compile_time_arg_val(10));

    constexpr uint32_t num_remote_senders = (num_blocks_inner_dim + num_blocks_per_shard - 1) / num_blocks_per_shard;
    uint32_t remote_sender_noc_x[num_remote_senders];
    uint32_t remote_sender_noc_y[num_remote_senders];
    if constexpr (transpose_mcast) {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = in0_mcast_noc_x[x];
            remote_sender_noc_y[i] = in0_mcast_noc_y[y];
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
            }
        }
    } else {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = in0_mcast_noc_x[x];
            remote_sender_noc_y[i] = in0_mcast_noc_y[y];
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
            }
        }
    }
    receiver_sem.set(VALID);

    // mcast_pipe (R6 role-flip): every grid core runs BOTH faces of the channel over the rotating
    // rounds. SENDER face below; the per-round RECEIVER face is built inside the loop (its ack
    // target rotates with block_id). One count works here: the factory always sets
    // in0_mcast_num_dests == in0_mcast_num_cores. The Pipe never counts self, so in-grid cores
    // pass num_dests - 1. Loopback is inferred per send(): extract (src == dst, block already in
    // cb_in0) -> EXCLUDE; non-extract (cb_in2 -> cb_in0) -> INCLUDE; out-of-grid -> EXCLUDE.
    // In-grid single-core (active == 0) collapses to the local-copy degenerate.
    constexpr uint32_t in0_pipe_active_cores =
        core_in_in0_receiver_mcast_grid ? in0_mcast_num_dests - 1 : in0_mcast_num_dests;
    dataflow_kernel_lib::Pipe<> in0_send_pipe(
        noc,
        dataflow_kernel_lib::McastRect{
            in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y},
        in0_pipe_active_cores,
        receiver_sem,  // data ready (S->R level flag)
        sender_sem);   // consumed (R->S counter)

    cb_in2.reserve_back(batch * in0_block_num_tiles);

    uint32_t in0_tensor_shard_read_addr = cb_in2.get_read_ptr();
    uint32_t in0_tensor_read_addr = 0;

    MatmulOpReceiver fused_op_receiver;
    if constexpr (fuse_op) {
        fused_op_receiver = MatmulOpReceiver(
            sender_id < num_remote_senders, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
        );
    }

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_h_dim_block_start_addr = in0_tensor_shard_read_addr;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    uint32_t block_id = block / num_blocks_per_shard;
                    // If used fused op, make block_id conform to ordering of tensor slices from all
                    // gather
                    if constexpr (fuse_op) {
                        block_id = fused_op_receiver.align_to_slice_and_sync(block, sender_id);
                    }

                    cb_in0.reserve_back(in0_block_num_tiles);

                    // All cores in receiver grid need to participate in receiving regardless if they produce output
                    // work or not. Otherwise, data corruption since we mcast from and to the same CB (eg.
                    // extract_shard_sub_blocks). If we only ever mcast with loopback src (ie. always to a different
                    // CB), we can have just the cores that produce work participate in receiving.
                    if constexpr (core_in_in0_receiver_mcast_grid) {
                        // Set in0 semaphore value to INVALID
                        receiver_sem.set(INVALID);
                    }

                    if (block_id == sender_id) {
                        // Operand 0
                        uint32_t in0_tensor_local_l1_write_addr = cb_in0.get_write_ptr();

                        if constexpr (extract_shard_sub_blocks) {
                            in0_tensor_read_addr = in0_tensor_local_l1_write_addr;

                            uint32_t l1_write_extract_shard_in0 = in0_tensor_local_l1_write_addr;
                            UnicastEndpoint self_ep;
                            uint32_t noc_shard_read_l1_addr = in0_tensor_current_inner_dim_block_start_addr;

                            for (uint32_t i = 0; i < out_block_h; i++) {
                                noc.async_read(
                                    self_ep,
                                    CoreLocalMem<uint32_t>(l1_write_extract_shard_in0),
                                    shard_read_width,
                                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = noc_shard_read_l1_addr},
                                    {});
                                l1_write_extract_shard_in0 += shard_read_width;
                                noc_shard_read_l1_addr += shard_read_stride;
                            }

                            in0_tensor_current_inner_dim_block_start_addr += shard_read_width;

                            noc.async_read_barrier();

                            if constexpr (in0_last_ktile_w > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t h = 0; h < out_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            (out_block_h - 1) * in0_block_w * in0_single_tile_size_bytes +
                                            w * in0_single_tile_size_bytes;
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        } else {
                            in0_tensor_read_addr = in0_tensor_current_inner_dim_block_start_addr;
                            in0_tensor_current_inner_dim_block_start_addr += in0_block_size_bytes;

                            if constexpr (in0_last_ktile_w > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            ((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes;
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        }

                        // mcast_pipe SENDER face: send() absorbs the ack wait (PRE_HANDSHAKE,
                        // in0_pipe_active_cores), the data mcast (mode inferred: extract src==dst ->
                        // EXCLUDE n-1; non-extract -> INCLUDE n; out-of-grid -> EXCLUDE n), the VALID
                        // flag mcast on the same VC, and the flush. In-grid single-core collapses to
                        // a local copy with no handshake/flush (raw skipped both there too). The
                        // sender does NOT wait its own flag: its block is in place by construction.
                        in0_send_pipe.send(in0_tensor_read_addr, in0_tensor_local_l1_write_addr, in0_block_size_bytes);
                    } else if constexpr (core_in_in0_receiver_mcast_grid) {
                        // mcast_pipe RECEIVER face: rebuilt per round, the ack target rotates with
                        // block_id. receive() acks the sender, waits VALID, clears the flag (H11);
                        // the top-of-loop INVALID reset stays raw — it also clears the stale VALID
                        // this core's own sender round leaves behind.
                        dataflow_kernel_lib::Pipe<> in0_recv_pipe(
                            noc,
                            dataflow_kernel_lib::McastRect::single_core(
                                remote_sender_noc_x[block_id], remote_sender_noc_y[block_id]),
                            1,
                            receiver_sem,
                            sender_sem);
                        in0_recv_pipe.receive();
                    }
                    cb_in0.push_back(in0_block_num_tiles);

                    // If core does not produce output block work, free cb_id_in0 immediately.
                    // This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced
                    // properly for cores that only send and have no compute / writer active. Technically, don't have to
                    // do this if cb_id_in0 is not double buffered.
                    if constexpr (!core_has_output_block_work) {
                        cb_in0.pop_front(in0_block_num_tiles);
                    }
                }
            }
            in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride;
        }
    }

    noc.async_write_barrier();
}
