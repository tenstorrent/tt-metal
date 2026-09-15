// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

#ifdef FUSE_OP
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#endif

// This is the Metal 2.0 fork of
// reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp, which still sits beside it
// and still serves the matmul factories that have not been ported. Changes to either copy should be
// evaluated for the other until the last legacy consumer migrates and the legacy copy is retired.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// The two multicast-destination coordinate lists arrive as runtime VARARGS rather than named
// arguments, because the kernel reaches them by index inside a loop whose bound (num_x / num_y) is
// a compile-time argument, not as distinct fields. Layout: noc_x occupies varargs [0, num_x) and
// noc_y occupies [num_x, num_x + num_y).

void kernel_main() {
    constexpr bool core_has_output_block_work = static_cast<bool>(get_arg(args::core_has_output_block_work));
    constexpr bool core_in_in0_receiver_mcast_grid = static_cast<bool>(get_arg(args::core_in_in0_receiver_mcast_grid));

    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    constexpr auto in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr auto in0_last_ktile_h = get_arg(args::in0_last_ktile_h);

    // in0/in1 common args
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // in0 mcast args
    constexpr auto in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    constexpr auto in0_mcast_num_cores = get_arg(args::in0_mcast_num_cores);
    constexpr auto num_x = get_arg(args::num_x);
    constexpr auto num_y = get_arg(args::num_y);
    constexpr bool transpose_mcast = static_cast<bool>(get_arg(args::transpose_mcast));
    constexpr auto shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr auto shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_block_h = get_arg(args::in0_block_h);

    constexpr auto batch = get_arg(args::batch);

    const uint32_t sender_id = get_arg(args::sender_id);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);

    // Bases into the runtime vararg block for the two coordinate lists (see the header comment).
    constexpr uint32_t in0_mcast_noc_x_base = 0;
    constexpr uint32_t in0_mcast_noc_y_base = num_x;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb::in0);
    constexpr DataFormat in0_data_format = get_dataformat(dfb::in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    // In case we need to send multiple blocks per shard, and shard height in tiles is greater than 1
    // Than we first need to extract the sub-blocks from the shard, and then send them to the destinations
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t out_block_h = shard_height_in_tiles / num_blocks_h_dim;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = shard_read_stride * in0_block_h;

    const Noc noc;
    // dfb::in0 is the multicast staging/destination buffer; dfb::in0_sharded is the borrowed view of
    // this core's resident in0 shard that the block is extracted from.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in2(dfb::in0_sharded);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    Semaphore sender_sem(sem::in0_mcast_sender);
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    Semaphore receiver_sem(sem::in0_mcast_receiver);

    constexpr uint32_t num_remote_senders = (num_blocks_inner_dim + num_blocks_per_shard - 1) / num_blocks_per_shard;
    uint32_t remote_sender_noc_x[num_remote_senders];
    uint32_t remote_sender_noc_y[num_remote_senders];
    if constexpr (transpose_mcast) {
        uint32_t x = 0;
        uint32_t y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = get_vararg(in0_mcast_noc_x_base + x);
            remote_sender_noc_y[i] = get_vararg(in0_mcast_noc_y_base + y);
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
            }
        }
    } else {
        uint32_t x = 0;
        uint32_t y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = get_vararg(in0_mcast_noc_x_base + x);
            remote_sender_noc_y[i] = get_vararg(in0_mcast_noc_y_base + y);
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
            }
        }
    }
    receiver_sem.set(VALID);

    dfb_in2.reserve_back(batch * in0_block_num_tiles);

    const uint32_t in0_tensor_shard_read_addr = dfb_in2.get_read_ptr();
    uint32_t in0_tensor_read_addr = 0;

#ifdef FUSE_OP
    // NOT CONVERTED TO METAL 2.0 -- this block is preserved verbatim from the legacy kernel and is
    // unreachable here: no Metal 2.0 factory may define FUSE_OP. MatmulOpReceiver consumes runtime
    // arguments positionally through a `uint32_t& rt_args_idx` cursor, and Metal 2.0 kernels address
    // their arguments by name, so there is no counter to hand it. Enabling this define will fail to
    // compile (deliberately, and loudly) until MatmulOpReceiver gains a named-argument interface.
    MatmulOpReceiver fused_op_receiver = MatmulOpReceiver(
        sender_id < num_remote_senders, /* wait_for_op_signal */
        rt_args_idx,
        num_blocks_inner_dim,
        in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
    );
#endif  // FUSE_OP

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_h_dim_block_start_addr = in0_tensor_shard_read_addr;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    uint32_t block_id = block / num_blocks_per_shard;
                    // If used fused op, make block_id conform to ordering of tensor slices from all
                    // gather
#ifdef FUSE_OP
                    block_id = fused_op_receiver.align_to_slice_and_sync(block, sender_id);
#endif  // FUSE_OP

                    dfb_in0.reserve_back(in0_block_num_tiles);

                    // All cores in receiver grid need to participate in receiving regardless if they produce output
                    // work or not. Otherwise, data corruption since we mcast from and to the same buffer (eg.
                    // extract_shard_sub_blocks). If we only ever mcast with loopback src (ie. always to a different
                    // buffer), we can have just the cores that produce work participate in receiving.
                    if constexpr (core_in_in0_receiver_mcast_grid) {
                        // Set in0 semaphore value to INVALID
                        receiver_sem.set(INVALID);
                    }

                    if (block_id == sender_id) {
                        // Operand 0
                        const uint32_t in0_tensor_local_l1_write_addr = dfb_in0.get_write_ptr();

                        if constexpr (extract_shard_sub_blocks) {
                            in0_tensor_read_addr = in0_tensor_local_l1_write_addr;

                            uint32_t l1_write_extract_shard_in0 = in0_tensor_local_l1_write_addr;
                            const UnicastEndpoint self_ep;
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
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t h = 0; h < out_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            ((h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            ((out_block_h - 1) * in0_block_w * in0_single_tile_size_bytes) +
                                            (w * in0_single_tile_size_bytes);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        } else {
                            in0_tensor_read_addr = in0_tensor_current_inner_dim_block_start_addr;
                            in0_tensor_current_inner_dim_block_start_addr += in0_block_size_bytes;

                            if constexpr (in0_last_ktile_w > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            ((h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            (((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        }

                        // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        if constexpr (core_in_in0_receiver_mcast_grid) {
                            // wait for every core in receiver grid EXCLUDING myself
                            sender_sem.wait(in0_mcast_num_dests - 1);
                        } else {
                            // wait for every core in receiver grid
                            sender_sem.wait(in0_mcast_num_dests);
                        }
                        sender_sem.set(0);

                        // Now we have the block in the buffer's address, we can mcast to dests!
                        if constexpr (core_in_in0_receiver_mcast_grid) {
                            // Mcast from/to same buffer
                            if constexpr (extract_shard_sub_blocks) {
                                // multicast to every core in receiver grid EXCLUDING myself
                                // Skip if there are no other cores since this core already has the data.
                                // Note: noc_async_write_multicast[_loopback_src] may hang if called with 0 cores.
                                if constexpr (in0_mcast_num_cores > 1) {
                                    const MulticastEndpoint mcast_dst;
                                    noc.async_write_multicast(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        mcast_dst,
                                        in0_block_size_bytes,
                                        in0_mcast_num_cores - 1,
                                        {},
                                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                                         .noc_y_start = in0_mcast_dest_noc_start_y,
                                         .noc_x_end = in0_mcast_dest_noc_end_x,
                                         .noc_y_end = in0_mcast_dest_noc_end_y,
                                         .addr = in0_tensor_local_l1_write_addr},
                                        true);
                                }
                            }
                            // Mcast from one buffer to another
                            else {
                                if constexpr (in0_mcast_num_cores == 1) {
                                    // noc_async_write if we only want to copy data between buffers locally
                                    const UnicastEndpoint ucast_dst;
                                    noc.async_write(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        ucast_dst,
                                        in0_block_size_bytes,
                                        {},
                                        {.noc_x = in0_mcast_dest_noc_start_x,
                                         .noc_y = in0_mcast_dest_noc_start_y,
                                         .addr = in0_tensor_local_l1_write_addr});
                                } else {
                                    // multicast to every core in receiver grid
                                    const MulticastEndpoint mcast_dst;
                                    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        mcast_dst,
                                        in0_block_size_bytes,
                                        in0_mcast_num_cores,
                                        {},
                                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                                         .noc_y_start = in0_mcast_dest_noc_start_y,
                                         .noc_x_end = in0_mcast_dest_noc_end_x,
                                         .noc_y_end = in0_mcast_dest_noc_end_y,
                                         .addr = in0_tensor_local_l1_write_addr},
                                        true);
                                }
                            }

                            // We should also multicast the flag to destinations
                            receiver_sem.set(VALID);
                            if constexpr (in0_mcast_num_cores > 1) {
                                receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                                    noc,
                                    in0_mcast_dest_noc_start_x,
                                    in0_mcast_dest_noc_start_y,
                                    in0_mcast_dest_noc_end_x,
                                    in0_mcast_dest_noc_end_y,
                                    in0_mcast_num_cores);
                            }
                        } else {
                            // If we are not part of receiver grid, always do a regular noc_async_write_multicast to all
                            // cores in receiver grid
                            const MulticastEndpoint mcast_dst;
                            noc.async_write_multicast(
                                CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                mcast_dst,
                                in0_block_size_bytes,
                                in0_mcast_num_cores,
                                {},
                                {.noc_x_start = in0_mcast_dest_noc_start_x,
                                 .noc_y_start = in0_mcast_dest_noc_start_y,
                                 .noc_x_end = in0_mcast_dest_noc_end_x,
                                 .noc_y_end = in0_mcast_dest_noc_end_y,
                                 .addr = in0_tensor_local_l1_write_addr},
                                true);

                            // We should also multicast the flag to destinations
                            receiver_sem.set(VALID);
                            receiver_sem.set_multicast(
                                noc,
                                in0_mcast_dest_noc_start_x,
                                in0_mcast_dest_noc_start_y,
                                in0_mcast_dest_noc_end_x,
                                in0_mcast_dest_noc_end_y,
                                in0_mcast_num_cores);
                        }
                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC).

                        // Flush is required because the semaphore multicast reads receiver_sem's L1
                        // address as the source value. Without a flush, the CPU can proceed to the next
                        // iteration and overwrite receiver_sem to INVALID before the NoC has read the
                        // VALID value from L1, causing receivers to see INVALID and hang.
                        // In single-core receiver-grid configurations, semaphore multicast may be compiled out;
                        // in that case, skip the flush to avoid an unnecessary stall.
                        if constexpr (!(core_in_in0_receiver_mcast_grid && (in0_mcast_num_cores == 1))) {
                            noc.async_writes_flushed();
                        }
                    } else if constexpr (core_in_in0_receiver_mcast_grid) {
                        // Increment remote sender's semaphore using pre-computed coordinates
                        sender_sem.up(noc, remote_sender_noc_x[block_id], remote_sender_noc_y[block_id], 1);
                    }

                    if constexpr (core_in_in0_receiver_mcast_grid) {
                        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                        receiver_sem.wait(VALID);
                    }
                    dfb_in0.push_back(in0_block_num_tiles);

                    // If core does not produce output block work, free dfb::in0 immediately.
                    // This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced
                    // properly for cores that only send and have no compute / writer active. Technically, don't have to
                    // do this if dfb::in0 is not double buffered.
                    if constexpr (!core_has_output_block_work) {
                        dfb_in0.pop_front(in0_block_num_tiles);
                    }
                }
            }
            in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride;
        }
    }

    noc.async_write_barrier();
}
