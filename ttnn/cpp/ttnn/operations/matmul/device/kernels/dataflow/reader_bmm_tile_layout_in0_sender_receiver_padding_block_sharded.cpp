// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    constexpr bool core_has_output_block_work = (bool)get_compile_time_arg_val(0);
    constexpr bool core_in_in0_receiver_mcast_grid = (bool)get_compile_time_arg_val(1);

    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(3);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(6));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(7);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(8);
    constexpr uint32_t num_x = get_compile_time_arg_val(9);
    constexpr uint32_t num_y = get_compile_time_arg_val(10);
    constexpr bool transpose_mcast = (bool)get_compile_time_arg_val(11);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(14);

    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr bool fuse_op = (bool)get_compile_time_arg_val(16);

    uint32_t rt_args_idx = 0;
    const uint32_t sender_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_x)));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_y)));

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    // In case we need to send multiple blocks per shard, and shard height in tiles is greater than 1
    // Than we first need to extract the sub-blocks from the shard, and then send them to the destinations
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    // L1 array
    constexpr uint32_t cb_l1_array = tt::CB::c_in5;
    uint32_t in0_mcast_sender_semaphore_valid_addr = get_write_ptr(cb_l1_array);
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_valid_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_valid_addr);
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in0_mcast_sender_semaphore_valid_addr_ptr[0] =
        VALID;  // Load const 1 to be used as semaphore valid value sent from sender to receivers

    constexpr uint32_t num_remote_senders = (num_blocks + num_blocks_per_shard - 1) / num_blocks_per_shard;
    uint64_t remote_sender_noc_addrs[num_remote_senders];
    if constexpr (transpose_mcast) {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_addrs[i] =
                get_noc_addr(in0_mcast_noc_x[x], in0_mcast_noc_y[y], in0_mcast_sender_semaphore_addr);
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
            }
        }
    } else {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_addrs[i] =
                get_noc_addr(in0_mcast_noc_x[x], in0_mcast_noc_y[y], in0_mcast_sender_semaphore_addr);
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
            }
        }
    }
    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

    uint64_t in0_mcast_receiver_semaphore_noc_addr =
        in0_multicast_data_noc | (uint64_t)in0_mcast_receiver_semaphore_addr;

    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, VALID);

    cb_reserve_back(cb_id_in2, batch * in0_block_num_tiles);

    uint32_t local_read_addr = 0;
    uint64_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        noc_shard_read_start_addr = get_noc_addr(get_read_ptr(cb_id_in2));
    } else {
        local_read_addr = get_read_ptr(cb_id_in2);
    }

    MatmulOpReceiver fused_op_receiver;
    if constexpr (fuse_op) {
        fused_op_receiver = MatmulOpReceiver(
            sender_id < num_remote_senders, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks,
            in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
        );
    }

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            uint32_t block_id = block / num_blocks_per_shard;
            if constexpr (fuse_op) {  // If used fused op, make block_id conform to ordering of tensor slices from all
                                      // gather
                block_id = fused_op_receiver.align_to_slice_and_sync(block, sender_id);
            }

            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            // All cores in receiver grid need to participate in receiving regardless if they produce output work or
            // not. Otherwise, data corruption since we mcast from and to the same CB (eg. extract_shard_sub_blocks). If
            // we only ever mcast with loopback src (ie. always to a different CB), we can have just the cores that
            // produce work participate in receiving.
            if constexpr (core_in_in0_receiver_mcast_grid) {
                // Set in0 semaphore value to INVALID
                noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);
            }

            if (block_id == sender_id) {
                // Operand 0
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

                if constexpr (extract_shard_sub_blocks) {
                    local_read_addr = l1_write_addr_in0;

                    uint32_t l1_write_extract_shard_in0 = l1_write_addr_in0;
                    uint64_t noc_shard_read_addr = noc_shard_read_start_addr;
                    noc_shard_read_start_addr += shard_read_width;

                    for (uint32_t i = 0; i < shard_height_in_tiles; i++) {
                        noc_async_read(noc_shard_read_addr, l1_write_extract_shard_in0, shard_read_width);

                        l1_write_extract_shard_in0 += shard_read_width;
                        noc_shard_read_addr += shard_read_stride;
                    }

                    noc_async_read_barrier();
                }

                // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its
                // value should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next
                // block
                if constexpr (core_in_in0_receiver_mcast_grid) {
                    // wait for every core in receiver grid EXCLUDING myself
                    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests - 1);
                } else {
                    // wait for every core in receiver grid
                    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                }
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                // Now we have the block in the CB address, we can mcast to dests!
                uint64_t in0_multicast_data_addr = in0_multicast_data_noc | l1_write_addr_in0;

                if constexpr (core_in_in0_receiver_mcast_grid) {
                    // Mcast from/to same CB
                    if constexpr (extract_shard_sub_blocks) {
                        // multicast to every core in receiver grid EXCLUDING myself
                        // Skip if there are no other cores since this core already has the data.
                        // Note: noc_async_write_multicast[_loopback_src] may hang if called with 0 cores.
                        if constexpr (in0_mcast_num_cores > 1) {
                            noc_async_write_multicast(
                                local_read_addr,
                                in0_multicast_data_addr,
                                in0_block_size_bytes,
                                in0_mcast_num_cores - 1,
                                true,
                                true);
                        }
                    }
                    // Mcast from different CB to another CB
                    else {
                        if constexpr (in0_mcast_num_cores == 1) {
                            // noc_async_write if we only want to copy data between CB locally
                            noc_async_write(local_read_addr, in0_multicast_data_addr, in0_block_size_bytes);
                        } else {
                            // multicast to every core in receiver grid
                            noc_async_write_multicast_loopback_src(
                                local_read_addr,
                                in0_multicast_data_addr,
                                in0_block_size_bytes,
                                in0_mcast_num_cores,
                                true,
                                true);
                        }
                    }

                    // We should also multicast the flag to destinations
                    if constexpr (in0_mcast_num_cores == 1) {
                        // All work is done on one core (the current one).
                        // noc_semaphore_set_multicast_loopback_src is a no-op in this case.
                        // Data needs to be written directly in the core.
                        in0_mcast_receiver_semaphore_addr_ptr[0] = in0_mcast_sender_semaphore_valid_addr_ptr[0];
                    } else {
                        noc_semaphore_set_multicast_loopback_src(
                            in0_mcast_sender_semaphore_valid_addr,
                            in0_mcast_receiver_semaphore_noc_addr,
                            in0_mcast_num_cores);
                    }
                } else {
                    // If we are not part of receiver grid, always do a regular noc_async_write_multicast to all cores
                    // in receiver grid
                    noc_async_write_multicast(
                        local_read_addr,
                        in0_multicast_data_addr,
                        in0_block_size_bytes,
                        in0_mcast_num_cores,
                        true,
                        true);

                    // We should also multicast the flag to destinations
                    noc_semaphore_set_multicast(
                        in0_mcast_sender_semaphore_valid_addr,
                        in0_mcast_receiver_semaphore_noc_addr,
                        in0_mcast_num_cores);
                }
                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc even though cmd bufs are different
                // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency which means data could be changed before
                //  write is issued.
                noc_async_writes_flushed();
#endif

                local_read_addr += in0_block_size_bytes;
            } else if constexpr (core_in_in0_receiver_mcast_grid) {
                uint64_t in0_mcast_sender_semaphore_noc_addr = remote_sender_noc_addrs[block_id];

                // Atomic increment source core counter
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            }

            if constexpr (core_in_in0_receiver_mcast_grid) {
                // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
            }

            cb_push_back(cb_id_in0, in0_block_num_tiles);

            // If core does not produce output block work, free cb_id_in0 immediately.
            // This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced properly for
            // cores that only send and have no compute / writer active. Technically, don't have to do this if cb_id_in0
            // is not double buffered.
            if constexpr (!core_has_output_block_work) {
                cb_pop_front(cb_id_in0, in0_block_num_tiles);
            }
        }
    }
}
