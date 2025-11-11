// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "pad_tile.hpp"

void kernel_main() {
    constexpr bool core_has_output_block_work = (bool)get_compile_time_arg_val(0);
    constexpr bool core_in_in0_receiver_mcast_grid = (bool)get_compile_time_arg_val(1);

    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(4);

    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(7);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(8));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(10);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(11);
    constexpr uint32_t num_x = get_compile_time_arg_val(12);
    constexpr uint32_t num_y = get_compile_time_arg_val(13);
    constexpr bool transpose_mcast = (bool)get_compile_time_arg_val(14);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(17);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(18);

    constexpr uint32_t batch = get_compile_time_arg_val(19);
    constexpr bool fuse_op = (bool)get_compile_time_arg_val(20);

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
    constexpr uint32_t out_block_h = shard_height_in_tiles / num_blocks_h_dim;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = shard_read_stride * in0_block_h;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    // L1 array
    constexpr uint32_t cb_l1_array = tt::CBIndex::c_6;
    uint32_t in0_mcast_sender_semaphore_valid_addr = get_write_ptr(cb_l1_array);
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_valid_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_valid_addr);
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in0_mcast_sender_semaphore_valid_addr_ptr[0] =
        VALID;  // Load const 1 to be used as semaphore valid value sent from sender to receivers

    constexpr uint32_t num_remote_senders = (num_blocks_inner_dim + num_blocks_per_shard - 1) / num_blocks_per_shard;
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

    uint32_t in0_tensor_shard_read_addr = get_read_ptr(cb_id_in2);
    uint32_t in0_tensor_read_addr = in0_tensor_shard_read_addr;

    cb_reserve_back(cb_id_in0, in0_block_num_tiles);

    const uint32_t block_id = 0;
    if (sender_id == block_id) {
        // Operand 0
        uint32_t in0_tensor_local_l1_write_addr = get_write_ptr(cb_id_in0);

        noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests - 1);
        noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_tensor_local_l1_write_addr;

        noc_async_write_multicast_loopback_src(
            in0_tensor_read_addr, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores, true);

        // We should also multicast the flag to destinations
        noc_semaphore_set_multicast_loopback_src(
            in0_mcast_sender_semaphore_valid_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);

        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
        // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
        // which means data could be changed before
        //  write is issued.
        noc_async_writes_flushed();
#endif
    } else {
        noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);
        uint64_t in0_mcast_sender_semaphore_noc_addr = remote_sender_noc_addrs[block_id];

        // Atomic increment source core counter
        noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
    }

    DPRINT << "Pushing back " << in0_block_num_tiles << " tiles to cb_id_in0" << ENDL();
    DPRINT << TileSlice(cb_id_in0, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 16}, true, true)
           << ENDL();
    cb_push_back(cb_id_in0, in0_block_num_tiles);
    noc_async_write_barrier();
}
