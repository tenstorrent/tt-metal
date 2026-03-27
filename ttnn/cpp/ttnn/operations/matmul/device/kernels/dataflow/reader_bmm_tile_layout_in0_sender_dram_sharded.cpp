// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(2);
    constexpr uint32_t in0_last_ktile_h = get_compile_time_arg_val(3);
    // in0 mcast args
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(6);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(7);
    // block args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // in0 mcast args
    constexpr uint32_t in0_mcast_dest_noc_start_x = get_compile_time_arg_val(9);
    constexpr uint32_t in0_mcast_dest_noc_start_y = get_compile_time_arg_val(10);
    constexpr uint32_t in0_mcast_dest_noc_end_x = get_compile_time_arg_val(11);
    constexpr uint32_t in0_mcast_dest_noc_end_y = get_compile_time_arg_val(12);
    // in0 semaphore always valid
    uint32_t in0_mcast_sender_valid_semaphore = get_semaphore(get_compile_time_arg_val(13));

    constexpr uint32_t num_blocks_per_shard = get_compile_time_arg_val(14);
    constexpr uint32_t num_storage_cores = num_blocks / num_blocks_per_shard;

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg_val<uint32_t>(0);
    // if not worker core, skip
    if (worker_core_type == 0) {
        return;
    }
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const bool is_last_ktile_padded = static_cast<bool>(get_arg_val<uint32_t>(2));

    tt_l1_ptr uint32_t* in0_mcast_sender_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + num_storage_cores));

    const uint32_t sender_block_id = sender_id * num_blocks_per_shard;

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_id_in2 = get_named_compile_time_arg_val("cb_in0_sharded");  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in2(cb_id_in2);
    experimental::Semaphore<> sender_sem(get_compile_time_arg_val(4));
    experimental::Semaphore<> receiver_sem(get_compile_time_arg_val(5));

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_receiver_semaphore_addr);

    uint32_t local_read_addr = cb_in2.get_read_ptr();

    if (worker_core_type == 1) {  // mcast sender + no compute

        for (uint32_t i = 0; i < num_blocks_per_shard; ++i) {
            const uint32_t block_id = sender_block_id + i;

            // Operand 0
            l1_write_addr_in0 = cb_in0.get_write_ptr();
            if (block_id % 2 != 0) {  // double buffer
                l1_write_addr_in0 += in0_block_size_bytes;
            }

            // copy start address of block, to be used for mcasting

            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
            sender_sem.wait(in0_mcast_num_dests);
            sender_sem.set(0);

            // Now we have the block in the CB address, we can mcast to dests!

            // Zero out padded regions for the very last tile
            if constexpr (in0_last_ktile_w > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    auto in0_last_ktile_ptr = local_read_addr + in0_block_size_bytes - in0_single_tile_size_bytes;
                    pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_ptr);
                }
            }
            if constexpr (in0_last_ktile_h > 0) {
                if (is_last_ktile_padded && (i == num_blocks_per_shard - 1)) {
                    auto in0_last_ktile_ptr = local_read_addr + in0_block_size_bytes - in0_single_tile_size_bytes;
                    pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                }
            }

#ifndef SKIP_MCAST
            // num_dests must not include source, since we are NOT really doing a local copy!
            experimental::MulticastEndpoint mcast_dst;
            noc.async_write_multicast(
                experimental::CoreLocalMem<uint32_t>(local_read_addr),
                mcast_dst,
                in0_block_size_bytes,
                in0_mcast_num_cores - 1,
                {},
                {.noc_x_start = in0_mcast_dest_noc_start_x,
                 .noc_y_start = in0_mcast_dest_noc_start_y,
                 .noc_x_end = in0_mcast_dest_noc_end_x,
                 .noc_y_end = in0_mcast_dest_noc_end_y,
                 .addr = l1_write_addr_in0},
                true);
#endif

            receiver_sem.set_multicast(
                noc,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
                in0_mcast_num_cores - 1);

            local_read_addr += in0_block_size_bytes;
        }

    } else if (worker_core_type == 2) {  // mcast sender + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = block / num_blocks_per_shard;

            cb_in0.reserve_back(in0_block_num_tiles);
            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);

            if (block_id == sender_id) {
                uint32_t l1_write_addr_in0 = cb_in0.get_write_ptr();
                // copy start address of block, to be used for mcasting

                // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                sender_sem.wait(in0_mcast_num_dests - 1);
                sender_sem.set(0);

                // Zero out padded regions for the very last tile
                if constexpr (in0_last_ktile_w > 0) {
                    if (is_last_ktile_padded && (block == num_blocks - 1)) {
                        auto in0_last_ktile_ptr = local_read_addr + in0_block_size_bytes - in0_single_tile_size_bytes;
                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_ptr);
                    }
                }
                if constexpr (in0_last_ktile_h > 0) {
                    if (is_last_ktile_padded && (block == num_blocks - 1)) {
                        auto in0_last_ktile_ptr = local_read_addr + in0_block_size_bytes - in0_single_tile_size_bytes;
                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(in0_last_ktile_ptr);
                    }
                }
#ifndef SKIP_MCAST
                experimental::MulticastEndpoint mcast_dst;
                noc.async_write_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
                    experimental::CoreLocalMem<uint32_t>(local_read_addr),
                    mcast_dst,
                    in0_block_size_bytes,
                    in0_mcast_num_cores,
                    {},
                    {.noc_x_start = in0_mcast_dest_noc_start_x,
                     .noc_y_start = in0_mcast_dest_noc_start_y,
                     .noc_x_end = in0_mcast_dest_noc_end_x,
                     .noc_y_end = in0_mcast_dest_noc_end_y,
                     .addr = l1_write_addr_in0},
                    true);
#endif
                noc_semaphore_set_multicast_loopback_src(
                    in0_mcast_sender_valid_semaphore, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores);

                local_read_addr += in0_block_size_bytes;

            } else {
                // Atomic increment source core counter
                sender_sem.up(noc, in0_mcast_sender_noc_x[block_id], in0_mcast_sender_noc_y[block_id], 1);
            }

            receiver_sem.wait(VALID);
            cb_in0.push_back(in0_block_num_tiles);
        }
    } else {  // mcast receiver + compute

        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = block / num_blocks_per_shard;

            // get the mcast sender noc

            // Operand 0
            cb_in0.reserve_back(in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);
            // Atomic increment source core counter
            sender_sem.up(noc, in0_mcast_sender_noc_x[block_id], in0_mcast_sender_noc_y[block_id], 1);
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            receiver_sem.wait(VALID);

            cb_in0.push_back(in0_block_num_tiles);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
