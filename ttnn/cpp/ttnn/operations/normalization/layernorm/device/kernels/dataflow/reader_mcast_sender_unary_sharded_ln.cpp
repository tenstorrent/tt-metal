// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "noc_addr_utils.h"

namespace df = norm::layernorm::device::kernels::dataflow;

/**
 * @brief This kernel implements the sender (coordinator) logic for
 *        the mean and variance calculations for the sharded layernorm
 *        kernels
 *
 * @details The kernel's objective is to coordinate a distributed (sharded)
 * mean and variance calculation, where shards are placed on other cores
 * in the communication network.The responsibilities of this kernel are:
 * 1. Do its partial mean/variance reduction for its assigned tiles
 * 2. Coordinate the waiting on partial results from the other cores
 * 3. Read partials from other cores to do its global combine
 *    for its assigned tiles.
 * 4. Coordinate the waiting on all global combined results to be ready
 * 5. Collect all global combined results and multicast to all cores
 *
 * @note If the reduce is two-stage, the kernel additionally waits
 *       on the combined results from the first stage and uses them
 *       in its own combine
 */
void kernel_main() {
    // ============================================================================
    // Kernel setup
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Compile-time arguments
    // ---------------------------------------------------------------------------
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr uint32_t block_h_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_all_to_all_workers_first_stage = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_tiles_per_worker_last = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_per_worker_last_bytes = get_compile_time_arg_val(9);
    constexpr bool row_major = (bool)get_compile_time_arg_val(10);
    constexpr uint32_t num_x = get_compile_time_arg_val(11);
    constexpr uint32_t num_y = get_compile_time_arg_val(12);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(13);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(14);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(15);
    uint32_t reduce_second_stage_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    constexpr bool rms_norm = get_compile_time_arg_val(17) == 1;
    constexpr bool use_welford = get_compile_time_arg_val(18) == 1;

    // ---------------------------------------------------------------------------
    // Runtime arguments
    // ---------------------------------------------------------------------------
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);
    const uint32_t start_x = get_arg_val<uint32_t>(4);
    const uint32_t start_y = get_arg_val<uint32_t>(5);

    df::L1Ptr in0_remote_noc_x = (df::L1Ptr)(get_arg_addr(6));
    df::L1Ptr in0_remote_noc_y = (df::L1Ptr)(get_arg_addr(6 + num_x));

    // ---------------------------------------------------------------------------
    // CB definitions
    // ---------------------------------------------------------------------------
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce

    // ---------------------------------------------------------------------------
    // Set up constants for the kernel
    // ---------------------------------------------------------------------------
    const uint32_t single_tile_size_bytes = get_tile_size(rms_norm ? cb_ex_partial2 : cb_ex_partial);

    // Compute the NOC addresses for remote cores that interact with this core
    df::RemoteNocAddrs<num_blocks> remote_noc_addrs{};
    df::compute_single_stage_noc_addrs<row_major, num_blocks>(
        remote_noc_addrs, in0_remote_noc_x, in0_remote_noc_y, start_x, start_y, num_x, num_y);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    // ============================================================================
    // Main kernel worker function
    // Performs partial reduction for its assigned tiles, coordinates
    // the waiting on partial and combined results for all cores, and performs
    // the multicast of the final results to all cores
    // ============================================================================
    const auto& global_reduce_sender = [&](const uint32_t cb_partial,
                                           const uint32_t cb_external,
                                           const uint32_t cb_ex,
                                           const uint32_t cb_ex_global,
                                           const uint32_t cb_reduce_first_stage,
                                           const uint32_t num_tiles_scaler) __attribute__((always_inline)) {
        // ============================================================================
        // Partial reduction
        // Partials stored `cb_partial`
        // ============================================================================

        // Wait for this core's partial results to be ready
        cb_wait_front(cb_partial, block_h * num_tiles_scaler);

        // Wait for partial results from the other cores to be ready.
        // Once all cores finish partials, notify the cores to
        // start reading remote data and doing their global combines
        if constexpr (num_blocks > 1) {
            *reduce_sender_semaphore_addr_ptr = VALID;
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks - 1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
            noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);
        }

        // ============================================================================
        // Combine partial results
        // Read from the partial buffers into the external buffer `cb_external`.
        // Will read a total of:
        // (num_blocks_first_stage + num_blocks_second_stage - 1) * num_tiles_scaler
        // tiles for each assigned tile row (or column, if not row-major).
        // For the second stage, read from `cb_reduce_first_stage` instead of `cb_partial`,
        // as it will contain the combined results from the first stage.
        // Combined results written to `cb_ex`.
        // ============================================================================

        // ---------------------------------------------------------------------------
        // Read remote partial data
        // ---------------------------------------------------------------------------
        uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
        uint32_t l1_read_addr_ex = 0;
        uint32_t block_index_stride = 0;
        if constexpr (use_two_stage_reduce) {
            l1_read_addr_ex = get_read_ptr(cb_reduce_first_stage);
            if constexpr (row_major) {
                block_index_stride = num_x;
            } else {
                block_index_stride = num_y;
            }
        }
        for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
            // Read in the partials from the cores in the same row
            cb_reserve_back(cb_external, num_blocks_first_stage * num_tiles_scaler);
            uint32_t l1_write_addr_external = get_write_ptr(cb_external);
            for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
                uint64_t noc_addr_ex_par = remote_noc_addrs[block] | l1_read_addr_ex_par;
                noc_async_read_one_packet(
                    noc_addr_ex_par, l1_write_addr_external, num_tiles_scaler * single_tile_size_bytes);
                l1_write_addr_external += num_tiles_scaler * single_tile_size_bytes;
            }
            l1_read_addr_ex_par += num_tiles_scaler * single_tile_size_bytes;
            noc_async_read_barrier();
            cb_push_back(cb_external, num_blocks_first_stage * num_tiles_scaler);

            // ---------------------------------------------------------------------------
            // Handle the two-stage reduce
            // ---------------------------------------------------------------------------
            if constexpr (use_two_stage_reduce) {
                // Wait for the signal to read in the combined results from
                // the first stage (from cores in our core column) are ready
                if (i == 0) {
                    noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage - 1);
                    noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                }

                // Read in the combined results from the cores in our column
                // into the external buffer
                uint32_t curr_block_index = block_index_stride;
                cb_reserve_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
                for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                    uint64_t noc_addr_ex = remote_noc_addrs[curr_block_index] | l1_read_addr_ex;
                    noc_async_read_one_packet(
                        noc_addr_ex, l1_write_addr_external, num_tiles_scaler * single_tile_size_bytes);
                    curr_block_index += block_index_stride;
                    l1_write_addr_external += num_tiles_scaler * single_tile_size_bytes;
                }
                l1_read_addr_ex += num_tiles_scaler * single_tile_size_bytes;
                noc_async_read_barrier();
                cb_push_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
            }
        }

        // ---------------------------------------------------------------------------
        // Wait for all final combined results to be ready
        // ---------------------------------------------------------------------------

        // Our results
        cb_wait_front(cb_ex, num_tiles_per_worker * num_tiles_scaler);

        // Other cores' results
        if constexpr (num_all_to_all_workers_first_stage > 1) {
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_all_to_all_workers_first_stage - 1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
        }

        // ============================================================================
        // Gather all final combined results and multicast to all cores.
        // Read from `cb_ex` into `cb_ex_global`, multicast `cb_ex_global` to all cores
        // ============================================================================

        // Gather final results
        l1_read_addr_ex = get_read_ptr(cb_ex);
        uint32_t l1_write_addr_ex_global = get_write_ptr(cb_ex_global);
        cb_reserve_back(cb_ex_global, block_h * num_tiles_scaler);
        for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
            uint64_t noc_addr_ex = remote_noc_addrs[block] | l1_read_addr_ex;
            uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1 ? num_tiles_per_worker_last_bytes
                                                                                       : num_tiles_per_worker_bytes;
            if constexpr (num_tiles_per_worker_bytes <= NOC_MAX_BURST_SIZE) {
                noc_async_read_one_packet(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
            } else {
                noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
            }
            l1_write_addr_ex_global += num_tiles_scaler * num_tiles_bytes;
        }
        noc_async_read_barrier();

        // Multicast
        uint32_t l1_read_addr_ex_global = get_read_ptr(cb_ex_global);
        cb_push_back(cb_ex_global, block_h * num_tiles_scaler);
        if constexpr (num_blocks > 1) {
            for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
                *reduce_sender_semaphore_addr_ptr = block + 2;

                uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1
                                               ? num_tiles_per_worker_last_bytes
                                               : num_tiles_per_worker_bytes;

                noc_async_write_multicast(
                    l1_read_addr_ex_global,
                    multicast_data_noc | l1_read_addr_ex_global,
                    num_tiles_scaler * num_tiles_bytes,
                    num_blocks - 1,
                    true);
                noc_semaphore_set_multicast(
                    reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);

                l1_read_addr_ex_global += num_tiles_scaler * num_tiles_bytes;
                noc_async_write_barrier();
            }
        }
    };

    if constexpr (!rms_norm) {
        // Welford processes 2 tiles at a time (mean and var)
        global_reduce_sender(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global, cb_ex, use_welford ? 2 : 1);
    }

    if constexpr (!use_welford) {
        global_reduce_sender(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global, cb_ex2, 1);
    }
}
