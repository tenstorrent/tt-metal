// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_addr_utils.h"

namespace df = norm::layernorm::device::kernels::dataflow;

/**
 * @brief This kernel implements reader (non-coordinator, i.e. non-sender) logic for
 *        the mean and variance calculations for the sharded layernorm
 *        kernels
 *
 * @details The kernel's objective is to handle this core's synchronization
 * with the coordinator. It does the following:
 * 1. Wait for this core's partial to results to be ready and notify the coordinator
 * 2. Wait for the coordinator to tell us when the other cores' partial results are ready
 * 3. Read the other cores' partial results so that this core can do its global combine
 * 4. Notify the coordinator when we've finished our combine
 * 5. Receive the final global mean and variance results multicasted from the coordinator
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
    const bool is_all_to_all_worker = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_all_to_all_workers = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_worker = get_compile_time_arg_val(6);
    constexpr uint32_t num_tiles_per_worker_last = get_compile_time_arg_val(7);
    constexpr bool row_major = (bool)get_compile_time_arg_val(8);
    constexpr uint32_t num_x = get_compile_time_arg_val(9);
    constexpr uint32_t num_y = get_compile_time_arg_val(10);
    constexpr bool use_two_stage_reduce = (bool)get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);
    uint32_t reduce_second_stage_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    constexpr bool rms_norm = get_compile_time_arg_val(15) == 1;
    constexpr bool use_welford = get_compile_time_arg_val(16) == 1;

    // ---------------------------------------------------------------------------
    // Runtime arguments
    // ---------------------------------------------------------------------------
    const bool is_last_all_to_all_worker = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(1);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(2);
    const uint32_t start_x = get_arg_val<uint32_t>(3);
    const uint32_t start_y = get_arg_val<uint32_t>(4);
    df::L1Ptr in0_remote_noc_x = (df::L1Ptr)(get_arg_addr(5));
    df::L1Ptr in0_remote_noc_y = (df::L1Ptr)(get_arg_addr(5 + num_x));

    // ---------------------------------------------------------------------------
    // CB definitions
    // ---------------------------------------------------------------------------
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;          // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;  // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;          // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce

    // ---------------------------------------------------------------------------
    // Set up constants for the kernel
    // ---------------------------------------------------------------------------
    // This is actually "num_tile_rows_to_read" but since column-major
    // reads columns instead of rows, that's not a great name.
    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;
    const uint32_t single_tile_size_bytes = get_tile_size(rms_norm ? cb_ex_partial2 : cb_ex_partial);  // tile size

    // Compute the NOC addresses for remote cores that interact with this core
    constexpr df::NumNocAddrs num_remote_noc_addrs_first_stage = is_all_to_all_worker ? num_blocks_first_stage : 1;
    constexpr df::NumNocAddrs num_remote_noc_addrs_second_stage = is_all_to_all_worker ? num_blocks_second_stage : 1;
    df::RemoteNocAddrs<num_remote_noc_addrs_first_stage> remote_noc_addrs_first_stage{};
    df::RemoteNocAddrs<num_remote_noc_addrs_second_stage> remote_noc_addrs_second_stage{};
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            df::compute_two_stage_noc_addrs<
                row_major,
                num_remote_noc_addrs_first_stage,
                num_remote_noc_addrs_second_stage>(
                remote_noc_addrs_first_stage,
                remote_noc_addrs_second_stage,
                in0_remote_noc_x,
                in0_remote_noc_y,
                start_x,
                start_y,
                num_x,
                num_y);
        } else {
            df::compute_single_stage_noc_addrs<row_major, num_remote_noc_addrs_first_stage>(
                remote_noc_addrs_first_stage, in0_remote_noc_x, in0_remote_noc_y, start_x, start_y, num_x, num_y);
        }
    } else {
        remote_noc_addrs_first_stage[0] = get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], 0);
    }

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(in0_remote_noc_x[0], in0_remote_noc_y[0], reduce_receiver_semaphore_addr);
    const uint64_t reduce_second_stage_receiver_semaphore_noc_addr =
        remote_noc_addrs_second_stage[0] | reduce_second_stage_semaphore_addr;

    // ============================================================================
    // Main kernel worker function
    // Waits on partial reduction, syncs with coordinator, reads
    // from other cores, signals when combine is done, receives multicast
    // ============================================================================
    const auto& global_reduce_receiver = [&](const uint32_t cb_partial,
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

        // Notify the sender that this core's partial results are ready
        noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
        noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);

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
        if constexpr (is_all_to_all_worker) {
            // ---------------------------------------------------------------------------
            // Read remote partial data
            // ---------------------------------------------------------------------------
            uint32_t l1_read_addr_ex_par = get_read_ptr(cb_partial);
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes * num_tiles_scaler;
            uint32_t l1_read_addr_ex = 0;
            if constexpr (use_two_stage_reduce) {
                l1_read_addr_ex = get_read_ptr(cb_reduce_first_stage);
            }
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                // Read from other cores in our core row
                cb_reserve_back(cb_external, num_blocks_first_stage * num_tiles_scaler);
                uint32_t l1_write_addr_external = get_write_ptr(cb_external);
                for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    uint64_t noc_addr_ex_par = remote_noc_addrs_first_stage[block] | l1_read_addr_ex_par;
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
                    if (is_second_stage_reader) {
                        // Wait for the signal that the combined results
                        // from the first stage are ready
                        if (i == 0) {
                            noc_semaphore_wait(reduce_second_stage_semaphore_addr_ptr, num_blocks_second_stage - 1);
                            noc_semaphore_set(reduce_second_stage_semaphore_addr_ptr, 0);
                        }

                        // Read from other cores in our core column
                        cb_reserve_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            uint64_t noc_addr_ex = remote_noc_addrs_second_stage[block + 1] | l1_read_addr_ex;
                            noc_async_read_one_packet(
                                noc_addr_ex, l1_write_addr_external, num_tiles_scaler * single_tile_size_bytes);
                            l1_write_addr_external += num_tiles_scaler * single_tile_size_bytes;
                        }
                        l1_read_addr_ex += num_tiles_scaler * single_tile_size_bytes;
                        noc_async_read_barrier();
                        cb_push_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
                    } else {
                        // If we're not a second stage reader (i.e. we're not in the top
                        // row of cores), we don't do any additional combines, so we just
                        // do a dummy push so that we move in lockstep with the other cores
                        cb_reserve_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
                        cb_push_back(cb_external, (num_blocks_second_stage - 1) * num_tiles_scaler);
                    }
                }
            }

            // ---------------------------------------------------------------------------
            // Wait for all final combined results to be ready and notify sender
            // ---------------------------------------------------------------------------
            if constexpr (use_two_stage_reduce) {
                if (is_second_stage_reader) {
                    cb_wait_front(cb_ex, num_tiles_to_read * num_tiles_scaler);
                    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
                } else {
                    cb_wait_front(cb_reduce_first_stage, num_tiles_to_read * num_tiles_scaler);
                    noc_semaphore_inc(reduce_second_stage_receiver_semaphore_noc_addr, 1);
                }
            } else {
                cb_wait_front(cb_ex, num_tiles_to_read * num_tiles_scaler);
                noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
            }
        }

        // ============================================================================
        // Receive the multicasted final results into `cb_ex_global`
        // ============================================================================
        for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
            uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last : num_tiles_per_worker;
            cb_reserve_back(cb_ex_global, num_tiles * num_tiles_scaler);
            noc_semaphore_wait_min(reduce_sender_semaphore_addr_ptr, block + 2);
            cb_push_back(cb_ex_global, num_tiles * num_tiles_scaler);
        }
    };

    if constexpr (!rms_norm) {
        // Welford processes 2 tiles at a time (mean and var)
        global_reduce_receiver(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global, cb_ex, use_welford ? 2 : 1);
    }

    if constexpr (!use_welford) {
        global_reduce_receiver(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global, cb_ex2, 1);
    }
}
