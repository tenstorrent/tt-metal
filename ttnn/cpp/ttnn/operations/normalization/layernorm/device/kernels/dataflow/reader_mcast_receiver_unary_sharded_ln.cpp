// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/dataflow_api.h"
#include "layernorm_dataflow_utils.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"

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
    // Set up experimental API objects
    // ---------------------------------------------------------------------------
    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(get_compile_time_arg_val(0));
    experimental::Semaphore<> reduce_sender_sem(get_compile_time_arg_val(1));
    experimental::Semaphore<> reduce_second_stage_sem(get_compile_time_arg_val(14));
    experimental::UnicastEndpoint remote_ep;

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;
    const uint32_t single_tile_size_bytes = get_tile_size(rms_norm ? cb_ex_partial2 : cb_ex_partial);

    // Compute the NOC coordinates for remote cores that interact with this core
    constexpr df::NumNocAddrs num_remote_noc_addrs_first_stage = is_all_to_all_worker ? num_blocks_first_stage : 1;
    constexpr df::NumNocAddrs num_remote_noc_addrs_second_stage = is_all_to_all_worker ? num_blocks_second_stage : 1;
    df::RemoteNocCoords<num_remote_noc_addrs_first_stage> remote_coords_first_stage{};
    df::RemoteNocCoords<num_remote_noc_addrs_second_stage> remote_coords_second_stage{};
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            df::compute_two_stage_noc_addrs<
                row_major,
                num_remote_noc_addrs_first_stage,
                num_remote_noc_addrs_second_stage>(
                remote_coords_first_stage,
                remote_coords_second_stage,
                in0_remote_noc_x,
                in0_remote_noc_y,
                start_x,
                start_y,
                num_x,
                num_y);
        } else {
            df::compute_single_stage_noc_addrs<row_major, num_remote_noc_addrs_first_stage>(
                remote_coords_first_stage, in0_remote_noc_x, in0_remote_noc_y, start_x, start_y, num_x, num_y);
        }
    } else {
        remote_coords_first_stage[0] = {in0_remote_noc_x[0], in0_remote_noc_y[0]};
    }

    // ============================================================================
    // Main kernel worker function
    // Waits on partial reduction, syncs with coordinator, reads
    // from other cores, signals when combine is done, receives multicast
    // ============================================================================
    const auto& global_reduce_receiver = [&](const uint32_t cb_partial_id,
                                             const uint32_t cb_external_id,
                                             const uint32_t cb_ex_id,
                                             const uint32_t cb_ex_global_id,
                                             const uint32_t cb_reduce_first_stage_id,
                                             const uint32_t num_tiles_scaler) __attribute__((always_inline)) {
        experimental::CircularBuffer cb_partial_obj(cb_partial_id);
        experimental::CircularBuffer cb_external_obj(cb_external_id);
        experimental::CircularBuffer cb_ex_obj(cb_ex_id);
        experimental::CircularBuffer cb_ex_global_obj(cb_ex_global_id);
        experimental::CircularBuffer cb_reduce_first_stage_obj(cb_reduce_first_stage_id);

        // ============================================================================
        // Partial reduction
        // ============================================================================

        cb_partial_obj.wait_front(block_h * num_tiles_scaler);

        reduce_sender_sem.set(INVALID);
        reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
        reduce_sender_sem.wait(VALID);

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
            uint32_t l1_read_addr_ex_par = cb_partial_obj.get_read_ptr();
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes * num_tiles_scaler;
            uint32_t l1_read_addr_ex = 0;
            if constexpr (use_two_stage_reduce) {
                l1_read_addr_ex = cb_reduce_first_stage_obj.get_read_ptr();
            }
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                cb_external_obj.reserve_back(num_blocks_first_stage * num_tiles_scaler);
                uint32_t write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                        remote_ep,
                        cb_external_obj,
                        num_tiles_scaler * single_tile_size_bytes,
                        {.noc_x = remote_coords_first_stage[block].x,
                         .noc_y = remote_coords_first_stage[block].y,
                         .addr = l1_read_addr_ex_par},
                        {.offset_bytes = write_offset});
                    write_offset += num_tiles_scaler * single_tile_size_bytes;
                }
                l1_read_addr_ex_par += num_tiles_scaler * single_tile_size_bytes;
                noc.async_read_barrier();
                cb_external_obj.push_back(num_blocks_first_stage * num_tiles_scaler);

                // ---------------------------------------------------------------------------
                // Handle the two-stage reduce
                // ---------------------------------------------------------------------------
                if constexpr (use_two_stage_reduce) {
                    if (is_second_stage_reader) {
                        if (i == 0) {
                            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                            reduce_second_stage_sem.set(0);
                        }

                        cb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                        write_offset = 0;
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                                remote_ep,
                                cb_external_obj,
                                num_tiles_scaler * single_tile_size_bytes,
                                {.noc_x = remote_coords_second_stage[block + 1].x,
                                 .noc_y = remote_coords_second_stage[block + 1].y,
                                 .addr = l1_read_addr_ex},
                                {.offset_bytes = write_offset});
                            write_offset += num_tiles_scaler * single_tile_size_bytes;
                        }
                        l1_read_addr_ex += num_tiles_scaler * single_tile_size_bytes;
                        noc.async_read_barrier();
                        cb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                    } else {
                        // If we're not a second stage reader (i.e. we're not in the top
                        // row of cores), we don't do any additional combines, so we just
                        // do a dummy push so that we move in lockstep with the other cores
                        cb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                        cb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                    }
                }
            }

            // ---------------------------------------------------------------------------
            // Wait for all final combined results to be ready and notify sender
            // ---------------------------------------------------------------------------
            if constexpr (use_two_stage_reduce) {
                if (is_second_stage_reader) {
                    cb_ex_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                    reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
                } else {
                    cb_reduce_first_stage_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                    reduce_second_stage_sem.up(
                        noc, remote_coords_second_stage[0].x, remote_coords_second_stage[0].y, 1);
                }
            } else {
                cb_ex_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
            }
        }

        // ============================================================================
        // Receive the multicasted final results into `cb_ex_global`
        // ============================================================================
        for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
            uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last : num_tiles_per_worker;
            cb_ex_global_obj.reserve_back(num_tiles * num_tiles_scaler);
            reduce_sender_sem.wait_min(block + 2);
            cb_ex_global_obj.push_back(num_tiles * num_tiles_scaler);
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
