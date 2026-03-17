// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "layernorm_dataflow_utils.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"

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
    constexpr uint32_t cb_ex_partial = get_named_compile_time_arg_val("cb_ex_partial");
    constexpr uint32_t cb_ex = get_named_compile_time_arg_val("cb_ex");
    constexpr uint32_t cb_ex_external = get_named_compile_time_arg_val("cb_ex_external");
    constexpr uint32_t cb_ex_partial2 = get_named_compile_time_arg_val("cb_ex_partial2");
    constexpr uint32_t cb_ex2 = get_named_compile_time_arg_val("cb_ex2");
    constexpr uint32_t cb_ex_external2 = get_named_compile_time_arg_val("cb_ex_external2");
    constexpr uint32_t cb_ex2pe = get_named_compile_time_arg_val("cb_ex2pe");
    constexpr uint32_t cb_ex_global = get_named_compile_time_arg_val("cb_ex_global");  // E[x] global reduce

    // ---------------------------------------------------------------------------
    // Set up experimental API objects
    // ---------------------------------------------------------------------------
    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(get_compile_time_arg_val(0));
    experimental::Semaphore<> reduce_sender_sem(get_compile_time_arg_val(1));
    experimental::Semaphore<> reduce_second_stage_sem(get_compile_time_arg_val(16));
    experimental::UnicastEndpoint remote_ep;
    experimental::MulticastEndpoint mcast_ep;

    const uint32_t single_tile_size_bytes = get_tile_size(rms_norm ? cb_ex_partial2 : cb_ex_partial);

    // Compute the NOC coordinates for remote cores that interact with this core
    df::RemoteNocCoords<num_blocks> remote_coords{};
    df::compute_single_stage_noc_addrs<row_major, num_blocks>(
        remote_coords, in0_remote_noc_x, in0_remote_noc_y, start_x, start_y, num_x, num_y);

    // ============================================================================
    // Main kernel worker function
    // Performs partial reduction for its assigned tiles, coordinates
    // the waiting on partial and combined results for all cores, and performs
    // the multicast of the final results to all cores
    // ============================================================================
    const auto& global_reduce_sender = [&](const uint32_t cb_partial_id,
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

        if constexpr (num_blocks > 1) {
            reduce_sender_sem.set(VALID);
            reduce_receiver_sem.wait(num_blocks - 1);
            reduce_receiver_sem.set(0);
            reduce_sender_sem.set_multicast(
                noc,
                mcast_dest_noc_start_x,
                mcast_dest_noc_start_y,
                mcast_dest_noc_end_x,
                mcast_dest_noc_end_y,
                num_blocks - 1);
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
        uint32_t l1_read_addr_ex_par = cb_partial_obj.get_read_ptr();
        uint32_t l1_read_addr_ex = 0;
        uint32_t block_index_stride = 0;
        if constexpr (use_two_stage_reduce) {
            l1_read_addr_ex = cb_reduce_first_stage_obj.get_read_ptr();
            if constexpr (row_major) {
                block_index_stride = num_x;
            } else {
                block_index_stride = num_y;
            }
        }
        for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
            cb_external_obj.reserve_back(num_blocks_first_stage * num_tiles_scaler);
            uint32_t write_offset = 0;
            for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
                noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                    remote_ep,
                    cb_external_obj,
                    num_tiles_scaler * single_tile_size_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_par},
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
                if (i == 0) {
                    reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                    reduce_second_stage_sem.set(0);
                }

                uint32_t curr_block_index = block_index_stride;
                cb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                    noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                        remote_ep,
                        cb_external_obj,
                        num_tiles_scaler * single_tile_size_bytes,
                        {.noc_x = remote_coords[curr_block_index].x,
                         .noc_y = remote_coords[curr_block_index].y,
                         .addr = l1_read_addr_ex},
                        {.offset_bytes = write_offset});
                    curr_block_index += block_index_stride;
                    write_offset += num_tiles_scaler * single_tile_size_bytes;
                }
                l1_read_addr_ex += num_tiles_scaler * single_tile_size_bytes;
                noc.async_read_barrier();
                cb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
            }
        }

        // ---------------------------------------------------------------------------
        // Wait for all final combined results to be ready
        // ---------------------------------------------------------------------------

        cb_ex_obj.wait_front(num_tiles_per_worker * num_tiles_scaler);

        if constexpr (num_all_to_all_workers_first_stage > 1) {
            reduce_receiver_sem.wait(num_all_to_all_workers_first_stage - 1);
            reduce_receiver_sem.set(0);
        }

        // ============================================================================
        // Gather all final combined results and multicast to all cores.
        // Read from `cb_ex` into `cb_ex_global`, multicast `cb_ex_global` to all cores
        // ============================================================================

        uint32_t l1_read_addr_ex_remote = cb_ex_obj.get_read_ptr();
        cb_ex_global_obj.reserve_back(block_h * num_tiles_scaler);
        uint32_t gather_write_offset = 0;
        // Account for num_tiles_scaler (2 for Welford, 1 otherwise) when checking
        // if the gather read fits in a single NOC packet.
        constexpr uint32_t gather_tiles_scaler = use_welford ? 2 : 1;
        for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
            uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1 ? num_tiles_per_worker_last_bytes
                                                                                       : num_tiles_per_worker_bytes;
            if constexpr (num_tiles_per_worker_bytes * gather_tiles_scaler <= NOC_MAX_BURST_SIZE) {
                noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                    remote_ep,
                    cb_ex_global_obj,
                    num_tiles_scaler * num_tiles_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_remote},
                    {.offset_bytes = gather_write_offset});
            } else {
                noc.async_read(
                    remote_ep,
                    cb_ex_global_obj,
                    num_tiles_scaler * num_tiles_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_remote},
                    {.offset_bytes = gather_write_offset});
            }
            gather_write_offset += num_tiles_scaler * num_tiles_bytes;
        }
        noc.async_read_barrier();

        uint32_t l1_read_addr_ex_global = cb_ex_global_obj.get_read_ptr();
        cb_ex_global_obj.push_back(block_h * num_tiles_scaler);
        if constexpr (num_blocks > 1) {
            uint32_t mcast_src_offset = 0;
            for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
                reduce_sender_sem.set(block + 2);

                uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1
                                               ? num_tiles_per_worker_last_bytes
                                               : num_tiles_per_worker_bytes;

                noc.async_write_multicast(
                    cb_ex_global_obj,
                    mcast_ep,
                    num_tiles_scaler * num_tiles_bytes,
                    num_blocks - 1,
                    {.offset_bytes = mcast_src_offset},
                    {.noc_x_start = mcast_dest_noc_start_x,
                     .noc_y_start = mcast_dest_noc_start_y,
                     .noc_x_end = mcast_dest_noc_end_x,
                     .noc_y_end = mcast_dest_noc_end_y,
                     .addr = l1_read_addr_ex_global + mcast_src_offset},
                    true);
                reduce_sender_sem.set_multicast(
                    noc,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_blocks - 1);

                mcast_src_offset += num_tiles_scaler * num_tiles_bytes;
                noc.async_write_barrier();
            }
        }
    };

    if constexpr (!rms_norm) {
        global_reduce_sender(cb_ex_partial, cb_ex_external, cb_ex, cb_ex_global, cb_ex, use_welford ? 2 : 1);
    }

    if constexpr (!use_welford) {
        global_reduce_sender(cb_ex_partial2, cb_ex_external2, cb_ex2pe, cb_ex_global, cb_ex2, 1);
    }
}
