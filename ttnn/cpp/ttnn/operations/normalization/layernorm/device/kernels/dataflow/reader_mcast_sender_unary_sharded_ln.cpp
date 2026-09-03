// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "layernorm_dataflow_utils.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

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
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto block_h = get_arg(args::block_h);
    constexpr auto num_all_to_all_workers_first_stage = get_arg(args::num_all_to_all_workers_first_stage);
    constexpr auto num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr auto num_tiles_per_worker_bytes = get_arg(args::num_tiles_per_worker_bytes);
    constexpr auto num_tiles_per_worker_last_bytes = get_arg(args::num_tiles_per_worker_last_bytes);
    constexpr bool row_major = (bool)get_arg(args::row_major);
    constexpr auto num_x = get_arg(args::num_x);
    constexpr auto num_y = get_arg(args::num_y);
    constexpr bool use_two_stage_reduce = (bool)get_arg(args::use_two_stage_reduce);
    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
    constexpr auto num_mcast_dests = get_arg(args::num_mcast_dests);
#ifdef RMSNORM
    constexpr bool rms_norm = true;
#else
    constexpr bool rms_norm = false;
#endif
#ifdef USE_WELFORD
    constexpr bool use_welford = true;
#else
    constexpr bool use_welford = false;
#endif

    // ---------------------------------------------------------------------------
    // Runtime arguments
    // ---------------------------------------------------------------------------
    const uint32_t mcast_dest_noc_start_x = get_arg(args::mcast_dest_noc_start_x);
    const uint32_t mcast_dest_noc_start_y = get_arg(args::mcast_dest_noc_start_y);
    const uint32_t mcast_dest_noc_end_x = get_arg(args::mcast_dest_noc_end_x);
    const uint32_t mcast_dest_noc_end_y = get_arg(args::mcast_dest_noc_end_y);
    const uint32_t start_x = get_arg(args::start_x);
    const uint32_t start_y = get_arg(args::start_y);

    // The multicast grid's NOC coordinates arrive as a positional block: num_x X coordinates followed
    // by num_y Y coordinates. Copy them into locals so the coordinate-walk helpers below keep taking a
    // pointer into memory rather than a per-element accessor.
    uint32_t remote_noc_x[num_x];
    uint32_t remote_noc_y[num_y];
    for (uint32_t i = 0; i < num_x; ++i) {
        remote_noc_x[i] = get_vararg(i);
    }
    for (uint32_t i = 0; i < num_y; ++i) {
        remote_noc_y[i] = get_vararg(num_x + i);
    }
    df::L1Ptr in0_remote_noc_x = (df::L1Ptr)remote_noc_x;
    df::L1Ptr in0_remote_noc_y = (df::L1Ptr)remote_noc_y;

    // ---------------------------------------------------------------------------
    // Set up experimental API objects
    // ---------------------------------------------------------------------------
    Noc noc;
    Semaphore reduce_receiver_sem(sem::reduce_receiver);
    Semaphore reduce_sender_sem(sem::reduce_sender);
    Semaphore reduce_second_stage_sem(sem::reduce_second_stage);
    UnicastEndpoint remote_ep;
    MulticastEndpoint mcast_ep;

    // RMSNorm only allocates the Var[x] partial buffer; the host skips the E[x] one.
#ifdef RMSNORM
    DataflowBuffer dfb_partial_size_ref(dfb::ex_partial2);
#else
    DataflowBuffer dfb_partial_size_ref(dfb::ex_partial);
#endif
    const uint32_t single_tile_size_bytes = dfb_partial_size_ref.get_tile_size();

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
    const auto& global_reduce_sender = [&](const uint32_t dfb_partial_id,
                                           const uint32_t dfb_external_id,
                                           const uint32_t dfb_ex_id,
                                           const uint32_t dfb_ex_global_id,
                                           const uint32_t dfb_reduce_first_stage_id,
                                           const uint32_t num_tiles_scaler) __attribute__((always_inline)) {
        DataflowBuffer dfb_partial_obj(dfb_partial_id);
        DataflowBuffer dfb_external_obj(dfb_external_id);
        DataflowBuffer dfb_ex_obj(dfb_ex_id);
        DataflowBuffer dfb_ex_global_obj(dfb_ex_global_id);
        DataflowBuffer dfb_reduce_first_stage_obj(dfb_reduce_first_stage_id);

        // ============================================================================
        // Partial reduction
        // ============================================================================

        dfb_partial_obj.wait_front(block_h * num_tiles_scaler);

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
                num_mcast_dests);
        }

        // ============================================================================
        // Combine partial results
        // Read from the partial buffers into the external buffer.
        // Will read a total of:
        // (num_blocks_first_stage + num_blocks_second_stage - 1) * num_tiles_scaler
        // tiles for each assigned tile row (or column, if not row-major).
        // For the second stage, read from the first-stage reduce buffer instead of the partial
        // buffer, as it will contain the combined results from the first stage.
        // Combined results written to the E[x] buffer.
        // ============================================================================

        // ---------------------------------------------------------------------------
        // Read remote partial data
        // ---------------------------------------------------------------------------
        uint32_t l1_read_addr_ex_par = dfb_partial_obj.get_read_ptr();
        uint32_t l1_read_addr_ex = 0;
        uint32_t block_index_stride = 0;
        if constexpr (use_two_stage_reduce) {
            l1_read_addr_ex = dfb_reduce_first_stage_obj.get_read_ptr();
            if constexpr (row_major) {
                block_index_stride = num_x;
            } else {
                block_index_stride = num_y;
            }
        }
        for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
            dfb_external_obj.reserve_back(num_blocks_first_stage * num_tiles_scaler);
            uint32_t write_offset = 0;
            for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    remote_ep,
                    dfb_external_obj,
                    num_tiles_scaler * single_tile_size_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_par},
                    {.offset_bytes = write_offset});
                write_offset += num_tiles_scaler * single_tile_size_bytes;
            }
            l1_read_addr_ex_par += num_tiles_scaler * single_tile_size_bytes;
            noc.async_read_barrier();
            dfb_external_obj.push_back(num_blocks_first_stage * num_tiles_scaler);

            // ---------------------------------------------------------------------------
            // Handle the two-stage reduce
            // ---------------------------------------------------------------------------
            if constexpr (use_two_stage_reduce) {
                if (i == 0) {
                    reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                    reduce_second_stage_sem.set(0);
                }

                uint32_t curr_block_index = block_index_stride;
                dfb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                    noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        remote_ep,
                        dfb_external_obj,
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
                dfb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
            }
        }

        // ---------------------------------------------------------------------------
        // Wait for all final combined results to be ready
        // ---------------------------------------------------------------------------

        dfb_ex_obj.wait_front(num_tiles_per_worker * num_tiles_scaler);
        dfb_partial_obj.pop_front(block_h * num_tiles_scaler);

        if constexpr (num_all_to_all_workers_first_stage > 1) {
            reduce_receiver_sem.wait(num_all_to_all_workers_first_stage - 1);
            reduce_receiver_sem.set(0);
        }

        // ============================================================================
        // Gather all final combined results and multicast to all cores.
        // Read from the E[x] buffer into the global buffer, multicast the global buffer to all cores
        // ============================================================================

        uint32_t l1_read_addr_ex_remote = dfb_ex_obj.get_read_ptr();
        dfb_ex_global_obj.reserve_back(block_h * num_tiles_scaler);
        uint32_t gather_write_offset = 0;
        // Account for num_tiles_scaler (2 for Welford, 1 otherwise) when checking
        // if the gather read fits in a single NOC packet.
        constexpr uint32_t gather_tiles_scaler = use_welford ? 2 : 1;
        for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
            uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1 ? num_tiles_per_worker_last_bytes
                                                                                       : num_tiles_per_worker_bytes;
            if constexpr (num_tiles_per_worker_bytes * gather_tiles_scaler <= NOC_MAX_BURST_SIZE) {
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    remote_ep,
                    dfb_ex_global_obj,
                    num_tiles_scaler * num_tiles_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_remote},
                    {.offset_bytes = gather_write_offset});
            } else {
                noc.async_read(
                    remote_ep,
                    dfb_ex_global_obj,
                    num_tiles_scaler * num_tiles_bytes,
                    {.noc_x = remote_coords[block].x, .noc_y = remote_coords[block].y, .addr = l1_read_addr_ex_remote},
                    {.offset_bytes = gather_write_offset});
            }
            gather_write_offset += num_tiles_scaler * num_tiles_bytes;
        }
        noc.async_read_barrier();

        uint32_t l1_read_addr_ex_global = dfb_ex_global_obj.get_read_ptr();
        dfb_ex_global_obj.push_back(block_h * num_tiles_scaler);
        if constexpr (num_blocks > 1) {
            uint32_t mcast_src_offset = 0;
            for (uint32_t block = 0; block < num_all_to_all_workers_first_stage; ++block) {
                reduce_sender_sem.set(block + 2);

                uint32_t num_tiles_bytes = block == num_all_to_all_workers_first_stage - 1
                                               ? num_tiles_per_worker_last_bytes
                                               : num_tiles_per_worker_bytes;

                noc.async_write_multicast(
                    dfb_ex_global_obj,
                    mcast_ep,
                    num_tiles_scaler * num_tiles_bytes,
                    num_mcast_dests,
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
                    num_mcast_dests);

                mcast_src_offset += num_tiles_scaler * num_tiles_bytes;
                noc.async_write_barrier();
            }
        }
    };

    // RMSNorm has no mean to reduce, so its buffers are not declared and the call is compiled out.
#ifndef RMSNORM
    global_reduce_sender(dfb::ex_partial, dfb::ex_external, dfb::ex, dfb::ex_global, dfb::ex, use_welford ? 2 : 1);
#endif

    // Welford produces the mean and variance together in the pass above, so it has no separate
    // variance reduction and those buffers are not declared either.
#ifndef USE_WELFORD
    global_reduce_sender(dfb::ex_partial2, dfb::ex_external2, dfb::ex2pe, dfb::ex_global, dfb::ex2, 1);
#endif
}
