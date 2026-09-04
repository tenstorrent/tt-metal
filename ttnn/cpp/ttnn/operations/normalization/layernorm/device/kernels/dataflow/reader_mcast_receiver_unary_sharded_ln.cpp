// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "layernorm_dataflow_utils.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

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
    // An idle core sits in a hole of a non-rectangular shard grid. It carries this program's dataflow
    // buffers and semaphores so the reduction's multicast has somewhere to land, and does no work of
    // its own, so its whole body is compiled out.
#ifndef IDLE_CORE

    // ============================================================================
    // Kernel setup
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Compile-time arguments
    // ---------------------------------------------------------------------------
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto block_h = get_arg(args::block_h);
    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr auto num_all_to_all_workers = get_arg(args::num_all_to_all_workers);
    constexpr auto num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr auto num_tiles_per_worker_last = get_arg(args::num_tiles_per_worker_last);
    constexpr bool row_major = (bool)get_arg(args::row_major);
    constexpr auto num_x = get_arg(args::num_x);
    constexpr auto num_y = get_arg(args::num_y);
    constexpr bool use_two_stage_reduce = (bool)get_arg(args::use_two_stage_reduce);
    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
#ifdef USE_WELFORD
    constexpr bool use_welford = true;
#else
    constexpr bool use_welford = false;
#endif

    // ---------------------------------------------------------------------------
    // Runtime arguments
    // ---------------------------------------------------------------------------
    const bool is_last_all_to_all_worker = get_arg(args::is_last_all_to_all_worker);
    const uint32_t all_to_all_tile_offset_bytes = get_arg(args::all_to_all_tile_offset_bytes);
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader);
    const uint32_t start_x = get_arg(args::start_x);
    const uint32_t start_y = get_arg(args::start_y);

    // The NOC coordinates of this core's remote peers arrive as a positional block: num_x X
    // coordinates followed by num_y Y coordinates. A core that only waits for the multicast is given
    // a 1 x 1 grid, so its block is just the sender's coordinate pair. Copy them into locals so the
    // coordinate-walk helpers below keep taking a pointer into memory.
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

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;
    // RMSNorm only allocates the Var[x] partial buffer; the host skips the E[x] one.
#ifdef RMSNORM
    DataflowBuffer dfb_partial_size_ref(dfb::ex_partial2);
#else
    DataflowBuffer dfb_partial_size_ref(dfb::ex_partial);
#endif
    const uint32_t single_tile_size_bytes = dfb_partial_size_ref.get_tile_size();

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
    const auto& global_reduce_receiver = [&](const uint32_t dfb_partial_id,
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

        reduce_sender_sem.set(INVALID);
        reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
        reduce_sender_sem.wait(VALID);

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
        if constexpr (is_all_to_all_worker) {
            uint32_t l1_read_addr_ex_par = dfb_partial_obj.get_read_ptr();
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes * num_tiles_scaler;
            uint32_t l1_read_addr_ex = 0;
            if constexpr (use_two_stage_reduce) {
                l1_read_addr_ex = dfb_reduce_first_stage_obj.get_read_ptr();
            }
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                dfb_external_obj.reserve_back(num_blocks_first_stage * num_tiles_scaler);
                uint32_t write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        remote_ep,
                        dfb_external_obj,
                        num_tiles_scaler * single_tile_size_bytes,
                        {.noc_x = remote_coords_first_stage[block].x,
                         .noc_y = remote_coords_first_stage[block].y,
                         .addr = l1_read_addr_ex_par},
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
                    if (is_second_stage_reader) {
                        if (i == 0) {
                            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                            reduce_second_stage_sem.set(0);
                        }

                        dfb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                        write_offset = 0;
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                                remote_ep,
                                dfb_external_obj,
                                num_tiles_scaler * single_tile_size_bytes,
                                {.noc_x = remote_coords_second_stage[block + 1].x,
                                 .noc_y = remote_coords_second_stage[block + 1].y,
                                 .addr = l1_read_addr_ex},
                                {.offset_bytes = write_offset});
                            write_offset += num_tiles_scaler * single_tile_size_bytes;
                        }
                        l1_read_addr_ex += num_tiles_scaler * single_tile_size_bytes;
                        noc.async_read_barrier();
                        dfb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                    } else {
                        // If we're not a second stage reader (i.e. we're not in the top
                        // row of cores), we don't do any additional combines, so we just
                        // do a dummy push so that we move in lockstep with the other cores
                        dfb_external_obj.reserve_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                        dfb_external_obj.push_back((num_blocks_second_stage - 1) * num_tiles_scaler);
                    }
                }
            }

            // ---------------------------------------------------------------------------
            // Wait for all final combined results to be ready and notify sender
            // ---------------------------------------------------------------------------
            if constexpr (use_two_stage_reduce) {
                if (is_second_stage_reader) {
                    dfb_ex_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                    reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
                } else {
                    dfb_reduce_first_stage_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                    reduce_second_stage_sem.up(
                        noc, remote_coords_second_stage[0].x, remote_coords_second_stage[0].y, 1);
                }
            } else {
                dfb_ex_obj.wait_front(num_tiles_to_read * num_tiles_scaler);
                reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
            }
        }

        // ============================================================================
        // Receive the multicasted final results into the global buffer
        // ============================================================================
        for (uint32_t block = 0; block < num_all_to_all_workers; ++block) {
            uint32_t num_tiles = block == num_all_to_all_workers - 1 ? num_tiles_per_worker_last : num_tiles_per_worker;
            dfb_ex_global_obj.reserve_back(num_tiles * num_tiles_scaler);
            reduce_sender_sem.wait_min(block + 2);
            dfb_ex_global_obj.push_back(num_tiles * num_tiles_scaler);
        }

        // The partial-reduction buffer is waited up front and read (locally and by remote cores)
        // during the combine; by here all those reads have completed, so pop it to leave the buffer
        // balanced.
        dfb_partial_obj.pop_front(block_h * num_tiles_scaler);
    };

    // RMSNorm has no mean to reduce, so its buffers are not declared and the call is compiled out.
#ifndef RMSNORM
    // Welford processes 2 tiles at a time (mean and var)
    global_reduce_receiver(dfb::ex_partial, dfb::ex_external, dfb::ex, dfb::ex_global, dfb::ex, use_welford ? 2 : 1);
#endif

    // Welford produces the mean and variance together in the pass above, so it has no separate
    // variance reduction and those buffers are not declared either.
#ifndef USE_WELFORD
    global_reduce_receiver(dfb::ex_partial2, dfb::ex_external2, dfb::ex2pe, dfb::ex_global, dfb::ex2, 1);
#endif

#endif  // IDLE_CORE
}
