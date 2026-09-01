// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

struct RemoteCoord {
    uint32_t x;
    uint32_t y;
};

// split REDUCE across cores
void kernel_main() {
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto block_h = get_arg(args::block_h);
    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr auto num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr auto num_tiles_per_worker_last = get_arg(args::num_tiles_per_worker_last);
    constexpr bool row_major = (bool)get_arg(args::row_major);
    constexpr auto num_x = get_arg(args::num_x);
    constexpr auto num_y = get_arg(args::num_y);
    constexpr bool use_two_stage_reduce = (bool)get_arg(args::use_two_stage_reduce);
    constexpr auto num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr auto num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
#ifdef RMSNORM
    constexpr bool rms_norm = true;
#else
    constexpr bool rms_norm = false;
#endif

    const bool is_last_all_to_all_worker = get_arg(args::is_last_all_to_all_worker);
    const uint32_t all_to_all_tile_offset_bytes = get_arg(args::all_to_all_tile_offset_bytes);
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader);
    const uint32_t start_x = get_arg(args::start_x);
    const uint32_t start_y = get_arg(args::start_y);

    // The NOC coordinates of this core's remote peers arrive as a positional block: num_x X
    // coordinates followed by num_y Y coordinates. A core that only waits for the multicast is given a
    // 1 x 1 grid, so its block is just the sender's coordinate pair.
    uint32_t in0_remote_noc_x[num_x];
    uint32_t in0_remote_noc_y[num_y];
    for (uint32_t i = 0; i < num_x; ++i) {
        in0_remote_noc_x[i] = get_vararg(i);
    }
    for (uint32_t i = 0; i < num_y; ++i) {
        in0_remote_noc_y[i] = get_vararg(num_x + i);
    }

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    Noc noc;
    Semaphore<> reduce_receiver_sem(sem::reduce_receiver);
    Semaphore<> reduce_sender_sem(sem::reduce_sender);
    Semaphore<> reduce_second_stage_sem(sem::reduce_second_stage);
    UnicastEndpoint remote_ep;

    DataflowBuffer dfb_ex_partial2_obj(dfb::ex_partial2);
    const uint32_t single_tile_size_bytes = dfb_ex_partial2_obj.get_tile_size();

    RemoteCoord remote_coords_first_stage[is_all_to_all_worker ? num_blocks_first_stage : 1];
    RemoteCoord remote_coords_second_stage[is_all_to_all_worker ? num_blocks_second_stage : 1];
    if constexpr (is_all_to_all_worker) {
        if constexpr (use_two_stage_reduce) {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks_first_stage; ++i) {
                remote_coords_first_stage[i] = {in0_remote_noc_x[x], in0_remote_noc_y[y]};
                if constexpr (row_major) {
                    ++x;
                    if (x == num_x) {
                        x = 0;
                    }
                } else {
                    ++y;
                    if (y == num_y) {
                        y = 0;
                    }
                }
            }
            if constexpr (row_major) {
                x = start_x;
                y = 0;
            } else {
                x = 0;
                y = start_y;
            }
            for (uint32_t i = 0; i < num_blocks_second_stage; ++i) {
                remote_coords_second_stage[i] = {in0_remote_noc_x[x], in0_remote_noc_y[y]};
                if constexpr (row_major) {
                    ++y;
                } else {
                    ++x;
                }
            }
        } else {
            uint32_t x = start_x, y = start_y;
            for (uint32_t i = 0; i < num_blocks; ++i) {
                remote_coords_first_stage[i] = {in0_remote_noc_x[x], in0_remote_noc_y[y]};
                if constexpr (row_major) {
                    ++x;
                    if (x == num_x) {
                        x = 0;
                        ++y;
                        if (y == num_y) {
                            y = 0;
                        }
                    }
                } else {
                    ++y;
                    if (y == num_y) {
                        y = 0;
                        ++x;
                        if (x == num_x) {
                            x = 0;
                        }
                    }
                }
            }
        }
    } else {
        remote_coords_first_stage[0] = {in0_remote_noc_x[0], in0_remote_noc_y[0]};
        remote_coords_second_stage[0] = {in0_remote_noc_x[0], in0_remote_noc_y[0]};
    }

    const auto& global_reduce_receiver = [&](const uint32_t dfb_partial_id,
                                             const uint32_t dfb_external_id,
                                             const uint32_t dfb_reduce_first_stage_id) __attribute__((always_inline)) {
        DataflowBuffer dfb_partial_obj(dfb_partial_id);
        DataflowBuffer dfb_external_obj(dfb_external_id);
        DataflowBuffer dfb_reduce_first_stage_obj(dfb_reduce_first_stage_id);

        uint32_t num_tiles_per_partial_result = 2;
        if constexpr (rms_norm) {
            num_tiles_per_partial_result = 1;
        }

        dfb_partial_obj.wait_front(num_tiles_per_partial_result * block_h);

        reduce_sender_sem.set(INVALID);
        reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
        reduce_sender_sem.wait(VALID);

        if constexpr (is_all_to_all_worker) {
            uint32_t l1_read_addr_ex_par = dfb_partial_obj.get_read_ptr();
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
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
            uint32_t write_offset = 0;
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                dfb_external_obj.reserve_back(num_tiles_per_partial_result * num_blocks_first_stage);
                write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; tile_idx++) {
                        noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                            remote_ep,
                            dfb_external_obj,
                            single_tile_size_bytes,
                            {.noc_x = remote_coords_first_stage[block].x,
                             .noc_y = remote_coords_first_stage[block].y,
                             .addr = l1_read_addr_ex_par + tile_idx * single_tile_size_bytes},
                            {.offset_bytes = write_offset});
                        write_offset += single_tile_size_bytes;
                    }
                }
                l1_read_addr_ex_par += single_tile_size_bytes;
                noc.async_read_barrier();
                dfb_external_obj.push_back(num_tiles_per_partial_result * num_blocks_first_stage);

                // read data from other cores - reduce first stage
                if constexpr (use_two_stage_reduce) {
                    if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                        if (i == 0) {
                            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                            reduce_second_stage_sem.set(0);
                        }
                        // read data from other cores - second stage reduce
                        // Mirror the sender and the first-stage block above: reserve/read/push
                        // num_tiles_per_partial_result tiles per block (E[x] and E[x^2] interleaved),
                        // not one, because the compute consumer pops
                        // num_tiles_per_partial_result * num_blocks_reduce.
                        dfb_external_obj.reserve_back(num_tiles_per_partial_result * (num_blocks_second_stage - 1));
                        write_offset = 0;
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; ++tile_idx) {
                                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                                    remote_ep,
                                    dfb_external_obj,
                                    single_tile_size_bytes,
                                    {.noc_x = remote_coords_second_stage[block + 1].x,
                                     .noc_y = remote_coords_second_stage[block + 1].y,
                                     .addr = l1_read_addr_ex + tile_idx * single_tile_size_bytes},
                                    {.offset_bytes = write_offset});
                                write_offset += single_tile_size_bytes;
                            }
                        }
                        l1_read_addr_ex += single_tile_size_bytes;
                        noc.async_read_barrier();
                        dfb_external_obj.push_back(num_tiles_per_partial_result * (num_blocks_second_stage - 1));
                    }
                }
            }

            dfb_reduce_first_stage_obj.wait_front(num_tiles_per_partial_result * num_tiles_to_read);
            reduce_second_stage_sem.up(noc, remote_coords_second_stage[0].x, remote_coords_second_stage[0].y, 1);
        }

        dfb_partial_obj.pop_front(num_tiles_per_partial_result * block_h);
    };
    global_reduce_receiver(dfb::ex_partial2, dfb::ex_external2, dfb::ex2);
    noc.async_atomic_barrier();
}
