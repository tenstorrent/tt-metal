// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

struct RemoteCoord {
    uint32_t x;
    uint32_t y;
};

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_h_size_bytes = get_arg(args::block_h_size_bytes);
    constexpr uint32_t num_all_to_all_workers_first_stage = get_arg(args::num_all_to_all_workers_first_stage);
    constexpr uint32_t num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr uint32_t num_tiles_per_worker_bytes = get_arg(args::num_tiles_per_worker_bytes);
    constexpr uint32_t num_tiles_per_worker_last = get_arg(args::num_tiles_per_worker_last);
    constexpr uint32_t num_tiles_per_worker_last_bytes = get_arg(args::num_tiles_per_worker_last_bytes);
    constexpr bool row_major = (bool)get_arg(args::row_major);
    constexpr uint32_t num_x = get_arg(args::num_x);
    constexpr uint32_t num_y = get_arg(args::num_y);
    constexpr bool use_two_stage_reduce = (bool)get_arg(args::use_two_stage_reduce);
    constexpr uint32_t num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr uint32_t num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
    constexpr bool rms_norm = get_arg(args::rms_norm) == 1;

    const uint32_t mcast_dest_noc_start_x = get_arg(args::mcast_start_x);
    const uint32_t mcast_dest_noc_start_y = get_arg(args::mcast_start_y);
    const uint32_t mcast_dest_noc_end_x = get_arg(args::mcast_end_x);
    const uint32_t mcast_dest_noc_end_y = get_arg(args::mcast_end_y);
    const uint32_t start_x = get_arg(args::start_x);
    const uint32_t start_y = get_arg(args::start_y);

    uint32_t in0_remote_noc_x_buf[num_x];
    uint32_t in0_remote_noc_y_buf[num_y];
    for (uint32_t i = 0; i < num_x; ++i) {
        in0_remote_noc_x_buf[i] = get_vararg(i);
    }
    for (uint32_t j = 0; j < num_y; ++j) {
        in0_remote_noc_y_buf[j] = get_vararg(num_x + j);
    }
    tt_l1_ptr uint32_t* in0_remote_noc_x = reinterpret_cast<tt_l1_ptr uint32_t*>(in0_remote_noc_x_buf);
    tt_l1_ptr uint32_t* in0_remote_noc_y = reinterpret_cast<tt_l1_ptr uint32_t*>(in0_remote_noc_y_buf);

    constexpr uint32_t cb_ex_partial2 = dfb::cb_ex_partial2;
    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_ex_external2 = dfb::cb_ex_external2;
    constexpr uint32_t cb_ex2_global = dfb::cb_in0_pre;  // c_14 in pre-allgather mode

    Noc noc;
    Semaphore<> reduce_receiver_sem(sem::reduce_receiver);
    Semaphore<> reduce_sender_sem(sem::reduce_sender);
    Semaphore<> reduce_second_stage_sem(sem::reduce_second_stage);
    UnicastEndpoint remote_ep;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);
    const DataFormat data_format = get_dataformat(cb_ex_partial2);

    RemoteCoord remote_coords[num_blocks];
    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_coords[i] = {in0_remote_noc_x[x], in0_remote_noc_y[y]};
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

    const auto& global_reduce_sender =
        [&](const uint32_t cb_partial_id, const uint32_t cb_external_id, const uint32_t cb_reduce_first_stage_id)
            __attribute__((always_inline)) {
                DataflowBuffer cb_partial_obj(cb_partial_id);
                DataflowBuffer cb_external_obj(cb_external_id);
                DataflowBuffer cb_reduce_first_stage_obj(cb_reduce_first_stage_id);

                uint32_t num_tiles_per_partial_result = 2;
                if constexpr (rms_norm) {
                    num_tiles_per_partial_result = 1;
                }

                // global reduce
                // wait for local data ready
                cb_partial_obj.wait_front(num_tiles_per_partial_result * block_h);

                // inc semaphore of other cores, tell other all-to-all workers to start
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

                // read data from other cores
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
                uint32_t write_offset = 0;
                for (uint32_t i = 0; i < num_tiles_per_worker; ++i) {
                    cb_external_obj.reserve_back(num_tiles_per_partial_result * num_blocks_first_stage);
                    write_offset = 0;
                    for (uint32_t block = 0; block < num_blocks_first_stage; ++block) {
                        for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; ++tile_idx) {
                            noc.async_read<Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                                remote_ep,
                                cb_external_obj,
                                single_tile_size_bytes,
                                {.noc_x = remote_coords[block].x,
                                 .noc_y = remote_coords[block].y,
                                 .addr = l1_read_addr_ex_par + tile_idx * single_tile_size_bytes},
                                {.offset_bytes = write_offset});
                            write_offset += single_tile_size_bytes;
                        }
                    }
                    l1_read_addr_ex_par += single_tile_size_bytes;
                    noc.async_read_barrier();
                    cb_external_obj.push_back(num_tiles_per_partial_result * num_blocks_first_stage);

                    // sync with second-stage all-to-all workers
                    if constexpr (use_two_stage_reduce) {
                        if (i == 0) {
                            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                            reduce_second_stage_sem.set(0);
                        }

                        uint32_t curr_block_index = block_index_stride;
                        cb_external_obj.reserve_back(num_tiles_per_partial_result * (num_blocks_second_stage - 1));
                        write_offset = 0;
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; ++tile_idx) {
                                noc.async_read<Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                                    remote_ep,
                                    cb_external_obj,
                                    single_tile_size_bytes,
                                    {.noc_x = remote_coords[curr_block_index].x,
                                     .noc_y = remote_coords[curr_block_index].y,
                                     .addr = l1_read_addr_ex + tile_idx * single_tile_size_bytes},
                                    {.offset_bytes = write_offset});
                                write_offset += single_tile_size_bytes;
                            }
                            curr_block_index += block_index_stride;
                        }
                        l1_read_addr_ex += single_tile_size_bytes;
                        noc.async_read_barrier();
                        cb_external_obj.push_back(
                            num_tiles_per_partial_result *
                            (num_blocks_second_stage -
                             1));  // push back partials from all cores -> compute can start reducing now
                    }
                }
            };
    global_reduce_sender(cb_ex_partial2, cb_ex_external2, cb_ex2);
    noc.async_write_barrier();
}
