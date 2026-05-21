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
    const bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr uint32_t num_all_to_all_workers = get_arg(args::num_all_to_all_workers_first_stage);
    constexpr uint32_t num_tiles_per_worker = get_arg(args::num_tiles_per_worker);
    constexpr uint32_t num_tiles_per_worker_last = get_arg(args::num_tiles_per_worker_last);
    constexpr bool row_major = (bool)get_arg(args::row_major);
    constexpr uint32_t num_x = get_arg(args::num_x);
    constexpr uint32_t num_y = get_arg(args::num_y);
    constexpr bool use_two_stage_reduce = (bool)get_arg(args::use_two_stage_reduce);
    constexpr uint32_t num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr uint32_t num_blocks_second_stage = get_arg(args::num_blocks_second_stage);
    constexpr bool rms_norm = get_arg(args::rms_norm) == 1;

    const bool is_last_all_to_all_worker = get_arg(args::is_last_all_to_all_worker);
    const uint32_t all_to_all_tile_offset_bytes = get_arg(args::all_to_all_offset_bytes);
    const bool is_second_stage_reader = get_arg(args::is_second_stage_reader);
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
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_remote_noc_x_buf);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_remote_noc_y_buf);

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    constexpr uint32_t cb_ex_partial2 = dfb::cb_ex_partial2;
    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_ex_external2 = dfb::cb_ex_external2;

    Noc noc;
    Semaphore<> reduce_receiver_sem(sem::reduce_receiver);
    Semaphore<> reduce_sender_sem(sem::reduce_sender);
    Semaphore<> reduce_second_stage_sem(sem::reduce_second_stage);
    UnicastEndpoint remote_ep;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);
    const DataFormat data_format = get_dataformat(cb_ex_partial2);

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

    const auto& global_reduce_receiver = [&](const uint32_t cb_partial_id,
                                             const uint32_t cb_external_id,
                                             const uint32_t cb_reduce_first_stage_id) __attribute__((always_inline)) {
        DataflowBuffer cb_partial_obj(cb_partial_id);
        DataflowBuffer cb_external_obj(cb_external_id);
        DataflowBuffer cb_reduce_first_stage_obj(cb_reduce_first_stage_id);

        uint32_t num_tiles_per_partial_result = 2;
        if constexpr (rms_norm) {
            num_tiles_per_partial_result = 1;
        }

        cb_partial_obj.wait_front(num_tiles_per_partial_result * block_h);

        reduce_sender_sem.set(INVALID);
        reduce_receiver_sem.up(noc, in0_remote_noc_x[0], in0_remote_noc_y[0], 1);
        reduce_sender_sem.wait(VALID);

        if constexpr (is_all_to_all_worker) {
            uint32_t l1_read_addr_ex_par = cb_partial_obj.get_read_ptr();
            l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
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
            for (uint32_t i = 0; i < num_tiles_to_read; i++) {
                cb_external_obj.reserve_back(num_tiles_per_partial_result * num_blocks_first_stage);
                write_offset = 0;
                for (uint32_t block = 0; block < num_blocks_first_stage; block++) {
                    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_partial_result; tile_idx++) {
                        noc.async_read<Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                            remote_ep,
                            cb_external_obj,
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
                cb_external_obj.push_back(num_tiles_per_partial_result * num_blocks_first_stage);

                // read data from other cores - reduce first stage
                if constexpr (use_two_stage_reduce) {
                    if (is_second_stage_reader) {  // gather data from a column of cores (if row major)
                        if (i == 0) {
                            reduce_second_stage_sem.wait(num_blocks_second_stage - 1);
                            reduce_second_stage_sem.set(0);
                        }
                        // read data from other cores - second stage reduce

                        write_offset = 0;
                        for (uint32_t block = 0; block < num_blocks_second_stage - 1; ++block) {
                            noc.async_read<Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
                                remote_ep,
                                cb_external_obj,
                                single_tile_size_bytes,
                                {.noc_x = remote_coords_second_stage[block + 1].x,
                                 .noc_y = remote_coords_second_stage[block + 1].y,
                                 .addr = l1_read_addr_ex},
                                {.offset_bytes = write_offset});
                            write_offset += single_tile_size_bytes;
                        }
                        l1_read_addr_ex += single_tile_size_bytes;
                        noc.async_read_barrier();
                        cb_external_obj.push_back(num_blocks_second_stage - 1);
                    }
                }
            }

            cb_reduce_first_stage_obj.wait_front(num_tiles_per_partial_result * num_tiles_to_read);
            reduce_second_stage_sem.up(noc, remote_coords_second_stage[0].x, remote_coords_second_stage[0].y, 1);
        }
    };
    global_reduce_receiver(cb_ex_partial2, cb_ex_external2, cb_ex2);
    noc.async_atomic_barrier();
}
