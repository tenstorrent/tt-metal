// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"

struct RemoteCoord {
    uint32_t x;
    uint32_t y;
};

// split REDUCE across cores
void kernel_main() {
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

    const bool is_last_all_to_all_worker = get_arg_val<uint32_t>(0);
    const uint32_t all_to_all_tile_offset_bytes = get_arg_val<uint32_t>(1);
    const bool is_second_stage_reader = get_arg_val<uint32_t>(2);
    const uint32_t start_x = get_arg_val<uint32_t>(3);
    const uint32_t start_y = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* in0_remote_noc_x = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));
    volatile tt_l1_ptr uint32_t* in0_remote_noc_y = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_x));

    const uint32_t num_tiles_to_read = is_last_all_to_all_worker ? num_tiles_per_worker_last : num_tiles_per_worker;

    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(get_compile_time_arg_val(0));
    experimental::Semaphore<> reduce_sender_sem(get_compile_time_arg_val(1));
    experimental::Semaphore<> reduce_second_stage_sem(get_compile_time_arg_val(14));
    experimental::UnicastEndpoint remote_ep;

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
        experimental::CircularBuffer cb_partial_obj(cb_partial_id);
        experimental::CircularBuffer cb_external_obj(cb_external_id);
        experimental::CircularBuffer cb_reduce_first_stage_obj(cb_reduce_first_stage_id);

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
                        noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
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
                            noc.async_read<experimental::Noc::TxnIdMode::DISABLED, NOC_MAX_BURST_SIZE>(
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
