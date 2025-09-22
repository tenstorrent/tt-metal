// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/welford_combine.hpp"

// split REDUCE across cores
void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t block_ht = get_compile_time_arg_val(3);
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
    constexpr uint32_t num_bytes_copy_to_combine = get_compile_time_arg_val(17);
    constexpr uint32_t num_combine_tiles_needed = get_compile_time_arg_val(18);
    constexpr uint32_t tile_height = get_compile_time_arg_val(19);
    constexpr uint32_t tile_width = get_compile_time_arg_val(20);
    constexpr uint32_t face_height = get_compile_time_arg_val(21);
    constexpr uint32_t face_width = get_compile_time_arg_val(22);

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(1);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);
    const uint32_t start_x = get_arg_val<uint32_t>(4);
    const uint32_t start_y = get_arg_val<uint32_t>(5);

    tt_l1_ptr uint32_t* in0_remote_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(6));
    tt_l1_ptr uint32_t* in0_remote_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(6 + num_x));

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;     // E[x] partial result
    constexpr uint32_t cb_ex_combine = tt::CBIndex::c_9;     // E[x] buffer for global combine
    constexpr uint32_t cb_varx_partial = tt::CBIndex::c_11;  // Var[x] partial result
    constexpr uint32_t cb_varx_combine = tt::CBIndex::c_12;  // Var[x] buffer for global combine
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;     // Final global E[x]
    constexpr uint32_t cb_varx_global = tt::CBIndex::c_10;   // Final global Var[x]

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial2);
    const DataFormat data_format = get_dataformat(cb_ex_partial2);

    uint64_t remote_noc_addrs[num_blocks];

    uint32_t x = start_x, y = start_y;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        remote_noc_addrs[i] = get_noc_addr(in0_remote_noc_x[x], in0_remote_noc_y[y], 0);
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

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_second_stage_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_second_stage_semaphore_addr);

    // wait for local partial data ready
    cb_wait_front(cb_ex_partial, block_ht);
    cb_wait_front(cb_varx_partial, block_ht);

    // Copy local partial data to combine buffers
    cb_reserve_back(cb_ex_combine, num_combine_tiles_needed);
    cb_reserve_back(cb_varx_combine, num_combine_tiles_needed);
    auto l1_read_addr_ex_par = get_read_ptr(cb_ex_partial);
    l1_read_addr_ex_par += all_to_all_tile_offset_bytes;
    auto l1_read_addr_varx_par = get_read_ptr(cb_varx_partial);
    l1_read_addr_varx_par += all_to_all_tile_offset_bytes;
    auto l1_write_addr_ex_combine = get_write_ptr(cb_ex_combine);
    auto l1_write_addr_varx_combine = get_write_ptr(cb_varx_combine);
    for (uint32_t i = 0; i < num_tiles_to_read; i++) {
        // Copy partial E[x]
        noc_async_read(l1_read_addr_ex_par, l1_write_addr_ex_combine, num_bytes_copy_to_combine);
        l1_read_addr_ex_par += single_tile_size_bytes;
        l1_write_addr_ex_combine += num_bytes_copy_to_combine;

        // Copy partial Var[x]
        noc_async_read(l1_read_addr_varx_par, l1_write_addr_varx_combine, num_bytes_copy_to_combine);
        l1_read_addr_varx_par += single_tile_size_bytes;
        l1_write_addr_varx_combine += num_bytes_copy_to_combine;
    }
    noc_async_read_barrier();

    cb_pop_front(cb_ex_partial, block_ht);
    cb_pop_front(cb_varx_partial, block_ht);

    // Wait for combine buffers in other cores to be ready
    noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_blocks - 1);
    noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

    // Copy combine buffers from all cores to this core's L1
    // TODO RM: Check if num_blocks > 1 before doing copy/combines
    auto means_combine_ptr = get_write_ptr(cb_ex_combine);
    auto vars_combine_ptr = get_write_ptr(cb_varx_combine);
    auto p_means_combine = reinterpret_cast<volatile uint16_t*>(means_combine_ptr);
    auto p_vars_combine = reinterpret_cast<volatile uint16_t*>(vars_combine_ptr);
    for (uint32_t block = 1; i < num_blocks; ++i) {
        uint64_t noc_addr_means = remote_noc_addrs[block] | means_combine_ptr;
        uint64_t noc_addr_vars = remote_noc_addrs[block] | vars_combine_ptr;
        noc_async_read(
            noc_addr_means, means_combine_ptr + block * num_bytes_copy_to_combine, num_bytes_copy_to_combine);
        noc_async_read(noc_addr_vars, vars_combine_ptr + block * num_bytes_copy_to_combine, num_bytes_copy_to_combine);
    }
    noc_async_read_barrier();

    // Do Welford combine of partial data across all cores
    // Write this to cb_ex_global and cb_varx_global
    auto global_means_ptr = get_write_ptr(cb_ex_global);
    auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
    auto global_vars_ptr = get_write_ptr(cb_varx_global);
    auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
    constexpr uint32_t combine_tile_stride = num_combine_tiles_needed * tile_height * tile_width;
    constexpr uint32_t face_stride = face_height * face_width;
    constexpr uint32_t one_tile_stride = tile_height * tile_width;

    // Do Welford combines to form half of the output
    // mean/var column vector tiles
    auto combine_half_column = [&](const auto global_offset) {
        for (uint32_t t = 0; t < block_ht; t++) {
            for (uint32_t r = 0; r < face_height; r++) {
                auto combine_result =
                    combine_welford_stats<num_blocks, tile_width, combine_tile_stride>(p_means_combine, p_vars_combine);
                p_global_means[global_offset + t * one_tile_stride + r * face_width] = combine_result.mean;
                p_global_vars[global_offset + t * tile_height + r] = combine_result.variance;
                p_means_combine += 2;  // Data stored in every other tile slot
                p_vars_combine += 2;   // Data stored in every other tile slot
            }
        }
    };

    // Do a combine for each row of each tile in block_ht.
    // Form column vectors in global mean and var tiles
    uint32_t global_offset{0};
    for (uint32_t t = 0; t < block_ht; t++) {
        // Form first half of mean and var
        // column vectors
        combine_half_column(global_offset);

        // After first face height of rows, move the global
        // mean/vars pointer to the beginning of the third face
        // to continue the column vector
        global_offset += 2 * face_stride;

        // Form second half of mean and var
        // column vectors
        combine_half_column(global_offset);

        // Advance the global mean and var pointers
        // to the beginning of the next tile
        global_offset += 2 * face_stride;
    }

    // Multicast result to all cores
    if constexpr (num_blocks > 1) {
        // Multicast global means
        noc_async_write_multicast(
            global_means_ptr,
            multicast_data_noc | global_means_ptr,
            block_ht * single_tile_size_bytes,
            num_blocks - 1,
            true);

        // Multicast global vars
        noc_async_write_multicast(
            global_vars_ptr,
            multicast_data_noc | global_vars_ptr,
            block_ht * single_tile_size_bytes,
            num_blocks - 1,
            true);

        noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_blocks - 1);
    }
}
