// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"

void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));

    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(9);

    // These are numbers in absolute terms, on a per group, per batch, per core with tiling
    constexpr uint32_t block_hw = get_compile_time_arg_val(10);

    constexpr uint32_t TILE_WIDTH = 32;

    const bool has_mcast_first_group = get_arg_val<uint32_t>(0);
    const bool has_mcast_last_group = get_arg_val<uint32_t>(1);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(3);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(4);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(5);
    const uint32_t num_mcast_cores_mid_group = get_arg_val<uint32_t>(6);

    // first mcast group
    uint32_t mcast_first_group_dest_noc_start_x;
    uint32_t mcast_first_group_dest_noc_start_y;
    uint32_t mcast_first_group_dest_noc_end_x;
    uint32_t mcast_first_group_dest_noc_end_y;
    // last mcast group
    uint32_t mcast_last_group_dest_noc_start_x;
    uint32_t mcast_last_group_dest_noc_start_y;
    uint32_t mcast_last_group_dest_noc_end_x;
    uint32_t mcast_last_group_dest_noc_end_y;

    tt_l1_ptr uint32_t* noc_coord_x;
    tt_l1_ptr uint32_t* noc_coord_y;

    // number of cores in mcast groups
    uint32_t num_mcast_cores_first_group;
    uint32_t num_mcast_cores_last_group;

    // noc addrs for first and last groups
    uint64_t reduce_sender_first_group_semaphore_noc_addr;
    uint64_t multicast_first_group_data_noc;
    uint64_t reduce_sender_last_group_semaphore_noc_addr;
    uint64_t multicast_last_group_data_noc;

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(11);

        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(11);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(7);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(8);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(9);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(10);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(11);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));

    } else {
        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(7));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(7 + num_mcast_cores));
    }

    const uint64_t reduce_sender_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        reduce_sender_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    if (has_mcast_first_group) {
        reduce_sender_first_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_first_group_data_noc = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            0);
    }
    if (has_mcast_last_group) {
        reduce_sender_last_group_semaphore_noc_addr = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            reduce_sender_semaphore_addr);

        multicast_last_group_data_noc = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            0);
    }

    auto reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, VALID);

    auto reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;         // sharded cb
    constexpr uint32_t cb_repack = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_12;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    const uint32_t num_bytes_read = datum_size_bytes;

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_reserve_back(cb_repack, per_core_N);
        uint32_t l1_write_addr_repack = get_write_ptr(cb_repack);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_repack, per_core_N);
    }
#endif

    for (uint32_t m = 0; m < num_batch_group; ++m) {
        // wait for local data ready
        cb_reserve_back(cb_ex_global, 2);
        cb_wait_front(cb_ex_partial, 2);

        // Read mean and variance arrays from cb_ex_partial, then combine using Welford
        auto local_read_ptr = get_read_ptr(cb_ex_partial);
        auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_read_ptr);
        auto p_local_vars = p_local_means + TILE_WIDTH * TILE_HEIGHT;

        auto local_result = combine_welford_stats<TILE_WIDTH, block_hw * TILE_WIDTH, 2>(p_local_means, p_local_vars);

        // Write this to cb_ex_global
        auto global_means_ptr = get_write_ptr(cb_ex_global);
        auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;
        auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
        p_global_means[0] = local_result.mean;
        p_global_vars[0] = local_result.variance;

        cb_pop_front(cb_ex_partial, 2);

        if constexpr (num_mcast_cores > 1) {
            // wait for all other cores data ready
            noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_mcast_cores - 1);
            noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);

            for (uint32_t i = 1; i < num_mcast_cores; ++i) {
                noc_async_read_one_packet(multicast_data_noc | global_means_ptr, global_means_ptr + i * 32, 32);
                noc_async_read_one_packet(multicast_data_noc | global_vars_ptr, global_vars_ptr + i * 32, 32);
            }
            noc_async_read_barrier();
        }

        // Read mean and variance arrays from cb_ex_global, then combine using Welford
        auto global_result =
            combine_welford_stats<num_mcast_cores, block_hw * 32 * 32, 16>(p_global_means, p_global_vars);

        // Write this to cb_ex_global
        p_global_means[0] = global_result.mean;
        p_global_vars[0] = global_result.variance;

        // mcast to other cores
        if constexpr (num_mcast_cores > 1) {
            noc_async_write_multicast(
                global_means_ptr,
                multicast_data_noc | global_means_ptr,
                num_bytes_read,
                num_mcast_cores_mid_group,
                true);
            noc_semaphore_set_multicast(
                reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_mcast_cores_mid_group, false);

            if (has_mcast_first_group) {
                noc_async_write_multicast(
                    global_means_ptr,
                    multicast_first_group_data_noc | global_means_ptr,
                    num_bytes_read,
                    num_mcast_cores_first_group,
                    true);
                noc_semaphore_set_multicast(
                    reduce_sender_semaphore_addr,
                    reduce_sender_first_group_semaphore_noc_addr,
                    num_mcast_cores_first_group,
                    false);
            }

            if (has_mcast_last_group) {
                noc_async_write_multicast(
                    global_means_ptr,
                    multicast_last_group_data_noc | global_means_ptr,
                    num_bytes_read,
                    num_mcast_cores_last_group,
                    true);
                noc_semaphore_set_multicast(
                    reduce_sender_semaphore_addr,
                    reduce_sender_last_group_semaphore_noc_addr,
                    num_mcast_cores_last_group,
                    false);
            }
            noc_async_write_barrier();
        }
        cb_push_back(cb_ex_global, 2);
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
#endif
}
