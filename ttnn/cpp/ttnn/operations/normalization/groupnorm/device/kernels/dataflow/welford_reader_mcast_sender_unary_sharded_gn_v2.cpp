// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_batches = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t tile_height = get_compile_time_arg_val(9);

    // These are numbers in absolute terms, on a per group, per batch, per core with tiling
    constexpr uint32_t block_hw = get_compile_time_arg_val(10);
    constexpr uint32_t num_groups = get_compile_time_arg_val(11);
    constexpr uint32_t tile_width = get_compile_time_arg_val(12);

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
    uint64_t multicast_first_group_data_noc;
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

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);

    if (has_mcast_first_group) {
        multicast_first_group_data_noc = get_noc_multicast_addr(
            mcast_first_group_dest_noc_start_x,
            mcast_first_group_dest_noc_start_y,
            mcast_first_group_dest_noc_end_x,
            mcast_first_group_dest_noc_end_y,
            0);
    }
    if (has_mcast_last_group) {
        multicast_last_group_data_noc = get_noc_multicast_addr(
            mcast_last_group_dest_noc_start_x,
            mcast_last_group_dest_noc_start_y,
            mcast_last_group_dest_noc_end_x,
            mcast_last_group_dest_noc_end_y,
            0);
    }

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    experimental::Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    reduce_sender_sem.set(VALID);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;

    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_in0(cb_in0_id);
    experimental::CircularBuffer cb_repack(cb_repack_id);
    experimental::CircularBuffer cb_repack_out(cb_repack_out_id);
    experimental::CircularBuffer cb_out0(cb_out0_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    // This is the stride between two consecutive local means/variances in the cb_ex_partial
    constexpr uint32_t local_stride = 2;
    constexpr uint32_t global_stride = NOC_L1_READ_ALIGNMENT_BYTES / 2;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    for (uint32_t m = 0; m < num_batches; ++m) {
        // Read mean and variance arrays from cb_ex_partial, then combine using Welford

        // wait for local data readykj
        cb_ex_partial.wait_front(2);
        auto local_means_ptr = cb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        cb_ex_global.reserve_back(2 * num_groups);
        auto global_means_ptr = cb_ex_global.get_write_ptr();
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        for (uint32_t m = 0; m < num_groups; ++m) {
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);

            auto local_result =
                combine_welford_stats<tile_width, block_hw * tile_width, local_stride>(p_local_means, p_local_vars);

            // Write this to cb_ex_global
            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            if constexpr (num_mcast_cores > 1) {
                // wait for all other cores data ready
                reduce_receiver_sem.wait(num_mcast_cores - 1);
                reduce_receiver_sem.set(0);

                for (uint32_t i = 1; i < num_mcast_cores; ++i) {
                    noc_async_read_one_packet(
                        multicast_data_noc | global_means_ptr,
                        global_means_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES,
                        NOC_L1_READ_ALIGNMENT_BYTES);
                    noc_async_read_one_packet(
                        multicast_data_noc | global_vars_ptr,
                        global_vars_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES,
                        NOC_L1_READ_ALIGNMENT_BYTES);
                }
                noc.async_read_barrier();
            }

            // Read mean and variance arrays from cb_ex_global, then combine using Welford
            auto global_result =
                combine_welford_stats<num_mcast_cores, block_hw * tile_width * tile_height, global_stride>(
                    p_global_means, p_global_vars);

            // Write this to cb_ex_global
            p_global_means[0] = global_result.mean;
            p_global_vars[0] = global_result.variance;

            // mcast to other cores
            if constexpr (num_mcast_cores > 1) {
                noc_async_write_multicast(
                    global_means_ptr,
                    multicast_data_noc | global_means_ptr,
                    2 * single_tile_size_bytes,
                    num_mcast_cores_mid_group,
                    true);
                reduce_sender_sem.set_multicast(
                    noc,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_mcast_cores_mid_group,
                    false);

                if (has_mcast_first_group) {
                    noc_async_write_multicast(
                        global_means_ptr,
                        multicast_first_group_data_noc | global_means_ptr,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_first_group,
                        true);
                    reduce_sender_sem.set_multicast(
                        noc,
                        mcast_first_group_dest_noc_start_x,
                        mcast_first_group_dest_noc_start_y,
                        mcast_first_group_dest_noc_end_x,
                        mcast_first_group_dest_noc_end_y,
                        num_mcast_cores_first_group,
                        false);
                }

                if (has_mcast_last_group) {
                    noc_async_write_multicast(
                        global_means_ptr,
                        multicast_last_group_data_noc | global_means_ptr,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_last_group,
                        true);
                    reduce_sender_sem.set_multicast(
                        noc,
                        mcast_last_group_dest_noc_start_x,
                        mcast_last_group_dest_noc_start_y,
                        mcast_last_group_dest_noc_end_x,
                        mcast_last_group_dest_noc_end_y,
                        num_mcast_cores_last_group,
                        false);
                }
                noc.async_write_barrier();
            }

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }
        cb_ex_partial.pop_front(2);
        cb_ex_global.push_back(2 * num_groups);
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
