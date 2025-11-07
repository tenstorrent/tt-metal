// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "tt-metalium/constants.hpp"
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"

void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));

    constexpr uint32_t num_batches = get_compile_time_arg_val(2);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(3);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(6);

    // These are numbers in absolute terms, on a per group, per batch without tiling
    constexpr uint32_t block_hw = get_compile_time_arg_val(8);
    constexpr uint32_t num_groups = get_compile_time_arg_val(9);

    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(1);

    auto reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;         // sharded cb
    constexpr uint32_t cb_repack = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_12;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);

    // This is the stride between two consecutive local means/variances in the cb_ex_partial
    constexpr uint32_t local_stride = 2;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tt::constants::TILE_HEIGHT;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_reserve_back(cb_repack, per_core_N);
        uint32_t l1_write_addr_repack = get_write_ptr(cb_repack);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_repack, per_core_N);
    }
#endif

    for (uint32_t i = 0; i < num_batches; ++i) {
        // Read mean and variance arrays from cb_ex_partial, then combine using Welford

        // wait for local data ready
        cb_wait_front(cb_ex_partial, 2);
        auto local_means_ptr = get_read_ptr(cb_ex_partial);
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        cb_reserve_back(cb_ex_global, 2 * num_groups);
        auto global_means_ptr = get_write_ptr(cb_ex_global);
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        for (uint32_t m = 0; m < num_groups; ++m) {
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);

            auto local_result =
                combine_welford_stats<tt::constants::TILE_WIDTH, block_hw * tt::constants::TILE_WIDTH, local_stride>(
                    p_local_means, p_local_vars);

            // Write this to cb_ex_global
            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            // Signal to sender that our partial data is ready
            noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);

            // Wait for sender to signal that it has sent the global data
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
            noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

        cb_pop_front(cb_ex_partial, 2);
        cb_push_back(cb_ex_global, 2 * num_groups);
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
#endif
}
