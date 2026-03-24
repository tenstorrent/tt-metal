// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "noc_parameters.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_named_compile_time_arg_val("reduce_receiver_semaphore_id");
    constexpr uint32_t reduce_sender_semaphore_id = get_named_compile_time_arg_val("reduce_sender_semaphore_id");

    constexpr uint32_t num_mcast_cores = get_named_compile_time_arg_val("num_cores_per_mcast_group");
    constexpr uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr uint32_t num_batches = get_named_compile_time_arg_val("num_batches");
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("TILE_HEIGHT");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");

    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr uint32_t num_rows_per_group = get_named_compile_time_arg_val("num_rows_per_group");

    constexpr auto src0_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);

    const bool has_mcast_first_group = get_arg_val<uint32_t>(5);
    const bool has_mcast_last_group = get_arg_val<uint32_t>(6);

    // mid mcast group
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(7);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(8);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(9);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(10);
    const uint32_t num_mcast_cores_mid_group = get_arg_val<uint32_t>(11);

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

    // first and last group mcast coordinates passed directly in async_write_multicast calls below

    if (has_mcast_first_group and has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(17);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(18);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(19);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(20);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(21);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(22));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(22 + num_mcast_cores));

    } else if (has_mcast_first_group and not has_mcast_last_group) {
        mcast_first_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_first_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_first_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_first_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_first_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else if (not has_mcast_first_group and has_mcast_last_group) {
        mcast_last_group_dest_noc_start_x = get_arg_val<uint32_t>(12);
        mcast_last_group_dest_noc_start_y = get_arg_val<uint32_t>(13);
        mcast_last_group_dest_noc_end_x = get_arg_val<uint32_t>(14);
        mcast_last_group_dest_noc_end_y = get_arg_val<uint32_t>(15);
        num_mcast_cores_last_group = get_arg_val<uint32_t>(16);

        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));

    } else {
        noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(12));
        noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(12 + num_mcast_cores));
    }

    experimental::Noc noc;
    experimental::Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    experimental::Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    reduce_sender_sem.set(VALID);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;

    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_in0(cb_in0_id);
    experimental::CircularBuffer cb_repack(cb_repack_id);
    experimental::CircularBuffer cb_repack_out(cb_repack_out_id);
    experimental::CircularBuffer cb_out0(cb_out0_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    constexpr uint32_t src0_tile_bytes = get_tile_size(cb_in0_id);

    constexpr uint32_t local_stride = 2;
    constexpr uint32_t global_stride = NOC_L1_READ_ALIGNMENT_BYTES / 2;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    experimental::UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
        out_block_hw_last = out_block_h_last * block_w;
    }

    uint32_t index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual, out_block_hw_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
                out_block_hw_actual = out_block_hw_last;
            } else {
                out_block_h_actual = out_block_h_normal;
                out_block_hw_actual = out_block_hw_normal;
            }

#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        experimental::CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }

        cb_ex_partial.wait_front(2);
        auto local_means_ptr = cb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        cb_ex_global.reserve_back(2 * num_groups);
        auto global_means_ptr = cb_ex_global.get_write_ptr();
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        for (uint32_t m = 0; m < num_groups; ++m) {
            // Read mean and variance arrays from cb_ex_partial, then combine using Welford
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);

            auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);

            // Write this to cb_ex_global
            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            if constexpr (num_mcast_cores > 1) {
                // Wait until all other cores have signaled that their partial data is ready
                reduce_receiver_sem.wait(num_mcast_cores - 1);
                reduce_receiver_sem.set(0);

                for (uint32_t i = 1; i < num_mcast_cores; ++i) {
                    experimental::UnicastEndpoint remote_ep;
                    noc.async_read(
                        remote_ep,
                        experimental::CoreLocalMem<uint32_t>(global_means_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES),
                        NOC_L1_READ_ALIGNMENT_BYTES,
                        {.noc_x = noc_coord_x[i], .noc_y = noc_coord_y[i], .addr = global_means_ptr},
                        {});
                    noc.async_read(
                        remote_ep,
                        experimental::CoreLocalMem<uint32_t>(global_vars_ptr + i * NOC_L1_READ_ALIGNMENT_BYTES),
                        NOC_L1_READ_ALIGNMENT_BYTES,
                        {.noc_x = noc_coord_x[i], .noc_y = noc_coord_y[i], .addr = global_vars_ptr},
                        {});
                }
                noc.async_read_barrier();
            }

            // Read mean and variance arrays from cb_ex_global, then combine using Welford
            auto global_result =
                combine_welford_stats<num_mcast_cores, num_channels_per_group * num_rows_per_group, global_stride>(
                    p_global_means, p_global_vars);

            // Write this to cb_ex_global
            p_global_means[0] = global_result.mean;
            p_global_vars[0] = global_result.variance;

            if constexpr (num_mcast_cores > 1) {
                // mcast to other cores
                experimental::MulticastEndpoint mcast_dst;
                noc.async_write_multicast(
                    experimental::CoreLocalMem<uint32_t>(global_means_ptr),
                    mcast_dst,
                    2 * single_tile_size_bytes,
                    num_mcast_cores_mid_group,
                    {},
                    {.noc_x_start = mcast_dest_noc_start_x,
                     .noc_y_start = mcast_dest_noc_start_y,
                     .noc_x_end = mcast_dest_noc_end_x,
                     .noc_y_end = mcast_dest_noc_end_y,
                     .addr = global_means_ptr},
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
                    experimental::MulticastEndpoint mcast_first_group_dst;
                    noc.async_write_multicast(
                        experimental::CoreLocalMem<uint32_t>(global_means_ptr),
                        mcast_first_group_dst,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_first_group,
                        {},
                        {.noc_x_start = mcast_first_group_dest_noc_start_x,
                         .noc_y_start = mcast_first_group_dest_noc_start_y,
                         .noc_x_end = mcast_first_group_dest_noc_end_x,
                         .noc_y_end = mcast_first_group_dest_noc_end_y,
                         .addr = global_means_ptr},
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
                    experimental::MulticastEndpoint mcast_last_group_dst;
                    noc.async_write_multicast(
                        experimental::CoreLocalMem<uint32_t>(global_means_ptr),
                        mcast_last_group_dst,
                        2 * single_tile_size_bytes,
                        num_mcast_cores_last_group,
                        {},
                        {.noc_x_start = mcast_last_group_dest_noc_start_x,
                         .noc_y_start = mcast_last_group_dest_noc_start_y,
                         .noc_x_end = mcast_last_group_dest_noc_end_x,
                         .noc_y_end = mcast_last_group_dest_noc_end_y,
                         .addr = global_means_ptr},
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

        mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual, out_block_hw_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
                out_block_hw_actual = out_block_hw_last;
            } else {
                out_block_h_actual = out_block_h_normal;
                out_block_hw_actual = out_block_hw_normal;
            }
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        experimental::CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        experimental::UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
