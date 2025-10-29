// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "tt-metalium/constants.hpp"
#include "noc_parameters.h"

void kernel_main() {
    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_named_compile_time_arg_val("reduce_receiver_semaphore_id"));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_named_compile_time_arg_val("reduce_sender_semaphore_id"));

    constexpr uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr uint32_t num_batches = get_named_compile_time_arg_val("num_batches");

    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t TILE_HEIGHT = get_named_compile_time_arg_val("TILE_HEIGHT");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
    constexpr uint32_t block_hw = get_named_compile_time_arg_val("block_hw");

    constexpr uint32_t num_cols_per_group = get_named_compile_time_arg_val("num_cols_per_group");
    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t block_w_last = get_named_compile_time_arg_val("block_w_last");
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_named_compile_time_arg_val("GROUP_SIZE_IS_POWER_OF_2");
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_named_compile_time_arg_val("GROUP_SIZE_SMALLER_THAN_TILE_W");
    constexpr uint32_t group_row_offset = get_named_compile_time_arg_val("group_row_offset");
    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    // These are numbers in absolute terms, on a per group, per batch without tiling
    constexpr uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr uint32_t num_rows_per_group = get_named_compile_time_arg_val("num_rows_per_group");

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tt::constants::TILE_WIDTH - num_cols_per_group;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t out_start_id = get_arg_val<uint32_t>(3);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);

    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(6);
    const auto reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    const auto reduce_receiver_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);
    constexpr uint32_t src0_tile_bytes = get_tile_size(cb_in0);

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    const uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
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
        uint32_t index_g_offset = 0;
        uint32_t row_offset = num_cols_per_group;

        cb_reserve_back(cb_ex_global, 2 * num_groups);
        auto global_means_ptr = get_write_ptr(cb_ex_global);
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        for (uint32_t m = 0; m < num_groups; ++m) {
            uint32_t out_block_start_id_offset = 0;
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
                uint32_t l1_write_addr = get_write_ptr(cb_in0);
                cb_reserve_back(cb_in0, out_block_hw_normal);
                for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                    for (uint32_t nt = 0; nt < block_w; nt++) {
                        noc_async_read_tile(
                            start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt + index_b_offset +
                                index_g_offset,
                            src_a,
                            l1_write_addr);
                        l1_write_addr += src0_tile_bytes;
                        noc_async_read_barrier();
                    }
                }
                cb_push_back(cb_in0, out_block_hw_normal);
#endif
                out_block_start_id_offset += out_block_h_actual * num_channels_tiles;
            }

            cb_wait_front(cb_ex_partial, 2);

            // Read mean and variance arrays from cb_ex_partial, then combine using Welford
            auto local_read_ptr = get_read_ptr(cb_ex_partial);
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_read_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_read_ptr + single_tile_size_bytes);

            auto local_result = combine_welford_stats<
                tt::constants::TILE_WIDTH,
                num_channels_per_group * num_rows_per_group / tt::constants::TILE_WIDTH,
                2>(p_local_means, p_local_vars);

            // Write this to cb_ex_global
            volatile uint16_t* p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            volatile uint16_t* p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            // Signal to sender that our partial data is ready
            noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);

            // Wait for sender to signal that it has sent the global data
            noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
            noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);

            cb_pop_front(cb_ex_partial, 2);

            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > tt::constants::TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }

        cb_push_back(cb_ex_global, 2 * num_groups);

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
                    cb_reserve_back(cb_in0, 1);
                    const uint32_t l1_write_addr = get_write_ptr(cb_in0);
                    noc_async_read_tile(start_id + index_b_offset + mt_offset + nt, src_a, l1_write_addr);
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        const uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
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
