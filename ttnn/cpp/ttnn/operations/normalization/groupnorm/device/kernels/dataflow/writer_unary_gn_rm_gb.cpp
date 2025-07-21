// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/dprint.h"

void kernel_main() {
    constexpr bool is_mcast_sender = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool input_mask_is_dram = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t num_cols_tile_gamma_beta = get_compile_time_arg_val(7);

    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(9);
    constexpr uint32_t per_core_N_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(11);

    constexpr uint32_t num_groups_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t num_batches_per_core = get_compile_time_arg_val(13);

    constexpr uint32_t num_cols_per_group = get_compile_time_arg_val(14);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(15);

    constexpr uint32_t block_w_last = get_compile_time_arg_val(16);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_compile_time_arg_val(17);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_compile_time_arg_val(18);
    constexpr uint32_t group_row_offset = get_compile_time_arg_val(19);
    constexpr uint32_t num_out_blocks = get_compile_time_arg_val(20);

    constexpr uint32_t block_h = get_compile_time_arg_val(21);
    constexpr uint32_t block_w = get_compile_time_arg_val(22);
    constexpr uint32_t block_hw = get_compile_time_arg_val(23);

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t page_size = get_compile_time_arg_val(25);

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t tile_w_minux_group_size = TILE_WIDTH - num_cols_per_group;
    uint32_t row_offset = num_cols_per_group;
    uint32_t index_g_offset = 0;
    uint32_t index_b_offset = 0;

    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t input_mask_addr = get_arg_val<uint32_t>(6);
    const uint32_t out_start_id = get_arg_val<uint32_t>(7);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(8);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(9);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(10);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_28;
    constexpr uint32_t cb_in = tt::CBIndex::c_29;

    // constexpr uint32_t block_w = 4;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma);
    const uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask);
    const DataFormat input_mask_data_format = get_dataformat(cb_input_mask);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out =
        (fuse_gamma or fuse_beta)
            ? (((fuse_gamma and not fuse_beta) or (not fuse_gamma and fuse_beta)) ? cb_in : cb_out0)
            : cb_out0;
#endif
    const DataFormat out_data_format = get_dataformat(cb_out);

    // input mask
    const InterleavedAddrGenFast<input_mask_is_dram> mask = {
        .bank_base_address = input_mask_addr,
        .page_size = input_mask_single_tile_size_bytes,
        .data_format = input_mask_data_format};

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
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

    index_b_offset = 0;
    constexpr uint32_t row_tile_max_index = num_cols_tile_gamma_beta;

    for (uint32_t b = 0; b < num_batches_per_core; ++b) {
        uint32_t input_mask_tile_id = input_mask_tile_start_id;
        index_g_offset = 0;
        row_offset = num_cols_per_group;

        for (uint32_t i = 0; i < num_groups_per_core; ++i) {
            cb_reserve_back(cb_input_mask, block_w);
            uint32_t l1_write_addr_input_mask = get_write_ptr(cb_input_mask);
            for (uint32_t j = 0; j < block_w; ++j) {
                noc_async_read_tile(input_mask_tile_id, mask, l1_write_addr_input_mask);
                l1_write_addr_input_mask += input_mask_single_tile_size_bytes;
                input_mask_tile_id += 1;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_mask, block_w);

            if (i == 0 and b == 0) {
                constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
                const uint32_t scalar_w = get_arg_val<uint32_t>(1);
                generate_reduce_scaler(cb_in_2, scalar_w);

                if constexpr (is_mcast_sender) {
                    constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
                    const uint32_t scalar_c = get_arg_val<uint32_t>(0);
                    generate_reduce_scaler(cb_in_4, scalar_c);
                }

                constexpr uint32_t eps_cb_id = tt::CBIndex::c_3;
                const uint32_t eps = get_arg_val<uint32_t>(2);
                generate_bcast_col_scalar(eps_cb_id, eps);

                if constexpr (fuse_gamma) {
                    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
                    auto gamma = get_interleaved_addr_gen<gamma_is_dram, page_size>(gamma_addr);

                    cb_reserve_back(cb_gamma, num_cols_tile_gamma_beta);

                    const uint32_t base_l1_write_addr_gamma = get_write_ptr(cb_gamma);
                    uint32_t l1_write_addr_gamma = base_l1_write_addr_gamma;

                    // We want this data to appear as the first row of the tile.
                    // This is 32B at the start of the first face, 32B at the start of the second face
                    // However we must read at a 64 byte granularity for Blackhole NOC compatibility on DRAM reads
                    // So instead of two 32B reads to the correct addresses, we read 64 bytes into the first face here
                    // Then later, copy the second set of 32 bytes into the start of the second face
                    // L1-L1 NOC transactions only need 16 byte alignment on BH, so this is legal after data is loaded
                    // to L1

                    // Read the first 64 bytes of the tile into the first face
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = gamma_tile_start_id + w;
                        uint64_t gamma_noc_addr = get_noc_addr(tile_id, gamma);

                        noc_async_read(gamma_noc_addr, l1_write_addr_gamma, 64);
                        l1_write_addr_gamma += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();

                    // Copy the second set of 32 bytes into the second face
                    l1_write_addr_gamma = base_l1_write_addr_gamma;

                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint64_t noc_read_addr = get_noc_addr(l1_write_addr_gamma + 32);
                        noc_async_read(noc_read_addr, l1_write_addr_gamma + 512, 32);
                        l1_write_addr_gamma += gamma_tile_bytes;
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_gamma, num_cols_tile_gamma_beta);
                }

                if constexpr (fuse_beta) {
                    // Just like gamma, we read at a 64 byte granularity for Blackhole NOC compatibility
                    // Then copy the second set of 32 bytes into the second face

                    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
                    auto beta = get_interleaved_addr_gen<beta_is_dram, page_size>(beta_addr);

                    cb_reserve_back(cb_beta, num_cols_tile_gamma_beta);

                    const uint32_t base_l1_write_addr_beta = get_write_ptr(cb_beta);
                    uint32_t l1_write_addr_beta = base_l1_write_addr_beta;

                    // Read the first 64 bytes of the tile into the first face
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = beta_tile_start_id + w;
                        uint64_t beta_noc_addr = get_noc_addr(tile_id, beta);
                        noc_async_read(beta_noc_addr, l1_write_addr_beta, 64);
                        l1_write_addr_beta += beta_tile_bytes;
                    }
                    noc_async_read_barrier();

                    // Copy the second set of 32 bytes into the second face
                    l1_write_addr_beta = base_l1_write_addr_beta;

                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint64_t noc_read_addr = get_noc_addr(l1_write_addr_beta + 32);
                        noc_async_read(noc_read_addr, l1_write_addr_beta + 512, 32);
                        l1_write_addr_beta += beta_tile_bytes;
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_beta, num_cols_tile_gamma_beta);
                }
            }

            // add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            const InterleavedAddrGenFast<out_is_dram> dst_a = {
                .bank_base_address = out_addr, .page_size = single_tile_size_bytes, .data_format = out_data_format};

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
                cb_wait_front(cb_out, out_block_hw_normal);
                uint32_t l1_read_addr = get_read_ptr(cb_out);

                for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                    for (uint32_t nt = 0; nt < block_w_curr; nt++) {
                        // Checks, only relavent to the last group, that we are not indexing out of bounds
                        // for the cases where our last group does not span the length of our max tile span for a group
                        if ((index_g_offset + nt) < row_tile_max_index) {
                            noc_async_write_tile(
                                out_start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt +
                                    index_b_offset + index_g_offset,
                                dst_a,
                                l1_read_addr);
                        }
                        l1_read_addr += single_tile_size_bytes;
                    }
                    // l1_read_addr += (single_tile_size_bytes * (block_w-block_w_curr));
                }
                out_block_start_id_offset += out_block_h_actual * num_channels_tiles;
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_block_hw_normal);
            }

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }
        index_b_offset += num_tiles_per_batch;
    }
}
