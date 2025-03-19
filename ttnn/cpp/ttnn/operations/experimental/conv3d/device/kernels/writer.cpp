// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    constexpr uint32_t cb_matmul_result_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_matmul_interm_tiled = get_compile_time_arg_val(3);
    constexpr uint32_t cb_reduction_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_worker_ack_back = get_compile_time_arg_val(5);
    constexpr uint32_t N = get_compile_time_arg_val(6);
    constexpr uint32_t T_out = get_compile_time_arg_val(7);
    constexpr uint32_t H_out = get_compile_time_arg_val(8);
    constexpr uint32_t W_out = get_compile_time_arg_val(9);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(10);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(11);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(12);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(13);
    constexpr uint32_t matmul_M_t = get_compile_time_arg_val(14);
    constexpr uint32_t matmul_K_t = get_compile_time_arg_val(15);
    constexpr uint32_t matmul_N_t = get_compile_time_arg_val(16);
    constexpr uint32_t num_patches_tile_padded = get_compile_time_arg_val(17);
    constexpr uint32_t out_row_size_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t C_out_block_bytes = get_compile_time_arg_val(19);  // padded to tile width
    constexpr bool use_bias = get_compile_time_arg_val(20) == 1;
    uint32_t semaphore_addr = get_semaphore(get_compile_time_arg_val(21));

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_reducer = get_arg_val<uint32_t>(argidx++);
    const uint32_t reducer_core_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t reducer_core_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(argidx++);

    const uint64_t reducer_semaphore_noc_addr = get_noc_addr(reducer_core_x, reducer_core_y, semaphore_addr);
    volatile tt_l1_ptr uint32_t* local_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    // Get worker core coordinates
    tt_l1_ptr uint32_t* worker_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(argidx));
    argidx += num_workers;
    tt_l1_ptr uint32_t* worker_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(argidx));

    constexpr uint32_t tile_bytes = get_tile_size(cb_weight_tiled);
    constexpr DataFormat data_format = get_dataformat(cb_weight_tiled);
    const InterleavedAddrGen<true> out_writer = {.bank_base_address = out_addr, .page_size = out_row_size_bytes};
    const InterleavedAddrGenFast<true> weight_reader = {
        .bank_base_address = weight_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> bias_reader = {
        .bank_base_address = bias_addr, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;
    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t C_out_t = C_out_num_blocks * matmul_N_t;

    for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
        const uint32_t c_in_offset_t = c_in_block * matmul_K_t;
        // Iterate only over assigned C_out blocks
        for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
            const uint32_t c_out_offset_t = c_out_block * matmul_N_t;

            // Read weights and bias for this block
            cb_reserve_back(cb_weight_tiled, weight_tiles);
            uint32_t weight_write_ptr = get_write_ptr(cb_weight_tiled);

            for (uint32_t row = c_in_offset_t; row < c_in_offset_t + matmul_K_t; row++) {
                for (uint32_t col = c_out_offset_t; col < c_out_offset_t + matmul_N_t; col++) {
                    uint32_t weight_idx = row * C_out_t + col;
                    noc_async_read_tile(weight_idx, weight_reader, weight_write_ptr);
                    weight_write_ptr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_weight_tiled, weight_tiles);

            if constexpr (use_bias) {
                if (is_reducer) {
                    cb_reserve_back(cb_bias_tiled, matmul_N_t);
                    uint32_t bias_write_ptr = get_write_ptr(cb_bias_tiled);
                    for (uint32_t i = c_out_offset_t; i < c_out_offset_t + matmul_N_t; i++) {
                        uint32_t bias_idx = i;
                        noc_async_read_tile(bias_idx, bias_reader, bias_write_ptr);
                        bias_write_ptr += tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_bias_tiled, matmul_N_t);
                }
            }

            // Write output for assigned ranges
            for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
                const uint32_t t_block_end = std::min(t_block + T_block_size, t_out_end);

                for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                    const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                    for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                        const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);

                        if (!is_reducer) {
                            // I'm a worker.
                            // Wait for compute to finish
                            cb_wait_front(cb_reduction_tiled, output_tiles);

                            // Reset our semaphore
                            *local_semaphore_addr_ptr = 0;

                            // Signal to reducer that we have data ready
                            noc_semaphore_inc(reducer_semaphore_noc_addr, 1);

                            // Wait for reducer to ack that it has read our data
                            noc_semaphore_wait(local_semaphore_addr_ptr, 1);

                            // Hanshake with compute so it can continue
                            cb_pop_front(cb_reduction_tiled, output_tiles);
                            cb_reserve_back(cb_worker_ack_back, 1);
                            cb_push_back(cb_worker_ack_back, 1);
                        } else {
                            // I'm a reducer.
                            // Wait for all workers to finish
                            noc_semaphore_wait(local_semaphore_addr_ptr, num_workers);

                            // Reset our semaphore
                            *local_semaphore_addr_ptr = 0;

                            const uint32_t worker_output_read_ptr = get_read_ptr(cb_matmul_interm_tiled);
                            for (uint32_t worker_idx = 0; worker_idx < num_workers; worker_idx++) {
                                // Read data from worker into reduction buffer
                                // Stall if compute has not cleared buffer
                                cb_reserve_back(cb_reduction_tiled, output_tiles);
                                uint32_t reduction_write_ptr = get_write_ptr(cb_reduction_tiled);
                                uint64_t worker_output_read_addr = get_noc_addr(
                                    worker_core_xs[worker_idx], worker_core_ys[worker_idx], worker_output_read_ptr);
                                for (uint32_t tile = 0; tile < output_tiles; tile++) {
                                    noc_async_read(worker_output_read_addr, reduction_write_ptr, tile_bytes);
                                    worker_output_read_addr += tile_bytes;
                                    reduction_write_ptr += tile_bytes;
                                }
                                noc_async_read_barrier();
                                cb_push_back(cb_reduction_tiled, output_tiles);

                                const uint64_t worker_semaphore_noc_addr = get_noc_addr(
                                    worker_core_xs[worker_idx], worker_core_ys[worker_idx], semaphore_addr);
                                noc_semaphore_inc(worker_semaphore_noc_addr, 1);
                            }

                            cb_wait_front(cb_matmul_result_rm, output_tiles);
                            uint32_t cb_read_ptr = get_read_ptr(cb_matmul_result_rm);

                            for (uint32_t t = t_block; t < t_block_end; ++t) {
                                for (uint32_t h = h_block; h < h_block_end; ++h) {
                                    for (uint32_t w = w_block; w < w_block_end; ++w) {
                                        uint32_t out_page_idx = t * H_out * W_out + h * W_out + w;
                                        uint64_t dst_addr = get_noc_addr(out_page_idx, out_writer);
                                        dst_addr += c_out_block * C_out_block_bytes;  // Using block index directly
                                        noc_async_write(cb_read_ptr, dst_addr, C_out_block_bytes);
                                        cb_read_ptr += C_out_block_bytes;
                                    }
                                }
                            }
                            noc_async_write_barrier();
                            cb_pop_front(cb_matmul_result_rm, output_tiles);
                        }
                    }
                }
            }
        }
    }
}
