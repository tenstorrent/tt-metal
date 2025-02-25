// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void dprint_rm(uint32_t cb_write_ptr, uint32_t num_rows, uint32_t num_cols) {
    volatile tt_l1_ptr uint16_t* ptr = (volatile tt_l1_ptr uint16_t*)cb_write_ptr;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
        DPRINT << "row " << i << ": ";
        for (uint32_t j = 0; j < num_cols; ++j) {
            DPRINT << ptr[idx] << " ";
            idx++;
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    constexpr uint32_t cb_matmul_result_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    constexpr uint32_t T_out = get_compile_time_arg_val(4);
    constexpr uint32_t H_out = get_compile_time_arg_val(5);
    constexpr uint32_t W_out = get_compile_time_arg_val(6);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(7);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(8);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(9);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t matmul_M_t = get_compile_time_arg_val(11);
    constexpr uint32_t matmul_K_t = get_compile_time_arg_val(12);
    constexpr uint32_t matmul_N_t = get_compile_time_arg_val(13);
    constexpr uint32_t num_patches_tile_padded = get_compile_time_arg_val(14);
    constexpr uint32_t out_row_size_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t C_out_block_bytes = get_compile_time_arg_val(16);  // padded to tile width
    constexpr bool use_bias = get_compile_time_arg_val(17) == 1;

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_start = get_arg_val<uint32_t>(argidx++);  // Block index
    const uint32_t c_out_block_end = get_arg_val<uint32_t>(argidx++);    // Block index
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);

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

    // Iterate only over assigned C_out blocks
    for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
        const uint32_t c_out_offset_t = c_out_block * matmul_N_t;

        // Read weights and bias for this block
        cb_reserve_back(cb_weight_tiled, weight_tiles);
        uint32_t weight_write_ptr = get_write_ptr(cb_weight_tiled);
        for (uint32_t row = 0; row < matmul_K_t; row++) {
            for (uint32_t col = 0; col < matmul_N_t; col++) {
                uint32_t weight_idx = row * C_out_t + col + c_out_offset_t;
                noc_async_read_tile(weight_idx, weight_reader, weight_write_ptr);
                weight_write_ptr += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_weight_tiled, weight_tiles);

        if constexpr (use_bias) {
            cb_reserve_back(cb_bias_tiled, matmul_N_t);
            uint32_t bias_write_ptr = get_write_ptr(cb_bias_tiled);
            for (uint32_t i = 0; i < matmul_N_t; i++) {
                uint32_t bias_idx = i + c_out_offset_t;
                noc_async_read_tile(bias_idx, bias_reader, bias_write_ptr);
                bias_write_ptr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_bias_tiled, matmul_N_t);
        }

        // Write output for assigned ranges
        for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
            const uint32_t t_block_end = std::min(t_block + T_block_size, t_out_end);

            for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                    const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);

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
