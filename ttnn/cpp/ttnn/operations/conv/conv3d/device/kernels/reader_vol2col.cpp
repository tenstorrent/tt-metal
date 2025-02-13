// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t T_in = get_compile_time_arg_val(1);
    constexpr uint32_t H_in = get_compile_time_arg_val(2);
    constexpr uint32_t W_in = get_compile_time_arg_val(3);
    constexpr uint32_t C_in = get_compile_time_arg_val(4);
    constexpr uint32_t padding_t = get_compile_time_arg_val(5);
    constexpr uint32_t padding_h = get_compile_time_arg_val(6);
    constexpr uint32_t padding_w = get_compile_time_arg_val(7);
    constexpr uint32_t kT = get_compile_time_arg_val(8);
    constexpr uint32_t kH = get_compile_time_arg_val(9);
    constexpr uint32_t kW = get_compile_time_arg_val(10);
    constexpr uint32_t T_out = get_compile_time_arg_val(11);
    constexpr uint32_t H_out = get_compile_time_arg_val(12);
    constexpr uint32_t W_out = get_compile_time_arg_val(13);
    constexpr uint32_t C_out = get_compile_time_arg_val(14);

    constexpr uint32_t cb_vol2col = get_compile_time_arg_val(15);
    // constexpr uint32_t cb_vol2col_out = get_compile_time_arg_val(16);

    constexpr uint32_t in_row_size_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t out_row_size_bytes = get_compile_time_arg_val(17);
    constexpr bool is_padding_zeros = get_compile_time_arg_val(18) == 1;

    /**
     * Implement vol2col. Produce one patch at a time, reading sticks from DRAM
     * directly into cb_vol2col. Write the patch out when it is constructed.
     *
     * Currentl does not support any padding.
     *
     *
     * TODO:
     * - handle non-aligned channels (16 seems to be alignment)
     * - handle padding (zeros and replicate)
     */

    uint32_t argidx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);

    constexpr bool is_dram = true;

    const InterleavedAddrGen<is_dram> in_reader = {.bank_base_address = in_addr, .page_size = in_row_size_bytes};

    const InterleavedAddrGen<true> out_reader = {.bank_base_address = out_addr, .page_size = out_row_size_bytes};

    constexpr uint32_t BF16_BYTES = 2;

    uint32_t cb_write_ptr = get_write_ptr(cb_vol2col);
    uint32_t out_page_idx = 0;
    for (uint32_t t = 0; t < T_out; t++) {
        for (uint32_t h = 0; h < H_out; h++) {
            for (uint32_t w = 0; w < W_out; w++) {
                // patch = input[:, t:t+kD, h:h+kH, w:w+kW, :].reshape(-1)
                for (uint32_t kt = 0; kt < kT; kt++) {
                    for (uint32_t kh = 0; kh < kH; kh++) {
                        for (uint32_t kw = 0; kw < kW; kw++) {
                            uint32_t cb_stick_idx = kt * kH * kW + kh * kW + kw;
                            uint32_t cb_write_offset = cb_stick_idx * C_in * BF16_BYTES;
                            uint32_t cb_write_addr = cb_write_ptr + cb_write_offset;

                            uint32_t t_idx = t + kt;
                            uint32_t h_idx = h + kh;
                            uint32_t w_idx = w + kw;

                            int32_t h_unpad_idx = h_idx - padding_h;
                            int32_t w_unpad_idx = w_idx - padding_w;

                            bool index_is_in_padding = h_unpad_idx < 0 || h_unpad_idx >= (int32_t)H_in ||
                                                       w_unpad_idx < 0 || w_unpad_idx >= (int32_t)W_in;
                            if (index_is_in_padding) {
                                if constexpr (is_padding_zeros) {
                                    constexpr uint32_t num_full_reads = in_row_size_bytes / MEM_ZEROS_SIZE;
                                    constexpr uint32_t partial_read_size = in_row_size_bytes % MEM_ZEROS_SIZE;
                                    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
                                    for (uint32_t i = 0; i < num_full_reads; ++i) {
                                        noc_async_read(zeros_noc_addr, cb_write_addr, MEM_ZEROS_SIZE);
                                        cb_write_addr += MEM_ZEROS_SIZE;
                                    }
                                    if (partial_read_size > 0) {
                                        noc_async_read(zeros_noc_addr, cb_write_addr, partial_read_size);
                                    }
                                    noc_async_read_barrier();
                                    continue;  // don't read from DRAM
                                } else {
                                    // padding replicate
                                    // 4 cases: h_unpad_idx < 0 or >= H_in, w_unpad_idx < 0 or >= W_in
                                    // Update indices for read
                                    if (h_unpad_idx < 0) {
                                        h_unpad_idx = 0;
                                    } else if (h_unpad_idx >= (int32_t)H_in) {
                                        h_unpad_idx = H_in - 1;
                                    }
                                    if (w_unpad_idx < 0) {
                                        w_unpad_idx = 0;
                                    } else if (w_unpad_idx >= (int32_t)W_in) {
                                        w_unpad_idx = W_in - 1;
                                    }
                                }
                            }
                            // Read the patch from cb_vol2col
                            // Write the patch to out_reader

                            uint32_t in_page_idx = t_idx * H_in * W_in + h_unpad_idx * W_in + w_unpad_idx;

                            in_reader.noc_async_read_page(in_page_idx, cb_write_addr);
                            noc_async_read_barrier();
                        }
                    }
                }
                // write patch to out_reader
                uint64_t dst_addr = get_noc_addr(out_page_idx, out_reader);
                noc_async_write(cb_write_ptr, dst_addr, out_row_size_bytes);
                noc_async_write_barrier();

                out_page_idx++;
            }
        }
    }
}
