// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline int32_t clampIndex(int32_t idx, int32_t lower_bound, int32_t upper_bound) {
    // If we're doing replicate padding, clamp idx into [lower_bound, upper_bound].
    if (idx < lower_bound) {
        return lower_bound;
    }
    if (idx > upper_bound) {
        return upper_bound;
    }
    return idx;
}

template <uint32_t in_row_size_bytes>
inline void zeroPad(uint32_t cb_write_addr) {
    // Zero-fill from MEM_ZEROS
    constexpr uint32_t num_full_reads = in_row_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = in_row_size_bytes % MEM_ZEROS_SIZE;
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    for (uint32_t i = 0; i < num_full_reads; ++i) {
        noc_async_read(zeros_noc_addr, cb_write_addr, MEM_ZEROS_SIZE);
        cb_write_addr += MEM_ZEROS_SIZE;
    }
    if (partial_read_size > 0) {
        noc_async_read(zeros_noc_addr, cb_write_addr, partial_read_size);
    }
}

void kernel_main() {
    constexpr uint32_t cb_vol2col = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t T_in = get_compile_time_arg_val(2);
    constexpr uint32_t H_in = get_compile_time_arg_val(3);
    constexpr uint32_t W_in = get_compile_time_arg_val(4);
    constexpr uint32_t C_in = get_compile_time_arg_val(5);
    constexpr uint32_t T_out = get_compile_time_arg_val(6);
    constexpr uint32_t H_out = get_compile_time_arg_val(7);
    constexpr uint32_t W_out = get_compile_time_arg_val(8);
    constexpr uint32_t C_out = get_compile_time_arg_val(9);
    constexpr uint32_t padding_t = get_compile_time_arg_val(10);
    constexpr uint32_t padding_h = get_compile_time_arg_val(11);
    constexpr uint32_t padding_w = get_compile_time_arg_val(12);
    constexpr uint32_t kT = get_compile_time_arg_val(13);
    constexpr uint32_t kH = get_compile_time_arg_val(14);
    constexpr uint32_t kW = get_compile_time_arg_val(15);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(16);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(17);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(18);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t in_row_size_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t C_in_block_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t out_row_size_bytes = get_compile_time_arg_val(22);
    constexpr bool is_padding_zeros = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(24);
    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(argidx++);
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

    // Interleaved address generators
    constexpr bool is_dram = true;
    const InterleavedAddrGen<is_dram> in_reader = {.bank_base_address = in_addr, .page_size = in_row_size_bytes};

    constexpr uint32_t num_patches = T_block_size * H_block_size * W_block_size;
    constexpr uint32_t H_in_W_in = H_in * W_in;

    for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
        const uint32_t c_in_offset_bytes = c_in_block * C_in_block_bytes;
        // Iterate only over assigned C_out blocks
        for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
            // 3D blocking loops over assigned ranges:
            for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
                const uint32_t t_block_end = std::min(t_block + T_block_size, t_out_end);

                for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                    const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                    for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                        const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);
                        // Now iterate through the sub-tile
                        cb_reserve_back(cb_vol2col, num_patches);
                        const uint32_t cb_write_ptr = get_write_ptr(cb_vol2col);
                        uint32_t cb_write_addr = cb_write_ptr;
                        for (uint32_t t = t_block; t < t_block_end; ++t) {
                            for (uint32_t h = h_block; h < h_block_end; ++h) {
                                for (uint32_t w = w_block; w < w_block_end; ++w) {
                                    // For each output coordinate (t, h, w),
                                    // gather the kT*kH*kW patch around (t,h,w).
                                    for (uint32_t kt = 0; kt < kT; kt++) {
                                        int32_t t_idx = (int32_t)(t + kt) - padding_t;
                                        const bool outside_t = (t_idx < 0 || t_idx >= (int32_t)T_in);
                                        t_idx = clampIndex(t_idx, 0, (int32_t)T_in - 1);

                                        for (uint32_t kh = 0; kh < kH; kh++) {
                                            int32_t h_idx = (int32_t)(h + kh) - padding_h;
                                            const bool outside_h = (h_idx < 0 || h_idx >= (int32_t)H_in);
                                            h_idx = clampIndex(h_idx, 0, (int32_t)H_in - 1);

                                            for (uint32_t kw = 0; kw < kW; kw++) {
                                                int32_t w_idx = (int32_t)(w + kw) - padding_w;
                                                const bool outside_w = (w_idx < 0 || w_idx >= (int32_t)W_in);
                                                const bool in_padding = (outside_t || outside_h || outside_w);
                                                w_idx = clampIndex(w_idx, 0, (int32_t)W_in - 1);

                                                if constexpr (is_padding_zeros) {
                                                    if (in_padding) {
                                                        // Zero fill
                                                        zeroPad<C_in_block_bytes>(cb_write_addr);
                                                        cb_write_addr += C_in_block_bytes;
                                                        continue;
                                                    }
                                                }

                                                // Now do the normal read from DRAM.
                                                // Flattened index in the input
                                                const uint32_t in_page_idx = (uint32_t)(t_idx)*H_in_W_in +
                                                                             (uint32_t)(h_idx)*W_in + (uint32_t)(w_idx);
                                                const uint64_t in_noc_addr =
                                                    in_reader.get_noc_addr(in_page_idx, c_in_offset_bytes /*offset*/);
                                                noc_async_read(in_noc_addr, cb_write_addr, C_in_block_bytes);

                                                cb_write_addr += C_in_block_bytes;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_vol2col, num_patches);
                        // End of w_block
                    }
                    // End of h_block
                }
                // End of t_block
            }
        }
    }
}
