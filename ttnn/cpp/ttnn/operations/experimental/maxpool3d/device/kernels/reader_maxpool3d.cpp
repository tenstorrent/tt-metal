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

template <uint32_t channels_bytes>
inline void zeroPad(uint32_t cb_write_addr) {
    // Zero-fill from MEM_ZEROS
    constexpr uint32_t num_full_reads = channels_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = channels_bytes % MEM_ZEROS_SIZE;
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
    constexpr uint32_t cb_input_window = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t T_in = get_compile_time_arg_val(2);
    constexpr uint32_t H_in = get_compile_time_arg_val(3);
    constexpr uint32_t W_in = get_compile_time_arg_val(4);
    constexpr uint32_t C = get_compile_time_arg_val(5);
    constexpr uint32_t T_out = get_compile_time_arg_val(6);
    constexpr uint32_t H_out = get_compile_time_arg_val(7);
    constexpr uint32_t W_out = get_compile_time_arg_val(8);
    constexpr uint32_t padding_t = get_compile_time_arg_val(9);
    constexpr uint32_t padding_h = get_compile_time_arg_val(10);
    constexpr uint32_t padding_w = get_compile_time_arg_val(11);
    constexpr uint32_t kernel_t = get_compile_time_arg_val(12);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(13);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(14);
    constexpr uint32_t stride_t = get_compile_time_arg_val(15);
    constexpr uint32_t stride_h = get_compile_time_arg_val(16);
    constexpr uint32_t stride_w = get_compile_time_arg_val(17);
    constexpr uint32_t in_page_size_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t channels_bytes = get_compile_time_arg_val(19);
    constexpr bool is_padding_zeros = get_compile_time_arg_val(20) == 1;

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);

    // Interleaved address generator
    constexpr bool is_dram = true;
    const InterleavedAddrGen<is_dram> in_reader = {.bank_base_address = in_addr, .page_size = in_page_size_bytes};

    constexpr uint32_t H_in_W_in = H_in * W_in;
    constexpr uint32_t window_size = kernel_t * kernel_h * kernel_w;

    // Debug: Print reader kernel parameters
    DPRINT << "READER KERNEL PARAMS:" << ENDL();
    DPRINT << "  Input dims: T_in=" << T_in << ", H_in=" << H_in << ", W_in=" << W_in << ENDL();
    DPRINT << "  Kernel: " << kernel_t << "x" << kernel_h << "x" << kernel_w << " = " << window_size << " sticks"
           << ENDL();
    DPRINT << "  Stride: " << stride_t << "x" << stride_h << "x" << stride_w << ENDL();
    DPRINT << "  H_in_W_in=" << H_in_W_in << ENDL();

    // Process assigned output range
    for (uint32_t t_out = t_out_start; t_out < t_out_end; t_out++) {
        for (uint32_t h_out = h_out_start; h_out < h_out_end; h_out++) {
            for (uint32_t w_out = w_out_start; w_out < w_out_end; w_out++) {
                // Reserve space in CB for 3D window
                cb_reserve_back(cb_input_window, window_size);
                const uint32_t cb_write_ptr = get_write_ptr(cb_input_window);
                uint32_t cb_write_addr = cb_write_ptr;

                // Calculate input window start position
                uint32_t t_in_start = t_out * stride_t;
                uint32_t h_in_start = h_out * stride_h;
                uint32_t w_in_start = w_out * stride_w;

                // Read 3D window: kernel_t × kernel_h × kernel_w sticks
                DPRINT << "READER: Processing window for output (" << t_out << "," << h_out << "," << w_out << ")"
                       << ENDL();
                DPRINT << "  Input window start: (" << t_in_start << "," << h_in_start << "," << w_in_start << ")"
                       << ENDL();
                DPRINT << "  Kernel size: " << kernel_t << "x" << kernel_h << "x" << kernel_w << " = " << window_size
                       << " sticks" << ENDL();

                uint32_t stick_count = 0;
                for (uint32_t kt = 0; kt < kernel_t; kt++) {
                    int32_t t_idx = (int32_t)(t_in_start + kt) - padding_t;
                    const bool outside_t = (t_idx < 0 || t_idx >= (int32_t)T_in);
                    t_idx = clampIndex(t_idx, 0, (int32_t)T_in - 1);

                    for (uint32_t kh = 0; kh < kernel_h; kh++) {
                        int32_t h_idx = (int32_t)(h_in_start + kh) - padding_h;
                        const bool outside_h = (h_idx < 0 || h_idx >= (int32_t)H_in);
                        h_idx = clampIndex(h_idx, 0, (int32_t)H_in - 1);

                        for (uint32_t kw = 0; kw < kernel_w; kw++) {
                            int32_t w_idx = (int32_t)(w_in_start + kw) - padding_w;
                            const bool outside_w = (w_idx < 0 || w_idx >= (int32_t)W_in);
                            const bool in_padding = (outside_t || outside_h || outside_w);
                            w_idx = clampIndex(w_idx, 0, (int32_t)W_in - 1);

                            if constexpr (is_padding_zeros) {
                                if (in_padding) {
                                    // Zero fill
                                    zeroPad<channels_bytes>(cb_write_addr);
                                    cb_write_addr += channels_bytes;
                                    continue;
                                }
                            }

                            // Read stick (all channels at this spatial position)
                            const uint32_t in_page_idx =
                                (uint32_t)(t_idx)*H_in_W_in + (uint32_t)(h_idx)*W_in + (uint32_t)(w_idx);

                            // Read stick (all channels at this spatial position)
                            const uint64_t in_noc_addr = in_reader.get_noc_addr(in_page_idx);
                            noc_async_read(in_noc_addr, cb_write_addr, channels_bytes);

                            cb_write_addr += channels_bytes;
                            stick_count++;
                        }
                    }
                }

                noc_async_read_barrier();
                cb_push_back(cb_input_window, window_size);
            }
        }
    }
}
