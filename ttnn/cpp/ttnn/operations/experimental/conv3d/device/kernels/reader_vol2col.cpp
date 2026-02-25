// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

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

template <bool is_sharded, typename Reader>
FORCE_INLINE void compute_page_id_and_offset_for_c_in_block(
    const Reader& reader,
    uint32_t in_page_idx,
    uint32_t c_in_offset_bytes,
    uint32_t in_row_size_bytes,
    uint32_t& in_page_id,
    uint32_t& in_offset_bytes) {
    if constexpr (is_sharded) {
        // For width/block sharded RowMajor tensors, a "row" is split into multiple pages of size in_row_size_bytes.
        // TensorAccessor expects page_id to be flattened over (row_idx, col_page_idx).
        //
        // For height sharded layouts, col_page_idx is always 0 and the channel selection is done via offset.
        const uint32_t col_page_idx = c_in_offset_bytes / in_row_size_bytes;
        in_offset_bytes = c_in_offset_bytes - (col_page_idx * in_row_size_bytes);
        const uint32_t width_in_pages = reader.dspec().tensor_shape()[1];
        in_page_id = in_page_idx * width_in_pages + col_page_idx;
    } else {
        in_page_id = in_page_idx;
        in_offset_bytes = c_in_offset_bytes;
    }
}

void kernel_main() {
    DeviceZoneScopedN("CONV3D-READER");
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
    constexpr uint32_t stride_t = get_compile_time_arg_val(25);
    constexpr uint32_t stride_h = get_compile_time_arg_val(26);
    constexpr uint32_t stride_w = get_compile_time_arg_val(27);
    constexpr uint32_t dilation_t = get_compile_time_arg_val(28);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(29);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(30);

    // L1 prefetch buffer parameters
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(31);
    constexpr uint32_t T_shard_max = get_compile_time_arg_val(32);
    constexpr uint32_t H_shard_max = get_compile_time_arg_val(33);
    constexpr uint32_t W_shard_max = get_compile_time_arg_val(34);

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

    // Tensor accessor for input tensor
    constexpr auto in_args = TensorAccessorArgs<35>();
    const auto in_reader = TensorAccessor(in_args, in_addr, in_row_size_bytes);

    constexpr uint32_t num_patches = T_block_size * H_block_size * W_block_size;
    constexpr uint32_t H_in_W_in = H_in * W_in;
    constexpr uint32_t T_in_H_in_W_in = T_in * H_in * W_in;

    // L1 prefetch: enabled for kernels > 1x1x1 (spatial reuse > 1)
    constexpr bool use_l1_prefetch =
        (kT > 1 || kH > 1 || kW > 1) && (dilation_t == 1 && dilation_h == 1 && dilation_w == 1);
    constexpr uint32_t H_shard_max_W_shard_max = H_shard_max * W_shard_max;

    // Reserve shard buffer once (used as scratch space, not streaming CB)
    uint32_t shard_l1_base = 0;
    uint64_t shard_noc_base = 0;
    if constexpr (use_l1_prefetch) {
        constexpr uint32_t shard_total = T_shard_max * H_shard_max_W_shard_max;
        cb_reserve_back(cb_input_shard, shard_total);
        shard_l1_base = get_write_ptr(cb_input_shard);
        shard_noc_base = get_noc_addr(shard_l1_base);
    }

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
        const uint32_t batch_page_base = batch_idx * T_in_H_in_W_in;
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

                            if constexpr (use_l1_prefetch) {
                                // ============================================================
                                // TWO-PHASE READER (L1 prefetch for kernels > 1x1x1)
                                // Phase 1: Gather receptive field DRAM -> L1 shard buffer
                                // Phase 2: Assemble vol2col patches L1 shard -> CB
                                // ============================================================

                                // Compute shard bounds for this spatial block
                                const int32_t t_shard_start =
                                    static_cast<int32_t>(t_block * stride_t) - static_cast<int32_t>(padding_t);
                                const int32_t h_shard_start =
                                    static_cast<int32_t>(h_block * stride_h) - static_cast<int32_t>(padding_h);
                                const int32_t w_shard_start =
                                    static_cast<int32_t>(w_block * stride_w) - static_cast<int32_t>(padding_w);
                                const uint32_t T_shard_cur = (t_block_end - 1 - t_block) * stride_t + kT;
                                const uint32_t H_shard_cur = (h_block_end - 1 - h_block) * stride_h + kH;
                                const uint32_t W_shard_cur = (w_block_end - 1 - w_block) * stride_w + kW;

                                // --- Phase 1: DRAM -> L1 Gather ---
                                {
                                    DeviceZoneScopedN("CONV3D-RD-GATHER");

                                    const bool shard_all_in_bounds =
                                        t_shard_start >= 0 &&
                                        (static_cast<uint32_t>(t_shard_start) + T_shard_cur) <= T_in &&
                                        h_shard_start >= 0 &&
                                        (static_cast<uint32_t>(h_shard_start) + H_shard_cur) <= H_in &&
                                        w_shard_start >= 0 &&
                                        (static_cast<uint32_t>(w_shard_start) + W_shard_cur) <= W_in;

                                    if (shard_all_in_bounds) {
                                        // Fast path: no boundary checks needed
                                        uint32_t page_t =
                                            batch_page_base + static_cast<uint32_t>(t_shard_start) * H_in_W_in;
                                        for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
                                            uint32_t page_h = page_t + static_cast<uint32_t>(h_shard_start) * W_in;
                                            for (uint32_t h_local = 0; h_local < H_shard_cur; h_local++) {
                                                uint32_t page_idx = page_h + static_cast<uint32_t>(w_shard_start);
                                                uint32_t shard_offset =
                                                    (t_local * H_shard_max_W_shard_max + h_local * W_shard_max) *
                                                    C_in_block_bytes;
                                                for (uint32_t w_local = 0; w_local < W_shard_cur; w_local++) {
                                                    const uint64_t noc_addr =
                                                        in_reader.get_noc_addr(page_idx, c_in_offset_bytes);
                                                    noc_async_read(
                                                        noc_addr, shard_l1_base + shard_offset, C_in_block_bytes);
                                                    page_idx++;
                                                    shard_offset += C_in_block_bytes;
                                                }
                                                page_h += W_in;
                                            }
                                            page_t += H_in_W_in;
                                        }
                                    } else {
                                        // Slow path: per-position boundary checking
                                        for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
                                            const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
                                            const bool t_outside = (t_in < 0 || t_in >= static_cast<int32_t>(T_in));
                                            const int32_t t_clamped =
                                                clampIndex(t_in, 0, static_cast<int32_t>(T_in) - 1);

                                            for (uint32_t h_local = 0; h_local < H_shard_cur; h_local++) {
                                                const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
                                                const bool h_outside = (h_in < 0 || h_in >= static_cast<int32_t>(H_in));
                                                const int32_t h_clamped =
                                                    clampIndex(h_in, 0, static_cast<int32_t>(H_in) - 1);

                                                uint32_t shard_offset =
                                                    (t_local * H_shard_max_W_shard_max + h_local * W_shard_max) *
                                                    C_in_block_bytes;

                                                for (uint32_t w_local = 0; w_local < W_shard_cur; w_local++) {
                                                    const int32_t w_in = w_shard_start + static_cast<int32_t>(w_local);
                                                    const bool w_outside =
                                                        (w_in < 0 || w_in >= static_cast<int32_t>(W_in));
                                                    const bool in_padding = t_outside || h_outside || w_outside;
                                                    const uint32_t shard_addr = shard_l1_base + shard_offset;

                                                    if (in_padding) {
                                                        if constexpr (is_padding_zeros) {
                                                            zeroPad<C_in_block_bytes>(shard_addr);
                                                        } else {
                                                            const int32_t w_clamped =
                                                                clampIndex(w_in, 0, static_cast<int32_t>(W_in) - 1);
                                                            const uint32_t page_idx =
                                                                batch_page_base +
                                                                static_cast<uint32_t>(t_clamped) * H_in_W_in +
                                                                static_cast<uint32_t>(h_clamped) * W_in +
                                                                static_cast<uint32_t>(w_clamped);
                                                            const uint64_t noc_addr =
                                                                in_reader.get_noc_addr(page_idx, c_in_offset_bytes);
                                                            noc_async_read(noc_addr, shard_addr, C_in_block_bytes);
                                                        }
                                                    } else {
                                                        const uint32_t page_idx =
                                                            batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                                            static_cast<uint32_t>(h_in) * W_in +
                                                            static_cast<uint32_t>(w_in);
                                                        const uint64_t noc_addr =
                                                            in_reader.get_noc_addr(page_idx, c_in_offset_bytes);
                                                        noc_async_read(noc_addr, shard_addr, C_in_block_bytes);
                                                    }

                                                    shard_offset += C_in_block_bytes;
                                                }
                                            }
                                        }
                                    }
                                    noc_async_read_barrier();
                                }

                                // --- Phase 2: L1 Vol2col -> CB ---
                                {
                                    DeviceZoneScopedN("CONV3D-RD-VOL2COL");
                                    cb_reserve_back(cb_vol2col, num_patches);
                                    uint32_t cb_write_addr = get_write_ptr(cb_vol2col);

                                    for (uint32_t t = t_block; t < t_block_end; t++) {
                                        const uint32_t t_base = (t - t_block) * stride_t;
                                        for (uint32_t h = h_block; h < h_block_end; h++) {
                                            const uint32_t h_base = (h - h_block) * stride_h;
                                            for (uint32_t w = w_block; w < w_block_end; w++) {
                                                const uint32_t w_base = (w - w_block) * stride_w;

                                                for (uint32_t kt = 0; kt < kT; kt++) {
                                                    const uint32_t t_local = t_base + kt;
                                                    for (uint32_t kh = 0; kh < kH; kh++) {
                                                        const uint32_t h_local = h_base + kh;
                                                        uint32_t shard_offset = (t_local * H_shard_max_W_shard_max +
                                                                                 h_local * W_shard_max + w_base) *
                                                                                C_in_block_bytes;
                                                        constexpr uint32_t kW_bytes = kW * C_in_block_bytes;
                                                        noc_async_read(
                                                            shard_noc_base + shard_offset, cb_write_addr, kW_bytes);
                                                        shard_offset += kW_bytes;
                                                        cb_write_addr += kW_bytes;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    noc_async_read_barrier();
                                    cb_push_back(cb_vol2col, num_patches);
                                }

                            } else {
                                // ============================================================
                                // DIRECT READER (for 1x1x1 or dilated kernels, no spatial reuse)
                                // ============================================================
                                const uint32_t t_block_s_start = t_block * stride_t;
                                const uint32_t t_block_s_end = t_block_end * stride_t;
                                const uint32_t h_block_s_start = h_block * stride_h;
                                const uint32_t h_block_s_end = h_block_end * stride_h;
                                const uint32_t w_block_s_start = w_block * stride_w;
                                const uint32_t w_block_s_end = w_block_end * stride_w;

                                cb_reserve_back(cb_vol2col, num_patches);
                                uint32_t cb_write_addr = get_write_ptr(cb_vol2col);

                                for (uint32_t t = t_block_s_start; t < t_block_s_end; t += stride_t) {
                                    for (uint32_t h = h_block_s_start; h < h_block_s_end; h += stride_h) {
                                        for (uint32_t w = w_block_s_start; w < w_block_s_end; w += stride_w) {
                                            for (uint32_t kt = 0; kt < kT; kt++) {
                                                int32_t t_idx = static_cast<int32_t>(t + kt * dilation_t) -
                                                                static_cast<int32_t>(padding_t);
                                                const bool outside_t =
                                                    (t_idx < 0 || t_idx >= static_cast<int32_t>(T_in));
                                                t_idx = clampIndex(t_idx, 0, static_cast<int32_t>(T_in) - 1);

                                                for (uint32_t kh = 0; kh < kH; kh++) {
                                                    int32_t h_idx = static_cast<int32_t>(h + kh * dilation_h) -
                                                                    static_cast<int32_t>(padding_h);
                                                    const bool outside_h =
                                                        (h_idx < 0 || h_idx >= static_cast<int32_t>(H_in));
                                                    h_idx = clampIndex(h_idx, 0, static_cast<int32_t>(H_in) - 1);

                                                    for (uint32_t kw = 0; kw < kW; kw++) {
                                                        int32_t w_idx = static_cast<int32_t>(w + kw * dilation_w) -
                                                                        static_cast<int32_t>(padding_w);
                                                        const bool outside_w =
                                                            (w_idx < 0 || w_idx >= static_cast<int32_t>(W_in));
                                                        const bool in_padding = outside_t || outside_h || outside_w;
                                                        w_idx = clampIndex(w_idx, 0, static_cast<int32_t>(W_in) - 1);

                                                        if constexpr (is_padding_zeros) {
                                                            if (in_padding) {
                                                                zeroPad<C_in_block_bytes>(cb_write_addr);
                                                                cb_write_addr += C_in_block_bytes;
                                                                continue;
                                                            }
                                                        }

                                                        const uint32_t page_idx =
                                                            batch_page_base + static_cast<uint32_t>(t_idx) * H_in_W_in +
                                                            static_cast<uint32_t>(h_idx) * W_in +
                                                            static_cast<uint32_t>(w_idx);
                                                        const uint64_t noc_addr =
                                                            in_reader.get_noc_addr(page_idx, c_in_offset_bytes);
                                                        noc_async_read(noc_addr, cb_write_addr, C_in_block_bytes);
                                                        cb_write_addr += C_in_block_bytes;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                noc_async_read_barrier();
                                cb_push_back(cb_vol2col, num_patches);
                            }
                            // End of w_block
                        }
                        // End of h_block
                    }
                    // End of t_block
                }
            }
        }
    }
}
