// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#if defined(PROFILE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#endif

// Pre-zero CB pages via NOC DMA from MEM_ZEROS so tile-alignment padding is zero.
// Uses MEM_ZEROS_SIZE-aligned transactions (same pattern as zero_out_tiles in conv_reader_common.hpp).
// padded_page_bytes must be a multiple of 16 to guarantee remainder alignment.
template <uint32_t padded_page_bytes>
FORCE_INLINE void pre_zero_pages(uint32_t write_addr, uint32_t num_pages) {
    static_assert(padded_page_bytes % 16 == 0, "CB page size must be 16-byte aligned for NOC transactions");
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t total = num_pages * padded_page_bytes;
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    while (total >= MEM_ZEROS_SIZE) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
        total -= MEM_ZEROS_SIZE;
    }
    if (total > 0) {
        noc_async_read(zeros_noc_addr, write_addr, total);
    }
    noc_async_read_barrier();
}

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

template <typename Reader>
FORCE_INLINE uint64_t
get_input_noc_addr(const Reader& reader, uint32_t in_page_idx, uint32_t c_in_offset_bytes, uint32_t in_row_size_bytes) {
    if constexpr (Reader::DSpec::tensor_shape_static) {
        if constexpr ((reader.dspec().rank() > 1) && (reader.dspec().tensor_shape()[1] > 1)) {
            // Width/block sharded RowMajor tensors may split a logical row across multiple pages.
            // Height-sharded inputs keep a single page per row and should use the direct path below.
            constexpr uint32_t width_in_pages = reader.dspec().tensor_shape()[1];
            const uint32_t col_page_idx = c_in_offset_bytes / in_row_size_bytes;
            const uint32_t in_offset_bytes = c_in_offset_bytes - (col_page_idx * in_row_size_bytes);
            ASSERT(col_page_idx < width_in_pages);

            const uint32_t in_page_id = in_page_idx * width_in_pages + col_page_idx;
            return reader.get_noc_addr(in_page_id, in_offset_bytes);
        }
    }

    return reader.get_noc_addr(in_page_idx, c_in_offset_bytes);
}

// Manages chunked CB writes: reserves TILE_HEIGHT pages, tracks patches written,
// pushes when full, and flushes remaining at the end of a block.
template <uint32_t cb_id, uint32_t padded_page_bytes, uint32_t patch_pad_bytes>
struct ChunkWriter {
    static constexpr uint32_t chunk_max = 32;  // TILE_HEIGHT
    uint32_t remaining;
    uint32_t chunk_size;
    uint32_t in_chunk;
    uint32_t write_addr;

    void init(uint32_t total_patches) {
        remaining = total_patches;
        in_chunk = 0;
        chunk_size = remaining < chunk_max ? remaining : chunk_max;
        cb_reserve_back(cb_id, chunk_size);
        write_addr = get_write_ptr(cb_id);
        if constexpr (patch_pad_bytes > 0) {
            pre_zero_pages<padded_page_bytes>(write_addr, chunk_size);
        }
    }

    // Call after writing one patch to write_addr. Returns true if more patches remain.
    // Callers that need to restore NOC state after a push should check the return value.
    bool advance() {
        if constexpr (patch_pad_bytes > 0) {
            write_addr += patch_pad_bytes;
        }
        in_chunk++;
        if (in_chunk == chunk_size) {
            noc_async_read_barrier();
            cb_push_back(cb_id, chunk_size);
            remaining -= chunk_size;
            in_chunk = 0;
            if (remaining > 0) {
                chunk_size = remaining < chunk_max ? remaining : chunk_max;
                cb_reserve_back(cb_id, chunk_size);
                write_addr = get_write_ptr(cb_id);
                if constexpr (patch_pad_bytes > 0) {
                    pre_zero_pages<padded_page_bytes>(write_addr, chunk_size);
                }
                return true;  // pushed and re-reserved — caller may need to restore NOC state
            }
        }
        return false;
    }

    void flush() {
        if (remaining > 0) {
            if (in_chunk > 0) {
                noc_async_read_barrier();
                cb_push_back(cb_id, chunk_size);
                remaining -= chunk_size;
            }
            while (remaining > 0) {
                chunk_size = remaining < chunk_max ? remaining : chunk_max;
                cb_reserve_back(cb_id, chunk_size);
                cb_push_back(cb_id, chunk_size);
                remaining -= chunk_size;
            }
        }
    }
};

// Copy one (t, h) row of patches from the L1 shard into the vol2col CB.
// Iterates over w positions in [w_block, w_block_end), extracting kT×kH×kW patches
// via one_packet NOC reads.  Calls chunk.advance() after each patch.
template <
    uint32_t kT,
    uint32_t kH,
    uint32_t kW,
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t stride_w,
    uint32_t cb_id,
    uint32_t padded_page_bytes,
    uint32_t patch_pad_bytes>
void vol2col_shard_to_cb(
    uint32_t shard_l1_base,
    uint64_t shard_noc_base,
    uint32_t t_base,
    uint32_t h_base,
    uint32_t w_block,
    uint32_t w_block_end,
    ChunkWriter<cb_id, padded_page_bytes, patch_pad_bytes>& chunk) {
    constexpr uint32_t kW_bytes = kW * C_in_block_bytes;
    noc_async_read_one_packet_set_state(shard_noc_base, kW_bytes);
    for (uint32_t w = w_block; w < w_block_end; w++) {
        const uint32_t w_base = (w - w_block) * stride_w;
        for (uint32_t kt = 0; kt < kT; kt++) {
            const uint32_t t_local = t_base + kt;
            for (uint32_t kh = 0; kh < kH; kh++) {
                const uint32_t h_local = h_base + kh;
                uint32_t shard_offset =
                    (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_base) * C_in_block_bytes;
                noc_async_read_one_packet_with_state(shard_l1_base + shard_offset, chunk.write_addr);
                chunk.write_addr += kW_bytes;
            }
        }
        if (chunk.advance()) {
            noc_async_read_one_packet_set_state(shard_noc_base, kW_bytes);
        }
    }
}

// Shift retained columns to the start of each shard row for sliding-window W reuse.
// With stride_w, adjacent w_blocks overlap by max(0, kW - stride_w) columns, not kW-1.
// After the shift, only (W_shard_cur - overlap) new columns need to be gathered from DRAM.
template <
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t kW,
    uint32_t stride_w>
void shift_retained_w_columns(uint32_t shard_l1_base, uint32_t T_shard_cur, uint32_t h_rows_gathered) {
    constexpr uint32_t overlap_w = kW > stride_w ? kW - stride_w : 0;
    constexpr uint32_t shift_bytes = overlap_w * C_in_block_bytes;
    constexpr uint32_t src_off = (W_shard_max - overlap_w) * C_in_block_bytes;
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        for (uint32_t h_local = 0; h_local < h_rows_gathered; h_local++) {
            const uint32_t row_base = (t_local * H_shard_max_W_shard_max + h_local * W_shard_max) * C_in_block_bytes;
            noc_async_read(get_noc_addr(shard_l1_base + row_base + src_off), shard_l1_base + row_base, shift_bytes);
        }
    }
    noc_async_read_barrier();
}

// Gather rows from DRAM into the L1 shard buffer.
// When check_padding=false, all positions are known to be in-bounds — skip per-position
// boundary checks and clamp/zeroPad logic (~3-6 RISC-V cycles saved per position).
template <
    uint32_t C_in_block_bytes,
    bool is_padding_zeros,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    bool check_padding,
    typename Reader>
void gather_rows_to_shard(
    const Reader& in_reader,
    uint32_t shard_l1_base,
    uint32_t batch_page_base,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count) {
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
        [[maybe_unused]] const bool t_outside = check_padding && (t_in < 0 || t_in >= static_cast<int32_t>(T_in));
        [[maybe_unused]] const int32_t t_clamped =
            check_padding ? clampIndex(t_in, 0, static_cast<int32_t>(T_in) - 1) : t_in;
        for (uint32_t h_local = h_start; h_local < h_end; h_local++) {
            const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
            [[maybe_unused]] const bool h_outside = check_padding && (h_in < 0 || h_in >= static_cast<int32_t>(H_in));
            [[maybe_unused]] const int32_t h_clamped =
                check_padding ? clampIndex(h_in, 0, static_cast<int32_t>(H_in) - 1) : h_in;
            uint32_t shard_offset =
                (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_col_start) * C_in_block_bytes;
            for (uint32_t w_idx = 0; w_idx < w_count; w_idx++) {
                const int32_t w_in = w_shard_start + static_cast<int32_t>(w_col_start + w_idx);
                const uint32_t shard_addr = shard_l1_base + shard_offset;
                if constexpr (check_padding) {
                    const bool w_outside = (w_in < 0 || w_in >= static_cast<int32_t>(W_in));
                    const bool in_padding = t_outside || h_outside || w_outside;
                    if (in_padding) {
                        if constexpr (is_padding_zeros) {
                            zeroPad<C_in_block_bytes>(shard_addr);
                        } else {
                            const int32_t w_clamped = clampIndex(w_in, 0, static_cast<int32_t>(W_in) - 1);
                            const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_clamped) * H_in_W_in +
                                                      static_cast<uint32_t>(h_clamped) * W_in +
                                                      static_cast<uint32_t>(w_clamped);
                            noc_async_read(
                                get_input_noc_addr(in_reader, page_idx, c_in_offset_bytes, in_row_size_bytes),
                                shard_addr,
                                C_in_block_bytes);
                        }
                    } else {
                        const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                                  static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                        noc_async_read(
                            get_input_noc_addr(in_reader, page_idx, c_in_offset_bytes, in_row_size_bytes),
                            shard_addr,
                            C_in_block_bytes);
                    }
                } else {
                    // Fast path: no padding checks
                    const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                              static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                    noc_async_read(
                        get_input_noc_addr(in_reader, page_idx, c_in_offset_bytes, in_row_size_bytes),
                        shard_addr,
                        C_in_block_bytes);
                }
                shard_offset += C_in_block_bytes;
            }
        }
    }
    noc_async_read_barrier();
}

#if defined(CONV3D_H_HALO)
// Halo-aware gather: for H-boundary positions (h_in < 0 or h_in >= H_in),
// reads from the compact halo buffer instead of zero-padding.
// Interior and T/W boundary positions use the standard path.
template <
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    typename Reader>
void gather_rows_halo(
    const Reader& in_reader,
    const Reader& halo_reader,
    uint32_t shard_l1_base,
    uint32_t batch_page_base,
    uint32_t batch_idx,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count,
    uint32_t h_halo_outer_dim_size,
    uint32_t h_halo_H,
    uint32_t h_halo_W,
    uint32_t h_halo_padding_h,
    uint32_t h_halo_padding_w,
    uint32_t h_halo_hbot_base,
    uint32_t h_halo_wleft_base,
    uint32_t h_halo_wright_base) {
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
        const bool t_outside = (t_in < 0 || t_in >= static_cast<int32_t>(T_in));
        const int32_t t_clamped = t_outside ? clampIndex(t_in, 0, static_cast<int32_t>(T_in) - 1) : t_in;
        for (uint32_t h_local = h_start; h_local < h_end; h_local++) {
            const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
            const bool h_outside = (h_in < 0 || h_in >= static_cast<int32_t>(H_in));
            const int32_t h_clamped = h_outside ? clampIndex(h_in, 0, static_cast<int32_t>(H_in) - 1) : h_in;
            uint32_t shard_offset =
                (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_col_start) * C_in_block_bytes;
            for (uint32_t w_idx = 0; w_idx < w_count; w_idx++) {
                const int32_t w_in = w_shard_start + static_cast<int32_t>(w_col_start + w_idx);
                const bool w_outside = (w_in < 0 || w_in >= static_cast<int32_t>(W_in));
                const int32_t w_clamped = w_outside ? clampIndex(w_in, 0, static_cast<int32_t>(W_in) - 1) : w_in;
                const uint32_t shard_addr = shard_l1_base + shard_offset;

                if (t_outside) {
                    // T boundary: zero-fill (causal padding, no T halo buffer)
                    zeroPad<C_in_block_bytes>(shard_addr);
                } else if (h_outside) {
                    // H boundary: read from H halo buffer (top or bottom half)
                    const uint32_t t_global = batch_idx * T_in + static_cast<uint32_t>(t_clamped);
                    uint32_t halo_page;
                    if (h_in < 0) {
                        // Top halo: h_in = -1 → pad_row = h_halo_padding_h + h_in
                        const uint32_t pad_row = h_halo_padding_h + static_cast<uint32_t>(h_in);
                        halo_page = t_global * h_halo_padding_h * h_halo_W + pad_row * h_halo_W +
                                    static_cast<uint32_t>(w_clamped);
                    } else {
                        // Bottom halo: h_in >= H_in → pad_row = h_in - H_in
                        const uint32_t pad_row = static_cast<uint32_t>(h_in) - H_in;
                        halo_page = h_halo_hbot_base + t_global * h_halo_padding_h * h_halo_W + pad_row * h_halo_W +
                                    static_cast<uint32_t>(w_clamped);
                    }
                    noc_async_read(
                        get_input_noc_addr(halo_reader, halo_page, c_in_offset_bytes, in_row_size_bytes),
                        shard_addr,
                        C_in_block_bytes);
                } else if (w_outside && h_halo_padding_w > 0) {
                    // W boundary: read from W halo buffer (left or right half)
                    const uint32_t t_global = batch_idx * T_in + static_cast<uint32_t>(t_clamped);
                    uint32_t halo_page;
                    if (w_in < 0) {
                        // Left halo: w_in = -1 → pad_col = h_halo_padding_w + w_in
                        const uint32_t pad_col = h_halo_padding_w + static_cast<uint32_t>(w_in);
                        halo_page = h_halo_wleft_base + t_global * h_halo_padding_w * h_halo_H + pad_col * h_halo_H +
                                    static_cast<uint32_t>(h_clamped);
                    } else {
                        // Right halo: w_in >= W_in → pad_col = w_in - W_in
                        const uint32_t pad_col = static_cast<uint32_t>(w_in) - W_in;
                        halo_page = h_halo_wright_base + t_global * h_halo_padding_w * h_halo_H + pad_col * h_halo_H +
                                    static_cast<uint32_t>(h_clamped);
                    }
                    noc_async_read(
                        get_input_noc_addr(halo_reader, halo_page, c_in_offset_bytes, in_row_size_bytes),
                        shard_addr,
                        C_in_block_bytes);
                } else if (w_outside) {
                    // W boundary but no W halo buffer: zero-fill
                    zeroPad<C_in_block_bytes>(shard_addr);
                } else {
                    // Interior: read from original unpadded input tensor
                    const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                              static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                    noc_async_read(
                        get_input_noc_addr(in_reader, page_idx, c_in_offset_bytes, in_row_size_bytes),
                        shard_addr,
                        C_in_block_bytes);
                }
                shard_offset += C_in_block_bytes;
            }
        }
    }
    noc_async_read_barrier();
}

// Halo-aware GATHER_ROWS macro: same signature as the non-halo version.
// Args: all_in_bounds, in_reader, shard_l1_base, batch_page_base, c_in_offset_bytes,
//       t_shard_start, T_shard_cur, h_shard_start, h_start, h_end,
//       w_shard_start, w_col_start, w_count
#define GATHER_ROWS(                 \
    all_in_bounds,                   \
    in_rdr,                          \
    shard_l1,                        \
    batch_pg,                        \
    c_in_off,                        \
    t_sh_st,                         \
    T_sh_cur,                        \
    h_sh_st,                         \
    h_st,                            \
    h_en,                            \
    w_sh_st,                         \
    w_col_st,                        \
    w_cnt)                           \
    do {                             \
        gather_rows_halo<            \
            C_in_block_bytes,        \
            H_shard_max_W_shard_max, \
            W_shard_max,             \
            T_in,                    \
            H_in,                    \
            W_in,                    \
            H_in_W_in,               \
            in_row_size_bytes>(      \
            in_rdr,                  \
            halo_reader,             \
            shard_l1,                \
            batch_pg,                \
            batch_idx,               \
            c_in_off,                \
            t_sh_st,                 \
            T_sh_cur,                \
            h_sh_st,                 \
            h_st,                    \
            h_en,                    \
            w_sh_st,                 \
            w_col_st,                \
            w_cnt,                   \
            h_halo_outer_dim_size,   \
            h_halo_H,                \
            h_halo_W,                \
            h_halo_padding_h,        \
            h_halo_padding_w,        \
            h_halo_hbot_base,        \
            h_halo_wleft_base,       \
            h_halo_wright_base);     \
    } while (0)
#else
// Dispatch to fast or slow gather based on runtime bounds check.
#define GATHER_ROWS(all_in_bounds, ...)  \
    do {                                 \
        if (all_in_bounds)               \
            gather_rows_to_shard<        \
                C_in_block_bytes,        \
                is_padding_zeros,        \
                H_shard_max_W_shard_max, \
                W_shard_max,             \
                T_in,                    \
                H_in,                    \
                W_in,                    \
                H_in_W_in,               \
                in_row_size_bytes,       \
                false>(__VA_ARGS__);     \
        else                             \
            gather_rows_to_shard<        \
                C_in_block_bytes,        \
                is_padding_zeros,        \
                H_shard_max_W_shard_max, \
                W_shard_max,             \
                T_in,                    \
                H_in,                    \
                W_in,                    \
                H_in_W_in,               \
                in_row_size_bytes,       \
                true>(__VA_ARGS__);      \
    } while (0)
#endif  // CONV3D_H_HALO

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

    // Padding bytes to append after each patch row to reach tile-aligned CB page width
    constexpr uint32_t patch_pad_bytes = get_compile_time_arg_val(35);
    constexpr uint32_t padded_page_bytes = kT * kH * kW * C_in_block_bytes + patch_pad_bytes;

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
#if defined(CONV3D_INPUT_PROGRESS_SEM)
    const uint32_t input_progress_sem_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t input_progress_t_batch_size = get_arg_val<uint32_t>(argidx++);
#else
    argidx += 2;
#endif
#if defined(CONV3D_H_HALO)
    const uint32_t h_halo_buffer_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_outer_dim_size = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_H = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_W = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_padding_h = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_padding_w = get_arg_val<uint32_t>(argidx++);
    // Compact buffer layout: [H-top | H-bot | W-left | W-right]
    const uint32_t h_halo_hbot_base = h_halo_outer_dim_size * h_halo_padding_h * h_halo_W;
    const uint32_t h_halo_wleft_base = 2u * h_halo_outer_dim_size * h_halo_padding_h * h_halo_W;
    const uint32_t h_halo_wright_base = h_halo_wleft_base + h_halo_outer_dim_size * h_halo_padding_w * h_halo_H;
#else
    argidx += 6;
#endif

    // Tensor accessor for input tensor
    constexpr auto in_args = TensorAccessorArgs<36>();
    const auto in_reader = TensorAccessor(in_args, in_addr, in_row_size_bytes);
#if defined(CONV3D_H_HALO)
    // Halo tensor uses same page layout as input (same alignment, same element size).
    // We reuse in_args since both are DRAM interleaved with the same page structure.
    const auto halo_reader = TensorAccessor(in_args, h_halo_buffer_addr, in_row_size_bytes);
#endif

    constexpr uint32_t num_patches = T_block_size * H_block_size * W_block_size;
    constexpr uint32_t H_in_W_in = H_in * W_in;
    constexpr uint32_t T_in_H_in_W_in = T_in * H_in * W_in;

    // L1 prefetch: enabled when the host allocated a shard buffer (T_shard_max > 0).
    // The host decides based on kernel size, dilation, and L1 budget.
    constexpr bool use_l1_prefetch = (T_shard_max > 0);
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
                    // NOTE: CONV3D_INPUT_PROGRESS_SEM (in-kernel semaphore spin) is disabled because
                    // the writer increments only its own L1 — readers on other cores never see the update.
                    // Inter-CQ ordering is instead guaranteed by the CQ1→CQ0 event mechanism at dispatch time.

                    for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                        const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                        // H rows persist across w_blocks for sliding window W reuse.
                        uint32_t h_rows_gathered = 0;
                        const int32_t t_shard_start =
                            static_cast<int32_t>(t_block * stride_t) - static_cast<int32_t>(padding_t);
                        const int32_t h_shard_start =
                            static_cast<int32_t>(h_block * stride_h) - static_cast<int32_t>(padding_h);
                        const uint32_t T_shard_cur = (t_block_end - 1 - t_block) * stride_t + kT;
                        const uint32_t H_shard_cur = (h_block_end - 1 - h_block) * stride_h + kH;
                        constexpr uint32_t kW_bytes = kW * C_in_block_bytes;
                        static_assert(kW_bytes <= NOC_MAX_BURST_SIZE, "kW_bytes exceeds NOC_MAX_BURST_SIZE");

                        // Precompute T/H bounds for shard_all_in_bounds (W is per-w_block).
                        const bool th_in_bounds =
                            t_shard_start >= 0 &&
                            (t_shard_start + static_cast<int32_t>(T_shard_cur) - 1) < static_cast<int32_t>(T_in) &&
                            h_shard_start >= 0 &&
                            (h_shard_start + static_cast<int32_t>(H_shard_cur) - 1) < static_cast<int32_t>(H_in);

                        for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                            const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);
#if defined(ABLATE_DM) || defined(ABLATE_READER_DM)
                            // Skip all DRAM gathers; push empty patches so compute can run.
                            {
                                constexpr uint32_t chunk_max = 32;
                                uint32_t patches_remaining = num_patches;
                                while (patches_remaining > 0) {
                                    const uint32_t chunk =
                                        patches_remaining < chunk_max ? patches_remaining : chunk_max;
                                    cb_reserve_back(cb_vol2col, chunk);
                                    cb_push_back(cb_vol2col, chunk);
                                    patches_remaining -= chunk;
                                }
                            }
#else
                            if constexpr (use_l1_prefetch) {
                                const int32_t w_shard_start =
                                    static_cast<int32_t>(w_block * stride_w) - static_cast<int32_t>(padding_w);
                                const uint32_t W_shard_cur = (w_block_end - 1 - w_block) * stride_w + kW;
                                const bool shard_all_in_bounds = th_in_bounds && w_shard_start >= 0 &&
                                                                 (w_shard_start + static_cast<int32_t>(W_shard_cur) -
                                                                  1) < static_cast<int32_t>(W_in);

                                // --- SLIDING WINDOW W + H-ROW INTERLEAVED GATHER ---
                                // For w_block > first: shift retained kW-1 columns to shard start,
                                // then gather only W_block new columns for existing h-rows.
                                // H rows persist across w_blocks — no re-gather for retained rows.
                                const bool is_first_w = (w_block == w_out_start);

                                // W overlap between adjacent w_blocks: kW - stride_w columns.
                                // No overlap when stride_w >= kW (each block reads entirely new data).
                                constexpr uint32_t overlap_w = kW > stride_w ? kW - stride_w : 0;

                                // Reset h_rows when no W overlap to retain or on first w_block
                                if (is_first_w || overlap_w == 0) {
                                    h_rows_gathered = 0;
                                }

                                if constexpr (overlap_w > 0) {
                                    if (!is_first_w && h_rows_gathered > 0) {
#if defined(PROFILE_ZONES)
                                    DeviceZoneScopedN("r-shard-w-shift");
#endif
                                    shift_retained_w_columns<
                                        C_in_block_bytes,
                                        H_shard_max_W_shard_max,
                                        W_shard_max,
                                        kW,
                                        stride_w>(shard_l1_base, T_shard_cur, h_rows_gathered);

                                    // Gather new W columns for existing h-rows
                                    const uint32_t new_w_cols = W_shard_cur - overlap_w;
                                    GATHER_ROWS(
                                        shard_all_in_bounds,
                                        in_reader,
                                        shard_l1_base,
                                        batch_page_base,
                                        c_in_offset_bytes,
                                        t_shard_start,
                                        T_shard_cur,
                                        h_shard_start,
                                        0u,
                                        h_rows_gathered,
                                        w_shard_start,
                                        overlap_w,
                                        new_w_cols);
                                    }
                                }

                                ChunkWriter<cb_vol2col, padded_page_bytes, patch_pad_bytes> chunk;
                                chunk.init(num_patches);

                                for (uint32_t t = t_block; t < t_block_end; t++) {
                                    const uint32_t t_base = (t - t_block) * stride_t;
                                    for (uint32_t h = h_block; h < h_block_end; h++) {
                                        const uint32_t h_base = (h - h_block) * stride_h;

                                        // Gather shard rows needed for this output h (incremental)
                                        const uint32_t h_needed = h_base + kH;
                                        if (h_needed > h_rows_gathered) {
#if defined(PROFILE_ZONES)
                                            DeviceZoneScopedN("r-shard-gather");
#endif
                                            GATHER_ROWS(
                                                shard_all_in_bounds,
                                                in_reader,
                                                shard_l1_base,
                                                batch_page_base,
                                                c_in_offset_bytes,
                                                t_shard_start,
                                                T_shard_cur,
                                                h_shard_start,
                                                h_rows_gathered,
                                                h_needed,
                                                w_shard_start,
                                                0u,
                                                W_shard_cur);
                                            h_rows_gathered = h_needed;
                                        }

                                        // Vol2col for this (t, h) across all w
                                        {
#if defined(PROFILE_ZONES)
                                            DeviceZoneScopedN("r-vol2col-copy");
#endif
                                            vol2col_shard_to_cb<
                                                kT,
                                                kH,
                                                kW,
                                                C_in_block_bytes,
                                                H_shard_max_W_shard_max,
                                                W_shard_max,
                                                stride_w,
                                                cb_vol2col,
                                                padded_page_bytes,
                                                patch_pad_bytes>(
                                                shard_l1_base,
                                                shard_noc_base,
                                                t_base,
                                                h_base,
                                                w_block,
                                                w_block_end,
                                                chunk);
                                        }
                                    }
                                }
                                chunk.flush();

                            } else {
                                // ============================================================
                                // DIRECT READER (for 1x1x1 or dilated kernels, no spatial reuse)
                                // Push patches in TILE_HEIGHT-sized chunks to keep cb_vol2col small.
                                // ============================================================
#if defined(PROFILE_ZONES)
                                DeviceZoneScopedN("r-direct-gather");
#endif
                                const uint32_t t_block_s_start = t_block * stride_t;
                                const uint32_t t_block_s_end = t_block_end * stride_t;
                                const uint32_t h_block_s_start = h_block * stride_h;
                                const uint32_t h_block_s_end = h_block_end * stride_h;
                                const uint32_t w_block_s_start = w_block * stride_w;
                                const uint32_t w_block_s_end = w_block_end * stride_w;

                                ChunkWriter<cb_vol2col, padded_page_bytes, patch_pad_bytes> chunk;
                                chunk.init(num_patches);

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
                                                                zeroPad<C_in_block_bytes>(chunk.write_addr);
                                                                chunk.write_addr += C_in_block_bytes;
                                                                continue;
                                                            }
                                                        }

                                                        const uint32_t page_idx =
                                                            batch_page_base + static_cast<uint32_t>(t_idx) * H_in_W_in +
                                                            static_cast<uint32_t>(h_idx) * W_in +
                                                            static_cast<uint32_t>(w_idx);
                                                        const uint64_t noc_addr = get_input_noc_addr(
                                                            in_reader, page_idx, c_in_offset_bytes, in_row_size_bytes);
                                                        noc_async_read(noc_addr, chunk.write_addr, C_in_block_bytes);
                                                        chunk.write_addr += C_in_block_bytes;
                                                    }
                                                }
                                            }

                                            chunk.advance();
                                        }
                                    }
                                }
                                chunk.flush();
                            }
#endif  // ABLATE_DM
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
