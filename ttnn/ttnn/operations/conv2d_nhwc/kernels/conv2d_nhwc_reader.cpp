// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc reader (NCRISC, NoC0) — regime R0.
//
// Block operations owned here:
//   fetch_window_block   : the m-block's input footprint from DRAM into cb_act_window
//                          (reader-private scratch), one read barrier per m-block. Columns are
//                          clipped to what the block reads (R10 column clip): a 2-D clip for a
//                          block inside one output row, a trimmed contiguous range otherwise.
//   gather_im2col_kblock : per K-block, per tile-row: 32 im2col rows x (bk*32) elements into
//                          cb_act_rm (bk tile-sized pages of RM rows), as L1->L1 NoC copies of
//                          each (pixel, kernel-row band) in-image run plus zero runs from
//                          MEM_ZEROS_BASE for padding. One read barrier per tile-row.
//
// No helper expresses a sliding-window gather with per-pixel padding runs
// (local_copy_helpers_dataflow copies one contiguous region per call), so this is the
// block operation built for it (op_design.md "Helpers considered and rejected").
// Phase 0 assumes dilation == 1, so a kernel-row band is one contiguous window run.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_act_window = get_compile_time_arg_val(0);
constexpr uint32_t cb_act_rm = get_compile_time_arg_val(1);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(3);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(5);
constexpr uint32_t band_tiles = get_compile_time_arg_val(6);
constexpr uint32_t tap_width = get_compile_time_arg_val(7);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(8);
constexpr uint32_t stick_bytes = get_compile_time_arg_val(9);
constexpr uint32_t kH = get_compile_time_arg_val(10);
constexpr uint32_t kW = get_compile_time_arg_val(11);
constexpr uint32_t H = get_compile_time_arg_val(12);
constexpr uint32_t W = get_compile_time_arg_val(13);
constexpr uint32_t H_out = get_compile_time_arg_val(14);
constexpr uint32_t W_out = get_compile_time_arg_val(15);
constexpr uint32_t M = get_compile_time_arg_val(16);
constexpr auto input_args = TensorAccessorArgs<17>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t tap_bytes = tap_width * elem_bytes;
constexpr uint32_t band_elems = band_tiles * TILE_DIM;
constexpr uint32_t kblock_elems = block_k_tiles * TILE_DIM;
constexpr uint32_t im2col_row_bytes = kblock_elems * elem_bytes;

FORCE_INLINE int32_t clamp_i(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

FORCE_INLINE void zero_run(uint64_t zeros_noc, uint32_t dst, uint32_t bytes) {
    while (bytes > 0) {
        const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros_noc, dst, chunk);
        dst += chunk;
        bytes -= chunk;
    }
}

// One im2col row (one output pixel) restricted to K-block columns [k0, k0 + kblock_elems).
FORCE_INLINE void gather_row(
    uint64_t zeros_noc,
    uint32_t window_base,
    int32_t win_row0,
    int32_t col_lo,
    int32_t win_w,
    uint32_t n,
    int32_t h_base,
    int32_t w_base,
    uint32_t k0,
    uint32_t dst) {
    // In-image tap columns of this pixel: kw in [kw_lo, kw_hi).
    const int32_t kw_lo = w_base < 0 ? -w_base : 0;
    int32_t kw_hi = static_cast<int32_t>(W) - w_base;
    if (kw_hi > static_cast<int32_t>(kW)) {
        kw_hi = kW;
    }
    const uint32_t v_lo = static_cast<uint32_t>(kw_lo) * tap_width;
    const uint32_t v_hi = kw_hi > kw_lo ? static_cast<uint32_t>(kw_hi) * tap_width : v_lo;

    uint32_t c = k0;
    const uint32_t end = k0 + kblock_elems;
    while (c < end) {
        const uint32_t kh = c / band_elems;
        const uint32_t o = c - kh * band_elems;
        uint32_t seg_end = band_elems;
        if (seg_end - o > end - c) {
            seg_end = o + (end - c);
        }
        const uint32_t seg_dst = dst + (c - k0) * elem_bytes;
        const int32_t h = h_base + static_cast<int32_t>(kh);
        if (h < 0 || h >= static_cast<int32_t>(H)) {
            zero_run(zeros_noc, seg_dst, (seg_end - o) * elem_bytes);
        } else {
            const uint32_t a = o > v_lo ? o : v_lo;
            const uint32_t b = seg_end < v_hi ? seg_end : v_hi;
            if (a >= b) {
                zero_run(zeros_noc, seg_dst, (seg_end - o) * elem_bytes);
            } else {
                if (a > o) {
                    zero_run(zeros_noc, seg_dst, (a - o) * elem_bytes);
                }
                // Window element of band offset a: stick (n,h,w_base) of the clipped window, plus a.
                const int32_t stick = (static_cast<int32_t>(n * H) + h - win_row0) * win_w + (w_base - col_lo);
                const int32_t src_elem = stick * static_cast<int32_t>(tap_width) + static_cast<int32_t>(a);
                noc_async_read(
                    get_noc_addr(window_base + static_cast<uint32_t>(src_elem) * elem_bytes),
                    seg_dst + (a - o) * elem_bytes,
                    (b - a) * elem_bytes);
                if (b < seg_end) {
                    zero_run(zeros_noc, seg_dst + (b - o) * elem_bytes, (seg_end - b) * elem_bytes);
                }
            }
        }
        c += seg_end - o;
    }
}

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t m_tiles_this_core = get_arg_val<uint32_t>(2);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(3);
    const int32_t padding = static_cast<int32_t>(get_arg_val<uint32_t>(4));
    const int32_t stride = static_cast<int32_t>(get_arg_val<uint32_t>(5));
    const int32_t dilation = static_cast<int32_t>(get_arg_val<uint32_t>(6));

    const auto input = TensorAccessor(input_args, input_addr);
    const uint32_t window_base = get_write_ptr(cb_act_window);
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    constexpr uint32_t hw_out = H_out * W_out;
    const uint32_t m_tile_end = m_tile_start + m_tiles_this_core;

    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        const uint32_t t0 = m_tile_start + m_block * block_m_tiles;
        const uint32_t t1 = (t0 + block_m_tiles < m_tile_end) ? t0 + block_m_tiles : m_tile_end;

        // ---- fetch_window_block (mirrors host _window_bounds) ----
        // Window stick of input (row, col) = (row - row_lo) * win_w + (col - col_lo).
        //   one output row : 2-D clip, every row holds [col_lo, last_hi], win_w = its width.
        //   several rows   : contiguous flattened range (win_w = W); the first / last input row
        //                    is trimmed to [col_lo, ...] / [..., last_hi] unless padding-clamped.
        const uint32_t pix0 = t0 * TILE_DIM;
        const uint32_t pix1 = ((t1 * TILE_DIM < M) ? t1 * TILE_DIM : M) - 1;
        const uint32_t n0 = pix0 / hw_out, r0 = pix0 - n0 * hw_out, ho0 = r0 / W_out;
        const uint32_t n1 = pix1 / hw_out, r1 = pix1 - n1 * hw_out, ho1 = r1 / W_out;
        const int32_t wo0 = static_cast<int32_t>(r0 - ho0 * W_out);
        const int32_t wo1 = static_cast<int32_t>(r1 - ho1 * W_out);
        constexpr int32_t Hi = static_cast<int32_t>(H);
        constexpr int32_t Wi = static_cast<int32_t>(W);
        const int32_t raw_lo = static_cast<int32_t>(ho0) * stride - padding;
        const int32_t raw_hi = static_cast<int32_t>(ho1) * stride - padding + dilation * static_cast<int32_t>(kH - 1);
        const int32_t row_lo = static_cast<int32_t>(n0 * H) + clamp_i(raw_lo, 0, Hi - 1);
        int32_t row_hi = static_cast<int32_t>(n1 * H) + clamp_i(raw_hi, 0, Hi - 1);
        row_hi = row_hi < row_lo ? row_lo : row_hi;
        const int32_t c_first = clamp_i(wo0 * stride - padding, 0, Wi - 1);
        const int32_t c_last = clamp_i(wo1 * stride - padding + dilation * static_cast<int32_t>(kW - 1), 0, Wi - 1);
        int32_t col_lo, last_hi, win_w, mid_lo, mid_hi;
        if (n0 == n1 && ho0 == ho1) {
            col_lo = c_first;
            last_hi = c_last < c_first ? c_first : c_last;
            win_w = last_hi - col_lo + 1;
            mid_lo = col_lo;
            mid_hi = last_hi;
        } else {
            col_lo = raw_lo >= 0 ? c_first : 0;
            last_hi = raw_hi <= Hi - 1 ? c_last : Wi - 1;
            win_w = Wi;
            mid_lo = 0;
            mid_hi = Wi - 1;
        }
        for (int32_t row = row_lo; row <= row_hi; ++row) {
            const int32_t c0 = row == row_lo ? col_lo : mid_lo;
            const int32_t c1 = row == row_hi ? last_hi : mid_hi;
            uint32_t dst = window_base + static_cast<uint32_t>((row - row_lo) * win_w + (c0 - col_lo)) * tap_bytes;
            uint32_t s = static_cast<uint32_t>(row) * W + static_cast<uint32_t>(c0);
            for (int32_t c = c0; c <= c1; ++c, ++s) {
                noc_async_read(input.get_noc_addr(s), dst, stick_bytes);
                dst += tap_bytes;
            }
        }
        noc_async_read_barrier();

        // ---- gather_im2col_kblock, per n-block (activation is reused across n) ----
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                const uint32_t k0 = k_block * kblock_elems;
                for (uint32_t t = 0; t < block_m_tiles; ++t) {
                    const uint32_t tile = t0 + t;
                    cb_reserve_back(cb_act_rm, block_k_tiles);
                    const uint32_t dst_base = get_write_ptr(cb_act_rm);
                    if (tile >= t1) {
                        // Ragged block tail: tile-row past the core's range.
                        zero_run(zeros_noc, dst_base, TILE_DIM * im2col_row_bytes);
                    } else {
                        uint32_t pix = tile * TILE_DIM;
                        uint32_t n = pix / hw_out;
                        uint32_t rem = pix - n * hw_out;
                        uint32_t ho = rem / W_out;
                        uint32_t wo = rem - ho * W_out;
                        for (uint32_t r = 0; r < TILE_DIM; ++r, ++pix) {
                            const uint32_t dst = dst_base + r * im2col_row_bytes;
                            if (pix >= M) {
                                zero_run(zeros_noc, dst, im2col_row_bytes);
                                continue;
                            }
                            gather_row(
                                zeros_noc,
                                window_base,
                                row_lo,
                                col_lo,
                                win_w,
                                n,
                                static_cast<int32_t>(ho) * stride - padding,
                                static_cast<int32_t>(wo) * stride - padding,
                                k0,
                                dst);
                            if (++wo == W_out) {
                                wo = 0;
                                if (++ho == H_out) {
                                    ho = 0;
                                    ++n;
                                }
                            }
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_act_rm, block_k_tiles);
                }
            }
        }
    }
}
