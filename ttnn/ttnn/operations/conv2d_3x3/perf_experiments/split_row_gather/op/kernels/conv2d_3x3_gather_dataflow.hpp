// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gather reads shared by the reader (NoC0) and the writer (NoC1): transaction-id tagged NoC reads, and the
// face-direct im2col gather.
//
// Face-direct (stride * C_in_p == 16, bf16 activation, no padded lanes): one output pixel steps through the slab
// by exactly one 16-element tile-face row, so the 16 im2col rows of a face are one contiguous run of slab pixels,
// and the gatherer writes the in0 tiles (bf16, face layout) itself, one read per face. A face that holds the
// kernel row's last tap also covers the next pixel, which lands in K-padding lanes against zero weight rows; it is
// a real input pixel (read only when inside the image), not stale L1. A face made only of K padding is never
// written: the reader zeroes it once, and every K-block starts at the CB base.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

namespace conv2d_3x3 {

// The NIU counts outstanding reads per transaction id in a counter capped at NOC_MAX_TRANSACTION_ID_COUNT; long
// batches are throttled to stay well below it.
constexpr uint32_t TRID_CHECK_EVERY_PACKETS = 64;
constexpr uint32_t TRID_MIN_FREE = (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2;

// fixed_reads: the kernel also issues read_local_fixed, whose command-buffer state every other read clobbers.
template <bool fixed_reads>
struct TaggedReads {
    Noc noc;
    uint32_t trid = 0;
    uint32_t packets_since_check = 0;
    // Bytes of the self-aimed fixed-size state the read command buffer holds, 0 if another read reprogrammed it.
    uint32_t fixed_state_bytes = 0;
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE, noc.get_noc_id());

    FORCE_INLINE void begin(uint32_t new_trid) {
        trid = new_trid;
        packets_since_check = 0;
        dataflow_kernel_lib::set_read_trid(noc, trid);
    }

    FORCE_INLINE void count_packets(uint32_t packets) {
        packets_since_check += packets;
        if (packets_since_check >= TRID_CHECK_EVERY_PACKETS) {
            packets_since_check = 0;
            while (noc_available_transactions(noc.get_noc_id(), trid) < TRID_MIN_FREE) {
            }
        }
    }

    FORCE_INLINE void read(uint64_t src_noc_addr, uint32_t dst_l1_addr, uint32_t bytes) {
        if constexpr (fixed_reads) {
            fixed_state_bytes = 0;
        }
        noc_async_read(src_noc_addr, dst_l1_addr, bytes);
        count_packets((bytes + NOC_MAX_BURST_SIZE - 1) / NOC_MAX_BURST_SIZE);
    }

    // One-packet L1 -> L1 copy on this core; consecutive calls of one size only rewrite the two addresses.
    template <uint32_t bytes>
    FORCE_INLINE void read_local_fixed(uint32_t src_l1_addr, uint32_t dst_l1_addr) {
        static_assert(fixed_reads);
        if (fixed_state_bytes != bytes) {
            dataflow_kernel_lib::set_read_state<bytes>(noc, src_l1_addr);
            fixed_state_bytes = bytes;
        }
        dataflow_kernel_lib::read_with_state<bytes>(noc, dst_l1_addr, src_l1_addr);
        count_packets(1);
    }

    // Zeros from the hardware zero region, which is MEM_ZEROS_SIZE bytes long.
    FORCE_INLINE void zero_fill(uint32_t dst_l1_addr, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
            read(zeros_noc, dst_l1_addr, chunk);
            dst_l1_addr += chunk;
            bytes -= chunk;
        }
    }

    FORCE_INLINE void wait(uint32_t which) { dataflow_kernel_lib::async_read_barrier_with_trid(noc, which); }
};

// Row gather (not tap-major, not face-direct): one im2col row per output pixel, [32, block_k_cols] per tile-row.
// Shared by the reader (NoC0) and, with the split row gather, the writer (NoC1): both gather whole tile-rows of
// the same K-block into their own reserved slots of cb_im2col_rm.
template <
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    uint32_t c_in,
    uint32_t c_in_p,
    uint32_t block_k_tiles,
    bool zero_im2col_once>
struct RowGather {
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t ELEM_BYTES = 2;
    static constexpr uint32_t block_k_cols = block_k_tiles * TILE_DIM;
    static constexpr uint32_t im2col_row_bytes = block_k_cols * ELEM_BYTES;
    static constexpr uint32_t tap_cols = 3 * c_in_p;
    static constexpr uint32_t pixel_bytes = c_in_p * ELEM_BYTES;
    static constexpr bool lanes_padded = c_in < c_in_p;

    // Output pixels [o, min(o + 32, o_end)) x K-block columns [col_lo, col_lo + block_k_cols) of kernel row kh, one
    // im2col row per pixel from rows_l1. Rows past o_end are not written (stale; never stored downstream).
    template <typename Reads>
#ifdef ROW_GATHER_NOINLINE
    __attribute__((noinline))
#else
    FORCE_INLINE
#endif
    static void
    tile_row(
        Reads& reads,
        uint64_t slab_noc,
        uint32_t rows_l1,
        uint32_t o,
        uint32_t o_end,
        int32_t kh,
        uint32_t col_lo,
        int32_t slab_x_base_kh) {
        const uint32_t col_hi = col_lo + block_k_cols;
        const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
        const int32_t kw_first = col_lo < tap_cols ? col_lo / c_in_p : 3;
        const int32_t kw_last = data_hi > col_lo ? (data_hi - 1) / c_in_p : -1;
        int32_t ho = o / output_width;
        int32_t wo = o % output_width;
        for (uint32_t r = 0; r < TILE_DIM && o < o_end; ++r, ++o) {
            const uint32_t row_l1 = rows_l1 + r * im2col_row_bytes;
            const int32_t h_in = ho * stride + kh * dilation - padding;
            const int32_t w_in0 = wo * stride - padding;
            const bool row_in_image = h_in >= 0 && h_in < input_height;
            const int32_t slab_x0 = slab_x_base_kh + h_in * input_width + w_in0;

            const bool combined = !lanes_padded && dilation == 1 && row_in_image && w_in0 + kw_first >= 0 &&
                                  w_in0 + kw_last < input_width && kw_first <= kw_last;
            if (combined) {
                // d = 1, no lanes: the touched taps are consecutive slab pixels, so one read covers them.
                reads.read(
                    slab_noc + slab_x0 * pixel_bytes + col_lo * ELEM_BYTES, row_l1, (data_hi - col_lo) * ELEM_BYTES);
            } else {
                for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
                    const uint32_t tap_lo = kw * c_in_p;
                    const uint32_t seg_lo = col_lo > tap_lo ? col_lo : tap_lo;
                    const uint32_t seg_hi = data_hi < tap_lo + c_in_p ? data_hi : tap_lo + c_in_p;
                    const uint32_t real_hi = seg_hi < tap_lo + c_in ? seg_hi : tap_lo + c_in;
                    const uint32_t dst = row_l1 + (seg_lo - col_lo) * ELEM_BYTES;
                    const int32_t w_in = w_in0 + kw * dilation;
                    if (row_in_image && w_in >= 0 && w_in < input_width) {
                        const int32_t slab_x = slab_x0 + kw * dilation;
                        if (real_hi > seg_lo) {
                            reads.read(
                                slab_noc + slab_x * pixel_bytes + (seg_lo - tap_lo) * ELEM_BYTES,
                                dst,
                                (real_hi - seg_lo) * ELEM_BYTES);
                        }
                        if constexpr (lanes_padded && !zero_im2col_once) {
                            const uint32_t lane_lo = real_hi > seg_lo ? real_hi : seg_lo;
                            if (seg_hi > lane_lo) {
                                reads.zero_fill(
                                    row_l1 + (lane_lo - col_lo) * ELEM_BYTES, (seg_hi - lane_lo) * ELEM_BYTES);
                            }
                        }
                    } else {
                        reads.zero_fill(dst, (seg_hi - seg_lo) * ELEM_BYTES);
                    }
                }
            }
            if constexpr (!zero_im2col_once) {
                if (col_hi > tap_cols) {
                    const uint32_t pad_lo = col_lo > tap_cols ? col_lo : tap_cols;
                    reads.zero_fill(row_l1 + (pad_lo - col_lo) * ELEM_BYTES, (col_hi - pad_lo) * ELEM_BYTES);
                }
            }

            if (++wo == static_cast<int32_t>(output_width)) {
                wo = 0;
                ++ho;
            }
        }
    }
};

// Split row gather: tile-row s of the kernel's global sequence (M-block, K-block, tile-row) belongs to the writer
// when s % period >= period - writer_rows. writer_rows_before(s) counts the writer's rows in [0, s).
template <uint32_t period, uint32_t writer_rows>
struct RowSplit {
    static_assert(period > 0 && writer_rows < period);
    static FORCE_INLINE bool writer_owns(uint32_t s) { return writer_rows > 0 && (s % period) >= period - writer_rows; }
    static FORCE_INLINE uint32_t writer_rows_before(uint32_t s) {
        const uint32_t phase = s % period;
        return (s / period) * writer_rows + (phase > period - writer_rows ? phase - (period - writer_rows) : 0);
    }
};

template <
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    uint32_t c_in_p,
    uint32_t tensor_kkh_tiles,
    uint32_t block_k_tiles,
    int32_t band_last_tap>
struct FaceGather {
    static constexpr uint32_t FACE_DIM = 16;
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t face_row_bytes = FACE_DIM * 2;
    static constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
    static constexpr uint32_t tile_bytes = 4 * face_bytes;
    static constexpr int32_t face_taps = c_in_p < FACE_DIM ? FACE_DIM / c_in_p : 1;
    static constexpr uint32_t pixel_bytes = c_in_p * 2;
    static constexpr uint32_t kh_per_k_block = block_k_tiles / tensor_kkh_tiles;

    // Output pixels [o_lo, o_hi) x taps [tap0, tap0 + face_taps) of kernel row kh into the face at dst, one face
    // row per pixel. Within one output image row the pixels whose taps all lie in the image are consecutive and
    // read consecutive slab bytes; the rest copy or zero tap by tap.
    template <typename Reads>
    static void face(
        Reads& reads,
        uint64_t slab_noc,
        uint32_t dst,
        uint32_t o_lo,
        uint32_t o_hi,
        int32_t kh,
        int32_t tap0,
        int32_t slab_x_base_kh) {
        int32_t ho = o_lo / output_width;
        int32_t wo = o_lo % output_width;
        for (uint32_t p = o_lo; p < o_hi;) {
            const int32_t run = o_hi - p < output_width - wo ? o_hi - p : output_width - wo;
            const int32_t h_in = ho * stride + kh * dilation - padding;
            if (h_in < 0 || h_in >= input_height) {
                reads.zero_fill(dst, run * face_row_bytes);
            } else {
                // Pixel i of the run reads its first tap at w0 + i * stride, its last at w0 + i * stride + last.
                const int32_t w0 = wo * stride - padding + tap0 * dilation;
                constexpr int32_t last = (face_taps - 1) * dilation;
                int32_t in_lo = w0 >= 0 ? 0 : (stride - 1 - w0) / stride;
                const int32_t room = input_width - 1 - last - w0;
                int32_t in_hi = room < 0 ? 0 : room / stride + 1;
                in_lo = in_lo < run ? in_lo : run;
                in_hi = in_hi < run ? in_hi : run;
                in_hi = in_hi > in_lo ? in_hi : in_lo;
                // Slab pixel of (h_in, 0); it may precede the slab, but every pixel read is inside it.
                const int32_t row_x = slab_x_base_kh + h_in * input_width;
                if (in_hi > in_lo) {
                    reads.read(
                        slab_noc + (row_x + w0 + in_lo * stride) * pixel_bytes,
                        dst + in_lo * face_row_bytes,
                        (in_hi - in_lo) * face_row_bytes);
                }
                for (int32_t i = 0; i < run; ++i) {
                    if (i == in_lo) {
                        i = in_hi;
                        if (i == run) {
                            break;
                        }
                    }
                    for (int32_t tap = 0; tap < face_taps; ++tap) {
                        const int32_t w = w0 + i * stride + tap * dilation;
                        const uint32_t tap_dst = dst + i * face_row_bytes + tap * pixel_bytes;
                        if (w >= 0 && w < input_width) {
                            reads.read(slab_noc + (row_x + w) * pixel_bytes, tap_dst, pixel_bytes);
                        } else {
                            reads.zero_fill(tap_dst, pixel_bytes);
                        }
                    }
                }
            }
            dst += run * face_row_bytes;
            p += run;
            wo = 0;
            ++ho;
        }
    }

    // Tile-rows [row_begin, row_end) of one K-block (kernel rows kh .. kh + kh_per_k_block - 1) of the M-block of
    // output pixels [o_first, o_end), into the K-block's tiles at tiles_l1. Rows past o_end stay stale: every
    // downstream phase is row-independent and those rows are never stored.
    template <typename Reads>
    static void tile_rows(
        Reads& reads,
        uint32_t tiles_l1,
        uint32_t row_begin,
        uint32_t row_end,
        uint32_t o_first,
        uint32_t o_end,
        int32_t kh,
        uint32_t slab_l1,
        const int32_t (&slab_x_base)[3]) {
        const uint64_t slab_noc = get_noc_addr(slab_l1, reads.noc.get_noc_id());
        for (uint32_t tile_row = row_begin; tile_row < row_end; ++tile_row) {
            const uint32_t tile_row_l1 = tiles_l1 + tile_row * block_k_tiles * tile_bytes;
            for (uint32_t half = 0; half < 2; ++half) {
                // Face rows: output pixels [f_lo, f_hi), rows half * 16.. of every tile of the tile-row.
                const uint32_t f_lo = o_first + tile_row * TILE_DIM + half * FACE_DIM;
                if (f_lo >= o_end) {
                    return;
                }
                const uint32_t f_hi = f_lo + FACE_DIM < o_end ? f_lo + FACE_DIM : o_end;
                const uint32_t faces_l1 = tile_row_l1 + half * 2 * face_bytes;
                const int32_t ho = f_lo / output_width;
                const int32_t wo = f_lo % output_width;
                const int32_t h_in0 = ho * stride - padding;
                const int32_t w_in0 = wo * stride - padding;
                // 16 pixels of one output row whose every tap read lies inside the image row: each face is one
                // run, or all zero when its kernel row is outside the image.
                const bool full_row = f_hi - f_lo == FACE_DIM && wo + FACE_DIM <= output_width && w_in0 >= 0 &&
                                      w_in0 + (FACE_DIM - 1) * stride + band_last_tap * dilation < input_width;
                for (uint32_t khb = 0; khb < kh_per_k_block; ++khb) {
                    const int32_t row_kh = kh + static_cast<int32_t>(khb);
                    const int32_t h_in = h_in0 + row_kh * dilation;
                    const bool row_in_image = h_in >= 0 && h_in < input_height;
                    const uint32_t run_l1 = slab_l1 + (slab_x_base[row_kh] + h_in * input_width + w_in0) * pixel_bytes;
                    for (uint32_t t = 0; t < tensor_kkh_tiles; ++t) {
                        const uint32_t tile_l1 = faces_l1 + (khb * tensor_kkh_tiles + t) * tile_bytes;
                        for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                            const int32_t tap0 = (t * 2 + face_col) * face_taps;
                            if (tap0 >= 3) {
                                continue;  // K padding only
                            }
                            const uint32_t face_l1 = tile_l1 + face_col * face_bytes;
                            if (full_row && row_in_image) {
                                reads.template read_local_fixed<face_bytes>(
                                    run_l1 + tap0 * dilation * pixel_bytes, face_l1);
                            } else if (full_row) {
                                reads.zero_fill(face_l1, face_bytes);
                            } else {
                                face(reads, slab_noc, face_l1, f_lo, f_hi, row_kh, tap0, slab_x_base[row_kh]);
                            }
                        }
                    }
                }
            }
        }
    }
};

}  // namespace conv2d_3x3
