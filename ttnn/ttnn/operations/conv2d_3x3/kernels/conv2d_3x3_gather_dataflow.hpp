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

struct TaggedReads {
    Noc noc;
    uint32_t trid = 0;
    uint32_t packets_since_check = 0;
    // The read command buffer holds the self-aimed fixed-size state of read_local_fixed; any other read
    // reprograms it.
    uint32_t fixed_state_bytes = 0;

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
        fixed_state_bytes = 0;
        noc_async_read(src_noc_addr, dst_l1_addr, bytes);
        count_packets((bytes + NOC_MAX_BURST_SIZE - 1) / NOC_MAX_BURST_SIZE);
    }

    // One-packet L1 -> L1 copy on this core; consecutive calls of one size only rewrite the two addresses.
    template <uint32_t bytes>
    FORCE_INLINE void read_local_fixed(uint32_t src_l1_addr, uint32_t dst_l1_addr) {
        if (fixed_state_bytes != bytes) {
            dataflow_kernel_lib::set_read_state<bytes>(noc, src_l1_addr);
            fixed_state_bytes = bytes;
        }
        dataflow_kernel_lib::read_with_state<bytes>(noc, dst_l1_addr, src_l1_addr);
        count_packets(1);
    }

    // Zeros from the hardware zero region, which is MEM_ZEROS_SIZE bytes long.
    FORCE_INLINE void zero_fill(uint32_t dst_l1_addr, uint32_t bytes) {
        const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE, noc.get_noc_id());
        while (bytes > 0) {
            const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
            read(zeros_noc, dst_l1_addr, chunk);
            dst_l1_addr += chunk;
            bytes -= chunk;
        }
    }

    FORCE_INLINE void wait(uint32_t which) { dataflow_kernel_lib::async_read_barrier_with_trid(noc, which); }
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
    static void face(
        TaggedReads& reads,
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
    static void tile_rows(
        TaggedReads& reads,
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
