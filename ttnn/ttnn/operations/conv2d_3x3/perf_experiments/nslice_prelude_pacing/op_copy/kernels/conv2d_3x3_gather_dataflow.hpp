// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gather reads shared by the reader (NoC0) and the writer (NoC1): transaction-id tagged NoC reads, the im2col
// gathers (row, tap-major, strided window) and the face-direct im2col gather.
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

    // A stateful read run reprogrammed the read command buffer.
    FORCE_INLINE void drop_fixed_state() {
        if constexpr (fixed_reads) {
            fixed_state_bytes = 0;
        }
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

// One im2col tile-row: output pixels [o, min(o + 32, o_end)) x the K-block's columns [col_lo, col_lo + block_k_cols)
// of kernel row kh, into rows_l1. Rows past o_end are not written: every phase downstream is row-independent and the
// writer never stores them. The row gather is a real call (inlined into kernel_main, its per-pixel loop measured
// slower: register pressure on the data-movement RISC); the tap-major body, a handful of runs, measured faster inlined.

// Row gather: one im2col row per output pixel ([32, block_k_cols]), walked in runs of one output image row.
//   - A run whose kernel row lies outside the image is all zero, and its rows are adjacent: one zero fill.
//   - In a d = 1 run with no padded lanes, the pixels whose combined read (the K-block's touched taps, consecutive slab
//     pixels) lies in the image form one window, found once per run. They read a fixed byte count from slab pixels
//     stride apart into im2col rows im2col_row_bytes apart, so the read command buffer keeps its source and size
//     (set_read_state) and each read only rewrites its two addresses (read_with_state).
//   - The other pixels (edges, dilated rows, padded lanes, a K-padding tail zeroed per row) take the per-tap body: one
//     read per in-image tap plus zero-fills for out-of-image taps, padded lanes (unless zeroed once) and kernel-row K
//     padding.
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
    // A row-gather K-block lies inside one kernel row (bk divides its tile count) and only the row's last K-block
    // reaches the K padding, so a combined read is one of two sizes.
    static constexpr uint32_t kkh_cols = (tap_cols + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
    static constexpr uint32_t full_read_bytes = block_k_cols * ELEM_BYTES;
    static constexpr uint32_t tail_read_bytes = (tap_cols + block_k_cols - kkh_cols) * ELEM_BYTES;
    static constexpr bool fixed_capable =
        !lanes_padded && dilation == 1 && full_read_bytes <= NOC_MAX_BURST_SIZE && block_k_cols <= kkh_cols;

    // n fixed-size reads: src_l1, src_l1 + stride * pixel_bytes, ... into dst_l1, dst_l1 + im2col_row_bytes, ...
    template <uint32_t bytes, typename Reads>
    FORCE_INLINE static void fixed_run(Reads& reads, uint32_t src_l1, uint32_t dst_l1, uint32_t n) {
        reads.drop_fixed_state();
        dataflow_kernel_lib::set_read_state<bytes>(reads.noc, src_l1);
        for (uint32_t i = 0; i < n; ++i) {
            dataflow_kernel_lib::read_with_state<bytes>(reads.noc, dst_l1, src_l1);
            reads.count_packets(1);
            src_l1 += stride * pixel_bytes;
            dst_l1 += im2col_row_bytes;
        }
    }

    // n consecutive pixels of one output image row, tap by tap: pixel i reads kernel row h_in at w_in0 + i * stride
    // (slab pixel slab_x0 + i * stride) into row_l1 + i * im2col_row_bytes.
    template <typename Reads>
    FORCE_INLINE static void generic_pixels(
        Reads& reads,
        uint64_t slab_noc,
        uint32_t row_l1,
        uint32_t n,
        bool row_in_image,
        int32_t w_in0,
        int32_t slab_x0,
        uint32_t col_lo) {
        const uint32_t col_hi = col_lo + block_k_cols;
        const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
        const int32_t kw_first = col_lo < tap_cols ? col_lo / c_in_p : 3;
        const int32_t kw_last = data_hi > col_lo ? (data_hi - 1) / c_in_p : -1;
        for (uint32_t i = 0; i < n; ++i, row_l1 += im2col_row_bytes, w_in0 += stride, slab_x0 += stride) {
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
        }
    }

    template <typename Reads>
    __attribute__((noinline)) static void tile_row(
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
        // The fixed window needs a combined read and no per-row K-padding zero fill between two reads.
        const bool fixed_ok = kw_first <= kw_last && (zero_im2col_once || col_hi <= tap_cols);
        // Output columns wo whose combined read lies in the image row: [wo_lo, wo_hi) (d = 1).
        const int32_t left = padding - kw_first;
        const int32_t wo_lo = left <= 0 ? 0 : (left + stride - 1) / stride;
        const int32_t right = input_width - 1 + padding - kw_last;
        const int32_t wo_hi = right < 0 ? 0 : right / stride + 1;
        const uint32_t slab_l1 = static_cast<uint32_t>(slab_noc);
        const uint32_t row_end = o + TILE_DIM < o_end ? o + TILE_DIM : o_end;
        int32_t ho = o / output_width;
        int32_t wo = o % output_width;
        uint32_t row_l1 = rows_l1;
        for (uint32_t p = o; p < row_end;) {
            const int32_t run = row_end - p < output_width - wo ? row_end - p : output_width - wo;
            const int32_t h_in = ho * stride + kh * dilation - padding;
            const bool row_in_image = h_in >= 0 && h_in < input_height;
            const int32_t w_in0 = wo * stride - padding;
            const int32_t slab_x0 = slab_x_base_kh + h_in * input_width + w_in0;
            if (!row_in_image) {
                // Every column of every row of the run is zero.
                reads.zero_fill(row_l1, run * im2col_row_bytes);
            } else {
                // Pixels [i_lo, i_hi) of the run take the fixed window, the rest the per-tap body.
                int32_t i_lo = 0;
                int32_t i_hi = 0;
                if constexpr (fixed_capable) {
                    if (fixed_ok) {
                        i_lo = wo_lo - wo < 0 ? 0 : (wo_lo - wo < run ? wo_lo - wo : run);
                        i_hi = wo_hi - wo < i_lo ? i_lo : (wo_hi - wo < run ? wo_hi - wo : run);
                    }
                }
                if (i_hi == i_lo) {
                    generic_pixels(reads, slab_noc, row_l1, run, true, w_in0, slab_x0, col_lo);
                } else {
                    if (i_lo > 0) {
                        generic_pixels(reads, slab_noc, row_l1, i_lo, true, w_in0, slab_x0, col_lo);
                    }
                    if constexpr (fixed_capable) {
                        const uint32_t src = slab_l1 + (slab_x0 + i_lo * stride) * pixel_bytes + col_lo * ELEM_BYTES;
                        const uint32_t dst = row_l1 + i_lo * im2col_row_bytes;
                        if constexpr (full_read_bytes == tail_read_bytes) {
                            fixed_run<full_read_bytes>(reads, src, dst, i_hi - i_lo);
                        } else if (col_hi <= tap_cols) {
                            fixed_run<full_read_bytes>(reads, src, dst, i_hi - i_lo);
                        } else {
                            fixed_run<tail_read_bytes>(reads, src, dst, i_hi - i_lo);
                        }
                    }
                    if (run > i_hi) {
                        generic_pixels(
                            reads,
                            slab_noc,
                            row_l1 + i_hi * im2col_row_bytes,
                            run - i_hi,
                            true,
                            w_in0 + i_hi * stride,
                            slab_x0 + i_hi * stride,
                            col_lo);
                    }
                }
            }
            row_l1 += run * im2col_row_bytes;
            p += run;
            wo = 0;
            ++ho;
        }
    }
};

// The core's im2col units (one tile-row of one K-block) in cb_im2col_rm push order. Unit s of M-block m is
// s - m * units_per_block into the block: K-block outer (the K-blocked walk) or tile-row outer (N-slice-outer).
template <bool tile_row_outer, uint32_t block_m_tiles, uint32_t num_k_blocks>
struct GatherUnits {
    static constexpr uint32_t per_block = block_m_tiles * num_k_blocks;
    static FORCE_INLINE uint32_t k_block(uint32_t s) {
        const uint32_t u = s % per_block;
        return tile_row_outer ? u % num_k_blocks : u / block_m_tiles;
    }
    static FORCE_INLINE uint32_t tile_row(uint32_t s) {
        const uint32_t u = s % per_block;
        return tile_row_outer ? u / num_k_blocks : u % block_m_tiles;
    }
};

// Tap-major (stride 1, whole unpadded or zero-lane taps): tap kw of the tile-row is a [32 pixels, c_in_p] block. The
// pixels of one output image row read consecutive slab pixels, so each in-image run is one read; out-of-image pixels
// read a 32-pixel zero block at zero_src_noc, one read per run. skip_kw (>= 0): that tap's block is left unwritten
// (center-tap direct: the reader writes its in0 tiles itself).
template <
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t padding,
    int32_t dilation,
    uint32_t c_in_p,
    uint32_t block_k_tiles,
    int32_t skip_kw = -1>
struct TapMajorGather {
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t pixel_bytes = c_in_p * 2;
    static constexpr uint32_t tap_cols = 3 * c_in_p;

    template <typename Reads>
    FORCE_INLINE static void tile_row(
        Reads& reads,
        uint64_t slab_noc,
        uint64_t zero_src_noc,
        uint32_t rows_l1,
        uint32_t o,
        uint32_t o_end,
        int32_t kh,
        uint32_t col_lo,
        int32_t slab_x_base_kh) {
        const uint32_t col_hi = col_lo + block_k_tiles * TILE_DIM;
        const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
        const int32_t kw_first = col_lo < tap_cols ? col_lo / c_in_p : 3;
        const int32_t kw_last = data_hi > col_lo ? (data_hi - 1) / c_in_p : -1;
        const uint32_t row_end = o + TILE_DIM < o_end ? o + TILE_DIM : o_end;
        for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
            if (kw == skip_kw) {
                continue;
            }
            uint32_t dst = rows_l1 + (kw - kw_first) * TILE_DIM * pixel_bytes;
            int32_t ho = o / output_width;
            int32_t wo = o % output_width;
            for (uint32_t p = o; p < row_end;) {
                const int32_t run = row_end - p < output_width - wo ? row_end - p : output_width - wo;
                const int32_t h_in = ho + kh * dilation - padding;
                const int32_t w_lo = wo - padding + kw * dilation;
                // Pixels [in_lo, in_hi) of the run are inside the image row; the rest are padding.
                int32_t in_lo = w_lo < 0 ? (-w_lo < run ? -w_lo : run) : 0;
                int32_t in_hi = input_width - w_lo < run ? input_width - w_lo : run;
                if (h_in < 0 || h_in >= input_height || in_hi < in_lo) {
                    in_lo = in_hi = run;
                }
                if (in_lo > 0) {
                    reads.read(zero_src_noc, dst, in_lo * pixel_bytes);
                }
                if (in_hi > in_lo) {
                    reads.read(
                        slab_noc + (slab_x_base_kh + h_in * input_width + w_lo + in_lo) * pixel_bytes,
                        dst + in_lo * pixel_bytes,
                        (in_hi - in_lo) * pixel_bytes);
                }
                if (run > in_hi) {
                    reads.read(zero_src_noc, dst + in_hi * pixel_bytes, (run - in_hi) * pixel_bytes);
                }
                dst += run * pixel_bytes;
                p += run;
                wo = 0;
                ++ho;
            }
        }
    }
};

// Tap-direct (N-slice-outer prelude, nslice_prelude_pacing experiment): compute tilizes each tap block of a unit from
// an L1 address the reader publishes per (unit slot, tap). A tap block [32 pixels, c_in_p] is already a contiguous run
// of slab pixels when the tile-row's valid pixels are one output-row run whose tap lies inside the image row: its
// address is the slab run itself (no copy). A tap whose kernel row lies outside the image, or a tile-row wholly past
// o_end, is the 32-pixel zero block. Only the other taps (a run touching the left / right padding, or a tile-row
// straddling two output rows) are gathered into the unit's own slot, as TapMajorGather does. Rows past o_end read
// whatever follows the run: every phase downstream is row-independent and the writer never stores them.
template <
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t padding,
    int32_t dilation,
    uint32_t c_in_p,
    uint32_t block_k_tiles>
struct TapDirect {
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t pixel_bytes = c_in_p * 2;
    static constexpr uint32_t tap_cols = 3 * c_in_p;

    // Returns the number of taps gathered (reads issued) into the slot.
    template <typename Reads>
    FORCE_INLINE static uint32_t tile_row(
        Reads& reads,
        uint32_t slab_l1,
        uint64_t slab_noc,
        uint32_t zero_l1,
        uint64_t zero_src_noc,
        uint32_t rows_l1,
        uint32_t o,
        uint32_t o_end,
        int32_t kh,
        uint32_t col_lo,
        int32_t slab_x_base_kh,
        volatile tt_l1_ptr uint32_t* srcs) {
        const uint32_t col_hi = col_lo + block_k_tiles * TILE_DIM;
        const uint32_t data_hi = col_hi < tap_cols ? col_hi : tap_cols;
        const int32_t kw_first = col_lo / c_in_p;
        const int32_t kw_last = (data_hi - 1) / c_in_p;
        const uint32_t row_end = o + TILE_DIM < o_end ? o + TILE_DIM : o_end;
        uint32_t gathered = 0;
        const int32_t ho0 = o / output_width;
        const int32_t wo0 = o % output_width;
        const int32_t n = static_cast<int32_t>(row_end) - static_cast<int32_t>(o);
        const bool one_run = n > 0 && wo0 + n <= static_cast<int32_t>(output_width);
        const int32_t h_in0 = ho0 + kh * dilation - padding;
        const bool row_out = h_in0 < 0 || h_in0 >= input_height;
        for (int32_t kw = kw_first; kw <= kw_last; ++kw) {
            const uint32_t dst = rows_l1 + (kw - kw_first) * TILE_DIM * pixel_bytes;
            const int32_t w_lo0 = wo0 - padding + kw * dilation;
            uint32_t src = dst;
#if defined(NPP_TD_FORCE_GATHER)
            // diagnosis: every tap gathered into the slot (the baseline's data movement, tilized via the table)
            if (n <= 0) {
                srcs[kw - kw_first] = src;
                continue;
            }
#elif defined(NPP_TD_ALLDIRECT)
            // diagnosis only (WRONG at the image edges): every in-image-row tap tilized from the slab, nothing gathered
            if (n <= 0 || row_out) {
                src = zero_l1;
            } else {
                src = slab_l1 + static_cast<uint32_t>(slab_x_base_kh + h_in0 * input_width + w_lo0) * pixel_bytes;
            }
#ifdef NPP_TD_ALIGN
            if (src != zero_l1) {
                src &= ~static_cast<uint32_t>(NPP_TD_ALIGN - 1);
            }
#endif
#ifdef NPP_TD_SLOTSRC
            src = dst;
            srcs[kw - kw_first] = src;
            continue;
#endif
#ifdef NPP_TD_ZEROSRC
            src = zero_l1;
#endif
#else
            if (n <= 0 || (one_run && row_out)) {
                src = zero_l1;
            } else if (one_run && w_lo0 >= 0 && w_lo0 + n <= input_width) {
                src = slab_l1 + static_cast<uint32_t>(slab_x_base_kh + h_in0 * input_width + w_lo0) * pixel_bytes;
            }
#endif
            srcs[kw - kw_first] = src;
            if (src != dst) {
                continue;
            }
            ++gathered;
            uint32_t d = dst;
            int32_t ho = ho0;
            int32_t wo = wo0;
            for (uint32_t p = o; p < row_end;) {
                const int32_t run = row_end - p < output_width - wo ? row_end - p : output_width - wo;
                const int32_t h_in = ho + kh * dilation - padding;
                const int32_t w_lo = wo - padding + kw * dilation;
                int32_t in_lo = w_lo < 0 ? (-w_lo < run ? -w_lo : run) : 0;
                int32_t in_hi = input_width - w_lo < run ? input_width - w_lo : run;
                if (h_in < 0 || h_in >= input_height || in_hi < in_lo) {
                    in_lo = in_hi = run;
                }
                if (in_lo > 0) {
                    reads.read(zero_src_noc, d, in_lo * pixel_bytes);
                }
                if (in_hi > in_lo) {
                    reads.read(
                        slab_noc + (slab_x_base_kh + h_in * input_width + w_lo + in_lo) * pixel_bytes,
                        d + in_lo * pixel_bytes,
                        (in_hi - in_lo) * pixel_bytes);
                }
                if (run > in_hi) {
                    reads.read(zero_src_noc, d + in_hi * pixel_bytes, (run - in_hi) * pixel_bytes);
                }
                d += run * pixel_bytes;
                p += run;
                wo = 0;
                ++ho;
            }
        }
        return gathered;
    }
};

// Strided window (stride 2, d 1, C_in_p whole tiles, one K-block per kernel row): im2col row i of a tile-row is the 3
// consecutive input pixels from 2 * wo_i - padding, and inside one output-row run row i + 1 starts two pixels after row
// i. So the run's rows are overlapping windows of one input run: pixel slot j of the unit holds input pixel
// 2 * wo_first - padding + j, and compute tilizes row i from slot 2 * i (row stride 2 pixels, row width 3 pixels), in
// place. One read per output-row run (plus zero fills at the image edges) replaces one read per output pixel. Two runs
// of one tile-row share one slot, which the host guarantees is zero for both (left padding / out-of-image right tap).
// The unit's last 16 B carry the L1 address compute tilizes it from: the unit itself, or (slab-direct) the slab run.
template <uint32_t output_width, int32_t input_height, int32_t input_width, int32_t padding, uint32_t c_in_p>
struct WindowGather {
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t pixel_bytes = c_in_p * 2;
    // 32 rows two pixels apart plus the last row's third pixel.
    static constexpr uint32_t unit_bytes = (2 * TILE_DIM + 1) * pixel_bytes;

    // Slab-direct: a full tile-row inside one output row whose whole window lies inside the image row is already a
    // contiguous run of slab pixels, so compute tilizes it from the slab; returns its L1 address, or 0 (copy it).
    static uint32_t direct_source(uint32_t slab_l1, uint32_t o, uint32_t o_end, int32_t kh, int32_t slab_x_base_kh) {
        if (o + TILE_DIM > o_end) {
            return 0;
        }
        const int32_t ho = o / output_width;
        const int32_t wo = o % output_width;
        const int32_t h_in = ho * 2 + kh - padding;
        const int32_t w0 = wo * 2 - padding;
        if (wo + static_cast<int32_t>(TILE_DIM) > static_cast<int32_t>(output_width) || h_in < 0 ||
            h_in >= input_height || w0 < 0 || w0 + 2 * static_cast<int32_t>(TILE_DIM) >= input_width) {
            return 0;
        }
        return slab_l1 + (slab_x_base_kh + h_in * input_width + w0) * pixel_bytes;
    }

    template <typename Reads>
    __attribute__((noinline)) static void tile_row(
        Reads& reads,
        uint64_t slab_noc,
        uint32_t dst_l1,
        uint32_t o,
        uint32_t o_end,
        int32_t kh,
        int32_t slab_x_base_kh) {
        const uint32_t row_end = o + TILE_DIM < o_end ? o + TILE_DIM : o_end;
        int32_t ho = o / output_width;
        int32_t wo = o % output_width;
        uint32_t slot_l1 = dst_l1;
        for (uint32_t p = o; p < row_end;) {
            const int32_t run = row_end - p < output_width - wo ? row_end - p : output_width - wo;
            const int32_t h_in = ho * 2 + kh - padding;
            const int32_t n_px = 2 * run + 1;
            if (h_in < 0 || h_in >= input_height) {
                reads.zero_fill(slot_l1, n_px * pixel_bytes);
            } else {
                // Slots [lo, hi) of the run are inside the image row.
                const int32_t w0 = wo * 2 - padding;
                const int32_t lo = w0 < 0 ? -w0 : 0;
                int32_t hi = input_width - w0 < n_px ? input_width - w0 : n_px;
                hi = hi > lo ? hi : lo;
                if (lo > 0) {
                    reads.zero_fill(slot_l1, lo * pixel_bytes);
                }
                if (hi > lo) {
                    reads.read(
                        slab_noc + (slab_x_base_kh + h_in * input_width + w0 + lo) * pixel_bytes,
                        slot_l1 + lo * pixel_bytes,
                        (hi - lo) * pixel_bytes);
                }
                if (n_px > hi) {
                    reads.zero_fill(slot_l1 + hi * pixel_bytes, (n_px - hi) * pixel_bytes);
                }
            }
            slot_l1 += 2 * run * pixel_bytes;
            p += run;
            wo = 0;
            ++ho;
        }
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
