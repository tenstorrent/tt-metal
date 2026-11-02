// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc im2col gather — shared by the reader (NCRISC) and, for the split gather (R3), the
// writer (BRISC). Include after defining GATHER_CT_BASE (index of this instance's first gather
// compile-time arg); everything lives in namespace conv_gather.
//
// conv2d_nhwc reader (NCRISC) — regimes R0 and R4/R5.
//
// Block operations owned here:
//   fetch_window_block   : R0: the m-block's input footprint from DRAM into cb_act_window
//                          (reader-private scratch), one read barrier per m-block. Columns are
//                          clipped to what the block reads (R10 column clip): a 2-D clip for a
//                          block inside one output row, a trimmed contiguous range otherwise.
//                          R4/R5 (sharded): the resident shard (cb_in_shard, zero-copy) is read in
//                          place; only the halo sticks of the window outside it are fetched into
//                          cb_act_window, one NoC read per owning same-column shard run.
//   gather_im2col_kblock : per K-block, per tile-row: 32 im2col rows x (bk*32) elements into
//                          cb_act_rm (bk tile-sized pages of RM rows), as L1->L1 NoC copies of
//                          each (pixel, kernel-row band) in-image run plus zero runs from
//                          MEM_ZEROS_BASE for padding. One read barrier per tile-row.
//
// No helper expresses a sliding-window gather with per-pixel padding runs
// (local_copy_helpers_dataflow copies one contiguous region per call), so this is the
// block operation built for it (op_design.md "Helpers considered and rejected").
// Phase 0 assumes dilation == 1, so a kernel-row band is one contiguous window run.
// R4/R5: tap_width / band_tiles are the channel slice's; the gather produces only this core's
// slice of K (num_k_blocks = the slice's K-blocks), which compute tilizes and the writer multicasts.
// R11 depthwise: per n-block one K-block of taps x bn*32 channels (per-tap tiles after tilize);
// R0 windows hold only the core's channel tiles, sharded cores read their own slice (no exchange).

#pragma once

#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv_gather {

constexpr uint32_t cb_act_window = get_compile_time_arg_val(GATHER_CT_BASE + 0);
constexpr uint32_t cb_act_rm = get_compile_time_arg_val(GATHER_CT_BASE + 1);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 2);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 3);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(GATHER_CT_BASE + 4);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(GATHER_CT_BASE + 5);
constexpr uint32_t band_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 6);
constexpr uint32_t tap_width = get_compile_time_arg_val(GATHER_CT_BASE + 7);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(GATHER_CT_BASE + 8);
constexpr uint32_t stick_bytes = get_compile_time_arg_val(GATHER_CT_BASE + 9);
constexpr uint32_t kH = get_compile_time_arg_val(GATHER_CT_BASE + 10);
constexpr uint32_t kW = get_compile_time_arg_val(GATHER_CT_BASE + 11);
constexpr uint32_t H = get_compile_time_arg_val(GATHER_CT_BASE + 12);
constexpr uint32_t W = get_compile_time_arg_val(GATHER_CT_BASE + 13);
constexpr uint32_t H_out = get_compile_time_arg_val(GATHER_CT_BASE + 14);
constexpr uint32_t W_out = get_compile_time_arg_val(GATHER_CT_BASE + 15);
constexpr uint32_t M = get_compile_time_arg_val(GATHER_CT_BASE + 16);
constexpr bool k_exchange = get_compile_time_arg_val(GATHER_CT_BASE + 17) != 0;  // R4/R5: shard-resident input
constexpr uint32_t cb_in_shard = get_compile_time_arg_val(GATHER_CT_BASE + 18);
constexpr uint32_t shard_h = get_compile_time_arg_val(GATHER_CT_BASE + 19);  // sticks per input shard (grid row)
constexpr uint32_t num_col_shards =
    get_compile_time_arg_val(GATHER_CT_BASE + 20);  // input shards in this core's column
// R11 depthwise: the im2col row of n-block nb is [tap][bn*32 channels at channel nb*bn*32 of the
// window / shard stick] (one K-block per n-block); dense: [kh][kw x tap_width] K-blocks.
constexpr bool depthwise = get_compile_time_arg_val(GATHER_CT_BASE + 21) != 0;
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 22);
// Tap-major K-blocks (stride 1, dilation 1, whole-tile taps): a K-block's RM data is, per tap, one
// contiguous 32-pixel x tap_width block (compute tilizes one tap wide), so the 32 pixels of a tap
// that lie in one output row are ONE contiguous run of input sticks: one read per run instead of one
// read per pixel. Off: [pixel][K-block columns] rows (band runs, one read per pixel per band).
constexpr bool tap_major = get_compile_time_arg_val(GATHER_CT_BASE + 23) != 0;
// Split gather (R3): m-blocks alternate between the reader (NCRISC, even m-blocks, cb_act_window /
// cb_act_rm) and the writer (BRISC, odd m-blocks, its own window / rm CBs: this header's CB ids are
// the instance's). 1 = the reader gathers every m-block.
constexpr uint32_t num_gather_roles = get_compile_time_arg_val(GATHER_CT_BASE + 24);
// TILE input (techniques 1 / 2): the gather reads an RM window that compute untilized from input
// tile-rows into cb_act_window (one contiguous run of whole tile-rows: stick s at
// (s - first_tile_row * 32) * tap_bytes). The reader supplies the tiles in cb_win_tiles:
//   sharded : once per kernel — pushes the resident own shard (zero-copy, compute untilizes it in
//             place) and fetches the halo tile-rows from their owning shards, one read per owner run.
//   DRAM    : per m-block — the window's tile-rows (win_tile_rows nominal) via the TensorAccessor.
constexpr bool tile_in = get_compile_time_arg_val(GATHER_CT_BASE + 25) != 0;
constexpr uint32_t cb_win_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 26);
constexpr uint32_t win_c_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 27);    // tiles per tile-row
constexpr uint32_t win_tile_rows = get_compile_time_arg_val(GATHER_CT_BASE + 28);  // DRAM: per m-block
// R9 (technique 3, 1x1 stride-1 TILE): the shard's tiles ARE in0 — the reader only publishes the
// resident shard (consumer: compute's matmul on HEIGHT, the writer's act multicast on BLOCK / WIDTH).
constexpr bool direct_in0 = get_compile_time_arg_val(GATHER_CT_BASE + 29) != 0;
// Dilation (Refinement 8): kernel tap (kh, kw) reads input (h_base + kh*DIL, w_base + kw*DIL). A
// kernel-row band is then kW runs of tap_width at stick stride DIL (one read per tap, not per band).
constexpr uint32_t DIL = get_compile_time_arg_val(GATHER_CT_BASE + 30);
// R0 unaligned input sticks (Refinement 8): a DRAM stick is read into a staging slot at the DRAM read
// alignment (stage_stride bytes, pre-zeroed once) and then copied tap_bytes wide into the window, so
// the window's pad lanes [stick_bytes, tap_bytes) stay zero (a zero weight times a stale NaN is NaN).
// 0 = direct DRAM -> window reads (tap_bytes a multiple of the DRAM alignment).
constexpr uint32_t stage_stride = get_compile_time_arg_val(GATHER_CT_BASE + 31);
constexpr uint32_t cb_stage = get_compile_time_arg_val(GATHER_CT_BASE + 32);
constexpr uint32_t stage_row_sticks = get_compile_time_arg_val(GATHER_CT_BASE + 33);  // per staging half
// R0: window sticks pre-zeroed once (stick_bytes < tap_bytes: pad lanes are never written).
constexpr uint32_t window_zero_bytes = get_compile_time_arg_val(GATHER_CT_BASE + 34);
// Byte stride of a source stick (window slot / shard stick / untilized TILE window stick). Equals
// tap_bytes unless the source sticks are wider than the K tap (a HEIGHT shard or a TILE window wider
// than the granule-padded C_in): then every tap is its own read, as with dilation.
constexpr uint32_t src_bytes = get_compile_time_arg_val(GATHER_CT_BASE + 35);
constexpr uint32_t in_row_tiles = get_compile_time_arg_val(GATHER_CT_BASE + 36);  // TILE input: tiles per tile-row
constexpr auto input_args = TensorAccessorArgs<GATHER_CT_BASE + 37>();
// The gather reads a contiguous stick source (resident shard + halo, or the untilized TILE window).
constexpr bool contiguous_src = k_exchange || tile_in;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t tap_bytes = tap_width * elem_bytes;
// A kernel-row band's kW taps are one contiguous source run (no dilation, sticks exactly one tap).
constexpr bool taps_contiguous = DIL == 1 && src_bytes == tap_bytes;
constexpr uint32_t band_elems = band_tiles * TILE_DIM;
constexpr uint32_t kblock_elems = block_k_tiles * TILE_DIM;
constexpr uint32_t im2col_row_bytes = kblock_elems * elem_bytes;
// Depthwise: bytes of one tap's channel run, and whether it spans the whole stick (then the kW
// taps of a kernel row are one contiguous run of sticks: one read per kernel row).
constexpr uint32_t dw_tap_bytes = block_n_tiles * TILE_DIM * elem_bytes;
constexpr bool dw_full_stick = dw_tap_bytes == tap_bytes;
// Depthwise kernel-row runs: the kW taps' sticks are contiguous (whole stick, no dilation).
constexpr bool dw_row_runs = dw_full_stick && taps_contiguous;
constexpr int32_t eff_kH = static_cast<int32_t>(DIL * (kH - 1) + 1);
constexpr int32_t eff_kW = static_cast<int32_t>(DIL * (kW - 1) + 1);

// In-image kernel columns of a pixel at w_base: kw in [kw_lo, kw_hi) (w_base + kw*DIL in [0, W)).
FORCE_INLINE void in_image_taps(int32_t w_base, int32_t& kw_lo, int32_t& kw_hi) {
    constexpr int32_t d = static_cast<int32_t>(DIL);
    kw_lo = w_base < 0 ? (-w_base + d - 1) / d : 0;
    const int32_t last = static_cast<int32_t>(W) - 1 - w_base;
    kw_hi = last < 0 ? 0 : last / d + 1;
    if (kw_hi > static_cast<int32_t>(kW)) {
        kw_hi = kW;
    }
    if (kw_hi < kw_lo) {
        kw_hi = kw_lo;
    }
}

FORCE_INLINE int32_t clamp_i(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Every im2col gather source (window / halo scratch, resident shard, MEM_ZEROS) is in this
// core's own L1, so the gather issues stateful reads against the local NoC xy, set once per
// m-block after the (remote) window / halo fetch: a per-read cost of a few register writes.
FORCE_INLINE void set_local_read_state() { noc_async_read_set_state(get_noc_addr(0)); }
FORCE_INLINE void local_read(uint32_t src, uint32_t dst, uint32_t bytes) { noc_async_read_with_state(src, dst, bytes); }

FORCE_INLINE void zero_run(uint64_t, uint32_t dst, uint32_t bytes) {
    while (bytes > 0) {
        const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        local_read(MEM_ZEROS_BASE, dst, chunk);
        dst += chunk;
        bytes -= chunk;
    }
}

// R0 source: the clipped DRAM window copy. Stick (row, w) -> (row - row0) / row_step * win_w + (w - col_lo).
// row_step = DIL for a dilated block inside one output row (only its kH kernel rows are fetched).
struct WindowSource {
    uint32_t base;
    int32_t row0;
    int32_t col_lo;
    int32_t win_w;
    int32_t row_step;
    FORCE_INLINE int32_t window_row(int32_t row) const {
        if constexpr (DIL == 1) {
            return row - row0;
        } else {
            return (row - row0) / row_step;
        }
    }
    // `bytes` starting at channel byte `chan_byte` of stick (row, w); contiguous across sticks.
    FORCE_INLINE void read(int32_t row, int32_t w, uint32_t chan_byte, uint32_t dst, uint32_t bytes) const {
        const int32_t stick = window_row(row) * win_w + (w - col_lo);
        local_read(base + static_cast<uint32_t>(stick) * src_bytes + chan_byte, dst, bytes);
    }
    // Address of stick (row, w_first) + chan_byte; a run of sticks [w_first, w_end) of one input
    // row is affine in w inside the window copy, so this always succeeds.
    FORCE_INLINE bool run_addr(int32_t row, int32_t w_first, int32_t, uint32_t chan_byte, uint32_t& a) const {
        a = base + static_cast<uint32_t>(window_row(row) * win_w + (w_first - col_lo)) * src_bytes + chan_byte;
        return true;
    }
};

// R4/R5 source: flattened stick s lives in the resident shard [own_lo, own_hi) (read in place),
// else in the halo scratch: [win_lo, own_lo) first, then [after_lo, ...).
struct ShardSource {
    uint32_t shard_base;
    uint32_t halo_base;
    int32_t own_lo;
    int32_t own_hi;
    int32_t win_lo;
    int32_t after_lo;
    int32_t n_before;
    FORCE_INLINE uint32_t addr(int32_t s) const {
        int32_t idx;
        uint32_t base;
        if (s < own_lo) {
            base = halo_base;
            idx = s - win_lo;
        } else if (s < own_hi) {
            base = shard_base;
            idx = s - own_lo;
        } else {
            base = halo_base;
            idx = n_before + s - after_lo;
        }
        return base + static_cast<uint32_t>(idx) * src_bytes;
    }
    // A run is contiguous in each source region; split it where it crosses own_lo / own_hi.
    FORCE_INLINE void read(int32_t row, int32_t w, uint32_t chan_byte, uint32_t dst, uint32_t bytes) const {
        int32_t s = row * static_cast<int32_t>(W) + w;
        while (bytes > 0) {
            uint32_t chunk = bytes;
            const int32_t region_end = s < own_lo ? own_lo : (s < own_hi ? own_hi : 0x7fffffff);
            if (region_end != 0x7fffffff) {
                const uint32_t avail = static_cast<uint32_t>(region_end - s) * src_bytes - chan_byte;
                chunk = avail < bytes ? avail : bytes;
            }
            local_read(addr(s) + chan_byte, dst, chunk);
            dst += chunk;
            bytes -= chunk;
            s = region_end;
            chan_byte = 0;
        }
    }
    // Address of stick (row, w_first) + chan_byte when sticks [w_first, w_end) of that input row
    // lie in one source region (halo-before / shard / halo-after, each affine in the stick).
    FORCE_INLINE bool run_addr(int32_t row, int32_t w_first, int32_t w_end, uint32_t chan_byte, uint32_t& a) const {
        const int32_t lo = row * static_cast<int32_t>(W) + w_first;
        const int32_t hi = row * static_cast<int32_t>(W) + w_end;  // exclusive
        if ((lo < own_lo && hi > own_lo) || (lo >= own_lo && lo < own_hi && hi > own_hi)) {
            return false;
        }
        a = addr(lo) + chan_byte;
        return true;
    }
};

// One im2col row (one output pixel) restricted to K-block columns [k0, k0 + kblock_elems).
template <typename Source>
FORCE_INLINE void gather_row(
    uint64_t zeros_noc, const Source& src, uint32_t n, int32_t h_base, int32_t w_base, uint32_t k0, uint32_t dst) {
    // In-image tap columns of this pixel: kw in [kw_lo, kw_hi).
    int32_t kw_lo, kw_hi;
    in_image_taps(w_base, kw_lo, kw_hi);
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
        const int32_t h = h_base + static_cast<int32_t>(kh * DIL);
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
                // Band offset a = tap a / tap_width (stick w_base + tap*DIL), channel a % tap_width.
                // DIL > 1: the taps' sticks are not adjacent — one read per tap.
                uint32_t x = a;
                while (x < b) {
                    const uint32_t tap = x / tap_width;
                    uint32_t x_end = taps_contiguous ? b : (tap + 1) * tap_width;
                    x_end = x_end < b ? x_end : b;
                    src.read(
                        static_cast<int32_t>(n * H) + h,
                        w_base + static_cast<int32_t>(tap * DIL),
                        (x - tap * tap_width) * elem_bytes,
                        seg_dst + (x - o) * elem_bytes,
                        (x_end - x) * elem_bytes);
                    x = x_end;
                }
                if (b < seg_end) {
                    zero_run(zeros_noc, seg_dst + (b - o) * elem_bytes, (seg_end - b) * elem_bytes);
                }
            }
        }
        c += seg_end - o;
    }
}

// A K-block's columns [k0, k0 + kblock_elems) split at kernel-row band boundaries: segment i is
// band kh[i], band offset tap[i] taps + chan[i] bytes, `bytes` long, at dst[i] in the im2col row.
// Built once per K-block; an interior pixel (whole kH x kW window in-image) then issues one read
// per segment with no padding / clipping arithmetic (gather_row is the general path for edges).
constexpr uint32_t dense_max_segments = (kblock_elems + band_elems - 1) / band_elems + 1;
// DIL > 1: a segment is one (band, tap) piece.
constexpr uint32_t dense_dil_segments =
    taps_contiguous ? dense_max_segments : (kblock_elems + tap_width - 1) / tap_width + 2 * dense_max_segments;
constexpr uint32_t max_segments = depthwise && kH * kW > dense_dil_segments ? kH* kW : dense_dil_segments;
struct KBlockSegments {
    uint32_t count;
    int32_t kh[max_segments];   // input row offset of the segment's kernel row (kh * DIL)
    int32_t tap[max_segments];  // input column offset of its first tap (kw * DIL)
    uint32_t chan[max_segments];
    uint32_t bytes[max_segments];
    uint32_t dst[max_segments];
    int32_t span[max_segments];  // sticks one read touches
};

FORCE_INLINE void build_segments(uint32_t k0, KBlockSegments& seg) {
    seg.count = 0;
    uint32_t c = k0;
    const uint32_t end = k0 + kblock_elems;
    while (c < end) {
        const uint32_t kh = c / band_elems;
        const uint32_t o = c - kh * band_elems;
        uint32_t seg_end = band_elems;
        if (seg_end - o > end - c) {
            seg_end = o + (end - c);
        }
        // Band columns >= kW * tap_width are K padding: zero-filled by the general path.
        const uint32_t real_end = seg_end < kW * tap_width ? seg_end : kW * tap_width;
        uint32_t x = o;
        while (x < real_end) {
            // DIL > 1: split at tap boundaries (each tap's stick is DIL sticks from the previous).
            uint32_t x_end = taps_contiguous ? real_end : (x / tap_width + 1) * tap_width;
            x_end = x_end < real_end ? x_end : real_end;
            const uint32_t i = seg.count++;
            seg.kh[i] = static_cast<int32_t>(kh * DIL);
            seg.tap[i] = static_cast<int32_t>((x / tap_width) * DIL);
            seg.chan[i] = (x % tap_width) * elem_bytes;
            seg.bytes[i] = (x_end - x) * elem_bytes;
            seg.dst[i] = (c - k0 + (x - o)) * elem_bytes;
            seg.span[i] = static_cast<int32_t>((seg.chan[i] + seg.bytes[i] + tap_bytes - 1) / tap_bytes);
            x = x_end;
        }
        c += seg_end - o;
    }
}

// Depthwise segments of n-block nb: one per kernel row (dw_full_stick: the kW taps' sticks are
// contiguous) or one per tap (a channel sub-range of each stick).
FORCE_INLINE void build_segments_dw(uint32_t n_block, KBlockSegments& seg) {
    const uint32_t chan0 = n_block * dw_tap_bytes;
    seg.count = 0;
    for (uint32_t kh = 0; kh < kH; ++kh) {
        if constexpr (dw_row_runs) {
            const uint32_t i = seg.count++;
            seg.kh[i] = static_cast<int32_t>(kh);
            seg.tap[i] = 0;
            seg.chan[i] = 0;
            seg.bytes[i] = kW * tap_bytes;
            seg.dst[i] = kh * kW * dw_tap_bytes;
            seg.span[i] = static_cast<int32_t>(kW);
        } else {
            for (uint32_t kw = 0; kw < kW; ++kw) {
                const uint32_t i = seg.count++;
                seg.kh[i] = static_cast<int32_t>(kh * DIL);
                seg.tap[i] = static_cast<int32_t>(kw * DIL);
                seg.chan[i] = chan0;
                seg.bytes[i] = dw_tap_bytes;
                seg.dst[i] = (kh * kW + kw) * dw_tap_bytes;
                seg.span[i] = 1;
            }
        }
    }
}

// One depthwise im2col row (one output pixel) of n-block nb: per tap, the n-block's channels of
// the tap's input stick, or zeros for a padding tap.
template <typename Source>
FORCE_INLINE void gather_row_dw(
    uint64_t zeros_noc, const Source& src, uint32_t n, int32_t h_base, int32_t w_base, uint32_t n_block, uint32_t dst) {
    const uint32_t chan0 = n_block * dw_tap_bytes;
    int32_t kw_lo, kw_hi;
    in_image_taps(w_base, kw_lo, kw_hi);
    for (uint32_t kh = 0; kh < kH; ++kh) {
        const uint32_t row_dst = dst + kh * kW * dw_tap_bytes;
        const int32_t h = h_base + static_cast<int32_t>(kh * DIL);
        if (h < 0 || h >= static_cast<int32_t>(H) || kw_hi <= kw_lo) {
            zero_run(zeros_noc, row_dst, kW * dw_tap_bytes);
            continue;
        }
        if (kw_lo > 0) {
            zero_run(zeros_noc, row_dst, static_cast<uint32_t>(kw_lo) * dw_tap_bytes);
        }
        const int32_t row = static_cast<int32_t>(n * H) + h;
        if constexpr (dw_row_runs) {
            src.read(
                row,
                w_base + kw_lo,
                0,
                row_dst + static_cast<uint32_t>(kw_lo) * dw_tap_bytes,
                static_cast<uint32_t>(kw_hi - kw_lo) * tap_bytes);
        } else {
            for (int32_t kw = kw_lo; kw < kw_hi; ++kw) {
                src.read(
                    row,
                    w_base + kw * static_cast<int32_t>(DIL),
                    chan0,
                    row_dst + static_cast<uint32_t>(kw) * dw_tap_bytes,
                    dw_tap_bytes);
            }
        }
        if (kw_hi < static_cast<int32_t>(kW)) {
            zero_run(
                zeros_noc,
                row_dst + static_cast<uint32_t>(kw_hi) * dw_tap_bytes,
                (kW - static_cast<uint32_t>(kw_hi)) * dw_tap_bytes);
        }
    }
}

// True when K-block columns outside the segments (band K padding) must be zeroed: then the

// fast path is not taken (it writes only the segments).
FORCE_INLINE bool kblock_has_k_padding(uint32_t k0) {
    uint32_t covered = 0;
    uint32_t c = k0;
    const uint32_t end = k0 + kblock_elems;
    while (c < end) {
        const uint32_t kh = c / band_elems;
        const uint32_t o = c - kh * band_elems;
        uint32_t seg_end = band_elems;
        if (seg_end - o > end - c) {
            seg_end = o + (end - c);
        }
        const uint32_t real_end = seg_end < kW * tap_width ? seg_end : kW * tap_width;
        covered += o < real_end ? real_end - o : 0;
        c += seg_end - o;
    }
    return covered != kblock_elems;
}

// gather_tile_row: 32 im2col rows (output pixels from `pix`) x one K-block into dst_base.
// Interior pixels of one output row form a run: per segment, the source advances by `stride`
// sticks per pixel, so the run is a tight stateful-read loop with an affine source address
// (one read per pixel per segment). Edge pixels (padding taps) and K-padded blocks take the
// general gather_row path (gather_row_dw for depthwise, where `k0` is the n-block index).
template <typename Source>
FORCE_INLINE void gather_tile_row(
    uint64_t zeros_noc,
    const Source& src,
    const KBlockSegments& seg,
    bool fast_ok,
    uint32_t k0,
    uint32_t pix,
    uint32_t dst_base,
    int32_t stride,
    int32_t padding,
    int32_t wo_int_lo,
    int32_t wo_int_hi) {
    constexpr uint32_t hw_out = H_out * W_out;
    uint32_t n = pix / hw_out;
    uint32_t rem = pix - n * hw_out;
    uint32_t ho = rem / W_out;
    uint32_t wo = rem - ho * W_out;
    const uint32_t step = static_cast<uint32_t>(stride) * src_bytes;
    uint32_t r = 0;
    while (r < TILE_DIM) {
        const uint32_t dst = dst_base + r * im2col_row_bytes;
        if (pix >= M) {
            zero_run(zeros_noc, dst, (TILE_DIM - r) * im2col_row_bytes);
            return;
        }
        const int32_t h_base = static_cast<int32_t>(ho) * stride - padding;
        const int32_t w_base = static_cast<int32_t>(wo) * stride - padding;
        uint32_t run = 1;
        if (fast_ok && h_base >= 0 && h_base + eff_kH <= static_cast<int32_t>(H) &&
            static_cast<int32_t>(wo) >= wo_int_lo && static_cast<int32_t>(wo) <= wo_int_hi) {
            run = static_cast<uint32_t>(wo_int_hi) + 1 - wo;
            run = run < TILE_DIM - r ? run : TILE_DIM - r;
            run = run < M - pix ? run : M - pix;
            const int32_t row0 = static_cast<int32_t>(n * H) + h_base;
            for (uint32_t i = 0; i < seg.count; ++i) {
                const int32_t row = row0 + seg.kh[i];
                const int32_t w_first = w_base + seg.tap[i];
                const uint32_t bytes = seg.bytes[i];
                uint32_t d = dst + seg.dst[i];
                uint32_t a;
                if (src.run_addr(
                        row, w_first, w_first + static_cast<int32_t>(run - 1) * stride + seg.span[i], seg.chan[i], a)) {
                    for (uint32_t j = 0; j < run; ++j) {
                        local_read(a, d, bytes);
                        a += step;
                        d += im2col_row_bytes;
                    }
                } else {
                    for (uint32_t j = 0; j < run; ++j) {
                        src.read(row, w_first + static_cast<int32_t>(j) * stride, seg.chan[i], d, bytes);
                        d += im2col_row_bytes;
                    }
                }
            }
        } else if constexpr (depthwise) {
            gather_row_dw(zeros_noc, src, n, h_base, w_base, k0, dst);
        } else {
            gather_row(zeros_noc, src, n, h_base, w_base, k0, dst);
        }
        r += run;
        pix += run;
        wo += run;
        if (wo == W_out) {
            wo = 0;
            if (++ho == H_out) {
                ho = 0;
                ++n;
            }
        }
    }
}

// gather_tile_row_taps (tap_major): 32 pixels from `pix` x the K-block's taps [k0 / tap_width, ...).
// Tap j of the block lands at dst_base + j * 32 * tap_bytes (tap_width/32 tile-sized RM pages: rows =
// pixels). Per tap, pixels whose tap stick is in-image form runs along an output row (stride 1: the
// source sticks are consecutive), each one read; padding taps and pixels >= M are zero runs.
template <typename Source>
FORCE_INLINE void gather_tile_row_taps(
    uint64_t zeros_noc, const Source& src, uint32_t k0, uint32_t pix0, uint32_t dst_base, int32_t padding) {
    constexpr uint32_t hw_out = H_out * W_out;
    constexpr uint32_t taps_per_block = kblock_elems / tap_width;
    constexpr uint32_t tap_block_bytes = TILE_DIM * tap_bytes;
    const uint32_t n0 = pix0 / hw_out;
    const uint32_t rem0 = pix0 - n0 * hw_out;
    const uint32_t ho0 = rem0 / W_out;
    const uint32_t wo0 = rem0 - ho0 * W_out;
    const uint32_t tap0 = k0 / tap_width;
    for (uint32_t j = 0; j < taps_per_block; ++j) {
        const uint32_t tap = tap0 + j;
        const int32_t kh = static_cast<int32_t>(tap / kW);
        const int32_t kw = static_cast<int32_t>(tap - (tap / kW) * kW);
        uint32_t dst = dst_base + j * tap_block_bytes;
        uint32_t pix = pix0, n = n0, ho = ho0, wo = wo0;
        uint32_t r = 0;
        while (r < TILE_DIM) {
            if (pix >= M) {
                zero_run(zeros_noc, dst, (TILE_DIM - r) * tap_bytes);
                break;
            }
            const int32_t h = static_cast<int32_t>(ho) - padding + kh;
            const int32_t w = static_cast<int32_t>(wo) - padding + kw;
            // Pixels left in this output row (and tile-row, and tensor).
            uint32_t run = W_out - wo;
            run = run < TILE_DIM - r ? run : TILE_DIM - r;
            run = run < M - pix ? run : M - pix;
            if (h < 0 || h >= static_cast<int32_t>(H)) {
                zero_run(zeros_noc, dst, run * tap_bytes);
            } else {
                // In-image span of the run: w in [0, W).
                uint32_t lead = w < 0 ? static_cast<uint32_t>(-w) : 0;
                lead = lead < run ? lead : run;
                const int32_t w_in = w + static_cast<int32_t>(lead);
                uint32_t body = run - lead;
                const int32_t room = static_cast<int32_t>(W) - w_in;
                if (room <= 0) {
                    body = 0;
                } else if (static_cast<uint32_t>(room) < body) {
                    body = static_cast<uint32_t>(room);
                }
                if (lead > 0) {
                    zero_run(zeros_noc, dst, lead * tap_bytes);
                }
                if (body > 0) {
                    src.read(static_cast<int32_t>(n * H) + h, w_in, 0, dst + lead * tap_bytes, body * tap_bytes);
                }
                const uint32_t tail = run - lead - body;
                if (tail > 0) {
                    zero_run(zeros_noc, dst + (lead + body) * tap_bytes, tail * tap_bytes);
                }
            }
            dst += run * tap_bytes;
            r += run;
            pix += run;
            wo += run;
            if (wo == W_out) {
                wo = 0;
                if (++ho == H_out) {
                    ho = 0;
                    ++n;
                }
            }
        }
    }
}

// gather_m_blocks: this instance's m-blocks (role, role + num_gather_roles, ...) — window / halo fetch
// then the im2col gather of every K-block into cb_act_rm. RT args start at rt_base (reader layout).
// weights_on_reader_risc: an optional Poll functor (the reader's non-blocking weight ring step,
// returns true if it issued NoC reads) is run while cb_act_rm is full and once per tile-row. The
// gather then tags its reads with GATHER_TRID and waits only on that id, so in-flight weight DRAM
// reads never stall a tile-row.
struct NoPoll {
    bool operator()() const { return false; }
};
constexpr uint32_t GATHER_TRID = 15;
template <typename Poll = NoPoll>
FORCE_INLINE void gather_m_blocks(uint32_t rt_base, uint32_t role, Poll poll = Poll{}) {
    constexpr bool polled = !std::is_same_v<Poll, NoPoll>;
    auto restore_gather_state = [&]() {
        set_local_read_state();
        noc_async_read_set_trid(GATHER_TRID);
    };
    auto run_poll = [&]() {
        if constexpr (polled) {
            if (poll()) {
                restore_gather_state();
            }
        }
    };
    [[maybe_unused]] const uint32_t input_addr = get_arg_val<uint32_t>(rt_base + 0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(rt_base + 1);
    const uint32_t m_tiles_this_core = get_arg_val<uint32_t>(rt_base + 2);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(rt_base + 3);
    const int32_t padding = static_cast<int32_t>(get_arg_val<uint32_t>(rt_base + 4));
    const int32_t stride = static_cast<int32_t>(get_arg_val<uint32_t>(rt_base + 5));
    constexpr int32_t dilation = static_cast<int32_t>(DIL);  // RT arg rt_base + 6 mirrors it
    // R0 depthwise: byte offset of this core's channel tiles in a DRAM input stick (else 0).
    [[maybe_unused]] const uint32_t win_chan_offset = get_arg_val<uint32_t>(rt_base + 7);
    // R4/R5 only: resident stick range, then (noc_x, noc_y) of each input shard in this column.
    const int32_t own_lo = k_exchange ? static_cast<int32_t>(get_arg_val<uint32_t>(rt_base + 8)) : 0;
    const int32_t own_hi = k_exchange ? static_cast<int32_t>(get_arg_val<uint32_t>(rt_base + 9)) : 0;
    const uint32_t shard_base = k_exchange ? get_read_ptr(cb_in_shard) : 0;
    if constexpr (direct_in0) {
        const uint32_t pages = num_m_blocks * block_m_tiles * block_k_tiles;  // whole shard tile-rows
        cb_reserve_back(cb_in_shard, pages);
        cb_push_back(cb_in_shard, pages);
        return;
    }

    // fetch_sticks: flattened sticks [p, q) of this channel slice from their owning shards
    // (same L1 address on every core), one NoC read per owner run.
    auto fetch_sticks = [&](int32_t p, int32_t q, uint32_t dst) {
        while (p < q) {
            const uint32_t r = static_cast<uint32_t>(p) / shard_h;
            int32_t e = static_cast<int32_t>((r + 1) * shard_h);
            e = e < q ? e : q;
            const uint32_t nx = get_arg_val<uint32_t>(rt_base + 10 + 2 * r);
            const uint32_t ny = get_arg_val<uint32_t>(rt_base + 11 + 2 * r);
            const uint32_t src = shard_base + (static_cast<uint32_t>(p) - r * shard_h) * src_bytes;
            const uint32_t bytes = static_cast<uint32_t>(e - p) * src_bytes;
            noc_async_read(get_noc_addr(nx, ny, src), dst, bytes);
            dst += bytes;
            p = e;
        }
    };

    const uint32_t window_base = get_write_ptr(cb_act_window);
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    // Unaligned C_in (R0): zero the lanes past each stick's data once — staging slots (copied
    // tap_bytes wide into the window) or the window itself (direct reads write stick_bytes only).
    auto zero_local = [&](uint32_t dst, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
            noc_async_read(zeros_noc, dst, chunk);
            dst += chunk;
            bytes -= chunk;
        }
    };
    [[maybe_unused]] uint32_t stage_base = 0;
    if constexpr (stage_stride > 0) {
        stage_base = get_write_ptr(cb_stage);
        zero_local(stage_base, 2 * stage_row_sticks * stage_stride);
    }
    if constexpr (window_zero_bytes > 0) {
        zero_local(window_base, window_zero_bytes);
    }
    if constexpr (stage_stride > 0 || window_zero_bytes > 0) {
        noc_async_read_barrier();
    }
    constexpr uint32_t hw_out = H_out * W_out;
    const uint32_t m_tile_end = m_tile_start + m_tiles_this_core;
    // Output columns whose whole (dilated) kW-tap window is in-image: [wo_int_lo, wo_int_hi].
    const int32_t wo_int_lo = (padding + stride - 1) / stride;
    const int32_t wo_int_num = static_cast<int32_t>(W) - eff_kW + padding;
    const int32_t wo_int_hi = wo_int_num >= 0 ? wo_int_num / stride : -1;

    // TILE: the untilized window as a contiguous stick source from stick `first_stick` on.
    auto tile_window_src = [&](int32_t first_stick) {
        const uint32_t base = get_read_ptr(cb_act_window);
        return ShardSource{base, base, first_stick, 0x7fffffff, first_stick, 0x7fffffff, 0};
    };
    ShardSource tile_src{};
    [[maybe_unused]] uint32_t tile_win_pages = 0;
    if constexpr (tile_in && k_exchange) {
        // ---- fetch_tile_window (technique 1): own shard in place + halo tile-rows, once ----
        constexpr uint32_t tile_rows_per_shard = shard_h >= TILE_DIM ? shard_h / TILE_DIM : 1;  // RM: unused
        const uint32_t tw = rt_base + 10 + 2 * num_col_shards;
        const uint32_t first_tr = get_arg_val<uint32_t>(tw + 0);
        const uint32_t n_before = get_arg_val<uint32_t>(tw + 1);
        const uint32_t n_own = get_arg_val<uint32_t>(tw + 2);
        const uint32_t n_after = get_arg_val<uint32_t>(tw + 3);
        const uint32_t in_tile_bytes = get_tile_size(cb_win_tiles);
        const uint32_t row_bytes = win_c_tiles * in_tile_bytes;
        // The resident shard is the own tile-rows' source: publish it to compute (no copy).
        cb_reserve_back(cb_in_shard, n_own * win_c_tiles);
        cb_push_back(cb_in_shard, n_own * win_c_tiles);
        // Tile-rows [p, q) from their owning shards (same L1 address on every core), one read per run.
        auto fetch_tile_rows = [&](uint32_t p, uint32_t q, uint32_t dst) {
            while (p < q) {
                const uint32_t r = p / tile_rows_per_shard;
                uint32_t e = (r + 1) * tile_rows_per_shard;
                e = e < q ? e : q;
                const uint32_t nx = get_arg_val<uint32_t>(rt_base + 10 + 2 * r);
                const uint32_t ny = get_arg_val<uint32_t>(rt_base + 11 + 2 * r);
                const uint32_t src = shard_base + (p - r * tile_rows_per_shard) * row_bytes;
                noc_async_read(get_noc_addr(nx, ny, src), dst, (e - p) * row_bytes);
                dst += (e - p) * row_bytes;
                p = e;
            }
        };
        const uint32_t halo_pages = (n_before + n_after) * win_c_tiles;
        if (halo_pages > 0) {
            cb_reserve_back(cb_win_tiles, halo_pages);
            const uint32_t dst = get_write_ptr(cb_win_tiles);
            fetch_tile_rows(first_tr, first_tr + n_before, dst);
            const uint32_t own_end = first_tr + n_before + n_own;
            fetch_tile_rows(own_end, own_end + n_after, dst + n_before * row_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_win_tiles, halo_pages);
        }
        tile_win_pages = (n_before + n_own + n_after) * win_c_tiles;
        cb_wait_front(cb_act_window, tile_win_pages);
        tile_src = tile_window_src(static_cast<int32_t>(first_tr * TILE_DIM));
    }

    for (uint32_t m_block = role; m_block < num_m_blocks; m_block += num_gather_roles) {
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
        // Dilated block inside one output row (R0 RM): fetch only its kH kernel rows (raw_lo + kh*DIL),
        // window row kh; rows between the taps are never read (mirrors host _window_sticks).
        const bool dil_rows = DIL > 1 && !tile_in && n0 == n1 && ho0 == ho1;
        WindowSource window_src{
            window_base,
            dil_rows ? static_cast<int32_t>(n0 * H) + raw_lo : row_lo,
            col_lo,
            win_w,
            dil_rows ? dilation : 1};
        ShardSource shard_src{};
        if constexpr (tile_in && k_exchange) {
            shard_src = tile_src;
        } else if constexpr (tile_in) {
            // ---- fetch_tile_window (technique 2): the m-block's window tile-rows from DRAM ----
            const uint32_t in_tile_bytes = get_tile_size(cb_win_tiles);
            const auto input = TensorAccessor(input_args, input_addr, in_tile_bytes);
            const uint32_t tr_lo = static_cast<uint32_t>(row_lo * static_cast<int32_t>(W) + col_lo) / TILE_DIM;
            const uint32_t tr_hi = static_cast<uint32_t>(row_hi * static_cast<int32_t>(W) + last_hi) / TILE_DIM;
            // Native groups: the core's group's tile columns start at win_chan_offset.
            const uint32_t ct0 = win_chan_offset / (TILE_DIM * elem_bytes);
            cb_reserve_back(cb_win_tiles, win_tile_rows * win_c_tiles);
            uint32_t dst = get_write_ptr(cb_win_tiles);
            for (uint32_t tr = tr_lo; tr <= tr_hi; ++tr) {
                for (uint32_t ct = 0; ct < win_c_tiles; ++ct) {
                    noc_async_read(input.get_noc_addr(tr * in_row_tiles + ct0 + ct), dst, in_tile_bytes);
                    dst += in_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_win_tiles, win_tile_rows * win_c_tiles);
            cb_wait_front(cb_act_window, win_tile_rows * win_c_tiles);
            shard_src = tile_window_src(static_cast<int32_t>(tr_lo * TILE_DIM));
        } else if constexpr (!k_exchange) {
            const auto input = TensorAccessor(input_args, input_addr);
            [[maybe_unused]] uint32_t half = 0;
            const int32_t row_first = dil_rows ? window_src.row0 : row_lo;
            const int32_t row_last = dil_rows ? window_src.row0 + dilation * static_cast<int32_t>(kH - 1) : row_hi;
            for (int32_t row = row_first; row <= row_last; row += window_src.row_step) {
                if (dil_rows && (row < static_cast<int32_t>(n0 * H) || row >= static_cast<int32_t>((n0 + 1) * H))) {
                    continue;  // a padding kernel row: never read
                }
                const int32_t c0 = (row == row_lo || dil_rows) ? col_lo : mid_lo;
                const int32_t c1 = (row == row_hi || dil_rows) ? last_hi : mid_hi;
                uint32_t dst =
                    window_base + static_cast<uint32_t>(window_src.window_row(row) * win_w + (c0 - col_lo)) * src_bytes;
                uint32_t s = static_cast<uint32_t>(row) * W + static_cast<uint32_t>(c0);
                if constexpr (stage_stride == 0) {
                    for (int32_t c = c0; c <= c1; ++c, ++s) {
                        noc_async_read(input.get_noc_addr(s, win_chan_offset), dst, stick_bytes);
                        dst += src_bytes;
                    }
                } else {
                    // Staged: DRAM -> aligned staging half (one barrier per chunk), then local copies
                    // into the packed window, overlapping the next chunk's DRAM reads (other half).
                    uint32_t left = static_cast<uint32_t>(c1 - c0 + 1);
                    while (left > 0) {
                        const uint32_t cnt = left < stage_row_sticks ? left : stage_row_sticks;
                        const uint32_t stg = stage_base + half * stage_row_sticks * stage_stride;
                        for (uint32_t j = 0; j < cnt; ++j, ++s) {
                            noc_async_read(input.get_noc_addr(s, win_chan_offset), stg + j * stage_stride, stick_bytes);
                        }
                        noc_async_read_barrier();
                        for (uint32_t j = 0; j < cnt; ++j) {
                            noc_async_read(get_noc_addr(stg + j * stage_stride), dst, tap_bytes);
                            dst += tap_bytes;
                        }
                        half ^= 1;
                        left -= cnt;
                    }
                }
            }
        } else {
            // ---- fetch_halo_block: window sticks [win_lo, win_hi] outside [own_lo, own_hi) ----
            // (mirrors host _window_stick_range / _halo_sticks)
            const int32_t win_lo = row_lo * static_cast<int32_t>(W) + col_lo;
            const int32_t win_hi = row_hi * static_cast<int32_t>(W) + last_hi;
            const int32_t b_end = own_lo < win_hi + 1 ? own_lo : win_hi + 1;
            const int32_t n_before = b_end > win_lo ? b_end - win_lo : 0;
            const int32_t after_lo = own_hi > win_lo ? own_hi : win_lo;
            const int32_t n_after = win_hi + 1 > after_lo ? win_hi + 1 - after_lo : 0;
            shard_src = ShardSource{shard_base, window_base, own_lo, own_hi, win_lo, after_lo, n_before};
            fetch_sticks(win_lo, win_lo + n_before, window_base);
            fetch_sticks(after_lo, after_lo + n_after, window_base + static_cast<uint32_t>(n_before) * src_bytes);
        }
        {
            MaybeDeviceZoneScope("reader_window_barrier");
            noc_async_read_barrier();
        }
        set_local_read_state();
        if constexpr (polled) {
            noc_async_read_set_trid(GATHER_TRID);
        }

        // ---- gather_im2col_kblock, per n-block (activation is reused across n) ----
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                MaybeDeviceZoneScope("reader_gather_kblock");
                // Depthwise: one K-block per n-block; `k0` carries the n-block index.
                const uint32_t k0 = depthwise ? n_block : k_block * kblock_elems;
                KBlockSegments seg;
                bool fast_ok = true;
                if constexpr (depthwise) {
                    build_segments_dw(n_block, seg);
                } else {
                    build_segments(k0, seg);
                    // K padding inside the block's columns: rows need zero runs -> general path only.
                    fast_ok = !kblock_has_k_padding(k0);
                }
                for (uint32_t t = 0; t < block_m_tiles; ++t) {
                    const uint32_t tile = t0 + t;
                    {
                        MaybeDeviceZoneScope("reader_rm_reserve");
                        if constexpr (polled) {
                            while (!cb_pages_reservable_at_back(cb_act_rm, block_k_tiles)) {
                                run_poll();
                            }
                        }
                        cb_reserve_back(cb_act_rm, block_k_tiles);
                    }
                    const uint32_t dst_base = get_write_ptr(cb_act_rm);
                    if (tile >= t1) {
                        // Ragged block tail: tile-row past the core's range.
                        zero_run(zeros_noc, dst_base, TILE_DIM * im2col_row_bytes);
                    } else if constexpr (tap_major && contiguous_src) {
                        gather_tile_row_taps(zeros_noc, shard_src, k0, tile * TILE_DIM, dst_base, padding);
                    } else if constexpr (tap_major) {
                        gather_tile_row_taps(zeros_noc, window_src, k0, tile * TILE_DIM, dst_base, padding);
                    } else if constexpr (contiguous_src) {
                        gather_tile_row(
                            zeros_noc,
                            shard_src,
                            seg,
                            fast_ok,
                            k0,
                            tile * TILE_DIM,
                            dst_base,
                            stride,
                            padding,
                            wo_int_lo,
                            wo_int_hi);
                    } else {
                        gather_tile_row(
                            zeros_noc,
                            window_src,
                            seg,
                            fast_ok,
                            k0,
                            tile * TILE_DIM,
                            dst_base,
                            stride,
                            padding,
                            wo_int_lo,
                            wo_int_hi);
                    }
                    if constexpr (polled) {
                        noc_async_read_barrier_with_trid(GATHER_TRID);
                    } else {
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_act_rm, block_k_tiles);
                    run_poll();
                }
            }
        }
        if constexpr (tile_in && !k_exchange) {
            cb_pop_front(cb_act_window, win_tile_rows * win_c_tiles);  // compute untilizes the next window
        }
    }
    if constexpr (tile_in && k_exchange) {
        cb_pop_front(cb_act_window, tile_win_pages);
    }
}

}  // namespace conv_gather
