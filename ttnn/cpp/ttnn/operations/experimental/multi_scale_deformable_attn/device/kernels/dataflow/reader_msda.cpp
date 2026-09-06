// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for fused multi-scale deformable attention.
//
// Produces one (input_tile, scalar_tile) pair per (p, corner) per output
// tile, where one output tile carries up to 32 queries (all from the same
// batch index n). The compute kernel multiplies them via
// mul_tiles_bcast<COL>: result[h, w] = input[h, w] * scalar[h, 0], so row h
// must hold query h's value-stick and scalar col-0 row h must hold that
// query's combined weight (attn * bilinear_corner).
//
// Tile face layout (bf16, 32x32 tile = 4 faces of 16x16, 512 B per face,
// 2048 B per tile):
//   TL face: offset    0..511   (rows  0..15, cols  0..15)
//   TR face: offset  512..1023  (rows  0..15, cols 16..31)
//   BL face: offset 1024..1535  (rows 16..31, cols  0..15)
//   BR face: offset 1536..2047  (rows 16..31, cols 16..31)
// Row r ∈ [0, 16) spans TL[r*32 .. r*32+31] + TR[512+r*32 .. 512+r*32+31].
// Row r ∈ [16, 32) spans BL[1024+(r-16)*32 ..] + BR[1536+(r-16)*32 ..].
// For COL bcast the scalar tile is read at col 0 of TL (rows 0..15) and
// col 0 of BL (rows 16..31): bytes 0, 32, 64, ..., 480 and 1024, 1056,
// ..., 1504. Non-col-0 lanes of the scalar tile are never written — the
// compute kernel calls mul_tiles_bcast<COL> with clear_fp32_dst_acc=true so
// DST is cleared on entry and only col-0 broadcasts contribute.
//
// Per-tile runtime args (3 per tile): (n, q_start, v_rows). 1 ≤ v_rows ≤ 32.
// Zero-fill contract:
//   * scalar tile: col 0 is explicitly written for all 32 rows. Tail rows
//     (r ≥ v_rows) and OOB-corner rows get bf16 0, so their contribution
//     zeroes out at the multiply.
//   * input tile: only valid rows (r < v_rows AND corner in-bounds) are
//     written; tail/OOB rows are left as whatever the CB slot held from a
//     previous iter. That's safe because the matching scalar lane is 0, so
//     stale bytes contribute 0 to the accumulator.

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/msda_tile_layout.hpp"

namespace {

// The RISC-V dataflow cores have no FPU, so every float operation here is a
// software routine costing ~100 cycles. The bilinear geometry runs once per
// (query row, point) and used to be ~47% of this kernel's time, so it is done
// in Q16.16 fixed point instead: grid coords live in [-1, 1] and the pixel
// coords they map to are bounded by the feature-map size, both of which fit
// integers with room to spare, and the weights only ever feed a bf16 result.

using fixed_point_arithmetic::fixed_frac;
using fixed_point_arithmetic::FIXED_HALF;
using fixed_point_arithmetic::fixed_mul;
using fixed_point_arithmetic::FIXED_ONE;
using fixed_point_arithmetic::fixed_one_minus;
using fixed_point_arithmetic::fixed_to_int;

// bf16 -> Q16.16, saturating at +-clamp_q16, which bounds the pixel-coord
// multiply below to int32. Inf/NaN saturate the same way.
//
// The clamp has to sit far enough out that the clamped coordinate still maps
// fully outside the feature map, or an out-of-range sample would come back
// in-bounds. With align_corners the mapping scales by (size - 1), so a 2-wide
// map needs +-3; without it the scale is size and +-2 is enough. The caller
// passes the right one for its mapping.
//
// The shared header has float_to_fixed but no bf16 entry point and no
// saturation, so this one stays local.
inline int32_t bf16_to_q16(uint16_t bf16, int32_t clamp_q16) {
    const int32_t exp = static_cast<int32_t>((bf16 >> 7) & 0xFFu);
    if (exp == 0) {
        return 0;  // zero or subnormal: below Q16.16 resolution anyway
    }
    const bool negative = (bf16 & 0x8000u) != 0;
    // value = (0x80 | mantissa) * 2^(exp - 127 - 7), so scaling by 2^16 shifts
    // the 8-bit significand left by (exp - 118).
    const int32_t shift = exp - 118;
    int32_t magnitude;
    if (shift >= 16) {
        // |value| >= 128 here, past any clamp this op passes, and shifting the
        // 8-bit significand that far would overflow int32.
        magnitude = clamp_q16;
    } else if (shift >= 0) {
        magnitude = static_cast<int32_t>((0x80u | (bf16 & 0x7Fu))) << shift;
    } else if (shift > -32) {
        magnitude = static_cast<int32_t>((0x80u | (bf16 & 0x7Fu))) >> (-shift);
    } else {
        magnitude = 0;
    }
    if (magnitude > clamp_q16) {
        magnitude = clamp_q16;
    }
    return negative ? -magnitude : magnitude;
}

// Precondition: v != 0. __builtin_clz(0) is undefined, and the fallback loop
// below would never terminate. Both callers check for zero first.
inline uint32_t count_leading_zeros(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_clz(v));
#else
    uint32_t n = 0;
    while ((v & 0x80000000u) == 0) {
        v <<= 1;
        ++n;
    }
    return n;
#endif
}

// (bf16 attention weight) * (Q16.16 corner weight) -> bf16, in integers.
//
// Non-finite attention weights come back as finite max magnitude with their
// sign, where the float path this replaces emitted inf or propagated NaN. That
// keeps a bad weight from poisoning the whole output tile, matching how the
// grid side of this kernel already treats non-finite sampling locations, but it
// does mean a caller debugging invalid weights sees a large finite number
// rather than a non-finite one.
//
// The shared header's fixed_to_bf16 converts a Q16.16 value on its own; there
// is no bf16-times-fixed entry point, and going through it would round the
// corner weight to bf16 before the multiply rather than after.
//
// The corner weight is normalised to a 16-bit significand before the multiply,
// so the 8-bit attention significand meets it with more precision than the
// bf16 result can hold -- same end value as multiplying in float32 and
// truncating, which is what this replaces.
inline uint16_t attn_times_weight_bf16(uint16_t attn_bf16, uint32_t weight_q16) {
    const int32_t attn_exp = static_cast<int32_t>((attn_bf16 >> 7) & 0xFFu);
    if (attn_exp == 0 || weight_q16 == 0) {
        return 0;
    }
    const uint32_t attn_significand = 0x80u | (attn_bf16 & 0x7Fu);  // 1.m, 8 bits

    // Normalise the weight to [2^15, 2^16): weight = significand * 2^(msb - 31).
    const int32_t weight_msb = 31 - static_cast<int32_t>(count_leading_zeros(weight_q16));
    const uint32_t weight_significand =
        (weight_msb >= 15) ? (weight_q16 >> (weight_msb - 15)) : (weight_q16 << (15 - weight_msb));

    // attn = attn_significand * 2^(attn_exp - 134); weight = weight_significand * 2^(weight_msb - 31).
    const uint32_t product = attn_significand * weight_significand;  // 23 or 24 bits
    const int32_t product_msb = 31 - static_cast<int32_t>(count_leading_zeros(product));
    const uint32_t mantissa = (product >> (product_msb - 7)) & 0x7Fu;
    const int32_t out_exp = (attn_exp - 134) + (weight_msb - 31) + product_msb + 127;
    if (out_exp <= 0) {
        return 0;  // underflows bf16's normal range
    }
    const uint16_t sign = static_cast<uint16_t>(attn_bf16 & 0x8000u);
    if (out_exp >= 0xFF) {
        // Saturate rather than emit inf, keeping the sign: a non-finite attn
        // weight would otherwise come back as +max and flip that corner's
        // contribution.
        return static_cast<uint16_t>(sign | (0xFEu << 7) | 0x7Fu);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(out_exp) << 7) | mantissa);
}

}  // namespace

constexpr uint32_t value_scratch_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t attn_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t input_tile_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(4);

constexpr uint32_t D = get_compile_time_arg_val(5);
constexpr uint32_t Q = get_compile_time_arg_val(6);
constexpr uint32_t P = get_compile_time_arg_val(7);
constexpr uint32_t h_in = get_compile_time_arg_val(8);
constexpr uint32_t w_in = get_compile_time_arg_val(9);
constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(10);
constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(11);
constexpr uint32_t attn_stick_nbytes = get_compile_time_arg_val(12);
constexpr bool ALIGN_CORNERS = get_compile_time_arg_val(13) != 0;

// See bf16_to_q16: align_corners scales by (size - 1), so a 2-wide map needs a
// wider clamp for an out-of-range coordinate to stay out of range.
constexpr int32_t GRID_CLAMP_Q16 = ALIGN_CORNERS ? 3 * FIXED_ONE : 2 * FIXED_ONE;

constexpr auto value_args = TensorAccessorArgs<14>();
constexpr auto grid_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto attn_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_MAX_ROWS = 32;
constexpr uint32_t HALF_STICK_NBYTES = 32;  // one face-row half: 16 bf16 (TL or TR portion of one row)
constexpr uint32_t HALF_WORDS = HALF_STICK_NBYTES / sizeof(uint32_t);
constexpr uint32_t TILE_NBYTES = 2048;  // bf16 32x32 tile

// A value stick carries D bf16 values (= D/2 uint32 words). One tile row
// holds 32 values (lo + hi face halves), so a stick spans N_D_TILES tiles
// laid side by side; the trailing tile is half-filled when D % 32 == 16.
// Derived from D (not value_stick_nbytes, which is alignment-padded).
constexpr uint32_t STICK_WORDS = D / 2;
constexpr uint32_t WORDS_PER_TILE_ROW = 2 * HALF_WORDS;
constexpr uint32_t N_D_TILES = (STICK_WORDS + WORDS_PER_TILE_ROW - 1) / WORDS_PER_TILE_ROW;
static_assert(D % 16 == 0 && D > 0, "D must be a positive multiple of 16");

void kernel_main() {
    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t grid_addr = get_arg_val<uint32_t>(1);
    const uint32_t attn_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(3);

    const auto value_acc = TensorAccessor(value_args, value_addr, value_stick_nbytes);
    const auto grid_acc = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto attn_acc = TensorAccessor(attn_args, attn_addr, attn_stick_nbytes);

    Noc noc;
    CircularBuffer value_scratch_cb(value_scratch_cb_index);
    CircularBuffer grid_cb(grid_cb_index);
    CircularBuffer attn_cb(attn_cb_index);
    CircularBuffer input_tile_cb(input_tile_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);

    constexpr int32_t h_in_i = static_cast<int32_t>(h_in);
    constexpr int32_t w_in_i = static_cast<int32_t>(w_in);

    // Reserve scratch CBs once and treat them as fixed linear L1 arenas.
    value_scratch_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t value_scratch_l1 = value_scratch_cb.get_write_ptr();
    grid_cb.reserve_back(TILE_MAX_ROWS * P);
    const uint32_t grid_scratch_l1 = grid_cb.get_write_ptr();
    attn_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t attn_scratch_l1 = attn_cb.get_write_ptr();

    // Per-(p, corner) precompute scratch (one entry per row in the current tile).
    uint16_t attn_bits_arr[TILE_MAX_ROWS];
    int32_t x0_arr[TILE_MAX_ROWS];
    int32_t y0_arr[TILE_MAX_ROWS];
    bool x0v_arr[TILE_MAX_ROWS];
    bool x1v_arr[TILE_MAX_ROWS];
    bool y0v_arr[TILE_MAX_ROWS];
    bool y1v_arr[TILE_MAX_ROWS];
    int32_t w_nw_arr[TILE_MAX_ROWS];
    int32_t w_ne_arr[TILE_MAX_ROWS];
    int32_t w_sw_arr[TILE_MAX_ROWS];
    int32_t w_se_arr[TILE_MAX_ROWS];

    uint32_t arg_idx = 4;
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const uint32_t n = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t q_start = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);

        // Stage attn for the v_rows queries (one P-wide stick each).
        for (uint32_t r = 0; r < v_rows; ++r) {
            CoreLocalMem<uint32_t> dst(attn_scratch_l1 + r * attn_stick_nbytes);
            noc.async_read(attn_acc, dst, attn_stick_nbytes, {.page_id = n * Q + (q_start + r)}, {.offset_bytes = 0});
        }
        // Stage grid for v_rows * P points (two bf16 each).
        for (uint32_t r = 0; r < v_rows; ++r) {
            const uint32_t base = n * (Q * P) + (q_start + r) * P;
            for (uint32_t p = 0; p < P; ++p) {
                CoreLocalMem<uint32_t> dst(grid_scratch_l1 + (r * P + p) * grid_stick_nbytes);
                noc.async_read(grid_acc, dst, grid_stick_nbytes, {.page_id = base + p}, {.offset_bytes = 0});
            }
        }
        noc.async_read_barrier();

        const uint32_t n_off = n * static_cast<uint32_t>(h_in_i * w_in_i);

        for (uint32_t p = 0; p < P; ++p) {
            // Precompute per-row geometry for this p.
            for (uint32_t r = 0; r < v_rows; ++r) {
                CoreLocalMem<volatile uint16_t> grid_ptr(grid_scratch_l1 + (r * P + p) * grid_stick_nbytes);
                CoreLocalMem<volatile uint16_t> attn_ptr(attn_scratch_l1 + r * attn_stick_nbytes);

                attn_bits_arr[r] = attn_ptr[p];

                // align_corners selects the pixel-coord mapping (mmcv default
                // is false: pixel = (g+1)*size/2 - 0.5; true variant uses
                // pixel = (g+1)*(size-1)/2). Scale first and halve afterwards,
                // through a 64-bit intermediate: halving (g + 1) up front would
                // keep the product in int32 but drop its low bit, and that
                // rounding is worth up to 0.5 * size / 2^16 of a pixel. For a
                // coordinate near zero -- where bf16 resolves finely -- that
                // exceeds the grid's own quantisation, so it is a real loss
                // rather than a free one.
                const int32_t gx_plus_one = bf16_to_q16(grid_ptr[0], GRID_CLAMP_Q16) + FIXED_ONE;
                const int32_t gy_plus_one = bf16_to_q16(grid_ptr[1], GRID_CLAMP_Q16) + FIXED_ONE;
                int32_t px_q16, py_q16;
                if constexpr (ALIGN_CORNERS) {
                    px_q16 = static_cast<int32_t>((static_cast<int64_t>(gx_plus_one) * (w_in_i - 1)) >> 1);
                    py_q16 = static_cast<int32_t>((static_cast<int64_t>(gy_plus_one) * (h_in_i - 1)) >> 1);
                } else {
                    px_q16 = static_cast<int32_t>((static_cast<int64_t>(gx_plus_one) * w_in_i) >> 1) - FIXED_HALF;
                    py_q16 = static_cast<int32_t>((static_cast<int64_t>(gy_plus_one) * h_in_i) >> 1) - FIXED_HALF;
                }

                // An arithmetic shift right is floor() for negatives too, and
                // what is left below the point is the interpolation fraction.
                const int32_t x0 = fixed_to_int(px_q16);
                const int32_t y0 = fixed_to_int(py_q16);
                const int32_t dx = fixed_frac(px_q16);
                const int32_t dy = fixed_frac(py_q16);

                x0_arr[r] = x0;
                y0_arr[r] = y0;
                x0v_arr[r] = (x0 >= 0) && (x0 < w_in_i);
                x1v_arr[r] = (x0 + 1 >= 0) && (x0 + 1 < w_in_i);
                y0v_arr[r] = (y0 >= 0) && (y0 < h_in_i);
                y1v_arr[r] = (y0 + 1 >= 0) && (y0 + 1 < h_in_i);

                const int32_t inv_dx = fixed_one_minus(dx);
                const int32_t inv_dy = fixed_one_minus(dy);
                w_nw_arr[r] = fixed_mul(inv_dx, inv_dy);
                w_ne_arr[r] = fixed_mul(dx, inv_dy);
                w_sw_arr[r] = fixed_mul(inv_dx, dy);
                w_se_arr[r] = fixed_mul(dx, dy);
            }

            for (uint32_t c = 0; c < 4; ++c) {
                // Hoist all c-invariant selectors out of the per-r loops below:
                // c picks which y/x validity array, which corner-weight array,
                // and the (dy, dx) offset to the corner.
                const int32_t dy_off = (c < 2) ? 0 : 1;
                const int32_t dx_off = (c & 1) ? 1 : 0;
                const bool* yv_arr = (c < 2) ? y0v_arr : y1v_arr;
                const bool* xv_arr = (c & 1) ? x1v_arr : x0v_arr;
                const int32_t* w_corner_arr = (c == 0)   ? w_nw_arr
                                              : (c == 1) ? w_ne_arr
                                              : (c == 2) ? w_sw_arr
                                                         : w_se_arr;

                // ---- INPUT TILES (N_D_TILES per (p, corner)) ----
                input_tile_cb.reserve_back(N_D_TILES);
                const uint32_t tile_l1 = input_tile_cb.get_write_ptr();

                // Issue NoC reads for all valid rows.
                for (uint32_t r = 0; r < v_rows; ++r) {
                    if (!(yv_arr[r] && xv_arr[r])) {
                        continue;
                    }
                    const uint32_t cy = static_cast<uint32_t>(y0_arr[r] + dy_off);
                    const uint32_t cx = static_cast<uint32_t>(x0_arr[r] + dx_off);
                    const uint32_t stick_idx = n_off + cy * w_in_i + cx;
                    CoreLocalMem<uint32_t> dst(value_scratch_l1 + r * value_stick_nbytes);
                    noc.async_read(value_acc, dst, value_stick_nbytes, {.page_id = stick_idx}, {.offset_bytes = 0});
                }
                noc.async_read_barrier();

                // Scatter sticks into face rows. Stick words [k*32row .. ] land in
                // d-tile k at the same row offsets. Invalid corners have stale
                // staging data but their scalar entry is zero — the multiply
                // contributes 0.
                for (uint32_t r = 0; r < v_rows; ++r) {
                    if (!(yv_arr[r] && xv_arr[r])) {
                        continue;
                    }
                    const auto off = msda_tile_layout::tile_row_offsets(r);
                    CoreLocalMem<volatile uint32_t> s(value_scratch_l1 + r * value_stick_nbytes);
                    for (uint32_t k = 0; k < N_D_TILES; ++k) {
                        const uint32_t base = k * WORDS_PER_TILE_ROW;
                        const uint32_t words_k =
                            (STICK_WORDS - base < WORDS_PER_TILE_ROW) ? (STICK_WORDS - base) : WORDS_PER_TILE_ROW;
                        const uint32_t lo_words = words_k < HALF_WORDS ? words_k : HALF_WORDS;
                        const uint32_t hi_words = words_k - lo_words;
                        const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
                        CoreLocalMem<volatile uint32_t> dl(ktile_l1 + off.lo);
                        CoreLocalMem<volatile uint32_t> dh(ktile_l1 + off.hi);
                        for (uint32_t i = 0; i < lo_words; ++i) {
                            dl[i] = s[base + i];
                        }
                        for (uint32_t i = 0; i < hi_words; ++i) {
                            dh[i] = s[base + HALF_WORDS + i];
                        }
                    }
                }

                // Tail rows (r ≥ v_rows) and OOB-corner rows are left untouched:
                // their scalar entry is zero (see scalar tile below), so any stale
                // bytes in input row r contribute 0 to L1 accumulation. Saves a
                // 16-row × 64-byte memset for tail tiles and skips work on full
                // tiles entirely. The same contract covers the unused hi halves of
                // a trailing half-filled d-tile (D % 32 == 16): the writer never
                // reads those lanes back.
                input_tile_cb.push_back(N_D_TILES);

                // ---- SCALAR TILE ----
                // LLK COL bcast reads only col 0 of TL face (rows 0..15) and BL
                // face (rows 16..31). Non-col-0 lanes are unused mathematically
                // (mul_tiles_bcast<COL> uses clear_fp32_dst_acc=true, so DST is
                // cleared on entry and only the col-0 broadcast contributes).
                // We therefore skip the 2-KiB full zero-fill and only write the
                // 32 col-0 bf16 lanes — 32× less L1 traffic per iter.
                scalar_tile_cb.reserve_back(1);
                const uint32_t s_tile_l1 = scalar_tile_cb.get_write_ptr();

                for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                    uint16_t bf = 0;
                    if (r < v_rows && yv_arr[r] && xv_arr[r]) {
                        bf = attn_times_weight_bf16(attn_bits_arr[r], static_cast<uint32_t>(w_corner_arr[r]));
                    }
                    // Rows ≥ v_rows OR invalid corners: bf stays 0 — explicitly
                    // overwrite col 0 because the CB slot may contain non-zero
                    // bf16 left by a previous tile where this row was valid.
                    CoreLocalMem<volatile uint16_t> p16(s_tile_l1 + msda_tile_layout::tile_col0_offset(r));
                    p16[0] = bf;
                }
                scalar_tile_cb.push_back(1);
            }
        }
    }
}
