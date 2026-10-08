// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the modulated deformable im2col.
//
// Per block of 32 output rows (pixels) it stages the rows' offsets and masks,
// then for each kernel position computes every row's bilinear sample point and
// hands compute the four corners as one row-major block, 32 rows of
// [corner 0 | corner 1 | corner 2 | corner 3] C-wide input sticks, landed by
// the NoC reads themselves, plus one scalar tile per corner whose column 0
// holds mask * corner_coefficient per row, rounded to nearest bfloat16.
// Compute tilizes the block and sums the four corners times their scalars,
// which is the masked bilinear sample.
//
// The geometry runs in Q16.16 integers: the data-movement RISC-Vs have no FPU,
// so a float operation is a software routine of about a hundred cycles, and
// this runs for every row and kernel position. Offsets saturate at
// +-OFFSET_CLAMP pixels, which keeps every coordinate inside int32 and puts a
// non-finite offset outside the input.
//
// A corner is read only when it lies inside the input and its coefficient is
// nonzero. Every other row, and every row past the block's end, is zeroed
// rather than skipped: a stale NaN or Inf times a zero scalar would still be
// NaN.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

constexpr uint32_t offset_arena_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t mask_arena_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t C = get_compile_time_arg_val(4);
constexpr int32_t IN_H = get_compile_time_arg_val(5);
constexpr int32_t IN_W = get_compile_time_arg_val(6);
constexpr uint32_t OUT_H = get_compile_time_arg_val(7);
constexpr uint32_t OUT_W = get_compile_time_arg_val(8);
constexpr uint32_t KERNEL_W = get_compile_time_arg_val(9);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(10);
constexpr int32_t STRIDE_H = get_compile_time_arg_val(11);
constexpr int32_t STRIDE_W = get_compile_time_arg_val(12);
constexpr int32_t PAD_H = get_compile_time_arg_val(13);
constexpr int32_t PAD_W = get_compile_time_arg_val(14);
constexpr int32_t DILATION_H = get_compile_time_arg_val(15);
constexpr int32_t DILATION_W = get_compile_time_arg_val(16);
// Page strides of the three row-major tensors, as their buffers lay them out.
constexpr uint32_t input_page_nbytes = get_compile_time_arg_val(17);
constexpr uint32_t offset_page_nbytes = get_compile_time_arg_val(18);
constexpr uint32_t mask_page_nbytes = get_compile_time_arg_val(19);
// The host validates that the input size, padding and kernel span stay below this.
constexpr int32_t OFFSET_CLAMP = get_compile_time_arg_val(20);

constexpr auto input_args = TensorAccessorArgs<21>();
constexpr auto offset_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto mask_args = TensorAccessorArgs<offset_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t CHANNEL_TILES = C / 32;
constexpr uint32_t CORNERS = 4;
constexpr uint32_t STICK_NBYTES = C * sizeof(uint16_t);  // the C channels read per corner
constexpr uint32_t BLOCK_ROW_NBYTES = CORNERS * STICK_NBYTES;

namespace {

using fixed_point_arithmetic::fixed_frac;
using fixed_point_arithmetic::fixed_mul;
using fixed_point_arithmetic::FIXED_ONE;
using fixed_point_arithmetic::fixed_to_int;
using fixed_point_arithmetic::int_to_fixed;

constexpr int32_t OFFSET_CLAMP_FIXED = int_to_fixed(OFFSET_CLAMP);

// bfloat16 -> Q16.16 in integers, saturating at +-clamp_fixed (at most 2^30). Zero, subnormals
// and values below Q16.16 resolution become 0; inf and NaN saturate with their sign.
inline int32_t bf16_to_fixed_saturating(uint16_t bf16, int32_t clamp_fixed) {
    const int32_t exponent = static_cast<int32_t>((bf16 >> 7) & 0xFFu);
    if (exponent == 0) {
        return 0;
    }
    // value = (0x80 | mantissa) * 2^(exponent - 134), so scaling by 2^16 shifts the 8-bit
    // significand left by (exponent - 118).
    const int32_t significand = static_cast<int32_t>(0x80u | (bf16 & 0x7Fu));
    const int32_t shift = exponent - 118;
    int32_t magnitude;
    if (shift > 22) {
        magnitude = clamp_fixed;  // the significand would overflow int32
    } else if (shift >= 0) {
        magnitude = significand << shift;
    } else if (shift > -32) {
        magnitude = significand >> (-shift);
    } else {
        magnitude = 0;
    }
    if (magnitude > clamp_fixed) {
        magnitude = clamp_fixed;
    }
    return (bf16 & 0x8000u) ? -magnitude : magnitude;
}

// bfloat16 * Q16.16 weight in (0, 1] -> bfloat16, rounded to nearest, in integers. The weight is
// normalised to a 16-bit significand, so the product holds more precision than the result.
inline uint16_t bf16_times_fixed_to_bf16(uint16_t value_bf16, uint32_t weight_fixed) {
    const int32_t value_exponent = static_cast<int32_t>((value_bf16 >> 7) & 0xFFu);
    if (value_exponent == 0 || weight_fixed == 0) {
        return 0;
    }
    const uint32_t value_significand = 0x80u | (value_bf16 & 0x7Fu);  // 1.m, 8 bits
    // weight = weight_significand * 2^(weight_msb - 31), weight_significand in [2^15, 2^16).
    const int32_t weight_msb = 31 - __builtin_clz(weight_fixed);
    const uint32_t weight_significand =
        (weight_msb >= 15) ? (weight_fixed >> (weight_msb - 15)) : (weight_fixed << (15 - weight_msb));
    const uint32_t product = value_significand * weight_significand;  // 23 or 24 bits
    int32_t product_msb = 31 - __builtin_clz(product);
    // One bit past the 8-bit significand, rounded half up; a carry moves up one binade.
    uint32_t significand = ((product >> (product_msb - 8)) + 1) >> 1;
    if (significand == 0x100u) {
        significand = 0x80u;
        ++product_msb;
    }
    // value = value_significand * 2^(value_exponent - 134).
    const int32_t out_exponent = (value_exponent - 134) + (weight_msb - 31) + product_msb + 127;
    if (out_exponent <= 0) {
        return 0;
    }
    const uint16_t sign = static_cast<uint16_t>(value_bf16 & 0x8000u);
    if (out_exponent >= 0xFF) {
        return static_cast<uint16_t>(sign | (0xFEu << 7) | 0x7Fu);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(out_exponent) << 7) | (significand & 0x7Fu));
}

// Column 0 of a bf16 tile row: rows 0-15 in face 0, rows 16-31 in face 2; a face is
// 16 x 16 bf16, so its rows are 32 B apart and the face 512 B.
inline uint32_t tile_col0_offset(uint32_t r) { return r < 16 ? r * 32 : 1024 + (r - 16) * 32; }

inline void zero_stick(uint32_t l1_addr) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t i = 0; i < STICK_NBYTES / sizeof(uint32_t); ++i) {
        dst[i] = 0;
    }
}

inline bool inside(int32_t index, int32_t extent) { return index >= 0 && index < extent; }

struct CornerGeometry {
    int32_t y0[TILE_ROWS];
    int32_t x0[TILE_ROWS];
    bool in_input[CORNERS][TILE_ROWS];
    uint16_t coeff_bf16[CORNERS][TILE_ROWS];  // mask * bilinear weight
};

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset_addr = get_arg_val<uint32_t>(1);
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    const auto input_acc = TensorAccessor(input_args, input_addr, input_page_nbytes);
    const auto offset_acc = TensorAccessor(offset_args, offset_addr, offset_page_nbytes);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_page_nbytes);

    Noc noc;
    CircularBuffer offset_arena_cb(offset_arena_cb_index);
    CircularBuffer mask_arena_cb(mask_arena_cb_index);
    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);

    // The arenas are reserved once and used as flat L1; nothing waits on them.
    offset_arena_cb.reserve_back(TILE_ROWS);
    const uint32_t offset_l1 = offset_arena_cb.get_write_ptr();
    mask_arena_cb.reserve_back(TILE_ROWS);
    const uint32_t mask_l1 = mask_arena_cb.get_write_ptr();

    CornerGeometry geom;
    uint32_t batch_base[TILE_ROWS];  // first input page of the row's image
    int32_t base_y[TILE_ROWS];       // h_out * stride - pad
    int32_t base_x[TILE_ROWS];       // w_out * stride - pad

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        const uint32_t row0 = block * TILE_ROWS;
        const uint32_t valid_rows = num_rows - row0 < TILE_ROWS ? num_rows - row0 : TILE_ROWS;

        for (uint32_t r = 0; r < valid_rows; ++r) {
            CoreLocalMem<uint32_t> off_dst(offset_l1 + r * offset_page_nbytes);
            noc.async_read(offset_acc, off_dst, offset_page_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
            CoreLocalMem<uint32_t> mask_dst(mask_l1 + r * mask_page_nbytes);
            noc.async_read(mask_acc, mask_dst, mask_page_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
        }

        // (b, h_out, w_out) of the block's first row, then stepped row by row.
        uint32_t b = row0 / (OUT_H * OUT_W);
        uint32_t h_o = (row0 / OUT_W) % OUT_H;
        uint32_t w_o = row0 % OUT_W;
        for (uint32_t r = 0; r < valid_rows; ++r) {
            batch_base[r] = b * IN_H * IN_W;
            base_y[r] = static_cast<int32_t>(h_o) * STRIDE_H - PAD_H;
            base_x[r] = static_cast<int32_t>(w_o) * STRIDE_W - PAD_W;
            if (++w_o == OUT_W) {
                w_o = 0;
                if (++h_o == OUT_H) {
                    h_o = 0;
                    ++b;
                }
            }
        }
        noc.async_read_barrier();

        for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
            const int32_t k_y = static_cast<int32_t>(kk / KERNEL_W) * DILATION_H;
            const int32_t k_x = static_cast<int32_t>(kk % KERNEL_W) * DILATION_W;

            for (uint32_t r = 0; r < valid_rows; ++r) {
                CoreLocalMem<volatile uint16_t> off(offset_l1 + r * offset_page_nbytes);
                CoreLocalMem<volatile uint16_t> msk(mask_l1 + r * mask_page_nbytes);
                const int32_t y =
                    int_to_fixed(base_y[r] + k_y) + bf16_to_fixed_saturating(off[2 * kk], OFFSET_CLAMP_FIXED);
                const int32_t x =
                    int_to_fixed(base_x[r] + k_x) + bf16_to_fixed_saturating(off[2 * kk + 1], OFFSET_CLAMP_FIXED);
                const uint16_t mask = msk[kk];

                // The arithmetic shift floors negative coordinates too, and the fraction is
                // taken against that floor.
                const int32_t y0 = fixed_to_int(y);
                const int32_t x0 = fixed_to_int(x);
                const int32_t ly = fixed_frac(y);
                const int32_t lx = fixed_frac(x);
                const int32_t hy = FIXED_ONE - ly;
                const int32_t hx = FIXED_ONE - lx;

                geom.y0[r] = y0;
                geom.x0[r] = x0;
                geom.in_input[0][r] = inside(y0, IN_H) && inside(x0, IN_W);
                geom.in_input[1][r] = inside(y0, IN_H) && inside(x0 + 1, IN_W);
                geom.in_input[2][r] = inside(y0 + 1, IN_H) && inside(x0, IN_W);
                geom.in_input[3][r] = inside(y0 + 1, IN_H) && inside(x0 + 1, IN_W);
                const int32_t weights[CORNERS] = {
                    fixed_mul(hy, hx), fixed_mul(hy, lx), fixed_mul(ly, hx), fixed_mul(ly, lx)};
                for (uint32_t c = 0; c < CORNERS; ++c) {
                    geom.coeff_bf16[c][r] =
                        geom.in_input[c][r] ? bf16_times_fixed_to_bf16(mask, static_cast<uint32_t>(weights[c])) : 0;
                }
            }

            corner_block_cb.reserve_back(CORNERS * CHANNEL_TILES);
            const uint32_t block_l1 = corner_block_cb.get_write_ptr();
            for (uint32_t c = 0; c < CORNERS; ++c) {
                const int32_t dy = c >> 1;
                const int32_t dx = c & 1;
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t dst_l1 = block_l1 + r * BLOCK_ROW_NBYTES + c * STICK_NBYTES;
                    // The page index below is valid only for a corner inside the input.
                    if (r >= valid_rows || !geom.in_input[c][r] || geom.coeff_bf16[c][r] == 0) {
                        zero_stick(dst_l1);
                        continue;
                    }
                    const uint32_t page = batch_base[r] + static_cast<uint32_t>(geom.y0[r] + dy) * IN_W +
                                          static_cast<uint32_t>(geom.x0[r] + dx);
                    CoreLocalMem<uint32_t> dst(dst_l1);
                    noc.async_read(input_acc, dst, STICK_NBYTES, {.page_id = page}, {.offset_bytes = 0});
                }
            }
            noc.async_read_barrier();
            corner_block_cb.push_back(CORNERS * CHANNEL_TILES);

            // Only column 0 of each row is read by the column broadcast.
            for (uint32_t c = 0; c < CORNERS; ++c) {
                scalar_tile_cb.reserve_back(1);
                const uint32_t scalar_l1 = scalar_tile_cb.get_write_ptr();
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    CoreLocalMem<volatile uint16_t> lane(scalar_l1 + tile_col0_offset(r));
                    lane[0] = r < valid_rows ? geom.coeff_bf16[c][r] : 0;
                }
                scalar_tile_cb.push_back(1);
            }
        }
    }
}
