// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the modulated deformable im2col.
//
// Per block of 32 output rows (pixels) it stages the rows' offsets and masks,
// then for each kernel position kk computes every row's bilinear sample point
// and hands compute the four corners as one row-major block, 32 rows of
// [corner 0 | corner 1 | corner 2 | corner 3] C-wide input sticks, landed by
// the NoC reads themselves, plus one scalar tile per corner whose column 0
// holds mask * corner_coefficient per row. Compute tilizes the block and sums
// the four corners times their scalars, which is the masked bilinear sample.
//
// Rows whose corner is outside the input, or past the block's end, are zeroed
// rather than skipped: a stale NaN or Inf times a zero scalar would still be
// NaN.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t offset_arena_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t mask_arena_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t C = get_compile_time_arg_val(4);
constexpr uint32_t IN_H = get_compile_time_arg_val(5);
constexpr uint32_t IN_W = get_compile_time_arg_val(6);
constexpr uint32_t OUT_H = get_compile_time_arg_val(7);
constexpr uint32_t OUT_W = get_compile_time_arg_val(8);
constexpr uint32_t KERNEL_W = get_compile_time_arg_val(9);
constexpr uint32_t KK = get_compile_time_arg_val(10);
constexpr int32_t STRIDE_H = get_compile_time_arg_val(11);
constexpr int32_t STRIDE_W = get_compile_time_arg_val(12);
constexpr int32_t PAD_H = get_compile_time_arg_val(13);
constexpr int32_t PAD_W = get_compile_time_arg_val(14);
constexpr int32_t DILATION_H = get_compile_time_arg_val(15);
constexpr int32_t DILATION_W = get_compile_time_arg_val(16);
constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(17);
constexpr uint32_t offset_stick_nbytes = get_compile_time_arg_val(18);
constexpr uint32_t mask_stick_nbytes = get_compile_time_arg_val(19);

constexpr auto input_args = TensorAccessorArgs<20>();
constexpr auto offset_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
constexpr auto mask_args = TensorAccessorArgs<offset_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t CHANNEL_TILES = C / 32;
constexpr uint32_t CORNERS = 4;
constexpr uint32_t STICK_NBYTES = C * sizeof(uint16_t);
constexpr uint32_t BLOCK_ROW_NBYTES = CORNERS * STICK_NBYTES;
// Column 0 of a bf16 tile row: rows 0-15 in face 0, rows 16-31 in face 2, 32 B apart.
constexpr uint32_t FACE_NBYTES = 512;
constexpr uint32_t FACE_ROW_NBYTES = 32;

namespace {

inline float bf16_to_float(uint16_t bf16) {
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

inline uint32_t tile_col0_offset(uint32_t r) {
    return r < 16 ? r * FACE_ROW_NBYTES : 2 * FACE_NBYTES + (r - 16) * FACE_ROW_NBYTES;
}

inline void zero_stick(uint32_t l1_addr) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t i = 0; i < STICK_NBYTES / sizeof(uint32_t); ++i) {
        dst[i] = 0;
    }
}

struct CornerGeometry {
    int32_t y0[TILE_ROWS];
    int32_t x0[TILE_ROWS];
    float coeff[CORNERS][TILE_ROWS];  // mask * bilinear weight per corner, 0 when the corner is outside
};

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t offset_addr = get_arg_val<uint32_t>(1);
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    const auto input_acc = TensorAccessor(input_args, input_addr, input_stick_nbytes);
    const auto offset_acc = TensorAccessor(offset_args, offset_addr, offset_stick_nbytes);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_stick_nbytes);

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
        const uint32_t v_rows = num_rows - row0 < TILE_ROWS ? num_rows - row0 : TILE_ROWS;

        for (uint32_t r = 0; r < v_rows; ++r) {
            CoreLocalMem<uint32_t> off_dst(offset_l1 + r * offset_stick_nbytes);
            noc.async_read(offset_acc, off_dst, offset_stick_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
            CoreLocalMem<uint32_t> mask_dst(mask_l1 + r * mask_stick_nbytes);
            noc.async_read(mask_acc, mask_dst, mask_stick_nbytes, {.page_id = row0 + r}, {.offset_bytes = 0});
        }

        // (b, h_out, w_out) of the block's first row, then stepped row by row.
        uint32_t b = row0 / (OUT_H * OUT_W);
        uint32_t h_o = (row0 / OUT_W) % OUT_H;
        uint32_t w_o = row0 % OUT_W;
        for (uint32_t r = 0; r < v_rows; ++r) {
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

        for (uint32_t kk = 0; kk < KK; ++kk) {
            const int32_t k_y = static_cast<int32_t>(kk / KERNEL_W) * DILATION_H;
            const int32_t k_x = static_cast<int32_t>(kk % KERNEL_W) * DILATION_W;

            for (uint32_t r = 0; r < v_rows; ++r) {
                CoreLocalMem<volatile uint16_t> off(offset_l1 + r * offset_stick_nbytes);
                CoreLocalMem<volatile uint16_t> msk(mask_l1 + r * mask_stick_nbytes);
                const float y = static_cast<float>(base_y[r] + k_y) + bf16_to_float(off[2 * kk]);
                const float x = static_cast<float>(base_x[r] + k_x) + bf16_to_float(off[2 * kk + 1]);
                const float m = bf16_to_float(msk[kk]);

                const int32_t y0 = static_cast<int32_t>(std::floor(y));
                const int32_t x0 = static_cast<int32_t>(std::floor(x));
                const float ly = y - static_cast<float>(y0);
                const float lx = x - static_cast<float>(x0);
                const bool y0_in = y0 >= 0 && y0 < static_cast<int32_t>(IN_H);
                const bool y1_in = y0 + 1 >= 0 && y0 + 1 < static_cast<int32_t>(IN_H);
                const bool x0_in = x0 >= 0 && x0 < static_cast<int32_t>(IN_W);
                const bool x1_in = x0 + 1 >= 0 && x0 + 1 < static_cast<int32_t>(IN_W);

                geom.y0[r] = y0;
                geom.x0[r] = x0;
                geom.coeff[0][r] = (y0_in && x0_in) ? m * (1.0f - ly) * (1.0f - lx) : 0.0f;
                geom.coeff[1][r] = (y0_in && x1_in) ? m * (1.0f - ly) * lx : 0.0f;
                geom.coeff[2][r] = (y1_in && x0_in) ? m * ly * (1.0f - lx) : 0.0f;
                geom.coeff[3][r] = (y1_in && x1_in) ? m * ly * lx : 0.0f;
            }

            corner_block_cb.reserve_back(CORNERS * CHANNEL_TILES);
            const uint32_t block_l1 = corner_block_cb.get_write_ptr();
            for (uint32_t c = 0; c < CORNERS; ++c) {
                const int32_t dy = c >> 1;
                const int32_t dx = c & 1;
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t dst_l1 = block_l1 + r * BLOCK_ROW_NBYTES + c * STICK_NBYTES;
                    if (r >= v_rows || geom.coeff[c][r] == 0.0f) {
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
                    lane[0] = r < v_rows ? float_to_bf16(geom.coeff[c][r]) : 0;
                }
                scalar_tile_cb.push_back(1);
            }
        }
    }
}
