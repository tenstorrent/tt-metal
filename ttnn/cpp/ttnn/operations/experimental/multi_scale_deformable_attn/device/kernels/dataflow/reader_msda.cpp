// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for fused multi-scale deformable attention.
//
// Produces (input_tile, scalar_tile) pairs per (d_tile, p, corner) per output
// query-group, where one query-group carries up to 32 queries (all from the
// same batch index n). The compute kernel multiplies them via
// mul_tiles_bcast<COL>: result[h, w] = input[h, w] * scalar[h, 0], so row h
// must hold query h's value-stick chunk and scalar col-0 row h must hold that
// query's combined weight (attn * bilinear_corner).
//
// D (value last dim) is packed into 32-wide tile columns. When D > 32 the
// reader emits one tile per 32-wide chunk (d_tile), reading each chunk via
// the NoC byte offset into the stick. When D < 32 (e.g. D=16) a single
// partial tile is emitted and only the occupied face-halves are filled.
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
// Per-tile runtime args (3 per query-group): (n, q_start, v_rows).
// 1 ≤ v_rows ≤ 32.
// Zero-fill contract:
//   * scalar tile: col 0 is explicitly written for all 32 rows. Tail rows
//     (r ≥ v_rows) and OOB-corner rows get bf16 0, so their contribution
//     zeroes out at the multiply.
//   * input tile: only valid rows (r < v_rows AND corner in-bounds) are
//     written; tail/OOB rows are left as whatever the CB slot held from a
//     previous iter. That's safe because the matching scalar lane is 0, so
//     stale bytes contribute 0 to the accumulator.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/device/kernels/msda_tile_layout.hpp"

namespace {

// Byte-identical to the `bfloat16_to_float` / `float_to_bfloat16` helpers in
// ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp
// and a handful of other reader kernels; duplicated here to keep the kernel
// dependency-free.
// TODO(#45742): consolidate these per-op copies into one shared kernel header.
inline float bf16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

// Scatter `nbytes` from a stick chunk into tile row `r`, walking face-halves
// (16 bf16 = 32 B each). For a full 32-wide chunk this fills lo then hi; for
// D=16 it fills only lo.
inline void copy_stick_chunk_to_tile_row(
    uint32_t tile_l1, uint32_t r, uint32_t stick_chunk_l1, uint32_t nbytes) {
    constexpr uint32_t FACE_ROW_NBYTES = msda_tile_layout::WITHIN_FACE_ROW_STRIDE;
    constexpr uint32_t FACE_ROW_WORDS = FACE_ROW_NBYTES / sizeof(uint32_t);

    const uint32_t num_halves = nbytes / FACE_ROW_NBYTES;
    const auto off = msda_tile_layout::tile_row_offsets(r);
    CoreLocalMem<volatile uint32_t> src(stick_chunk_l1);

    for (uint32_t h = 0; h < num_halves; ++h) {
        const uint32_t dst_off = (h == 0) ? off.lo : off.hi;
        CoreLocalMem<volatile uint32_t> dst(tile_l1 + dst_off);
        const uint32_t src_word = h * FACE_ROW_WORDS;
        for (uint32_t i = 0; i < FACE_ROW_WORDS; ++i) {
            dst[i] = src[src_word + i];
        }
    }
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

constexpr auto value_args = TensorAccessorArgs<14>();
constexpr auto grid_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto attn_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_MAX_ROWS = 32;
constexpr uint32_t ELEMENT_SIZE = 2;  // bf16
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t NUM_D_TILES = (D + TILE_WIDTH - 1) / TILE_WIDTH;

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
    float w_attn_arr[TILE_MAX_ROWS];
    int32_t x0_arr[TILE_MAX_ROWS];
    int32_t y0_arr[TILE_MAX_ROWS];
    bool x0v_arr[TILE_MAX_ROWS];
    bool x1v_arr[TILE_MAX_ROWS];
    bool y0v_arr[TILE_MAX_ROWS];
    bool y1v_arr[TILE_MAX_ROWS];
    float w_nw_arr[TILE_MAX_ROWS];
    float w_ne_arr[TILE_MAX_ROWS];
    float w_sw_arr[TILE_MAX_ROWS];
    float w_se_arr[TILE_MAX_ROWS];

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

        // d_tile is outermost over the reduction so compute can accumulate one
        // width-tile of D at a time (D=64 => two face-pair tiles per group).
        for (uint32_t d_tile = 0; d_tile < NUM_D_TILES; ++d_tile) {
            const uint32_t d_start = d_tile * TILE_WIDTH;
            const uint32_t d_chunk = (D - d_start) < TILE_WIDTH ? (D - d_start) : TILE_WIDTH;
            const uint32_t chunk_nbytes = d_chunk * ELEMENT_SIZE;
            const uint32_t chunk_offset = d_start * ELEMENT_SIZE;

            for (uint32_t p = 0; p < P; ++p) {
                // Precompute per-row geometry for this p (independent of d_tile).
                // Recomputed per d_tile; cheap vs NoC and keeps nesting simple.
                for (uint32_t r = 0; r < v_rows; ++r) {
                    CoreLocalMem<volatile uint16_t> grid_ptr(grid_scratch_l1 + (r * P + p) * grid_stick_nbytes);
                    CoreLocalMem<volatile uint16_t> attn_ptr(attn_scratch_l1 + r * attn_stick_nbytes);

                    const float gx = bf16_to_float(grid_ptr[0]);
                    const float gy = bf16_to_float(grid_ptr[1]);
                    w_attn_arr[r] = bf16_to_float(attn_ptr[p]);

                    // align_corners selects the pixel-coord mapping (mmcv default
                    // is false: pixel = (g+1)*size/2 - 0.5; true variant uses
                    // pixel = (g+1)*(size-1)/2).
                    float px, py;
                    if constexpr (ALIGN_CORNERS) {
                        px = (gx + 1.0f) * 0.5f * static_cast<float>(w_in_i - 1);
                        py = (gy + 1.0f) * 0.5f * static_cast<float>(h_in_i - 1);
                    } else {
                        px = (gx + 1.0f) * 0.5f * static_cast<float>(w_in_i) - 0.5f;
                        py = (gy + 1.0f) * 0.5f * static_cast<float>(h_in_i) - 0.5f;
                    }

                    const int32_t x0 = static_cast<int32_t>(std::floor(px));
                    const int32_t y0 = static_cast<int32_t>(std::floor(py));
                    const float dx = px - static_cast<float>(x0);
                    const float dy = py - static_cast<float>(y0);

                    x0_arr[r] = x0;
                    y0_arr[r] = y0;
                    x0v_arr[r] = (x0 >= 0) && (x0 < w_in_i);
                    x1v_arr[r] = (x0 + 1 >= 0) && (x0 + 1 < w_in_i);
                    y0v_arr[r] = (y0 >= 0) && (y0 < h_in_i);
                    y1v_arr[r] = (y0 + 1 >= 0) && (y0 + 1 < h_in_i);
                    w_nw_arr[r] = (1.0f - dx) * (1.0f - dy);
                    w_ne_arr[r] = dx * (1.0f - dy);
                    w_sw_arr[r] = (1.0f - dx) * dy;
                    w_se_arr[r] = dx * dy;
                }

                for (uint32_t c = 0; c < 4; ++c) {
                    // Hoist all c-invariant selectors out of the per-r loops below:
                    // c picks which y/x validity array, which corner-weight array,
                    // and the (dy, dx) offset to the corner.
                    const int32_t dy_off = (c < 2) ? 0 : 1;
                    const int32_t dx_off = (c & 1) ? 1 : 0;
                    const bool* yv_arr = (c < 2) ? y0v_arr : y1v_arr;
                    const bool* xv_arr = (c & 1) ? x1v_arr : x0v_arr;
                    const float* w_corner_arr =
                        (c == 0) ? w_nw_arr : (c == 1) ? w_ne_arr : (c == 2) ? w_sw_arr : w_se_arr;

                    // ---- INPUT TILE ----
                    input_tile_cb.reserve_back(1);
                    const uint32_t tile_l1 = input_tile_cb.get_write_ptr();

                    // Issue NoC reads for all valid rows (one D-chunk per stick).
                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (!(yv_arr[r] && xv_arr[r])) {
                            continue;
                        }
                        const uint32_t cy = static_cast<uint32_t>(y0_arr[r] + dy_off);
                        const uint32_t cx = static_cast<uint32_t>(x0_arr[r] + dx_off);
                        const uint32_t stick_idx = n_off + cy * w_in_i + cx;
                        CoreLocalMem<uint32_t> dst(value_scratch_l1 + r * value_stick_nbytes);
                        noc.async_read(
                            value_acc,
                            dst,
                            chunk_nbytes,
                            {.page_id = stick_idx},
                            {.offset_bytes = chunk_offset});
                    }
                    noc.async_read_barrier();

                    // Scatter stick chunks into face-halves of each row.
                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (!(yv_arr[r] && xv_arr[r])) {
                            continue;
                        }
                        copy_stick_chunk_to_tile_row(
                            tile_l1, r, value_scratch_l1 + r * value_stick_nbytes, chunk_nbytes);
                    }

                    // Tail rows (r ≥ v_rows) and OOB-corner rows are left untouched:
                    // their scalar entry is zero (see scalar tile below), so any stale
                    // bytes in input row r contribute 0 to L1 accumulation.
                    input_tile_cb.push_back(1);

                    // ---- SCALAR TILE ----
                    // Same scalar for every d_tile of this (p, corner): COL bcast
                    // broadcasts one weight across the whole tile width.
                    scalar_tile_cb.reserve_back(1);
                    const uint32_t s_tile_l1 = scalar_tile_cb.get_write_ptr();

                    for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                        uint16_t bf = 0;
                        if (r < v_rows) {
                            const float combined =
                                (yv_arr[r] && xv_arr[r]) ? (w_attn_arr[r] * w_corner_arr[r]) : 0.0f;
                            bf = float_to_bf16(combined);
                        }
                        CoreLocalMem<volatile uint16_t> p16(s_tile_l1 + msda_tile_layout::tile_col0_offset(r));
                        p16[0] = bf;
                    }
                    scalar_tile_cb.push_back(1);
                }
            }
        }
    }
}
