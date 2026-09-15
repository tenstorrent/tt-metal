// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared reader body for fused multi-scale deformable attention.
//
// V1 (reader_msda_v1.cpp) and V2 (reader_msda_v2.cpp) differ in exactly one
// step: how a normalized (x, y) sampling location for (row, level, point) is
// obtained. Everything else — staging, geometry, boundary handling, the NoC
// gather of the four bilinear neighbours, the tile scatter and the scalar-tile
// contract with the compute kernel — lives here and is compiled once per reader.
//
// The sample stream produced for the compute kernel
// ------------------------------------------------
// One output block carries up to 32 queries of a single (batch, head), packed
// vertically into N_D_TILES tiles laid side by side (32 value-channels each).
// For that block the reader emits 4 * L * P (input_tiles, scalar_tile) pairs —
// one per (level, point, bilinear corner). The compute kernel evaluates
//   dest[row, col] = input[row, col] * scalar[row, 0]        (COL broadcast)
// and accumulates in L1, so the bilinear blend never reaches compute: the
// reader folds the corner coefficient into the scalar it is already sending.
//
//   scalar[row] = attention_weight * corner_coefficient
//
// where corner_coefficient is one of (1-dx)(1-dy), dx(1-dy), (1-dx)dy, dx dy.
// Summing the four corners of a point therefore *is* the bilinear
// interpolation, and summing over (level, point) is the MSDA reduction — one
// accumulator, no intermediate sampled-value tensor.
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B) is documented in
// ../msda_tile_layout.hpp.
//
// Zero-fill contract (relied on by compute_msda.cpp):
//   * scalar tile: col 0 of TL/BL is written for all 32 rows on every emission.
//     Rows >= v_rows and rows whose corner is out of bounds get bf16 0. Other
//     lanes are never written — mul_tiles_bcast<COL> clears DST on entry and
//     only col 0 broadcasts.
//   * input tile: only rows that are in range AND in bounds are written. Stale
//     bytes in the other rows are harmless precisely because their scalar lane
//     is 0.
//
// Runtime-arg layout (identical for both readers):
//   [0]                 value buffer address
//   [1]                 attention_weights buffer address
//   [2]                 sampling_locations (V1) / sampling_offsets (V2) address
//   [3]                 reference_points address (V2; 0 for V1)
//   [4]                 num_output_tiles
//   [5 .. 5+3L)         per level l: H_l, W_l, level_start_index_l
//   [5+3L ...]          per output tile: b, head, q_start, v_rows

// NOTE: kernels are JIT-compiled as C++17. No abbreviated function templates
// (`const auto&` parameters), no concepts — spell template parameters out.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"

namespace fused_msda {

// Byte-identical to the helpers in grid_sample_reader_common.hpp and the
// sibling MSDA reader; duplicated to keep the kernel dependency-free.
// TODO(#45742): consolidate the per-op copies into one shared kernel header.
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

}  // namespace fused_msda

// ---------------------------------------------------------------------------
// Compile-time args (same indices for both readers; the program factory emits
// them in this order and appends the TensorAccessor args from index 24).
// ---------------------------------------------------------------------------
constexpr uint32_t value_scratch_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t attn_scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t loc_scratch_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t input_tile_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t scalar_tile_cb_index = get_compile_time_arg_val(4);
[[maybe_unused]] constexpr uint32_t ref_scratch_cb_index = get_compile_time_arg_val(5);

constexpr uint32_t D = get_compile_time_arg_val(6);
constexpr uint32_t Q = get_compile_time_arg_val(7);
constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(8);
constexpr uint32_t NUM_LEVELS = get_compile_time_arg_val(9);
constexpr uint32_t NUM_POINTS = get_compile_time_arg_val(10);
constexpr uint32_t NUM_KEYS = get_compile_time_arg_val(11);                   // S
[[maybe_unused]] constexpr uint32_t NUM_REFS = get_compile_time_arg_val(12);  // R (V2; 0 for V1)

constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(13);
constexpr uint32_t attn_stick_nbytes = get_compile_time_arg_val(14);
constexpr uint32_t loc_stick_nbytes = get_compile_time_arg_val(15);
[[maybe_unused]] constexpr uint32_t ref_stick_nbytes = get_compile_time_arg_val(16);

constexpr bool ALIGN_CORNERS = get_compile_time_arg_val(17) != 0;
constexpr bool LOC_IN_GRID_SPACE = get_compile_time_arg_val(18) != 0;
constexpr bool LOC_PACKED = get_compile_time_arg_val(19) != 0;
constexpr bool ATTN_PACKED = get_compile_time_arg_val(20) != 0;
[[maybe_unused]] constexpr uint32_t REF_MODE = get_compile_time_arg_val(21);  // 0 = level, 1 = pillar
constexpr bool VALUE_PACKED = get_compile_time_arg_val(22) != 0;
constexpr uint32_t value_page_nbytes = get_compile_time_arg_val(23);

constexpr uint32_t MSDA_TENSOR_ACCESSOR_ARG_BASE = 24;

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------
constexpr uint32_t TILE_MAX_ROWS = fused_msda_tile_layout::TILE_MAX_ROWS;
constexpr uint32_t TILE_NBYTES = fused_msda_tile_layout::TILE_NBYTES;
constexpr uint32_t HALF_STICK_NBYTES = 32;  // one face-row half: 16 bf16
constexpr uint32_t HALF_WORDS = HALF_STICK_NBYTES / sizeof(uint32_t);
constexpr uint32_t WORDS_PER_TILE_ROW = 2 * HALF_WORDS;

// A value stick carries D bf16 (= D/2 uint32 words) and spans N_D_TILES tiles
// side by side; the trailing tile is half-filled when D % 32 == 16. Derived
// from D, not from value_stick_nbytes, which is alignment-padded.
constexpr uint32_t STICK_WORDS = D / 2;
constexpr uint32_t N_D_TILES = (STICK_WORDS + WORDS_PER_TILE_ROW - 1) / WORDS_PER_TILE_ROW;
static_assert(D % 16 == 0 && D > 0, "D must be a positive multiple of 16");

// Staging fan-out per query row.
constexpr uint32_t ATTN_STICKS_PER_ROW = ATTN_PACKED ? 1u : NUM_LEVELS;
constexpr uint32_t LOC_STICKS_PER_ROW = LOC_PACKED ? 1u : (NUM_LEVELS * NUM_POINTS);

// Cap on the per-level geometry table held in the reader's stack frame. Must
// match MAX_LEVELS in fused_msda_device_operation.cpp, which validates against it.
constexpr uint32_t MSDA_MAX_LEVELS = 8;

namespace fused_msda {

struct LevelGeom {
    uint32_t height;       // H_l
    uint32_t width;        // W_l
    uint32_t start_index;  // sum_{k<l} H_k * W_k
    float inv_width;       // 1 / W_l, precomputed: the RISC has no FP divide
    float inv_height;      // 1 / H_l
};

// Normalized (x, y) -> continuous pixel coordinates on level `g`.
// See README.md §3 for the four (LOC_IN_GRID_SPACE, ALIGN_CORNERS) cases.
inline void to_pixel(float x, float y, const LevelGeom& g, float& px, float& py) {
    const float w = static_cast<float>(g.width);
    const float h = static_cast<float>(g.height);
    if constexpr (LOC_IN_GRID_SPACE) {
        if constexpr (ALIGN_CORNERS) {
            px = (x + 1.0f) * 0.5f * (w - 1.0f);
            py = (y + 1.0f) * 0.5f * (h - 1.0f);
        } else {
            px = (x + 1.0f) * 0.5f * w - 0.5f;
            py = (y + 1.0f) * 0.5f * h - 0.5f;
        }
    } else {
        if constexpr (ALIGN_CORNERS) {
            px = x * (w - 1.0f);
            py = y * (h - 1.0f);
        } else {
            px = x * w - 0.5f;
            py = y * h - 0.5f;
        }
    }
}

// Per-row geometry for one (level, point), reused across the four corners.
struct RowGeometry {
    float attn_weight[TILE_MAX_ROWS];
    int32_t x0[TILE_MAX_ROWS];
    int32_t y0[TILE_MAX_ROWS];
    bool x0_valid[TILE_MAX_ROWS];
    bool x1_valid[TILE_MAX_ROWS];
    bool y0_valid[TILE_MAX_ROWS];
    bool y1_valid[TILE_MAX_ROWS];
    float w_nw[TILE_MAX_ROWS];
    float w_ne[TILE_MAX_ROWS];
    float w_sw[TILE_MAX_ROWS];
    float w_se[TILE_MAX_ROWS];
};

// Reads attention_weights[.., l, p] for row r out of the staged arena.
inline float staged_attn(uint32_t attn_arena_l1, uint32_t r, uint32_t l, uint32_t p) {
    const uint32_t stick = ATTN_PACKED ? 0u : l;
    const uint32_t elem = ATTN_PACKED ? (l * NUM_POINTS + p) : p;
    CoreLocalMem<volatile uint16_t> ptr(attn_arena_l1 + (r * ATTN_STICKS_PER_ROW + stick) * attn_stick_nbytes);
    return bf16_to_float(ptr[elem]);
}

// L1 byte address of the staged (x, y) pair for row r, level l, point p, in an
// arena laid out with LOC_STICKS_PER_ROW sticks per row.
inline uint32_t staged_loc_addr(uint32_t loc_arena_l1, uint32_t r, uint32_t l, uint32_t p) {
    if constexpr (LOC_PACKED) {
        return loc_arena_l1 + r * loc_stick_nbytes + (l * NUM_POINTS + p) * 2u * sizeof(uint16_t);
    } else {
        return loc_arena_l1 + (r * LOC_STICKS_PER_ROW + (l * NUM_POINTS + p)) * loc_stick_nbytes;
    }
}

// Issues the NoC reads that stage attention_weights for one output block. No
// barrier: the caller batches it with the location source's own staging.
template <typename AttnAccessor>
inline void stage_attention_weights(
    Noc& noc,
    const AttnAccessor& attn_acc,
    uint32_t attn_arena_l1,
    uint32_t b,
    uint32_t q_start,
    uint32_t head,
    uint32_t v_rows) {
    for (uint32_t r = 0; r < v_rows; ++r) {
        const uint32_t q = q_start + r;
        const uint32_t bqh = (b * Q + q) * NUM_HEADS + head;
        for (uint32_t s = 0; s < ATTN_STICKS_PER_ROW; ++s) {
            const uint32_t page = ATTN_PACKED ? bqh : (bqh * NUM_LEVELS + s);
            CoreLocalMem<uint32_t> dst(attn_arena_l1 + (r * ATTN_STICKS_PER_ROW + s) * attn_stick_nbytes);
            noc.async_read(attn_acc, dst, attn_stick_nbytes, {.page_id = page}, {.offset_bytes = 0});
        }
    }
}

// Issues the NoC reads that stage a (B, Q, H, L, P, 2)- or (B, Q, H, L*P*2)-shaped
// location-like tensor (V1 locations, V2 offsets) for one output block.
template <typename LocAccessor>
inline void stage_location_tensor(
    Noc& noc,
    const LocAccessor& loc_acc,
    uint32_t loc_arena_l1,
    uint32_t b,
    uint32_t q_start,
    uint32_t head,
    uint32_t v_rows) {
    for (uint32_t r = 0; r < v_rows; ++r) {
        const uint32_t q = q_start + r;
        const uint32_t bqh = (b * Q + q) * NUM_HEADS + head;
        for (uint32_t s = 0; s < LOC_STICKS_PER_ROW; ++s) {
            const uint32_t page = LOC_PACKED ? bqh : (bqh * NUM_LEVELS * NUM_POINTS + s);
            CoreLocalMem<uint32_t> dst(loc_arena_l1 + (r * LOC_STICKS_PER_ROW + s) * loc_stick_nbytes);
            noc.async_read(loc_acc, dst, loc_stick_nbytes, {.page_id = page}, {.offset_bytes = 0});
        }
    }
}

// ---------------------------------------------------------------------------
// The shared reader body.
//
// `loc_src` supplies the only V1/V2-specific step:
//     void stage(Noc&, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows);
//     void location(uint32_t r, uint32_t l, uint32_t p, const LevelGeom&, float& x, float& y) const;
// `stage` issues NoC reads without barriering; this function barriers once for
// everything the block needs.
// ---------------------------------------------------------------------------
template <typename ValueAccessor, typename AttnAccessor, typename LocSrc>
inline void reader_main(const ValueAccessor& value_acc, const AttnAccessor& attn_acc, LocSrc& loc_src) {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(4);

    LevelGeom levels[MSDA_MAX_LEVELS];
    for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
        const uint32_t h = get_arg_val<uint32_t>(5 + 3 * l);
        const uint32_t w = get_arg_val<uint32_t>(5 + 3 * l + 1);
        levels[l].height = h;
        levels[l].width = w;
        levels[l].start_index = get_arg_val<uint32_t>(5 + 3 * l + 2);
        levels[l].inv_width = 1.0f / static_cast<float>(w);
        levels[l].inv_height = 1.0f / static_cast<float>(h);
    }

    Noc noc;
    CircularBuffer value_scratch_cb(value_scratch_cb_index);
    CircularBuffer attn_scratch_cb(attn_scratch_cb_index);
    CircularBuffer input_tile_cb(input_tile_cb_index);
    CircularBuffer scalar_tile_cb(scalar_tile_cb_index);

    // Scratch CBs are reserved once and used as fixed linear L1 arenas; they
    // are never pushed, so nothing downstream waits on them.
    value_scratch_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t value_arena_l1 = value_scratch_cb.get_write_ptr();
    attn_scratch_cb.reserve_back(TILE_MAX_ROWS * ATTN_STICKS_PER_ROW);
    const uint32_t attn_arena_l1 = attn_scratch_cb.get_write_ptr();

    RowGeometry geom;

    uint32_t arg_idx = 5 + 3 * NUM_LEVELS;
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const uint32_t b = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t head = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t q_start = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);

        stage_attention_weights(noc, attn_acc, attn_arena_l1, b, q_start, head, v_rows);
        loc_src.stage(noc, b, q_start, head, v_rows);
        noc.async_read_barrier();

        // Base page of this (batch, head) pair in the (B, S, H, D) value tensor.
        const uint32_t value_batch_base = b * NUM_KEYS;

        for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
            const LevelGeom& g = levels[l];
            const int32_t w_i = static_cast<int32_t>(g.width);
            const int32_t h_i = static_cast<int32_t>(g.height);

            for (uint32_t p = 0; p < NUM_POINTS; ++p) {
                for (uint32_t r = 0; r < v_rows; ++r) {
                    float x, y;
                    loc_src.location(r, l, p, g, x, y);
                    geom.attn_weight[r] = staged_attn(attn_arena_l1, r, l, p);

                    float px, py;
                    to_pixel(x, y, g, px, py);

                    const int32_t x0 = static_cast<int32_t>(std::floor(px));
                    const int32_t y0 = static_cast<int32_t>(std::floor(py));
                    const float dx = px - static_cast<float>(x0);
                    const float dy = py - static_cast<float>(y0);

                    geom.x0[r] = x0;
                    geom.y0[r] = y0;
                    geom.x0_valid[r] = (x0 >= 0) && (x0 < w_i);
                    geom.x1_valid[r] = (x0 + 1 >= 0) && (x0 + 1 < w_i);
                    geom.y0_valid[r] = (y0 >= 0) && (y0 < h_i);
                    geom.y1_valid[r] = (y0 + 1 >= 0) && (y0 + 1 < h_i);
                    geom.w_nw[r] = (1.0f - dx) * (1.0f - dy);
                    geom.w_ne[r] = dx * (1.0f - dy);
                    geom.w_sw[r] = (1.0f - dx) * dy;
                    geom.w_se[r] = dx * dy;
                }

                for (uint32_t c = 0; c < 4; ++c) {
                    // Hoist every c-invariant selector: c picks the (dy, dx)
                    // step to the corner, which validity arrays gate it, and
                    // which bilinear coefficient it carries.
                    const int32_t dy_off = (c < 2) ? 0 : 1;
                    const int32_t dx_off = (c & 1) ? 1 : 0;
                    const bool* yv = (c < 2) ? geom.y0_valid : geom.y1_valid;
                    const bool* xv = (c & 1) ? geom.x1_valid : geom.x0_valid;
                    const float* w_corner = (c == 0)   ? geom.w_nw
                                            : (c == 1) ? geom.w_ne
                                            : (c == 2) ? geom.w_sw
                                                       : geom.w_se;

                    // ---- INPUT TILES ----
                    input_tile_cb.reserve_back(N_D_TILES);
                    const uint32_t tile_l1 = input_tile_cb.get_write_ptr();

                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (!(yv[r] && xv[r])) {
                            continue;
                        }
                        const uint32_t cy = static_cast<uint32_t>(geom.y0[r] + dy_off);
                        const uint32_t cx = static_cast<uint32_t>(geom.x0[r] + dx_off);
                        // Canonical (B, S, H, D): one page per (b, s, h).
                        // Packed (B, S, H*D): one page per (b, s), head at byte offset h*D*2.
                        const uint32_t s = g.start_index + cy * g.width + cx;
                        uint32_t page;
                        uint32_t offset_bytes;
                        if constexpr (VALUE_PACKED) {
                            page = value_batch_base + s;
                            offset_bytes = head * (D * 2u);
                        } else {
                            page = (value_batch_base + s) * NUM_HEADS + head;
                            offset_bytes = 0;
                        }
                        CoreLocalMem<uint32_t> dst(value_arena_l1 + r * value_stick_nbytes);
                        // Both page_id and offset_bytes belong to the *source* pack: async_read is
                        // (src, dst, size, src_args, dst_args), so an offset in the 5th argument
                        // would shift the L1 destination instead of the DRAM source page.
                        noc.async_read(
                            value_acc,
                            dst,
                            value_stick_nbytes,
                            {.page_id = page, .offset_bytes = offset_bytes},
                            {.offset_bytes = 0});
                    }
                    noc.async_read_barrier();

                    // Scatter each staged stick across the N_D_TILES face rows.
                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (!(yv[r] && xv[r])) {
                            continue;
                        }
                        const auto off = fused_msda_tile_layout::tile_row_offsets(r);
                        CoreLocalMem<volatile uint32_t> src(value_arena_l1 + r * value_stick_nbytes);
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
                                dl[i] = src[base + i];
                            }
                            for (uint32_t i = 0; i < hi_words; ++i) {
                                dh[i] = src[base + HALF_WORDS + i];
                            }
                        }
                    }
                    input_tile_cb.push_back(N_D_TILES);

                    // ---- SCALAR TILE ----
                    // Only col 0 of TL/BL is read by mul_tiles_bcast<COL>, so
                    // 32 bf16 lanes are written instead of a 2 KiB zero-fill.
                    scalar_tile_cb.reserve_back(1);
                    const uint32_t scalar_l1 = scalar_tile_cb.get_write_ptr();
                    for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                        uint16_t bf = 0;
                        if (r < v_rows && yv[r] && xv[r]) {
                            bf = float_to_bf16(geom.attn_weight[r] * w_corner[r]);
                        }
                        // Rows >= v_rows and out-of-bounds corners must be
                        // written as 0, not skipped: the CB slot may still hold
                        // a nonzero lane from an earlier emission.
                        CoreLocalMem<volatile uint16_t> lane(scalar_l1 + fused_msda_tile_layout::tile_col0_offset(r));
                        lane[0] = bf;
                    }
                    scalar_tile_cb.push_back(1);
                }
            }
        }
    }
}

}  // namespace fused_msda
