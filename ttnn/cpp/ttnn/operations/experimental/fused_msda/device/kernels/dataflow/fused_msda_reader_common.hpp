// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared reader body for fused multi-scale deformable attention.
//
// V1 (reader_msda_v1.cpp) and V2 (reader_msda_v2.cpp) differ in exactly one
// step: which bf16 operands a (row, level, point) sampling position is built
// from. Everything else — staging, the hand-off of the geometry to the SFPU, the
// NoC gather of the four bilinear neighbours, the tile scatter and the input-tile
// contract with the compute kernel — lives here and is compiled once per reader.
//
// No float arithmetic happens in this file, by design: the dataflow RISC has no
// FPU, so every float operation here would cost ~140 cycles of soft-float
// emulation. The reader moves bf16 *bit patterns* into tiles and decodes the
// floored corner the SFPU hands back with integer shifts.
//
// The operand names differ either side of the boundary: `geom_*` carries what
// the host constants call the *primary* operand (the location for V1, the
// reference point for V2) and `offset_*` the *secondary* one.
//
// The two streams crossing the reader/compute boundary
// ----------------------------------------------------
// One output block carries up to 32 queries of a single (batch, head), packed
// vertically into N_D_TILES tiles laid side by side (32 value-channels each).
// For that block, per sampling point (one (level, point) pair):
//
//   reader -> compute   geom_x, geom_y, attn_tile     column-0 tiles, one per
//                       [+ offset_x, offset_y for V2]  point
//   compute -> reader   x0, y0                        floor(px), floor(py) as
//                                                      bf16 integers
//   reader -> compute   4 x input_tiles               the gathered corners
//   compute -> compute  4 x scalar_tile               attn * corner coefficient
//
// The compute kernel evaluates dest[row, col] = input[row, col] * scalar[row, 0]
// (COL broadcast) and accumulates in L1, so summing the four corners of a point
// *is* the bilinear interpolation and summing over (level, point) is the MSDA
// reduction — one accumulator, no intermediate sampled-value tensor.
//
// Pipelining. The reader pushes point j+1's geometry tiles *before* it waits for
// point j's corners, and the compute kernel solves point j+1's geometry before
// reducing point j. Without that one-point lookahead the two would ping-pong:
// the reader idle through every geometry solve, compute idle through every
// gather. See the CB depths in fused_msda_program_factory.cpp, which are what
// make the lookahead legal.
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B) is documented in
// ../msda_tile_layout.hpp.
//
// Contract with compute_msda.cpp
//   * geom/offset/attn tiles: column 0 is written for all 32 rows on every
//     emission; rows >= v_rows get bf16 0. A zero attn lane is what makes a tail
//     row's scalar zero, so tail rows cost nothing downstream. Columns 1..31 are
//     zeroed once per CB slot at startup so an uninitialised L1 bit pattern
//     never reaches the SFPU as a NaN.
//   * input tile: rows that are in range AND in bounds hold the gathered value
//     stick; every other row is explicitly zeroed. The scalar is built by the
//     compute kernel, which has no bounds information, so nothing downstream can
//     annihilate a stale row.
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

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"

// ---------------------------------------------------------------------------
// Compile-time args (same indices for both readers; the program factory emits
// them in this order and appends the TensorAccessor args from
// MSDA_TENSOR_ACCESSOR_ARG_BASE onwards).
// ---------------------------------------------------------------------------
constexpr uint32_t value_scratch_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t attn_scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t loc_scratch_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t input_tile_cb_index = get_compile_time_arg_val(3);
[[maybe_unused]] constexpr uint32_t ref_scratch_cb_index = get_compile_time_arg_val(4);

constexpr uint32_t D = get_compile_time_arg_val(5);
constexpr uint32_t Q = get_compile_time_arg_val(6);
constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(7);
constexpr uint32_t NUM_LEVELS = get_compile_time_arg_val(8);
constexpr uint32_t NUM_POINTS = get_compile_time_arg_val(9);
constexpr uint32_t NUM_KEYS = get_compile_time_arg_val(10);                   // S
[[maybe_unused]] constexpr uint32_t NUM_REFS = get_compile_time_arg_val(11);  // R (V2; 0 for V1)

constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(12);
constexpr uint32_t attn_stick_nbytes = get_compile_time_arg_val(13);
constexpr uint32_t loc_stick_nbytes = get_compile_time_arg_val(14);
[[maybe_unused]] constexpr uint32_t ref_stick_nbytes = get_compile_time_arg_val(15);

constexpr bool LOC_PACKED = get_compile_time_arg_val(16) != 0;
constexpr bool ATTN_PACKED = get_compile_time_arg_val(17) != 0;
[[maybe_unused]] constexpr uint32_t REF_MODE = get_compile_time_arg_val(18);  // 0 = level, 1 = pillar
constexpr bool VALUE_PACKED = get_compile_time_arg_val(19) != 0;
constexpr uint32_t value_page_nbytes = get_compile_time_arg_val(20);
constexpr bool FROM_OFFSETS = get_compile_time_arg_val(21) != 0;

// Reader <-> compute tile pipes.
constexpr uint32_t geom_x_cb_index = get_compile_time_arg_val(22);
constexpr uint32_t geom_y_cb_index = get_compile_time_arg_val(23);
constexpr uint32_t offset_x_cb_index = get_compile_time_arg_val(24);
constexpr uint32_t offset_y_cb_index = get_compile_time_arg_val(25);
constexpr uint32_t attn_tile_cb_index = get_compile_time_arg_val(26);
constexpr uint32_t x0_cb_index = get_compile_time_arg_val(27);
constexpr uint32_t y0_cb_index = get_compile_time_arg_val(28);
constexpr uint32_t geom_cb_pages = get_compile_time_arg_val(29);
constexpr uint32_t MSDA_LAST_SCALAR_CT_ARG = 29;
constexpr uint32_t MSDA_TENSOR_ACCESSOR_ARG_BASE = MSDA_LAST_SCALAR_CT_ARG + 1;

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------
constexpr uint32_t TILE_MAX_ROWS = fused_msda_tile_layout::TILE_MAX_ROWS;
constexpr uint32_t TILE_NBYTES = fused_msda_tile_layout::TILE_NBYTES;
constexpr uint32_t TILE_WORDS = TILE_NBYTES / sizeof(uint32_t);
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

// Sampling points per output block: the reduction runs over (level, point).
constexpr uint32_t POINTS_PER_BLOCK = NUM_LEVELS * NUM_POINTS;

// Cap on the per-level geometry table held in the reader's stack frame. Must
// match MAX_LEVELS in fused_msda_device_operation.cpp, which validates against it.
constexpr uint32_t MSDA_MAX_LEVELS = 8;

namespace fused_msda {

struct LevelGeom {
    uint32_t height;       // H_l
    uint32_t width;        // W_l
    uint32_t start_index;  // sum_{k<l} H_k * W_k
};

// Decodes a bf16 that holds an exact integer, with shifts only.
//
// The compute kernel floored px on the SFPU, so the value is integral by
// construction, and going through float here would put soft-float on a core
// that has no FPU. bf16 carries 8 significant bits, so every integer up to 256
// is exact; `derive_shapes` in fused_msda_device_operation.cpp rejects feature
// maps larger than that, which is what makes an in-bounds corner's index
// exact.
//
// Anything too large to decode (including inf and NaN, whose exponent field is
// 0xFF) is mapped to a magnitude no feature map can reach, so it fails the
// bounds test rather than aliasing into it through a wrapped shift.
constexpr int32_t BF16_INT_OUT_OF_RANGE = 1 << 24;

inline int32_t bf16_exact_int(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    const bool negative = (bits & 0x80000000u) != 0;
    if (exp < 0) {
        return 0;  // |v| < 1, and v is integral, so v == 0
    }
    if (exp > 23) {
        return negative ? -BF16_INT_OUT_OF_RANGE : BF16_INT_OUT_OF_RANGE;
    }
    const uint32_t mant = (bits & 0x7FFFFFu) | 0x800000u;
    const int32_t m = static_cast<int32_t>(mant >> (23 - exp));
    return negative ? -m : m;
}

// Per-row corner indices and bounds flags for one (level, point), reused across
// the four corners.
struct RowGeometry {
    int32_t x0[TILE_MAX_ROWS];
    int32_t y0[TILE_MAX_ROWS];
    bool x0_valid[TILE_MAX_ROWS];
    bool x1_valid[TILE_MAX_ROWS];
    bool y0_valid[TILE_MAX_ROWS];
    bool y1_valid[TILE_MAX_ROWS];
};

// Reads the raw bf16 attention_weights[.., l, p] for row r out of the staged arena.
inline uint16_t staged_attn_bits(uint32_t attn_arena_l1, uint32_t r, uint32_t l, uint32_t p) {
    const uint32_t stick = ATTN_PACKED ? 0u : l;
    const uint32_t elem = ATTN_PACKED ? (l * NUM_POINTS + p) : p;
    CoreLocalMem<volatile uint16_t> ptr(attn_arena_l1 + (r * ATTN_STICKS_PER_ROW + stick) * attn_stick_nbytes);
    return ptr[elem];
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

// Zeroes every page of a reader-produced tile CB, once, before the block loop.
//
// Only column 0 of these tiles carries a query; the other 31 columns are never
// written again and would otherwise reach the SFPU as whatever bit pattern L1
// happened to hold, which decodes to NaN often enough to matter. The CB is
// reserved but not pushed, so this is a plain write over its L1 extent.
//
// `pages` must be that CB's own depth, and this must run before the first
// push_back on it — the write starts at the write pointer, which is only at
// page 0 while the CB has never been pushed.
inline void zero_tile_cb(uint32_t cb_index, uint32_t pages) {
    CircularBuffer cb(cb_index);
    cb.reserve_back(pages);
    CoreLocalMem<volatile uint32_t> base(cb.get_write_ptr());
    for (uint32_t i = 0; i < pages * TILE_WORDS; ++i) {
        base[i] = 0;
    }
}

// Writes bf16 `value` into column 0 of tile row `r`.
inline void write_col0(uint32_t tile_l1, uint32_t r, uint16_t value) {
    CoreLocalMem<volatile uint16_t> lane(tile_l1 + fused_msda_tile_layout::tile_col0_offset(r));
    lane[0] = value;
}

// Zeroes one row of one input tile group, for a corner the gather skipped.
inline void zero_input_row(uint32_t tile_l1, uint32_t r) {
    const auto off = fused_msda_tile_layout::tile_row_offsets(r);
    for (uint32_t k = 0; k < N_D_TILES; ++k) {
        const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
        CoreLocalMem<volatile uint32_t> lo(ktile_l1 + off.lo);
        CoreLocalMem<volatile uint32_t> hi(ktile_l1 + off.hi);
        for (uint32_t i = 0; i < HALF_WORDS; ++i) {
            lo[i] = 0;
            hi[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// The shared reader body.
//
// `loc_src` supplies the only V1/V2-specific step:
//     void stage(Noc&, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows);
//     void primary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const;
//     void secondary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const;
// Both accessors move bf16 bit patterns, never floats. `secondary` carries the
// raw sampling offset and is required only when FROM_OFFSETS — its call site is
// a discarded statement otherwise, so a V1 source need not provide it. `stage`
// issues NoC reads without barriering; this function barriers once for
// everything the block needs.
// ---------------------------------------------------------------------------
template <typename ValueAccessor, typename AttnAccessor, typename LocSrc>
inline void reader_main(const ValueAccessor& value_acc, const AttnAccessor& attn_acc, LocSrc& loc_src) {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(4);

    LevelGeom levels[MSDA_MAX_LEVELS];
    for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
        levels[l].height = get_arg_val<uint32_t>(5 + 3 * l);
        levels[l].width = get_arg_val<uint32_t>(5 + 3 * l + 1);
        levels[l].start_index = get_arg_val<uint32_t>(5 + 3 * l + 2);
    }

    Noc noc;
    CircularBuffer value_scratch_cb(value_scratch_cb_index);
    CircularBuffer attn_scratch_cb(attn_scratch_cb_index);
    CircularBuffer input_tile_cb(input_tile_cb_index);
    CircularBuffer geom_x_cb(geom_x_cb_index);
    CircularBuffer geom_y_cb(geom_y_cb_index);
    CircularBuffer offset_x_cb(offset_x_cb_index);
    CircularBuffer offset_y_cb(offset_y_cb_index);
    CircularBuffer attn_tile_cb(attn_tile_cb_index);
    CircularBuffer x0_cb(x0_cb_index);
    CircularBuffer y0_cb(y0_cb_index);

    // Scratch CBs are reserved once and used as fixed linear L1 arenas; they
    // are never pushed, so nothing downstream waits on them.
    value_scratch_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t value_arena_l1 = value_scratch_cb.get_write_ptr();
    attn_scratch_cb.reserve_back(TILE_MAX_ROWS * ATTN_STICKS_PER_ROW);
    const uint32_t attn_arena_l1 = attn_scratch_cb.get_write_ptr();

    // All five operand pipes are allocated kGeomCbPages deep by the factory,
    // which is what geom_cb_pages carries.
    zero_tile_cb(geom_x_cb_index, geom_cb_pages);
    zero_tile_cb(geom_y_cb_index, geom_cb_pages);
    zero_tile_cb(attn_tile_cb_index, geom_cb_pages);
    if constexpr (FROM_OFFSETS) {
        zero_tile_cb(offset_x_cb_index, geom_cb_pages);
        zero_tile_cb(offset_y_cb_index, geom_cb_pages);
    }

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

        // Hands one sampling point's operands to the SFPU as column-0 tiles.
        auto push_point_geometry = [&](uint32_t j) {
            const uint32_t l = j / NUM_POINTS;
            const uint32_t p = j - l * NUM_POINTS;

            geom_x_cb.reserve_back(1);
            geom_y_cb.reserve_back(1);
            attn_tile_cb.reserve_back(1);
            const uint32_t gx_l1 = geom_x_cb.get_write_ptr();
            const uint32_t gy_l1 = geom_y_cb.get_write_ptr();
            const uint32_t at_l1 = attn_tile_cb.get_write_ptr();
            uint32_t ox_l1 = 0;
            uint32_t oy_l1 = 0;
            if constexpr (FROM_OFFSETS) {
                offset_x_cb.reserve_back(1);
                offset_y_cb.reserve_back(1);
                ox_l1 = offset_x_cb.get_write_ptr();
                oy_l1 = offset_y_cb.get_write_ptr();
            }

            for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                // Rows past v_rows go out as zero. A zero attn lane makes the
                // scalar zero, which is the same contract the reduction already
                // relies on for a partial trailing tile.
                uint16_t gx = 0;
                uint16_t gy = 0;
                uint16_t av = 0;
                uint16_t ox = 0;
                uint16_t oy = 0;
                if (r < v_rows) {
                    loc_src.primary(r, l, p, gx, gy);
                    av = staged_attn_bits(attn_arena_l1, r, l, p);
                    if constexpr (FROM_OFFSETS) {
                        loc_src.secondary(r, l, p, ox, oy);
                    }
                }
                write_col0(gx_l1, r, gx);
                write_col0(gy_l1, r, gy);
                write_col0(at_l1, r, av);
                if constexpr (FROM_OFFSETS) {
                    write_col0(ox_l1, r, ox);
                    write_col0(oy_l1, r, oy);
                }
            }

            geom_x_cb.push_back(1);
            geom_y_cb.push_back(1);
            attn_tile_cb.push_back(1);
            if constexpr (FROM_OFFSETS) {
                offset_x_cb.push_back(1);
                offset_y_cb.push_back(1);
            }
        };

        // One point of lookahead, matching the compute kernel's. Point 0 goes
        // out before the loop so compute can start solving it while the reader
        // is still preparing point 1.
        push_point_geometry(0);

        for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
            if (j + 1 < POINTS_PER_BLOCK) {
                push_point_geometry(j + 1);
            }

            const uint32_t l = j / NUM_POINTS;
            const LevelGeom& g = levels[l];
            const int32_t w_i = static_cast<int32_t>(g.width);
            const int32_t h_i = static_cast<int32_t>(g.height);

            // Corners solved on the SFPU. Only the decode, the bounds test and
            // the page index stay here, all in integer arithmetic.
            x0_cb.wait_front(1);
            y0_cb.wait_front(1);
            const uint32_t x0_l1 = x0_cb.get_read_ptr();
            const uint32_t y0_l1 = y0_cb.get_read_ptr();
            for (uint32_t r = 0; r < v_rows; ++r) {
                const uint32_t col0 = fused_msda_tile_layout::tile_col0_offset(r);
                CoreLocalMem<volatile uint16_t> x0_src(x0_l1 + col0);
                CoreLocalMem<volatile uint16_t> y0_src(y0_l1 + col0);
                const int32_t x0 = bf16_exact_int(x0_src[0]);
                const int32_t y0 = bf16_exact_int(y0_src[0]);
                geom.x0[r] = x0;
                geom.y0[r] = y0;
                geom.x0_valid[r] = (x0 >= 0) && (x0 < w_i);
                geom.x1_valid[r] = (x0 + 1 >= 0) && (x0 + 1 < w_i);
                geom.y0_valid[r] = (y0 >= 0) && (y0 < h_i);
                geom.y1_valid[r] = (y0 + 1 >= 0) && (y0 + 1 < h_i);
            }
            x0_cb.pop_front(1);
            y0_cb.pop_front(1);

            for (uint32_t c = 0; c < 4; ++c) {
                // Hoist every c-invariant selector: c picks the (dy, dx) step to
                // the corner and which validity arrays gate it.
                //
                // c = 0, 1, 2, 3 is NW, NE, SW, SE, and must stay in lockstep
                // with the four corner_weight calls in msda_geometry.hpp::point
                // that build the matching scalar tiles.
                const int32_t dy_off = (c < 2) ? 0 : 1;
                const int32_t dx_off = (c & 1) ? 1 : 0;
                const bool* yv = (c < 2) ? geom.y0_valid : geom.y1_valid;
                const bool* xv = (c & 1) ? geom.x1_valid : geom.x0_valid;

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

                // Scatter each staged stick across the N_D_TILES face rows, and
                // zero the rows that had no corner to gather.
                //
                // The zeroing is load-bearing: the scalar comes from a compute
                // kernel that cannot know which corners fell outside the feature
                // map, so a stale row would be multiplied by a live weight — a
                // high-error-ratio bug that a PCC gate passes.
                for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                    if (r >= v_rows || !(yv[r] && xv[r])) {
                        zero_input_row(tile_l1, r);
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
            }
        }
    }
}

}  // namespace fused_msda
