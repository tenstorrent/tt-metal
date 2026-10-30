// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared reader body. V1 (reader_msda_v1.cpp) and V2 (reader_msda_v2.cpp)
// differ in which bf16 operands a (row, level, point) sampling position is
// built from, and only V2 can take the tiled geometry (see below), which also
// needs its source to stage offsets into a block and fill the reference tile.
// Everything else — staging, the hand-off of the geometry to compute, the NoC
// gather of the four bilinear neighbours, the tile scatter and the input-tile
// contract with the compute kernel — lives here and is compiled once per
// reader.
//
// No float arithmetic happens here. The dataflow RISC has no FPU, so every
// float operation would cost ~140 cycles of soft-float emulation. This file
// moves bit patterns; compute hands back each corner already bounds-tested and
// indexed as an integer word (msda_corner_word.hpp).
// geom_x/geom_y are float32 when REF_FP32 is set and compute unpacks them
// straight into its 32-bit DST, so the reference point reaches the SFPU
// unrounded; every other tile here is bf16.
// The row-major gather itself is msda_gather_rows.hpp, shared with the writer
// when the gather is split.
//
// The reader pushes point j+1's geometry tiles before it gathers point j.
// Without that lookahead the reader is idle through every geometry solve and
// compute is idle through every gather. Where the push sits depends on the
// gather:
//
//   reader-only gather: push j+1; wait corner(j); gather j.
//   split gather:       wait corner(j); post j to the writer; push j+1; gather
//                       the reader's rows of j.
//
// The split order cannot deadlock: compute produces corner(j) before it waits
// for j+1's tiles and holds at most one point ahead, so the CB depths in
// fused_msda_program_factory.cpp cover both orders.
//
// Tiled geometry (TILED_GEOMETRY, msda_tiled_geometry.hpp) moves the hand-off
// from points to blocks. The reader NoC-reads a block's offsets and attn rows
// into a row-major page and writes the reference tile only when the query
// block changes; compute tilizes them, solves every point at once and returns
// one corner tile per block, point j's word in column 16 + j. Block t+1 is
// staged as soon as block t's corner tile arrives and published after the
// first gather barrier of block t, so compute can solve it early in block t.
//
// Tile face layout is documented in ../msda_tile_layout.hpp. bf16 tiles are
// 2048 B; geom_x/geom_y are 4096 B when REF_FP32 is set (per-point path only).
//
// Contract with compute_msda.cpp
//   * geom/attn tiles: column 0 (and on V2 the offset in column 16 of the geom
//     tiles) is written for all 32 rows on every emission; rows >= v_rows get
//     0 in that tile's format (float32 for geom_x/geom_y when REF_FP32 is set,
//     bf16 otherwise). The other columns are zeroed
//     once per CB slot at startup so an uninitialised L1 bit pattern never
//     reaches the SFPU as a NaN.
//   * corner tiles come back float32 with the corner word in column 16
//     (msda_geometry.hpp).
//   * tiled geometry instead: rows < v_rows of the staged page are NoC-read,
//     the reference tile is rewritten for rows < v_rows when the query block
//     changes, and every other row keeps a previous block's values (zero at
//     startup), so those rows' scalars are not zero. That is harmless: rows
//     never mix, and the writer emits only rows < v_rows.
//   * input tile: rows that are in range AND in bounds hold the gathered value
//     stick; every other row holds zero or an earlier stick (the CB is zeroed
//     once at startup). Compute zeroes the scalar of every corner whose bounds flags
//     are clear, so the gathers skip those corners without writing them.
//   * with RM_STAGING the four corners of a point go over as one row-major
//     block (row r = [NW | NE | SW | SE], D bf16 each) that compute tilizes;
//     the same rule applies per corner slot. With SPLIT_GATHER the writer
//     produces part of the block; the reader pushes it once both are done.
//
// Runtime-arg layout (identical for both readers):
//   [0]                 value buffer address
//   [1]                 attention_weights buffer address
//   [2]                 sampling_locations (V1) / sampling_offsets (V2) address
//   [3]                 reference_points address (V2; 0 for V1)
//   [4]                 num_output_tiles
//   [5 .. 5+3L)         per level l: H_l (unread; compute bounds the corner),
//                       W_l, level_start_index_l
//   [5+3L .. 5+11L)     TILED_GEOMETRY only: per level l, the fp32 bit patterns
//                       of (primary, secondary, bias, extent) for x, then for y
//   [...]               per output tile: b, head, q_start, v_rows

// NOTE: kernels are JIT-compiled as C++17. No abbreviated function templates
// (`const auto&` parameters), no concepts — spell template parameters out.

#pragma once

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_corner_word.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tiled_geometry.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/msda_gather_rows.hpp"

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

struct ReaderGatherCfg {
    static constexpr uint32_t D = ::D;
    static constexpr uint32_t NUM_KEYS = ::NUM_KEYS;
    static constexpr uint32_t NUM_HEADS = ::NUM_HEADS;
    static constexpr bool VALUE_PACKED = ::VALUE_PACKED;
    static constexpr uint32_t STICK_NBYTES = value_stick_nbytes;
};

// Reader <-> compute tile pipes.
constexpr uint32_t geom_x_cb_index = get_compile_time_arg_val(22);
constexpr uint32_t geom_y_cb_index = get_compile_time_arg_val(23);
constexpr uint32_t attn_tile_cb_index = get_compile_time_arg_val(24);
constexpr uint32_t corner_cb_index = get_compile_time_arg_val(25);
constexpr uint32_t geom_cb_pages = get_compile_time_arg_val(26);
constexpr bool RM_STAGING = get_compile_time_arg_val(27) != 0;
constexpr bool SPLIT_GATHER = get_compile_time_arg_val(28) != 0;
constexpr uint32_t gather_ready_sem_id = get_compile_time_arg_val(29);
constexpr uint32_t gather_done_sem_id = get_compile_time_arg_val(30);
constexpr uint32_t gather_mailbox_cb_index = get_compile_time_arg_val(31);
constexpr uint32_t gather_mailbox_nbytes = get_compile_time_arg_val(32);  // allocated page; 0 without SPLIT_GATHER
constexpr bool TILED_GEOMETRY = get_compile_time_arg_val(33) != 0;
constexpr uint32_t geom_rm_cb_index = get_compile_time_arg_val(34);
constexpr uint32_t ref_tile_cb_index = get_compile_time_arg_val(35);
constexpr uint32_t const_cb_index = get_compile_time_arg_val(36);
constexpr uint32_t MSDA_LAST_SCALAR_CT_ARG = 36;
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

static_assert(
    !TILED_GEOMETRY ||
        (FROM_OFFSETS && LOC_PACKED && ATTN_PACKED && POINTS_PER_BLOCK == fused_msda_tiled_geometry::POINTS &&
         NUM_REFS <= fused_msda_tiled_geometry::MAX_REFS &&
         loc_stick_nbytes == fused_msda_tiled_geometry::RM_ATTN_OFFSET &&
         attn_stick_nbytes <= fused_msda_tiled_geometry::RM_ROW_NBYTES - fused_msda_tiled_geometry::RM_ATTN_OFFSET),
    "tiled geometry needs packed V2 operands with L * P == 16 that fit the row-major block");

// Cap on the per-level geometry table held in the reader's stack frame. Must
// match MAX_LEVELS in fused_msda_device_operation.cpp, which validates against it.
constexpr uint32_t MSDA_MAX_LEVELS = 8;

namespace fused_msda {

struct LevelGeom {
    uint32_t width;        // W_l
    uint32_t start_index;  // sum_{k<l} H_k * W_k
};

// Per-row NW corner and bounds flags for one (level, point), reused across the
// four corners.
struct RowGeometry {
    uint32_t nw[TILE_MAX_ROWS];  // (b, s) row of the NW corner
    uint8_t flags[TILE_MAX_ROWS];
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

// Issues the NoC reads that stage attention_weights for one output block, row
// r at attn_arena_l1 + r * row_nbytes. No barrier: the caller batches it with
// the location source's own staging.
template <typename AttnAccessor>
inline void stage_attention_weights(
    Noc& noc,
    const AttnAccessor& attn_acc,
    uint32_t attn_arena_l1,
    uint32_t b,
    uint32_t q_start,
    uint32_t head,
    uint32_t v_rows,
    uint32_t row_nbytes = ATTN_STICKS_PER_ROW * attn_stick_nbytes) {
    for (uint32_t r = 0; r < v_rows; ++r) {
        const uint32_t q = q_start + r;
        const uint32_t bqh = (b * Q + q) * NUM_HEADS + head;
        for (uint32_t s = 0; s < ATTN_STICKS_PER_ROW; ++s) {
            const uint32_t page = ATTN_PACKED ? bqh : (bqh * NUM_LEVELS + s);
            CoreLocalMem<uint32_t> dst(attn_arena_l1 + r * row_nbytes + s * attn_stick_nbytes);
            noc.async_read(attn_acc, dst, attn_stick_nbytes, {.page_id = page}, {.offset_bytes = 0});
        }
    }
}

// Issues the NoC reads that stage a (B, Q, H, L, P, 2)- or (B, Q, H, L*P*2)-shaped
// location-like tensor (V1 locations, V2 offsets) for one output block, row r
// at loc_arena_l1 + r * row_nbytes.
template <typename LocAccessor>
inline void stage_location_tensor(
    Noc& noc,
    const LocAccessor& loc_acc,
    uint32_t loc_arena_l1,
    uint32_t b,
    uint32_t q_start,
    uint32_t head,
    uint32_t v_rows,
    uint32_t row_nbytes = LOC_STICKS_PER_ROW * loc_stick_nbytes) {
    for (uint32_t r = 0; r < v_rows; ++r) {
        const uint32_t q = q_start + r;
        const uint32_t bqh = (b * Q + q) * NUM_HEADS + head;
        for (uint32_t s = 0; s < LOC_STICKS_PER_ROW; ++s) {
            const uint32_t page = LOC_PACKED ? bqh : (bqh * NUM_LEVELS * NUM_POINTS + s);
            CoreLocalMem<uint32_t> dst(loc_arena_l1 + r * row_nbytes + s * loc_stick_nbytes);
            noc.async_read(loc_acc, dst, loc_stick_nbytes, {.page_id = page}, {.offset_bytes = 0});
        }
    }
}

// Zeroes every page of a reader-produced tile CB, once, before the block loop.
//
// Only column 0 of these tiles (and column 16 of geom_x / geom_y on V2) carries
// a query; the other columns are never written again and would otherwise reach
// the SFPU as whatever bit pattern L1 happened to hold, which decodes to NaN often enough to matter. The CB is
// reserved but not pushed, so this is a plain write over its L1 extent.
//
// `pages` must be that CB's own depth, and this must run before the first
// push_back on it — the write starts at the write pointer, which is only at
// page 0 while the CB has never been pushed.
inline void zero_tile_cb(uint32_t cb_index, uint32_t pages, uint32_t tile_nbytes = TILE_NBYTES) {
    CircularBuffer cb(cb_index);
    cb.reserve_back(pages);
    CoreLocalMem<volatile uint32_t> base(cb.get_write_ptr());
    // `tile_nbytes` must be the CB's page size: a float32 geometry tile is
    // 4096 B, and zeroing the bf16 size would leave its upper half as stale L1.
    const uint32_t words = tile_nbytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < pages * words; ++i) {
        base[i] = 0;
    }
}

// Writes bf16 `value` into column 0 of tile row `r`.
inline void write_col0(uint32_t tile_l1, uint32_t r, uint16_t value) {
    CoreLocalMem<volatile uint16_t> lane(tile_l1 + fused_msda_tile_layout::tile_col0_offset(r));
    lane[0] = value;
}

// Writes a float32 bit pattern into column 0 of tile row `r`.
inline void write_col0_fp32(uint32_t tile_l1, uint32_t r, uint32_t value) {
    CoreLocalMem<volatile uint32_t> lane(tile_l1 + fused_msda_tile_layout::tile_col0_offset_fp32(r));
    lane[0] = value;
}

// Column 16 of tile row `r`, the SFPU lane of column 0 one face to the right.
inline void write_col16(uint32_t tile_l1, uint32_t r, uint16_t value) {
    CoreLocalMem<volatile uint16_t> lane(tile_l1 + fused_msda_tile_layout::tile_col16_offset(r));
    lane[0] = value;
}

inline void write_col16_fp32(uint32_t tile_l1, uint32_t r, uint32_t value) {
    CoreLocalMem<volatile uint32_t> lane(tile_l1 + fused_msda_tile_layout::tile_col16_offset_fp32(r));
    lane[0] = value;
}

// Writes bf16 `value` into element (r, c) of a tile.
inline void write_tile_elem(uint32_t tile_l1, uint32_t r, uint32_t c, uint16_t value) {
    CoreLocalMem<volatile uint16_t> elem(tile_l1 + fused_msda_tile_layout::tile_elem_offset(r, c));
    elem[0] = value;
}

constexpr uint16_t BF16_ONE = 0x3F80;

// The reference point r(l, p) that sampling point (l, p) offsets from; see
// reader_msda_v2.cpp.
inline uint32_t ref_index(uint32_t l, uint32_t p) { return (REF_MODE == 0) ? l : (p % NUM_REFS); }

// Readies everything the tiled geometry keeps across blocks: writes and pushes
// the constants CB (M_off, M_ref, M_extent, E_j; see msda_tiled_geometry.hpp),
// and zeroes the reference tile, setting its bias column, and the row-major
// staging page. Staging writes only rows < v_rows; the rest keep what an
// earlier block left, which never reaches the output (see the contract at the
// top of this file). `arg` is the first per-level constant runtime arg, fp32
// bit patterns whose low halves the factory has checked to be zero.
inline void init_tiled_geometry(Noc& noc, uint32_t arg) {
    namespace tg = fused_msda_tiled_geometry;
    CircularBuffer const_cb(const_cb_index);
    CircularBuffer ref_tile_cb(ref_tile_cb_index);
    CircularBuffer geom_rm_cb(geom_rm_cb_index);
    const_cb.reserve_back(tg::CONST_TILES);
    ref_tile_cb.reserve_back(1);
    // The row-major CB has never been pushed, so the write pointer is at page
    // 0 and this covers all of it.
    geom_rm_cb.reserve_back(tg::RM_CB_PAGES);
    noc.async_write_zeros(const_cb, tg::CONST_TILES * TILE_NBYTES);
    noc.async_write_zeros(ref_tile_cb, TILE_NBYTES);
    noc.async_write_zeros(geom_rm_cb, tg::RM_CB_PAGES * TILE_NBYTES);
    noc.write_zeros_l1_barrier();

    const uint32_t m_off = const_cb.get_write_ptr() + tg::CONST_M_OFF * TILE_NBYTES;
    const uint32_t m_ref = const_cb.get_write_ptr() + tg::CONST_M_REF * TILE_NBYTES;
    const uint32_t m_extent = const_cb.get_write_ptr() + tg::CONST_M_EXTENT * TILE_NBYTES;
    for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
        const uint32_t l = j / NUM_POINTS;
        const uint32_t p = j - l * NUM_POINTS;
        const uint32_t k = ref_index(l, p);
        // (primary, secondary, bias, extent) of axis 0 = x, 1 = y.
        auto level_bf16 = [&](uint32_t axis, uint32_t i) {
            return static_cast<uint16_t>(
                get_arg_val<uint32_t>(arg + tg::CONSTANTS_PER_LEVEL * l + tg::CONSTANTS_PER_AXIS * axis + i) >> 16);
        };
        write_tile_elem(m_ref, k, j, level_bf16(0, 0));
        write_tile_elem(m_off, 2 * j, j, level_bf16(0, 1));
        write_tile_elem(m_ref, tg::BIAS_COL, j, level_bf16(0, 2));
        write_tile_elem(m_ref, 16 + k, 16 + j, level_bf16(1, 0));
        write_tile_elem(m_off, 2 * j + 1, 16 + j, level_bf16(1, 1));
        write_tile_elem(m_ref, tg::BIAS_COL, 16 + j, level_bf16(1, 2));
        write_tile_elem(m_extent, tg::BIAS_COL, j, level_bf16(1, 3));
        write_tile_elem(m_extent, tg::BIAS_COL, 16 + j, level_bf16(0, 3));
        write_tile_elem(const_cb.get_write_ptr() + (tg::CONST_E_FIRST + j) * TILE_NBYTES, j, 0, BF16_ONE);
    }
    const_cb.push_back(tg::CONST_TILES);

    for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
        write_tile_elem(ref_tile_cb.get_write_ptr(), r, tg::BIAS_COL, BF16_ONE);
    }
}

// Row-major staging: the four corners of a point share one input-CB block of
// 32 rows, each row [NW | NE | SW | SE] of D bf16, which compute tilizes on the
// unpacker. Only used when D is a whole number of tile widths; the gather
// itself is fused_msda_gather::gather_rows.
static_assert(!RM_STAGING || D % 32 == 0, "row-major staging needs D to be a multiple of the tile width");
static_assert(!SPLIT_GATHER || RM_STAGING, "the split gather is a row-major staging path");
static_assert(
    !SPLIT_GATHER || sizeof(fused_msda_gather::PointArgs) <= gather_mailbox_nbytes,
    "PointArgs no longer fits the gather mailbox page");
static_assert(!RM_STAGING || value_stick_nbytes == D * 2u, "row-major staging packs unpadded D*2-byte sticks");

// The input-tile writes below are plain, not volatile, so the compiler can
// batch the loads and stores: a volatile word copy costs ~14 cycles, and the
// scatter is most of the reader's time. Nothing reads these tiles until the
// push_back that follows, and the caller fences the compiler before it.

// Zeroes row r of one input tile group, for a corner the gather skipped.
inline void zero_input_row(uint32_t tile_l1, uint32_t r) {
    const auto off = fused_msda_tile_layout::tile_row_offsets(r);
#pragma GCC unroll 4
    for (uint32_t k = 0; k < N_D_TILES; ++k) {
        const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
        uint32_t* lo = &CoreLocalMem<uint32_t>(ktile_l1 + off.lo)[0];
        uint32_t* hi = &CoreLocalMem<uint32_t>(ktile_l1 + off.hi)[0];
#pragma GCC unroll 8
        for (uint32_t i = 0; i < HALF_WORDS; ++i) {
            lo[i] = 0;
            hi[i] = 0;
        }
    }
}

// Copies one staged D-wide stick into row r of an input tile group.
inline void scatter_stick(uint32_t tile_l1, uint32_t r, uint32_t src_l1) {
    const auto off = fused_msda_tile_layout::tile_row_offsets(r);
    const uint32_t* src = &CoreLocalMem<uint32_t>(src_l1)[0];
#pragma GCC unroll 4
    for (uint32_t k = 0; k < N_D_TILES; ++k) {
        const uint32_t base = k * WORDS_PER_TILE_ROW;
        const uint32_t words_k = (STICK_WORDS - base < WORDS_PER_TILE_ROW) ? (STICK_WORDS - base) : WORDS_PER_TILE_ROW;
        const uint32_t lo_words = words_k < HALF_WORDS ? words_k : HALF_WORDS;
        const uint32_t hi_words = words_k - lo_words;
        const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
        uint32_t* dl = &CoreLocalMem<uint32_t>(ktile_l1 + off.lo)[0];
        uint32_t* dh = &CoreLocalMem<uint32_t>(ktile_l1 + off.hi)[0];
#pragma GCC unroll 8
        for (uint32_t i = 0; i < lo_words; ++i) {
            dl[i] = src[base + i];
        }
#pragma GCC unroll 8
        for (uint32_t i = 0; i < hi_words; ++i) {
            dh[i] = src[base + HALF_WORDS + i];
        }
    }
}

// ---------------------------------------------------------------------------
// The shared reader body.
//
// `loc_src` supplies the only V1/V2-specific step:
//     void stage(Noc&, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows);
//     void primary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const;
//     void primary(uint32_t r, uint32_t l, uint32_t p, uint32_t& x, uint32_t& y) const;  // REF_FP32
//     void secondary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const;
// Accessors move bit patterns, never values this file interprets. `primary` is
// bf16, or the float32 bit pattern when REF_FP32 is set. `secondary` is always
// bf16 and is required only when FROM_OFFSETS — its call site is a discarded
// statement otherwise, so a V1 source need not provide it. `stage`
// issues NoC reads without barriering; this function barriers once for
// everything the block needs. A source owns its arenas and may skip reads whose
// data they already hold from the previous tile, so nothing here may write to
// them.
//
// With TILED_GEOMETRY the source also provides (and `stage` goes unused):
//     bool stage_tiled(Noc&, b, q_start, head, v_rows, uint32_t off_l1, uint32_t off_row_nbytes);
//         stages the offsets at off_l1 and the reference points into its
//         arena; true when the reference points were re-read
//     void fill_ref_tile(uint32_t tile_l1, uint32_t v_rows) const;
// ---------------------------------------------------------------------------
template <typename ValueAccessor, typename AttnAccessor, typename LocSrc>
inline void reader_main(const ValueAccessor& value_acc, const AttnAccessor& attn_acc, LocSrc& loc_src) {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(4);

    LevelGeom levels[MSDA_MAX_LEVELS];
    for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
        levels[l].width = get_arg_val<uint32_t>(5 + 3 * l + 1);
        levels[l].start_index = get_arg_val<uint32_t>(5 + 3 * l + 2);
    }

    Noc noc;
    CircularBuffer value_scratch_cb(value_scratch_cb_index);
    CircularBuffer attn_scratch_cb(attn_scratch_cb_index);
    CircularBuffer input_tile_cb(input_tile_cb_index);
    CircularBuffer geom_x_cb(geom_x_cb_index);
    CircularBuffer geom_y_cb(geom_y_cb_index);
    CircularBuffer attn_tile_cb(attn_tile_cb_index);
    CircularBuffer corner_cb(corner_cb_index);

    // Scratch CBs are reserved once and used as fixed linear L1 arenas; they
    // are never pushed, so nothing downstream waits on them.
    value_scratch_cb.reserve_back(TILE_MAX_ROWS);
    const uint32_t value_arena_l1 = value_scratch_cb.get_write_ptr();
    [[maybe_unused]] uint32_t attn_arena_l1 = 0;

    uint32_t arg_idx = 5 + 3 * NUM_LEVELS;
    if constexpr (TILED_GEOMETRY) {
        init_tiled_geometry(noc, arg_idx);
        arg_idx += fused_msda_tiled_geometry::CONSTANTS_PER_LEVEL * NUM_LEVELS;
    } else {
        attn_scratch_cb.reserve_back(TILE_MAX_ROWS * ATTN_STICKS_PER_ROW);
        attn_arena_l1 = attn_scratch_cb.get_write_ptr();
        // All three operand pipes are allocated kGeomCbPages deep by the
        // factory, which is what geom_cb_pages carries.
#ifdef REF_FP32
        zero_tile_cb(geom_x_cb_index, geom_cb_pages, fused_msda_tile_layout::FP32_TILE_NBYTES);
        zero_tile_cb(geom_y_cb_index, geom_cb_pages, fused_msda_tile_layout::FP32_TILE_NBYTES);
#else
        zero_tile_cb(geom_x_cb_index, geom_cb_pages);
        zero_tile_cb(geom_y_cb_index, geom_cb_pages);
#endif
        zero_tile_cb(attn_tile_cb_index, geom_cb_pages);
    }

    // On Blackhole the gathers leave out-of-bounds slots unwritten and only
    // compute's zero weight cancels them (msda_gather_rows.hpp). Zeroing the CB
    // once keeps uninitialised L1 out of those slots; an earlier inf or NaN
    // value stick is cancelled by the FPU itself. The CB has never been pushed,
    // so the write pointer is at page 0.
    {
        const auto& iface = get_local_cb_interface(input_tile_cb_index);
        input_tile_cb.reserve_back(iface.fifo_size / iface.fifo_page_size);
        noc.async_write_zeros(input_tile_cb, iface.fifo_size);
        noc.write_zeros_l1_barrier();
    }

    RowGeometry geom;

    // Split gather: the reader posts each point to the mailbox and bumps
    // `ready`; the writer gathers its rows and bumps `done`. Both counters only
    // grow, so neither side resets anything.
    [[maybe_unused]] const uint32_t gather_mailbox_l1 =
        SPLIT_GATHER ? CircularBuffer(gather_mailbox_cb_index).get_write_ptr() : 0u;
    [[maybe_unused]] Semaphore<> gather_ready(gather_ready_sem_id);
    [[maybe_unused]] Semaphore<> gather_done(gather_done_sem_id);
    [[maybe_unused]] uint32_t gather_seq = 0;

    struct Unit {
        uint32_t b;
        uint32_t head;
        uint32_t q_start;
        uint32_t v_rows;
    };
    const uint32_t unit_args = arg_idx;
    auto unit = [&](uint32_t t) {
        const uint32_t base = unit_args + 4 * t;
        return Unit{
            get_arg_val<uint32_t>(base),
            get_arg_val<uint32_t>(base + 1),
            get_arg_val<uint32_t>(base + 2),
            get_arg_val<uint32_t>(base + 3)};
    };

    // Tiled geometry: block t's offsets and attn rows go into one row-major
    // page, its reference points into the source's arena. Nothing is pushed
    // until a barrier has covered the reads (publish_block). With
    // TILED_GEOMETRY off, `if constexpr` drops these bodies, so a source need
    // not define stage_tiled / fill_ref_tile.
    [[maybe_unused]] CircularBuffer geom_rm_cb(geom_rm_cb_index);
    [[maybe_unused]] CircularBuffer ref_tile_cb(ref_tile_cb_index);
    [[maybe_unused]] auto stage_block = [&](uint32_t t) {
        if constexpr (TILED_GEOMETRY) {
            namespace tg = fused_msda_tiled_geometry;
            geom_rm_cb.reserve_back(tg::RM_TILES);
            const uint32_t rm_l1 = geom_rm_cb.get_write_ptr();
            const Unit u = unit(t);
            stage_attention_weights(
                noc, attn_acc, rm_l1 + tg::RM_ATTN_OFFSET, u.b, u.q_start, u.head, u.v_rows, tg::RM_ROW_NBYTES);
            return loc_src.stage_tiled(noc, u.b, u.q_start, u.head, u.v_rows, rm_l1, tg::RM_ROW_NBYTES);
        } else {
            return false;
        }
    };
    // The reference tile is one page, so a block whose reference points did
    // not change pushes the same bytes again.
    [[maybe_unused]] auto publish_block = [&](uint32_t t, bool refs_restaged) {
        if constexpr (TILED_GEOMETRY) {
            ref_tile_cb.reserve_back(1);
            if (refs_restaged) {
                loc_src.fill_ref_tile(ref_tile_cb.get_write_ptr(), unit(t).v_rows);
            }
            ref_tile_cb.push_back(1);
            geom_rm_cb.push_back(fused_msda_tiled_geometry::RM_TILES);
        }
    };
    if constexpr (TILED_GEOMETRY) {
        if (num_output_tiles > 0) {
            const bool refs_restaged = stage_block(0);
            noc.async_read_barrier();
            publish_block(0, refs_restaged);
        }
    }

    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const Unit u = unit(t);
        const uint32_t b = u.b;
        const uint32_t head = u.head;
        const uint32_t q_start = u.q_start;
        const uint32_t v_rows = u.v_rows;

        [[maybe_unused]] uint32_t block_corner_l1 = 0;
        [[maybe_unused]] bool next_staged = false;
        [[maybe_unused]] bool next_refs_restaged = false;
        if constexpr (TILED_GEOMETRY) {
            corner_cb.wait_front(1);
            // Read across the whole block, and wait_front does not invalidate
            // the RISC L1 data cache, which may hold this page from two blocks
            // ago.
            invalidate_l1_cache();
            block_corner_l1 = corner_cb.get_read_ptr();
            if (t + 1 < num_output_tiles) {
                next_refs_restaged = stage_block(t + 1);
                next_staged = true;
            }
        } else {
            stage_attention_weights(noc, attn_acc, attn_arena_l1, b, q_start, head, v_rows);
            loc_src.stage(noc, b, q_start, head, v_rows);
            noc.async_read_barrier();
        }

        // Base page of this (batch, head) pair in the (B, S, H, D) value tensor.
        const uint32_t value_batch_base = b * NUM_KEYS;

        // Hands one sampling point's operands to the SFPU: column 0, and on V2 the
        // offset in column 16.
        auto push_point_geometry = [&](uint32_t j) {
            const uint32_t l = j / NUM_POINTS;
            const uint32_t p = j - l * NUM_POINTS;

            geom_x_cb.reserve_back(1);
            geom_y_cb.reserve_back(1);
            attn_tile_cb.reserve_back(1);
            const uint32_t gx_l1 = geom_x_cb.get_write_ptr();
            const uint32_t gy_l1 = geom_y_cb.get_write_ptr();
            const uint32_t at_l1 = attn_tile_cb.get_write_ptr();

            for (uint32_t r = 0; r < TILE_MAX_ROWS; ++r) {
                // Rows past v_rows go out as zero, so their attn lane, and with
                // it their scalar, is zero.
#ifdef REF_FP32
                uint32_t gx = 0;
                uint32_t gy = 0;
#else
                uint16_t gx = 0;
                uint16_t gy = 0;
#endif
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
#ifdef REF_FP32
                write_col0_fp32(gx_l1, r, gx);
                write_col0_fp32(gy_l1, r, gy);
#else
                write_col0(gx_l1, r, gx);
                write_col0(gy_l1, r, gy);
#endif
                write_col0(at_l1, r, av);
                if constexpr (FROM_OFFSETS) {
                    // The offset rides next to the primary operand, in the geom
                    // tile's format; bf16 -> float32 is exact.
#ifdef REF_FP32
                    write_col16_fp32(gx_l1, r, static_cast<uint32_t>(ox) << 16);
                    write_col16_fp32(gy_l1, r, static_cast<uint32_t>(oy) << 16);
#else
                    write_col16(gx_l1, r, ox);
                    write_col16(gy_l1, r, oy);
#endif
                }
            }

            geom_x_cb.push_back(1);
            geom_y_cb.push_back(1);
            attn_tile_cb.push_back(1);
        };

        // A tiled block's corner tile stays at the front until its last point.
        auto release_corners = [&](uint32_t j) {
            if (!TILED_GEOMETRY || j + 1 == POINTS_PER_BLOCK) {
                corner_cb.pop_front(1);
            }
        };

        // One point of lookahead, matching the compute kernel's (see the
        // pipelining note at the top of this file). Iteration j pushes j+1, so
        // point 0 goes out here.
        if constexpr (!TILED_GEOMETRY) {
            push_point_geometry(0);
        }

        for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
            // Exactly one push of point j+1 per iteration, none with the tiled
            // geometry: here without the split gather, after the post with it.
            // SPLIT_GATHER implies RM_STAGING (static_assert above), so the
            // second site always exists.
            if constexpr (!SPLIT_GATHER && !TILED_GEOMETRY) {
                if (j + 1 < POINTS_PER_BLOCK) {
                    push_point_geometry(j + 1);
                }
            }

            const uint32_t l = j / NUM_POINTS;
            const LevelGeom& g = levels[l];
            // The gathers read row r's word at corner_l1 +
            // tile_col16_offset_fp32(r), so a tiled block's point j is its
            // tile shifted by j columns.
            uint32_t corner_l1;
            if constexpr (TILED_GEOMETRY) {
                corner_l1 = block_corner_l1 + j * sizeof(uint32_t);
            } else {
                corner_cb.wait_front(1);
                corner_l1 = corner_cb.get_read_ptr();
            }

            if constexpr (RM_STAGING) {
                input_tile_cb.reserve_back(4 * N_D_TILES);
                const fused_msda_gather::PointArgs pt{
                    .corner_l1 = corner_l1,
                    .block_l1 = input_tile_cb.get_write_ptr(),
                    .v_rows = v_rows,
                    .start_index = g.start_index,
                    .width = g.width,
                    .batch = b,
                    .head = head,
                };
                if constexpr (SPLIT_GATHER) {
                    fused_msda_gather::post_point(gather_mailbox_l1, pt);
                    gather_ready.set(++gather_seq);
                    if (!TILED_GEOMETRY && j + 1 < POINTS_PER_BLOCK) {
                        push_point_geometry(j + 1);
                    }
                    fused_msda_gather::gather_rows<ReaderGatherCfg>(
                        noc, value_acc, pt, 0, fused_msda_gather::SPLIT_ROW<ValueAccessor>);
                    noc.async_read_barrier();
                    // The writer also reads this point's corner tile, so it is
                    // popped only after done.
                    gather_done.wait(gather_seq);
                } else {
                    fused_msda_gather::gather_rows<ReaderGatherCfg>(noc, value_acc, pt, 0, TILE_MAX_ROWS);
                    noc.async_read_barrier();
                }
                release_corners(j);
                input_tile_cb.push_back(4 * N_D_TILES);
            } else {
                namespace cw = fused_msda_corner_word;
                // nw_base + index is the NW corner's (b, s) row. Unsigned
                // wraparound keeps every in-bounds sum exact; the sum itself
                // wraps below zero only for an out-of-bounds NW, never read.
                const uint32_t nw_base = value_batch_base + g.start_index - (g.width + 1);
                for (uint32_t r = 0; r < v_rows; ++r) {
                    const auto w = cw::decode(CoreLocalMem<volatile uint32_t>(
                        corner_l1 + fused_msda_tile_layout::tile_col16_offset_fp32(r))[0]);
                    geom.nw[r] = nw_base + w.index;
                    geom.flags[r] = static_cast<uint8_t>(w.flags);
                }
                release_corners(j);

                for (uint32_t c = 0; c < 4; ++c) {
                    // c = 0, 1, 2, 3 is NW, NE, SW, SE, and must stay in lockstep
                    // with the four scalar packs in msda_geometry.hpp::point
                    // that build the matching scalar tiles.
                    const uint32_t dy = c >> 1;
                    const uint32_t dx = c & 1;
                    const uint32_t need = (cw::FLAG_Y0 << dy) | (cw::FLAG_X0 << dx);
                    const uint32_t step = dy * g.width + dx;
                    auto in_bounds = [&](uint32_t r) { return (geom.flags[r] & need) == need; };

                    input_tile_cb.reserve_back(N_D_TILES);
                    const uint32_t tile_l1 = input_tile_cb.get_write_ptr();

                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (!in_bounds(r)) {
                            continue;
                        }
                        // Canonical (B, S, H, D): one page per (b, s, h).
                        // Packed (B, S, H*D): one page per (b, s), head at byte offset h*D*2.
                        const uint32_t bs = geom.nw[r] + step;
                        uint32_t page;
                        uint32_t offset_bytes;
                        if constexpr (VALUE_PACKED) {
                            page = bs;
                            offset_bytes = head * (D * 2u);
                        } else {
                            page = bs * NUM_HEADS + head;
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
                    // The scatter reads the arena with plain loads; keep them
                    // behind the barrier, whose fence is not a compiler barrier.
                    asm volatile("" ::: "memory");

                    // Scatter each staged stick across the N_D_TILES face rows.
                    // A skipped row is zeroed where ZERO_SKIPPED_SLOTS asks for it
                    // (msda_gather_rows.hpp).
                    for (uint32_t r = 0; r < v_rows; ++r) {
                        if (in_bounds(r)) {
                            scatter_stick(tile_l1, r, value_arena_l1 + r * value_stick_nbytes);
                        } else if constexpr (fused_msda_gather::ZERO_SKIPPED_SLOTS) {
                            zero_input_row(tile_l1, r);
                        }
                    }
                    // The tile writes above are plain stores; keep them ahead of
                    // the push that hands the tiles to compute.
                    asm volatile("" ::: "memory");
                    input_tile_cb.push_back(N_D_TILES);
                }
            }

            // Every path above has barriered its reads by now, which covers
            // the next block's staging too.
            if constexpr (TILED_GEOMETRY) {
                if (j == 0 && next_staged) {
                    publish_block(t + 1, next_refs_restaged);
                }
            }
        }
    }
}

}  // namespace fused_msda
