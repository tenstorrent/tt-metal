// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Walk and Map of the Quasar-native binary_ng kernels. The reader and both compute kernels include this header, so
// they agree on every page. The factory includes it for the mode values.
//
// Walk: the output tiles that one tile counter of a thread holds, in order. Consecutive tiles of a counter are a
// constant number of core-local tiles apart, so each step moves the output page by one constant delta, or, in the
// shard walk, by one of two: inside a shard row and across its edge.
// Map: the page of an operand that an output tile reads. Each output dim has a page stride per operand, and the
// stride is 0 where the operand's dim is 1.
// A cursor holds a counter's output position as mixed-radix digits and each operand's page. A step is adds and one
// compare per digit, so no kernel divides per tile.

// The kernels take A_MODE and B_MODE as these values. FULL, ROW, COL and SCALAR read through their Map: a whole tile,
// one tile row, one tile column or one tile. VALUE is one value for the whole op, with no Walk and no Map.
#define QSR_MODE_FULL 0
#define QSR_MODE_ROW 1
#define QSR_MODE_COL 2
#define QSR_MODE_SCALAR 3
#define QSR_MODE_VALUE 4

#ifdef KERNEL_BUILD
// A dense program defines none of A_MODE, B_MODE, A_OUTER_BCAST and B_OUTER_BCAST, and each kernel compiles to its
// dense form. A program in which some operand's Map is not the identity defines all four.
#ifndef A_MODE
#define A_MODE QSR_MODE_FULL
#define B_MODE QSR_MODE_FULL
#define A_OUTER_BCAST 0
#define B_OUTER_BCAST 0
#endif
// For #if. An identity operand reads the output's own page. A held operand (COL, SCALAR) is expanded once per page
// change and held. A VALUE operand is one ring entry per compute thread.
#define A_IDENTITY (A_MODE == QSR_MODE_FULL && !A_OUTER_BCAST)
#define B_IDENTITY (B_MODE == QSR_MODE_FULL && !B_OUTER_BCAST)
#define QSR_MAPPED (!A_IDENTITY || !B_IDENTITY)
#define A_ROW (A_MODE == QSR_MODE_ROW)
#define B_ROW (B_MODE == QSR_MODE_ROW)
#define A_HELD (A_MODE == QSR_MODE_COL || A_MODE == QSR_MODE_SCALAR)
#define B_HELD (B_MODE == QSR_MODE_COL || B_MODE == QSR_MODE_SCALAR)
#define A_VALUE (A_MODE == QSR_MODE_VALUE)
#define B_VALUE (B_MODE == QSR_MODE_VALUE)
#endif

namespace binary_ng_qsr {

enum class OperandMode : uint32_t {
    FULL = QSR_MODE_FULL,
    ROW = QSR_MODE_ROW,
    COL = QSR_MODE_COL,
    SCALAR = QSR_MODE_SCALAR,
    VALUE = QSR_MODE_VALUE,
};

// The output page as mixed-radix digits, least significant first: tw, th, c, n, d, e and nd, where e is dim -6 and nd
// the dims past it. The top digit has no radix, so a walk past the last page, over pad slots, keeps counting.
inline constexpr uint32_t kNumDigits = 7;

struct Digits {
    uint32_t v[kNumDigits] = {};
};

// The radices of the six low digits: the output's Wt, Ht, C, N, D and E.
struct OutputDims {
    uint32_t radix[kNumDigits - 1] = {};
};

// An operand's page stride per output digit, in the digits' order. It is 0 where the operand's dim is 1.
struct OperandMap {
    uint32_t stride[kNumDigits] = {};
};

// One divide per digit. A kernel splits a page only at entry.
inline Digits split_page(uint32_t page, const OutputDims& dims) {
    Digits digits;
    for (uint32_t i = 0; i + 1 < kNumDigits; ++i) {
        digits.v[i] = page % dims.radix[i];
        page /= dims.radix[i];
    }
    digits.v[kNumDigits - 1] = page;
    return digits;
}

inline uint32_t page_of(const Digits& digits, const OperandMap& map) {
    uint32_t page = 0;
    for (uint32_t i = 0; i < kNumDigits; ++i) {
        page += digits.v[i] * map.stride[i];
    }
    return page;
}

// The output page of core-local tile k: the shard walk of a shard `shard_width` tiles wide in a tensor `row_tiles`
// tiles wide, the bank walk over `bank_stride` banks, or, with both 0, the linear walk.
inline uint32_t walk_page(uint32_t start, uint32_t k, uint32_t shard_width, uint32_t row_tiles, uint32_t bank_stride) {
    if (shard_width != 0) {
        return start + (k / shard_width) * row_tiles + k % shard_width;
    }
    if (bank_stride != 0) {
        return start + k * bank_stride;
    }
    return start + k;
}

// The step from one tile of a counter to the next, `tile_step` core-local tiles on. In the shard walk a step whose
// column passes the shard row's edge takes delta[1], and every other step takes delta[0]. page_step and page_wrap hold
// what a step and a carry out of each digit add to each operand's page.
struct WalkStep {
    OutputDims dims;
    Digits delta[2];
    uint32_t shard_width = 0;
    uint32_t col_step = 0;
    uint32_t page_step[2][2] = {};               // [a or b][delta]
    uint32_t page_wrap[2][kNumDigits - 1] = {};  // [a or b][digit]
};

// The shard walk needs shard_width <= row_tiles. A wider shard makes the step across a shard row's edge negative, and
// the unsigned delta wraps. The factory gives a mapped program no such shard.
inline WalkStep make_walk_step(
    const OutputDims& dims,
    uint32_t tile_step,
    uint32_t shard_width,
    uint32_t row_tiles,
    uint32_t bank_stride,
    const OperandMap& map_a,
    const OperandMap& map_b) {
    WalkStep step;
    step.dims = dims;
    uint32_t delta_in_row = tile_step;
    uint32_t delta_past_edge = tile_step;
    if (shard_width != 0) {
        delta_in_row = (tile_step / shard_width) * row_tiles + tile_step % shard_width;
        delta_past_edge = delta_in_row + row_tiles - shard_width;
        step.shard_width = shard_width;
        step.col_step = tile_step % shard_width;
    } else if (bank_stride != 0) {
        delta_in_row = tile_step * bank_stride;
        delta_past_edge = delta_in_row;
    }
    step.delta[0] = split_page(delta_in_row, dims);
    step.delta[1] = split_page(delta_past_edge, dims);
    const OperandMap* maps[2] = {&map_a, &map_b};
    for (uint32_t x = 0; x < 2; ++x) {
        for (uint32_t j = 0; j < 2; ++j) {
            step.page_step[x][j] = page_of(step.delta[j], *maps[x]);
        }
        // A carry out of digit i takes radix[i] from it and adds 1 to digit i + 1. Unsigned wrap is exact here.
        for (uint32_t i = 0; i + 1 < kNumDigits; ++i) {
            step.page_wrap[x][i] = maps[x]->stride[i + 1] - dims.radix[i] * maps[x]->stride[i];
        }
    }
    return step;
}

// One tile counter's place in its walk: the output digits, the column in the shard row, each operand's page, and
// which pages the last step changed (bit 0 a, bit 1 b). The first tile of a counter counts as a change of both.
struct Cursor {
    Digits position;
    uint32_t col = 0;
    uint32_t page[2] = {};
    uint32_t changed = 3;
};

inline Cursor make_cursor(
    uint32_t first_page, uint32_t first_col, const OutputDims& dims, const OperandMap& map_a, const OperandMap& map_b) {
    Cursor cursor;
    cursor.position = split_page(first_page, dims);
    cursor.col = first_col;
    cursor.page[0] = page_of(cursor.position, map_a);
    cursor.page[1] = page_of(cursor.position, map_b);
    return cursor;
}

inline void advance(Cursor& cursor, const WalkStep& step) {
    bool past_edge = false;
    if (step.shard_width != 0) {
        cursor.col += step.col_step;
        if (cursor.col >= step.shard_width) {
            cursor.col -= step.shard_width;
            past_edge = true;
        }
    }
    const uint32_t page_a = cursor.page[0];
    const uint32_t page_b = cursor.page[1];
    // A select, not an index: an index into the two deltas compiles to a multiply.
    const Digits& delta = past_edge ? step.delta[1] : step.delta[0];
    cursor.page[0] += past_edge ? step.page_step[0][1] : step.page_step[0][0];
    cursor.page[1] += past_edge ? step.page_step[1][1] : step.page_step[1][0];
    uint32_t carry = 0;
    for (uint32_t i = 0; i + 1 < kNumDigits; ++i) {
        uint32_t digit = cursor.position.v[i] + delta.v[i] + carry;
        carry = 0;
        if (digit >= step.dims.radix[i]) {
            digit -= step.dims.radix[i];
            carry = 1;
            cursor.page[0] += step.page_wrap[0][i];
            cursor.page[1] += step.page_wrap[1][i];
        }
        cursor.position.v[i] = digit;
    }
    cursor.position.v[kNumDigits - 1] += delta.v[kNumDigits - 1] + carry;
    cursor.changed = (cursor.page[0] != page_a ? 1u : 0u) | (cursor.page[1] != page_b ? 2u : 0u);
}

// The size of the next compute batch: at most `limit` tiles, and no change of a page in `held_mask` (bit 0 a, bit 1 b)
// after its first tile. The cursor ends on the tile after the batch. `first_changed` receives the first tile's bits.
inline uint32_t next_batch(
    Cursor& cursor, const WalkStep& step, uint32_t limit, uint32_t held_mask, uint32_t& first_changed) {
    first_changed = cursor.changed;
    uint32_t tiles = 1;
    advance(cursor, step);
    while (tiles < limit && (cursor.changed & held_mask) == 0) {
        ++tiles;
        advance(cursor, step);
    }
    return tiles;
}

// Runs fn(n, first_changed) on each compute batch of a thread's my_tiles tiles. A batch ends at the next multiple of
// batch_size, as in the same-shape loop, or earlier where a page in held_mask changes, so every tile of a batch reads
// one held entry. A ring holds a multiple of batch_size per thread and wraps only after a batch, so no batch crosses
// its end.
template <typename Fn>
inline void for_each_batch(
    Cursor& cursor, const WalkStep& step, uint32_t my_tiles, uint32_t batch_size, uint32_t held_mask, Fn fn) {
    for (uint32_t done = 0; done < my_tiles;) {
        const uint32_t to_multiple = batch_size - done % batch_size;
        const uint32_t left = my_tiles - done;
        uint32_t first_changed = 0;
        const uint32_t n = next_batch(cursor, step, left < to_multiple ? left : to_multiple, held_mask, first_changed);
        fn(n, first_changed);
        done += n;
    }
}

}  // namespace binary_ng_qsr
