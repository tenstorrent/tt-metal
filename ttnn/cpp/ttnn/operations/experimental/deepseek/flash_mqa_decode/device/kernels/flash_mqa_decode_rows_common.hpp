// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_common.hpp"

// Chunk plan of the block mode (num_rows R > 1, a DSpark verify block at positions p .. p + R - 1). Reader, writer
// and compute derive it from the same lengths, so they agree on every core's chunks, masks and reduction trees.
// The rows may be split over row_groups groups of cores (group rg attends rows rg, rg + row_groups, ...): each
// group runs the S-blocks of its rows on its own cores, so the rows run in parallel; a chunk shared by a group's rows
// is read once per group.
//
// Chunk indices are shared by all rows: g < ring_chunks are ring chunks (slots [g * kc, g * kc + kc)), g ==
// ring_chunks is the staging chunk (the block's new kv rows in its first tile row), and g > ring_chunks is selected
// chunk i = g - ring_chunks - 1 (row j's own rows, or the shared dense rows). Row j attends indices [0, len(j)).
// S-block b holds the indices g = b (mod num_s_blocks), as in the one-row mode, so a chunk shared by several rows
// (ring, staging, dense) is read once into L1 and attended by every row from there; row j's tree runs over the
// active S-blocks [0, min(len(j), num_s_blocks)).
//
// Independent rows (a B=4 batch step: row j is one token of its own session at position p_j): every row group holds
// one row (row_groups == num_rows), row j's ring is the slot at ring_first_row_j of a shared kv, and row j attends its
// ring's positions max(0, p_j - (W - 1)) .. p_j - 1, its own new kv row (key j of the staging chunk) and its n_j
// selected rows. Each row's plan is built from its own p_j, as a block row 0 at p_j would be; nothing is shared
// between row groups.
namespace flash_mqa_decode_rows {

using flash_mqa_decode::div_up;
using flash_mqa_decode::min_u32;

constexpr std::uint32_t kMaxRows = 8;

struct Plan {
    std::uint32_t num_rows;
    std::uint32_t window;  // ring capacity W
    std::uint32_t k_chunk_size;
    std::uint32_t p;            // position of block row 0; the ring holds positions <= p - 1
    std::uint32_t pm;           // (p - 1) mod W, the newest ring slot (p > 0)
    std::uint32_t ring_chunks;  // chunks covering min(p, W) ring slots
    std::uint32_t shared_len;   // indices [0, shared_len) are shared by the rows
    std::uint32_t n[kMaxRows];  // selected rows per block row
    bool independent;           // rows of independent sessions (see above): the plan is one row's, at its own p
};

// The lengths tensor's words a core reads: [p, ring_first_row, selected_first_row, 0, n_0 .. n_{R-1}] for a block,
// [n_0 .. n_{R-1}, p_0 .. p_{R-1}, ring_first_row_0 .., selected_first_row_0 ..] for independent rows.
constexpr std::uint32_t lengths_words(std::uint32_t num_rows, bool independent) {
    return independent ? 4 * num_rows : 4 + num_rows;
}

// One core's view of the lengths: the p its plan uses (row j's own with independent rows), each row's n, and the
// ring / selected first rows it reads from.
struct RowLengths {
    std::uint32_t p;
    std::uint32_t ring_first_row;
    std::uint32_t selected_first_row;
    std::uint32_t n[kMaxRows];
};

template <typename Ptr>
inline RowLengths parse_lengths(Ptr lengths, std::uint32_t num_rows, bool independent, std::uint32_t row) {
    RowLengths out{};
    if (independent) {
        for (std::uint32_t j = 0; j < num_rows; ++j) {
            out.n[j] = lengths[j];
        }
        out.p = lengths[num_rows + row];
        out.ring_first_row = lengths[2 * num_rows + row];
        out.selected_first_row = lengths[3 * num_rows + row];
    } else {
        for (std::uint32_t j = 0; j < num_rows; ++j) {
            out.n[j] = lengths[4 + j];
        }
        out.p = lengths[0];
        out.ring_first_row = lengths[1];
        out.selected_first_row = lengths[2];
    }
    return out;
}

// A core attends the rows row_begin, row_begin + row_step, ... (its row group); shared_len covers those rows.
inline Plan make_plan(
    std::uint32_t num_rows,
    std::uint32_t window,
    std::uint32_t k_chunk_size,
    std::uint32_t selected_capacity,
    bool dense,
    std::uint32_t p,
    const std::uint32_t* n,
    std::uint32_t row_begin = 0,
    std::uint32_t row_step = 1,
    bool independent = false) {
    Plan plan{};
    plan.independent = independent;
    plan.num_rows = num_rows;
    plan.window = window;
    plan.k_chunk_size = k_chunk_size;
    plan.p = p;
    plan.pm = p > 0 ? (p - 1) % window : 0;
    plan.ring_chunks = div_up(min_u32(p, window), k_chunk_size);
    std::uint32_t max_sel_chunks = 0;
    for (std::uint32_t j = 0; j < num_rows; ++j) {
        plan.n[j] = min_u32(n[j], selected_capacity);
    }
    for (std::uint32_t j = row_begin; j < num_rows; j += row_step) {
        const std::uint32_t c = div_up(plan.n[j], k_chunk_size);
        max_sel_chunks = c > max_sel_chunks ? c : max_sel_chunks;
    }
    // Independent rows share nothing (each row group reads its own ring, dense rows and staging key).
    plan.shared_len = independent ? 0 : plan.ring_chunks + 1 + (dense ? max_sel_chunks : 0);
    return plan;
}

inline std::uint32_t row_len(const Plan& plan, std::uint32_t j) {
    return plan.ring_chunks + 1 + div_up(plan.n[j], plan.k_chunk_size);
}

inline std::uint32_t row_active(const Plan& plan, std::uint32_t j, std::uint32_t num_s_blocks) {
    return min_u32(row_len(plan, j), num_s_blocks);
}

// Row j's valid ring slots form the circular interval of the newest lim = min(p, W - 1 - j) slots ending at pm:
// [a, pm] with a = pm + 1 - lim, wrapping to [a + W, W) when a < 0. An independent row is a block row 0 at its p.
inline std::int32_t ring_interval_start(const Plan& plan, std::uint32_t j) {
    const std::uint32_t lim = min_u32(plan.p, plan.window - 1 - (plan.independent ? 0 : j));
    return static_cast<std::int32_t>(plan.pm + 1) - static_cast<std::int32_t>(lim);
}

inline std::uint32_t overlap(std::int32_t a0, std::int32_t a1, std::int32_t b0, std::int32_t b1) {
    const std::int32_t lo = a0 > b0 ? a0 : b0;
    const std::int32_t hi = a1 < b1 ? a1 : b1;
    return hi > lo ? static_cast<std::uint32_t>(hi - lo) : 0;
}

// Keys of chunk g that row j attends (g < row_len(j)); the chunk needs a mask when this is below k_chunk_size.
inline std::uint32_t valid_rows(const Plan& plan, std::uint32_t j, std::uint32_t g) {
    const std::uint32_t kc = plan.k_chunk_size;
    if (g < plan.ring_chunks) {
        const std::int32_t lo = static_cast<std::int32_t>(g * kc);
        const std::int32_t hi = lo + static_cast<std::int32_t>(kc);
        const std::int32_t a = ring_interval_start(plan, j);
        const std::int32_t end = static_cast<std::int32_t>(plan.pm + 1);
        if (a >= 0) {
            return overlap(a, end, lo, hi);
        }
        const std::int32_t w = static_cast<std::int32_t>(plan.window);
        return overlap(0, end, lo, hi) + overlap(a + w, w, lo, hi);
    }
    if (g == plan.ring_chunks) {
        // A block row attends the new rows 0 .. j; an independent row only its own (key j, see staging_key_valid).
        return plan.independent ? 1 : j + 1;
    }
    const std::uint32_t first = (g - plan.ring_chunks - 1) * kc;
    return min_u32(kc, plan.n[j] - first);
}

// Whether key `key` of ring chunk g is a slot row j attends.
inline bool ring_key_valid(const Plan& plan, std::int32_t a, std::uint32_t slot) {
    const std::int32_t s = static_cast<std::int32_t>(slot);
    if (s > static_cast<std::int32_t>(plan.pm)) {
        return a < 0 && s >= a + static_cast<std::int32_t>(plan.window);
    }
    return s >= a;
}

// Whether key i of the staging chunk is one row j attends.
inline bool staging_key_valid(const Plan& plan, std::uint32_t j, std::uint32_t i) {
    return plan.independent ? i == j : i <= j;
}

// Pages (ms tiles) S-block c pushes into its receive CB for row j: the partials it merges plus the sink peer.
inline std::uint32_t receive_pages(
    const Plan& plan, std::uint32_t c, std::uint32_t j, std::uint32_t num_s_blocks, std::uint32_t num_tree_steps) {
    const std::uint32_t active = row_active(plan, j, num_s_blocks);
    if (c >= active) {
        return 0;
    }
    return flash_mqa_decode::num_received(c, active, num_tree_steps) + (c == 0 ? 1 : 0);
}

}  // namespace flash_mqa_decode_rows
