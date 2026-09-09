// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "adaptive_chunk.hpp"

// Device-side expert -> row-group assignment for the GROUPED unified_routed_expert_ffn
// program factory, shared by the reader (BRISC), writer (NCRISC) and compute (TRISC
// UNPACK) kernels. Pure integer arithmetic, no NoC/CB/register APIs, so it is valid
// in every translation unit.
//
// The three kernels MUST compute the identical assignment from the identical
// counts/idx L1 pages, or the row groups disagree on which expert they process and
// the UP_SPLIT go/done sequence between reader and writer desynchronises. Keep every
// operation here deterministic (no floating point, fixed tie-breaks).
namespace group_assign {

constexpr uint32_t kMaxExperts = 32;  // LPT stack arrays; host validates experts_per_chip <= this
constexpr uint32_t kMaxGroups = 16;   // host validates num_row_groups <= this
constexpr uint32_t kTileHeight = 32;

// Column ownership of an N-axis core. Contiguous (legacy): core gx owns tiles
// [gx*per_core_N, (gx+1)*per_core_N). Strided ("band" mode, grid_x == num DRAM
// banks): core gx owns tiles {i*grid_x + gx}, i.e. exactly the tiles that the
// interleaved allocator places in DRAM bank gx (bank = page % num_banks when the
// row width in tiles is a multiple of num_banks), so a K-block of this core's
// weights is one contiguous DRAM run.
inline uint32_t global_col(uint32_t gx, uint32_t i, uint32_t per_core_N, uint32_t grid_x, uint32_t strided) {
    return strided ? (i * grid_x + gx) : (gx * per_core_N + i);
}

// Work items. A local expert's M-chunks are split into up to `pieces` contiguous ranges
// ("items") so that a large expert (many chunks) is spread over several row groups instead
// of serialising on one; every item re-streams the weights once per chunk exactly as the
// legacy chunk loop does, so the total work is unchanged — only the balance improves.
// Pieces of one expert touch disjoint token rows, so they may run concurrently.
constexpr uint32_t kMaxItems = 64;

struct Plan {
    uint32_t n_items = 0;
    uint8_t expert[kMaxItems];   // local expert id
    uint8_t chunk_b[kMaxItems];  // first M-chunk of the item
    uint8_t chunk_e[kMaxItems];  // one past the last M-chunk
    uint8_t group[kMaxItems];    // row group that runs it
};

// Item cost in tile-row-equivalents: every chunk re-streams the full weights
// (fixed_cost_tiles) plus the token rows it computes.
inline uint32_t item_cost(uint32_t chunks, uint32_t tiles, uint32_t fixed_cost_tiles) {
    return chunks * fixed_cost_tiles + tiles;
}

// Deterministic plan: identical in reader/compute/writer (same L1 counts page, same
// integer code). Zero-token experts produce no item.
inline void build_plan(
    const volatile tt_l1_ptr uint32_t* counts_ptr,
    const volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t num_experts,
    uint32_t num_groups,
    uint32_t chunk_M_max,
    uint32_t num_chunks_max,
    uint32_t m_tiles_full,
    uint32_t fixed_cost_tiles,
    Plan& plan) {
    // At most kMaxItems items in total: cap the pieces per expert accordingly.
    uint32_t max_pieces = kMaxItems / (num_experts > 0 ? num_experts : 1u);
    if (max_pieces < 1) {
        max_pieces = 1;
    }
    uint16_t cost[kMaxItems];
    uint8_t order[kMaxItems];
    uint32_t n = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t count_value = counts_ptr[idx_ptr[e]];
        const uint32_t raw_tiles = (count_value + kTileHeight - 1) / kTileHeight;
        const uint32_t tiles = adaptive_chunk::clamp_count_tiles(raw_tiles, chunk_M_max, num_chunks_max, m_tiles_full);
        const uint32_t n_chunks = adaptive_chunk::num_chunks(tiles, chunk_M_max);
        if (n_chunks == 0) {
            continue;
        }
        const uint32_t pieces = (n_chunks < max_pieces) ? n_chunks : max_pieces;
        const uint32_t base = n_chunks / pieces;
        const uint32_t rem = n_chunks % pieces;
        uint32_t c = 0;
        for (uint32_t pc = 0; pc < pieces && n < kMaxItems; ++pc) {
            const uint32_t len = base + (pc < rem ? 1u : 0u);
            const uint32_t c_e = c + len;
            const uint32_t last_tile = (c_e * chunk_M_max < tiles) ? c_e * chunk_M_max : tiles;
            const uint32_t item_tiles = last_tile - c * chunk_M_max;
            plan.expert[n] = static_cast<uint8_t>(e);
            plan.chunk_b[n] = static_cast<uint8_t>(c);
            plan.chunk_e[n] = static_cast<uint8_t>(c_e);
            const uint32_t cst = item_cost(len, item_tiles, fixed_cost_tiles);
            cost[n] = static_cast<uint16_t>(cst > 0xFFFFu ? 0xFFFFu : cst);
            order[n] = static_cast<uint8_t>(n);
            ++n;
            c = c_e;
        }
    }
    plan.n_items = n;
    // Insertion sort: cost descending, item id ascending on ties (deterministic).
    for (uint32_t i = 1; i < n; ++i) {
        const uint8_t key = order[i];
        uint32_t j = i;
        while (j > 0) {
            const uint8_t prev = order[j - 1];
            const bool prev_before_key = (cost[prev] > cost[key]) || (cost[prev] == cost[key] && prev < key);
            if (prev_before_key) {
                break;
            }
            order[j] = prev;
            --j;
        }
        order[j] = key;
    }
    // Greedy LPT onto the least-loaded group (lowest group id on ties).
    uint32_t load[kMaxGroups];
    for (uint32_t g = 0; g < num_groups; ++g) {
        load[g] = 0;
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint8_t it = order[k];
        uint32_t best = 0;
        for (uint32_t g = 1; g < num_groups; ++g) {
            if (load[g] < load[best]) {
                best = g;
            }
        }
        plan.group[it] = static_cast<uint8_t>(best);
        load[best] += cost[it];
    }
}

// Split-read: the `total` tile-rows of a weight K-block are shared by the R rows of a
// column group; row r reads rows [begin, end) (contiguous, all rows covered, sizes
// differ by at most one).
inline void slice_rows(uint32_t total, uint32_t R, uint32_t r, uint32_t& begin, uint32_t& end) {
    const uint32_t base = total / R;
    const uint32_t rem = total % R;
    begin = r * base + (r < rem ? r : rem);
    end = begin + base + (r < rem ? 1u : 0u);
}

}  // namespace group_assign
