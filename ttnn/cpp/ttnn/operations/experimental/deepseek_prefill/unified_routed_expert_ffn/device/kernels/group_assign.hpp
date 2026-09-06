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

// Relative cost of one expert on one row group: every M-chunk re-streams the full
// weights (fixed_cost_tiles, in tile-row-equivalents of compute) plus the rows
// themselves. Zero-token experts cost nothing and are skipped everywhere.
inline uint32_t expert_cost(uint32_t count_tiles, uint32_t chunk_M_max, uint32_t fixed_cost_tiles) {
    if (count_tiles == 0) {
        return 0;
    }
    return adaptive_chunk::num_chunks(count_tiles, chunk_M_max) * fixed_cost_tiles + count_tiles;
}

// Deterministic greedy LPT (longest processing time first): sort local experts by
// (cost desc, local id asc), then hand each to the least-loaded group (lowest group id
// on ties). Writes assign[e] in [0, num_groups) for e in [0, num_experts).
//
// counts_ptr / idx_ptr are the resident L1 scratch pages every kernel already holds
// (counts indexed by GLOBAL expert id via idx_ptr[local]). Counts are clamped exactly
// like the chunk loops clamp them (adaptive_chunk::clamp_count_tiles) so the cost
// model and the executed work agree.
inline void lpt_assign(
    const volatile tt_l1_ptr uint32_t* counts_ptr,
    const volatile tt_l1_ptr uint32_t* idx_ptr,
    uint32_t num_experts,
    uint32_t num_groups,
    uint32_t chunk_M_max,
    uint32_t num_chunks_max,
    uint32_t m_tiles_full,
    uint32_t fixed_cost_tiles,
    uint32_t* assign) {
    uint32_t cost[kMaxExperts];
    uint32_t order[kMaxExperts];
    uint32_t load[kMaxGroups];
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t count_value = counts_ptr[idx_ptr[e]];
        const uint32_t raw_tiles = (count_value + kTileHeight - 1) / kTileHeight;
        const uint32_t tiles = adaptive_chunk::clamp_count_tiles(raw_tiles, chunk_M_max, num_chunks_max, m_tiles_full);
        cost[e] = expert_cost(tiles, chunk_M_max, fixed_cost_tiles);
        order[e] = e;
    }
    // Insertion sort: cost descending, local id ascending on ties.
    for (uint32_t i = 1; i < num_experts; ++i) {
        const uint32_t key = order[i];
        uint32_t j = i;
        while (j > 0) {
            const uint32_t prev = order[j - 1];
            const bool prev_before_key = (cost[prev] > cost[key]) || (cost[prev] == cost[key] && prev < key);
            if (prev_before_key) {
                break;
            }
            order[j] = prev;
            --j;
        }
        order[j] = key;
    }
    for (uint32_t g = 0; g < num_groups; ++g) {
        load[g] = 0;
    }
    for (uint32_t k = 0; k < num_experts; ++k) {
        const uint32_t e = order[k];
        uint32_t best = 0;
        for (uint32_t g = 1; g < num_groups; ++g) {
            if (load[g] < load[best]) {
                best = g;
            }
        }
        assign[e] = best;
        load[best] += cost[e];
    }
}

}  // namespace group_assign
