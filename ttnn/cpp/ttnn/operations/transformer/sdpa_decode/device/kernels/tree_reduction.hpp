// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/******************************************************************************
 *                    Tree Reduction Parameters (host + device)               *
 ******************************************************************************/

// The ONE implementation of the SDPA-decode reduction tree. The host calls it when it lays out
// the static per-padded-row groups; the kernels call it when active-row allocation
// (tenstorrent/tt-metal#59300) sizes a group at runtime. Keeping a single copy is what makes
// the static and the active-row paths reduce in the same order for the same group size.
//
// Binary tree over N ranks: rank 0 is the root, vid = (N - 1) - rank gives the tree position.
// At round r a rank whose low (r + 1) vid bits are all ones receives from vid - 2^r; every
// other rank sends to vid + 2^trailing_ones(vid) at round trailing_ones(vid). A rank whose
// parent vid falls past N (non-power-of-two N) is an orphan and sends to the root instead.

constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS = 6;  // Supports up to 2^6 = 64 cores per head
constexpr uint32_t TREE_REDUCTION_NONE = 0xFFFFFFFFu;

struct TreeReductionParams {
    uint32_t num_rounds = 0;                              // ceil(log2(N))
    bool is_root = false;                                 // final reducer (rank 0)
    uint32_t parent_core_in_group = TREE_REDUCTION_NONE;  // TREE_REDUCTION_NONE if root
    uint32_t send_at_round = TREE_REDUCTION_NONE;         // TREE_REDUCTION_NONE if root
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS] = {
        TREE_REDUCTION_NONE,
        TREE_REDUCTION_NONE,
        TREE_REDUCTION_NONE,
        TREE_REDUCTION_NONE,
        TREE_REDUCTION_NONE,
        TREE_REDUCTION_NONE};  // TREE_REDUCTION_NONE if no child
};
static_assert(
    MAX_TREE_REDUCTION_ROUNDS == 6,
    "children_per_round initialiser and the writer's semaphore tables are sized for 6 rounds");

inline TreeReductionParams get_tree_reduction_params(uint32_t rank, uint32_t N) {
    TreeReductionParams p;
    if (N <= 1) {
        p.is_root = true;
        return p;
    }

    const uint32_t vid = (N - 1) - rank;
    p.num_rounds = 32 - __builtin_clz(N - 1);  // ceil(log2(N))
    p.is_root = (rank == 0);

    // Children: at round r, vid receives from (vid - 2^r) if vid's low (r+1) bits are all 1
    for (uint32_t r = 0; r < p.num_rounds; r++) {
        const uint32_t mask = (2u << r) - 1;
        if ((vid & mask) == mask) {
            const uint32_t child_vid = vid - (1u << r);
            if (child_vid < N) {
                p.children_per_round[r] = (N - 1) - child_vid;
            }
        }
    }

    // Parent: send at round = trailing 1s in vid; parent_vid = vid + 2^round
    if (!p.is_root) {
        const uint32_t trailing_ones = __builtin_ctz(~vid);
        const uint32_t parent_vid = vid + (1u << trailing_ones);
        // If parent_vid >= N (non-power-of-2), orphan sends to root (rank 0)
        p.parent_core_in_group = (parent_vid < N) ? (N - 1) - parent_vid : 0;
        p.send_at_round = trailing_ones;
    } else {
        // Root collects orphans: ranks whose natural parent_vid >= N
        for (uint32_t c = 1; c < N; c++) {
            const uint32_t cv = (N - 1) - c;
            const uint32_t t = __builtin_ctz(~cv);
            if (cv + (1u << t) >= N && p.children_per_round[t] == TREE_REDUCTION_NONE) {
                p.children_per_round[t] = c;
            }
        }
    }
    return p;
}
