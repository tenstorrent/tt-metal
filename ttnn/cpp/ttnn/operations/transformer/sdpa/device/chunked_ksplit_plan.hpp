// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ttnn::prim {

// Host side of the chunked K split (kernels/chunked_ksplit.hpp). The work items, units * k_splits slices numbered
// unit * k_splits + slice, are dealt to the cores in contiguous ranges (an even split, the remainder one item each
// on the first cores). Each core runs the slices it holds of one unit as one piece, so a unit is cut into as many
// pieces as cores it spans; piece j of a unit writes output plane j. `pieces` is the most any unit has (the number
// of planes), and `first_slot[i]` the plane of core i's first piece.
struct ChunkedKSplitPlan {
    uint32_t pieces = 1;
    std::vector<uint32_t> first_slot;
};

inline std::pair<uint32_t, uint32_t> chunked_ksplit_core_range(uint32_t total_items, uint32_t num_cores, uint32_t i) {
    const uint32_t base = total_items / num_cores;
    const uint32_t extra = total_items % num_cores;
    return {(i * base) + std::min(i, extra), base + (i < extra ? 1 : 0)};
}

inline ChunkedKSplitPlan chunked_ksplit_plan(uint32_t total_items, uint32_t k_splits, uint32_t num_cores) {
    ChunkedKSplitPlan plan;
    plan.first_slot.assign(num_cores, 0);
    const uint32_t units = total_items / k_splits;
    // cores that ran slices of each unit so far
    std::vector<uint32_t> cores_so_far(units, 0);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto [start, count] = chunked_ksplit_core_range(total_items, num_cores, i);
        if (count == 0) {
            continue;
        }
        const uint32_t first_unit = start / k_splits;
        const uint32_t last_unit = (start + count - 1) / k_splits;
        plan.first_slot[i] = cores_so_far[first_unit];
        for (uint32_t u = first_unit; u <= last_unit; ++u) {
            plan.pieces = std::max(plan.pieces, ++cores_so_far[u]);
        }
    }
    return plan;
}

}  // namespace ttnn::prim
