// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_factory_utils.hpp"

#include <algorithm>
#include <set>

#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim::kda_factory_detail {

KdaPrepWorkDist distribute_prep(tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap) {
    const uint32_t max_cores = std::min<uint32_t>(grid.x * grid.y, core_cap);
    const uint32_t count = std::min(total, max_cores);
    TT_FATAL(count > 0, "KDA work distribution needs at least one item (total={})", total);
    const uint32_t base = total / count;
    const uint32_t remainder = total % count;

    KdaPrepWorkDist distribution;
    distribution.cores.reserve(count);
    distribution.wi_start.reserve(count);
    distribution.wi_count.reserve(count);
    std::set<tt::tt_metal::CoreRange> ranges;
    uint32_t offset = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const tt::tt_metal::CoreCoord core{index % grid.x, index / grid.x};
        const uint32_t item_count = base + (index < remainder ? 1u : 0u);
        distribution.cores.push_back(core);
        distribution.wi_start.push_back(offset);
        distribution.wi_count.push_back(item_count);
        ranges.insert(tt::tt_metal::CoreRange{core, core});
        offset += item_count;
    }
    distribution.core_set = tt::tt_metal::CoreRangeSet{ranges};
    return distribution;
}

}  // namespace ttnn::experimental::prim::kda_factory_detail
