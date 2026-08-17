// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim::kda_factory_detail {

struct KdaPrepWorkDist {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    tt::tt_metal::CoreRangeSet core_set;
};

KdaPrepWorkDist distribute_prep(tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap);

}  // namespace ttnn::experimental::prim::kda_factory_detail
