// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program geometry of fused_recurrent_gated_delta_rule. This build runs one whole head per core and accepts only the
// defaults; the V-split units and the units looped per core of the next steps fill the fields from a cost model.

#pragma once

#include <cstdint>
#include <optional>

namespace ttnn::transformer {

struct FusedRecurrentGdnProgramConfig {
    // NV: V-slices per head, 1, 2 or 4 (V / 32 / NV tile columns per unit). nullopt = the op's pick.
    std::optional<uint32_t> num_slices;
    // Units (b, hv, V-slice) looped per core. nullopt = balanced to within one across the grid.
    std::optional<uint32_t> units_per_core;
};

// The per-token CBs and the ring layout are sized for at most this many tokens in one call.
inline constexpr uint32_t kFusedRecurrentGdnMaxTokensPerCall = 32;

}  // namespace ttnn::transformer
