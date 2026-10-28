// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program config of fused_recurrent_gated_delta_rule: the geometry of the v2 plan (V-split units per head and
// units looped per core), which the op's cost model chooses when a field is left unset. This build runs one whole
// head per core, so only the defaults are accepted; stage 2 of the plan fills the fields from the cost model and
// accepts the rest.

#pragma once

#include <cstdint>
#include <optional>

namespace ttnn::transformer {

struct FusedRecurrentGdnProgramConfig {
    // NV: V-slices per head, 1, 2 or 4 (V / 32 / NV tile columns per unit). nullopt = the cost model's pick.
    std::optional<uint32_t> num_slices;
    // Units (b, hv, V-slice) looped per core. nullopt = balanced to within one across the grid.
    std::optional<uint32_t> units_per_core;
};

// Tokens per call the op accepts: single-token decode and speculative verify with up to 31 draft tokens. Later
// stages size their per-token scratch (index pages, circular buffers) from this constant.
inline constexpr uint32_t kFusedRecurrentGdnMaxTokensPerCall = 32;

}  // namespace ttnn::transformer
