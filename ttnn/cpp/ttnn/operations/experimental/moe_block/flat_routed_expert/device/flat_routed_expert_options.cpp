// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

const char* to_string(FlatDriscTiles tiles) {
    switch (tiles) {
        case FlatDriscTiles::both: return "both";
        case FlatDriscTiles::free_only: return "free_only";
        case FlatDriscTiles::noc1_endpoint_only: return "noc1_endpoint_only";
    }
    return "?";
}

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    return {u(x_helpers), pin_ratio, u(rows_hint), u(x_ready), static_cast<uint32_t>(drisc_tiles)};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "x_helpers {} pin_ratio {} rows_hint {} x_ready {} drisc_tiles {}",
        x_helpers,
        pin_ratio,
        rows_hint,
        x_ready,
        to_string(drisc_tiles));
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
