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

const char* to_string(FlatDriscProbe probe) {
    switch (probe) {
        case FlatDriscProbe::none: return "none";
        case FlatDriscProbe::timing: return "timing";
        case FlatDriscProbe::dma_only: return "dma_only";
        case FlatDriscProbe::write_only: return "write_only";
        case FlatDriscProbe::loop_only: return "loop_only";
        case FlatDriscProbe::call_time: return "call_time";
    }
    return "?";
}

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    return {
        u(x_helpers),
        pin_ratio,
        u(rows_hint),
        u(x_ready),
        static_cast<uint32_t>(drisc_tiles),
        u(drisc_ub),
        u(drisc_dma_cap),
        drisc_mpfe,
        u(drisc_early_free),
        static_cast<uint32_t>(drisc_probe)};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "x_helpers {} pin_ratio {} rows_hint {} x_ready {} drisc_tiles {} drisc_ub {} drisc_dma_cap {} drisc_mpfe "
        "{:#x} "
        "drisc_early_free {} drisc_probe {}",
        x_helpers,
        pin_ratio,
        rows_hint,
        x_ready,
        to_string(drisc_tiles),
        drisc_ub,
        drisc_dma_cap,
        drisc_mpfe,
        drisc_early_free,
        to_string(drisc_probe));
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
