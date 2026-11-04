// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

const char* to_string(FlatGuSource source) {
    switch (source) {
        case FlatGuSource::tensix: return "tensix";
        case FlatGuSource::tensix_split: return "tensix_split";
        case FlatGuSource::tensix_noc1: return "tensix_noc1";
        case FlatGuSource::placed: return "placed";
        case FlatGuSource::auto_tensix: return "auto_tensix";
    }
    return "?";
}

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    std::vector<uint32_t> k{
        u(x_helpers),
        pin_ratio,
        u(rows_hint),
        u(x_ready),
        static_cast<uint32_t>(gu_source),
        u(gu_read_depth),
        u(gu_read_pieces),
        static_cast<uint32_t>(gu_place.size())};
    k.insert(k.end(), gu_place.begin(), gu_place.end());
    return k;
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "x_helpers {} pin_ratio {} rows_hint {} x_ready {} gu_source {} gu_read_depth {} gu_read_pieces {} gu_place {} "
        "words",
        x_helpers,
        pin_ratio,
        rows_hint,
        x_ready,
        to_string(gu_source),
        gu_read_depth,
        gu_read_pieces,
        gu_place.size());
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
