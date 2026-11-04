// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    return {pin_ratio, u(rows_hint), u(gu_read_pieces)};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format("pin_ratio {} rows_hint {} gu_read_pieces {}", pin_ratio, rows_hint, gu_read_pieces);
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
