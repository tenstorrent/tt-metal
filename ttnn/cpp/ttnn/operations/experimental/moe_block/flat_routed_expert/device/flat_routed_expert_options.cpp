// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    return {u(x_helpers), pin_ratio, u(rows_hint), u(x_ready)};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format("x_helpers {} pin_ratio {} rows_hint {} x_ready {}", x_helpers, pin_ratio, rows_hint, x_ready);
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
