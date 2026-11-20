// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    return {static_cast<uint32_t>(arena_bytes), x_bf16 ? 1u : 0u, h_bf16 ? 1u : 0u};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "arena_bytes {}, x {}, h {}", arena_bytes, x_bf16 ? "bf16" : "bfp8", h_bf16 ? "bf16" : "bfp8");
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
