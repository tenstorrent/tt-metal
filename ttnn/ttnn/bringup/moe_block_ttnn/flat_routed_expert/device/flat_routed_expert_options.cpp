// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::bringup::moe_block::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    std::vector<uint32_t> k{static_cast<uint32_t>(arena_bytes), x_bf16 ? 1u : 0u, h_bf16 ? 1u : 0u};
    if (weights_nk) {  // (bring-up option: off keeps the source's key; tagged so it never aliases another option)
        k.push_back(0x6e6b0000u);
    }
    if (x_min_piece_bytes) {  // (bring-up option: 0 keeps the source's key)
        k.push_back(0x78700000u);
        k.push_back(x_min_piece_bytes);
    }
    return k;
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "arena_bytes {}, x {}, h {}{}{}",
        arena_bytes,
        x_bf16 ? "bf16" : "bfp8",
        h_bf16 ? "bf16" : "bfp8",
        weights_nk ? ", weights_nk" : "",
        x_min_piece_bytes ? fmt::format(", x_min_piece_bytes {}", x_min_piece_bytes) : std::string());
}

}  // namespace ttnn::operations::bringup::moe_block::flat_routed_expert
