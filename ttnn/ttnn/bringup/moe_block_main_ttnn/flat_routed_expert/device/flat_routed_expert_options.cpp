// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation_types.hpp"

#include <fmt/format.h>

namespace ttnn::operations::bringup::moe_block_main::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    std::vector<uint32_t> k{pin_ratio, u(arena_bytes), u(gu_read_pieces)};
    if (weights_nk) {  // (bring-up option: off keeps the source's key)
        k.push_back(1);
    }
    return k;
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "pin_ratio {} arena_bytes {} gu_read_pieces {}{}",
        pin_ratio,
        arena_bytes,
        gu_read_pieces,
        weights_nk ? " weights_nk" : "");
}

}  // namespace ttnn::operations::bringup::moe_block_main::flat_routed_expert
