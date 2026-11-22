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
    if (x_min_piece_bytes) {  // (bring-up option: 0 keeps the source's key; tagged so it never aliases weights_nk)
        k.push_back(0x78700000u);
        k.push_back(x_min_piece_bytes);
    }
    return k;
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "pin_ratio {} arena_bytes {} gu_read_pieces {}{}{}",
        pin_ratio,
        arena_bytes,
        gu_read_pieces,
        weights_nk ? " weights_nk" : "",
        x_min_piece_bytes ? fmt::format(" x_min_piece_bytes {}", x_min_piece_bytes) : std::string());
}

}  // namespace ttnn::operations::bringup::moe_block_main::flat_routed_expert
