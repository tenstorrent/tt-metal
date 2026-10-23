// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ttnn::operations::ccl::common {

inline constexpr std::size_t num_fabric_directions = 4;
using FabricDirections = std::array<bool, num_fabric_directions>;

constexpr uint32_t fabric_directions_to_mask(const FabricDirections& directions) {
    uint32_t mask = 0;
    for (std::size_t direction = 0; direction < directions.size(); ++direction) {
        mask |= static_cast<uint32_t>(directions[direction]) << direction;
    }
    return mask;
}

constexpr FabricDirections fabric_direction_mask_to_directions(uint32_t mask) {
    FabricDirections directions{};
    for (std::size_t direction = 0; direction < directions.size(); ++direction) {
        directions[direction] = (mask & (1U << direction)) != 0;
    }
    return directions;
}

}  // namespace ttnn::operations::ccl::common
