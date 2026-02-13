// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <functional>
#include <optional>
#include <variant>

namespace ttnn {

Tensor manual_seed(
    const std::variant<uint32_t, Tensor>& seeds,
    std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
    const std::optional<std::variant<uint32_t, Tensor>>& user_ids = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
