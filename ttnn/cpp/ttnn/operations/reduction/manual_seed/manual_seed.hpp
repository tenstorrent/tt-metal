// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <functional>
#include <optional>
namespace ttnn::operations::reduction {

// NOTE: This OP does not return anything, but register_operation currently does not handle void return types.
struct ExecuteManualSeed {
    static Tensor invoke(
        const std::variant<uint32_t, Tensor>& seeds,
        std::optional<std::reference_wrapper<MeshDevice>> device = std::nullopt,
        const std::optional<std::variant<uint32_t, Tensor>>& user_ids = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace ttnn::operations::reduction

namespace ttnn {

constexpr auto manual_seed =
    ttnn::register_operation<"ttnn::manual_seed", ttnn::operations::reduction::ExecuteManualSeed>();

}  // namespace ttnn
