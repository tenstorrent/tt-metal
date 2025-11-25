// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction {

// NOTE: This OP does not return anything, but register_operation currently does not handle void return types.
struct ExecuteManualSeed {
    static Tensor invoke(
        MeshDevice& device,
        std::variant<uint32_t, Tensor> seeds,
        std::optional<std::variant<uint32_t, Tensor>> user_ids = std::nullopt);
};

}  // namespace ttnn::operations::reduction

namespace ttnn {

constexpr auto manual_seed =
    ttnn::register_operation<"ttnn::manual_seed", ttnn::operations::reduction::ExecuteManualSeed>();

}  // namespace ttnn
