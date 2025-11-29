// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement::nonzero {

struct ExecuteNonZeroIndices {
    static std::vector<Tensor> invoke(
        const Tensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace ttnn::operations::data_movement::nonzero

namespace ttnn {

constexpr auto nonzero =
    ttnn::register_operation<"ttnn::nonzero", ttnn::operations::data_movement::nonzero::ExecuteNonZeroIndices>();

}  // namespace ttnn
