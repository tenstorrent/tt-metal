// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::reduction {

struct ArgMaxOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<int>& dim = std::nullopt,
        bool keepdim = false,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
        bool use_multicore = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::reduction

constexpr auto argmax = ttnn::register_operation<"ttnn::argmax", ttnn::operations::reduction::ArgMaxOperation>();

}  // namespace ttnn
