// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <functional>

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::reduction {

    struct ProdOperation {
        static Tensor invoke(
            const Tensor& input,
            bool all_dimensions = false,
            int64_t dim = 0,
            const std::optional<MemoryConfig>& memory_config = std::nullopt);

        static Tensor invoke(
            const Tensor& input,
            const Tensor& output,
            std::vector<int64_t> &dims,
            const std::optional<MemoryConfig>& memory_config = std::nullopt);
    };

} // namespace operations::reduction

constexpr auto prod = ttnn::register_operation_with_auto_launch_op<"ttnn::prod", ttnn::operations::reduction::ProdOperation>();

} // namespace ttnn
