// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <functional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations::reduction {

namespace prod {
        Tensor _prod(
            const Tensor& input_a,
            bool all_dimensions,
            int64_t dim,
            const MemoryConfig& output_mem_config);

        Tensor _prod_nc(
            const Tensor &input,
            const Tensor &output,
            std::vector<int64_t> &dims,
            const MemoryConfig &output_mem_config);
}


    struct ExecuteProdOp {
        static Tensor operator()(
            const Tensor& input,
            bool all_dimensions = false,
            int64_t dim = 0,
            const std::optional<MemoryConfig>& memory_config = std::nullopt) {
            return prod::_prod(input, all_dimensions, dim, memory_config.value_or(input.memory_config()));
        }
        static Tensor operator()(
            const Tensor& input,
            const Tensor& output,
            std::vector<int64_t> &dims,
            const std::optional<MemoryConfig>& memory_config = std::nullopt) {
            return prod::_prod_nc(input, output, dims, memory_config.value_or(input.memory_config()));
        }
    };

} // namespace operations::reduction

constexpr auto prod = ttnn::register_operation_with_auto_launch_op<"ttnn::prod", ttnn::operations::reduction::ExecuteProdOp>();

} // namespace ttnn
