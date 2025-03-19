// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
namespace ttnn::operations::moreh::moreh_dot_backward {
struct MorehDotBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& other,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& other_grad,
        const std::optional<MemoryConfig>& memory_config);

    static OptionalTensors create_async_optional_output_tensors(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& other,
        const std::optional<const Tensor>& input_grad,
        const std::optional<const Tensor>& other_grad,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_dot_backward

namespace ttnn {
constexpr auto moreh_dot_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_dot_backward",
    ttnn::operations::moreh::moreh_dot_backward::MorehDotBackward>();
}  // namespace ttnn
