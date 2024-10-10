// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
namespace ttnn::operations::moreh::moreh_dot_backward {
struct MorehDotBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &output_grad,
        const Tensor &input,
        const Tensor &other,
        const std::optional<const Tensor> &input_grad,
        const std::optional<const Tensor> &other_grad,
        const std::optional<MemoryConfig> &mem_config);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>> &optional_inputs);

    static std::vector<bool> create_async_return_flag(
        const Tensor &output_grad,
        const Tensor &input,
        const Tensor &other,
        const std::optional<const Tensor> &input_grad,
        const std::optional<const Tensor> &other_grad,
        const std::optional<MemoryConfig> &mem_config);
};
}  // namespace ttnn::operations::moreh::moreh_dot_backward

namespace ttnn {
constexpr auto moreh_dot_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_dot_backward",
    ttnn::operations::moreh::moreh_dot_backward::MorehDotBackward>();
}  // namespace ttnn
