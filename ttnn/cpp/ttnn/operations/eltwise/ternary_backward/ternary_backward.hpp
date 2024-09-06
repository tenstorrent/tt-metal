
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"

namespace ttnn {

namespace operations::ternary_backward {

using OptionalTensor = std::optional<Tensor>;

struct AddcmulBackwardOperation {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        float alpha,
        const MemoryConfig &memory_config);
};

struct AddcdivBackwardOperation {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        float alpha,
        const MemoryConfig &memory_config);
};

struct WhereBackwardOperation {
    static std::vector<OptionalTensor> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        OptionalTensor input_a_grad = std::nullopt,
        OptionalTensor input_b_grad = std::nullopt);
};

struct LerpBackwardOperation {
    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const Tensor &input_tensor_c_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt);

};

}  // operations::ternary_backward

//type 1
constexpr auto addcmul_bw = ttnn::register_operation<
    "ttnn::addcmul_bw",
    operations::ternary_backward::AddcmulBackwardOperation>();
constexpr auto addcdiv_bw = ttnn::register_operation<
    "ttnn::addcdiv_bw",
    operations::ternary_backward::AddcdivBackwardOperation>();
constexpr auto where_bw = ttnn::register_operation<
    "ttnn::where_bw",
    operations::ternary_backward::WhereBackwardOperation>();
constexpr auto lerp_bw = ttnn::register_operation<
    "ttnn::lerp_bw",
    operations::ternary_backward::LerpBackwardOperation>();
}  // namespace ttnn
