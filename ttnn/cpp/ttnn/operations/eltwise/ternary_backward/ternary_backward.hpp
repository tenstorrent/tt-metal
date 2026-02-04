
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
        const Tensor& grad_tensor,
        const Tensor& input_a,
        const Tensor& tensor1,
        const Tensor& tensor2,
        float value,
        const MemoryConfig& output_mem_config);
};

struct AddcdivBackwardOperation {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor,
        const Tensor& input_a,
        const Tensor& tensor1,
        const Tensor& tensor2,
        float value,
        const MemoryConfig& output_mem_config);
};

struct WhereBackwardOperation {
    static std::vector<OptionalTensor> invoke(
        const Tensor& grad_tensor,
        const Tensor& condition,
        const Tensor& input_a,
        const Tensor& other,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
        OptionalTensor input_grad = std::nullopt,
        OptionalTensor other_grad = std::nullopt);
};

struct LerpBackwardOperation {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor,
        const Tensor& input_a,
        const Tensor& end,
        const Tensor& weight,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor,
        const Tensor& input_a,
        const Tensor& end,
        float weight,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace operations::ternary_backward

// type 1
constexpr auto addcmul_bw =
    ttnn::register_operation<"ttnn::addcmul_bw", operations::ternary_backward::AddcmulBackwardOperation>();
constexpr auto addcdiv_bw =
    ttnn::register_operation<"ttnn::addcdiv_bw", operations::ternary_backward::AddcdivBackwardOperation>();
constexpr auto where_bw =
    ttnn::register_operation<"ttnn::where_bw", operations::ternary_backward::WhereBackwardOperation>();
constexpr auto lerp_bw =
    ttnn::register_operation<"ttnn::lerp_bw", operations::ternary_backward::LerpBackwardOperation>();
}  // namespace ttnn
