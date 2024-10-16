// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/normalization/softmax/device/softmax_op.hpp"

namespace ttnn {
namespace operations::transformer {

template <bool in_place>
struct ExecuteAttentionSoftmax {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               const std::optional<int>& head_size_arg = std::nullopt,
                               const std::optional<const ttnn::Tensor>& attention_mask = std::nullopt,
                               const ttnn::operations::normalization::SoftmaxProgramConfig& program_config =
                                   ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
                               const std::optional<bool> causal_mask = false,
                               const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto attention_softmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::transformer::attention_softmax",
                                                 ttnn::operations::transformer::ExecuteAttentionSoftmax<false>>();

constexpr auto attention_softmax_ =
    ttnn::register_operation_with_auto_launch_op<"ttnn::transformer::attention_softmax_",
                                                 ttnn::operations::transformer::ExecuteAttentionSoftmax<true>>();

}  // namespace transformer

}  // namespace ttnn
