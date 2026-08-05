// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/generalized_moe_gate.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_device_operation.hpp"

namespace ttnn::experimental::deepseek::moe {

std::tuple<ttnn::Tensor, ttnn::Tensor> generalized_moe_gate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& bias_tensor,
    const ttnn::Tensor& input_indices_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid,
    uint32_t topk,
    bool output_softmax,
    bool grouped) {
    auto [operation_attributes, tensors_args] =
        ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::GeneralizedMoeGateDeviceOperation::invoke(
            input_tensor,
            bias_tensor,
            input_indices_tensor,
            output_tensor,
            output_indices_tensor,
            eps,
            scaling_factor,
            enable_sigmoid,
            topk,
            output_softmax,
            grouped);

    return ttnn::device_operation::launch<
        ttnn::operations::experimental::deepseek::moe::generalized_moe_gate::GeneralizedMoeGateDeviceOperation>(
        operation_attributes, tensors_args);
}

}  // namespace ttnn::experimental::deepseek::moe
