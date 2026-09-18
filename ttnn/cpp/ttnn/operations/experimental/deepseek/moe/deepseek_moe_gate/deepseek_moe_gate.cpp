// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/deepseek_moe_gate.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/deepseek_moe_gate_device_operation.hpp"

namespace ttnn::experimental::deepseek::moe {

std::tuple<ttnn::Tensor, ttnn::Tensor> deepseek_moe_gate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& bias_tensor,
    const ttnn::Tensor& input_indices_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid) {
    auto [operation_attributes, tensors_args] =
        ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::DeepseekMoeGateDeviceOperation::invoke(
            input_tensor,
            bias_tensor,
            input_indices_tensor,
            output_tensor,
            output_indices_tensor,
            eps,
            scaling_factor,
            enable_sigmoid);

    return ttnn::device_operation::launch<
        ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::DeepseekMoeGateDeviceOperation>(
        operation_attributes, tensors_args);
}

}  // namespace ttnn::experimental::deepseek::moe
