// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "device/softmax_types.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteSoftmax {
    // softmax
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int dim_arg,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteScaleMaskSoftmax {
    // scale_mask_softmax
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteSoftmaxInPlace {

    // softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteScaleMaskSoftmaxInPlace {

    // scale_mask_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const bool is_causal_mask = false,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteScaleCausalMaskHWSoftmaxInPlace {

    // scale_causal_mask_hw_dims_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor> mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto softmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::softmax", ttnn::operations::normalization::ExecuteSoftmax>();
constexpr auto scale_mask_softmax = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_mask_softmax",
    ttnn::operations::normalization::ExecuteScaleMaskSoftmax>();
constexpr auto softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::softmax_in_place",
    ttnn::operations::normalization::ExecuteSoftmaxInPlace>();
constexpr auto scale_mask_softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_mask_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleMaskSoftmaxInPlace>();
constexpr auto scale_causal_mask_hw_dims_softmax_in_place = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scale_causal_mask_hw_dims_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleCausalMaskHWSoftmaxInPlace>();

}  // namespace ttnn
