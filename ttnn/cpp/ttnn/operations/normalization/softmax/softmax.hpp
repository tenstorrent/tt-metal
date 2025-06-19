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
        int dim_arg,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

struct ExecuteScaleMaskSoftmax {
    // scale_mask_softmax
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        bool is_causal_mask = false,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

struct ExecuteSoftmaxInPlace {
    // softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

struct ExecuteScaleMaskSoftmaxInPlace {
    // scale_mask_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        bool is_causal_mask = false,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

struct ExecuteScaleCausalMaskHWSoftmaxInPlace {
    // scale_causal_mask_hw_dims_softmax_in_place
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

}  // namespace operations::normalization

constexpr auto softmax = ttnn::register_operation<"ttnn::softmax", ttnn::operations::normalization::ExecuteSoftmax>();
constexpr auto scale_mask_softmax =
    ttnn::register_operation<"ttnn::scale_mask_softmax", ttnn::operations::normalization::ExecuteScaleMaskSoftmax>();
constexpr auto softmax_in_place =
    ttnn::register_operation<"ttnn::softmax_in_place", ttnn::operations::normalization::ExecuteSoftmaxInPlace>();
constexpr auto scale_mask_softmax_in_place = ttnn::register_operation<
    "ttnn::scale_mask_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleMaskSoftmaxInPlace>();
constexpr auto scale_causal_mask_hw_dims_softmax_in_place = ttnn::register_operation<
    "ttnn::scale_causal_mask_hw_dims_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleCausalMaskHWSoftmaxInPlace>();

}  // namespace ttnn
