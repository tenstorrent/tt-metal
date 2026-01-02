// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/groupnorm_device_operation_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

namespace operations::normalization {

struct ExecuteGroupNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int num_groups,
        float epsilon,
        const std::optional<ttnn::Tensor>& input_mask = std::nullopt,
        const std::optional<ttnn::Tensor>& weight = std::nullopt,
        const std::optional<ttnn::Tensor>& bias = std::nullopt,
        const std::optional<ttnn::Tensor>& reciprocals = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<CoreGrid> core_grid = std::nullopt,
        std::optional<bool> inplace = std::nullopt,
        std::optional<ttnn::Layout> output_layout = std::nullopt,
        std::optional<int> num_out_blocks = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<ttnn::Tensor>& negative_mask = std::nullopt,
        bool use_welford = false);
};

struct ExecuteGroupNormV3 {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int num_groups,
        float epsilon,
        const std::optional<ttnn::Tensor>& weight = std::nullopt,
        const std::optional<ttnn::Tensor>& bias = std::nullopt,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<CoreGrid> core_grid = std::nullopt,
        std::optional<bool> inplace = std::nullopt,
        std::optional<int> chunk_size = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto group_norm =
    ttnn::register_operation<"ttnn::group_norm", ttnn::operations::normalization::ExecuteGroupNorm>();

constexpr auto group_norm_v3 =
    ttnn::register_operation<"ttnn::group_norm_v3", ttnn::operations::normalization::ExecuteGroupNormV3>();

}  // namespace ttnn
