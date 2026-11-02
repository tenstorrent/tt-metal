// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar (Metal 2.0) batch_norm, inference mode only. Same contract as ttnn::batch_norm with
// training=False: out[n,c,h,w] = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c].
Tensor batch_norm(
    const Tensor& input,
    std::optional<Tensor> running_mean = std::nullopt,
    std::optional<Tensor> running_var = std::nullopt,
    bool training = false,
    float eps = 1e-05,
    float momentum = 0.1,
    const std::optional<Tensor>& weight = std::nullopt,
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
