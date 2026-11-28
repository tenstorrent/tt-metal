// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

std::tuple<ttnn::Tensor, ttnn::Tensor> rms_norm_residual(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& weight,
    float epsilon,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

bool rms_norm_residual_supported(const ttnn::Tensor& input, const ttnn::Tensor& residual);

}  // namespace ttnn::experimental::llama31_8b
