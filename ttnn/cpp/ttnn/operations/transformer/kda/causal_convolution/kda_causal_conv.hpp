// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> kda_causal_conv1d_split(
    const ttnn::Tensor&,
    const ttnn::Tensor&,
    const ttnn::Tensor&,
    const ttnn::Tensor&,
    const ttnn::Tensor&,
    const ttnn::Tensor&,
    uint32_t,
    uint32_t,
    uint32_t,
    const std::optional<ttnn::MemoryConfig>& = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& = std::nullopt);

}  // namespace ttnn::transformer
