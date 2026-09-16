// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor rms_norm_pre_all_gather_bw(
    const Tensor& input_tensor,
    const Tensor& output_grad,
    const Tensor& stats,
    float epsilon = 1e-12,
    const std::optional<const Tensor>& weight = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
