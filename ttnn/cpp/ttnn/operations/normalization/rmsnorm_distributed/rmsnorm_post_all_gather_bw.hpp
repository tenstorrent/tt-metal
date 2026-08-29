// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Returns {input_grad, weight_grad}; weight_grad is nullopt when weight is unset, and is local to
// this shard -- gamma is sharded alongside the input, so no cross-device reduction is needed.
std::vector<std::optional<Tensor>> rms_norm_post_all_gather_bw(
    const Tensor& input_tensor,
    const Tensor& output_grad,
    const Tensor& stats,
    const Tensor& bw_stats,
    float epsilon = 1e-12,
    const std::optional<const Tensor>& weight = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
