// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

struct MorehClipGradNorm {
    static Tensor invoke(
        const std::vector<Tensor>& inputs,
        float max_norm,
        float norm_type,
        bool error_if_nonfinite,
        const std::optional<const Tensor>& total_norm,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn {
constexpr auto moreh_clip_grad_norm = ttnn::register_operation<
    "ttnn::moreh_clip_grad_norm",
    ttnn::operations::moreh::moreh_clip_grad_norm::MorehClipGradNorm>();
}  // namespace ttnn
