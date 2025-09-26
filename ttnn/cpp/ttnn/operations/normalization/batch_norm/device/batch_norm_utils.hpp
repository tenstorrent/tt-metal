// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace tt::tt_metal {
class Tensor;
};

namespace ttnn::operations::normalization::batch_norm::utils {

// resolve an optional compute kernel config or compute
// a default based on input tensor's data type
DeviceComputeKernelConfig resolve_compute_kernel_config(
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config, const tt::tt_metal::Tensor& input);

}  // namespace ttnn::operations::normalization::batch_norm::utils
