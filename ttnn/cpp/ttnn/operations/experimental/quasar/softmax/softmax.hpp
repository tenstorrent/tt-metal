// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar (Metal 2.0) softmax. Same contract as ttnn::softmax for the interleaved, non-sharded,
// last-dim case: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))) along the last dimension
// (numeric_stable=False skips the max subtraction).
Tensor softmax(
    const Tensor& input_tensor,
    int dim = -1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);

}  // namespace ttnn::operations::experimental::quasar
