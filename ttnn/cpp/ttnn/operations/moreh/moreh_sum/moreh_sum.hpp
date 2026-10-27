// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Deprecated: use ttnn::sum. Note that with keepdim=false, moreh_sum keeps reduced H/W axes as size 1.
Tensor moreh_sum(
    const Tensor& input,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim = std::nullopt,
    bool keepdim = false,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn

namespace ttnn::operations::moreh::moreh_sum {

// moreh_sum without the deprecation warning, for moreh ops that rely on its keepdim=false shapes.
Tensor moreh_sum(
    const Tensor& input,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
}  // namespace ttnn::operations::moreh::moreh_sum
