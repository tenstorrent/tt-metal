// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor affine_exclusive_scan(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& initial_state,
    uint32_t groups_per_head,
    const std::optional<ttnn::Tensor>& tail_a = std::nullopt,
    const std::optional<ttnn::Tensor>& tail_b = std::nullopt,
    const std::optional<ttnn::Tensor>& tail_state = std::nullopt,
    const std::optional<ttnn::Tensor>& wrap_indicator = std::nullopt,
    uint32_t wrap_group = 0,
    bool split_in_group = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& actual_start = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    uint32_t local_rows = 0);

}  // namespace ttnn::experimental::kda
