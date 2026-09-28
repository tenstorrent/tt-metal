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
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const ttnn::Tensor& tail_a,
    const ttnn::Tensor& tail_b,
    const ttnn::Tensor& tail_entry_states,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0);

}  // namespace ttnn::experimental::kda
