// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

std::tuple<ttnn::Tensor, ttnn::Tensor> entry_correction(
    const ttnn::Tensor& zero_output,
    const ttnn::Tensor& identity_output,
    const ttnn::Tensor& final_a,
    const ttnn::Tensor& final_b,
    const ttnn::Tensor& entry,
    const ttnn::Tensor& tail_entry,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::kda
