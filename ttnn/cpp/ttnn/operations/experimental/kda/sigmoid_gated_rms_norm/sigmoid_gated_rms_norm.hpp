// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor sigmoid_gated_rms_norm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    uint32_t num_heads,
    float epsilon = 1e-5f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    ttnn::DataType output_dtype = ttnn::DataType::FLOAT32,
    uint32_t gate_column_offset = 0,
    const std::optional<ttnn::Tensor>& activated_gate = std::nullopt,
    const std::optional<ttnn::Tensor>& prefetch_source = std::nullopt,
    const std::optional<ttnn::Tensor>& prefetch_jobs = std::nullopt,
    const std::vector<tt::tt_metal::CoreCoord>& prefetch_cores = {},
    uint32_t prefetch_l1_begin = 0,
    uint32_t prefetch_l1_end = 0);

}  // namespace ttnn::experimental::kda
