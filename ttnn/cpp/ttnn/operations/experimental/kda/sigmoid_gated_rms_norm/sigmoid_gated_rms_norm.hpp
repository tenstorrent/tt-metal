// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

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
    const std::string& gate_activation = "sigmoid",
    uint32_t gate_col_offset_tiles = 0);

}  // namespace ttnn::experimental::kda
