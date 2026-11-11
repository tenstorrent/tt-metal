// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// DeepSeek V4.1 prefill Engram gate: everything ``DeepSeekV41PrefillEngram.forward`` does after ``wkv``, in one
// program. Same math as ``engram_gate`` (the decode op), but on DRAM-interleaved TILE operands at prefill sizes.
ttnn::Tensor engram_gate_prefill(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& gate_weight,
    float eps,
    float clamp_value = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek
