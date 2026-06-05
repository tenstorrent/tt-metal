// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Depthwise 1D FIR filter with taps shared across all channels:
//   y[b, t, c] = sum_{j<K} taps[j] * x[b, t*stride + j, c]
// Input/output are (B, T_pad, C) ROW_MAJOR FLOAT32; T_out = (T_pad - K) / stride + 1.
ttnn::Tensor conv1d_depthwise(
    const ttnn::Tensor& input_tensor,
    const std::vector<float>& taps,
    uint32_t stride = 1,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
