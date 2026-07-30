// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::experimental::reduction {

// out[b][0][h][w] = sum_c input[b][c][h][w] * weight[b][c][h][0]
//
// A reduction over `dim` fused with the per-row weighting that would otherwise
// need a separate `mul` and its full-size intermediate. The product is MAC'd
// into the accumulator, so the tensor is read once instead of three times.
//
// `weight` carries one scalar per (c, h) and is broadcast along the last dim,
// which is what `BroadcastType::COL` does natively — no transpose, and no
// padding beyond the tile width the layout already imposes.
//
// Only `dim == 1` is implemented. See the device op's validation.
ttnn::Tensor fast_weighted_reduce_nc(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
