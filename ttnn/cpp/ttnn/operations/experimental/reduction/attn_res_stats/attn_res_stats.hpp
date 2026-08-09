// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::experimental::reduction {

// The head of an AttnRes score pass: both rank-local `d`-reductions of `v` from
// one pass over it, stacked into the layout the statistics collective wants.
//
//   out[c]     = sum_d v[c][n][d] * v[c][n][d]
//   out[C + c] = sum_d v[c][n][d] * q[d]
//
// `v` is `[1, C, N, D]` and `q` is `[1, 1, 1, D]`; the result is `[1, 2C, N, 1]`.
// Unfused these are two separate streams of `v` — an RMSNorm statistics kernel
// and a matmul against `q` as a column — plus the slice, concat and typecast that
// bring their outputs into one tensor for the collective. Both reductions read
// the same row, so the row is made resident once and reduced twice.
//
// A row of `v` has to fit in L1 alongside `q` and one transformed copy, which
// bounds `D`.
ttnn::Tensor attn_res_stats(
    const ttnn::Tensor& v,
    const ttnn::Tensor& q,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
