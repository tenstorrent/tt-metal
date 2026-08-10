// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::experimental::reduction {

// The residual accumulation and the RMSNorm statistics that read it, in one pass:
//
//   total = a + b
//   stats = [sum(total * total), sum(total * q)]
//
// A residual stream's next read consumes exactly what its accumulation just wrote, so
// unfused the sum makes a full round trip through DRAM between two device programs. Here
// the sum is packed straight out of dest into both the reduce operand and the writer's
// buffer, so it is written once and never read back.
//
// `stats` is `[1, 2 * C, N, 1]`: each candidate's sum of squares, then its dots, stacked
// candidate-major the way a gathering collective leaves them. `q` is one row broadcast
// down the tokens. The addends must be bfloat16 — the sum runs on the FPU, whose source
// registers would cost a wider one precision an unfused add does not.
std::array<ttnn::Tensor, 2> attn_res_accum_stats(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& q,
    const std::optional<ttnn::DataType>& stats_dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
