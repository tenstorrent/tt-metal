// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::fused_partial_rope {

// Fused partial RoPE (deepseek_v4_flash `_apply_rope`) as a single device op.
//
// Applies interleaved RoPE to the trailing `rope_dim` channels of a height-sharded
// `[1, 1, rows, D]` input and passes the leading `D - rope_dim` "nope" channels through
// untouched:
//
//   out[..., :D-Rd] = x[..., :D-Rd]
//   out[..., D-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin   (HiFi4)
//
// `cos`/`sin` are `[1, 1, rows, rope_dim]` tables height-sharded on the same core grid as
// `input`; `trans_mat` is a single [32, 32] `rotate_half` tile (replicated). One tile-row
// (32 rows) is processed per core, so the op uses ceil(rows / 32) cores. Returns a new
// tensor with the same (height-sharded) spec as `input`.
ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope

namespace ttnn::experimental {
using operations::experimental::transformer::fused_partial_rope::fused_partial_rope;
}  // namespace ttnn::experimental
