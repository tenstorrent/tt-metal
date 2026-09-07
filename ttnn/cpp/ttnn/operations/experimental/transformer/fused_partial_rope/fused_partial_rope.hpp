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
// Applies interleaved RoPE to the trailing `rope_dim` channels of a height- or
// width-sharded `[1, 1, rows, D]` input (TILE or ROW_MAJOR) and passes the leading
// `D - rope_dim` "nope" channels through untouched:
//
//   out[..., :D-Rd] = x[..., :D-Rd]
//   out[..., D-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin   (HiFi4)
//
// `cos`/`sin` are `[1, 1, rows, rope_dim]` (or a single broadcast row) DRAM-interleaved
// TILE tables; `trans_mat` is a single [32, 32] `rotate_half` tile (replicated).
// TILE X is processed as 32x32 tiles. ROW_MAJOR X is processed as 1x32 faces (one
// row of 32 elements per tile) and requires the broadcast cos/sin row. Output
// layout matches the input.
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
