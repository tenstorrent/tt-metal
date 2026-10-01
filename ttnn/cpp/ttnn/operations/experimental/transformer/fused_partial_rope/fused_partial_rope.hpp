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
// Applies interleaved RoPE independently to each `head_dim`-wide block of a
// height- or width-sharded `[1, 1, rows, D]` input (TILE or ROW_MAJOR). `D` must
// be a multiple of `head_dim` (pass 0 to use `D`, i.e. a single block). Within
// each block the trailing `rope_dim` channels are rotated and the leading
// `head_dim - rope_dim` "nope" channels pass through:
//
//   for each block b of `head_dim` channels:
//     out[..., b, :Hd-Rd] = x[..., b, :Hd-Rd]
//     out[..., b, Hd-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin   (HiFi4)
//
// `cos`/`sin` are `[1, 1, rows, rope_dim]` (or a single broadcast row) DRAM-interleaved
// TILE tables, shared across every head block of a row; `trans_mat` is a single
// [32, 32] `rotate_half` tile (replicated). TILE X is processed as 32x32 tiles.
// ROW_MAJOR X is processed as 1x32 faces (one row of 32 elements per tile) and
// requires the broadcast cos/sin row. Output layout matches the input.
ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t head_dim = 0);

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope

namespace ttnn::experimental {
using operations::experimental::transformer::fused_partial_rope::fused_partial_rope;
}  // namespace ttnn::experimental
