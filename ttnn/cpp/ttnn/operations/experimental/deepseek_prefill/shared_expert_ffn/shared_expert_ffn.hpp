// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "device/shared_expert_ffn_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

// Shared-expert GLU FFN for DeepSeek-family prefill on Blackhole, in ONE device op:
//   y = act(x @ gate_proj, x @ up_proj) @ down_proj
//
// typecast + unified_routed_expert_moe + mesh_partition(dim=-1) in one op, running the unified
// routed-expert program unchanged as a single expert that owns every row of x:
//   * x is read as given -- TILE bf16, TILE bf8_b or ROW_MAJOR bf16 -- so no typecast;
//   * down_proj may be one mesh column's emb slice, (hidden, emb / num_cols): the output is then
//     that column's slice, which is what mesh_partition of the full output kept.
// The output is a fresh TILE bf8_b tensor.
//
// Args:
//   x: (1, .., 1, M, emb), DRAM-interleaved, M a multiple of 32.
//   gate_proj / up_proj: (emb, hidden). down_proj: (hidden, N). TILE, DRAM-interleaved.
//   expert_token_counts / global_expert_idx_table / expert_region_offsets: ROW_MAJOR UINT32 DRAM
//     vectors describing the one expert -- [M], [0] and [0]. Built once by the caller.
// Keyword args:
//   activation: Silu (default), SwiGluOai, SituGlu or ClampedSiluGlu (constants baked as in
//     unified_routed_expert_moe).
//   compute_kernel_config: optional.
//   core_grid_origin: top-left logical core of the fixed 11x8 worker rectangle.
// Returns: (1, .., 1, M, N) TILE bf8_b, DRAM-interleaved.
ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const CoreCoord& core_grid_origin = CoreCoord{0, 0});

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::shared_expert_ffn::shared_expert_ffn;
}  // namespace ttnn
