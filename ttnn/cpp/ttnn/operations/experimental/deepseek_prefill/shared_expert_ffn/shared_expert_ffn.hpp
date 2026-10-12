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

// Fused dense (shared-expert) GLU FFN for DeepSeek-family prefill on Blackhole, in ONE device op:
//   y = act(x @ gate_proj, x @ up_proj) @ down_proj
//
// It is typecast + unified_routed_expert_moe + mesh_partition(dim=-1, cluster_axis) in one op:
//   * the unified routed-expert program runs as a single expert owning every row of x, so it
//     takes no counts / index / region-offset tensors;
//   * x is read as given -- TILE bf16, TILE bf8_b or ROW_MAJOR bf16 -- so no typecast;
//   * with cluster_axis, the device at index i along it computes only emb slice i of the output,
//     reading just those columns of the replicated down_proj.
// The output is a fresh TILE bf8_b tensor.
//
// Args:
//   x: (1, .., 1, M, emb), DRAM-interleaved, M a multiple of 32.
//   gate_proj / up_proj: (emb, hidden). down_proj: (hidden, emb). TILE, DRAM-interleaved,
//     replicated on every device.
// Keyword args:
//   activation: Silu (default), SwiGluOai, SituGlu or ClampedSiluGlu (constants baked as in
//     unified_routed_expert_moe).
//   compute_kernel_config: optional.
//   core_grid_origin: top-left logical core of the fixed 11x8 worker rectangle.
//   cluster_axis: mesh axis to partition the output's emb over; nullopt => full width.
// Returns: (1, .., 1, M, emb / mesh_shape[cluster_axis]) TILE bf8_b, DRAM-interleaved.
ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const CoreCoord& core_grid_origin = CoreCoord{0, 0},
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::shared_expert_ffn::shared_expert_ffn;
}  // namespace ttnn
