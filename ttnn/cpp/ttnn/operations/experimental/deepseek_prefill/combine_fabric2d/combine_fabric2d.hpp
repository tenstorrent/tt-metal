// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Allocates and returns the combined output: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim)
// BFLOAT16 ROW_MAJOR per device. See the nanobind docstring for what each tensor carries.
ttnn::Tensor combine_fabric2d(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t fwd_vc_pages,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& forwarding_buffer = std::nullopt);

// The forwarding_buffer for combine_fabric2d calls with these arguments, emb_dim being the token width. Create it
// once per mesh, before the first forward, and pass it on every call: left to the op it is allocated at cache miss,
// while that forward's tensors are live, and every cached program keeps its own.
ttnn::Tensor combine_fabric2d_forwarding_buffer(
    ttnn::MeshDevice* mesh_device,
    uint32_t emb_dim,
    uint32_t fwd_vc_pages,
    uint32_t cluster_axis = 0,
    uint32_t num_links = 2);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::combine_fabric2d::combine_fabric2d;
}  // namespace ttnn
