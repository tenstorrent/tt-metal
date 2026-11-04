// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "device/dispatch_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Returns {dispatched_buffer, metadata}, both per-device and ROW_MAJOR.
std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const std::optional<ttnn::Tensor>& padding_config,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t fwd_vc_pages,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<ttnn::Tensor>& forwarding_buffer = std::nullopt,
    const std::optional<ttnn::Tensor>& staging_buffer = std::nullopt);

// {forwarding_buffer, staging_buffer} for dispatch_fabric2d calls with these arguments, emb_dim being the token
// width. Create them once per mesh, before the first forward, and pass them on every call: left to the op they are
// allocated at cache miss, while that forward's tensors are live, and every cached program keeps its own.
std::array<ttnn::Tensor, 2> dispatch_fabric2d_buffers(
    ttnn::MeshDevice* mesh_device,
    uint32_t emb_dim,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis = 0,
    uint32_t num_links = 1,
    uint32_t fwd_vc_pages = 64);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::dispatch_fabric2d::dispatch_fabric2d;
}  // namespace ttnn
