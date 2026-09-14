// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// VSA fine-stage attention fused with the SP-ring all-gather of K/V (Blackhole mesh, non-causal).
// Equivalent to vsa_sdpa(q, all_gather(k, dim=2), all_gather(v, dim=2), ...) in raw-selection streaming
// mode, up to bf16 rounding order: this device's K and V shards are forwarded around the ring by the op's own
// multi-worker gather (vsa_kv_gather_*.cpp, token-major so every head's blocks land progressively) while the
// attention consumes the blocks already landed. One program per device; no sub-devices. See VSA_RING_SDPA_SPEC.md.
//   q                              [1, H, S_local, d]              bf16 TILE, this device's query rows
//   k, v                           [1, H, T_local, d]              bf16 TILE, this device's K and V shards
//   indices                        [1, H, S_local/64, W]           uint32 ROW_MAJOR global (padded-per-shard) block ids
//   block_counts                   [1, 1, 1, Wc]                   uint32 ROW_MAJOR, global
//   persistent_output_buffer_k/v   [1, H, T_local*ring_size, d]    the all-gather ping-pong buffers for k and v
// multi_device_global_semaphore: two GlobalSemaphores [direction 0, direction 1] for the gather.
// num_workers_per_link: gather workers per direction per link (behind a fabric MUX when > 1); the senders
// (2*links*(workers + mux) cores) fill the compute grid's first rows, the VSA engine the rest.
ttnn::Tensor vsa_ring_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& block_counts,
    const ttnn::Tensor& persistent_output_buffer_k,
    const ttnn::Tensor& persistent_output_buffer_v,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t num_workers_per_link = 2,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<float> scale = std::nullopt,
    uint32_t block_size = 64,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint32_t list_len = 0,
    std::vector<uint32_t> exempt_ids = {},
    std::optional<ttnn::Tensor> dense_row_mask = std::nullopt,
    uint32_t coarse_slots_shift = 0,
    uint32_t coarse_real_per_shard = 0,
    std::vector<uint32_t> dense_row_hint = {});

}  // namespace ttnn::transformer
