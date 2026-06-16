// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <umd/device/types/arch.hpp>

#include "moe_compute.hpp"
#include "device/moe_compute_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const uint32_t output_height_shard_dim,
    const uint32_t intermediate_size,
    const bool has_bias,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& num_links,
    const std::optional<ttnn::CoreRangeSet>& mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<ttnn::GlobalSemaphore>& optional_cross_device_semaphore,
    const std::optional<ttnn::experimental::prim::detail::MoEActivationFunction>& activation_type,
    const bool compute_only,
    const std::optional<uint32_t>& num_shared_experts_per_device) {
    // bh_ring_size is intentionally not exposed on the public API. The matmul ring size is
    // auto-detected from the architecture: 8 on Blackhole, 12 on Wormhole (one per DRAM bank).
    // It remains a tunable knob on the ttnn::prim::moe_compute entry point.
    const std::optional<uint32_t> bh_ring_size = (tilize_input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 8 : 12;
    return ttnn::prim::moe_compute(
        tilize_input_tensor,
        tilize_expert_indices_tensor,
        tilize_expert_scores_tensor,
        tilize_expert_mapping_tensor,
        matmul_w0_w1_tensor,
        matmul_w2_tensor,
        layer_id,
        output_height_shard_dim,
        intermediate_size,
        has_bias,
        cluster_axis,
        topology,
        num_links,
        mux_core_range_set,
        output_memory_config,
        optional_output_tensor,
        optional_cross_device_semaphore,
        activation_type,
        compute_only,
        bh_ring_size,
        num_shared_experts_per_device);
}

std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores) {
    return ttnn::prim::get_moe_combine_cores(mesh_device, combine_token_parallel_cores, combine_data_parallel_cores);
}

ttnn::CoreRange get_moe_worker_mcast_bounding_box(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const uint32_t bh_ring_size) {
    return ttnn::prim::get_moe_worker_mcast_bounding_box(
        mesh_device, combine_token_parallel_cores, combine_data_parallel_cores, hidden_size, bh_ring_size);
}
}  // namespace ttnn::experimental
