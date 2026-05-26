// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

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
    const std::optional<uint32_t>& bh_ring_size) {
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
        bh_ring_size);
}

std::vector<ttnn::CoreCoord> get_moe_combine_cores(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set) {
    return ttnn::prim::get_moe_combine_cores(
        mesh_device, combine_token_parallel_cores, combine_data_parallel_cores, hidden_size, mux_core_range_set);
}

ttnn::CoreCoord get_moe_tilize_drain_core(
    ttnn::MeshDevice* mesh_device,
    const uint32_t combine_token_parallel_cores,
    const uint32_t combine_data_parallel_cores,
    const uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set) {
    return ttnn::prim::get_moe_tilize_drain_core(
        mesh_device, combine_token_parallel_cores, combine_data_parallel_cores, hidden_size, mux_core_range_set);
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
