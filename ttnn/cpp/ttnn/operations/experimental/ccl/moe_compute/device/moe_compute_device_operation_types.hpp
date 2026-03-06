// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeParams {
    // MoE compute attributes
    uint32_t layer_id;
    uint32_t output_height_shard_dim;
    uint32_t output_width_shard_dim;
    std::optional<uint32_t> cluster_axis;
    // a2a combine  attributes
    std::optional<uint32_t> combine_num_links;
    std::optional<uint32_t> combine_token_parallel_core_dim;
    std::optional<uint32_t> combine_data_parallel_core_dim;
    tt::tt_fabric::Topology combine_topology = tt::tt_fabric::Topology::Ring;
    CoreRangeSet mux_core_range_set{};
    std::optional<MemoryConfig> output_memory_config;
    std::optional<GlobalSemaphore> optional_cross_device_semaphore;

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("output_height_shard_dim", output_height_shard_dim);
        attrs.emplace_back("output_width_shard_dim", output_width_shard_dim);
        attrs.emplace_back("cluster_axis", cluster_axis);
        // a2a combine attributes
        attrs.emplace_back("combine_num_links", combine_num_links);
        attrs.emplace_back("combine_token_parallel_core_dim", combine_token_parallel_core_dim);
        attrs.emplace_back("combine_data_parallel_core_dim", combine_data_parallel_core_dim);
        attrs.emplace_back("combine_topology", combine_topology);
        attrs.emplace_back("mux_core_range_set", mux_core_range_set);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("optional_cross_device_semaphore", optional_cross_device_semaphore);
        return attrs;
    }
};

struct MoEComputeInputs {
    const ttnn::Tensor& tilize_input_tensor;
    const ttnn::Tensor& tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_expert_mapping_tensor;
    const ttnn::Tensor& matmul_w0_w1_tensor;
    const ttnn::Tensor& matmul_w2_tensor;
};

}  // namespace ttnn::experimental::prim
