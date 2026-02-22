// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SelectiveReduceCombineParams {
    uint32_t hidden_size;
    uint32_t batch_size;
    uint32_t seq_size;
    uint32_t select_experts_k;
    uint32_t experts;
    uint32_t num_links;

    std::optional<uint32_t> axis;
    tt::tt_fabric::Topology topology;

    uint32_t num_token_parallel_cores;
    uint32_t num_data_parallel_cores;
    std::vector<ttnn::CoreCoord> worker_cores;
    CoreRangeSet mux_core_range_set;
    ttnn::MemoryConfig output_memory_config;
    std::optional<GlobalSemaphore> optional_cross_device_semaphore;

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("hidden_size", hidden_size);
        attrs.emplace_back("batch_size", batch_size);
        attrs.emplace_back("seq_size", seq_size);
        attrs.emplace_back("select_experts_k", select_experts_k);
        attrs.emplace_back("experts", experts);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("axis", axis);
        attrs.emplace_back("num_token_parallel_cores", num_token_parallel_cores);
        attrs.emplace_back("num_data_parallel_cores", num_data_parallel_cores);
        attrs.emplace_back("worker_cores", worker_cores);
        attrs.emplace_back("mux_core_range_set", mux_core_range_set);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("optional_cross_device_semaphore", optional_cross_device_semaphore);

        return attrs;
    }
};

struct SelectiveReduceCombineTensors {
    ttnn::Tensor dense_input_tensor;
    ttnn::Tensor dense_metadata_tensor;
    ttnn::Tensor dense_token_maps_tensor;
    ttnn::Tensor dense_token_counts_tensor;
    std::optional<ttnn::Tensor> optional_output_tensor;
};

}  // namespace ttnn::experimental::prim
