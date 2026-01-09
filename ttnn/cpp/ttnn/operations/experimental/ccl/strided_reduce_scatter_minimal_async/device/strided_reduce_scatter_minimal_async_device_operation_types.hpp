// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "tt-metalium/sub_device.hpp"

namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_minimal_async::detail {

struct operation_attributes_t {
    uint32_t dim;
    uint32_t num_links;
    uint32_t ring_size;
    MemoryConfig output_mem_config;
    std::optional<MemoryConfig> optional_intermediate_mem_config;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<GlobalSemaphore> barrier_semaphore;
    bool using_persistent_buffers;
    tt::tt_metal::SubDeviceId sub_device_id;
    std::optional<uint32_t> cluster_axis;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;
    // Strided-specific parameters
    std::optional<uint32_t> tiles_per_chunk;
    std::optional<uint32_t> mm_cores_y;
    std::optional<uint32_t> mm_block_ht;
    std::optional<uint32_t> mm_block_wt;

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("optional_intermediate_mem_config", optional_intermediate_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("using_persistent_buffers", using_persistent_buffers);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("chunks_per_sync", chunks_per_sync);
        attrs.emplace_back("num_workers_per_link", num_workers_per_link);
        attrs.emplace_back("num_buffers_per_channel", num_buffers_per_channel);
        // Strided-specific attributes
        attrs.emplace_back("tiles_per_chunk", tiles_per_chunk);
        attrs.emplace_back("mm_cores_y", mm_cores_y);
        attrs.emplace_back("mm_block_ht", mm_block_ht);
        attrs.emplace_back("mm_block_wt", mm_block_wt);
        return attrs;
    }
};

struct tensor_args_t {
    const ttnn::Tensor& input_tensor;
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor;
    const std::optional<ttnn::Tensor>& optional_output_tensor;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_minimal_async::detail
