// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::neighbor_pad {

struct operation_attributes_t {
    uint32_t dim = 0;
    uint32_t padding_left = 0;
    uint32_t padding_right = 0;
    std::string padding_mode;
    uint32_t cluster_axis = 0;
    GlobalSemaphore final_semaphore;    // Not default constructible
    GlobalSemaphore barrier_semaphore;  // Not default constructible
    uint32_t num_links = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    uint32_t ring_size = 0;
    std::optional<uint32_t> secondary_cluster_axis;
    std::optional<std::vector<uint32_t>> secondary_mesh_shape;

    // Constructor required because GlobalSemaphore is not default constructible
    operation_attributes_t(
        uint32_t dim,
        uint32_t padding_left,
        uint32_t padding_right,
        const std::string& padding_mode,
        uint32_t cluster_axis,
        const GlobalSemaphore& final_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        uint32_t num_links,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        uint32_t ring_size,
        std::optional<uint32_t> secondary_cluster_axis,
        const std::optional<std::vector<uint32_t>>& secondary_mesh_shape) :
        dim(dim),
        padding_left(padding_left),
        padding_right(padding_right),
        padding_mode(padding_mode),
        cluster_axis(cluster_axis),
        final_semaphore(final_semaphore),
        barrier_semaphore(barrier_semaphore),
        num_links(num_links),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        ring_size(ring_size),
        secondary_cluster_axis(secondary_cluster_axis),
        secondary_mesh_shape(secondary_mesh_shape ? std::make_optional(*secondary_mesh_shape) : std::nullopt) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("padding_left", padding_left);
        attrs.emplace_back("padding_right", padding_right);
        attrs.emplace_back("padding_mode", padding_mode);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("final_semaphore", final_semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("secondary_cluster_axis", secondary_cluster_axis);
        attrs.emplace_back("secondary_mesh_shape", secondary_mesh_shape);
        return attrs;
    }
};

struct tensor_args_t {
   Tensor input_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::experimental::ccl::neighbor_pad
