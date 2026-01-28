// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt_stl/reflection.hpp>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct AllGatherConcatParams {
    uint32_t dim = 0;
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology{};
    GlobalSemaphore semaphore;  // Not default constructible
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    uint32_t num_heads = 0;
    bool use_noc1_only = false;
    uint32_t cluster_axis = 0;

    AllGatherConcatParams(
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        uint32_t num_heads,
        bool use_noc1_only,
        uint32_t cluster_axis) :
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        num_heads(num_heads),
        use_noc1_only(use_noc1_only),
        cluster_axis(cluster_axis) {}
    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("num_heads", num_heads);
        attrs.emplace_back("use_noc1_only", use_noc1_only);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct AllGatherConcatInputs {
    Tensor input_tensor;
    Tensor buffer_tensor;
};

}  // namespace ttnn::experimental::prim
