// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AllGatherParams {
    int32_t dim = 0;
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    tt::tt_fabric::Topology topology{};
    std::optional<uint32_t> cluster_axis;

    AllGatherParams(
        int32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        tt::tt_fabric::Topology topology,
        std::optional<uint32_t> cluster_axis) :
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        cluster_axis(cluster_axis) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct AllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::experimental::prim
