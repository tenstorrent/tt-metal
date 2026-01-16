// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::ccl::broadcast {

struct operation_attributes_t {
    MeshCoordinate sender_coord;  // No default constructor
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    tt::tt_fabric::Topology topology{};
    std::optional<uint32_t> cluster_axis;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    operation_attributes_t(
        const MeshCoordinate& sender_coord_,
        uint32_t num_links_,
        uint32_t ring_size_,
        const MemoryConfig& output_mem_config_,
        tt::tt_fabric::Topology topology_,
        std::optional<uint32_t> cluster_axis_,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id_) :
        sender_coord(sender_coord_),
        num_links(num_links_),
        ring_size(ring_size_),
        output_mem_config(output_mem_config_),
        topology(topology_),
        cluster_axis(cluster_axis_),
        sub_device_id(sub_device_id_) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("sender_coord", sender_coord);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct tensor_args_t {
    Tensor input_tensor;
};

}  // namespace ttnn::operations::ccl::broadcast
