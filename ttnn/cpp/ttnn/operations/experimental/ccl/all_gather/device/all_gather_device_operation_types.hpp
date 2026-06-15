// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AllGatherParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    std::optional<uint32_t> cluster_axis;
    // Per-axis info.
    // An inactive axis has num_devices = 1, num_links = 0, and Linear topology.
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    // Number of devices participating in the collective
    uint32_t num_devices = 0;
    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    AllGatherParams(
        int32_t dim,
        MemoryConfig output_mem_config,
        std::optional<uint32_t> cluster_axis,
        std::array<tt::tt_fabric::Topology, 2> axis_topology,
        std::array<uint32_t, 2> axis_num_devices,
        std::array<uint32_t, 2> axis_num_links,
        uint32_t num_devices,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
        std::optional<CoreRangeSet> sub_core_grid) :
        dim(dim),
        output_mem_config(std::move(output_mem_config)),
        cluster_axis(cluster_axis),
        axis_topology(axis_topology),
        axis_num_devices(axis_num_devices),
        axis_num_links(axis_num_links),
        num_devices(num_devices),
        subdevice_id(subdevice_id),
        sub_core_grid(std::move(sub_core_grid)) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("axis0_topology", axis_topology[0]);
        attrs.emplace_back("axis1_topology", axis_topology[1]);
        attrs.emplace_back("axis0_num_devices", axis_num_devices[0]);
        attrs.emplace_back("axis1_num_devices", axis_num_devices[1]);
        attrs.emplace_back("axis0_num_links", axis_num_links[0]);
        attrs.emplace_back("axis1_num_links", axis_num_links[1]);
        attrs.emplace_back("num_devices", num_devices);
        attrs.emplace_back("sub_core_grid", sub_core_grid);
        return attrs;
    }
};

struct AllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::experimental::prim
