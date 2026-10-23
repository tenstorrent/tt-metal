// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::ccl {

// The program-cache hash and the profiler's op metadata both come from attribute_names + attribute_values(), every
// struct field must be added to both lists. Miss one and two different configs will share a cached program.
// NOTE: auto-reflection can't be used since it value-initializes the struct, and TensorSpec has no default ctor.
struct AllGatherParams {
    // Gather dim: counted from the end of the shape, so always negative. A padded shape can outrank the
    // logical one (a tiled rank-<2 tensor is promoted to rank 2, extra dims prepended), so this is the only
    // format valid against both. Use Shape::get_normalized_index() where a non-negative axis is needed.
    int32_t dim_from_end = -1;

    tt::tt_metal::TensorSpec output_spec;
    std::optional<uint32_t> cluster_axis;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config{};
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<bool, 2> axis_is_straight{};  // is the axis wired as a straight physical line
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective
    size_t packet_size = 0;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim_from_end",
        "output_spec",
        "cluster_axis",
        "fabric_config",
        "axis_topology",
        "axis_is_straight",
        "axis_num_devices",
        "axis_num_links",
        "num_devices",
        "packet_size",
        "subdevice_id",
        "sub_core_grid");
    auto attribute_values() const {
        return std::forward_as_tuple(
            dim_from_end,
            output_spec,
            cluster_axis,
            fabric_config,
            axis_topology,
            axis_is_straight,
            axis_num_devices,
            axis_num_links,
            num_devices,
            packet_size,
            subdevice_id,
            sub_core_grid);
    }

    // "true 2D" = a 2D fabric config with both axes active
    bool is_true_2d() const {
        return ::tt::tt_fabric::is_2d_fabric_config(fabric_config) && axis_num_devices[0] > 1 &&
               axis_num_devices[1] > 1;
    }

    // The active axis of an effectively-1D topology
    uint32_t get_1d_axis() const {
        TT_FATAL(!is_true_2d(), "get_1d_axis is undefined for true 2D topologies");
        if (cluster_axis.has_value()) {
            return cluster_axis.value();
        }
        for (uint32_t axis = 0; axis < axis_num_devices.size(); ++axis) {
            if (axis_num_devices[axis] > 1) {
                return axis;
            }
        }
        return 0;
    }
};

struct AllGatherInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_tensor;
};

}  // namespace ttnn::operations::ccl
