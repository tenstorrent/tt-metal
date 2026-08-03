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

namespace ttnn::operations::ccl {

// The program-cache hash is computed automatically by reflecting over the members
// below and hashing each one. This is safe only because every field here is a
// stable, structural value.
//
// To add a volatile field later (e.g. a semaphore or raw pointer), do not rely
// on this auto-hashing -- define attribute_names + attribute_values() to list
// exactly what to hash.
struct AllGatherParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    std::optional<uint32_t> cluster_axis;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config{};
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective
    size_t packet_size = 0;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

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
