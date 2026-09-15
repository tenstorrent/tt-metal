// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct AllToAllAsyncGenericParams {
    const uint32_t in_dim;
    const uint32_t out_dim;
    const uint32_t num_links;
    const uint32_t num_devices;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const std::optional<uint32_t> cluster_axis;
    const tt::tt_fabric::Topology axis_topology;
    const bool axis_is_straight;
    const std::vector<tt::tt_metal::CoreCoord> drain_logical_core_candidates;
    const std::vector<tt::tt_metal::CoreCoord> drain_virtual_cores;
};

struct AllToAllAsyncGenericInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
