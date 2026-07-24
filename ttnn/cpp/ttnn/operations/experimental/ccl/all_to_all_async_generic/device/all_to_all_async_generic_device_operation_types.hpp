// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/sub_device.hpp>
#include <optional>
#include <tuple>

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

    static constexpr auto attribute_names = std::forward_as_tuple(
        "in_dim",
        "out_dim",
        "num_links",
        "num_devices",
        "output_mem_config",
        "topology",
        "sub_device_id",
        "cluster_axis");
    auto attribute_values() const {
        return std::make_tuple(
            in_dim, out_dim, num_links, num_devices, output_mem_config, topology, sub_device_id, cluster_axis);
    }
};

struct AllToAllAsyncGenericInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
