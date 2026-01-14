// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async {

struct operation_attributes_t {
    matmul::operation_attributes_t matmul_struct;
    std::vector<IDevice*> devices;
    uint32_t dim{};
    uint32_t num_links{};
    uint32_t ring_size{};
    tt::tt_metal::MemoryConfig output_memory_config;
    ttnn::ccl::Topology topology{};
    tt::tt_metal::GlobalSemaphore semaphore;  // Not default constructible
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;

    // Constructor required because GlobalSemaphore is not default constructible
    operation_attributes_t(
        matmul::operation_attributes_t matmul_struct,
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        tt::tt_metal::MemoryConfig output_memory_config,
        ttnn::ccl::Topology topology,
        tt::tt_metal::GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis) :
        matmul_struct(std::move(matmul_struct)),
        devices(std::move(devices)),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_memory_config(std::move(output_memory_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("devices", devices);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        if (sub_device_id.has_value()) {
            attrs.emplace_back("sub_device_id", sub_device_id.value());
        }
        if (cluster_axis.has_value()) {
            attrs.emplace_back("cluster_axis", cluster_axis.value());
        }
        return attrs;
    }
};

struct tensor_return_value_t {
    Tensor mm;
    Tensor aggregated;
};

struct spec_return_value_t {
    TensorSpec mm;
    TensorSpec aggregated;
};

struct tensor_args_t {
    Tensor input0;
    Tensor input1;
    Tensor intermediate;
};

}  // namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async
