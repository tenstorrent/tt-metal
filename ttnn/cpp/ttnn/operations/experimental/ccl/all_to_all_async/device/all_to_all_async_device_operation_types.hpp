// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <optional>

namespace ttnn::experimental::prim {

struct AllToAllAsyncParams {
    const uint32_t in_dim;
    const uint32_t out_dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const ttnn::MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const ttnn::GlobalSemaphore semaphore;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    AllToAllAsyncParams(
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        ttnn::GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id) :
        in_dim(in_dim),
        out_dim(out_dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id) {}

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("in_dim", in_dim);
        attrs.emplace_back("out_dim", out_dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        if (sub_device_id.has_value()) {
            attrs.emplace_back("sub_device_id", sub_device_id.value());
        }
        return attrs;
    }
};

struct AllToAllAsyncInputs {
    Tensor input_tensor;
    Tensor persistent_intermediate_buffer;
    Tensor persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
