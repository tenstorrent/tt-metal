// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl::slice_reshard_async {

struct operation_attributes_t {
    std::vector<IDevice*> devices;
    uint32_t dim;
    uint32_t output_dim_offset;
    uint32_t output_dim_shape;
    uint32_t cluster_axis;
    GlobalSemaphore final_semaphore;
    GlobalSemaphore barrier_semaphore;
    uint32_t num_links;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    uint32_t ring_size;

    // Constructor required because GlobalSemaphore is not default constructible
    operation_attributes_t(
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t output_dim_offset,
        uint32_t output_dim_shape,
        uint32_t cluster_axis,
        const GlobalSemaphore& final_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        uint32_t num_links,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        uint32_t ring_size) :
        devices(std::move(devices)),
        dim(dim),
        output_dim_offset(output_dim_offset),
        output_dim_shape(output_dim_shape),
        cluster_axis(cluster_axis),
        final_semaphore(final_semaphore),
        barrier_semaphore(barrier_semaphore),
        num_links(num_links),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        ring_size(ring_size) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("output_dim_offset", output_dim_offset);
        attrs.emplace_back("output_dim_shape", output_dim_shape);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("final_semaphore", final_semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("ring_size", ring_size);
        return attrs;
    }
};

struct tensor_args_t {
    Tensor input;
};

}  // namespace ttnn::operations::experimental::ccl::slice_reshard_async
