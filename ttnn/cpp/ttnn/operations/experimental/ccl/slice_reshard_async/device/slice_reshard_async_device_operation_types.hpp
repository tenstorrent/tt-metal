// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

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

namespace ttnn::experimental::prim {

struct SliceReshardAsyncParams {
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
    SliceReshardAsyncParams(
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

    // Program-cache hash / canonical-key fields.
    static constexpr auto attribute_names = std::make_tuple(
        "dim",
        "output_dim_offset",
        "output_dim_shape",
        "cluster_axis",
        "num_links",
        "output_mem_config",
        "topology",
        "ring_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            dim, output_dim_offset, output_dim_shape, cluster_axis, num_links, output_mem_config, topology, ring_size);
    }
};

}  // namespace ttnn::experimental::prim
