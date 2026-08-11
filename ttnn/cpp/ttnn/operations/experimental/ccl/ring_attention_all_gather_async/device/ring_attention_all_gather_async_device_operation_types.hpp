// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operation.hpp"

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::experimental::prim {

struct RingAttentionAllGatherAsyncParams {
    std::vector<IDevice*> devices;
    int32_t dim = 0;
    uint32_t num_links = 1;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR;

    RingAttentionAllGatherAsyncParams(
        std::vector<IDevice*> devices,
        int32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        std::optional<uint32_t> cluster_axis,
        ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR) :
        devices(std::move(devices)),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        core_allocation_strategy(core_allocation_strategy) {}

    // Restrict program-cache hashing and the canonical key to structure-affecting fields only.
    // Excludes runtime-only `devices` (raw IDevice* pointers), `semaphore` (GlobalSemaphore objects
    // whose addresses are passed to runtime args), and `core_allocation_strategy` (effectively
    // constant; not part of the historical custom hash). `sub_device_id` is the structural source of
    // the worker-core range set the previous custom hash encoded.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim", "num_links", "ring_size", "output_mem_config", "topology", "sub_device_id", "cluster_axis");
    auto attribute_values() const {
        return std::forward_as_tuple(
            dim, num_links, ring_size, output_mem_config, topology, sub_device_id, cluster_axis);
    }
};

struct RingAttentionAllGatherAsyncInputs {
    std::vector<Tensor> input_tensor;
    std::vector<std::optional<Tensor>> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
