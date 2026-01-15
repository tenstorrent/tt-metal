// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::all_gather_async {

struct operation_attributes_t {
    int32_t dim = 0;
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
    bool use_all_gather_async_llama_sharded = false;
    bool use_optimal_ccl_for_llama = false;
    std::optional<GlobalSemaphore> barrier_semaphore;
    bool using_persistent_buffers = false;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;
    bool reverse_order = false;
    std::optional<CoreRangeSet> sub_core_grid;

    operation_attributes_t(
        int32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis,
        bool use_all_gather_async_llama_sharded,
        bool use_optimal_ccl_for_llama,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        bool reverse_order,
        const std::optional<CoreRangeSet>& sub_core_grid) :
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        use_all_gather_async_llama_sharded(use_all_gather_async_llama_sharded),
        use_optimal_ccl_for_llama(use_optimal_ccl_for_llama),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        chunks_per_sync(chunks_per_sync),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        reverse_order(reverse_order),
        sub_core_grid(sub_core_grid) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("sub_device_id", sub_device_id);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("use_all_gather_async_llama_sharded", use_all_gather_async_llama_sharded);
        attrs.emplace_back("use_optimal_ccl_for_llama", use_optimal_ccl_for_llama);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("using_persistent_buffers", using_persistent_buffers);
        attrs.emplace_back("chunks_per_sync", chunks_per_sync);
        attrs.emplace_back("num_workers_per_link", num_workers_per_link);
        attrs.emplace_back("num_buffers_per_channel", num_buffers_per_channel);
        attrs.emplace_back("reverse_order", reverse_order);
        attrs.emplace_back("sub_core_grid", sub_core_grid);
        return attrs;
    }
};

struct tensor_args_t {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::operations::experimental::ccl::all_gather_async
