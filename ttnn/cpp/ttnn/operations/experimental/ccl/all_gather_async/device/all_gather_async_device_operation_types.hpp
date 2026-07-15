// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

struct AllGatherAsyncParams {
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
    bool use_all_gather_async_via_broadcast = true;
    std::optional<GlobalSemaphore> barrier_semaphore;
    bool using_persistent_buffers = false;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;
    bool reverse_order = false;
    std::optional<CoreRangeSet> sub_core_grid;
    // Gather only a single index along dim 0 (gather dim must be != 0); output dim-0 collapses to 1.
    std::optional<uint32_t> batch_slice_idx;
    // Gather only the valid prefix (elements, tile-aligned) along the height (rank-2) gather dim of an
    // overallocated input; limits data movement without shrinking the output.
    std::optional<uint32_t> valid_gather_extent;

    AllGatherAsyncParams(
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
        bool use_all_gather_async_via_broadcast,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        bool reverse_order,
        const std::optional<CoreRangeSet>& sub_core_grid,
        std::optional<uint32_t> batch_slice_idx = std::nullopt,
        std::optional<uint32_t> valid_gather_extent = std::nullopt) :
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
        use_all_gather_async_via_broadcast(use_all_gather_async_via_broadcast),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        chunks_per_sync(chunks_per_sync),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        reverse_order(reverse_order),
        sub_core_grid(sub_core_grid),
        batch_slice_idx(batch_slice_idx),
        valid_gather_extent(valid_gather_extent) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "sub_device_id",
        "cluster_axis",
        "use_all_gather_async_llama_sharded",
        "use_optimal_ccl_for_llama",
        "use_all_gather_async_via_broadcast",
        "barrier_semaphore_present",
        "using_persistent_buffers",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "reverse_order",
        "sub_core_grid",
        // The tiled path patches the batch index and valid extent through runtime args. The direct-broadcast
        // factory currently builds its row mapping into runtime args at program creation, so include both values
        // in its cache key. Its valid extent also changes the compact output shape.
        "batch_slice_present",
        "direct_broadcast_batch_slice_idx",
        "direct_broadcast_valid_gather_extent");
    auto attribute_values() const {
        return std::make_tuple(
            dim,
            num_links,
            ring_size,
            output_mem_config,
            topology,
            sub_device_id,
            cluster_axis,
            use_all_gather_async_llama_sharded,
            use_optimal_ccl_for_llama,
            use_all_gather_async_via_broadcast,
            barrier_semaphore.has_value(),
            using_persistent_buffers,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            reverse_order,
            sub_core_grid,
            batch_slice_idx.has_value(),
            use_all_gather_async_via_broadcast ? batch_slice_idx : std::nullopt,
            use_all_gather_async_via_broadcast ? valid_gather_extent : std::nullopt);
    }
};

struct AllGatherAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
