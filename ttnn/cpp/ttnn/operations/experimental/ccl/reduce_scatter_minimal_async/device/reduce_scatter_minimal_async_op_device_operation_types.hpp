// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include <algorithm>
#include <array>

#include <tt-metalium/runtime_args_data.hpp>
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Common argument bindings shared by standalone and fused CCL programs.
struct ReduceScatterProgramArtifacts {
    // Cache the binding objects, not their payload pointers: dispatch may relocate data().
    std::reference_wrapper<tt::tt_metal::RuntimeArgsData> reader_common_args;
    std::reference_wrapper<tt::tt_metal::RuntimeArgsData> writer_common_args;
    using RuntimeArgs = std::array<uint32_t, ttnn::ccl::ReduceScatterCommonArgs::count>;
    static RuntimeArgs collect_runtime_args(
        bool is_ring,
        const std::optional<GlobalSemaphore>& barrier,
        const std::vector<GlobalSemaphore>& semaphores,
        const Tensor& input,
        const Tensor& intermediate,
        const Tensor& output,
        const std::optional<Tensor>& penult = std::nullopt);

    void override_runtime_arguments(const RuntimeArgs& args) const {
        std::copy(args.begin(), args.end(), reader_common_args.get().data());
        std::copy(args.begin(), args.end(), writer_common_args.get().data());
    }
};

struct ReduceScatterMinimalAsyncParams {
    uint32_t dim;
    uint32_t num_links;
    uint32_t ring_size;
    MemoryConfig output_mem_config;
    std::optional<MemoryConfig> optional_intermediate_mem_config;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<GlobalSemaphore> barrier_semaphore;
    bool using_persistent_buffers;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    std::optional<ttnn::MeshShape> mesh_shape;
    std::vector<tt::tt_fabric::FabricNodeId> fabric_nodes;

    // Compile-time attributes drive the default program-cache reflection hash and the canonical key
    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "optional_intermediate_mem_config",
        "topology",
        "has_barrier_semaphore",
        "using_persistent_buffers",
        "sub_device_id",
        "cluster_axis",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "compute_kernel_config",
        "mesh_shape",
        "fabric_nodes");
    auto attribute_values() const {
        // Reference stored attributes; the computed presence flag must remain an owned value.
        return std::tuple_cat(
            std::tie(dim, num_links, ring_size, output_mem_config, optional_intermediate_mem_config, topology),
            std::make_tuple(barrier_semaphore.has_value()),
            std::tie(
                using_persistent_buffers,
                sub_device_id,
                cluster_axis,
                chunks_per_sync,
                num_workers_per_link,
                num_buffers_per_channel,
                compute_kernel_config,
                mesh_shape,
                fabric_nodes));
    }
};

struct ReduceScatterMinimalAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_intermediate_tensor;
    std::optional<Tensor> optional_output_tensor;
    // Ring contiguous fast path only (Ring topology, scatter dim != 0): caller-provided persistent
    // penult intermediate (see reduce_scatter_ring_penult_intermediate_staging_spec). When
    // absent and the contiguous path applies, create_output_tensors allocates one and returns it at index 2.
    std::optional<Tensor> optional_penult_intermediate_tensor;
};

}  // namespace ttnn::experimental::prim

#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.hpp"

namespace ttnn::experimental::prim {

// Forwarder kept for callers outside the experimental/ccl tree.
inline void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    ttnn::experimental::ccl::reduce_scatter_common_validates(
        input_tensor, topology, dim, num_links, ring_size, memory_config, optional_output_tensor);
}

}  // namespace ttnn::experimental::prim
