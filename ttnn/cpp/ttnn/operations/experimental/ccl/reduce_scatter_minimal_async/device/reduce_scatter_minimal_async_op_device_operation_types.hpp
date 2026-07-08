// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
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

// Shared struct for program artifacts - used for caching kernel handles and core info
struct ReduceScatterProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t num_directions_per_link;
    uint32_t num_workers_per_direction;
    uint32_t num_mux_cores_per_direction_per_link;
    uint32_t num_cores_per_link;
    uint32_t normalized_dim;
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
        "compute_kernel_config");
    auto attribute_values() const {
        return std::make_tuple(
            dim,
            num_links,
            ring_size,
            output_mem_config,
            optional_intermediate_mem_config,
            topology,
            barrier_semaphore.has_value(),
            using_persistent_buffers,
            sub_device_id,
            cluster_axis,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            compute_kernel_config);
    }
};

struct ReduceScatterMinimalAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_intermediate_tensor;
    std::optional<Tensor> optional_output_tensor;
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
