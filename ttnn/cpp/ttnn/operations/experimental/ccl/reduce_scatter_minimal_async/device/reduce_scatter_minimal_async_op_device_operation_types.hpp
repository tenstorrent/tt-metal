// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

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

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("optional_intermediate_mem_config", optional_intermediate_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("using_persistent_buffers", using_persistent_buffers);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("chunks_per_sync", chunks_per_sync);
        attrs.emplace_back("num_workers_per_link", num_workers_per_link);
        attrs.emplace_back("num_buffers_per_channel", num_buffers_per_channel);
        return attrs;
    }
};

struct ReduceScatterMinimalAsyncInputs {
    Tensor input_tensor;
    std::optional<Tensor> optional_intermediate_tensor;
    std::optional<Tensor> optional_output_tensor;
};

// Common validation function
void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor);

}  // namespace ttnn::experimental::prim
