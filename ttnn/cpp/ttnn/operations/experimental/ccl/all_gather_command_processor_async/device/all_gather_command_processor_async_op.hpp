// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllGatherCommandProcessorAsync {
    std::vector<IDevice*> devices;
    const uint32_t ring_size;
    const uint32_t dim;
    const GlobalSemaphore semaphore;
    const uint32_t num_links;
    const MemoryConfig output_memory_config;
    const ccl::Topology topology;
    std::optional<uint32_t> cluster_axis;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    AllGatherCommandProcessorAsync(
        std::vector<IDevice*> devices,
        uint32_t ring_size,
        uint32_t dim,
        GlobalSemaphore semaphore,
        uint32_t num_links,
        MemoryConfig output_memory_config,
        ccl::Topology topology,
        std::optional<uint32_t> cluster_axis,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) :
        devices(std::move(devices)),
        ring_size(ring_size),
        dim(dim),
        semaphore(std::move(semaphore)),
        num_links(num_links),
        output_memory_config(std::move(output_memory_config)),
        topology(topology),
        cluster_axis(cluster_axis),
        sub_device_id(sub_device_id) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("cluster_axis", cluster_axis);

        return attrs;
    }

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks all_gather_command_processor_async_multi_core_with_workers(
    const Tensor& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t ring_size,
    uint32_t ring_index,
    uint32_t dim,
    GlobalSemaphore semaphore,
    uint32_t num_links,
    ccl::Topology topology,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_command_processor_async(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

std::vector<Tensor> all_gather_command_processor_async(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);
}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
