// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct ReduceScatterMinimalAsync {
    std::vector<IDevice*> devices;
    uint32_t dim;
    uint32_t num_links;
    uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;

    ReduceScatterMinimalAsync(
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis = std::nullopt) :
        devices(std::move(devices)),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis) {}

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

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<uint32_t>& cluster_axis);

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0));

tt::tt_metal::operation::ProgramWithCallbacks ring_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0));

tt::tt_metal::operation::ProgramWithCallbacks line_reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    CoreCoord core_grid_offset = CoreCoord(0, 0));

namespace operations {
namespace experimental {
namespace ccl {

Tensor reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
