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
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllToAllAsyncGeneric {
    const uint32_t in_dim;
    const uint32_t out_dim;
    const uint32_t num_links;
    const uint32_t num_devices;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;

    AllToAllAsyncGeneric(
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t num_links,
        uint32_t num_devices,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis) :
        in_dim(in_dim),
        out_dim(out_dim),
        num_links(num_links),
        num_devices(num_devices),
        output_mem_config(output_mem_config),
        topology(topology),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("in_dim", in_dim);
        attrs.emplace_back("out_dim", out_dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("num_devices", num_devices);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("cluster_axis", cluster_axis);

        return attrs;
    }

    // Method declarations (implementations will be needed elsewhere)
    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors,
        const GlobalSemaphore& init_barrier_semaphore,
        const GlobalSemaphore& final_barrier_semaphore) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
};

// Add declaration for the AllToAll program function
tt::tt_metal::operation::ProgramWithCallbacks all_to_all_async_generic_program(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    std::optional<MeshCoordinate> target_device,
    std::optional<MeshCoordinate> forward_coord,
    std::optional<MeshCoordinate> backward_coord,
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t num_links,
    uint32_t num_devices,
    uint32_t device_index,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& final_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace operations {
namespace experimental {
namespace ccl {

// Add declaration for all_to_all_async
Tensor all_to_all_async_generic(
    const Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
