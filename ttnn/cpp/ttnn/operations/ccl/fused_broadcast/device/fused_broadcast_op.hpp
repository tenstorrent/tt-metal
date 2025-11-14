// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
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

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

struct FusedBroadcast {
    const MeshCoordinate root_coord;
    const MeshCoordinate mesh_shape;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const tt::tt_fabric::Topology topology;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    FusedBroadcast(
        MeshCoordinate root_coord,
        MeshCoordinate mesh_shape,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        tt::tt_fabric::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id) :
        root_coord(root_coord),
        mesh_shape(mesh_shape),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        sub_device_id(sub_device_id) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("root_coord", root_coord);
        attrs.emplace_back("mesh_shape", mesh_shape);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
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
        const MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors,
        const GlobalSemaphore& coordination_semaphore,
        const GlobalSemaphore& barrier_semaphore) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks fused_broadcast_multicore(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    tt::tt_fabric::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace operations::ccl {

ttnn::Tensor fused_broadcast_impl(
    const Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    uint32_t num_links,
    uint32_t ring_size,
    tt::tt_fabric::Topology topology,
    const MemoryConfig& output_mem_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

}  // namespace operations::ccl

}  // namespace ttnn
