// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#include "ttnn/run_operation.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct Broadcast {
    const MeshCoordinate sender_coord;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;

    Broadcast(
        MeshCoordinate sender_coord,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis) :
        sender_coord(std::move(sender_coord)),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("sender_coord", sender_coord);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("output_mem_config", output_mem_config);
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
        std::vector<Tensor>& output_tensors,
        const GlobalSemaphore& init_barrier_semaphore,
        const GlobalSemaphore& final_barrier_semaphore) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks broadcast_multicore(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& self_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

namespace operations::ccl {

Tensor broadcast(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

}  // namespace operations::ccl

}  // namespace ttnn
