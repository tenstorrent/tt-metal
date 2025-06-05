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

struct AllGatherConcat {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const uint32_t num_heads;
    const uint32_t cluster_axis;

    AllGatherConcat(
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        uint32_t num_heads,
        uint32_t cluster_axis) :
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        num_heads(num_heads),
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
        attrs.emplace_back("num_heads", num_heads);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

// All Gather Variants
std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);

tt::tt_metal::operation::ProgramWithCallbacks all_gather_concat_llama_sharded(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const uint32_t num_heads);

tt::tt_metal::operation::ProgramWithCallbacks all_gather_concat_llama_sharded_subgrids(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const uint32_t num_heads);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_concat(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const uint32_t num_heads,
    const MemoryConfig& memory_config,
    const std::optional<uint32_t> num_links = std::nullopt,
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
