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

struct AllReduceAsync {
    const uint32_t num_links;
    const uint32_t ring_size;
    const DataType dtype;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    bool use_noc1_only;
    bool use_optimal_ccl_for_llama;
    uint32_t cluster_axis;
    const distributed::MeshDevice* mesh_device;

    AllReduceAsync(
        uint32_t num_links,
        uint32_t ring_size,
        DataType dtype,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        bool use_noc1_only,
        bool use_optimal_ccl_for_llama,
        uint32_t cluster_axis,
        const distributed::MeshDevice* mesh_device) :
        num_links(num_links),
        ring_size(ring_size),
        dtype(dtype),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        use_noc1_only(use_noc1_only),
        use_optimal_ccl_for_llama(use_optimal_ccl_for_llama),
        cluster_axis(cluster_axis),
        mesh_device(mesh_device) {}
    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("use_noc1_only", use_noc1_only);
        attrs.emplace_back("use_optimal_ccl_for_llama", use_optimal_ccl_for_llama);
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
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace ccl {
namespace all_reduce_async_detail {
AllReduceAsync create_all_reduce_async_struct(
    const Tensor& input_tensor,
    uint32_t num_links,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama);

}  // namespace all_reduce_async_detail
}  // namespace ccl

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores);

tt::tt_metal::operation::ProgramWithCallbacks all_reduce_async_minimal_multi_core_with_workers(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    DataType output_dtype,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    std::optional<DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_noc1_only = false,
    bool use_optimal_ccl_for_llama = false);

std::vector<Tensor> all_reduce_async(
    const std::vector<Tensor>& input_tensors,
    Tensor& buffer_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_noc1_only = false,
    bool use_optimal_ccl_for_llama = false);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
