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

enum class AllGatherAsyncVersion {
    GENERIC = 0,
    LLAMA_MINIMAL_SHARDED = 1,
    MINIMAL_INTERLEAVED = 2,
};

struct AllGatherAsync {
    std::vector<IDevice*> devices;
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
    bool use_optimal_ccl_for_llama;
    std::optional<uint32_t> chunks_per_sync;
    std::optional<uint32_t> num_workers_per_link;
    std::optional<uint32_t> num_buffers_per_channel;

    AllGatherAsync(
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> cluster_axis,
        bool use_optimal_ccl_for_llama,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel) :
        devices(std::move(devices)),
        dim(dim),
        num_links(num_links),
        ring_size(ring_size),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        cluster_axis(cluster_axis),
        use_optimal_ccl_for_llama(use_optimal_ccl_for_llama),
        chunks_per_sync(chunks_per_sync),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel) {}

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
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("use_optimal_ccl_for_llama", use_optimal_ccl_for_llama);
        attrs.emplace_back("chunks_per_sync", chunks_per_sync);
        attrs.emplace_back("num_worker_per_link", num_workers_per_link);
        attrs.emplace_back("num_buffers_per_channel", num_buffers_per_channel);
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

    AllGatherAsyncVersion select_version(const Tensor& input_tensor) const;
};

// All Gather Variants
std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset = CoreCoord(0, 0));
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_multi_core_with_workers(
    const Tensor& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    GlobalSemaphore semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id);
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved(
    const Tensor& input_tensor,
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
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel);
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
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
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset = CoreCoord(0, 0));
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_llama_sharded(
    const Tensor& input_tensor,
    IDevice* target_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool use_optimal_ccl_for_llama);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_async(
    const Tensor& input_tensor,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false);

Tensor all_gather_async(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    bool use_optimal_ccl_for_llama = false,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

std::vector<Tensor> all_gather_async(
    const std::vector<Tensor>& input_tensors,
    uint32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false);

Tensor all_gather_async(
    const Tensor& input_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false);

std::vector<Tensor> all_gather_async(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
