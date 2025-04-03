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
#include "cpp/ttnn/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllReduceAsync {
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    bool enable_persistent_fabric_mode;

    AllReduceAsync(
        std::optional<IDevice*> forward_device,
        std::optional<IDevice*> backward_device,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        bool enable_persistent_fabric_mode) :
        forward_device(forward_device),
        backward_device(backward_device),
        num_links(num_links),
        ring_size(ring_size),
        ring_index(ring_index),
        output_mem_config(output_mem_config),
        topology(topology),
        semaphore(semaphore),
        sub_device_id(sub_device_id),
        enable_persistent_fabric_mode(enable_persistent_fabric_mode) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);

        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    const tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace ccl {
namespace all_reduce_async_detail {
AllReduceAsync create_all_reduce_async_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode);

}  // namespace all_reduce_async_detail
}  // namespace ccl

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, bool persistent_fabric_mode, const CoreRangeSet& available_cores);

tt::tt_metal::operation::ProgramWithCallbacks all_reduce_async_minimal_multi_core_with_workers(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode);

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    bool enable_persistent_fabric_mode = false);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
