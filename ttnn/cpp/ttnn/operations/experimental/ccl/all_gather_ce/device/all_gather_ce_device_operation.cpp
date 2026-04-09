// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_ce_device_operation.hpp"

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

AllGatherCeDeviceOperation::program_factory_t AllGatherCeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AllGatherCeDefaultMeshWorkloadFactory{};
}

void AllGatherCeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        !args.use_all_gather_async_llama_sharded,
        "ttnn.experimental.all_gather_ce only implements the minimal default path; use all_gather_async for the "
        "llama-sharded variant.");
    TT_FATAL(
        !args.use_all_gather_async_via_broadcast,
        "ttnn.experimental.all_gather_ce only implements the minimal default path; use all_gather_async for "
        "via-broadcast.");
    TT_FATAL(args.semaphore.size() == 2, "all_gather_ce requires exactly 2 global semaphores");
    AllGatherAsyncDeviceOperation::validate_on_program_cache_miss(args, tensor_args);
}

AllGatherCeDeviceOperation::spec_return_value_t AllGatherCeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return AllGatherAsyncDeviceOperation::compute_output_specs(args, tensor_args);
}

AllGatherCeDeviceOperation::tensor_return_value_t AllGatherCeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return AllGatherAsyncDeviceOperation::create_output_tensors(args, tensor_args);
}

ttsl::hash::hash_t AllGatherCeDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherCeDeviceOperation::compute_program_hash is called");

    auto subdevice_id = args.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    if (args.sub_core_grid.has_value()) {
        subdevice_core_range_set = subdevice_core_range_set.intersection(args.sub_core_grid.value());
    }

    constexpr uint32_t kAllGatherCeProgramTag = 0xACEC0DEu;
    return tt::tt_metal::operation::hash_operation<AllGatherCeDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_mem_config,
        args.topology,
        args.cluster_axis,
        args.barrier_semaphore.has_value(),
        args.using_persistent_buffers,
        args.chunks_per_sync,
        args.num_workers_per_link,
        args.num_buffers_per_channel,
        args.use_all_gather_async_llama_sharded,
        args.use_optimal_ccl_for_llama,
        args.reverse_order,
        subdevice_core_range_set,
        tensor_args,
        kAllGatherCeProgramTag);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor all_gather_ce(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    bool use_all_gather_async_via_broadcast,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const MeshDevice* optional_mesh_device) {
    auto [params, inputs] = experimental::prim::all_gather_async_build_operation_args(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        use_all_gather_async_llama_sharded,
        use_all_gather_async_via_broadcast,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        reverse_order,
        sub_core_grid,
        optional_mesh_device);
    return ttnn::device_operation::launch<experimental::prim::AllGatherCeDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim
