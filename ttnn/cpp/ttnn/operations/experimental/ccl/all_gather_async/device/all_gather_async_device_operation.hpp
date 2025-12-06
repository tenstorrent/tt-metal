// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_async_device_operation_types.hpp"
#include "all_gather_async_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::all_gather_async {

struct AllGatherAsyncDeviceOperation {
    using operation_attributes_t = all_gather_async::operation_attributes_t;
    using tensor_args_t = all_gather_async::tensor_args_t;
    using spec_return_value_t = all_gather_async::spec_return_value_t;
    using tensor_return_value_t = all_gather_async::tensor_return_value_t;
    using program_factory_t = std::variant<DefaultMeshWorkloadFactory, LlamaShardedMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        const std::optional<uint32_t>& cluster_axis,
        bool use_optimal_ccl_for_llama,
        bool use_all_gather_async_llama_sharded,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        const std::optional<uint32_t>& chunks_per_sync,
        const std::optional<uint32_t>& num_workers_per_link,
        const std::optional<uint32_t>& num_buffers_per_channel,
        bool reverse_order,
        const std::optional<CoreRangeSet>& sub_core_grid,
        const MeshDevice* mesh_device);
};

}  // namespace ttnn::operations::experimental::ccl::all_gather_async

namespace ttnn {
using AllGatherProgramArtifacts = operations::experimental::ccl::all_gather_async::AllGatherProgramArtifacts;

// Builder function that creates kernels and returns artifacts
AllGatherProgramArtifacts build_all_gather_async_minimal_default_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

// Runtime argument override function
void all_gather_async_minimal_default_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::vector<GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& output);

// TODO: Remove this once dependent ops are migrated to the new infra
tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_default_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    bool reverse_order = false,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn
namespace ttnn::prim {
constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::prim::all_gather_async",
    ttnn::operations::experimental::ccl::all_gather_async::AllGatherAsyncDeviceOperation>();
}  // namespace ttnn::prim
