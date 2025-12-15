// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <vector>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async_new {

struct RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_forward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_forward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_reader_backward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_backward_kernel_id{};
    std::vector<CoreCoord> sender_worker_cores;
    uint32_t num_inputs{};
    uint32_t reader_sender_rt_offset{};
    uint32_t writer_sender_rt_offset{};
    uint32_t num_links{};
};

struct RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory {
    using shared_variables_t = RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables;

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    using operation_attributes_t = ring_attention_all_gather_async_new::operation_attributes_t;

    using tensor_args_t = ring_attention_all_gather_async_new::tensor_args_t;

    using tensor_return_value_t = ring_attention_all_gather_async_new::tensor_return_value_t;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<
        RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::shared_variables_t>
    ring_attention_all_gather_async_multi_core_with_workers_helper(
        tt::tt_metal::Program& program,
        const std::vector<Tensor>& input_tensor,
        IDevice* target_device,
        std::optional<IDevice*> forward_device,
        std::optional<IDevice*> backward_device,
        std::vector<Tensor>& output_tensor,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        ttnn::ccl::Topology topology,
        const std::vector<GlobalSemaphore>& semaphore,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
        CoreCoord core_grid_offset = CoreCoord(0, 0));
};
}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async_new
