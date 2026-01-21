// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_device_operation_types.hpp"

namespace ttnn::experimental::prim {
struct DeepseekMinimalBroadcastProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
        tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
        tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
        tt::tt_metal::GlobalSemaphore semaphore;
        tt::tt_metal::GlobalSemaphore barrier_semaphore;
        uint32_t ring_index = 0;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const DeepseekMinimalBroadcastParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const DeepseekMinimalBroadcastInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DeepseekMinimalBroadcastParams& operation_attributes,
        const DeepseekMinimalBroadcastInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const DeepseekMinimalBroadcastParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const DeepseekMinimalBroadcastInputs& tensor_args,
        Tensor& output_tensor,
        const tt::tt_metal::GlobalSemaphore& semaphore,
        const tt::tt_metal::GlobalSemaphore& barrier_semaphore);
};

}  // namespace ttnn::experimental::prim
