// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "width_sharded_all_reduce_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::prim {

struct WidthShardedAllReduceSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    tt::tt_metal::KernelHandle reduction_reader_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
    CoreRangeSet output_tensor_cores;
    tt::tt_metal::CBHandle cb_out{};
    tt::tt_metal::CBHandle cb_reduction{};
    std::shared_ptr<Tensor> scratch;
    GlobalSemaphore out_ready_semaphore;
};

struct WidthShardedAllReduceMeshWorkloadFactory {
    using shared_variables_t = WidthShardedAllReduceSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const WidthShardedAllReduceParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const WidthShardedAllReduceInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const WidthShardedAllReduceParams& operation_attributes,
        const WidthShardedAllReduceInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const WidthShardedAllReduceParams& operation_attributes,
        const ttnn::MeshCoordinate& coord,
        const WidthShardedAllReduceInputs& tensor_args,
        Tensor& output_tensor,
        const std::shared_ptr<Tensor>& scratch,
        const GlobalSemaphore& out_ready_semaphore);
};

}  // namespace ttnn::prim
