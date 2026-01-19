// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct AllGatherConcatSharedVariables {
    std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
    uint32_t num_concat_worker_cores = 0;
    tt::tt_metal::CBHandle cb_q_output{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    tt::tt_metal::KernelHandle concat_reader_kernel_id{};
};

struct AllGatherConcatMeshWorkloadFactory {
    using shared_variables_t = AllGatherConcatSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherConcatParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherConcatParams& operation_attributes,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherConcatParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherConcatInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
