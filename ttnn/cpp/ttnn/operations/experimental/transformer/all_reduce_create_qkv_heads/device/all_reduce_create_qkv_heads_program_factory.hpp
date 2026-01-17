// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_create_qkv_heads_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/core_coord.hpp>

#include <vector>

namespace ttnn::experimental::prim {

struct AllReduceCreateQkvHeadsSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    std::vector<CoreCoord> sender_worker_cores;
    tt::tt_metal::CBHandle cb_out{};
    tt::tt_metal::CBHandle cb_reduction{};
    std::vector<CoreCoord> output_cores_vec;
    tt::tt_metal::KernelHandle reduction_reader_kernel_id{};
    tt::tt_metal::KernelHandle reduction_writer_kernel_id{};
};

struct AllReduceCreateQkvHeadsMeshWorkloadFactory {
    using shared_variables_t = AllReduceCreateQkvHeadsSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
