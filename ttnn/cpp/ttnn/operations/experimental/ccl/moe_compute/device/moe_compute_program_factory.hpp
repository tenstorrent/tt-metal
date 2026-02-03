// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_compute_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeMeshWorkloadFactory {
    struct shared_variables_t {
        // Tilize kernel handles
        std::vector<tt::tt_metal::KernelHandle> tilize_kernel_handles;

        // Tilize cores
        std::vector<CoreCoord> tilize_cores;

        // Matmul kernel handles
        std::vector<tt::tt_metal::KernelHandle> matmul_kernel_handles;

        // Matmul cores
        std::vector<CoreCoord> matmul_cores;

        // CB handle for shared global sharded tensor
        tt::tt_metal::CBHandle sharded_output_cb_handle;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoEComputeParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MoEComputeParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MoEComputeParams& operation_attributes,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
