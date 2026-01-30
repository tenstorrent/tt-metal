// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores;
        std::vector<CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id;
        tt::tt_metal::KernelHandle in0_receiver_kernels_id;
        tt::tt_metal::KernelHandle in1_sender_kernels_id;
        tt::tt_metal::KernelHandle in1_receiver_kernels_id;
        bool transpose_core_grid;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor);
};

}  // namespace ttnn::experimental::prim
