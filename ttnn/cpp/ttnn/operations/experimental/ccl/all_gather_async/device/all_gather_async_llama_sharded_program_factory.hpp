// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include <tt-metalium/runtime_args_data.hpp>

#include "all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct LlamaShardedMeshWorkloadFactory {
    struct shared_variables_t {
        // Cache binding objects; data() follows dispatch/trace payload relocation.
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> reader_args;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> writer_args;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherAsyncParams& operation_attributes,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const AllGatherAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
