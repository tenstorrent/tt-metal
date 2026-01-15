// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::ccl::slice_reshard_async::program {

struct SliceReshardAsyncSharedVariables {
    uint32_t num_directions = 0;
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
};

struct SliceReshardAsyncProgramFactory {
    using shared_variables_t = SliceReshardAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& args,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinate& mesh_coord,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ccl::slice_reshard_async::program
