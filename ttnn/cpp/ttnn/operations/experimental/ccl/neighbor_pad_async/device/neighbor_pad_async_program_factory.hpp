// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl::neighbor_pad {

struct NeighborPadAsyncSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    // Additional local-copy workers (do not send over fabric)
    std::vector<tt::tt_metal::KernelHandle> local_reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> local_writer_kernel_ids;
    std::vector<tt::tt_metal::CoreCoord> local_copy_core_coords;  // logical coords used for local copy
    uint32_t num_links = 0;
    uint32_t num_directions = 0;  // Always 2 (left/right padding)
};

struct NeighborPadAsyncMeshWorkloadFactory {
    using shared_variables_t = NeighborPadAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ccl::neighbor_pad
