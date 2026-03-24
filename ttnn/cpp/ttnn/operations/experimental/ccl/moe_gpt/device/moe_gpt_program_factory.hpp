// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_gpt_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt::program {

struct MoEGPTSharedVariables {
    // CB handles for sharded circular buffers
    std::map<std::string, tt::tt_metal::CBHandle> cb_handles_sharded;

    // Kernel handles (matmul)
    std::vector<tt::tt_metal::KernelHandle> kernel_handles;

    // Kernel handles (tilize)
    std::vector<tt::tt_metal::KernelHandle> tilize_kernel_handles;

    // Kernel handle (combine dm1)
    std::optional<tt::tt_metal::KernelHandle> combine_kernel_handle;

    // Matmul cores
    std::vector<CoreCoord> worker_cores;

    // Tilize cores
    std::vector<CoreCoord> tilize_cores;

    // Combine cores
    std::vector<CoreCoord> combine_cores;
};

struct MoEGPTMeshWorkloadFactory {
    using shared_variables_t = MoEGPTSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& args,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::moe_gpt::program
