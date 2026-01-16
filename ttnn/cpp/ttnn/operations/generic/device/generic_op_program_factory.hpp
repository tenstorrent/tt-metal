// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

namespace ttnn::operations::generic::program {

struct GenericMeshProgramFactory {
    struct shared_variables_t {
        uint32_t num_kernel_handles{};
        std::vector<tt::tt_metal::CBHandle> cb_handles;
    };

    struct mesh_shared_variables_t {
        shared_variables_t program_shared_variables;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<mesh_shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_mesh_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const tt::tt_metal::ProgramDescriptor& program_descriptor,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::generic::program
