// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dropout_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::dropout::program {

struct DropoutSharedVariables {
    tt::tt_metal::KernelHandle dropout_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle dropout_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle dropout_kernel_group_1_id = 0;
    tt::tt_metal::KernelHandle dropout_kernel_group_2_id = 0;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
};

struct DropoutProgramFactory {
    using shared_variables_t = DropoutSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const DropoutParams& args, const DropoutInputs& tensor_args, Tensor& output);

    // operation_attributes_t with some seed value
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& output);
};

struct DropoutMeshWorkloadFactory {
    using shared_variables_t = DropoutSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    // Dropout generates N different programs, but they differ only in the per-device seed set as a runtime argument.
    // TODO: when heterogenous runtime arguments are supported, create a single program for all devices, and only
    // override the runtime arguments for each device. In addition, use `CachedMeshWorkload` instead of
    // `AdaptedCachedMeshWorkload`, as only a single `shared_variables_t` is needed.
    static cached_mesh_workload_t create_mesh_workload(
        const DropoutParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const DropoutInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DropoutParams& args,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::dropout::program
