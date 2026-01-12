// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::operations::matmul::program {

struct MatmulMultiCoreReuseOptimizedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle mm_kernel_in0_reader_id{};
        tt::tt_metal::KernelHandle mm_kernel_in1_reader_writer_id{};
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_src1{};
        tt::tt_metal::CBHandle cb_output{};
        uint32_t num_cores{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

struct MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory {
    using shared_variables_t = MatmulMultiCoreReuseOptimizedProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::matmul::program
