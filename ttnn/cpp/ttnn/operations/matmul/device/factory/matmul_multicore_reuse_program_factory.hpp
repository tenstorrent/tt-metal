// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores_x{};
        uint32_t num_blocks_y{};
        uint32_t num_blocks_x{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

struct MatmulMeshWorkloadMultiCoreReuseFactory {
    using shared_variables_t = MatmulMultiCoreReuseProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ttnn::prim::MatmulParams& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
