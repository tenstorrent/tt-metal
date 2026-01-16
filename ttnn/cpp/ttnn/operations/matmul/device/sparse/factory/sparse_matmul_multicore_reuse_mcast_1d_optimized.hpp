// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"

namespace ttnn::operations::sparse_matmul::program {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> kernels;
        std::vector<tt::tt_metal::CBHandle> cbs;
        bool extract_shard_sub_blocks{};
        CoreCoord start_core;
        std::vector<CoreCoord> cores;
        uint32_t num_cores_with_work{};
        matmul::Matmul1DType type{};
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

struct SparseMatmulMeshWorkloadMultiCoreReuseMcast1DFactory {
    using shared_variables_t = SparseMatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::sparse_matmul::program
