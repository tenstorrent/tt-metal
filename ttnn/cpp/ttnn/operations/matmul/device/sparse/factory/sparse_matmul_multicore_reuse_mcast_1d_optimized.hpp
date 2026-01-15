// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"

namespace ttnn::prim {

struct SparseMatmulMultiCoreReuseMcast1DProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> kernels;
        std::vector<tt::tt_metal::CBHandle> cbs;
        bool extract_shard_sub_blocks{};
        CoreCoord start_core;
        std::vector<CoreCoord> cores;
        uint32_t num_cores_with_work{};
        ttnn::prim::Matmul1DType type{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

struct SparseMatmulMeshWorkloadMultiCoreReuseMcast1DFactory {
    using shared_variables_t = SparseMatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ttnn::prim::SparseMatmulParams& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
