// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async::program {

struct LlamaAllGatherMatmulAsyncSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id{};
    tt::tt_metal::KernelHandle worker_receiver_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> sender_worker_cores;
    std::vector<tt::tt_metal::CoreCoord> intermediate_cores_vec;
    uint32_t ring_index{};
    matmul::program::matmul_mcast_1d_common_override_variables_t matmul_shared_variables;
};

struct LlamaAllGatherMatmulAsyncProgramFactory {
    using shared_variables_t = LlamaAllGatherMatmulAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& args,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async::program
