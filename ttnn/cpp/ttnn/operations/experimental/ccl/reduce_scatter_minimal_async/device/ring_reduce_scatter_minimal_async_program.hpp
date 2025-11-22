// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include <optional>
#include <vector>

// Includes for program factory
#include "reduce_scatter_minimal_async_types.hpp"
#include "reduce_scatter_minimal_async_program_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring {

struct RingReduceScatterMinimalAsyncProgramFactory {
    using shared_variables_t = ReduceScatterProgramArtifacts;

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

void override_runtime_args(
    tt::tt_metal::Program& program,
    const ReduceScatterProgramArtifacts& artifacts,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output);

tt::tt_metal::operation::ProgramWithCallbacks populate_program_legacy(
    tt::tt_metal::Program& program,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t&
        operation_attributes,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t& tensor_args,
    Tensor& intermediate_tensor,
    Tensor& output_tensor,
    const ::ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::mesh_runtime_params_t&
        mesh_runtime_params);

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::program::ring
