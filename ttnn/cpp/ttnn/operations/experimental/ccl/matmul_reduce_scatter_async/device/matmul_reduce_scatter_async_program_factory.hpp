// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/run_operation.hpp"
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::program {

struct MatmulReduceScatterAsyncSharedVariables {
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        matmul_override_callback;
    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        reduce_scatter_override_callback;

    // Shared parameters needed for override_runtime_arguments
    // Store mesh coordinate for proper tensor mapping
    ttnn::MeshCoordinate mesh_coord;
};

struct MatmulReduceScatterAsyncMeshWorkloadFactory {
    using shared_variables_t = MatmulReduceScatterAsyncSharedVariables;
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

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::program
