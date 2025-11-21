// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::program {

struct StridedAllGatherMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        ttnn::operations::experimental::ccl::strided_all_gather_async::program::StridedAllGatherAsyncProgramFactory::
            shared_variables_t ag_shared_variables;
        ttnn::operations::experimental::minimal_matmul::minimal_matmul_override_variables_t mm_shared_variables;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static cached_program_t create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static cached_program_t strided_all_gather_minimal_matmul_async_program(
        /* General Params */
        const Tensor& input_tensor,
        Tensor& all_gather_output_tensor,
        const Tensor& weight_tensor,
        Tensor& matmul_output_tensor,
        IDevice* target_device,
        const MeshCoordinate& target_device_coord,
        const std::optional<MeshCoordinate>& forward_coord,
        const std::optional<MeshCoordinate>& backward_coord,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        ttnn::ccl::Topology topology,
        const std::vector<GlobalSemaphore>& semaphore,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> num_workers_per_direction_opt,
        std::optional<uint32_t> num_buffers_per_channel,
        CoreCoord core_grid_offset,

        /* Matmul Params */
        const std::optional<const Tensor>& bias,
        const std::optional<operations::unary::UnaryWithParam>& fused_activation,
        operations::experimental::minimal_matmul::MinimalMatmulConfig config,
        DeviceComputeKernelConfig compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::program
