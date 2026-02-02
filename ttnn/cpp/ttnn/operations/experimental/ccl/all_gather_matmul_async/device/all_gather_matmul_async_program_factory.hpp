// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct AllGatherMatmulAsyncSharedVariables {
    std::variant<
        std::monostate,
        ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t,
        ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t>
        matmul_shared_variables;
    AllGatherProgramArtifacts all_gather_async_shared_variables;
};

struct AllGatherMatmulAsyncMeshWorkloadFactory {
    using shared_variables_t = AllGatherMatmulAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherMatmulAsyncParams& operation_attributes,
        const AllGatherMatmulAsyncInputs& tensor_args,
        AllGatherMatmulAsyncResult& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const Tensor& input_tensor,
        Tensor& all_gather_output_tensor,
        const Tensor& weight_tensor,
        Tensor& matmul_output_tensor,

        /* All Gather Params */
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
        bool using_persistent_buffers,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_direction_opt,
        std::optional<uint32_t> num_buffers_per_channel,
        CoreCoord core_grid_offset,

        /* Matmul Params */
        const std::optional<Tensor>& bias,
        bool bcast_batch,
        DeviceComputeKernelConfig compute_kernel_config,
        const operations::matmul::MatmulProgramConfig& program_config,
        bool untilize_out);
};

}  // namespace ttnn::experimental::prim
