// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail {

// Use StridedReduceScatterProgramArtifacts as the shared variables type for consistency
using RingStridedReduceScatterSharedVariables = StridedReduceScatterProgramArtifacts;

struct RingStridedReduceScatterMeshWorkloadFactory {
    using shared_variables_t = RingStridedReduceScatterSharedVariables;
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

// Builder function for ring topology - creates program artifacts
StridedReduceScatterProgramArtifacts build_ring_strided_reduce_scatter_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler>& mm_fused_op_signaler,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    std::optional<uint32_t> mm_cores_y,
    uint32_t mm_block_ht,
    uint32_t mm_block_wt,
    std::optional<uint32_t> mm_N_full_block_wt,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    // Rolling L1 window over the fused matmul's output, in M blocks. When set, the input tensor
    // holds only mm_window_blocks M blocks per core (slot m % mm_window_blocks) instead of the whole
    // matmul result, so its height no longer describes the reduce-scatter's geometry — pass the true
    // height in tiles as mm_logical_Ht. Both unset = the input tensor is the full matmul output.
    std::optional<uint32_t> mm_window_blocks = std::nullopt,
    std::optional<uint32_t> mm_logical_Ht = std::nullopt,
    // Caller-owned RS->MM credit array for the rolling window; allocated per program when absent.
    const std::optional<const Tensor>& mm_credit_counters = std::nullopt,
    // Optional fused addcmul at the final RS write step.
    // output = addcmul_a + fused_ternary_scalar * rs_result * addcmul_b
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor1 = std::nullopt,  // residual a [M, D/tp]
    const std::optional<const Tensor>& addcmul_input_tensor2 = std::nullopt,  // gate b [1, D/tp]
    // Caller-owned scratch for the per-MM-core progress counters (fused MM->RS signaling only): uint32
    const std::optional<const Tensor>& mm_progress_counters = std::nullopt);

// Override runtime arguments helper for ring topology
void ring_strided_reduce_scatter_async_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_links,
    uint32_t num_directions_per_link,
    uint32_t num_workers_per_direction,
    uint32_t num_mux_cores_per_direction_per_link,
    uint32_t num_cores_per_link,
    const std::optional<tt::tt_metal::GlobalSemaphore>& barrier_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphore,
    const Tensor& input,
    const Tensor& intermed,
    const Tensor& output,
    uint32_t reader_addcmul_rt_arg_offset = 0,
    const std::optional<const Tensor>& addcmul_a = std::nullopt,
    const std::optional<const Tensor>& addcmul_b = std::nullopt);

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail
