// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

// Include RS types
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_ring_program_factory.hpp"

#include <sstream>
#include <type_traits>

using namespace tt::constants;

// Import the RS program artifacts type
using ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts;

// Forward declarations for functions defined in namespace ttnn
// (defined in strided_reduce_scatter_async_program.cpp)
namespace ttnn {
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
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt,
    std::optional<uint32_t> mm_N_block_wt,
    std::optional<uint32_t> chunk_width_in_mm_blocks);

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
    const Tensor& output);
}  // namespace ttnn

namespace ttnn::experimental::prim {

MinimalMatmulStridedReduceScatterAsyncProgramFactory::cached_mesh_workload_t
MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_mesh_workload(
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MinimalMatmulStridedReduceScatterAsyncParams& attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(range);

        // Override RS runtime arguments
        // output_tensor[0] = MM output = RS input
        // output_tensor[1] = RS intermediate
        // output_tensor[2] = RS output
        ::ttnn::ring_strided_reduce_scatter_async_helper_override_runtime_arguments(
            program,
            shared_variables.rs_shared_variables.reader_kernel_id,
            shared_variables.rs_shared_variables.writer_kernel_id,
            shared_variables.rs_shared_variables.all_cores,
            attributes.num_links,
            shared_variables.rs_shared_variables.num_directions_per_link,
            shared_variables.rs_shared_variables.num_workers_per_direction,
            shared_variables.rs_shared_variables.num_mux_cores_per_direction_per_link,
            shared_variables.rs_shared_variables.num_cores_per_link,
            attributes.barrier_semaphore,
            attributes.semaphore,
            output_tensor.at(0),   // RS input = MM output
            output_tensor.at(1),   // RS intermediate
            output_tensor.at(2));  // RS output

        // Override MM runtime arguments
        auto cached_program_proxy = ttnn::experimental::prim::MinimalMatmulProgramFactory::cached_program_t::proxy(
            program, shared_variables.mm_shared_variables);

        ttnn::experimental::prim::MinimalMatmulProgramFactory::override_runtime_arguments(
            cached_program_proxy,
            attributes.matmul_struct,
            {tensor_args.input_tensor, tensor_args.weight_tensor, tensor_args.bias, std::nullopt},
            {output_tensor.at(0)});
    }
}

ttnn::device_operation::CachedProgram<MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t>
minimal_matmul_strided_reduce_scatter_async_program(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    Tensor& rs_intermediate_tensor,
    Tensor& rs_output_tensor,

    /* Reduce Scatter Params */
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord reduce_scatter_core_grid_offset,
    std::optional<uint32_t> chunk_width_in_mm_blocks,

    /* Matmul Params */
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    ttnn::experimental::prim::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt::tt_metal::Program program{};

    // Derive matmul geometry parameters for the RS factory.
    // The matmul factory normally transposes its core grid when M > N, but
    // transposition is disabled when fusing with SRS (the RS iteration structure
    // requires mm_N_block_wt <= slice_Wt, which can be violated when transposing).
    // So we always use the non-transposed layout: M parallelized on y, N on x.
    constexpr uint32_t TILE_WIDTH = 32;
    uint32_t mm_cores_y_val = config.compute_with_storage_grid_size.y;
    uint32_t mm_block_ht_val = config.M_block_size;
    uint32_t mm_block_wt_val = config.N_block_size;

    // Compute mm_N_block_wt: total N tiles per core (N parallelized on x)
    uint32_t N_tiles = weight_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_cores_x = config.compute_with_storage_grid_size.x;
    uint32_t padded_N_tiles = tt::round_up(N_tiles, num_cores_x);
    uint32_t mm_N_block_wt_val = padded_N_tiles / num_cores_x;

    // =========================================================================
    // STEP 1: Create the Reduce Scatter program FIRST
    //
    // The RS factory creates a semaphore on the RS reader cores and records
    // their NOC coordinates. This info is captured in srs_fused_op_signaler,
    // which is then passed to the matmul factory in step 2.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler =
        ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler();
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> empty_rs_fused_op_signaler = std::nullopt;

    auto rs_shared_variables = ::ttnn::build_ring_strided_reduce_scatter_async_program_artifacts(
        program,
        matmul_output_tensor,    // RS input = MM output
        rs_intermediate_tensor,  // RS intermediate (scratch)
        target_device_coord,
        forward_coord,
        backward_coord,
        rs_output_tensor,  // RS output
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        empty_rs_fused_op_signaler,  // RS -> next op signaling (not used)
        srs_fused_op_signaler,       // MM -> RS signaling (populated by RS factory)
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        reduce_scatter_core_grid_offset,
        mm_cores_y_val,
        mm_block_ht_val,
        mm_block_wt_val,
        mm_N_block_wt_val,
        chunk_width_in_mm_blocks);

    // =========================================================================
    // STEP 2: Create the Matmul program SECOND
    //
    // The matmul factory receives the populated srs_fused_op_signaler, which
    // contains the RS reader cores' NOC coordinates and semaphore ID. The
    // matmul kernels use this to signal the RS when output blocks are ready.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_mm_fused_op_signaler;

    auto mm_shared_variables = ttnn::experimental::prim::minimal_matmul_factory_helper(
        program,
        input_tensor,   // MM input (activations)
        weight_tensor,  // MM weights
        bias,
        fused_activation,
        config,
        matmul_output_tensor,  // MM output (= RS input)
        compute_kernel_config,
        empty_mm_fused_op_signaler,  // No AG -> MM signaling
        srs_fused_op_signaler);      // MM -> RS signaling (populated from step 1)

    return {std::move(program), {rs_shared_variables, mm_shared_variables}};
}

ttnn::device_operation::CachedProgram<MinimalMatmulStridedReduceScatterAsyncProgramFactory::shared_variables_t>
MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_at(
    const MinimalMatmulStridedReduceScatterAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);

    // output_tensor[0] = MM output (= RS input)
    // output_tensor[1] = RS intermediate
    // output_tensor[2] = RS output
    return minimal_matmul_strided_reduce_scatter_async_program(
        tensor_args.input_tensor,   // MM input (activations)
        tensor_args.weight_tensor,  // MM weights
        output_tensor[0],           // MM output = RS input
        output_tensor[1],           // RS intermediate
        output_tensor[2],           // RS output

        /* Reduce Scatter Params */
        mesh_coordinate,
        forward_coord,
        backward_coord,
        attributes.dim,
        attributes.num_links,
        attributes.ring_size,
        device_index,
        attributes.topology,
        attributes.semaphore,
        attributes.barrier_semaphore,
        attributes.using_persistent_buffers,
        attributes.sub_device_id,
        attributes.chunks_per_sync,
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.reduce_scatter_core_grid_offset,
        attributes.chunk_width_in_mm_blocks,

        /* Matmul Params */
        tensor_args.bias,
        attributes.matmul_struct.fused_activation,
        attributes.matmul_struct.config.value(),
        attributes.matmul_struct.compute_kernel_config);
}

}  // namespace ttnn::experimental::prim
