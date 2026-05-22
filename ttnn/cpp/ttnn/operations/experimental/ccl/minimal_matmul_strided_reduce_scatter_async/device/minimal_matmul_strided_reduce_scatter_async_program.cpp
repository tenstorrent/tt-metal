// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
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

using namespace tt::constants;

// Import the RS program artifacts type
using ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts;

// Forward declaration for the ProgramDescriptor (Contract-2) variant of the
// strided reduce-scatter ring builder (defined in
// strided_reduce_scatter_async_program.cpp in namespace ttnn).
namespace ttnn {
StridedReduceScatterProgramArtifacts build_ring_strided_reduce_scatter_async_program_artifacts_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
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
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor1 = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor2 = std::nullopt);
}  // namespace ttnn

namespace ttnn::experimental::prim {

namespace {

// Build a ProgramDescriptor for one mesh coord by composing the RS ring builder
// and the matmul helper.
//
// Step 1: build the RS section first.  The RS builder allocates its reader-side
// semaphore + records the RS reader cores' NOC coordinates into the
// srs_fused_op_signaler.  Step 2 then feeds that populated signaler to the
// matmul helper so the matmul writer kernels know where to signal once each
// output block is ready.  Order matters: the matmul depends on the RS reader
// core layout.
tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const MinimalMatmulStridedReduceScatterAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    // output_tensor[0] = MM output (= RS input)
    // output_tensor[1] = RS intermediate (scratch)
    // output_tensor[2] = RS output
    auto& matmul_output_tensor = output_tensor[0];
    auto& rs_intermediate_tensor = output_tensor[1];
    auto& rs_output_tensor = output_tensor[2];

    const uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, attributes.cluster_axis);

    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);

    const auto& config = attributes.matmul_struct.config.value();

    // Derive matmul geometry parameters for the RS factory.
    // The matmul factory normally transposes its core grid when M > N, but
    // transposition is disabled when fusing with SRS (the RS iteration structure
    // requires mm_N_block_wt <= slice_Wt, which can be violated when transposing).
    // So we always use the non-transposed layout: M parallelized on y, N on x.
    const uint32_t mm_cores_y_val = config.compute_with_storage_grid_size.y;
    const uint32_t mm_block_ht_val = config.M_block_size;
    const uint32_t mm_block_wt_val = config.N_block_size;

    // Compute mm_N_full_block_wt: total N tiles per core (N parallelized on x)
    const uint32_t N_tiles = weight_tensor.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t num_cores_x = config.compute_with_storage_grid_size.x;
    const uint32_t padded_N_tiles = tt::round_up(N_tiles, num_cores_x);
    const uint32_t mm_N_full_block_wt_val = padded_N_tiles / num_cores_x;

    // =========================================================================
    // STEP 1: Build the Reduce Scatter section FIRST
    //
    // The RS factory creates a semaphore on the RS reader cores and records
    // their NOC coordinates. This info is captured in srs_fused_op_signaler,
    // which is then passed to the matmul factory in step 2.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler =
        ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler();
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> empty_rs_fused_op_signaler = std::nullopt;

    (void)::ttnn::build_ring_strided_reduce_scatter_async_program_artifacts_descriptor(
        desc,
        matmul_output_tensor,    // RS input = MM output
        rs_intermediate_tensor,  // RS intermediate (scratch)
        mesh_coordinate,
        forward_coord,
        backward_coord,
        rs_output_tensor,  // RS output
        attributes.dim,
        attributes.num_links,
        attributes.ring_size,
        device_index,
        attributes.topology,
        attributes.semaphore,
        attributes.barrier_semaphore,
        attributes.using_persistent_buffers,
        attributes.sub_device_id,
        empty_rs_fused_op_signaler,  // RS -> next op signaling (not used)
        srs_fused_op_signaler,       // MM -> RS signaling (populated by RS builder)
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.reduce_scatter_core_grid_offset,
        mm_cores_y_val,
        mm_block_ht_val,
        mm_block_wt_val,
        mm_N_full_block_wt_val,
        attributes.chunk_width_in_mm_blocks,
        // Phase 2: fuse addcmul at the RS final write step (not in MM kernel)
        attributes.fused_ternary_scalar,
        tensor_args.addcmul_input_tensor1,
        tensor_args.addcmul_input_tensor2);

    // =========================================================================
    // STEP 2: Build the Matmul section SECOND
    //
    // The matmul factory receives the populated srs_fused_op_signaler, which
    // contains the RS reader cores' NOC coordinates and semaphore ID. The
    // matmul kernels use this to signal the RS when output blocks are ready.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_mm_fused_op_signaler;

    std::vector<Tensor> mm_output_tensors = {matmul_output_tensor};
    (void)ttnn::experimental::prim::minimal_matmul_factory_helper_common_descriptor(
        desc,
        input_tensor,   // MM input (activations)
        weight_tensor,  // MM weights
        tensor_args.bias,
        attributes.matmul_struct.fused_activation,
        config,
        mm_output_tensors,  // MM output (= RS input)
        attributes.matmul_struct.compute_kernel_config,
        empty_mm_fused_op_signaler,  // No AG -> MM signaling
        1,                           // N_chunks = 1
        std::nullopt,                // ternary fused in RS, not MM
        std::nullopt,
        std::nullopt,
        srs_fused_op_signaler);  // MM -> RS signaling (populated from step 1)

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_descriptor_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
