// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_program.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/math.hpp"

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Build a ProgramDescriptor for one mesh coord by composing the matmul helper
// and the strided all-gather builder.
//
// Step 1: build the matmul section first.  The matmul helper populates the
// MinimalMatmulFusedOpSignaler with its receiver cores' NOC coordinates +
// semaphore IDs.  Step 2 then feeds that populated signaler (forwarded as a
// StridedAllGatherFusedOpSignaler via init_fused_op) to the all-gather builder
// so its sender writer kernels know whom to signal once each tensor slice is
// locally available.  Order matters: the all-gather depends on the matmul
// receiver core layout.
tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    // output_tensor[0] = AG output (= MM input)
    // output_tensor[1] = MM output
    auto& all_gather_output_tensor = output_tensor[0];
    auto& matmul_output_tensor = output_tensor[1];

    const uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, attributes.strided_all_gather_async_struct.cluster_axis);

    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor,
        mesh_coordinate,
        1,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor,
        mesh_coordinate,
        -1,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.cluster_axis);

    const auto& config = attributes.matmul_struct.config.value();

    // =========================================================================
    // STEP 1: Build the Matmul section FIRST
    //
    // The matmul helper allocates its fused-op receiver semaphores and records
    // the receiver cores' NOC coordinates into matmul_fused_op_signaler. That
    // info is then forwarded to the all-gather builder in step 2.
    // =========================================================================
    constexpr uint32_t TILE_WIDTH_LOCAL = 32;
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler();
    matmul_fused_op_signaler->init_all_gather(
        attributes.strided_all_gather_async_struct.ring_size,
        device_index,
        input_tensor.padded_shape()[3] / TILE_WIDTH_LOCAL,
        attributes.strided_all_gather_async_struct.topology,
        attributes.read_local_slice_from_input,
        attributes.read_local_slice_from_input ? std::optional<const Tensor>(input_tensor) : std::nullopt);

    // Matmul - wrap single output tensor in vector for unified interface
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> empty_srs_fused_op_signaler;
    std::vector<Tensor> matmul_output_tensors = {matmul_output_tensor};
    (void)ttnn::experimental::prim::minimal_matmul_factory_helper_common_descriptor(
        desc,
        all_gather_output_tensor,
        weight_tensor,
        tensor_args.bias,
        attributes.matmul_struct.fused_activation,
        config,
        matmul_output_tensors,
        attributes.matmul_struct.compute_kernel_config,
        matmul_fused_op_signaler,
        1,  // N_chunks = 1 for single output
        std::nullopt,
        std::nullopt,
        std::nullopt,
        empty_srs_fused_op_signaler);

    // =========================================================================
    // STEP 2: Build the All Gather section SECOND
    //
    // The all-gather helper receives the populated all_gather_fused_op_signaler
    // (initialized from matmul_fused_op_signaler), which contains the matmul
    // receiver cores' NOC coordinates and semaphore IDs. The all-gather sender
    // workers use this to signal the matmul when output blocks are ready.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    (void)StridedAllGatherAsyncProgramFactory::strided_all_gather_async_minimal_default_helper_descriptor(
        desc,
        input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        all_gather_output_tensor,
        attributes.strided_all_gather_async_struct.dim,
        attributes.strided_all_gather_async_struct.num_links,
        attributes.strided_all_gather_async_struct.ring_size,
        device_index,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.semaphore,
        all_gather_fused_op_signaler,
        attributes.read_local_slice_from_input,
        attributes.strided_all_gather_async_struct.num_workers_per_link,
        attributes.strided_all_gather_async_struct.num_buffers_per_channel,
        matmul_fused_op_signaler->num_fused_op_cores_to_signal,
        config.M_block_size,
        config.K_block_size,
        attributes.all_gather_core_grid_offset);

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor StridedAllGatherMinimalMatmulAsyncProgramFactory::create_workload_descriptor(
    const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
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
