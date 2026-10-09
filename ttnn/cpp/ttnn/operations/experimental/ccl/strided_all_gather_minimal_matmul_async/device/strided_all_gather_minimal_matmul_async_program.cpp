// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>
#include <cstdlib>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <sstream>
#include <type_traits>

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_fabric_bound_program_factory.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// The matmul kernels are pushed first, so the all-gather kernels start right after them.
constexpr uint32_t strided_ag_mm_matmul_first_kernel_index = 0;
constexpr uint32_t strided_ag_mm_all_gather_first_kernel_index =
    strided_ag_mm_matmul_first_kernel_index + minimal_matmul_fabric_bound_layout::kNumKernels;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::ProgramDescriptor StridedAllGatherMinimalMatmulAsyncProgramFactory::create_descriptor(
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "strided_all_gather_minimal_matmul_async builds one program per mesh coordinate; no coordinate was given");
    const auto& mesh_coordinate = mesh_dispatch_coordinate.value();
    const auto& ag_attributes = attributes.strided_all_gather_async_struct;
    const auto& input_tensor = tensor_args.input_tensor;
    Tensor& all_gather_output_tensor = output_tensor[0];
    const bool read_local_slice_from_input = attributes.read_local_slice_from_input;
    const auto& config = attributes.matmul_struct.config.value();

    uint32_t ring_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, ag_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, ag_attributes.topology, ag_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, ag_attributes.topology, ag_attributes.cluster_axis);

    // Matmul outputs are all tensors after the all-gather output (one per chunk).
    std::vector<Tensor> matmul_output_tensors(output_tensor.begin() + 1, output_tensor.end());

    tt::tt_metal::ProgramDescriptor desc;

    // Create a matmul signal info object that gets populated by the matmul kernel
    uint32_t TILE_WIDTH = 32;
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler();
    matmul_fused_op_signaler->init_all_gather(
        ag_attributes.ring_size,
        ring_index,
        input_tensor.padded_shape()[3] / TILE_WIDTH,
        ag_attributes.topology,
        read_local_slice_from_input,
        read_local_slice_from_input ? std::optional<const Tensor>(input_tensor) : std::nullopt);

    // Option W (writer-signals-matmul): the matmul cores keep the legacy 3 semaphores

    // Matmul outputs: one tensor per chunk (chunks == 1 -> single output).
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> empty_srs_fused_op_signaler;
    TT_FATAL(
        desc.kernels.size() == strided_ag_mm_matmul_first_kernel_index,
        "strided_all_gather_minimal_matmul_async matmul kernels must start at index {}",
        strided_ag_mm_matmul_first_kernel_index);
    minimal_matmul_fabric_bound_factory_helper_common(
        desc,
        all_gather_output_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias,
        attributes.matmul_struct.fused_activation,
        config,
        matmul_output_tensors,
        attributes.matmul_struct.compute_kernel_config,
        matmul_fused_op_signaler,
        static_cast<uint32_t>(matmul_output_tensors.size()),  // N_chunks
        attributes.matmul_struct.fused_ternary_scalar,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        empty_srs_fused_op_signaler,
        attributes.matmul_struct.fuse_swiglu);

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    TT_FATAL(
        desc.kernels.size() == strided_ag_mm_all_gather_first_kernel_index,
        "strided_all_gather_minimal_matmul_async all-gather kernels must start at index {}",
        strided_ag_mm_all_gather_first_kernel_index);
    strided_all_gather_async_minimal_default_helper(
        desc,
        input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        all_gather_output_tensor,
        ag_attributes.dim,
        ag_attributes.num_links,
        ag_attributes.ring_size,
        ring_index,
        ag_attributes.topology,
        ag_attributes.semaphore,
        all_gather_fused_op_signaler,
        read_local_slice_from_input,
        ag_attributes.num_workers_per_link,
        ag_attributes.num_buffers_per_channel,
        matmul_fused_op_signaler->num_fused_op_cores_to_signal,
        config.M_block_size,
        config.K_block_size,
        attributes.all_gather_core_grid_offset,
        attributes.mm_signal_aggregator_mode);

    return desc;
}

void StridedAllGatherMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const Tensor& all_gather_output_tensor = output_tensor.at(0);

    strided_all_gather_async_patch_runtime_args(
        program,
        strided_ag_mm_all_gather_first_kernel_index,
        attributes.strided_all_gather_async_struct,
        tensor_args.input_tensor,
        all_gather_output_tensor,
        /*fused=*/true);

    // The all-gather output is the matmul's in0; the matmul outputs are all tensors after it (one per chunk).
    std::vector<Tensor> matmul_output_tensors(output_tensor.begin() + 1, output_tensor.end());
    minimal_matmul_fabric_bound_patch_runtime_args(
        program,
        strided_ag_mm_matmul_first_kernel_index,
        all_gather_output_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias,
        attributes.read_local_slice_from_input ? std::optional<const Tensor>(tensor_args.input_tensor) : std::nullopt,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        matmul_output_tensors);
}

}  // namespace ttnn::experimental::prim
