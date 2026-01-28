// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <sstream>
#include <type_traits>

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

StridedAllGatherMinimalMatmulAsyncProgramFactory::cached_mesh_workload_t
StridedAllGatherMinimalMatmulAsyncProgramFactory::create_mesh_workload(
    const StridedAllGatherMinimalMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
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

void StridedAllGatherMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(range);
        StridedAllGatherAsyncProgramFactory::override_runtime_arguments_per_program(
            shared_variables.ag_shared_variables,
            program,
            attributes.strided_all_gather_async_struct,
            StridedAllGatherAsyncInputs(tensor_args.input_tensor),
            output_tensor.at(0));

        auto cached_program_proxy = ttnn::experimental::prim::MinimalMatmulProgramFactory::cached_program_t::proxy(
            program, shared_variables.mm_shared_variables);

        ttnn::experimental::prim::MinimalMatmulProgramFactory::override_runtime_arguments(
            cached_program_proxy,
            attributes.matmul_struct,
            {output_tensor.at(0), tensor_args.weight_tensor, tensor_args.bias, tensor_args.input_tensor},
            {output_tensor.at(1)});
    }
}

ttnn::device_operation::CachedProgram<StridedAllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t>
strided_all_gather_minimal_matmul_async_program(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    bool read_local_slice_from_input,

    /* All Gather Params */
    IDevice* /*target_device*/,
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    ttnn::experimental::prim::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt::tt_metal::Program program{};

    // Create a matmul signal info object that gets populated by the matmul kernel
    uint32_t TILE_WIDTH = 32;
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler();
    matmul_fused_op_signaler->init_all_gather(
        ring_size,
        ring_index,
        input_tensor.padded_shape()[3] / TILE_WIDTH,
        topology,
        read_local_slice_from_input,
        read_local_slice_from_input ? std::optional<const Tensor>(input_tensor) : std::nullopt);

    // Matmul
    auto mm_shared_variables = ttnn::experimental::prim::minimal_matmul_factory_helper(
        program,
        all_gather_output_tensor,
        weight_tensor,
        bias,
        fused_activation,
        config,
        matmul_output_tensor,
        compute_kernel_config,
        matmul_fused_op_signaler);

    // Create the all gather fused op signaler
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    // All Gather
    StridedAllGatherAsyncProgramFactory::shared_variables_t ag_shared_variables =
        StridedAllGatherAsyncProgramFactory::strided_all_gather_async_minimal_default_helper(
            program,
            input_tensor,
            target_device_coord,
            forward_coord,
            backward_coord,
            all_gather_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            all_gather_fused_op_signaler,
            read_local_slice_from_input,
            std::nullopt,
            num_workers_per_direction_opt,
            num_buffers_per_channel,
            matmul_fused_op_signaler->num_fused_op_cores_to_signal,
            config.M_block_size,
            config.K_block_size,
            core_grid_offset);

    return {std::move(program), {ag_shared_variables, mm_shared_variables}};
}

ttnn::device_operation::CachedProgram<StridedAllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t>
StridedAllGatherMinimalMatmulAsyncProgramFactory::create_at(
    const StridedAllGatherMinimalMatmulAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const StridedAllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    auto* mesh_device = tensor_args.input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.input_tensor.device();

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, attributes.strided_all_gather_async_struct.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        1,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        -1,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.cluster_axis);

    // Return the StridedAllGatherMinimalMatmulAsync program with callbacks
    return strided_all_gather_minimal_matmul_async_program(
        tensor_args.input_tensor,   // input_tensor
        output_tensor[0],           // all_gather_output_tensor
        tensor_args.weight_tensor,  // weight_tensor
        output_tensor[1],           // matmul_output_tensor
        attributes.read_local_slice_from_input,

        /* All Gather Params */
        target_device,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        attributes.strided_all_gather_async_struct.dim,
        attributes.strided_all_gather_async_struct.num_links,
        attributes.strided_all_gather_async_struct.ring_size,
        device_index,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.semaphore,
        attributes.strided_all_gather_async_struct.num_workers_per_link,
        attributes.strided_all_gather_async_struct.num_buffers_per_channel,
        attributes.all_gather_core_grid_offset,

        /* Matmul Params */
        tensor_args.bias,  // Bias
        attributes.matmul_struct.fused_activation,
        attributes.matmul_struct.config.value(),
        attributes.matmul_struct.compute_kernel_config);
}

}  // namespace ttnn::experimental::prim
