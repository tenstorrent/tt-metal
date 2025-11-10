// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"

namespace ttnn {
namespace ccl {
namespace strided_all_gather_minimal_matmul_async_detail {

StridedAllGatherMinimalMatmulAsync create_strided_all_gather_minimal_matmul_async_struct(
    const ttnn::StridedAllGatherAsync& strided_all_gather_struct_input,
    const operations::experimental::minimal_matmul::MinimalMatmulOp& matmul_struct_input,
    const CoreCoord all_gather_core_grid_offset,
    const std::vector<IDevice*>& devices) {
    return ttnn::StridedAllGatherMinimalMatmulAsync{
        strided_all_gather_struct_input, matmul_struct_input, all_gather_core_grid_offset, devices};
}

}  // namespace strided_all_gather_minimal_matmul_async_detail
}  // namespace ccl

void StridedAllGatherMinimalMatmulAsync::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 2, "StridedAllGatherMinimalMatmulAsync requires 2 input tensors: [input, weight]");
    auto& input_tensor = input_tensors[0];
    // auto& weight_tensor = input_tensors[1];

    TT_FATAL(
        std::all_of(
            input_tensors.begin(), input_tensors.end(), [](const auto& t) { return t.logical_shape().rank() == 4; }),
        "StridedAllGatherMinimalMatmulAsync requires input tensors to be of rank 4");

    // if (output_tensors[0].has_value()) {
    //     auto& strided_all_gather_output_tensor = output_tensors.at(0).value();
    //     // All Gather validate
    //     this->strided_all_gather_async_struct.validate_with_output_tensors(
    //         {input_tensor}, {strided_all_gather_output_tensor});
    //     // Matmul validate
    //     this->matmul_struct.validate({strided_all_gather_output_tensor, weight_tensor}, optional_input_tensors);
    // }

    // All Gather Matmul validate
    TT_FATAL(
        this->strided_all_gather_async_struct.dim == 3,
        "StridedAllGatherMinimalMatmulAsync requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        input_tensor.padded_shape()[0] == 1 && input_tensor.padded_shape()[1] == 1,
        "StridedAllGatherMinimalMatmulAsync requires input tensor to have batch size of 1.");
}

std::vector<ttnn::TensorSpec> StridedAllGatherMinimalMatmulAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // All Gather shape
    ttnn::TensorSpec strided_all_gather_output_shape =
        this->strided_all_gather_async_struct.compute_output_specs({input_tensors[0]})[0];

    // Matmul shape
    ttnn::TensorSpec minimal_matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[0], input_tensors[1]})[0];

    return {strided_all_gather_output_shape, minimal_matmul_output_specs};
}

std::vector<Tensor> StridedAllGatherMinimalMatmulAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    // All Gather output tensor
    ttnn::Tensor strided_all_gather_output_tensor =
        this->strided_all_gather_async_struct.create_output_tensors({input_tensors[0]})[0];

    // Matmul output tensor
    ttnn::Tensor minimal_matmul_output_tensor =
        this->matmul_struct.create_output_tensors({strided_all_gather_output_tensor, input_tensors[1]})[0];

    return {strided_all_gather_output_tensor, minimal_matmul_output_tensor};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks StridedAllGatherMinimalMatmulAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ttnn::ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks StridedAllGatherMinimalMatmulAsync::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at physical coordinate {} is called", mesh_coord);
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensors[0].device();

    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(
        input_tensors[0], mesh_coord, this->strided_all_gather_async_struct.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensors[0],
        mesh_coord,
        1,
        this->strided_all_gather_async_struct.topology,
        this->strided_all_gather_async_struct.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensors[0],
        mesh_coord,
        -1,
        this->strided_all_gather_async_struct.topology,
        this->strided_all_gather_async_struct.cluster_axis);

    // Return the StridedAllGatherMinimalMatmulAsync program with callbacks
    return strided_all_gather_minimal_matmul_async_program(
        input_tensors[0],   // input_tensor
        output_tensors[0],  // all_gather_output_tensor
        input_tensors[1],   // weight_tensor
        output_tensors[1],  // matmul_output_tensor

        /* All Gather Params */
        target_device,
        mesh_coord,
        forward_coord,
        backward_coord,
        this->strided_all_gather_async_struct.dim,
        this->strided_all_gather_async_struct.num_links,
        this->strided_all_gather_async_struct.ring_size,
        device_index,
        this->strided_all_gather_async_struct.topology,
        this->strided_all_gather_async_struct.semaphore,
        this->strided_all_gather_async_struct.barrier_semaphore,
        this->strided_all_gather_async_struct.sub_device_id,
        this->strided_all_gather_async_struct.num_workers_per_link,
        this->strided_all_gather_async_struct.num_buffers_per_channel,
        this->all_gather_core_grid_offset,

        /* Matmul Params */
        optional_input_tensors[0],  // Bias
        this->matmul_struct.fused_activation,
        this->matmul_struct.config.value(),
        this->matmul_struct.compute_kernel_config);
}

tt::tt_metal::operation::Hash StridedAllGatherMinimalMatmulAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<StridedAllGatherMinimalMatmulAsync>(
        this->strided_all_gather_async_struct.dim,
        this->strided_all_gather_async_struct.num_links,
        this->strided_all_gather_async_struct.ring_size,
        this->strided_all_gather_async_struct.output_mem_config,
        this->strided_all_gather_async_struct.topology,
        this->strided_all_gather_async_struct.sub_device_id.has_value(),
        this->strided_all_gather_async_struct.sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  shard_builder::HalProgrammableCoreType::TENSIX,
                  this->strided_all_gather_async_struct.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        this->strided_all_gather_async_struct.cluster_axis,
        this->strided_all_gather_async_struct.barrier_semaphore.has_value(),
        this->strided_all_gather_async_struct.num_workers_per_link,
        this->strided_all_gather_async_struct.num_buffers_per_channel,
        this->all_gather_core_grid_offset,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

std::vector<ttnn::Tensor> strided_all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord strided_all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config_mm,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const minimal_matmul::MinimalMatmulConfig> config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias);
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    /* AllGather setup */
    ttnn::StridedAllGatherAsync strided_all_gather_async_struct = ttnn::StridedAllGatherAsync(
        devices,
        dim,
        num_links,
        devices.size(),
        memory_config_ag.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        sub_device_id,
        /*cluster_axis=*/std::nullopt,
        barrier_semaphore,
        /*tiles_per_chunk=*/std::nullopt,
        num_workers_per_link,
        num_buffers_per_channel,
        config->compute_with_storage_grid_size.y,
        config->M_block_size,
        config->K_block_size);

    // // Create the all gather output tensor used as input (activation) to the matmul
    // ttnn::Tensor strided_all_gather_out_tensor =
    //     strided_all_gather_async_struct.create_output_tensors({input_tensor})[0];

    /* Matmul setup */
    operations::experimental::minimal_matmul::MinimalMatmulOp matmul_struct =
        operations::experimental::minimal_matmul::MinimalMatmulOp{
            .config = config,
            .fused_activation = std::move(fused_activation),
            .output_mem_config = memory_config_mm,
            .compute_kernel_config = compute_kernel_config.value()};

    return tt::tt_metal::operation::run(
        ttnn::ccl::strided_all_gather_minimal_matmul_async_detail::
            create_strided_all_gather_minimal_matmul_async_struct(
                /* All Gather Params */
                strided_all_gather_async_struct,
                /* Matmul params */
                matmul_struct,
                /* Fusion params */
                strided_all_gather_core_grid_offset,
                devices),
        {input_tensor, weight_tensor},
        optional_input_tensors,
        {});
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
