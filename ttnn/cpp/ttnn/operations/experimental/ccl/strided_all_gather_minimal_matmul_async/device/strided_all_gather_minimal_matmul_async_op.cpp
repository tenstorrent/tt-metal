// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"

using matmul_device_operation_t = ttnn::operations::experimental::minimal_matmul::MinimalMatmulDeviceOperation;

namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async {

StridedAllGatherMinimalMatmulAsync::program_factory_t StridedAllGatherMinimalMatmulAsync::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::StridedAllGatherMinimalMatmulAsyncProgramFactory{};
}

void StridedAllGatherMinimalMatmulAsync::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

void StridedAllGatherMinimalMatmulAsync::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.strided_all_gather_async_struct.dim == 3,
        "StridedAllGatherMinimalMatmulAsync requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        tensor_args.input_tensor.padded_shape()[0] == 1 && tensor_args.input_tensor.padded_shape()[1] == 1,
        "StridedAllGatherMinimalMatmulAsync requires input tensor to have batch size of 1.");
}

StridedAllGatherMinimalMatmulAsync::spec_return_value_t StridedAllGatherMinimalMatmulAsync::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // All Gather shape
    ttnn::TensorSpec strided_all_gather_output_shape =
        strided_all_gather_async::StridedAllGatherAsync::compute_output_specs(
            attributes.strided_all_gather_async_struct,
            strided_all_gather_async::tensor_args_t{tensor_args.input_tensor});

    // Matmul shape
    ttnn::TensorSpec minimal_matmul_output_specs = matmul_device_operation_t::compute_output_specs(
        attributes.matmul_struct, {tensor_args.input_tensor, tensor_args.weight_tensor});

    return {strided_all_gather_output_shape, minimal_matmul_output_specs};
}

StridedAllGatherMinimalMatmulAsync::tensor_return_value_t StridedAllGatherMinimalMatmulAsync::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // All Gather output tensor
    ttnn::Tensor strided_all_gather_output_tensor =
        strided_all_gather_async::StridedAllGatherAsync::create_output_tensors(
            attributes.strided_all_gather_async_struct,
            strided_all_gather_async::tensor_args_t{tensor_args.input_tensor, tensor_args.persistent_output_buffer});

    // Matmul output tensor
    ttnn::Tensor minimal_matmul_output_tensor = matmul_device_operation_t::create_output_tensors(
        attributes.matmul_struct, {strided_all_gather_output_tensor, tensor_args.weight_tensor});

    return {strided_all_gather_output_tensor, minimal_matmul_output_tensor};
}

tt::tt_metal::operation::Hash StridedAllGatherMinimalMatmulAsync::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "StridedAllGatherMinimalMatmulAsync::compute_program_hash is called");

    auto program_factory = select_program_factory(attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<StridedAllGatherMinimalMatmulAsync>(
        attributes.strided_all_gather_async_struct.dim,
        attributes.strided_all_gather_async_struct.num_links,
        attributes.strided_all_gather_async_struct.ring_size,
        attributes.strided_all_gather_async_struct.output_mem_config,
        attributes.strided_all_gather_async_struct.topology,
        attributes.strided_all_gather_async_struct.cluster_axis,
        attributes.strided_all_gather_async_struct.tiles_per_chunk,
        attributes.strided_all_gather_async_struct.num_workers_per_link,
        attributes.strided_all_gather_async_struct.num_buffers_per_channel,
        attributes.strided_all_gather_async_struct.mm_cores_y,
        attributes.strided_all_gather_async_struct.mm_block_ht,
        attributes.strided_all_gather_async_struct.mm_block_wt,
        attributes.matmul_struct,
        attributes.all_gather_core_grid_offset,
        attributes.read_local_slice_from_input,
        attributes.ag_op,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::StridedAllGatherMinimalMatmulAsync::
    tensor_return_value_t
    strided_all_gather_minimal_matmul_async(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        const uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const CoreCoord strided_all_gather_core_grid_offset,
        const uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config_ag,
        const ttnn::ccl::Topology topology,
        std::optional<uint32_t> cluster_axis,
        const std::optional<const Tensor>& bias,
        const std::optional<MemoryConfig>& memory_config_mm,
        std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
        std::optional<const ttnn::operations::experimental::minimal_matmul::MinimalMatmulConfig> config,
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<bool> read_local_slice_from_input) {
    using OperationType = ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::
        StridedAllGatherMinimalMatmulAsync;

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias);
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    /* AllGather setup */
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    ttnn::operations::experimental::ccl::strided_all_gather_async::operation_attributes_t
        strided_all_gather_async_struct =
            ttnn::operations::experimental::ccl::strided_all_gather_async::operation_attributes_t(
                devices,
                dim,
                num_links,
                num_devices,
                memory_config_ag.value_or(input_tensor.memory_config()),
                topology,
                multi_device_global_semaphore,
                cluster_axis,
                /*tiles_per_chunk=*/std::nullopt,
                num_workers_per_link,
                num_buffers_per_channel,
                config->compute_with_storage_grid_size.y,
                config->M_block_size,
                config->K_block_size);

    /* Matmul setup */
    auto matmul_struct = decltype(ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async::
                                      operation_attributes_t::matmul_struct){
        .config = config,
        .fused_activation = std::move(fused_activation),
        .output_mem_config = memory_config_mm,
        .compute_kernel_config = compute_kernel_config.value()};
    ttnn::operations::experimental::ccl::strided_all_gather_async::StridedAllGatherAsync ag_op{};

    bool read_local_from_input = read_local_slice_from_input.value_or(false);

    auto operation_attributes = OperationType::operation_attributes_t{
        strided_all_gather_async_struct,
        matmul_struct,
        strided_all_gather_core_grid_offset,
        read_local_from_input,
        devices,
        ag_op};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, weight_tensor, persistent_output_buffer, bias};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
