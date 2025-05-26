// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/reduce_scatter_matmul/device/rs_matmul_op.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl {

void AllGatherRS::validate_on_program_cache_hit(
    const LlamaReduceScatterDeviceOperation& rs_struct,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const operations::matmul::Matmul& matmul_struct,
    const matmul_tensor_args_t& matmul_tensor_args) const {
    matmul_struct.validate({matmul_tensor_args.input_tensor, matmul_tensor_args.weight_tensor}, {std::nullopt}, {});
    rs_struct.validate_on_program_cache_hit(operation_attributes, tensor_args);
}

void AllGatherRS::validate_on_program_cache_miss(
    const LlamaReduceScatterDeviceOperation& rs_struct,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const operations::matmul::Matmul& matmul_struct,
    const matmul_tensor_args_t& matmul_tensor_args) const {
    matmul_struct.validate({matmul_tensor_args.input_tensor, matmul_tensor_args.weight_tensor}, {std::nullopt}, {});
    rs_struct.validate_on_program_cache_miss(operation_attributes, tensor_args);
}

std::vector<ttnn::TensorSpec> AllGatherRS::compute_output_specs(
    const LlamaReduceScatterDeviceOperation& rs_struct,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const operations::matmul::Matmul& matmul_struct,
    const matmul_tensor_args_t& matmul_tensor_args) const {
    // All Gather shape
    ttnn::TensorSpec reduce_scatter_output_spec = rs_struct.compute_output_specs(operation_attributes, tensor_args);
    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        matmul_struct.compute_output_specs({matmul_tensor_args.input_tensor, matmul_tensor_args.weight_tensor}, {})[0];

    return {matmul_output_specs, reduce_scatter_output_spec};
}

std::vector<Tensor> AllGatherRS::create_output_tensors(
    const LlamaReduceScatterDeviceOperation& rs_struct,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const operations::matmul::Matmul& matmul_struct,
    const matmul_tensor_args_t& matmul_tensor_args) const {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        matmul_struct.create_output_tensors({matmul_tensor_args.input_tensor, matmul_tensor_args.weight_tensor}, {})[0];
    ttnn::Tensor rs_output_tensor = rs_struct.create_output_tensors(operation_attributes, tensor_args);

    return {matmul_output_tensor, rs_output_tensor};
}

std::tuple<
    LlamaReduceScatterDeviceOperation,
    LlamaReduceScatterDeviceOperation::operation_attributes_t,
    LlamaReduceScatterDeviceOperation::tensor_args_t,
    operations::matmul::Matmul,
    AllGatherRS::matmul_tensor_args_t>
AllGatherRS::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,  // mm1 used
    const ttnn::Tensor& rs_tensor,      // rs1
    ttnn::Tensor& intermediate_packet_buffer,
    const int32_t dim,
    const GlobalSemaphore& semaphore,
    const tt::tt_metal::SubDeviceId subdevice_id,
    const uint32_t cluster_axis,
    const uint32_t ring_devices,
    const uint32_t num_links,                                                            // default 1
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,                           // default std::nullopt
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,                           // default std::nullopt
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,    // default std::nullopt
    const std::optional<const GlobalCircularBuffer>& global_cb,                          // default std::nullopt
    const std::optional<const ttnn::CoreGrid> core_grid,                                 // default std::nullopt
    const bool transpose_a,                                                              // degault false
    const bool transpose_b,                                                              // default false
    const std::optional<const DataType> dtype,                                           // default std::nullopt
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // default std::nullopt
    const std::optional<const std::string>& activation,                                  // default std::nullopt
    const std::optional<const tt::tt_metal::Tile>& output_tile,                          // default std::nullopt
    const std::optional<Tensor>& optional_output_tensor                                  // default std::nullopt
) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.get_logical_shape());
    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.get_dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            output_tile,
            global_cb});
    LlamaReduceScatterDeviceOperation rs_struct;

    auto [operation_attributes, tensor_args] = rs_struct.invoke(
        rs_tensor,
        intermediate_packet_buffer,
        dim,
        semaphore,
        subdevice_id,
        cluster_axis,
        ring_devices,
        num_links,
        memory_config_rs);
    return {
        rs_struct,
        operation_attributes,
        tensor_args,
        matmul_struct,
        matmul_tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor}};
}

}  // namespace ttnn::operations::experimental::ccl
