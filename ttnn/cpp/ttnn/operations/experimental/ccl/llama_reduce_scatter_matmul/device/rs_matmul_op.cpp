// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl {

Matmul_RS::program_factory_t Matmul_RS::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Matmul_RS_PF{};
}

void Matmul_RS::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.second_weight_tensor.has_value()) {
        operation_attributes.matmul.validate(
            {tensor_args.matmul.input_tensor,
             tensor_args.matmul.weight_tensor,
             tensor_args.second_weight_tensor.value()},
            {std::nullopt},
            {});
    } else {
        operation_attributes.matmul.validate(
            {tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {std::nullopt}, {});
    }
    operation_attributes.rs.validate_on_program_cache_hit(operation_attributes.rs_op, tensor_args.rs);
}

void Matmul_RS::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.second_weight_tensor.has_value()) {
        operation_attributes.matmul.validate(
            {tensor_args.matmul.input_tensor,
             tensor_args.matmul.weight_tensor,
             tensor_args.second_weight_tensor.value()},
            {std::nullopt},
            {});
    } else {
        operation_attributes.matmul.validate(
            {tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {std::nullopt}, {});
    }
    operation_attributes.rs.validate_on_program_cache_miss(operation_attributes.rs_op, tensor_args.rs);
}

Matmul_RS::spec_return_value_t Matmul_RS::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Reduce Scatter shape
    ttnn::TensorSpec reduce_scatter_output_spec =
        operation_attributes.rs.compute_output_specs(operation_attributes.rs_op, tensor_args.rs);
    // Matmul shape
    if (tensor_args.second_weight_tensor.has_value()) {
        auto matmul_output_specs = operation_attributes.matmul.compute_output_specs(
            {tensor_args.matmul.input_tensor,
             tensor_args.matmul.weight_tensor,
             tensor_args.second_weight_tensor.value()},
            {});
        return {matmul_output_specs.at(0), matmul_output_specs.at(1), reduce_scatter_output_spec};
    } else {
        ttnn::TensorSpec matmul_output_specs = operation_attributes.matmul.compute_output_specs(
            {tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {})[0];
        return {matmul_output_specs, reduce_scatter_output_spec};
    }
}

Matmul_RS::tensor_return_value_t Matmul_RS::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    Tensor rs_output_tensor = operation_attributes.rs.create_output_tensors(operation_attributes.rs_op, tensor_args.rs);
    if (tensor_args.second_weight_tensor.has_value()) {
        return {tensor_args.matmul_output_tensors.at(0), tensor_args.matmul_output_tensors.at(1), rs_output_tensor};
    } else {
        Tensor matmul_output_tensor = operation_attributes.matmul.create_output_tensors(
            {tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {})[0];
        return {matmul_output_tensor, rs_output_tensor};
    }
}

std::tuple<Matmul_RS::operation_attributes_t, Matmul_RS::tensor_args_t> Matmul_RS::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,                   // mm1 used
    const std::optional<const ttnn::Tensor>& rs_tensor,  // rs1
    ttnn::Tensor& intermediate_packet_buffer,
    const int32_t dim,
    const GlobalSemaphore& semaphore,
    const uint32_t cluster_axis,
    const uint32_t ring_devices,
    const uint32_t num_links,
    const tt::tt_metal::SubDeviceId& subdevice_id,
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
    const std::optional<Tensor>& optional_output_tensor,                                 // default std::nullopt
    tt::tt_fabric::Topology topology,
    bool use_noc1_only,
    const std::optional<const ttnn::Tensor>& second_weight_tensor) {
    TT_FATAL(
        rs_tensor.has_value() ^ second_weight_tensor.has_value(),
        "Exactly one of rs_tensor or second_weight_tensor must have a value");
    LlamaReduceScatterDeviceOperation rs_struct{};
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    auto matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::Matmul{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            output_tile,
            global_cb});
    if (second_weight_tensor.has_value()) {
        std::vector<Tensor> matmul_output_tensors =
            matmul_struct.create_output_tensors({input_tensor, weight_tensor, second_weight_tensor.value()}, {});
        auto new_rs_tensor = matmul_output_tensors.at(0);
        return {
            operation_attributes_t{
                rs_struct,
                LlamaReduceScatterDeviceOperation::operation_attributes_t{
                    .dim = (dim < 0 ? uint32_t(new_rs_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
                    .cross_device_semaphore = semaphore,
                    .subdevice_id = subdevice_id,
                    .cluster_axis = cluster_axis,
                    .output_mem_config = memory_config_rs,
                    .ring_devices = ring_devices,
                    .num_links = num_links,
                    .topology = topology,
                    .use_noc1_only = use_noc1_only},
                matmul_struct},
            tensor_args_t{
                LlamaReduceScatterDeviceOperation::tensor_args_t{new_rs_tensor, intermediate_packet_buffer},
                matmul_tensor_args_t{input_tensor, weight_tensor},
                matmul_output_tensors,
                second_weight_tensor}};
    } else {
        auto new_rs_tensor = rs_tensor.value();
        return {
            operation_attributes_t{
                rs_struct,
                LlamaReduceScatterDeviceOperation::operation_attributes_t{
                    .dim = (dim < 0 ? uint32_t(new_rs_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
                    .cross_device_semaphore = semaphore,
                    .subdevice_id = subdevice_id,
                    .cluster_axis = cluster_axis,
                    .output_mem_config = memory_config_rs,
                    .ring_devices = ring_devices,
                    .num_links = num_links,
                    .topology = topology,
                    .use_noc1_only = use_noc1_only},
                matmul_struct},
            tensor_args_t{
                LlamaReduceScatterDeviceOperation::tensor_args_t{new_rs_tensor, intermediate_packet_buffer},
                matmul_tensor_args_t{input_tensor, weight_tensor},
                {},
                std::nullopt}};
    }
}

}  // namespace ttnn::operations::experimental::ccl
