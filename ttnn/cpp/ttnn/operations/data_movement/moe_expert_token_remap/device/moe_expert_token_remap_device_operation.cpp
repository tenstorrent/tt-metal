// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
// #include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include "moe_expert_token_remap_device_operation.hpp"

namespace ttnn::operations::data_movement {

MoeExpertTokenRemapDeviceOperation::program_factory_t MoeExpertTokenRemapDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Multicore{};
}

void MoeExpertTokenRemapDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& topk_tensor = tensor_args.topk_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;

    TT_FATAL(topk_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "topk tensor must be in row major layout");
    TT_FATAL(
        metadata_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Metadata tensor must be in row major layout");

    TT_FATAL(metadata_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Metadata tensor must be uint16");

    TT_FATAL(mapping_tensor.dtype() == tt::tt_metal::DataType::UINT16, "Mapping tensor must be uint16");
    TT_FATAL(mapping_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Mapping tensor must be in row major layout");

    auto mesh_device = tensor_args.topk_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();

    const auto num_devices = mesh_view.num_devices();

    const auto& topk_shape = topk_tensor.logical_shape();
    const auto& metadata_shape = metadata_tensor.logical_shape();
    const auto& mapping_shape = mapping_tensor.logical_shape();

    TT_FATAL(
        mapping_shape[-1] == num_devices,
        "Last mapping_tensor dim should be num_devices {}, got {}",
        num_devices,
        mapping_shape[-1]);

    TT_FATAL(
        mapping_shape[-2] == topk_shape[-1],
        "Expected mapping shape dim 2 to be equal to topk dim 3 (number of experts), got {} and {} respectively",
        mapping_shape[-2],
        topk_shape[-1]);

    TT_FATAL(
        metadata_shape[1] == topk_shape[1],
        "Expected metadata dim 1 to be equal to topk dim 1 (batch size), got {} and {}, respectively",
        metadata_shape[1],
        topk_shape[1]);
}

MoeExpertTokenRemapDeviceOperation::spec_return_value_t MoeExpertTokenRemapDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    const auto& mapping_shape = tensor_args.mapping_tensor.logical_shape();
    const auto& metadata_shape = tensor_args.metadata_tensor.logical_shape();

    auto mesh_device = tensor_args.topk_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();

    const auto num_devices = mesh_view.num_devices();

    const auto batch_size = metadata_shape[1];
    const auto seq_size = metadata_shape[2];
    const auto experts = mapping_shape[2];

    const uint32_t num_local_experts = experts / num_devices;

    const auto output_shape = ttnn::Shape({1, batch_size, seq_size, num_local_experts});

    const auto mem_config = operation_attributes.output_mem_config.value_or(MemoryConfig());
    return TensorSpec(
        Shape(output_shape),
        TensorLayout(tensor_args.topk_tensor.dtype(), PageConfig(tensor_args.topk_tensor.layout()), mem_config));
}

MoeExpertTokenRemapDeviceOperation::tensor_return_value_t MoeExpertTokenRemapDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_spec, tensor_args.topk_tensor.device()));
}

std::
    tuple<MoeExpertTokenRemapDeviceOperation::operation_attributes_t, MoeExpertTokenRemapDeviceOperation::tensor_args_t>
    MoeExpertTokenRemapDeviceOperation::invoke(
        const ttnn::Tensor& topk_tensor,
        const ttnn::Tensor& mapping_tensor,
        const ttnn::Tensor& metadata_tensor,
        const std::optional<ttnn::MemoryConfig>& output_mem_config,
        const std::optional<ttnn::Tensor>& optional_output_tensor) {
    return {
        operation_attributes_t{.output_mem_config = output_mem_config},
        tensor_args_t{
            .topk_tensor = topk_tensor,
            .mapping_tensor = mapping_tensor,
            .metadata_tensor = metadata_tensor,
            .optional_output_tensor = optional_output_tensor}};
}

}  // namespace ttnn::operations::data_movement
