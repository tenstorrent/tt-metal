// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_gdn_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

void NlpCreateHeadsGdnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto input_shape = input_tensor.padded_shape();

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to TM need to be on device! {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());
    TT_FATAL(!input_tensor.is_sharded(), "GDN create-heads supports only interleaved input");
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config layout must be INTERLEAVED but got {}",
        operation_attributes.output_mem_config.memory_layout());

    TT_FATAL(input_shape[1] == 1, "Unsupported input shape[1] {} is not equal to 1", input_shape[1]);
    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0, "Unsupported input height {} is not tile aligned", input_shape[2]);
    TT_FATAL(
        operation_attributes.head_dim % TILE_WIDTH == 0,
        "head_dim {} must be tile aligned",
        operation_attributes.head_dim);
    const uint32_t expected_w =
        (operation_attributes.num_q_heads + operation_attributes.num_k_heads + operation_attributes.num_v_heads) *
        operation_attributes.head_dim;
    TT_FATAL(
        input_shape[3] == expected_w, "Fused width {} must equal (Nq+Nk+Nv)*head_dim = {}", input_shape[3], expected_w);
}

void NlpCreateHeadsGdnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

NlpCreateHeadsGdnDeviceOperation::spec_return_value_t NlpCreateHeadsGdnDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    if (tensor_args.optional_output_tensors.size() == 3) {
        const auto& output_tensors = tensor_args.optional_output_tensors;
        return {
            output_tensors.at(0)->tensor_spec(),
            output_tensors.at(1)->tensor_spec(),
            output_tensors.at(2)->tensor_spec()};
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto input_shape = input_tensor.logical_shape();
    const auto sequence_length = input_shape[2];
    const auto head_dim = operation_attributes.head_dim;

    const Shape q_output_shape({input_shape[0], operation_attributes.num_q_heads, sequence_length, head_dim});
    const Shape k_output_shape({input_shape[0], operation_attributes.num_k_heads, sequence_length, head_dim});
    const Shape v_output_shape({input_shape[0], operation_attributes.num_v_heads, sequence_length, head_dim});

    auto make_spec = [&](const Shape& shape) {
        return TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config));
    };
    return {make_spec(q_output_shape), make_spec(k_output_shape), make_spec(v_output_shape)};
}

NlpCreateHeadsGdnDeviceOperation::tensor_return_value_t NlpCreateHeadsGdnDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    if (tensor_args.optional_output_tensors.size() == 3) {
        const auto& output_tensors = tensor_args.optional_output_tensors;
        return {output_tensors.at(0).value(), output_tensors.at(1).value(), output_tensors.at(2).value()};
    }
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), input_tensor.device()),
        create_device_tensor(std::get<1>(output_specs), input_tensor.device()),
        create_device_tensor(std::get<2>(output_specs), input_tensor.device()),
    };
}

NlpCreateHeadsGdnDeviceOperation::program_factory_t NlpCreateHeadsGdnDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return Interleaved{};
}

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_gdn(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::NlpCreateHeadsGdnDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_k_heads = num_k_heads,
        .num_v_heads = num_v_heads,
        .head_dim = head_dim,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .optional_output_tensors = optional_output_tensors.value_or(std::vector<std::optional<Tensor>>{})};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
