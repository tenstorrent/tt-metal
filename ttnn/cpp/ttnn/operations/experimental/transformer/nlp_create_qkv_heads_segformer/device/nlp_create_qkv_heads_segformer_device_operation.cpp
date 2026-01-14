// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_segformer {

NlpCreateHeadsSegformerDeviceOperation::program_factory_t
NlpCreateHeadsSegformerDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::NlpCreateQkvHeadsSegformerProgramFactory{};
}

void NlpCreateHeadsSegformerDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NlpCreateHeadsSegformerDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());

    TT_FATAL(
        input_shape[2] % tt::constants::TILE_HEIGHT == 0,
        "Input shape[2] ({}) must be divisible by TILE_HEIGHT ({})",
        input_shape[2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        input_shape[3] % tt::constants::TILE_HEIGHT == 0,
        "Input shape[3] ({}) must be divisible by TILE_HEIGHT ({})",
        input_shape[3],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config layout must be INTERLEAVED but got {}",
        args.output_mem_config.memory_layout());

    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    TT_FATAL(
        optional_output_tensors.empty() || optional_output_tensors.size() == 3,
        "Optional output tensors must be empty or have exactly 3 tensors (Q, K, V), but got {}",
        optional_output_tensors.size());
    if (optional_output_tensors.size() == 3) {
        TT_FATAL(
            optional_output_tensors[0].has_value() && optional_output_tensors[1].has_value() &&
                optional_output_tensors[2].has_value(),
            "All 3 optional output tensors must have values when provided");
    }
}

spec_return_value_t NlpCreateHeadsSegformerDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.output_mem_config.is_sharded()) {
        TT_FATAL(false, "Sharded output memory config is not supported for nlp_create_qkv_heads_segformer");
        TensorSpec spec(
            ttnn::Shape({}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::INVALID,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                args.output_mem_config));
        return {spec, spec, spec};
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const auto head_dim = 32;                                      // head_dim is hard-coded = 32
    auto num_heads = input_shape[3] / tt::constants::TILE_HEIGHT;  // head_dim is hard-coded = 32
    TensorSpec spec(
        ttnn::Shape({input_shape[0], num_heads, input_shape[2], head_dim}),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), args.output_mem_config));
    return {spec, spec, spec};
}

tensor_return_value_t NlpCreateHeadsSegformerDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    if (!optional_output_tensors.empty()) {
        return {
            optional_output_tensors.at(0).value(),
            optional_output_tensors.at(1).value(),
            optional_output_tensors.at(2).value()};
    }
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), tensor_args.input_tensor.device()),
        create_device_tensor(std::get<1>(output_specs), tensor_args.input_tensor.device()),
        create_device_tensor(std::get<2>(output_specs), tensor_args.input_tensor.device())};
}

}  // namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_segformer

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_segformer(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::nlp_create_qkv_heads_segformer::
        NlpCreateHeadsSegformerDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = output_mem_config,
    };
    auto tensor_args =
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensors = optional_output_tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
