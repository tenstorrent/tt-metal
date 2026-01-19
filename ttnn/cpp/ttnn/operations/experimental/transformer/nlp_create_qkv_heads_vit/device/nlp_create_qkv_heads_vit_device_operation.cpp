// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

NlpCreateHeadsVitDeviceOperation::program_factory_t NlpCreateHeadsVitDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return NlpCreateQkvHeadsVitProgramFactory{};
}

void NlpCreateHeadsVitDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NlpCreateHeadsVitDeviceOperation::validate_on_program_cache_miss(
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
    TT_FATAL((input_shape == ttnn::Shape({input_shape[0], 1, input_shape[2], 2304})), "Unsupported input shape");
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config layout must be INTERLEAVED but got {}",
        args.output_mem_config.memory_layout());

    if (tensor_args.optional_output_tensors.has_value()) {
        const auto& opt_outputs = *tensor_args.optional_output_tensors;
        TT_FATAL(
            opt_outputs.size() == 3,
            "Optional output tensors must have exactly 3 elements (Q, K, V), but got {}",
            opt_outputs.size());
        for (size_t i = 0; i < opt_outputs.size(); ++i) {
            TT_FATAL(
                opt_outputs[i].has_value(),
                "All 3 optional output tensors must have values, but tensor at index {} is nullopt",
                i);
        }
    }
}

NlpCreateHeadsVitDeviceOperation::spec_return_value_t NlpCreateHeadsVitDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    }
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    TensorSpec spec(
        Shape({input_shape[0], 12, input_shape[2], 64}),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), args.output_mem_config));
    return {spec, spec, spec};
}

NlpCreateHeadsVitDeviceOperation::tensor_return_value_t NlpCreateHeadsVitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    if (optional_output_tensors.has_value()) {
        tensor_return_value_t outputs;
        outputs.reserve(3);
        for (const auto& opt_tensor : *optional_output_tensors) {
            outputs.push_back(opt_tensor.value());
        }
        return outputs;
    }

    auto output_specs = compute_output_specs(args, tensor_args);
    tensor_return_value_t outputs;
    outputs.reserve(output_specs.size());
    for (const auto& spec : output_specs) {
        outputs.push_back(create_device_tensor(spec, tensor_args.input_tensor.device()));
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> nlp_create_qkv_heads_vit(
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    using OperationType = ttnn::experimental::prim::NlpCreateHeadsVitDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = output_mem_config,
    };
    auto tensor_args =
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensors = optional_output_tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
