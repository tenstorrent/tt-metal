// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/moe/device/moe_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>

#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"
#include "ttnn/operations/reduction/moe/device/moe_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::moe {

MoeDeviceOperation::program_factory_t MoeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoeProgramFactory{};
}

void MoeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& expert_mask_tensor = tensor_args.expert_mask;
    const auto& topk_mask_tensor = tensor_args.topk_mask;

    auto input_shape = input_tensor.padded_shape();
    auto topk_shape = topk_mask_tensor.padded_shape();
    auto expert_shape = expert_mask_tensor.padded_shape();

    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());
    TT_FATAL(args.k == 32, "K must be equal to 32, pad with -infinity if necessary to get 32, got {}", args.k);

    TT_FATAL(
        input_shape[-1] >= 64,
        "Input shape inner dim {} must be a multiple of 64, pad with -infinity if necessary",
        input_shape[-1]);
    TT_FATAL(
        (input_shape[-1] & (input_shape[-1] - 1)) == 0,
        "Input shape inner dim {} must be a power of 2, pad with -infinity if necessary",
        input_shape[-1]);
    TT_FATAL(
        (input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-2]) {} must be a multiple of 32",
        input_shape[0] * input_shape[1] * input_shape[2]);

    TT_FATAL(args.output_memory_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    TT_FATAL(topk_shape[-1] == args.k, "Topk shape inner dim must be equal to k, got {}", topk_shape[-1]);
    TT_FATAL(
        expert_shape[-1] == input_shape[-1],
        "Expert shape inner dim must be equal to input_shape[-1], got {}",
        expert_shape[-1]);
    TT_FATAL(topk_shape[-2] == 32, "Topk shape inner dim must be equal to 32, got {}", topk_shape[-2]);
    TT_FATAL(expert_shape[-2] == 32, "Expert shape inner dim must be equal to 32, got {}", expert_shape[-2]);
}

TensorSpec MoeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    auto output_shape = input_tensor.logical_shape();
    output_shape[-1] = 1;
    return TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), args.output_memory_config));
}

Tensor MoeDeviceOperation::create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::moe

namespace ttnn::prim {
ttnn::Tensor moe(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output_tensor) {
    using OperationType = ttnn::operations::reduction::moe::MoeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .k = k,
            .output_memory_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG)},
        OperationType::tensor_args_t{
            .input = input_tensor,
            .expert_mask = expert_mask_tensor,
            .topk_mask = topk_mask_tensor,
            .preallocated_output = preallocated_output_tensor});
}
}  // namespace ttnn::prim
