// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

void MoeGroupedTopkDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;

    TT_FATAL(scores.storage_type() == StorageType::DEVICE, "Scores tensor must be on device");
    TT_FATAL(bias.storage_type() == StorageType::DEVICE, "Bias tensor must be on device");
    TT_FATAL(scores.buffer() != nullptr, "Scores tensor must be allocated");
    TT_FATAL(bias.buffer() != nullptr, "Bias tensor must be allocated");

    TT_FATAL(scores.dtype() == DataType::FLOAT32, "Scores tensor must be FLOAT32");
    TT_FATAL(scores.layout() == Layout::TILE, "Scores tensor must be TILE layout");
    TT_FATAL(bias.dtype() == DataType::FLOAT32, "Bias tensor must be FLOAT32");
    TT_FATAL(bias.layout() == Layout::TILE, "Bias tensor must be TILE layout");
    TT_FATAL(scores.logical_shape() == bias.logical_shape(), "Scores and bias must have the same shape");

    TT_FATAL(
        attributes.summed_experts_per_group == 2,
        "summed_experts_per_group must be 2 at the moment. Got {}",
        attributes.summed_experts_per_group);

    TT_FATAL(attributes.n_groups == 8, "n_groups must be 8 at the moment. Got {}", attributes.n_groups);
    TT_FATAL(attributes.topk_groups == 4, "topk_groups must be 4 at the moment. Got {}", attributes.topk_groups);

    TT_FATAL(scores.logical_shape()[-1] == 256, "Experts must be 256. Got {}", scores.logical_shape()[-1]);

    TT_FATAL(
        attributes.n_activated_experts == 8,
        "n_activated_experts must be 8 at the moment. Got {}",
        attributes.n_activated_experts);
}

MoeGroupedTopkDeviceOperation::spec_return_value_t MoeGroupedTopkDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    auto shape = scores.logical_shape();

    auto output_shape = shape;
    output_shape[-1] = attributes.n_activated_experts;

    return std::array<TensorSpec, 2>{
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                DataType::BFLOAT16, tt::tt_metal::PageConfig(scores.layout()), attributes.output_mem_config)),
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                DataType::UINT16, tt::tt_metal::PageConfig(scores.layout()), attributes.output_mem_config))};
}

MoeGroupedTopkDeviceOperation::tensor_return_value_t MoeGroupedTopkDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attributes, tensor_args);
    return std::array<Tensor, 2>{
        create_device_tensor(specs[0], tensor_args.scores.device()),
        create_device_tensor(specs[1], tensor_args.scores.device())};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::MoeGroupedTopkDeviceOperation::tensor_return_value_t
moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    const std::optional<MemoryConfig>& output_mem_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::MoeGroupedTopkDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        output_mem_config.value_or(scores.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{scores, bias};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
