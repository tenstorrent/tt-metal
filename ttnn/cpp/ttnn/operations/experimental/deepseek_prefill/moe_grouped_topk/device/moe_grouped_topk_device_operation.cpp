// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

void MoeGroupedTopkDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;
    const auto& padding_config = tensor_args.padding_config;

    TT_FATAL(scores.storage_type() == ttnn::StorageType::DEVICE, "Scores tensor must be on device");
    TT_FATAL(bias.storage_type() == ttnn::StorageType::DEVICE, "Bias tensor must be on device");
    TT_FATAL(scores.buffer() != nullptr, "Scores tensor must be allocated");
    TT_FATAL(bias.buffer() != nullptr, "Bias tensor must be allocated");

    TT_FATAL(scores.dtype() == tt::tt_metal::DataType::FLOAT32, "Scores tensor must be FLOAT32");
    TT_FATAL(scores.layout() == tt::tt_metal::Layout::TILE, "Scores tensor must be TILE layout");
    TT_FATAL(bias.dtype() == tt::tt_metal::DataType::FLOAT32, "Bias tensor must be FLOAT32");
    TT_FATAL(bias.layout() == tt::tt_metal::Layout::TILE, "Bias tensor must be TILE layout");
    TT_FATAL(scores.logical_shape() == bias.logical_shape(), "Scores and bias must have the same shape");

    // Optional per-device [num_real_tokens, pad_side] config used to sentinel-mark padded token rows.
    if (padding_config.has_value()) {
        TT_FATAL(
            padding_config->storage_type() == ttnn::StorageType::DEVICE, "Padding config tensor must be on device");
        TT_FATAL(padding_config->buffer() != nullptr, "Padding config tensor must be allocated");
        TT_FATAL(
            padding_config->dtype() == tt::tt_metal::DataType::UINT32, "Padding config tensor must be UINT32");
        TT_FATAL(
            padding_config->layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Padding config tensor must be ROW_MAJOR layout");
        TT_FATAL(
            padding_config->logical_shape()[-1] >= 2,
            "Padding config tensor must contain at least [num_real_tokens, pad_side]");
    }

    const uint32_t experts = scores.logical_shape()[-1];

    if (attributes.n_groups == 1) {
        // Single expert group: grouped routing collapses to a plain top-k over all experts.
        // topk_groups / summed_experts_per_group are unused by the kernel on this path. The expert
        // count is variable (any tile-aligned width, e.g. Kimi's 384), not hardcoded to DeepSeek.
        TT_FATAL(experts % 32 == 0, "Number of experts must be a multiple of the tile width (32). Got {}", experts);
        TT_FATAL(attributes.n_activated_experts > 0, "n_activated_experts must be > 0");
        TT_FATAL(
            attributes.n_activated_experts <= 64,
            "n_activated_experts must be <= 64 (topk limit). Got {}",
            attributes.n_activated_experts);
        TT_FATAL(
            attributes.n_activated_experts <= experts,
            "n_activated_experts ({}) must be <= experts ({})",
            attributes.n_activated_experts,
            experts);
        return;
    }

    // Grouped path (e.g. DeepSeek). Group routing currently assumes one tile (32 experts) per group.
    TT_FATAL(
        attributes.summed_experts_per_group == 2,
        "summed_experts_per_group must be 2 at the moment. Got {}",
        attributes.summed_experts_per_group);

    TT_FATAL(attributes.n_groups == 8, "n_groups must be 8 at the moment. Got {}", attributes.n_groups);
    TT_FATAL(attributes.topk_groups == 4, "topk_groups must be 4 at the moment. Got {}", attributes.topk_groups);

    TT_FATAL(experts == 256, "Experts must be 256. Got {}", experts);

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

    return std::array<tt::tt_metal::TensorSpec, 2>{
        tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(scores.layout()),
                attributes.output_mem_config)),
        tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT16,
                tt::tt_metal::PageConfig(scores.layout()),
                attributes.output_mem_config))};
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
    bool stable_sort,
    ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::ScoreFunc score_func,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& padding_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::MoeGroupedTopkDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        stable_sort,
        score_func,
        output_mem_config.value_or(scores.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{scores, bias, padding_config};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
