// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_hash_gate_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

void MoeHashGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& scores = tensor_args.scores;
    const auto& input_ids = tensor_args.input_ids;
    const auto& tid2eid = tensor_args.tid2eid;
    const auto& padding_config = tensor_args.padding_config;

    TT_FATAL(scores.storage_type() == ttnn::StorageType::DEVICE, "Scores tensor must be on device");
    TT_FATAL(input_ids.storage_type() == ttnn::StorageType::DEVICE, "input_ids tensor must be on device");
    TT_FATAL(tid2eid.storage_type() == ttnn::StorageType::DEVICE, "tid2eid tensor must be on device");
    TT_FATAL(scores.buffer() != nullptr, "Scores tensor must be allocated");
    TT_FATAL(input_ids.buffer() != nullptr, "input_ids tensor must be allocated");
    TT_FATAL(tid2eid.buffer() != nullptr, "tid2eid tensor must be allocated");

    TT_FATAL(scores.dtype() == tt::tt_metal::DataType::FLOAT32, "Scores tensor must be FLOAT32");
    TT_FATAL(scores.layout() == tt::tt_metal::Layout::TILE, "Scores tensor must be TILE layout");
    TT_FATAL(input_ids.dtype() == tt::tt_metal::DataType::UINT32, "input_ids tensor must be UINT32");
    TT_FATAL(input_ids.layout() == tt::tt_metal::Layout::ROW_MAJOR, "input_ids tensor must be ROW_MAJOR layout");
    TT_FATAL(tid2eid.dtype() == tt::tt_metal::DataType::UINT16, "tid2eid tensor must be UINT16");
    TT_FATAL(tid2eid.layout() == tt::tt_metal::Layout::ROW_MAJOR, "tid2eid tensor must be ROW_MAJOR layout");

    // Optional per-device [num_real_tokens, pad_side] config used to sentinel-mark padded token rows.
    if (padding_config.has_value()) {
        TT_FATAL(
            padding_config->storage_type() == ttnn::StorageType::DEVICE, "Padding config tensor must be on device");
        TT_FATAL(padding_config->buffer() != nullptr, "Padding config tensor must be allocated");
        TT_FATAL(padding_config->dtype() == tt::tt_metal::DataType::UINT32, "Padding config tensor must be UINT32");
        TT_FATAL(
            padding_config->layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Padding config tensor must be ROW_MAJOR layout");
        TT_FATAL(
            padding_config->logical_shape()[-1] >= 2,
            "Padding config tensor must contain at least [num_real_tokens, pad_side]");
    }

    const uint32_t experts = scores.logical_shape()[-1];
    TT_FATAL(experts % 32 == 0, "Number of experts must be a multiple of the tile width (32). Got {}", experts);
    TT_FATAL(attributes.n_activated_experts > 0, "n_activated_experts must be > 0");
    TT_FATAL(
        attributes.n_activated_experts <= 32,
        "n_activated_experts must be <= 32 (single index tile). Got {}",
        attributes.n_activated_experts);
    TT_FATAL(
        attributes.n_activated_experts <= experts,
        "n_activated_experts ({}) must be <= experts ({})",
        attributes.n_activated_experts,
        experts);
    // tid2eid rows hold at least n_activated_experts expert ids (row may be padded for alignment).
    TT_FATAL(
        tid2eid.logical_shape()[-1] >= attributes.n_activated_experts,
        "tid2eid last dim ({}) must be >= n_activated_experts ({})",
        tid2eid.logical_shape()[-1],
        attributes.n_activated_experts);
    // input_ids must supply one vocabulary id per score token row (the reader indexes tid2eid by these).
    const uint32_t num_token_rows = scores.logical_shape().volume() / experts;
    TT_FATAL(
        input_ids.logical_shape().volume() >= num_token_rows,
        "input_ids ({} ids) must supply at least one id per score token row ({})",
        input_ids.logical_shape().volume(),
        num_token_rows);
}

MoeHashGateDeviceOperation::spec_return_value_t MoeHashGateDeviceOperation::compute_output_specs(
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

MoeHashGateDeviceOperation::tensor_return_value_t MoeHashGateDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attributes, tensor_args);
    return std::array<Tensor, 2>{
        create_device_tensor(specs[0], tensor_args.scores.device()),
        create_device_tensor(specs[1], tensor_args.scores.device())};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::MoeHashGateDeviceOperation::tensor_return_value_t
moe_hash_gate(
    const Tensor& scores,
    const Tensor& input_ids,
    const Tensor& tid2eid,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::ScoreFunc score_func,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& padding_config) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::MoeHashGateDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        n_activated_experts, route_scale, epsilon, score_func, output_mem_config.value_or(scores.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{scores, input_ids, tid2eid, padding_config};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
