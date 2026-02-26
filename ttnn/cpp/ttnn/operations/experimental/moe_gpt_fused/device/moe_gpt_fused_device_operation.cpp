// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_fused_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused {

MoEGPTFusedDeviceOperation::program_factory_t MoEGPTFusedDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEGPTFusedProgramFactory{};
}

void MoEGPTFusedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEGPTFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
    TT_FATAL(args.num_experts >= 1, "Number of experts must be at least 1, got {}", args.num_experts);
    TT_FATAL(args.experts_per_device >= 1, "Experts per device must be at least 1, got {}", args.experts_per_device);
}

MoEGPTFusedDeviceOperation::spec_return_value_t MoEGPTFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Output: BLOCK_SHARDED L1 tensor in ROW_MAJOR layout
    // Shape: [E * tokens, K] = [128, 2880]
    // Sharded across 12 combine cores in a 4×3 logical grid (4 height × 3 width)
    // Shard shape: [E * tokens / 4, K / 3] = [32, 960]
    auto input_shape = tensor_args.input_tensor.logical_shape();
    uint32_t K = input_shape[-1];  // hidden_size = 2880
    uint32_t E = operation_attributes.experts_per_device;
    uint32_t tokens = 32;

    uint32_t height_shard_dim = 4;
    uint32_t width_shard_dim = 3;
    uint32_t shard_height = E * tokens / height_shard_dim;  // 32
    uint32_t shard_width = K / width_shard_dim;             // 960

    // Combine cores: 2 columns × 6 rows at CoreRange({5,0},{6,5})
    // ShardOrientation::ROW_MAJOR maps logical grid rows to height shards
    auto shard_spec = tt::tt_metal::ShardSpec(
        CoreRangeSet(CoreRange({1, 0}, {3, 3})),
        {shard_height, shard_width},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);

    auto mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    tt::tt_metal::TensorLayout tensor_layout(
        tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config);

    auto output_spec = tt::tt_metal::TensorSpec(ttnn::Shape({E * tokens, K}), tensor_layout);

    return {output_spec};
}

MoEGPTFusedDeviceOperation::tensor_return_value_t MoEGPTFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_specs.size());
    for (const auto& spec : output_specs) {
        output_tensors.push_back(tt::tt_metal::create_device_tensor(spec, tensor_args.input_tensor.device()));
    }
    return output_tensors;
}

std::tuple<MoEGPTFusedDeviceOperation::operation_attributes_t, MoEGPTFusedDeviceOperation::tensor_args_t>
MoEGPTFusedDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& expert_indices,
    const Tensor& expert_scores,
    const Tensor& w0_w1_tensor,
    const Tensor& w2_tensor,
    uint32_t num_experts,
    uint32_t layer_id,
    uint32_t experts_per_device) {
    return {
        operation_attributes_t{
            .num_experts = num_experts, .layer_id = layer_id, .experts_per_device = experts_per_device},
        tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices = expert_indices,
            .expert_scores = expert_scores,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor}};
}

}  // namespace ttnn::operations::experimental::moe_gpt_fused
