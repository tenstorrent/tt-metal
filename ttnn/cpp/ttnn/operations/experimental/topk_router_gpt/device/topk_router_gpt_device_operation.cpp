// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_device_operation.hpp"

#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::topk_router_gpt {

void TopkRouterGptDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void TopkRouterGptDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;
    const auto& bias = tensor_args.bias_tensor;

    TT_FATAL(attrs.k >= 1 && attrs.k <= 32, "topk_router_gpt requires k in range [1, 32], got {}", attrs.k);

    TT_FATAL(
        attrs.num_experts == 64 || attrs.num_experts == 128,
        "topk_router_gpt supports num_experts=64 or 128, got {}",
        attrs.num_experts);

    // Validate input tensor properties
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device, got storage_type {}",
        input.storage_type());
    TT_FATAL(input.buffer() != nullptr, "Input tensor must have allocated buffer");
    TT_FATAL(input.layout() == Layout::TILE, "Input tensor must have TILE layout, got {}", input.layout());
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input tensor must have BFLOAT16 dtype, got {}", input.dtype());

    // Validate weight tensor properties
    TT_FATAL(
        weight.storage_type() == StorageType::DEVICE,
        "Weight tensor must be on device, got storage_type {}",
        weight.storage_type());
    TT_FATAL(weight.buffer() != nullptr, "Weight tensor must have allocated buffer");
    TT_FATAL(weight.layout() == Layout::TILE, "Weight tensor must have TILE layout, got {}", weight.layout());
    TT_FATAL(weight.dtype() == DataType::BFLOAT16, "Weight tensor must have BFLOAT16 dtype, got {}", weight.dtype());

    // Validate bias tensor properties
    TT_FATAL(
        bias.storage_type() == StorageType::DEVICE,
        "Bias tensor must be on device, got storage_type {}",
        bias.storage_type());
    TT_FATAL(bias.buffer() != nullptr, "Bias tensor must have allocated buffer");
    TT_FATAL(bias.layout() == Layout::TILE, "Bias tensor must have TILE layout, got {}", bias.layout());
    TT_FATAL(bias.dtype() == DataType::BFLOAT16, "Bias tensor must have BFLOAT16 dtype, got {}", bias.dtype());

    // Validate input shape. GPT uses [32, hidden_dim]; GLM decode uses
    // [1, 1, tokens, hidden_dim] with one physical tile of token rows.
    auto input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() >= 2, "Input tensor must have rank >= 2, got rank {}", input_shape.rank());
    auto tokens = input_shape[-2];
    auto hidden_dim = input_shape[-1];
    TT_FATAL(
        (attrs.num_experts == 128 && input_shape.rank() == 2 && tokens == 32) ||
            (attrs.num_experts == 64 && tokens >= 1 && tokens <= 32),
        "Unsupported input shape {} for num_experts={}",
        input_shape,
        attrs.num_experts);
    TT_FATAL(hidden_dim % 32 == 0, "Input hidden_dim must be divisible by 32 (tile size), got {}", hidden_dim);

    // Validate weight's final two dimensions: [hidden_dim, num_experts].
    auto weight_shape = weight.logical_shape();
    TT_FATAL(weight_shape.rank() >= 2, "Weight tensor must have rank >= 2, got rank {}", weight_shape.rank());
    TT_FATAL(
        weight_shape[-2] == hidden_dim,
        "Weight tensor dim 0 must match input hidden_dim {}, got {}",
        hidden_dim,
        weight_shape[-2]);
    TT_FATAL(
        weight_shape[-1] == attrs.num_experts,
        "Weight tensor dim 1 must match num_experts {}, got {}",
        attrs.num_experts,
        weight_shape[-1]);

    // Validate bias's final dimensions. Its token dimension may be one because
    // TILE padding physically broadcasts the correction-bias row for GLM B1.
    auto bias_shape = bias.logical_shape();
    TT_FATAL(bias_shape.rank() >= 2, "Bias tensor must have rank >= 2, got rank {}", bias_shape.rank());
    TT_FATAL(
        bias_shape[-1] == attrs.num_experts,
        "Bias tensor dim 1 must match num_experts {}, got {}",
        attrs.num_experts,
        bias_shape[-1]);

    // Validate all tensors are on the same device
    TT_FATAL(weight.device() == input.device(), "All tensors must be on the same device");
    TT_FATAL(bias.device() == input.device(), "All tensors must be on the same device");
}

spec_return_value_t TopkRouterGptDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto input_shape = tensor_args.input_tensor.logical_shape();
    auto dram = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    if (attrs.num_experts == 64) {
        auto tokens = input_shape[-2];
        auto output_shape = ttnn::Shape({1, 1, tokens, attrs.k});
        auto idx_spec = TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(DataType::UINT16, tt::tt_metal::PageConfig(Layout::TILE), dram));
        auto wgt_spec = TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), dram));
        return {idx_spec, wgt_spec};
    }

    uint32_t k_padded = tt::round_up(attrs.k, 8);
    auto B = input_shape[0];
    // Slot 0: indices_rm [B, k_padded] uint16 RM
    auto idx_spec = TensorSpec(
        ttnn::Shape({B, k_padded}),
        tt::tt_metal::TensorLayout(DataType::UINT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), dram));

    // Slot 1: weights_rm [B, k_padded] bf16 RM
    auto wgt_spec = TensorSpec(
        ttnn::Shape({B, k_padded}),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), dram));

    return {idx_spec, wgt_spec};
}

tensor_return_value_t TopkRouterGptDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    auto idx_tensor = create_device_tensor(std::get<0>(specs), device);
    auto wgt_tensor = create_device_tensor(std::get<1>(specs), device);
    return {idx_tensor, wgt_tensor};
}

std::tuple<TopkRouterGptDeviceOperation::operation_attributes_t, TopkRouterGptDeviceOperation::tensor_args_t>
TopkRouterGptDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const Tensor& bias_tensor,
    uint32_t k,
    uint32_t num_experts) {
    return {
        operation_attributes_t{.k = k, .num_experts = num_experts},
        tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor, .bias_tensor = bias_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
