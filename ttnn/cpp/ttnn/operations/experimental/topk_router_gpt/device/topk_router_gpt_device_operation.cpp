// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        attrs.num_experts == 128,
        "topk_router_gpt only supports num_experts=128 (hardcoded 4-group architecture), got {}",
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

    // Validate input shape: one physical tile of up to 32 logical tokens. The
    // kernel always computes that tile; sub-tile decode batches consume only
    // their logical rows.
    auto input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 2, "Input tensor must be rank 2, got rank {}", input_shape.rank());
    auto B = input_shape[0];
    auto hidden_dim = input_shape[1];
    TT_FATAL(B >= 1 && B <= 32, "topk_router_gpt requires batch_size in [1, 32], got {}", B);
    TT_FATAL(
        input.padded_shape()[0] == 32,
        "topk_router_gpt requires one physical 32-row input tile, got padded batch size {}",
        input.padded_shape()[0]);
    TT_FATAL(hidden_dim % 32 == 0, "Input hidden_dim must be divisible by 32 (tile size), got {}", hidden_dim);

    // Validate weight shape: [hidden_dim, num_experts]
    auto weight_shape = weight.logical_shape();
    TT_FATAL(weight_shape.rank() == 2, "Weight tensor must be rank 2, got rank {}", weight_shape.rank());
    TT_FATAL(
        weight_shape[0] == hidden_dim,
        "Weight tensor dim 0 must match input hidden_dim {}, got {}",
        hidden_dim,
        weight_shape[0]);
    TT_FATAL(
        weight_shape[1] == attrs.num_experts,
        "Weight tensor dim 1 must match num_experts {}, got {}",
        attrs.num_experts,
        weight_shape[1]);

    // Validate bias shape: [B, num_experts] (pre-broadcast)
    auto bias_shape = bias.logical_shape();
    TT_FATAL(bias_shape.rank() == 2, "Bias tensor must be rank 2, got rank {}", bias_shape.rank());
    TT_FATAL(bias_shape[0] == B, "Bias tensor dim 0 must match batch size {}, got {}", B, bias_shape[0]);
    TT_FATAL(
        bias.padded_shape()[0] == 32,
        "topk_router_gpt requires one physical 32-row bias tile, got padded batch size {}",
        bias.padded_shape()[0]);
    TT_FATAL(
        bias_shape[1] == attrs.num_experts,
        "Bias tensor dim 1 must match num_experts {}, got {}",
        attrs.num_experts,
        bias_shape[1]);

    // Validate all tensors are on the same device
    TT_FATAL(weight.device() == input.device(), "All tensors must be on the same device");
    TT_FATAL(bias.device() == input.device(), "All tensors must be on the same device");
}

spec_return_value_t TopkRouterGptDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto input_shape = tensor_args.input_tensor.logical_shape();
    auto B = input_shape[0];
    auto l1_rm = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1};

    uint32_t k_padded = tt::round_up(attrs.k, 8);

    const auto logical_shape = ttnn::Shape({B, attrs.k});
    const auto padded_shape = ttnn::Shape({32, k_padded});

    // The data-movement kernel emits all 32 physical rows. Preserve the
    // caller's logical B while allocating those rows in the RM backing buffer.
    const auto idx_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        DataType::UINT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), l1_rm, logical_shape, padded_shape);
    auto idx_spec = tt::tt_metal::TensorSpec(logical_shape, idx_layout);

    const auto wgt_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), l1_rm, logical_shape, padded_shape);
    auto wgt_spec = tt::tt_metal::TensorSpec(logical_shape, wgt_layout);

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

namespace ttnn::experimental {
std::tuple<Tensor, Tensor> topk_router_gpt(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& bias_tensor,
    uint32_t k,
    uint32_t num_experts) {
    auto [operation_attributes, tensors_args] =
        operations::experimental::topk_router_gpt::TopkRouterGptDeviceOperation::invoke(
            input_tensor, weight_tensor, bias_tensor, k, num_experts);

    return ttnn::device_operation::launch<operations::experimental::topk_router_gpt::TopkRouterGptDeviceOperation>(
        operation_attributes, tensors_args);
}
}  // namespace ttnn::experimental
