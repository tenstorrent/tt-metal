// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "router_front_end_device_operation.hpp"

#include <cstdint>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t kTileSize = tt::constants::TILE_HEIGHT;

void check_interleaved(const tt::tt_metal::MemoryConfig& memory_config, std::string_view name) {
    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "router_front_end: {} must be interleaved, got {}",
        name,
        memory_config.memory_layout());
}

}  // namespace

// The framework rejects host tensors before this runs.
RouterFrontEndLayout RouterFrontEndDeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "router_front_end: input must be BFLOAT16, got {}", input.dtype());
    TT_FATAL(input.layout() == Layout::TILE, "router_front_end: input must be TILE layout, got {}", input.layout());
    TT_FATAL(weight.dtype() == DataType::FLOAT32, "router_front_end: weight must be FLOAT32, got {}", weight.dtype());
    TT_FATAL(weight.layout() == Layout::TILE, "router_front_end: weight must be TILE layout, got {}", weight.layout());
    check_interleaved(weight.memory_config(), "weight");
    TT_FATAL(weight.device() == input.device(), "router_front_end: weight must be on the device of input");
    check_interleaved(attributes.scores_memory_config, "scores_memory_config");
    check_interleaved(attributes.rows_memory_config, "rows_memory_config");

    const auto& shape = input.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1,
        "router_front_end: input must be 1 x 1 x tokens x hidden, got {}",
        shape);
    const uint32_t num_tokens = shape[2];
    const uint32_t hidden = shape[3];
    TT_FATAL(num_tokens >= 1, "router_front_end: input has no tokens");
    TT_FATAL(hidden % kTileSize == 0, "router_front_end: hidden must be whole tiles, got {}", hidden);

    const auto& weight_shape = weight.logical_shape();
    TT_FATAL(
        weight_shape.rank() >= 2 && weight_shape[-2] == hidden,
        "router_front_end: weight must be {} x experts, got {}",
        hidden,
        weight_shape);
    for (int dim = 0; dim + 2 < static_cast<int>(weight_shape.rank()); ++dim) {
        TT_FATAL(
            weight_shape[dim] == 1, "router_front_end: weight's leading dimensions must be 1, got {}", weight_shape);
    }
    const uint32_t num_experts = weight_shape[-1];
    TT_FATAL(
        num_experts >= 1 && num_experts <= kTileSize,
        "router_front_end: experts must be 1 to {}, got {}",
        kTileSize,
        num_experts);

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), attributes.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "router_front_end: compute_kernel_config must enable fp32_dest_acc_en");

    return RouterFrontEndLayout{
        .num_tokens = num_tokens,
        .hidden_tiles = hidden / kTileSize,
        .num_experts = num_experts,
        .num_blocks = tt::div_up(num_tokens, kTileSize),
    };
}

void RouterFrontEndDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

RouterFrontEndDeviceOperation::spec_return_value_t RouterFrontEndDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;
    using tt::tt_metal::TensorSpec;

    const auto layout = compute_layout(attributes, tensor_args);
    return {
        TensorSpec(
            ttnn::Shape({1, 1, layout.num_tokens, layout.num_experts}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attributes.scores_memory_config)),
        TensorSpec(
            ttnn::Shape({1, 1, layout.num_tokens, layout.hidden_tiles * kTileSize}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attributes.rows_memory_config)),
    };
}

RouterFrontEndDeviceOperation::tensor_return_value_t RouterFrontEndDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto [scores, rows] = compute_output_specs(attributes, tensor_args);
    auto* device = tensor_args.input.device();
    return {create_device_tensor(scores, device), create_device_tensor(rows, device)};
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

std::tuple<Tensor, Tensor> router_front_end(
    const Tensor& input,
    const Tensor& weight,
    const tt::tt_metal::MemoryConfig& scores_memory_config,
    const tt::tt_metal::MemoryConfig& rows_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::RouterFrontEndDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .scores_memory_config = scores_memory_config,
            .rows_memory_config = rows_memory_config,
            .compute_kernel_config = compute_kernel_config},
        Operation::tensor_args_t{.input = input, .weight = weight});
}

}  // namespace ttnn::prim
