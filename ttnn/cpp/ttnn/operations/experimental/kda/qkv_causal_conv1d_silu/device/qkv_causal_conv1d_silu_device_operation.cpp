// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_device_tensor(const Tensor& tensor, const char* name, Layout layout) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "qkv_causal_conv1d_silu: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == layout, "qkv_causal_conv1d_silu: {} has unsupported layout", name);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "qkv_causal_conv1d_silu: {} must be BFLOAT16", name);
    TT_FATAL(!tensor.is_sharded(), "qkv_causal_conv1d_silu: {} must use interleaved memory", name);
}

}  // namespace

QkvCausalConv1dSiluOperation::program_factory_t QkvCausalConv1dSiluOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QkvCausalConv1dSiluProgramFactory{};
}

void QkvCausalConv1dSiluOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_device_tensor(in.input, "input", Layout::ROW_MAJOR);
    check_device_tensor(in.history, "history", Layout::ROW_MAJOR);
    check_device_tensor(in.tap0, "tap0", Layout::TILE);
    check_device_tensor(in.tap1, "tap1", Layout::TILE);
    check_device_tensor(in.tap2, "tap2", Layout::TILE);
    check_device_tensor(in.tap3, "tap3", Layout::TILE);

    auto* const input_device = in.input.device();
    for (const auto* tensor : std::array{&in.history, &in.tap0, &in.tap1, &in.tap2, &in.tap3}) {
        TT_FATAL(tensor->device() == input_device, "qkv_causal_conv1d_silu: all inputs must be on the same device");
    }

    TT_FATAL(
        attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be positive");
    TT_FATAL(
        attrs.q_width % tt::constants::TILE_WIDTH == 0 && attrs.k_width % tt::constants::TILE_WIDTH == 0 &&
            attrs.v_width % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be tile aligned");
    const uint64_t channels =
        static_cast<uint64_t>(attrs.q_width) + static_cast<uint64_t>(attrs.k_width) + attrs.v_width;

    const auto& input_shape = in.input.logical_shape();
    const auto& history_shape = in.history.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == 1 && input_shape[1] == attrs.sequence &&
            input_shape[2] == channels,
        "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    TT_FATAL(
        history_shape.rank() == 3 && history_shape[0] == 1 && history_shape[1] == 3 && history_shape[2] == channels,
        "qkv_causal_conv1d_silu: history must be [1,3,Q+K+V]");
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "qkv_causal_conv1d_silu: sequence must be positive and tile aligned");

    for (const auto [tensor, name] : std::array{
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"},
             std::pair{&in.tap3, "tap3"}}) {
        TT_FATAL(
            tensor->logical_volume() == channels, "qkv_causal_conv1d_silu: {} logical volume must equal Q+K+V", name);
    }
    TT_FATAL(
        !attrs.output_mem_config.is_sharded(),
        "qkv_causal_conv1d_silu: output memory configuration must be interleaved");
}

QkvCausalConv1dSiluOperation::spec_return_value_t QkvCausalConv1dSiluOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
}

QkvCausalConv1dSiluOperation::tensor_return_value_t QkvCausalConv1dSiluOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.input.device()),
        create_device_tensor(specs[1], in.input.device()),
        create_device_tensor(specs[2], in.input.device())};
}

std::vector<Tensor> qkv_causal_conv1d_silu(
    const Tensor& input,
    const Tensor& history,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    return ttnn::device_operation::launch<QkvCausalConv1dSiluOperation>(
        QkvCausalConv1dSiluParams{
            .sequence = static_cast<uint32_t>(input_shape[1]),
            .q_width = q_width,
            .k_width = k_width,
            .v_width = v_width,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        QkvCausalConv1dSiluInputs{
            .input = input, .history = history, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3});
}

}  // namespace ttnn::experimental::prim
