// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_causal_conv_device_operation.hpp"

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {
void check(const Tensor& tensor, const char* name, DataType dtype) {
    TT_FATAL(tensor.layout() == Layout::TILE, "kda_causal_conv1d_split: {} must be TILE layout", name);
    TT_FATAL(tensor.dtype() == dtype, "kda_causal_conv1d_split: {} has wrong dtype", name);
    TT_FATAL(tensor.buffer() != nullptr, "kda_causal_conv1d_split: {} must be on device", name);
}
}  // namespace

KdaCausalConvOperation::program_factory_t KdaCausalConvOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaCausalConvProgramFactory{};
}

void KdaCausalConvOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    TT_FATAL(in.input.layout() == Layout::ROW_MAJOR, "kda_causal_conv1d_split: input must be ROW_MAJOR");
    TT_FATAL(in.input.dtype() == DataType::BFLOAT16, "kda_causal_conv1d_split: input must be BFLOAT16");
    TT_FATAL(in.input.buffer() != nullptr, "kda_causal_conv1d_split: input must be on device");
    TT_FATAL(in.state.layout() == Layout::ROW_MAJOR, "kda_causal_conv1d_split: state must be ROW_MAJOR");
    TT_FATAL(in.state.dtype() == DataType::BFLOAT16, "kda_causal_conv1d_split: state must be BFLOAT16");
    TT_FATAL(in.state.buffer() != nullptr, "kda_causal_conv1d_split: state must be on device");
    check(in.tap0, "tap0", DataType::BFLOAT16);
    check(in.tap1, "tap1", DataType::BFLOAT16);
    check(in.tap2, "tap2", DataType::BFLOAT16);
    check(in.tap3, "tap3", DataType::BFLOAT16);
    const uint32_t channels = attrs.q_width + attrs.k_width + attrs.v_width;
    const auto& xs = in.input.logical_shape();
    const auto& ss = in.state.logical_shape();
    TT_FATAL(xs.rank() == 3 && xs[0] == 1 && xs[1] == attrs.sequence && xs[2] == channels, "input must be [1,T,Q+K+V]");
    TT_FATAL(ss.rank() == 3 && ss[0] == 1 && ss[1] == 3 && ss[2] == channels, "state must be [1,3,Q+K+V]");
    TT_FATAL(attrs.sequence % 32 == 0, "sequence must be tile aligned");
    TT_FATAL(
        attrs.q_width % 32 == 0 && attrs.k_width % 32 == 0 && attrs.v_width % 32 == 0,
        "Q/K/V widths must be tile aligned");
    TT_FATAL(in.tap0.logical_volume() == channels, "tap0 width mismatch");
    TT_FATAL(in.tap1.logical_volume() == channels, "tap1 width mismatch");
    TT_FATAL(in.tap2.logical_volume() == channels, "tap2 width mismatch");
    TT_FATAL(in.tap3.logical_volume() == channels, "tap3 width mismatch");
}

KdaCausalConvOperation::spec_return_value_t KdaCausalConvOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
}

KdaCausalConvOperation::tensor_return_value_t KdaCausalConvOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.input.device()),
        create_device_tensor(specs[1], in.input.device()),
        create_device_tensor(specs[2], in.input.device())};
}

std::vector<Tensor> kda_causal_conv1d_split(
    const Tensor& input,
    const Tensor& state,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = input.logical_shape();
    return ttnn::device_operation::launch<KdaCausalConvOperation>(
        KdaCausalConvParams{
            .sequence = static_cast<uint32_t>(shape[1]),
            .q_width = q_width,
            .k_width = k_width,
            .v_width = v_width,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        KdaCausalConvInputs{.input = input, .state = state, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3});
}

}  // namespace ttnn::prim
