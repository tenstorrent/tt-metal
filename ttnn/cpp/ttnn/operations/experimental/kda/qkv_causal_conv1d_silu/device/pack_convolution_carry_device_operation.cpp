// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_convolution_carry_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PackConvolutionCarryOperation::program_factory_t PackConvolutionCarryOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PackConvolutionCarryProgramFactory{};
}

void PackConvolutionCarryOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr std::string_view operation_name = "pack_convolution_carry";
    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::ROW_MAJOR, operation_name, "input");
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    check_allocated_device_tensor(in.wrap_indicator, operation_name, "wrap_indicator");
    check_layout(in.wrap_indicator, Layout::TILE, operation_name, "wrap_indicator");
    check_dtype(in.wrap_indicator, DataType::FLOAT32, operation_name, "wrap_indicator");
    check_interleaved(in.wrap_indicator, operation_name, "wrap_indicator");
    check_same_device(in.input, in.wrap_indicator, operation_name, "wrap_indicator");
    TT_FATAL(
        in.wrap_indicator.logical_volume() >= 1,
        "pack_convolution_carry: wrap_indicator must contain at least one scalar");

    const auto& shape = in.input.logical_shape();
    TT_FATAL(
        shape.rank() == 3 && shape[0] == 1 && shape[1] == attrs.sequence && shape[2] == attrs.channels,
        "pack_convolution_carry: input must be [1,T,C]");
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "pack_convolution_carry: sequence must be positive and tile aligned");
    TT_FATAL(
        attrs.channels > 0 && attrs.channels % tt::constants::TILE_WIDTH == 0,
        "pack_convolution_carry: channels must be positive and tile aligned");
    TT_FATAL(
        attrs.wrap_row >= attrs.history_rows && attrs.wrap_row < attrs.sequence &&
            attrs.wrap_row % tt::constants::TILE_HEIGHT == 0,
        "pack_convolution_carry: wrap_row {} must be tile aligned and inside [{}, {})",
        attrs.wrap_row,
        attrs.history_rows,
        attrs.sequence);
    TT_FATAL(
        attrs.history_rows > 0 && 2 * attrs.history_rows <= tt::constants::TILE_HEIGHT,
        "pack_convolution_carry: twice history_rows must fit one tile");
    check_output_interleaved(attrs.output_mem_config, operation_name);
    check_compute_config(attrs.compute_kernel_config, operation_name);
}

PackConvolutionCarryOperation::spec_return_value_t PackConvolutionCarryOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {TensorSpec(Shape({1, tt::constants::TILE_HEIGHT, attrs.channels}), layout)};
}

PackConvolutionCarryOperation::tensor_return_value_t PackConvolutionCarryOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.input.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PackConvolutionCarryOperation::tensor_return_value_t>
PackConvolutionCarryOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;
    const KdaFpuWork work{};
    const std::array<const Tensor*, 2> inputs = {&in.input, &in.wrap_indicator};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::vector<Tensor> pack_convolution_carry(
    const Tensor& input,
    const Tensor& wrap_indicator,
    uint32_t wrap_row,
    uint32_t history_rows,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 3, "pack_convolution_carry: input must be [1,T,C]");
    return ttnn::device_operation::launch<PackConvolutionCarryOperation>(
        PackConvolutionCarryParams{
            .sequence = static_cast<uint32_t>(shape[1]),
            .channels = static_cast<uint32_t>(shape[2]),
            .wrap_row = wrap_row,
            .history_rows = history_rows,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        PackConvolutionCarryInputs{.input = input, .wrap_indicator = wrap_indicator});
}

}  // namespace ttnn::experimental::prim
