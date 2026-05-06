// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion_device_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

YUVConversionDeviceOperation::program_factory_t YUVConversionDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return YUVConversionProgramFactory{};
}

void YUVConversionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    TT_FATAL(in.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(in.buffer() != nullptr, "Input buffer must be allocated");
    TT_FATAL(in.dtype() == DataType::BFLOAT16, "Input must be bfloat16");
    TT_FATAL(in.layout() == Layout::ROW_MAJOR, "Input must be row-major (CHWT)");

    const auto& shape = in.logical_shape();
    TT_FATAL(shape.rank() == 4, "Input must be 4D (C, H, W, T)");
    TT_FATAL(shape[0] == 3, "Input must have C=3 (RGB)");
    TT_FATAL(shape[1] % 2 == 0, "H must be even for 4:2:0 subsampling");
    TT_FATAL(shape[2] % 2 == 0, "W must be even for 4:2:0 subsampling");
}

void YUVConversionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

YUVConversionDeviceOperation::spec_return_value_t YUVConversionDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& shape = tensor_args.input.logical_shape();
    uint32_t H = shape[1], W = shape[2], T = shape[3];

    auto mem_cfg = attrs.output_memory_config;
    auto uint8_layout = TensorLayout(DataType::UINT8, Layout::ROW_MAJOR, mem_cfg);

    TensorSpec y_spec(ttnn::Shape{1, H, W, T}, uint8_layout);
    TensorSpec u_spec(ttnn::Shape{1, H / 2, W / 2, T}, uint8_layout);
    TensorSpec v_spec(ttnn::Shape{1, H / 2, W / 2, T}, uint8_layout);

    return {y_spec, u_spec, v_spec};
}

YUVConversionDeviceOperation::tensor_return_value_t YUVConversionDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [y_spec, u_spec, v_spec] = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input.device();
    return {
        create_device_tensor(y_spec, device),
        create_device_tensor(u_spec, device),
        create_device_tensor(v_spec, device),
    };
}

ttsl::hash::hash_t YUVConversionDeviceOperation::compute_program_hash(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    return operation::hash_operation<YUVConversionDeviceOperation>(
        in.dtype(), in.memory_config(), in.logical_shape().volume());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    const ttnn::experimental::prim::YUVCoefficients& coefficients,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using Op = ttnn::experimental::prim::YUVConversionDeviceOperation;

    auto op_attrs = Op::operation_attributes_t{
        .coefficients = coefficients,
        .output_memory_config = memory_config.value_or(input.memory_config()),
    };
    auto tensor_args = Op::tensor_args_t{.input = input};
    return ttnn::device_operation::launch<Op>(op_attrs, tensor_args);
}

}  // namespace ttnn::prim
