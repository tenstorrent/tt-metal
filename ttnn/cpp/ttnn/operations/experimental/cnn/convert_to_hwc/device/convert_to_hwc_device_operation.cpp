// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "convert_to_hwc_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ConvertToHWCDeviceOperation::program_factory_t ConvertToHWCDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ConvertToHWCProgramFactory{};
}

void ConvertToHWCDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ConvertToHWCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input = tensor_args.input;
    const auto& shape = input.logical_shape();
    const auto& C = shape[-2];

    TT_FATAL(shape.size() == 4, "Input shape must be rank 4 (was rank {})", shape.size());
    TT_FATAL(shape[0] == 1, "Expected input tensor to be shape [1, B, C, HW] (shape was {})", shape);
    TT_FATAL(C <= TILE_HEIGHT, "C must be less than or equal to 32 (was {})", C);

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input tensor must be in row-major layout");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");

    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Input tensor must be width sharded");
    TT_FATAL(
        args.memory_config.is_sharded() && args.memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must be height sharded");
}

TensorSpec ConvertToHWCDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& shape = tensor_args.input.logical_shape();
    const int B = shape[1];
    const int C = shape[2];
    const int HW = shape[3];

    // Output needs to be multiple of alignment requirement to guarantee aligned copies
    const auto alignment_elements = compute_alignment_requirement_in_elements(tensor_args.input);
    const auto output_channels = tt::round_up(C, alignment_elements);

    return TensorSpec(
        Shape({1, 1, B * HW, output_channels}),
        TensorLayout(args.dtype, PageConfig(Layout::ROW_MAJOR), args.memory_config));
}

Tensor ConvertToHWCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ConvertToHWCDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::stl::hash::hash_objects_with_default_seed(
        tt::stl::hash::type_hash<ConvertToHWCDeviceOperation>, args, tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::ConvertToHWCDeviceOperation::tensor_return_value_t convert_to_hwc(
    const Tensor& input, const MemoryConfig& memory_config, const DataType& dtype) {
    using OperationType = ttnn::experimental::prim::ConvertToHWCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config,
        .dtype = dtype,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
