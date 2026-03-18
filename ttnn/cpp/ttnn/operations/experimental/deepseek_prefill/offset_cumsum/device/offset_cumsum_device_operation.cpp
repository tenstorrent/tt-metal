// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
void OffsetCumsumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::UINT32, "Only UINT32 is supported for inputs!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(
        input_shape.size() == 2,
        "Input tensor must be 2D [num_devices, n_routed_experts], got {} dimensions",
        input_shape.size());
}

OffsetCumsumDeviceOperation::spec_return_value_t OffsetCumsumDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    const auto& logical_shape = input_tensor.logical_shape();
    uint32_t H = logical_shape[-2];
    uint32_t W = logical_shape[-1];

    auto layout = tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::UINT32,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

    auto offsets_spec = TensorSpec(ttnn::Shape({H, W}), layout);
    auto totals_spec = TensorSpec(ttnn::Shape({1, W}), layout);
    return {offsets_spec, totals_spec};
}

tt::stl::hash::hash_t OffsetCumsumDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    const auto& input_shape = input_tensor.padded_shape();
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<OffsetCumsumDeviceOperation>(
        args, input_tensor.dtype(), input_tensor.memory_config(), input_shape);
    return hash;
}

OffsetCumsumDeviceOperation::tensor_return_value_t OffsetCumsumDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    auto output_specs = compute_output_specs(args, input_tensor);
    auto offsets_tensor = create_device_tensor(output_specs[0], input_tensor.device());
    auto totals_tensor = create_device_tensor(output_specs[1], input_tensor.device());
    return {offsets_tensor, totals_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 2> offset_cumsum(const Tensor& input_tensor) {
    using OperationType = ttnn::experimental::prim::OffsetCumsumDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, input_tensor);
}

}  // namespace ttnn::prim
