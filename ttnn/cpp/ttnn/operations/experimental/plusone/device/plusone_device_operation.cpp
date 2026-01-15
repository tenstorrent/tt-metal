// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

PlusOneDeviceOperation::program_factory_t PlusOneDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PlusOneProgramFactory{};
}

void PlusOneDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void PlusOneDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& input_tensor) {
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::INT32 || input_tensor.dtype() == tt::tt_metal::DataType::UINT32,
        "Only INT32 and UINT32 is supported for inputs!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.size() >= 1 && input_shape.size() <= 4, "must have 1 to 4 dimensions for input tensor");
}

PlusOneDeviceOperation::spec_return_value_t PlusOneDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& input_tensor) {
    return input_tensor.tensor_spec();
}

tt::stl::hash::hash_t PlusOneDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    const auto& input_shape = input_tensor.padded_shape();
    // Hash operation attributes (both sub_core_grids and skip_negative_entries affect program structure)
    // and specific tensor properties that affect program structure (dtype, memory_config, shape)
    // rather than the whole tensor to avoid including runtime-only properties like buffer addresses
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<PlusOneDeviceOperation>(
        args,                          // Includes sub_core_grids and skip_negative_entries
        input_tensor.dtype(),          // Affects CB data format and element size
        input_tensor.memory_config(),  // Affects buffer type (DRAM/L1) and sharding
        input_shape);                  // Affects W, H, aligned_input_page_size, core groups

    return hash;
}

PlusOneDeviceOperation::tensor_return_value_t PlusOneDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& input_tensor) {
    return input_tensor;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor plus_one(
    const Tensor& input_tensor, const std::optional<CoreRangeSet>& sub_core_grids, bool skip_negative_entries) {
    using OperationType = ttnn::experimental::prim::PlusOneDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .sub_core_grids = sub_core_grids, .skip_negative_entries = skip_negative_entries};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, input_tensor);
}

}  // namespace ttnn::prim
