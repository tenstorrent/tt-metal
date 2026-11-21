// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
void PlusOneDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensor) {
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::INT32 || input_tensor.dtype() == tt::tt_metal::DataType::UINT32,
        "Only INT32 and UINT32 is supported for inputs!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.size() >= 1 && input_shape.size() <= 4, "must have 1 to 4 dimensions for input tensor");

    if (input_tensor.is_sharded()) {
        // The kernel increments the input shard in place, so it can only run on cores that hold a shard.
        TT_FATAL(input_tensor.shard_spec().has_value(), "plus_one: a sharded input must have a 2D shard spec");
        if (operation_attributes.sub_core_grids.has_value()) {
            const auto& shard_grid = input_tensor.shard_spec()->grid;
            TT_FATAL(
                operation_attributes.sub_core_grids->contains(shard_grid) &&
                    shard_grid.contains(*operation_attributes.sub_core_grids),
                "plus_one: sub_core_grids must match the input tensor's shard grid for a sharded input, got {} and {}",
                operation_attributes.sub_core_grids->str(),
                shard_grid.str());
        }
    }
}

PlusOneDeviceOperation::spec_return_value_t PlusOneDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& input_tensor) {
    return input_tensor.tensor_spec();
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
