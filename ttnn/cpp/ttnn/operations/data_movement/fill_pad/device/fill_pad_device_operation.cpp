// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "fill_pad_program_factory.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

FillPadDeviceOperation::program_factory_t FillPadDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    // L1-sharded: each core processes its own shard (no cross-core NOC).
    // DRAM interleaved / DRAM-sharded (rare): fall through to the regular factory.
    if (input.is_sharded() && input.memory_config().is_l1()) {
        return FillPadL1ShardedProgramFactory{};
    }
    return FillPadProgramFactory{};
}

ttsl::hash::hash_t FillPadDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto program_factory = select_program_factory(operation_attributes, tensor_args);
    const auto& input = tensor_args.input;
    const auto tile_height = input.tensor_spec().tile().get_height();
    const auto& padded_shape = input.padded_shape();
    if (!std::holds_alternative<FillPadProgramFactory>(program_factory) || input.logical_shape().rank() < 2 ||
        input.logical_shape()[-2] % tile_height == 0) {
        return tt::tt_metal::operation::hash_operation<FillPadDeviceOperation>(
            operation_attributes, input.tensor_spec(), program_factory.index());
    }
    // FillPadProgramFactory takes the bottom remainder (H_mod32) as a runtime arg: every logical height
    // with the same padded height shares one program. Hash a canonical height in that bucket.
    auto canonical_shape = input.logical_shape();
    canonical_shape[-2] = padded_shape[-2] - 1;
    const tt::tt_metal::TensorSpec canonical(canonical_shape, input.tensor_spec().tensor_layout());
    TT_FATAL(canonical.padded_shape() == padded_shape, "FillPad canonical height changed the padded shape");
    return tt::tt_metal::operation::hash_operation<FillPadDeviceOperation>(
        operation_attributes, canonical, program_factory.index());
}

void FillPadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(input_tensor.layout() == TILE_LAYOUT, "FillPad should only be used for tile layout");
    TT_FATAL(detail::data_type_to_size.contains(input_tensor.dtype()), "Unsupported datatype {}", input_tensor.dtype());
}

tt::tt_metal::TensorSpec FillPadDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return input_tensor.tensor_spec();
}

Tensor FillPadDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return input_tensor;
}

ttnn::Tensor fill_pad(
    const Tensor& input,
    ttnn::PadValue fill_value,
    const MemoryConfig& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::prim::FillPadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .fill_value = fill_value,
            .output_mem_config = output_memory_config,
            .sub_core_grids = sub_core_grids,
        },
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
