// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deepseek_b1::mcast {

McastDeviceOperation::program_factory_t McastDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return McastProgramFactory{};
}

void McastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate output tensor is provided
    TT_FATAL(tensor_args.optional_output_tensor.has_value(), "Output tensor must be provided");

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_args.optional_output_tensor.value();

    // Validate input tensor
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");

    // Validate that input is sharded on one core
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    TT_FATAL(input_tensor.memory_config().shard_spec().has_value(), "Input tensor must have a shard spec");
    const auto& input_shard_spec = *input_tensor.memory_config().shard_spec();
    TT_FATAL(
        input_shard_spec.grid.num_cores() == 1,
        "Input tensor must be sharded on exactly one core, but found {} cores",
        input_shard_spec.grid.num_cores());

    // Validate output tensor
    TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output must be on device");
    TT_FATAL(output_tensor.layout() == input_tensor.layout(), "Output must have the same layout as input");

    // Validate output is sharded on multiple cores
    TT_FATAL(output_tensor.memory_config().is_sharded(), "Output tensor must be sharded");
    TT_FATAL(output_tensor.memory_config().shard_spec().has_value(), "Output tensor must have a shard spec");

    // Validate that output shard size matches input shard size
    const auto& output_shard_spec = *output_tensor.memory_config().shard_spec();
    TT_FATAL(
        input_shard_spec.shape == output_shard_spec.shape,
        "Output shard shape must match input shard shape. Input: {}, Output: {}",
        input_shard_spec.shape,
        output_shard_spec.shape);
    TT_FATAL(output_shard_spec.grid.ranges().size() == 1, "Output tensor must have exactly one shard range");
}

void McastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

McastDeviceOperation::spec_return_value_t McastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return input_tensor.tensor_spec();
}

McastDeviceOperation::tensor_return_value_t McastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For this operation, the output tensor must be pre-allocated
    TT_FATAL(tensor_args.optional_output_tensor.has_value(), "Output tensor must be provided for mcast operation");
    return tensor_args.optional_output_tensor.value();
}

std::tuple<McastDeviceOperation::operation_attributes_t, McastDeviceOperation::tensor_args_t>
McastDeviceOperation::invoke(const Tensor& input_tensor, const Tensor& output_tensor, uint32_t noc) {
    return {
        operation_attributes_t{.noc = static_cast<tt::tt_metal::NOC>(noc)},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::deepseek_b1::mcast
